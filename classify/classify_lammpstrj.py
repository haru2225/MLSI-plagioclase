#!/usr/bin/env python3
"""Classify every selected frame of a LAMMPS trajectory with an MLSI model.

The script writes per-frame class counts/fractions to CSV.  It streams frames
through OVITO and does not retain the full trajectory in memory.  Optionally it
also stores the per-atom class and confidence of each frame as compressed .npz.

MLSI runs on the GPU automatically when torch.cuda.is_available(); pass
--require-gpu to abort instead of silently falling back to the (very slow) CPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from ovito.io import import_file

# MLSI is imported from this repository (../src) unless MLSI_SRC points to another "src" directory.
sys.path.insert(0, os.environ.get("MLSI_SRC") or str(Path(__file__).resolve().parent.parent / "src"))
from MLSI import MLSI  # noqa: E402


def check_gpu() -> str:
    """Return the GPU name after proving a CUDA kernel really runs on it."""
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. Use `singularity exec --nv` and request a GPU "
            "(ngpus=1); drop --require-gpu to run on the CPU."
        )
    name = torch.cuda.get_device_name(0)
    try:
        (torch.ones(8, device="cuda") + 1).sum().item()
    except RuntimeError as exc:  # e.g. "no kernel image is available"
        raise SystemExit(
            f"CUDA device {name} is visible but PyTorch {torch.__version__} "
            f"cannot run kernels on it: {exc}"
        )
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path("model_ep50"))
    parser.add_argument("--output", type=Path, default=None,
                        help="CSV path (default: <trajectory stem>_classification.csv)")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None,
                        help="Inclusive; defaults to the final frame")
    parser.add_argument("--every-nth", type=int, default=1)
    parser.add_argument("--cutoff", type=float, default=0.0,
                        help="Confidence below this value is labeled Other")
    parser.add_argument("--atoms-per-batch", type=int, default=None, metavar="N",
                        help="Atoms per network forward pass (model default: 16384). "
                             "Lower it on CUDA out-of-memory; labels are unchanged.")
    parser.add_argument("--save-labels", type=Path, default=None, metavar="DIR",
                        help="Also write per-atom id/structure/score of each frame "
                             "to DIR/<stem>_frame<N>.npz")
    parser.add_argument("--require-gpu", action="store_true",
                        help="Exit with an error unless a working CUDA GPU is present")
    args = parser.parse_args()

    trajectory = args.trajectory.resolve()
    model_dir = args.model_dir.resolve()
    output = (args.output or trajectory.with_name(
        f"{trajectory.stem}_classification.csv")).resolve()
    if not trajectory.is_file():
        parser.error(f"trajectory not found: {trajectory}")
    if not (model_dir / "settings.json").is_file() or not (
        model_dir / "torch_checkpoint_best.tar"
    ).is_file():
        parser.error(f"model settings/checkpoint missing in {model_dir}")
    if args.every_nth < 1 or args.start_frame < 0:
        parser.error("start-frame must be >= 0 and every-nth must be >= 1")
    if args.atoms_per_batch is not None and args.atoms_per_batch < 8:
        parser.error("atoms-per-batch must be >= 8")

    # MLSI classifies 8 * settings["batch_size"] atoms per forward pass.  Atoms
    # are independent in the network (eval mode), so a smaller batch gives the
    # same labels with less GPU memory.  Override it in a temporary model copy.
    model_copy = None
    if args.atoms_per_batch is not None:
        model_copy = tempfile.TemporaryDirectory(prefix="mlsi_model_")
        shutil.copy2(model_dir / "torch_checkpoint_best.tar", model_copy.name)
        overridden = json.loads((model_dir / "settings.json").read_text())
        overridden["batch_size"] = args.atoms_per_batch // 8
        Path(model_copy.name, "settings.json").write_text(json.dumps(overridden))
        model_dir = Path(model_copy.name)

    if args.require_gpu:
        device = f"GPU {check_gpu()}"
    else:
        device = "GPU " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"torch {torch.__version__} on {device}", flush=True)

    settings = json.loads((model_dir / "settings.json").read_text())
    classes = {int(k): v for k, v in settings["classes"].items()}
    print(f"atoms per forward pass: {8 * settings['batch_size']}", flush=True)
    pipeline = import_file(str(trajectory), multiple_frames=True)
    total_frames = pipeline.source.num_frames
    end = total_frames - 1 if args.end_frame is None else args.end_frame
    if end >= total_frames or end < args.start_frame:
        parser.error(f"frame range invalid for {total_frames} frames")

    pipeline.modifiers.append(MLSI(
        model_dir=str(model_dir), cutoff=args.cutoff,
        use_neighbor_finder=True, use_KDTree=False, buffer=0.0,
    ))

    fields = ["frame", "timestep", "particles", "score_mean", "score_min", "score_max"]
    fields.extend(f"{name}_{suffix}" for name in classes.values()
                  for suffix in ("count", "frac"))
    fields.extend(("Other_count", "Other_frac"))
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.save_labels:
        args.save_labels.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        frames = range(args.start_frame, end + 1, args.every_nth)
        for frame in frames:
            started = time.perf_counter()
            data = pipeline.compute(frame=frame)
            structures = np.asarray(data.particles["Structure"], dtype=np.int32)
            scores = np.asarray(data.particles["Score"], dtype=np.float64)
            row = {
                "frame": frame,
                "timestep": int(data.attributes.get("Timestep", frame)),
                "particles": int(structures.size),
                "score_mean": float(scores.mean()) if scores.size else 0.0,
                "score_min": float(scores.min()) if scores.size else 0.0,
                "score_max": float(scores.max()) if scores.size else 0.0,
            }
            for class_id, name in classes.items():
                count = int(np.count_nonzero(structures == class_id))
                row[f"{name}_count"] = count
                row[f"{name}_frac"] = count / structures.size if structures.size else 0.0
            other = int(np.count_nonzero(structures < 0))
            row["Other_count"] = other
            row["Other_frac"] = other / structures.size if structures.size else 0.0
            writer.writerow(row)
            handle.flush()
            if args.save_labels:
                ids = data.particles["Particle Identifier"] if (
                    "Particle Identifier" in data.particles) else np.arange(structures.size)
                np.savez_compressed(
                    args.save_labels / f"{trajectory.stem}_frame{frame:05d}.npz",
                    id=np.asarray(ids).astype(np.int32),
                    structure=structures.astype(np.int8),
                    score=scores.astype(np.float32),
                    timestep=row["timestep"],
                )
            print(f"frame {frame + 1}/{total_frames}: {structures.size:,} atoms "
                  f"in {time.perf_counter() - started:.1f} s", flush=True)

    print(f"summary: {output}")


if __name__ == "__main__":
    main()
