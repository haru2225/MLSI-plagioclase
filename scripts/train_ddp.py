"""Multi-GPU training of the MLSI DG-CNN classifier (PyTorch DistributedDataParallel).

    torchrun --standalone --nproc_per_node=<GPUs> train_ddp.py --points 64 [options]

Same model, loss, optimizer, scheduler and output files as train_pytorch.py (settings.json,
torch_checkpoint_best.tar, ...), so the result can be used with the MLSI OVITO modifier unchanged.
Differences, all for GPU throughput:
  * every GPU runs its own process; --batch-size is per GPU (effective batch = batch-size x GPUs)
  * the training samples are loaded once into memory (onto the GPU if they fit), so no DataLoader workers are needed
  * random rotations are applied on the GPU, batch-wise
  * optional bf16 autocast (--amp)
"""
import argparse
import datetime
import json
import math
import os
import shutil
import time

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from Model import DGCNN_cls


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--points", type=int, default=64, help="neighbours per sample (needs 1000000_<points>/)")
    p.add_argument("--data-dir", default=None, help="directory with numpy_data.h5 and (rev_)mapping.json")
    p.add_argument("--n-samples", type=int, default=500000, help="samples per class used for training+validation")
    p.add_argument("--split", type=float, default=0.8, help="training fraction")
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=128, help="per GPU")
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate for one GPU (batch-size samples)")
    p.add_argument("--lr-scaling", choices=("none", "sqrt", "linear"), default="sqrt",
                   help="scale --lr with the number of GPUs (effective batch size)")
    p.add_argument("--eta-min", type=float, default=1e-5)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--emb-dims", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=123456)
    p.add_argument("--amp", action="store_true", help="bf16 autocast (faster on A100/H100; check accuracy)")
    p.add_argument("--sync-bn", action="store_true", help="synchronise BatchNorm statistics over GPUs")
    p.add_argument("--data-on-gpu", choices=("auto", "yes", "no"), default="auto",
                   help="keep the samples in GPU memory (auto: if they take < 40%% of it)")
    p.add_argument("--scratch", default=os.environ.get("TMPDIR", ""), help="node-local directory for the h5 copy")
    p.add_argument("--out", default=None, help="output directory (default training/<date-time>)")
    return p.parse_args()


def setup():
    if "RANK" in os.environ:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local = 0, 1, 0
    if not torch.cuda.is_available():  # CPU only for a quick test of the script
        return rank, world, torch.device("cpu")
    torch.cuda.set_device(local)
    return rank, world, torch.device("cuda", local)


def load_data(h5name, mapname, n_samples, rank):
    """All classes as one (n_classes * n_samples, points, 3) float32 array and the labels."""
    with open(mapname) as f:
        rev_mapping = json.load(f)
    with open(mapname.replace("rev_", "")) as f:
        mapping = {int(k): v for k, v in json.load(f).items()}
    names = list(rev_mapping)
    with h5py.File(h5name, "r") as hf:
        shape = hf[names[0]].shape[1:]
        x = np.empty((len(names) * n_samples,) + shape, dtype=np.float32)  # filled in place: no second copy
        y = np.empty(len(names) * n_samples, dtype=np.int64)
        for i, name in enumerate(names):
            n = len(hf[name])
            assert n >= n_samples, f"{name}: {n} samples < --n-samples {n_samples}"
            hf[name].read_direct(x, np.s_[:n_samples], np.s_[i * n_samples:(i + 1) * n_samples])
            y[i * n_samples:(i + 1) * n_samples] = rev_mapping[name]
            if rank == 0:
                print(f"  {rev_mapping[name]:3d} {name:20s} {n} samples", flush=True)
    return torch.from_numpy(x), torch.from_numpy(y), mapping


def random_rotations(n, device, gen):
    """n uniformly distributed rotation matrices (QR of Gaussian matrices)."""
    a = torch.randn(n, 3, 3, device=device, generator=gen)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r, dim1=1, dim2=2)).unsqueeze(1)
    det = torch.det(q)
    q[:, :, 0] *= det.unsqueeze(1)
    return q


def confusion(true, pred, n):
    return torch.bincount(true * n + pred, minlength=n * n).view(n, n)


def scores(cm):
    cm = cm.double()
    per_class = torch.diagonal(cm) / cm.sum(1).clamp(min=1)
    return (torch.diagonal(cm).sum() / cm.sum()).item(), per_class.mean().item(), per_class.tolist()


def main():
    args = parse_args()
    rank, world, device = setup()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data_dir = args.data_dir or f"1000000_{args.points}"
    h5name = os.path.join(data_dir, "numpy_data.h5")
    tmp_copy = None
    if args.scratch and os.path.isdir(args.scratch):
        # one copy per node (not per GPU), only if the node-local disk has room for it; removed after the data is loaded
        tmp_copy = os.path.join(args.scratch, f"mlsi_{os.environ.get('PBS_JOBID', os.getpid())}_numpy_data.h5")
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            need = os.path.getsize(h5name) * 1.2
            if shutil.disk_usage(args.scratch).free > need:
                shutil.copy(h5name, tmp_copy)
            else:
                print(f"not enough space in {args.scratch}: reading {h5name} directly", flush=True)
        if world > 1:
            dist.barrier()
        if os.path.exists(tmp_copy):
            h5name = tmp_copy
        else:
            tmp_copy = None

    if rank == 0:
        print(f"GPUs: {world} x {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}", flush=True)
    x_all, y_all, mapping = load_data(h5name, os.path.join(data_dir, "rev_mapping.json"), args.n_samples, rank)
    n_classes = len(mapping)
    if world > 1:
        dist.barrier()  # every rank has read the file
    if tmp_copy and int(os.environ.get("LOCAL_RANK", 0)) == 0 and os.path.exists(tmp_copy):
        os.remove(tmp_copy)

    # the same train/validation split on every GPU
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(y_all), generator=g)
    n_train = int(len(perm) * args.split)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    if device.type == "cuda":
        free = torch.cuda.get_device_properties(device).total_memory
        on_gpu = args.data_on_gpu == "yes" or (args.data_on_gpu == "auto" and x_all.numel() * 4 < 0.4 * free)
    else:
        on_gpu = True  # CPU test: "device memory" is the host memory
    if on_gpu:
        x_all, y_all = x_all.to(device), y_all.to(device)
    else:
        x_all = x_all.pin_memory()

    lr = args.lr * {"none": 1.0, "sqrt": math.sqrt(world), "linear": float(world)}[args.lr_scaling]
    settings = {
        "h5fname": os.path.abspath(os.path.join(data_dir, "numpy_data.h5")),
        "mapfname": os.path.abspath(os.path.join(data_dir, "rev_mapping.json")),
        "num_pts": args.points, "seed": args.seed, "batch_size": args.batch_size, "lr": lr, "eta_min": args.eta_min,
        "k": args.k, "dim9": False, "emb_dims": args.emb_dims, "dropout": args.dropout, "epochs": args.epochs,
        "split": args.split, "n_samples": args.n_samples, "schedule_divisor": 1, "num_outputs": n_classes,
        "classes": mapping, "gpus": world, "effective_batch_size": args.batch_size * world, "amp_bf16": args.amp,
        "score_cutoff_other": 0.5,
    }

    model = DGCNN_cls(settings, output_channels=n_classes).to(device)
    if args.sync_bn and world > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp = nn.parallel.DistributedDataParallel(model, device_ids=[device.index]) if world > 1 else model
    opt = torch.optim.Adam(ddp.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, args.epochs, eta_min=args.eta_min)
    loss_func = nn.functional.cross_entropy

    dname = args.out or os.path.join("training", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    if rank == 0:
        os.makedirs(dname, exist_ok=True)
        with open(os.path.join(dname, "settings.json"), "w") as f:
            json.dump(settings, f, indent=1)
        log = open(os.path.join(dname, "training.log"), "w")
        log.write("epoch train_loss train_acc train_avg_acc test_loss test_acc test_avg_acc\n")
        detail = open(os.path.join(dname, "training_detail.log"), "w")
        detail.write("dataset\titeration\t" + "\t".join(mapping[i] for i in range(n_classes)) + "\t\n")
        print("Output directory", dname, flush=True)

    aug = torch.Generator(device=device).manual_seed(args.seed + 1000 * rank)

    def batches(idx, shuffle, epoch):
        if shuffle:
            g = torch.Generator().manual_seed(args.seed + epoch)
            idx = idx[torch.randperm(len(idx), generator=g)]
        idx = idx[rank::world]
        nb = len(idx) // args.batch_size if shuffle else math.ceil(len(idx) / args.batch_size)
        if shuffle and world > 1:  # every GPU must run the same number of steps
            t = torch.tensor(nb, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            nb = int(t)
        for b in range(nb):
            sel = idx[b * args.batch_size:(b + 1) * args.batch_size]
            if on_gpu:
                sel = sel.to(device)
                yield x_all[sel], y_all[sel]
            else:
                yield x_all[sel].to(device, non_blocking=True), y_all[sel].to(device, non_blocking=True)

    def run(idx, train, epoch):
        ddp.train(train)
        cm = torch.zeros(n_classes, n_classes, dtype=torch.long, device=device)
        tot = torch.zeros(2, dtype=torch.float64, device=device)  # loss sum, count
        with torch.set_grad_enabled(train):
            for x, y in batches(idx, train, epoch):
                if train:
                    x = torch.bmm(x, random_rotations(len(x), device, aug))
                x = x.permute(0, 2, 1).contiguous()
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp):
                    logits = ddp(x)
                    loss = loss_func(logits.float(), y)
                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                tot += torch.stack([loss.detach().double() * len(y), torch.tensor(float(len(y)), device=device,
                                                                                    dtype=torch.float64)])
                cm += confusion(y, logits.argmax(1), n_classes)
        if world > 1:
            dist.all_reduce(cm)
            dist.all_reduce(tot)
        return (tot[0] / tot[1]).item(), cm

    best = -1.0
    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss, tr_cm = run(train_idx, True, epoch)
        sched.step()
        te_loss, te_cm = run(val_idx, False, epoch)
        if rank != 0:
            continue
        tr_acc, tr_bal, tr_pc = scores(tr_cm)
        te_acc, te_bal, te_pc = scores(te_cm)
        print(f"epoch {epoch}: train loss {tr_loss:.5f} acc {tr_acc:.5f} | test loss {te_loss:.5f} acc {te_acc:.5f} "
              f"avg {te_bal:.5f} | {time.time() - t0:.0f} s", flush=True)
        log.write(f"{epoch} {tr_loss} {tr_acc:.6f} {tr_bal:.6f} {te_loss} {te_acc} {te_bal}\n")
        log.flush()
        detail.write(f"Train\t{epoch}\t" + "\t".join(map(str, tr_pc)) + "\t\n")
        detail.write(f"Test\t{epoch}\t" + "\t".join(map(str, te_pc)) + "\t\n")
        detail.flush()
        state = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(),
                 "scheduler_state_dict": sched.state_dict(), "loss": tr_loss}
        if te_acc >= best:
            best = te_acc
            torch.save(state, os.path.join(dname, "torch_checkpoint_best.tar"))
            torch.save(model, os.path.join(dname, "torch_model_best.t7"))
            np.savetxt(os.path.join(dname, "confusion_best.txt"), te_cm.cpu().numpy(), fmt="%d",
                       header=" ".join(mapping[i] for i in range(n_classes)))
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            torch.save(state, os.path.join(dname, f"torch_checkpoint_{epoch}.tar"))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
