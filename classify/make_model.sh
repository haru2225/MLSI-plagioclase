#!/bin/bash
# Make a model directory for classify_lammpstrj.py from a (running or finished) training run.
#   bash classify/make_model.sh scripts/training/<run> [epoch]       # default epoch: 50  -> classify/model_ep<epoch>
# MLSI reads torch_checkpoint_best.tar, so the chosen checkpoint is copied under that name (a snapshot:
# it does not change while the training continues).  Checkpoints exist for epochs 0, 10, 20, ...
set -eu
RUN="${1:?usage: make_model.sh <training run dir> [epoch]}"
EP="${2:-50}"
OUT="$(dirname "$0")/model_ep$EP"
[ -f "$RUN/torch_checkpoint_$EP.tar" ] || { echo "no $RUN/torch_checkpoint_$EP.tar; available:"; ls "$RUN" | grep checkpoint; exit 1; }
mkdir -p "$OUT"
cp "$RUN/settings.json" "$OUT/settings.json"
cp "$RUN/torch_checkpoint_$EP.tar" "$OUT/torch_checkpoint_best.tar"
echo "model: $OUT"
