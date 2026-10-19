#!/bin/bash
# Submit the whole chain (extract -> h5 -> train) from the repository root on the login node:
#   bash pbs/submit_all.sh                    # POINTS=64, all GPUs of the node
#   POINTS=32 bash pbs/submit_all.sh
set -eu
cd "$(dirname "$0")/.."
J1=$(qsub -v POINTS="${POINTS:-64}" pbs/01_extract.pbs)
J2=$(qsub -v POINTS="${POINTS:-64}" -W depend=afterokarray:"$J1" pbs/02_h5.pbs)
J3=$(qsub -v POINTS="${POINTS:-64}" -W depend=afterok:"$J2" pbs/03_train.pbs)
echo "extract $J1 -> h5 $J2 -> train $J3"
