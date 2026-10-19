# Sourced by the PBS scripts.  Everything can be overridden with `qsub -v NAME=value`.
cd "$PBS_O_WORKDIR"
. /etc/profile.d/modules.sh
module purge
module load cuda/12.4

SIF_IMAGE="${SIF_IMAGE:-$PBS_O_WORKDIR/mlsi-train-pytorch-2.5.0-cu124.sif}"   # singularity build --fakeroot <this> singularity/Singularity.train.def
POINTS="${POINTS:-64}"                  # neighbours per sample (8, 16, 32, 64)
DATA_ROOT="${DATA_ROOT:-$PBS_O_WORKDIR/scripts}"   # contains md_training_data/; receives 1000000_<POINTS>/ and training/
[ -r "$SIF_IMAGE" ] || { echo "ERROR: container not found: $SIF_IMAGE" >&2; exit 1; }

# run <command ...> inside the container with the GPUs visible
run() {
    singularity exec --nv --bind "$PBS_O_WORKDIR:$PBS_O_WORKDIR" --bind "$DATA_ROOT:$DATA_ROOT" \
        --pwd "$DATA_ROOT" "$SIF_IMAGE" "$@"
}
