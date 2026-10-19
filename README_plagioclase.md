# MLSI for plagioclase

Fork of [MLSI](https://github.com/nnn911/MLSI) (Utt, Erhard et al., MIT) with a training set for plagioclase
(albite, anorthite, labradorite) and its polymorphs and high-pressure breakdown products, plus everything to
generate the data and to train the DG-CNN classifier on a GPU cluster (PBS + Singularity).

Classes (19, from the folder names of `md_training_data/`): plagioclase, jadeite, lingunite, grossular, kyanite,
andalusite, sillimanite, corundum, casio3_perovskite, cas_phase, cf_naalsio4, quartz, coesite, tridymite,
cristobalilte, stishovite, seifertite, melt, amorphous. Atoms with a softmax score below 0.5 are assigned to
`other` (no training data).

| Directory | Content |
|---|---|
| `scripts/lammps_training_data/` | LAMMPS inputs, SHIK potential, structure builder and `run_all.sh`; see its `README.md` for the labels and how each is generated |
| `docs/training_data_methods.md` | methods text (English) with references |
| `singularity/Singularity.train.def` | container: PyTorch 2.5.0 + CUDA 12.4 + OVITO 3.10.1 + h5py |
| `pbs/` | PBS jobs: `01_extract.pbs` (OVITO, per label) -> `02_h5.pbs` -> `03_train.pbs` (multi-GPU); `submit_all.sh` chains them |
| `scripts/train_ddp.py` | multi-GPU (DistributedDataParallel) training, writes the same files as `train_pytorch.py` |

## Get the data

The MD snapshots (about 5 GB, 500 frames per file) are release assets, one archive per label:

```bash
gh release download v1 -R haru2225/MLSI-plagioclase -D scripts/        # or download from the Releases page
cd scripts && for f in md_training_data_*.tar.gz; do tar xzf "$f"; done  # creates scripts/md_training_data/<label>/
```

## Train on the cluster

```bash
singularity build --fakeroot mlsi-train-pytorch-2.5.0-cu124.sif singularity/Singularity.train.def
# set `#PBS -P` (project) and, if needed, the queue and ngpus in pbs/*.pbs
bash pbs/submit_all.sh                 # POINTS=64 by default
```

Notes on the data: the `melt` and `amorphous` classes were used as generated; the SiO2 melt/glass files
(`SiO2_*`) were computed with BKS, all crystals with SHIK. The `amorphous` set is partial (30 of the 80 planned files). Two 1 GPa tridymite files were dropped because the structure collapsed.
