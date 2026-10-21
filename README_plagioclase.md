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
git clone https://github.com/haru2225/MLSI-plagioclase.git && cd MLSI-plagioclase
cp -r /path/to/md_training_data_merged scripts/        # or the Release archives, see above
singularity build --fakeroot mlsi-train-pytorch-2.5.0-cu124.sif singularity/Singularity.train.def   # once
qsub -P <project> mlsi.pbs
```

`mlsi.pbs` runs everything in one job: extraction of 1e6 local environments per label, the h5 file and the
multi-GPU training (all GPUs of the job). Resources can be changed on the command line, e.g.
`qsub -P <project> -l select=1:ncpus=32:mpiprocs=1:ompthreads=1:mem=256gb:ngpus=8 mlsi.pbs`, options with
`-v POINTS=64,BATCH=512,EPOCHS=250,AMP=1`. Finished steps are skipped when the job is submitted again.
The three-job version (`pbs/submit_all.sh`) does the same in separate jobs.
The result is in `scripts/training/<date>_64pts/` (`torch_checkpoint_best.tar`, `settings.json`).

## Classify trajectories

Put the `.lammpstrj` files in `classify/` (sub-folders are fine), then, with a finished or running training run:

```bash
bash classify/make_model.sh scripts/training/<run> 50              # snapshot of the epoch-50 checkpoint -> classify/model_ep50
qsub -P run classify/classify_all.pbs                               # one GPU, all files
qsub -P run -J 0-3 -v NTASKS=4 classify/classify_all.pbs            # 4 GPUs in parallel
tail -f classify/results/logs/<file>.log ; cat classify/results/progress_*.txt     # live progress
```
Results: `classify/results/<file>_classification.csv` (class counts and fractions per frame; scores < 0.5 are `Other`).

Notes on the data: the `melt` and `amorphous` classes were used as generated; the SiO2 melt/glass files
(`SiO2_*`) were computed with BKS, all crystals with SHIK. The `amorphous` set is partial (30 of the 80 planned files). Two 1 GPa tridymite files were dropped because the structure collapsed.
