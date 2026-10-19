# BKS potential for SiO2 (van Beest, Kramer, van Santen, PRL 64, 1955 (1990))
# with the 30-6 core correction of Saika-Voivod et al., PRE 69, 041503 (2004).
# Atom types: 1 = Si, 2 = O.  Charges (Si +2.4, O -1.2) are stored in the data files.
# Requires: ${potdir} pointing to this directory.
# Long-range Coulomb: pppm by default; -var kstyle ewald switches to Ewald (e.g. for LAMMPS builds
# whose FFT library crashes in pppm).

variable        kstyle index pppm
variable        kacc   index 1.0e-5

pair_style      hybrid/overlay coul/long 10.0 table linear 10000
pair_coeff      * * coul/long
pair_coeff      1 2 table ${potdir}/BKS_SiO2.table SiO 10.0
pair_coeff      2 2 table ${potdir}/BKS_SiO2.table OO 10.0
kspace_style    ${kstyle} ${kacc}

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes
