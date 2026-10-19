# SHIK potential (Sundararaman, Huang, Ispas, Kob, J. Chem. Phys. 150, 154505 (2019);
# SiO2 part from J. Chem. Phys. 148, 194504 (2018)).
# Atom types: 1 = Si, 2 = O, 3 = Al, 4 = Na, 5 = Ca.
# Charges are stored in the data files: Si 1.7755, Al 1.6334, Na 0.6018, Ca 1.3967 and
# q_O = -(sum of cation charges)/N_O, i.e. the composition-dependent oxygen charge of SHIK.
# Coulomb: Wolf form of the paper = coul/dsf with alpha = 0, cutoff 10 A.
# Short range (8 A, smoothed) is tabulated; Si-Al, Na-Al, Ca-Al and Na-Ca are Coulomb only.
# Requires: ${potdir} pointing to this directory.

pair_style      hybrid/overlay coul/dsf 0.0 10.0 table linear 10000
pair_coeff      * * coul/dsf
pair_coeff      1 1 table ${potdir}/SHIK.table Si-Si 8.0
pair_coeff      1 2 table ${potdir}/SHIK.table O-Si  8.0
pair_coeff      2 2 table ${potdir}/SHIK.table O-O   8.0
pair_coeff      2 3 table ${potdir}/SHIK.table O-Al  8.0
pair_coeff      3 3 table ${potdir}/SHIK.table Al-Al 8.0
pair_coeff      2 4 table ${potdir}/SHIK.table O-Na  8.0
pair_coeff      1 4 table ${potdir}/SHIK.table Si-Na 8.0
pair_coeff      4 4 table ${potdir}/SHIK.table Na-Na 8.0
pair_coeff      2 5 table ${potdir}/SHIK.table O-Ca  8.0
pair_coeff      1 5 table ${potdir}/SHIK.table Si-Ca 8.0
pair_coeff      5 5 table ${potdir}/SHIK.table Ca-Ca 8.0

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes
