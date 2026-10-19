#!/usr/bin/env bash
# MLSI training data: LAMMPS driver.
#
#   ./run_all.sh <stage>          run one stage: crystal | melt | quench | compress | tensile
#   ./run_all.sh all              run all stages in order (melt -> quench -> compress/tensile depend on each other)
#   ./run_all.sh list <stage>     print the job ids of a stage (one per line)
#   ./run_all.sh job <job-id>     run a single job, e.g. from a PBS/Slurm array
#
# Environment variables:
#   LMP      LAMMPS executable             (default: lmp)
#   NP       MPI ranks per job             (default: 1; >1 runs "$MPIRUN -np $NP $LMP")
#   NJOBS    jobs of a stage run at once   (default: 16; 16 single-core jobs, NP x NJOBS = number of cores)
#   MPIRUN   MPI launcher                  (default: mpirun)
#   LMPARGS  extra LAMMPS arguments, e.g. "-sf omp -pk omp 4" or "-var kstyle ewald"
#
# Finished jobs leave logs/<job>.done and are skipped when the stage is run again.
set -euo pipefail
cd "$(dirname "$0")"
SELF="$PWD/$(basename "$0")"

# ------------------------------------------------------------------ conditions (edit here)
# crystals: name:output-folder:potential (structures/crystal/<name>.data)
CRYSTALS="albite:plagioclase:SHIK anorthite:plagioclase:SHIK labradorite:plagioclase:SHIK
          jadeite:jadeite:SHIK lingunite:lingunite:SHIK
          grossular:grossular:SHIK kyanite:kyanite:SHIK
          quartz:quartz:SHIK coesite:coesite:SHIK cristobalite:cristobalilte:SHIK tridymite:tridymite:SHIK
          stishovite:stishovite:SHIK seifertite:seifertite:SHIK
          beta_quartz:quartz:SHIK beta_tridymite:tridymite:SHIK
          corundum:corundum:SHIK casio3_perovskite:casio3_perovskite:SHIK cas_phase:cas_phase:SHIK cf_naalsio4:cf_naalsio4:SHIK
          andalusite:andalusite:SHIK sillimanite:sillimanite:SHIK monalbite:plagioclase:SHIK anorthite_I1:plagioclase:SHIK"
CRYSTAL_T="300 500"             # K
CRYSTAL_P="0 1"                 # GPa; per-crystal override: CRYSTAL_P_<name>
# per-crystal temperatures (K): beta forms, monalbite and I-1 anorthite are only run hot
# (SHIK: the beta classes start from the alpha structures and become beta on heating)
# SHIK quartz: the time-averaged Si-O-Si angle grows from 148 deg (100 K) to 157 deg (>= 600 K): alpha below ~400 K, beta above ~600 K
CRYSTAL_T_quartz="100 300"
CRYSTAL_T_beta_quartz="700 1000"
CRYSTAL_T_cristobalite="500 700 850 1000"   # SHIK cristobalite is the cubic beta form (alpha collapses below ~500 K)
CRYSTAL_T_beta_tridymite="700 850 1000"
# the low-density cristobalite and beta-tridymite are only held at ~0 GPa: they collapse to denser structures at 1 GPa
CRYSTAL_P_cristobalite="0"
CRYSTAL_P_beta_tridymite="0"
CRYSTAL_T_monalbite="800 1100"
# high-pressure phases: stable only at high pressure (CF-type NaAlSiO4 stable above ~24 GPa)
CRYSTAL_P_cf_naalsio4="25 30"
CRYSTAL_T_anorthite_I1="500 800"

# glass/melt compositions: name:potential (structures/glass_init/<name>.data)
GLASSES="SiO2:SHIK albite:SHIK anorthite:SHIK labradorite:SHIK"
MELT_T1=5000                    # K, melt snapshots and melting
MELT_T2=4000                    # K, melt snapshots and start of the quench
GLASS_P="0 60"                  # GPa, quench pressures = start/end of the (de)compression
QUENCH_RATES="1 10"             # K/ps (1e12 and 1e13 K/s)
COMPRESS_T="300 500 800 1000"   # K
COMPRESS_RATE=0.1               # GPa/ps
TENSILE_GLASS_P="0"             # GPa, quench pressure of the glass models that are stretched
TENSILE_T="300 800"             # K
TENSILE_ERATE=0.01              # 1/ps (1e10 1/s), uniaxial tension along z
TENSILE_STRAIN=0.6              # final engineering strain (fracture happens well before)

# ------------------------------------------------------------------ helpers
LMP=${LMP:-lmp}
NP=${NP:-1}
MPIRUN=${MPIRUN:-mpirun}
NJOBS=${NJOBS:-16}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}   # one core per job; avoids oversubscription by threaded libraries
LMPARGS=${LMPARGS:-}
OUT=md_training_data
mkdir -p logs work "$OUT"

seed_of() { echo $(( $(printf '%s' "$1" | cksum | cut -d' ' -f1) % 900000 + 1000 )); }
other_p() { local lo hi; read -r lo hi <<< "$GLASS_P"; [[ $1 == "$lo" ]] && echo "$hi" || echo "$lo"; }
lookup() { local key=$1 list=$2 item; for item in $list; do [[ ${item%%:*} == "$key" ]] && { echo "$item"; return; }; done
           echo "unknown name: $key" >&2; return 1; }

run_lmp() {  # run_lmp <job-id> <input> <-var ...>
    local id=$1 input=$2 tag=${1//:/_}
    shift 2
    if [[ -f logs/$tag.done ]]; then echo "skip   $id"; return 0; fi
    local launch=("$LMP")
    (( NP > 1 )) && launch=("$MPIRUN" -np "$NP" "$LMP")
    echo "start  $id  ($(date '+%F %T'))"
    # shellcheck disable=SC2086
    if "${launch[@]}" -in "$input" "$@" -log "logs/$tag.log" -screen none $LMPARGS > "logs/$tag.stdout" 2>&1; then
        touch "logs/$tag.done"
        echo "done   $id  ($(date '+%F %T'))"
    else
        echo "FAILED $id  (see logs/$tag.log, logs/$tag.stdout)" >&2
        return 1
    fi
}

list_jobs() {
    local item name pot t p r
    case $1 in
    crystal)
        for item in $CRYSTALS; do
            name=${item%%:*}
            local pv=CRYSTAL_P_$name tv=CRYSTAL_T_$name
            for t in ${!tv:-$CRYSTAL_T}; do for p in ${!pv:-$CRYSTAL_P}; do echo "crystal:$name:$t:$p"; done; done
        done ;;
    melt)
        for item in $GLASSES; do for p in $GLASS_P; do echo "melt:${item%%:*}:$p"; done; done ;;
    quench)
        for item in $GLASSES; do for p in $GLASS_P; do for r in $QUENCH_RATES; do
            echo "quench:${item%%:*}:$p:$r"; done; done; done ;;
    compress)
        for item in $GLASSES; do for p in $GLASS_P; do for r in $QUENCH_RATES; do for t in $COMPRESS_T; do
            echo "compress:${item%%:*}:$p:$r:$t"; done; done; done; done ;;
    tensile)
        for item in $GLASSES; do for p in $TENSILE_GLASS_P; do for r in $QUENCH_RATES; do for t in $TENSILE_T; do
            echo "tensile:${item%%:*}:$p:$r:$t"; done; done; done; done ;;
    *) echo "unknown stage: $1" >&2; return 1 ;;
    esac
}

run_job() {
    local id=$1 stage name a b c item phase pot
    IFS=: read -r stage name a b c <<< "$id"
    case $stage in
    crystal)  # a=T b=P
        item=$(lookup "$name" "$CRYSTALS"); IFS=: read -r _ phase pot <<< "$item"
        mkdir -p "$OUT/$phase"
        run_lmp "$id" in/in.crystal -var data "structures/crystal/$name.data" -var pot "$pot" \
            -var T "$a" -var P "$b" -var seed "$(seed_of "$id")" \
            -var dump "$OUT/$phase/${name}_T${a}K_P${b}GPa.lammpstrj" ;;
    melt)  # a=P
        item=$(lookup "$name" "$GLASSES"); pot=${item##*:}
        mkdir -p "$OUT/melt"
        run_lmp "$id" in/in.melt -var data "structures/glass_init/$name.data" -var pot "$pot" \
            -var P "$a" -var seed "$(seed_of "$id")" -var T1 "$MELT_T1" -var T2 "$MELT_T2" \
            -var dump1 "$OUT/melt/${name}_T${MELT_T1}K_P${a}GPa.lammpstrj" \
            -var dump2 "$OUT/melt/${name}_T${MELT_T2}K_P${a}GPa.lammpstrj" \
            -var liquid "work/liquid_${name}_P${a}GPa.data" ;;
    quench)  # a=P b=rate
        item=$(lookup "$name" "$GLASSES"); pot=${item##*:}
        [[ -f work/liquid_${name}_P${a}GPa.data ]] || { echo "missing work/liquid_${name}_P${a}GPa.data: run melt first" >&2; return 1; }
        run_lmp "$id" in/in.quench -var liquid "work/liquid_${name}_P${a}GPa.data" -var pot "$pot" \
            -var P "$a" -var rate "$b" -var T2 "$MELT_T2" \
            -var glass "work/glass_${name}_P${a}GPa_q${b}Kps.data" ;;
    compress)  # a=P0 b=rate c=T
        item=$(lookup "$name" "$GLASSES"); pot=${item##*:}
        local p1; p1=$(other_p "$a")
        [[ -f work/glass_${name}_P${a}GPa_q${b}Kps.data ]] || { echo "missing work/glass_${name}_P${a}GPa_q${b}Kps.data: run quench first" >&2; return 1; }
        mkdir -p "$OUT/amorphous"
        run_lmp "$id" in/in.compress -var glass "work/glass_${name}_P${a}GPa_q${b}Kps.data" -var pot "$pot" \
            -var T "$c" -var P0 "$a" -var P1 "$p1" -var prate "$COMPRESS_RATE" -var seed "$(seed_of "$id")" \
            -var dump "$OUT/amorphous/${name}_q${b}Kps_P${a}to${p1}GPa_T${c}K.lammpstrj" ;;
    tensile)  # a=P (quench) b=rate c=T
        item=$(lookup "$name" "$GLASSES"); pot=${item##*:}
        [[ -f work/glass_${name}_P${a}GPa_q${b}Kps.data ]] || { echo "missing work/glass_${name}_P${a}GPa_q${b}Kps.data: run quench first" >&2; return 1; }
        mkdir -p "$OUT/amorphous"
        run_lmp "$id" in/in.tensile -var glass "work/glass_${name}_P${a}GPa_q${b}Kps.data" -var pot "$pot" \
            -var T "$c" -var erate "$TENSILE_ERATE" -var strain "$TENSILE_STRAIN" -var seed "$(seed_of "$id")" \
            -var dump "$OUT/amorphous/${name}_q${b}Kps_P${a}GPa_tensile_T${c}K.lammpstrj" ;;
    *) echo "unknown job: $id" >&2; return 1 ;;
    esac
}

run_stage() {
    echo "=== stage $1: $(list_jobs "$1" | wc -l | tr -d ' ') jobs, NJOBS=$NJOBS, NP=$NP"
    list_jobs "$1" | xargs -P "$NJOBS" -n 1 bash "$SELF" job
}

case ${1:-} in
    crystal|melt|quench|compress|tensile) run_stage "$1" ;;
    all) for s in crystal melt quench compress tensile; do run_stage "$s"; done ;;
    list) list_jobs "${2:?stage}" ;;
    job) run_job "${2:?job id}" ;;
    *) sed -n '2,18p' "$SELF"; exit 1 ;;
esac
