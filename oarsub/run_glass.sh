#!/usr/bin/env bash
#OAR --name smb_glass
#OAR -l /nodes=1/core=4,walltime=03:00:00
#OAR --stdout oarsub/logs/%jobid%.glass.out
#OAR --stderr oarsub/logs/%jobid%.glass.err
#
# FAMILY B -- GLASS clustering calibration, 9 LS10 samples x 4 NSIDE = 36 cells.
#
# Each cell root-finds the cl_amplitude whose mock sigma_hat reproduces that
# sample's measured sigma_hat, over several seeds so the answer carries a
# scatter.  The published calibration covered ONE cell with a single seed and a
# three-point scan, which is not a fit.
#
# ONE JOB, not an array: a probe costs ~3 s, so all 36 cells run in well under
# an hour.  A 36-element array would spend 36 of the 100 queue slots (GRICAD
# refuses a submission leaving more than 100 waiting) to save nothing.
#
# A fixed --scan cannot replace the fit: the required amplitude spans more than
# an order of magnitude across the nine samples -- the smallest needs ~1.6e-1
# against the 5e-4 default, a factor of 320 -- so any grid wide enough to
# bracket them all is far too coarse to land on one.
#
#   oarsub --project <proj> -S "./oarsub/run_glass.sh <tag> <nseeds> [first] [last]"
set -euo pipefail
cd "$(dirname "$0")/.."
source oarsub/_campaign_env.sh

TAG="${1:-${OAR_JOB_ID:-local}}"
NSEEDS="${2:-5}"
campaign_activate_env
campaign_threads >/dev/null

NN=${#SMB_NSIDES[@]}
NCELL=$(( ${#SMB_SAMPLES[@]} * NN ))
FIRST="${3:-0}"
LAST="${4:-$(( NCELL - 1 ))}"
# Honour an array submission if one is used anyway.  OAR_ARRAY_SIZE does NOT
# exist on this build -- a plain job and an array element look identical apart
# from the index -- so the caller states the range explicitly and only a caller
# that gives none is treated as an array element.  Without this every element of
# a 36-wide array ran all 36 cells into the same output paths at once.
if [ -z "${3:-}" ] && [ -n "${OAR_ARRAY_INDEX:-}" ] \
   && [ "${OAR_ARRAY_INDEX}" -gt 0 ] 2>/dev/null; then
    FIRST=$(( OAR_ARRAY_INDEX - 1 )); LAST="${FIRST}"
    echo "-- no explicit range: treating OAR_ARRAY_INDEX=${OAR_ARRAY_INDEX} as one cell"
fi
echo "== cells ${FIRST}..${LAST} of ${NCELL}, ${NSEEDS} seeds each"

for IDX in $(seq "${FIRST}" "${LAST}"); do
    SAMPLE="${SMB_SAMPLES[$(( IDX / NN ))]}"
    NSIDE="${SMB_NSIDES[$(( IDX % NN ))]}"
    NS4="$(printf '%04d' "${NSIDE}")"

    # LS10_VLIM_ANY_<logM>_Mstar_12.0_<zmin>_z_<zmax>_N_<ngal>
    IFS='_' read -r -a F <<< "${SAMPLE}"
    ZMIN="${F[6]}"; ZMAX="${F[8]}"; NGAL=$(( 10#${F[10]} ))

    # Target this sample's own residual scatter, not the fiducial cell's.
    PJ="${SMB_PKG}/data/sys_weights/${SAMPLE}_NSIDE${NS4}_params.json"
    SIGMA="$(python - "${PJ}" <<'PYEOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
if p.exists():
    d = json.load(p.open())
    v = d.get("sigma_hat_ols") or d.get("sigma_hat_add")
    print(v if v else "")
else:
    print("")
PYEOF
)"
    SIGMA_ARG=()
    if [ -n "${SIGMA}" ]; then
        SIGMA_ARG=(--sigma-hat-data "${SIGMA}")
    else
        echo "!! no params.json at ${PJ}; using the script default sigma_hat_data"
    fi

    OUT="${SMB_RESULTS}/glass_calibration/${TAG}/${SAMPLE}_NSIDE${NS4}"
    echo "-- cell ${IDX}: ${SAMPLE} NSIDE=${NSIDE} n_gal=${NGAL} z=[${ZMIN},${ZMAX}] sigma=${SIGMA:-default}"
    for k in $(seq 0 $(( NSEEDS - 1 ))); do
        SEED=$(( 11 + 1000 * k ))
        python characterisation/calibrate_glass_clustering.py \
            --nside "${NSIDE}" --n-gal "${NGAL}" \
            --z-range "${ZMIN}" "${ZMAX}" \
            "${SIGMA_ARG[@]}" \
            --fit --fit-iters 6 --seed "${SEED}" \
            --out-dir "${OUT}/seed${SEED}"
    done
done
echo "== glass DONE on $(hostname)"
