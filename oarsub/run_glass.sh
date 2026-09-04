#!/usr/bin/env bash
#OAR --name smb_glass
#OAR -l /nodes=1/core=4,walltime=02:00:00
#OAR --array 36
#OAR --stdout oarsub/logs/%jobid%.glass.out
#OAR --stderr oarsub/logs/%jobid%.glass.err
#
# FAMILY B -- GLASS clustering calibration, 9 LS10 samples x 4 NSIDE = 36 cells.
#
# Each cell root-finds the `cl_amplitude` whose mock sigma_hat reproduces that
# sample's measured sigma_hat, then repeats over seeds so the answer carries a
# scatter.  The laptop run covered ONE cell (2.8 %) with a single seed and a
# 3-point scan, which is not a fit.
#
#   oarsub --project <proj> -S "./oarsub/run_glass.sh <tag> <nseeds>"
set -euo pipefail
cd "$(dirname "$0")/.."
source oarsub/_campaign_env.sh

TAG="${1:-${OAR_ARRAY_ID:-local}}"
NSEEDS="${2:-5}"
campaign_activate_env
campaign_threads >/dev/null

# OAR_ARRAY_INDEX is 1-based.  Cell order: sample-major, NSIDE-minor.
IDX=$(( ${OAR_ARRAY_INDEX:-1} - 1 ))
NN=${#SMB_NSIDES[@]}
SAMPLE="${SMB_SAMPLES[$(( IDX / NN ))]}"
NSIDE="${SMB_NSIDES[$(( IDX % NN ))]}"

# LS10_VLIM_ANY_<logM>_Mstar_12.0_<zmin>_z_<zmax>_N_<ngal>
IFS='_' read -r -a F <<< "${SAMPLE}"
ZMIN="${F[6]}"; ZMAX="${F[8]}"; NGAL=$(( 10#${F[10]} ))

# The target is this sample's own residual scatter, not the fiducial cell's.
# sigma_hat_ols is the additive-model scatter the calibration is matching.
PJ="${SMB_PKG}/data/sys_weights/${SAMPLE}_NSIDE$(printf '%04d' "${NSIDE}")_params.json"
SIGMA="$(python - "${PJ}" <<'PYEOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
if not p.exists():
    print("")                      # let the script keep its own default
else:
    d = json.load(p.open())
    print(d.get("sigma_hat_ols") or d.get("sigma_hat_add") or "")
PYEOF
)"
SIGMA_ARG=()
if [ -n "${SIGMA}" ]; then
    SIGMA_ARG=(--sigma-hat-data "${SIGMA}")
else
    echo "!! no params.json at ${PJ}; using the script default sigma_hat_data"
fi

OUT="${SMB_RESULTS}/glass_calibration/${TAG}/${SAMPLE}_NSIDE$(printf '%04d' "${NSIDE}")"
mkdir -p "${OUT}"
echo "== cell ${IDX}: ${SAMPLE} NSIDE=${NSIDE} n_gal=${NGAL} z=[${ZMIN},${ZMAX}] sigma=${SIGMA:-default}"

# One invocation per seed; the aggregate scatter over seeds is the point.
for k in $(seq 0 $(( NSEEDS - 1 ))); do
    SEED=$(( 11 + 1000 * k ))
    echo "-- seed ${SEED}"
    python characterisation/calibrate_glass_clustering.py \
        --nside "${NSIDE}" --n-gal "${NGAL}" \
        --z-range "${ZMIN}" "${ZMAX}" \
        "${SIGMA_ARG[@]}" \
        --seed "${SEED}" \
        --out-dir "${OUT}/seed${SEED}"
done
echo "== glass cell ${IDX} DONE on $(hostname)"
