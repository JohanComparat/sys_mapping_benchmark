#!/usr/bin/env bash
#OAR --name smb_simtests
#OAR -l /nodes=1/core=8,walltime=12:00:00
#OAR --array 75
#OAR --stdout oarsub/logs/%jobid%.simtests.out
#OAR --stderr oarsub/logs/%jobid%.simtests.err
#
# FAMILY F -- simulation recovery tests over seeds.
#
# 3 NSIDE x 25 elements x 2 seeds each = 150 runs (50 seeds per NSIDE).  Every
# published frac_helped is a fraction of ten from ONE realisation, so the +0.73
# correlation the docs report has no error bar; seeds are the whole point here.
#
# GLASS only: the Uchuu mocks are ~40 GB and deliberately not staged.
#
#   oarsub --project <proj> -S "./oarsub/run_simtests.sh <tag> <seeds_per_task>"
set -euo pipefail
cd "$(dirname "$0")/.."
source oarsub/_campaign_env.sh

TAG="${1:-${OAR_ARRAY_ID:-local}}"
SPT="${2:-2}"
campaign_activate_env
campaign_threads >/dev/null

NSIDES=(32 64 128)
CHUNKS=25
IDX=$(( ${OAR_ARRAY_INDEX:-1} - 1 ))
NSIDE="${NSIDES[$(( IDX / CHUNKS ))]}"
CHUNK=$(( IDX % CHUNKS ))

for j in $(seq 0 $(( SPT - 1 ))); do
    SEED=$(( CHUNK * SPT + j ))
    OUT="${SMB_RESULTS}/simulations/${TAG}/seed$(printf '%03d' "${SEED}")"
    echo "-- NSIDE=${NSIDE} seed=${SEED} -> ${OUT}"
    python "${SMB_PKG}/scripts/run_simulation_tests.py" \
        --nside "${NSIDE}" \
        --n-glass 500000 \
        --glass-only \
        --methods OLS ISD-1 ElasticNet MCMC-add MCMC-comb \
        --syst-dir "${SMB_DATA}/systematics/$(printf '%04d' "${NSIDE}")" \
        --seed "${SEED}" \
        --output-dir "${OUT}"
done
echo "== simtests cell ${IDX} DONE on $(hostname)"
