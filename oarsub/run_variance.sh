#!/usr/bin/env bash
#OAR --name smb_variance
#OAR -l /nodes=1/core=4,walltime=04:00:00
#OAR --array 20
#OAR --stdout oarsub/logs/%jobid%.variance.out
#OAR --stderr oarsub/logs/%jobid%.variance.err
#
# FAMILY C -- variance-inflation cube.
#
# 100 cells = 2 NSIDE x 5 n_sys x 5 Cl slopes x 2 sigma_clus, at n_real=2000.
# One array element per (NSIDE, slope, sigma_clus); the 5 n_sys values run
# inside a single call because the script takes them as a list.  20 elements is
# well under the 94 GRICAD leaves in the waiting queue.
#
# The laptop grid was 6 cells at n_real=150 -- an FPR(3 sigma) estimate built on
# three events -- and a later scan overwrote it in place.  Outputs here are
# run-tagged so that cannot recur.
#
#   oarsub --project <proj> -S "./oarsub/run_variance.sh <tag> <n_real>"
set -euo pipefail
cd "$(dirname "$0")/.."
source oarsub/_campaign_env.sh

TAG="${1:-${OAR_ARRAY_ID:-local}}"
NREAL="${2:-2000}"
campaign_activate_env
campaign_threads >/dev/null

NSIDES=(32 64)
SLOPES=(1.0 1.5 2.0 2.5 3.0)
SIGCLUS=(0.2 0.4)
NSYS=(3 5 7 9 11)

IDX=$(( ${OAR_ARRAY_INDEX:-1} - 1 ))
NSL=${#SLOPES[@]}; NSC=${#SIGCLUS[@]}
NSIDE="${NSIDES[$(( IDX / (NSL * NSC) ))]}"
REM=$(( IDX % (NSL * NSC) ))
SLOPE="${SLOPES[$(( REM / NSC ))]}"
SC="${SIGCLUS[$(( REM % NSC ))]}"

CELL="ns${NSIDE}_slope${SLOPE}_sc${SC}"
echo "== cell ${IDX}: ${CELL} n_sys=${NSYS[*]} n_real=${NREAL}"

python characterisation/run_variance_inflation.py \
    --nsides "${NSIDE}" \
    --n-sys "${NSYS[@]}" \
    --n-real "${NREAL}" \
    --sigma-clus "${SC}" \
    --cl-slopes "${SLOPE}" \
    --seed $(( 1234 + IDX )) \
    --tag "${CELL}" \
    --out-dir "${SMB_RESULTS}/variance_inflation/${TAG}"

echo "== variance cell ${IDX} DONE on $(hostname)"
