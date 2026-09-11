#!/usr/bin/env bash
#OAR --name smb_simtests
#OAR -l /nodes=1/core=8,walltime=24:00:00
#OAR --array 75
#OAR --stdout oarsub/logs/%jobid%.simtests.out
#OAR --stderr oarsub/logs/%jobid%.simtests.err
#
# FAMILY F -- simulation recovery tests over seeds.
#
# 3 NSIDE x 25 elements x 2 seeds each = 150 runs (50 seeds per NSIDE).
# All six methods, ISD-3 included: it was absent from the runner's --methods
# choices until the rewrite, so no ISD-3 column exists in any earlier campaign.  Every
# published frac_helped is a fraction of ten from ONE realisation, so the +0.73
# correlation the docs report has no error bar; seeds are the whole point here.
#
# GLASS only: the Uchuu mocks are ~40 GB and deliberately not staged.
#
# 24 h, not 12: at 12 h the NSIDE 32 and 64 tiers finished 50/50 but 25 of the
# NSIDE-128 elements were killed at the wall, leaving that tier at 12/50.  The
# cost is superlinear in pixel count and the two seeds per element share a job.
#
# --syst-dir is the PARENT of the NSIDE directories: load_systematic_maps
# appends "%04d" % nside itself.  Passing the NSIDE dir gave .../0128/0128/.
#
#   oarsub --project <proj> -S "./oarsub/run_simtests.sh <tag> <seeds_per_task> [offset]"
set -euo pipefail
cd "$(dirname "$0")/.."
source oarsub/_campaign_env.sh

TAG="${1:-${OAR_ARRAY_ID:-local}}"
SPT="${2:-2}"
# Offset into the 75-element cell list, so F can be submitted in chunks that fit
# the queue.  $OAR_ARRAY_INDEX restarts at 1 for every chunk.
OFF="${3:-0}"
campaign_activate_env
campaign_threads >/dev/null

# ISD's stopping rule is a Delta chi^2 normalised on contamination-free mocks.
# Without it the threshold is in raw Delta chi^2 units, the iteration keeps
# re-selecting templates it has already corrected, and the amplitudes overshoot --
# so the calibration is not optional here even though the flag is.  The null
# depends on the footprint, resolution and surface density but not on the injected
# contamination, so it is computed once per mock source and reused across all nine
# configurations of a cell.
ISD_NMOCK="${ISD_NMOCK:-30}"

NSIDES=(32 64 128)
CHUNKS=25
IDX=$(( ${OAR_ARRAY_INDEX:-1} - 1 + OFF ))
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
        --methods OLS ISD-1 ISD-3 ElasticNet MCMC-add MCMC-comb \
        --isd-n-mocks "${ISD_NMOCK}" \
        --syst-dir "${SMB_DATA}/systematics" \
        --seed "${SEED}" \
        --output-dir "${OUT}"
done
echo "== simtests cell ${IDX} DONE on $(hostname)"
