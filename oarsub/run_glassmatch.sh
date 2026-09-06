#!/usr/bin/env bash
#OAR --name smb_glassmatch
#OAR -l /nodes=1/core=4,walltime=06:00:00
#OAR --array 18
#OAR --stdout oarsub/logs/%jobid%.glassmatch.out
#OAR --stderr oarsub/logs/%jobid%.glassmatch.err
#
# FAMILY M -- match each mock to its own sample's large-scale clustering.
#
# This runs BEFORE family E and replaces family B as E's input.  B fitted one
# number per cell against the total pixel variance; the variance is dominated by
# the many small-scale modes while the calibration depends on the few
# large-scale ones, so B tuned the wrong scales.  What a null has to reproduce
# is the two-halo clustering of THIS sample at THIS resolution on THIS
# footprint, which is the scale systematic templates vary on.  There is no
# universal spectrum and no shortcut: one match per cell.
#
# NSIDE 32 and 64 only.  No calibrated statistic in the paper is evaluated at
# 128 or 256, and the scalar fit did not converge there either -- below roughly
# five galaxies per pixel the mock's shot noise alone exceeds the target scatter.
#
#   oarsub --project <proj> -S "./oarsub/run_glassmatch.sh <tag> [n_iter] [n_seeds]"
set -euo pipefail
cd "$(dirname "$0")/.."
source oarsub/_campaign_env.sh

TAG="${1:-${OAR_ARRAY_ID:-local}}"
NITER="${2:-25}"
NSEEDS="${3:-5}"
campaign_activate_env
campaign_threads >/dev/null

NSIDES=(32 64)
IDX=$(( ${OAR_ARRAY_INDEX:-1} - 1 ))
SAMPLE="${SMB_SAMPLES[$(( IDX / 2 ))]}"
NSIDE="${NSIDES[$(( IDX % 2 ))]}"

OUT="${SMB_RESULTS}/glass_match/${TAG}"
mkdir -p "${OUT}"
echo "== cell ${IDX}: ${SAMPLE} NSIDE=${NSIDE}  (${NITER} iterations x ${NSEEDS} seeds)"

python -u characterisation/match_glass_to_data.py \
    --sample "${SAMPLE}" \
    --catalog-dir "${SMB_DATA}/sweep/BGS_VLIM_Mstar" \
    --nside "${NSIDE}" \
    --n-iter "${NITER}" \
    --n-seeds "${NSEEDS}" \
    --out-dir "${OUT}"

# The run reports "NOT converged" whenever a band sits inside its own
# cosmic-variance floor but outside the requested tolerance, which is expected
# at low l.  So the post-condition is that a spectrum was WRITTEN, not that the
# script called itself converged.
PJ="${OUT}/${SAMPLE}_NSIDE$(printf '%04d' "${NSIDE}")_match.json"
python "${REPO}/oarsub/check_match_cell.py" "${PJ}" || exit 1

echo "== glassmatch cell ${IDX} DONE on $(hostname)"
