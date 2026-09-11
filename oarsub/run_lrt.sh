#!/usr/bin/env bash
#OAR --name smb_lrt
#OAR -l /nodes=1/core=8,walltime=48:00:00
#OAR --array 18
#OAR --stdout oarsub/logs/%jobid%.lrt.out
#OAR --stderr oarsub/logs/%jobid%.lrt.err
#
# FAMILY E -- mock-calibrated LRT, 9 samples x NSIDE {32,64} = 18 cells.
#
# DEPENDS ON FAMILY B.  The published null was drawn from mocks far less
# clustered than the data, which makes every p-value optimistic; `p` is also
# floored at 0.032 by N=31 mocks.
#
# The amplitude is read PER CELL from family B, not passed as one number:
# B measured 0.104 for the logM 9.0 sample and 0.017 for logM 11.0, an order of
# magnitude apart, so one global value would be wrong for almost every sample.
# Give the calibration tag; a fallback may follow for cells B did not fit.
#
#   oarsub --project <proj> -S "./oarsub/run_lrt.sh <tag> <n_mocks> <calib_tag> [fallback]"
set -euo pipefail
cd "$(dirname "$0")/.."
source oarsub/_campaign_env.sh

TAG="${1:-${OAR_ARRAY_ID:-local}}"
NMOCK="${2:-50}"
CALIB_TAG="${3:-}"
FALLBACK="${4:-}"
if [ -z "${CALIB_TAG}" ]; then
    echo "!! no calibration tag given.  Run family B first, then:" >&2
    echo "   ./oarsub/submit_campaign.sh <tag> E <calib_tag>" >&2
    exit 1
fi
campaign_activate_env
campaign_threads >/dev/null

NSIDES=(32 64)
# An optional comma-separated cell list (5th argument) selects a subset, so a
# handful of cells can be re-run without recomputing the ones that are already
# right -- an LRT cell costs 14 h and the grid has 18 of them.  Same pattern as
# family M.  Without it the array index is the cell index, as before.
CELLS="${5:-}"
if [ -n "${CELLS}" ]; then
    IFS=',' read -r -a _CELL_ARR <<< "${CELLS}"
    IDX="${_CELL_ARR[$(( ${OAR_ARRAY_INDEX:-1} - 1 ))]}"
else
    IDX=$(( ${OAR_ARRAY_INDEX:-1} - 1 ))
fi
SAMPLE="${SMB_SAMPLES[$(( IDX / 2 ))]}"
NSIDE="${NSIDES[$(( IDX % 2 ))]}"

CATDIR="${SMB_DATA}/sweep/BGS_VLIM_Mstar"
TPLDIR="${SMB_DATA}/systematics/$(printf '%04d' "${NSIDE}")"
OUT="${SMB_RESULTS}/ls10_mocklrt/${TAG}/NSIDE$(printf '%04d' "${NSIDE}")"
mkdir -p "${OUT}"

# Median over B's converged seeds for THIS sample and NSIDE.
CELLDIR="${SMB_RESULTS}/glass_calibration/${CALIB_TAG}/${SAMPLE}_NSIDE$(printf '%04d' "${NSIDE}")"
CLAMP="$(python "${REPO}/oarsub/median_amplitude.py" "${CELLDIR}")"
if [ -z "${CLAMP}" ]; then
    if [ -n "${FALLBACK}" ]; then
        echo "!! family B did not converge for this cell; using fallback ${FALLBACK}"
        CLAMP="${FALLBACK}"
    else
        echo "!! no converged family-B amplitude for ${SAMPLE} NSIDE=${NSIDE}" >&2
        echo "   (calibration tag '${CALIB_TAG}'); pass a fallback as the 4th argument" >&2
        exit 1
    fi
fi

# A matched spectrum for this cell supersedes the scalar amplitude entirely: the
# scalar was fitted to total pixel variance, which the small scales dominate,
# while the calibration depends on the large ones.
MATCH_DIR="${SMB_RESULTS}/glass_match/${CALIB_TAG}"
MATCH_ARG=()
# Any matched spectrum for this SAMPLE will do -- load_matched_cl picks the finest
# *validated* one and falls through when the fit at this resolution did not pass,
# because the spectrum belongs to the sample and its footprint and resolution limits
# what can be checked rather than what can be used.  Testing for the exact NSIDE here
# would send a cell to the parametric null merely because its own resolution failed.
if compgen -G "${MATCH_DIR}/${SAMPLE}_NSIDE*_match.json" >/dev/null; then
    MATCH_ARG=(--lrt-null-cl-file "${MATCH_DIR}")
    echo "== cell ${IDX}: ${SAMPLE} NSIDE=${NSIDE} n_mocks=${NMOCK} "\
         "null=MATCHED spectrum from ${CALIB_TAG}"
else
    echo "== cell ${IDX}: ${SAMPLE} NSIDE=${NSIDE} n_mocks=${NMOCK} "\
         "null=parametric, cl_amplitude=${CLAMP} (no matched spectrum in ${MATCH_DIR})"
fi
python "${SMB_PKG}/scripts/run_ls10_analysis.py" \
    --catalog-dir "${CATDIR}" \
    --sample "${SAMPLE}" \
    --template-dir "${TPLDIR}" \
    --nside "${NSIDE}" \
    --sampler auto \
    --lrt-null-mocks "${NMOCK}" \
    --lrt-null-cl-amplitude "${CLAMP}" \
    "${MATCH_ARG[@]}" \
    --no-rst \
    --output-dir "${OUT}"

# --resume-null is deliberately NOT used.  It MERGES new draws into the stored
# null_lambda, so resuming a null built at 5e-4 with draws at the corrected
# amplitude would blend two different clustering amplitudes into one
# distribution -- the opposite of what this family exists to do.  It also
# silently skips all work when the output directory is fresh, which is how the
# first pass of E "succeeded" in 40 seconds having computed nothing.

# Verify the cell actually produced a mock-calibrated null.  A family that can
# exit 0 having done nothing is worse than one that crashes.
PJ="${OUT}/${SAMPLE}_NSIDE$(printf '%04d' "${NSIDE}")_params.json"
python "${REPO}/oarsub/check_lrt_cell.py" "${PJ}" "${NMOCK}" || exit 1

echo "== lrt cell ${IDX} DONE on $(hostname)"
