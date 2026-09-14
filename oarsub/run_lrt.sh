#!/usr/bin/env bash
#OAR --name smb_lrt
#OAR -l /nodes=1/core=8,walltime=48:00:00
#OAR --array 18
#OAR --stdout oarsub/logs/%jobid%.lrt.out
#OAR --stderr oarsub/logs/%jobid%.lrt.err
#
# FAMILY E -- mock-calibrated LRT, 9 samples x NSIDE {32,64} = 18 cells.
#
# The null is drawn from each sample's own matched spectrum in matched_spectra/,
# fitted band by band to the data and validated on held-out seeds.  A scalar
# amplitude cannot stand in for it: the default under-clusters LS10 about 25x in
# variance, and one number cannot set the shape of a spectrum, which is what the
# error inflation depends on.  run_ls10_analysis.py refuses to build a null without
# one, so this family has no fallback to a parametric null.
#
#   oarsub --project <proj> -S "./oarsub/run_lrt.sh <tag> <n_mocks> [cells]"
set -euo pipefail
cd "$(dirname "$0")/.."
source oarsub/_campaign_env.sh

TAG="${1:-${OAR_ARRAY_ID:-local}}"
NMOCK="${2:-50}"
campaign_activate_env
campaign_threads >/dev/null

NSIDES=(32 64)
# An optional comma-separated cell list (3rd argument) selects a subset, so a
# handful of cells can be re-run without recomputing the ones that are already
# right -- an LRT cell costs 14 h and the grid has 18 of them.  Same pattern as
# family M.  Without it the array index is the cell index, as before.
CELLS="${3:-}"
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

# The spectra travel with the campaign snapshot, so a tag records which it used.
MATCH_DIR="${REPO}/matched_spectra"
echo "== cell ${IDX}: ${SAMPLE} NSIDE=${NSIDE} n_mocks=${NMOCK} null=matched spectrum"
python "${SMB_PKG}/scripts/run_ls10_analysis.py" \
    --catalog-dir "${CATDIR}" \
    --sample "${SAMPLE}" \
    --template-dir "${TPLDIR}" \
    --nside "${NSIDE}" \
    --sampler auto \
    --lrt-null-mocks "${NMOCK}" \
    --null-cl-file "${MATCH_DIR}" \
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
