#!/usr/bin/env bash
#OAR --name smb_lrt
#OAR -l /nodes=1/core=8,walltime=48:00:00
#OAR --array 18
#OAR --stdout oarsub/logs/%jobid%.lrt.out
#OAR --stderr oarsub/logs/%jobid%.lrt.err
#
# FAMILY E -- mock-calibrated LRT, 9 samples x NSIDE {32,64} = 18 cells.
#
# DEPENDS ON FAMILY B.  The published null was drawn from mocks ~25x under-
# clustered relative to the data (mock sigma_hat 0.117 vs data 0.397), which
# makes every p-value optimistic; `p` is also floored at 0.032 by N=31 mocks.
# Pass B's fitted cl_amplitude as the third argument -- running this with the
# 5e-4 default just reproduces the bias it exists to remove.
#
#   oarsub --project <proj> -S "./oarsub/run_lrt.sh <tag> <n_mocks> <cl_amplitude>"
set -euo pipefail
cd "$(dirname "$0")/.."
source oarsub/_campaign_env.sh

TAG="${1:-${OAR_ARRAY_ID:-local}}"
NMOCK="${2:-50}"
CLAMP="${3:-}"
if [ -z "${CLAMP}" ]; then
    echo "!! no cl_amplitude given.  Read it off family B before submitting:" >&2
    echo "   ./oarsub/campaign_status.sh glass <tag>" >&2
    exit 1
fi
campaign_activate_env
campaign_threads >/dev/null

NSIDES=(32 64)
IDX=$(( ${OAR_ARRAY_INDEX:-1} - 1 ))
SAMPLE="${SMB_SAMPLES[$(( IDX / 2 ))]}"
NSIDE="${NSIDES[$(( IDX % 2 ))]}"

CATDIR="${SMB_DATA}/sweep/BGS_VLIM_Mstar"
TPLDIR="${SMB_DATA}/systematics/$(printf '%04d' "${NSIDE}")"
OUT="${SMB_RESULTS}/ls10_mocklrt/${TAG}/NSIDE$(printf '%04d' "${NSIDE}")"
mkdir -p "${OUT}"

echo "== cell ${IDX}: ${SAMPLE} NSIDE=${NSIDE} n_mocks=${NMOCK} cl_amplitude=${CLAMP}"
python "${SMB_PKG}/scripts/run_ls10_analysis.py" \
    --catalog-dir "${CATDIR}" \
    --sample "${SAMPLE}" \
    --template-dir "${TPLDIR}" \
    --nside "${NSIDE}" \
    --sampler auto \
    --lrt-null-mocks "${NMOCK}" \
    --lrt-null-cl-amplitude "${CLAMP}" \
    --resume-null \
    --no-rst \
    --output-dir "${OUT}"

echo "== lrt cell ${IDX} DONE on $(hostname)"
