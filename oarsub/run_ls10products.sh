#!/usr/bin/env bash
#OAR --name smb_ls10prod
#OAR -l /nodes=1/core=8,walltime=12:00:00
#OAR --array 36
#OAR --stdout oarsub/logs/%jobid%.ls10prod.out
#OAR --stderr oarsub/logs/%jobid%.ls10prod.err
#
# FAMILY P -- regenerate the shipped LS10 weight products.
#
# The WEIGHT_ISD1 and WEIGHT_ISD3 columns as shipped are wrong twice over.  They
# were produced by polynomial_ols_decontamination, which is not ISD; and the
# writer rebuilt them as 1/(1 + a_hat . t), a *linear* reconstruction, when ISD's
# own weight is the cumulative product prod_j 1/(1 + F_j(t_j)).  Both are fixed in
# the library, and the scripts now read result["weights"] instead of recomputing.
# This family writes files that match.
#
# NOT family E: E attaches the mock-calibrated LRT, which is what makes it cost
# 14 h a cell.  The weight products need no null, so this runs in minutes and the
# two are kept apart deliberately.
#
# 9 samples x NSIDE {32, 64, 128, 256} = 36 cells, one per array element.  The
# paper's calibrated results use 32 and 64; 128 and 256 are shipped alongside
# them, so they have to carry the same weight convention or a consumer picking
# the finest available map gets the old one.
#
# The third argument overrides the resolution list, so one half can be
# regenerated without re-running the other.  The submitting environment is not
# propagated to jobs on this site, hence positional rather than exported.
#
# It is COMMA separated, not space separated: oarsub takes the script and its
# arguments as one string, so a space inside the list becomes another argument
# and the list silently truncates to its first entry.
#
#   oarsub --project <proj> -S "./oarsub/run_ls10products.sh <tag> [offset] [128,256]"
#
# "auto" in place of the list gives each sample its own resolution: the finest
# NSIDE up to NSIDE_MAX (128) at which its footprint holds MIN_PER_PIXEL (25)
# galaxies per pixel on average.  One cell per sample, array 9.
#
#   oarsub --project <proj> -S "./oarsub/run_ls10products.sh <tag> 0 auto"
set -euo pipefail
cd "$(dirname "$0")/.."
source oarsub/_campaign_env.sh

TAG="${1:-${OAR_ARRAY_ID:-local}}"
OFF="${2:-0}"
NSIDE_LIST="${3:-32,64,128,256}"
campaign_activate_env
campaign_threads >/dev/null

# ISD's stopping rule is a Delta chi^2 normalised on contamination-free mocks;
# without it the threshold is in raw units and the amplitudes overshoot.
ISD_NMOCK="${ISD_NMOCK:-30}"

CATDIR="${SMB_DATA}/sweep/BGS_VLIM_Mstar"

if [ "${NSIDE_LIST}" = "auto" ]; then
    NSIDE_MAX="${NSIDE_MAX:-128}"
    MIN_PER_PIXEL="${MIN_PER_PIXEL:-25}"
    IDX=$(( ${OAR_ARRAY_INDEX:-1} - 1 + OFF ))
    if [ "${IDX}" -ge ${#SMB_SAMPLES[@]} ]; then
        echo "!! cell ${IDX} is past the end of SMB_SAMPLES (${#SMB_SAMPLES[@]}); " \
             "auto mode takes an array of ${#SMB_SAMPLES[@]}" >&2
        exit 1
    fi
    SAMPLE="${SMB_SAMPLES[${IDX}]}"
    # Templates at the finest resolution; the analysis downgrades them to the one
    # the occupancy rule chooses.
    TPLDIR="${SMB_DATA}/systematics/$(printf '%04d' "${NSIDE_MAX}")"
    OUT="${SMB_RESULTS}/ls10_products/${TAG}/auto"
    mkdir -p "${OUT}"
    echo "== cell ${IDX}: ${SAMPLE} NSIDE<=${NSIDE_MAX} at >=${MIN_PER_PIXEL} galaxies/pixel" \
         "isd_n_mocks=${ISD_NMOCK}"
    python "${SMB_PKG}/scripts/run_ls10_analysis.py" \
        --catalog-dir "${CATDIR}" \
        --sample "${SAMPLE}" \
        --template-dir "${TPLDIR}" \
        --nside "${NSIDE_MAX}" \
        --min-per-pixel "${MIN_PER_PIXEL}" \
        --sampler auto \
        --isd-n-mocks "${ISD_NMOCK}" \
        --null-cl-file "${REPO}/matched_spectra" \
        --no-rst \
        --output-dir "${OUT}"
    # The chosen resolution is in the filename; there is exactly one for the sample.
    shopt -s nullglob
    WEIGHTS=("${OUT}/${SAMPLE}"_NSIDE*_WEIGHTS.fits)
    if [ "${#WEIGHTS[@]}" -ne 1 ]; then
        echo "!! expected one ${SAMPLE}_NSIDE*_WEIGHTS.fits in ${OUT}, found ${#WEIGHTS[@]}" >&2
        exit 1
    fi
    python "${REPO}/oarsub/check_ls10_products.py" "${WEIGHTS[0]}" || exit 1
    echo "== ls10products cell ${IDX} DONE on $(hostname)"
    exit 0
fi

IFS=',' read -r -a NSIDES <<< "${NSIDE_LIST}"
# A cell index past the end of SMB_SAMPLES means the array size and the
# resolution list disagree.  Say so rather than dying on an unbound variable.
NN="${#NSIDES[@]}"
IDX=$(( ${OAR_ARRAY_INDEX:-1} - 1 + OFF ))
if [ $(( IDX / NN )) -ge ${#SMB_SAMPLES[@]} ]; then
    echo "!! cell ${IDX}: sample index $(( IDX / NN )) is past the end of" \
         "SMB_SAMPLES (${#SMB_SAMPLES[@]}); the array size must be" \
         "${#SMB_SAMPLES[@]} x ${NN} = $(( ${#SMB_SAMPLES[@]} * NN )) for" \
         "NSIDE list '${NSIDE_LIST}'" >&2
    exit 1
fi
SAMPLE="${SMB_SAMPLES[$(( IDX / NN ))]}"
NSIDE="${NSIDES[$(( IDX % NN ))]}"

TPLDIR="${SMB_DATA}/systematics/$(printf '%04d' "${NSIDE}")"
OUT="${SMB_RESULTS}/ls10_products/${TAG}/NSIDE$(printf '%04d' "${NSIDE}")"
mkdir -p "${OUT}"

echo "== cell ${IDX}: ${SAMPLE} NSIDE=${NSIDE} isd_n_mocks=${ISD_NMOCK}"

python "${SMB_PKG}/scripts/run_ls10_analysis.py" \
    --catalog-dir "${CATDIR}" \
    --sample "${SAMPLE}" \
    --template-dir "${TPLDIR}" \
    --nside "${NSIDE}" \
    --sampler auto \
    --isd-n-mocks "${ISD_NMOCK}" \
    --null-cl-file "${REPO}/matched_spectra" \
    --no-rst \
    --output-dir "${OUT}"

# A family that can exit 0 having done nothing is worse than one that crashes.
python "${REPO}/oarsub/check_ls10_products.py" \
    "${OUT}/${SAMPLE}_NSIDE$(printf '%04d' "${NSIDE}")_WEIGHTS.fits" || exit 1
echo "== ls10products cell ${IDX} DONE on $(hostname)"
