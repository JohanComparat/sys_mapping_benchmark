#!/usr/bin/env bash
#OAR --name smb_respages
#OAR -l /nodes=1/core=8,walltime=48:00:00
#OAR --array 5
#OAR --stdout oarsub/logs/%jobid%.respages.out
#OAR --stderr oarsub/logs/%jobid%.respages.err
#
# FAMILY R -- the runs behind the sys_mapping results pages.
#
# One array element per page run, each the documented command of that page on the
# campaign's package snapshot:
#
#   1 progressive   results_progressive_contamination  45 mocks, NSIDE 32
#   2 systests      results_systematic_tests           32 configs x 6 methods, NSIDE 32
#   3 mockanalysis  results_mock_analysis              100 synthetic mocks, NSIDE 64
#   4 simtests64    results_simulation_tests           GLASS + Uchuu, NSIDE 64
#   5 simtests32    results_simulation_tests           GLASS + Uchuu, NSIDE 32
#
# MCMC-add is the analytic posterior and MCMC-comb NUTS (1000 warmup, 1000 draws per
# chain); every ISD run is calibrated on uncontaminated realisations by the script.
# The GLASS universe of the simulation tests is the parametric spectrum 5e-4; the
# Uchuu null is the matched spectrum of the LS10 sample the Uchuu mock was built for.
# The Uchuu catalogues are staged under ${HOME}/data/Uchuu.
#
# 48 h for every element: an array shares one walltime, and the 100-mock run is the
# longest.  run_mock_analysis.py --resume keeps the mocks already fitted, so a
# resubmission of element 3 continues rather than restarts.
#
# Each element writes ${SMB_RESULTS}/results_pages/<tag>/<cell>/ with run.log and,
# on success, DONE.
#
#   oarsub --project <proj> -S "./oarsub/run_results_pages.sh <tag> [offset]"
set -euo pipefail
cd "$(dirname "$0")/.."
source oarsub/_campaign_env.sh

TAG="${1:-${OAR_ARRAY_ID:-local}}"
OFF="${2:-0}"
campaign_activate_env
NCORE="$(campaign_threads)"

CELLS=(progressive systests mockanalysis simtests64 simtests32)
IDX=$(( ${OAR_ARRAY_INDEX:-1} - 1 + OFF ))
CELL="${CELLS[${IDX}]}"
OUT="${SMB_RESULTS}/results_pages/${TAG}/${CELL}"
mkdir -p "${OUT}"
rm -f "${OUT}/DONE"
[ -r "${REPO}/pkg_version.txt" ] && cp "${REPO}/pkg_version.txt" "${OUT}/"
{ echo "host=$(hostname)"; echo "ncore=${NCORE}"; echo "start=$(date -Is)";
  grep -m1 'model name' /proc/cpuinfo || true; } > "${OUT}/machine.txt"

S="${SMB_PKG}/scripts"
UCHUU_TAG="MOCK_VLIM_ANY_10.65_Mstar_12.0_0.05_z_0.26_N_0923373"
UCHUU="${HOME}/data/Uchuu/FullSky/mock_catalogues/${UCHUU_TAG}/${UCHUU_TAG}_DATA.fits.gz"
UCHUU_CL_SAMPLE="LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228"

echo "== results pages cell ${IDX} (${CELL}) -> ${OUT}"
run () { "$@" 2>&1 | tee -a "${OUT}/run.log"; }

case "${CELL}" in
  progressive)
    run python "${S}/run_mock_analysis_progressive.py" \
        --nside 32 --n-sys 4 --n-mocks-per-case 5 \
        --snr-threshold 2.0 --sigma 0.15 \
        --output-dir "${OUT}" ;;
  systests)
    run python "${S}/run_systematic_tests.py" \
        --nside 32 --output-dir "${OUT}" ;;
  mockanalysis)
    run python "${S}/run_mock_analysis.py" \
        --synthetic --n-mocks 100 --n-sys 3 --nside 64 --resume \
        --output-dir "${OUT}" ;;
  simtests64|simtests32)
    NS="${CELL#simtests}"
    NGAL=500000
    run python "${S}/run_simulation_tests.py" \
        --nside "${NS}" --n-glass "${NGAL}" \
        --methods OLS ISD-1 ISD-3 ElasticNet MCMC-add MCMC-comb \
        --cl-amplitude 5e-4 \
        --uchuu-data "${UCHUU}" \
        --uchuu-null-cl-file "${REPO}/matched_spectra/${UCHUU_CL_SAMPLE}_NSIDE$(printf '%04d' "${NS}")_match.json" \
        --syst-dir "${SMB_DATA}/systematics" \
        --output-dir "${OUT}"
    TPL=""; [ "${NS}" = 32 ] && TPL="--no-templates"
    run python "${S}/plot_simulation_tests.py" \
        --nside "${NS}" ${TPL} \
        --results-json "${OUT}/nside$(printf '%04d' "${NS}")/results_summary.json" \
        --syst-dir "${SMB_DATA}/systematics/" \
        --uchuu-data "${UCHUU}" --cl-amplitude 5e-4 \
        --output-dir "${OUT}/figures" ;;
  *) echo "!! no cell ${IDX}"; exit 1 ;;
esac

# tee hides the exit status of the python it follows; pipefail restores it, so a
# failed run stops here without writing the sentinel.
echo "end=$(date -Is)" >> "${OUT}/machine.txt"
touch "${OUT}/DONE"
echo "== results pages cell ${IDX} (${CELL}) DONE on $(hostname)"
