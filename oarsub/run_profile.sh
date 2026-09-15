#!/usr/bin/env bash
#OAR --name smb_profile
#OAR -l /nodes=1/core=8,walltime=03:00:00
#OAR --stdout oarsub/logs/%jobid%.profile.out
#OAR --stderr oarsub/logs/%jobid%.profile.err
#
# The profile behind the "wall time by library" table of sys_mapping docs/coverage.rst:
# run_ls10_analysis.py on log M* >= 10.0 at NSIDE 32 under cProfile, with the XLA
# compilation log, on a whole node.
#
#   oarsub --project <proj> -S "./oarsub/run_profile.sh <tag>"
#
# Then, from the sys_mapping checkout, with the pulled files:
#   python scripts/coverage_report.py coverage.json --profile ls10_ns32.prof \
#       --compile-log run.err --profile-label "..."
set -euo pipefail
cd "$(dirname "$0")/.."
source oarsub/_campaign_env.sh

TAG="${1:-${OAR_JOB_ID:-local}}"
campaign_activate_env
NCORE="$(campaign_threads)"

OUT="${SMB_RESULTS}/profile/${TAG}"
mkdir -p "${OUT}"
{ echo "host=$(hostname)"; echo "ncore=${NCORE}"; echo "start=$(date -Is)";
  grep -m1 'model name' /proc/cpuinfo || true;
  echo "loadavg_start=$(cut -d' ' -f1-3 /proc/loadavg)";
  git -C "${SMB_PKG}" rev-parse --short HEAD 2>/dev/null | sed 's/^/pkg_head=/' || true; } > "${OUT}/machine.txt"

SAMPLE="LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238"
DATA="${SMB_DATA}"
JAX_LOG_COMPILES=1 python -m cProfile -o "${OUT}/ls10_ns32.prof" \
    "${SMB_PKG}/scripts/run_ls10_analysis.py" \
    --catalog-dir "${DATA}/sweep/BGS_VLIM_Mstar" --sample "${SAMPLE}" \
    --template-dir "${DATA}/systematics/0032" --nside 32 \
    --significance-n-mocks 40 --isd-n-mocks 10 \
    --null-cl-file "${REPO}/matched_spectra" \
    --nuts-warmup 500 --nuts-samples 500 --no-rst --force \
    --output-dir "${OUT}/out" > "${OUT}/run.out" 2> "${OUT}/run.err"
{ echo "end=$(date -Is)"; echo "loadavg_end=$(cut -d' ' -f1-3 /proc/loadavg)"; } >> "${OUT}/machine.txt"
touch "${OUT}/DONE"
