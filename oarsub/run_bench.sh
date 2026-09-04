#!/usr/bin/env bash
#OAR --name smb_bench
#OAR -l /cpumodel=1/nodes=1/core=8,walltime=12:00:00
#OAR --stdout oarsub/logs/%jobid%.bench.out
#OAR --stderr oarsub/logs/%jobid%.bench.err
#
# FAMILY D -- timing benchmark grid.
#
# /cpumodel=1 is not decoration: dahu nodes are heterogeneous and this job IS
# the measurement, so every row must come off one CPU model or the table mixes
# hardware.  For the same reason it is a single job, not an array: array
# elements land on whatever is free.
#
# The published table stops at NSIDE 64 and every MCMC row is a single draw
# (mad = 0, i.e. no dispersion at all).  This extends to 128/256 and repeats.
#
#   oarsub --project <proj> -S "./oarsub/run_bench.sh <tag> <n_repeat>"
set -euo pipefail
cd "$(dirname "$0")/.."
source oarsub/_campaign_env.sh

TAG="${1:-${OAR_JOB_ID:-local}}"
NREP="${2:-5}"
campaign_activate_env
NCORE="$(campaign_threads)"

OUT="${SMB_RESULTS}/benchmarks/${TAG}"
mkdir -p "${OUT}"
# Record the hardware alongside the numbers -- a timing table without the CPU
# it was measured on is not reproducible.
{ echo "host=$(hostname)"; echo "ncore=${NCORE}"; echo "date=$(date -Is)";
  grep -m1 'model name' /proc/cpuinfo || true; } > "${OUT}/machine.txt"

echo "== benchmark grid, n_repeat=${NREP}, ${NCORE} cores"
python benchmark/benchmark_pipeline.py \
    --nsides 16 32 64 128 256 \
    --n-sys 5 11 \
    --n-repeat "${NREP}" \
    --nuts 400 \
    --out-dir "${OUT}"

echo "== bench DONE on $(hostname)"
