#!/usr/bin/env bash
# =============================================================================
# Submit one or more job families.  Runs ON THE CLUSTER FRONTEND.
#
#   ./oarsub/submit_campaign.sh <tag> A            # env build, first and alone
#   ./oarsub/submit_campaign.sh <tag> B C D F      # the independent families
#   ./oarsub/submit_campaign.sh <tag> E 3.8e-2     # LRT, after reading B
#
# The submitting environment is NOT propagated to OAR jobs on this site, so
# every setting a job needs is passed positionally rather than exported.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
source oarsub/_campaign_env.sh
PROJECT="$(campaign_project)"
mkdir -p oarsub/logs "${SMB_RESULTS}"

TAG="${1:?usage: submit_campaign.sh <tag> <family...>}"; shift
[ $# -gt 0 ] || { echo "!! no family given (A B C D E F)"; exit 1; }

sub () {   # sub <script+args> [extra oarsub flags...]
    local cmd="$1"; shift
    echo "-- oarsub --project ${PROJECT} $* -S \"${cmd}\""
    oarsub --project "${PROJECT}" "$@" -S "${cmd}" | grep -E 'OAR_JOB_ID|OAR_ARRAY_ID' || true
}

for fam in "$@"; do
  case "${fam}" in
    A) echo "== A  environment build (1 job, core=16, 3 h)"
       sub "./oarsub/build_env.sh" ;;
    B) echo "== B  GLASS calibration (array 36, core=4, 2 h)"
       sub "./oarsub/run_glass.sh ${TAG} 5" ;;
    C) echo "== C  variance cube (array 20 x 5 n_sys = 100 cells, core=4, 4 h)"
       sub "./oarsub/run_variance.sh ${TAG} 2000" ;;
    D) echo "== D  benchmark grid (1 job, cpumodel-pinned, core=8, 12 h)"
       sub "./oarsub/run_bench.sh ${TAG} 5" ;;
    E) # E is the one family that takes a value from another: B's fitted amplitude.
       CLAMP="${1:-}"
       if [ -z "${CLAMP}" ]; then
           echo "!! family E needs B's cl_amplitude:" >&2
           echo "   ./oarsub/submit_campaign.sh ${TAG} E <cl_amplitude>" >&2
           exit 1
       fi
       shift
       echo "== E  mock-calibrated LRT (array 18, core=8, 48 h, cl_amplitude=${CLAMP})"
       sub "./oarsub/run_lrt.sh ${TAG} 50 ${CLAMP}" ;;
    F) echo "== F  simulation tests (array 75 x 2 seeds, core=8, 12 h)"
       sub "./oarsub/run_simtests.sh ${TAG} 2" ;;
    *) echo "!! unknown family '${fam}' (expected A B C D E F)"; exit 1 ;;
  esac
done

echo
echo "== queue"
oarstat -u "${USER}" 2>/dev/null | tail -n +1 | head -20 || true
echo "== progress:  ./oarsub/campaign_status.sh ${TAG}"
