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
REPO_LOGS="$(pwd)/oarsub/logs"
mkdir -p "${REPO_LOGS}" "${SMB_RESULTS}"

TAG="${1:?usage: submit_campaign.sh <tag> <family...>}"; shift
[ $# -gt 0 ] || { echo "!! no family given (A B C D E F)"; exit 1; }

# Submit from an immutable per-tag snapshot of oarsub/, never from the working
# tree.  bash reads a job script incrementally, so editing a run_*.sh while its
# job is running makes the job resume at a stale offset -- in practice it dies
# with "error reading input file: Stale file handle", which is exactly how the
# first pass of family B was lost.  A snapshot also records what a given tag
# actually ran, which the working tree stops being the moment anything changes.
SNAP="${WORK}/campaigns/${TAG}"
mkdir -p "${SNAP}"
rsync -a --delete --exclude logs oarsub/ "${SNAP}/oarsub/"
ln -sfn "${REPO_LOGS}" "${SNAP}/oarsub/logs"
rsync -a --exclude '.git' --exclude '__pycache__' --exclude 'oarsub' \
      --exclude 'results' ./ "${SNAP}/" 2>/dev/null || true
echo "== snapshot: ${SNAP}"
cd "${SNAP}"

# GRICAD refuses any submission that would leave more than 100 jobs waiting.
# Ask how much room there is rather than finding out from a rejection.
MAX_WAITING=100
campaign_waiting () {
    oarstat -u "${USER}" --json 2>/dev/null | python3 -c \
      'import json,sys; d=json.load(sys.stdin); print(sum(v["state"]=="Waiting" for v in d.values()))' \
      2>/dev/null || echo 0
}
campaign_free_slots () {
    local w; w="$(campaign_waiting)"
    echo $(( MAX_WAITING - w ))
}

sub () {   # sub <script+args> [extra oarsub flags...]
    local cmd="$1"; shift
    echo "-- oarsub --project ${PROJECT} $* -S \"${cmd}\""
    local out rc
    out="$(oarsub --project "${PROJECT}" "$@" -S "${cmd}" 2>&1)"; rc=$?
    # A refused submission must be loud.  Swallowing it leaves a family silently
    # absent from a campaign that otherwise looks like it was submitted.
    if [ "${rc}" -ne 0 ] || ! grep -q 'OAR_JOB_ID' <<< "${out}"; then
        echo "!! submission FAILED:" >&2
        sed 's/^/   /' <<< "${out}" >&2
        return 1
    fi
    grep -cE 'OAR_JOB_ID' <<< "${out}" | sed 's/^/   submitted /;s/$/ job(s)/'
}

while [ $# -gt 0 ]; do
  fam="$1"; shift
  case "${fam}" in
    A) echo "== A  environment build (1 job, core=16, 3 h)"
       sub "./oarsub/build_env.sh" ;;
    B) echo "== B  GLASS calibration (1 job, 36 cells x 5 seeds, core=4, 3 h)"
       sub "./oarsub/run_glass.sh ${TAG} 5 0 35" ;;
    C) echo "== C  variance cube (array 20 x 5 n_sys = 100 cells, core=4, 4 h)"
       sub "./oarsub/run_variance.sh ${TAG} 2000" ;;
    D) echo "== D  benchmark grid (1 job, cpumodel-pinned, core=8, 48 h)"
       sub "./oarsub/run_bench.sh ${TAG} 5 2" ;;
    E) # E is the one family that takes a value from another: B's fitted amplitude.
       CALIB="${1:-}"
       if [ -z "${CALIB}" ]; then
           echo "!! family E needs family B's calibration tag:" >&2
           echo "   ./oarsub/submit_campaign.sh ${TAG} E <calib_tag> [fallback_amplitude]" >&2
           exit 1
       fi
       shift
       FB=""
       if [[ "${1:-}" =~ ^[0-9.eE+-]+$ ]]; then FB="$1"; shift; fi
       echo "== E  mock-calibrated LRT (array 18, core=8, 48 h, per-cell amplitude from ${CALIB})"
       sub "./oarsub/run_lrt.sh ${TAG} 50 ${CALIB} ${FB}" ;;
    F) # 75 elements rarely fit beside B and C, so F is submitted in chunks with
       # an explicit offset.  Re-run `submit_campaign.sh <tag> F` as the queue
       # drains; campaign_status.sh names the seeds still missing.
       TOTAL_F=75
       FREE="$(campaign_free_slots)"
       if [ "${FREE}" -gt 5 ]; then FREE=$(( FREE - 5 )); fi   # keep a small margin
       # An optional numeric offset may follow "F"; anything else is a family.
       OFF=0
       if [[ "${1:-}" =~ ^[0-9]+$ ]]; then OFF="$1"; shift; fi
       N=$(( TOTAL_F - OFF ))
       if [ "${N}" -gt "${FREE}" ]; then N="${FREE}"; fi
       if [ "${OFF}" -ge "${TOTAL_F}" ]; then
           echo "== F  all ${TOTAL_F} elements already submitted"
       elif [ "${N}" -le 0 ]; then
           echo "== F  no queue room (${FREE} slots free); retry when B/C drain"
       else
           echo "== F  simulation tests (elements $(( OFF + 1 ))-$(( OFF + N )) of ${TOTAL_F}, core=8, 24 h)"
           sub "./oarsub/run_simtests.sh ${TAG} 2 ${OFF}" --array "${N}"
           if [ $(( OFF + N )) -lt "${TOTAL_F}" ]; then
             echo "   $(( TOTAL_F - OFF - N )) element(s) left:  ./oarsub/submit_campaign.sh ${TAG} F $(( OFF + N ))"
           fi
       fi ;;
    *) echo "!! unknown family '${fam}' (expected A B C D E F)"; exit 1 ;;
  esac
done

echo
echo "== queue"
oarstat -u "${USER}" 2>/dev/null | tail -n +1 | head -20 || true
echo "== progress:  ./oarsub/campaign_status.sh ${TAG}"
