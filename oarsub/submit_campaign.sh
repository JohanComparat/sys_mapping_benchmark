#!/usr/bin/env bash
# =============================================================================
# Submit one or more job families.  Runs ON THE CLUSTER FRONTEND.
#
#   ./oarsub/submit_campaign.sh <tag> A            # env build, first and alone
#   ./oarsub/submit_campaign.sh <tag> B C D F      # the independent families
#   ./oarsub/submit_campaign.sh <tag> M            # match the mocks per sample
#   ./oarsub/submit_campaign.sh <tag> E <calib_tag> # LRT, after M (or B)
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

# Snapshot the library too.  Without this, SMB_PKG is one live checkout shared by
# every running job: an rsync mid-campaign changes what array elements that have
# not yet started will import, so two cells of the same tag can run different
# code and the tag records neither.  _campaign_env.sh prefers ${SNAP}/pkg when it
# exists.
_PKG_SRC="${SMB_PKG:-${HOME}/${SMB_PKG_REMOTE:-software/sys_mapping}}"
if [ -d "${_PKG_SRC}/sys_mapping" ]; then
    rsync -a --exclude '.git' --exclude '__pycache__' --exclude '_build' \
          --exclude 'results' "${_PKG_SRC}/" "${SNAP}/pkg/"
    {
        echo "source:   ${_PKG_SRC}"
        echo "snapshot: $(date -Is)"
        if [ -r "${_PKG_SRC}/PKG_VERSION.txt" ]; then
            # A staged copy carries no .git; the stager writes this instead.
            sed 's/^/          /' "${_PKG_SRC}/PKG_VERSION.txt"
        elif command -v git >/dev/null && [ -d "${_PKG_SRC}/.git" ]; then
            echo "head:     $(git -C "${_PKG_SRC}" rev-parse HEAD 2>/dev/null || echo unknown)"
            if [ -n "$(git -C "${_PKG_SRC}" status --porcelain 2>/dev/null)" ]; then
                echo "dirty:    yes"
                git -C "${_PKG_SRC}" status --porcelain 2>/dev/null | sed 's/^/          /'
            else
                echo "dirty:    no"
            fi
        fi
    } > "${SNAP}/pkg_version.txt"
    echo "== package snapshot: ${SNAP}/pkg  ($(sed -n 's/^head: *//p' "${SNAP}/pkg_version.txt" | cut -c1-12))"
else
    echo "!! no package at ${_PKG_SRC}; jobs will read the live checkout" >&2
fi
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
    M) # An optional comma-separated cell list reruns a subset, so a failed
       # handful does not mean recomputing the cells that already produced a
       # validated spectrum.
       CELLS=""
       if [[ "${1:-}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then CELLS="$1"; shift; fi
       if [ -n "${CELLS}" ]; then
           N_M=$(awk -F, '{print NF}' <<< "${CELLS}")
           echo "== M  matching, cells ${CELLS} only (${N_M} of 18, core=4, 6 h)"
           sub "./oarsub/run_glassmatch.sh ${TAG} 25 5 ${CELLS}" --array "${N_M}"
       else
           echo "== M  match each mock to its sample's large-scale clustering"
           echo "      (array 9: one per sample, NSIDE set to reach rp=10 Mpc/h, core=4, 12 h)"
           sub "./oarsub/run_glassmatch.sh ${TAG} 25 5"
       fi ;;
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
       # An optional comma-separated cell list re-runs a subset.
       ECELLS=""
       if [[ "${1:-}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then ECELLS="$1"; shift; fi
       if [ -n "${ECELLS}" ]; then
           N_E=$(awk -F, '{print NF}' <<< "${ECELLS}")
           echo "== E  mock-calibrated LRT, cells ${ECELLS} only (${N_E} of 18, core=8, 48 h)"
           sub "./oarsub/run_lrt.sh ${TAG} 50 ${CALIB} '${FB}' ${ECELLS}" --array "${N_E}"
       else
           echo "== E  mock-calibrated LRT (array 18, core=8, 48 h, per-cell amplitude from ${CALIB})"
           sub "./oarsub/run_lrt.sh ${TAG} 50 ${CALIB} ${FB}"
       fi ;;
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
    H) # H is two families in one: the six-method comparison on a non-linear
       # response, and the ISD hyper-parameter sweep.  The sweep is ISD-only and
       # costs ~0.07 s a fit, so it is submitted whole; the comparison carries
       # MCMC-comb at 156 s a config and is chunked like F.
       MODE="capability"
       if [[ "${1:-}" =~ ^(capability|sweep)$ ]]; then MODE="$1"; shift; fi
       # NSIDE is positional in the runner; OAR does not carry the environment.
       HNS=32
       if [[ "${1:-}" =~ ^(32|64|128|256)$ ]]; then HNS="$1"; shift; fi
       if [ "${MODE}" = "sweep" ]; then
           HWALL=2; [ "${HNS}" -ge 128 ] && HWALL=8
           echo "== H  ISD hyper-parameter sweep at NSIDE ${HNS} (array 10 x 2 seeds, core=4, ${HWALL} h)"
           sub "./oarsub/run_isdtests.sh ${TAG} sweep 2 0 ${HNS}" \
               --array 10 -l "/nodes=1/core=4,walltime=${HWALL}:00:00"
       else
           TOTAL_H=25
           FREE="$(campaign_free_slots)"
           if [ "${FREE}" -gt 5 ]; then FREE=$(( FREE - 5 )); fi
           OFF=0
           if [[ "${1:-}" =~ ^[0-9]+$ ]]; then OFF="$1"; shift; fi
           N=$(( TOTAL_H - OFF ))
           if [ "${N}" -gt "${FREE}" ]; then N="${FREE}"; fi
           if [ "${OFF}" -ge "${TOTAL_H}" ]; then
               echo "== H  all ${TOTAL_H} elements already submitted"
           elif [ "${N}" -le 0 ]; then
               echo "== H  no queue room (${FREE} slots free); retry when E/F drain"
           else
               echo "== H  ISD capability tests at NSIDE ${HNS} (elements $(( OFF + 1 ))-$(( OFF + N )) of ${TOTAL_H}, core=8, 12 h)"
               sub "./oarsub/run_isdtests.sh ${TAG} capability 2 ${OFF} ${HNS}" --array "${N}"
               if [ $(( OFF + N )) -lt "${TOTAL_H}" ]; then
                 echo "   $(( TOTAL_H - OFF - N )) element(s) left:  ./oarsub/submit_campaign.sh ${TAG} H $(( OFF + N ))"
               fi
           fi
       fi ;;
    P) # Regenerate the shipped LS10 weight products.  Cheap -- no LRT null --
       # so it runs as a plain array with no chunking.  An optional argument
       # gives the resolutions; the array size must match 9 x their count, and
       # the #OAR --array header in the script covers the default four.
       PNS="${1:-}"
       if [ -n "${PNS}" ] && [[ "${PNS}" != [A-Z] ]]; then
           shift
           PNS="${PNS// /,}"          # oarsub takes one string; commas survive it
           PN=$(tr ',' '\n' <<< "${PNS}" | grep -c .)
           # SMB_P_ARRAY/SMB_P_OFFSET submit a slice, for a smoke test of one
           # cell before committing the whole family to a code change.
           P_N="${SMB_P_ARRAY:-$((9 * PN))}"
           P_OFF="${SMB_P_OFFSET:-0}"
           echo "== P  LS10 weight products, NSIDE ${PNS} (array ${P_N}, offset ${P_OFF}, core=8, 12 h)"
           sub "./oarsub/run_ls10products.sh ${TAG} ${P_OFF} ${PNS}" \
               -l "/nodes=1/core=8,walltime=12:00:00" --array "${P_N}"
       else
           echo "== P  LS10 weight products (array 36, core=8, 12 h)"
           sub "./oarsub/run_ls10products.sh ${TAG}"
       fi ;;
    *) echo "!! unknown family '${fam}' (expected A B C D E F H M P)"; exit 1 ;;
  esac
done

echo
echo "== queue"
oarstat -u "${USER}" 2>/dev/null | tail -n +1 | head -20 || true
echo "== progress:  ./oarsub/campaign_status.sh ${TAG}"
