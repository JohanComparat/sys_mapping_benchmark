#!/usr/bin/env bash
# =============================================================================
# What actually landed, per family.  Runs on the cluster frontend.
#
#   ./oarsub/campaign_status.sh <tag> [family...]
#
# This NAMES THE MISSING CELLS rather than counting files.  A count tells you
# 34/36 and nothing about which two; a partial campaign whose gaps you cannot
# name is a campaign you have to rerun whole.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
source oarsub/_campaign_env.sh
TAG="${1:?usage: campaign_status.sh <tag> [family...]}"; shift
FAMS=("$@"); [ ${#FAMS[@]} -gt 0 ] || FAMS=(B C D E F)

miss=0
report () {   # report <label> <path>
    if [ -e "$2" ]; then return 0; fi
    echo "   MISSING  $1"; miss=$(( miss + 1 ))
}

for fam in "${FAMS[@]}"; do
case "${fam}" in
B) echo "== B  GLASS calibration  (expect 36 cells x 5 seeds)"
   R="${SMB_RESULTS}/glass_calibration/${TAG}"
   for s in "${SMB_SAMPLES[@]}"; do for n in "${SMB_NSIDES[@]}"; do
     for k in 0 1 2 3 4; do
       report "${s} NSIDE${n} seed$(( 11 + 1000*k ))" \
              "${R}/${s}_NSIDE$(printf '%04d' "${n}")/seed$(( 11 + 1000*k ))/glass_calibration.json"
   done; done; done
   # The number family E is waiting on.
   python - "${R}" <<'PYEOF' || true
import json, pathlib, statistics, sys
root = pathlib.Path(sys.argv[1])
vals, unconverged = [], []
for p in root.rglob("glass_calibration.json"):
    d = json.load(p.open())
    fit = d.get("fit") or {}
    if fit.get("converged") and isinstance(fit.get("cl_amplitude"), (int, float)):
        vals.append(fit["cl_amplitude"])
    else:
        unconverged.append(str(p.parent.relative_to(root)))
if vals:
    m = statistics.median(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"   cl_amplitude over {len(vals)} runs: median {m:.4g}  sd {sd:.3g}")
    print(f"   -> submit_campaign.sh <tag> E {m:.4g}")
if unconverged:
    print(f"   {len(unconverged)} run(s) did NOT converge, e.g. {unconverged[0]}")
PYEOF
   ;;
C) echo "== C  variance cube  (expect 20 elements)"
   R="${SMB_RESULTS}/variance_inflation/${TAG}"
   for n in 32 64; do for sl in 1.0 1.5 2.0 2.5 3.0; do for sc in 0.2 0.4; do
     report "ns${n} slope${sl} sigma_clus${sc}" "${R}/ns${n}_slope${sl}_sc${sc}/variance_inflation.csv"
   done; done; done ;;
D) echo "== D  benchmark grid  (expect 1 job)"
   report "benchmarks.csv" "${SMB_RESULTS}/benchmarks/${TAG}/benchmarks.csv" ;;
E) echo "== E  mock-calibrated LRT  (expect 18 cells)"
   R="${SMB_RESULTS}/ls10_mocklrt/${TAG}"
   for s in "${SMB_SAMPLES[@]}"; do for n in 32 64; do
     report "${s} NSIDE${n}" \
            "${R}/NSIDE$(printf '%04d' "${n}")/${s}_NSIDE$(printf '%04d' "${n}")_params.json"
   done; done ;;
F) echo "== F  simulation tests  (expect 50 seeds x 3 NSIDE)"
   R="${SMB_RESULTS}/simulations/${TAG}"
   for k in $(seq 0 49); do for n in 32 64 128; do
     report "seed${k} NSIDE${n}" \
            "${R}/seed$(printf '%03d' "${k}")/nside$(printf '%04d' "${n}")/results_summary.json"
   done; done ;;
esac
done

echo
if [ "${miss}" -eq 0 ]; then
    echo "== every expected cell is present.  ./oarsub/pull_results.sh ${TAG}"
else
    echo "== ${miss} cell(s) missing (listed above)"
fi
echo "-- queue:"; oarstat -u "${USER}" 2>/dev/null | head -15 || true
