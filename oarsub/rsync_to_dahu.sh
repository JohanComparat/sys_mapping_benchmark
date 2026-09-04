#!/usr/bin/env bash
# Push the two checkouts and the staged input data to the cluster.
# Runs on YOUR machine.   ./oarsub/rsync_to_dahu.sh [--dry-run] [tier0|tier1|tier2|code]
set -euo pipefail
cd "$(dirname "$0")/.."
source oarsub/site.sh 2>/dev/null || true
HOST="${SMB_SSH_HOST:?set SMB_SSH_HOST in oarsub/site.sh}"
BENCH_REMOTE="${SMB_REPO_REMOTE:-software/sys_mapping_benchmark}"
PKG_REMOTE="${SMB_PKG_REMOTE:-software/sys_mapping}"
DATA_REMOTE="${SMB_DATA_REMOTE:-data/legacysurvey/dr10}"
PKG_LOCAL="${PKG_LOCAL:-$HOME/software/sys_mapping}"
DATA_LOCAL="${DATA_LOCAL:-$HOME/data/legacysurvey/dr10}"

DRY=""; [ "${1:-}" = "--dry-run" ] && { DRY="-n"; shift; }
WHAT="${1:-all}"
R=(rsync -avz --mkpath ${DRY} --exclude '.git' --exclude '__pycache__'
   --exclude '*.pyc' --exclude '*.egg-info' --exclude 'oarsub/logs/*'
   --exclude '.claude' --exclude '.pytest_cache' --exclude '.coverage'
   --exclude 'dist' --exclude 'dist_*' --exclude '*.ipynb_checkpoints')

if [ "$WHAT" = all ] || [ "$WHAT" = code ]; then
  echo "== code: benchmark repo -> ${HOST}:${BENCH_REMOTE}"
  "${R[@]}" --exclude 'results/*' ./ "${HOST}:${BENCH_REMOTE}/"
  echo "== code: sys_mapping package -> ${HOST}:${PKG_REMOTE}"
  # The package only: its data/ and results/ are outputs and are regenerable.
  "${R[@]}" --exclude 'data/*' --exclude 'results/*' --exclude 'docs/_build' \
            --exclude 'logs/*' "${PKG_LOCAL}/" "${HOST}:${PKG_REMOTE}/"
fi

# Tier 0 -- 7.7 MB.  Everything the characterisation scripts open, plus the
# timing harness (which reads nothing).
if [ "$WHAT" = all ] || [ "$WHAT" = tier0 ]; then
  echo "== tier0: templates 32/64 + params/summaries/sweeps (~7.7 MB)"
  "${R[@]}" "${DATA_LOCAL}/systematics/0032/" "${HOST}:${DATA_REMOTE}/systematics/0032/"
  "${R[@]}" "${DATA_LOCAL}/systematics/0064/" "${HOST}:${DATA_REMOTE}/systematics/0064/"
  "${R[@]}" --include '*/' --include '*_params.json' --exclude '*' \
        "${PKG_LOCAL}/data/sys_weights/" "${HOST}:${PKG_REMOTE}/data/sys_weights/"
  "${R[@]}" --include '*/' --include 'results_summary.json' --exclude '*' \
        "${PKG_LOCAL}/data/simulations/" "${HOST}:${PKG_REMOTE}/data/simulations/"
  "${R[@]}" --include 'detectability_sweep_*.csv' --exclude '*' \
        "${PKG_LOCAL}/results/" "${HOST}:${PKG_REMOTE}/results/"
fi

# Tier 1 -- +84 MB.  Unlocks NSIDE 128/256 for cross-terms and variance.
if [ "$WHAT" = all ] || [ "$WHAT" = tier1 ]; then
  echo "== tier1: templates 128/256 (~84 MB)"
  "${R[@]}" "${DATA_LOCAL}/systematics/0128/" "${HOST}:${DATA_REMOTE}/systematics/0128/"
  "${R[@]}" "${DATA_LOCAL}/systematics/0256/" "${HOST}:${DATA_REMOTE}/systematics/0256/"
fi

# Tier 2 -- +2.4 GB.  Only the LRT needs these.  Deliberately NOT the
# HPX_*-JK100 subdirs or the wprp FITS: nothing in this campaign reads them.
if [ "$WHAT" = all ] || [ "$WHAT" = tier2 ]; then
  echo "== tier2: LS10 DATA/RAND catalogues (~2.4 GB)"
  "${R[@]}" --include '*_DATA.fits' --include '*_RAND.fits' --exclude '*' \
        "${DATA_LOCAL}/sweep/BGS_VLIM_Mstar/" "${HOST}:${DATA_REMOTE}/sweep/BGS_VLIM_Mstar/"
  echo "== tier2: existing mock-LRT params.json (for --resume-null, ~144 KB)"
  "${R[@]}" --include '*/' --include '*_params.json' --exclude '*' \
        "${PKG_LOCAL}/results/ls10_mocklrt/" "${HOST}:${PKG_REMOTE}/results/ls10_mocklrt/"
fi

echo
echo "== next, on dahu:"
echo "   cd ~/${BENCH_REMOTE} && oarsub --project \${SMB_PROJECT} -S ./oarsub/build_env.sh"
