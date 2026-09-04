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

# rsync exits 12/23/30/255 on a dropped connection.  Each retry resumes from the
# partial file, so N attempts finish a transfer one attempt cannot.
rs () {
    local try
    for try in 1 2 3 4 5; do
        if "${R[@]}" "$@"; then return 0; fi
        echo "-- attempt ${try} failed (rsync $?); resuming in 10 s" >&2
        sleep 10
    done
    echo "!! transfer failed after 5 attempts: $*" >&2
    return 1
}
WHAT="${1:-all}"
APPEND=()
# --partial makes a dropped connection resumable; --append-verify (added for the
# tier-2 catalogues only, via APPEND below) resumes a large file from its current
# size instead of restarting it.  It is deliberately NOT used for the code push:
# source files change in place, so appending is meaningless there and rsync
# rightly warns on every changed file.
#
# The 2.4 GB tier-2 push through the gateway does not survive in one piece: the
# 2.4 GB tier-2 push through the gateway does not survive in one piece, and
# without these a retry restarts every file from zero.  ServerAlive* keeps the
# ssh channel from being reaped while rsync is checksumming a large file.
RSH="ssh -o ServerAliveInterval=20 -o ServerAliveCountMax=6 -o TCPKeepAlive=yes"
R=(rsync -avz --mkpath ${DRY} --rsh="${RSH}" --timeout=300
   --partial
   --exclude '.git' --exclude '__pycache__'
   --exclude '*.pyc' --exclude '*.egg-info' --exclude 'oarsub/logs/*'
   --exclude '.claude' --exclude '.pytest_cache' --exclude '.coverage'
   --exclude 'dist' --exclude 'dist_*' --exclude '*.ipynb_checkpoints')

if [ "$WHAT" = all ] || [ "$WHAT" = code ]; then
  echo "== code: benchmark repo -> ${HOST}:${BENCH_REMOTE}"
  rs --exclude 'results/*' ./ "${HOST}:${BENCH_REMOTE}/"
  echo "== code: sys_mapping package -> ${HOST}:${PKG_REMOTE}"
  # The package only: its data/ and results/ are outputs and are regenerable.
  rs --exclude 'data/*' --exclude 'results/*' --exclude 'docs/_build' \
            --exclude 'logs/*' "${PKG_LOCAL}/" "${HOST}:${PKG_REMOTE}/"
fi

# Tier 0 -- 7.7 MB.  Everything the characterisation scripts open, plus the
# timing harness (which reads nothing).
if [ "$WHAT" = all ] || [ "$WHAT" = tier0 ]; then
  echo "== tier0: templates 32/64 + params/summaries/sweeps (~7.7 MB)"
  rs "${DATA_LOCAL}/systematics/0032/" "${HOST}:${DATA_REMOTE}/systematics/0032/"
  rs "${DATA_LOCAL}/systematics/0064/" "${HOST}:${DATA_REMOTE}/systematics/0064/"
  rs --include '*/' --include '*_params.json' --exclude '*' \
        "${PKG_LOCAL}/data/sys_weights/" "${HOST}:${PKG_REMOTE}/data/sys_weights/"
  rs --include '*/' --include 'results_summary.json' --exclude '*' \
        "${PKG_LOCAL}/data/simulations/" "${HOST}:${PKG_REMOTE}/data/simulations/"
  rs --include 'detectability_sweep_*.csv' --exclude '*' \
        "${PKG_LOCAL}/results/" "${HOST}:${PKG_REMOTE}/results/"
fi

# Tier 1 -- +84 MB.  Unlocks NSIDE 128/256 for cross-terms and variance.
if [ "$WHAT" = all ] || [ "$WHAT" = tier1 ]; then
  echo "== tier1: templates 128/256 (~84 MB)"
  rs "${DATA_LOCAL}/systematics/0128/" "${HOST}:${DATA_REMOTE}/systematics/0128/"
  rs "${DATA_LOCAL}/systematics/0256/" "${HOST}:${DATA_REMOTE}/systematics/0256/"
fi

# Tier 2 -- +2.4 GB.  Only the LRT needs these.  Deliberately NOT the
# HPX_*-JK100 subdirs or the wprp FITS: nothing in this campaign reads them.
if [ "$WHAT" = all ] || [ "$WHAT" = tier2 ]; then
  APPEND=(--append-verify)          # large, immutable: resume rather than restart
  echo "== tier2: LS10 DATA/RAND catalogues (~2.4 GB)"
  rs --include '*_DATA.fits' --include '*_RAND.fits' --exclude '*' \
        "${APPEND[@]}" "${DATA_LOCAL}/sweep/BGS_VLIM_Mstar/" \
        "${HOST}:${DATA_REMOTE}/sweep/BGS_VLIM_Mstar/"
  echo "== tier2: existing mock-LRT params.json (for --resume-null, ~144 KB)"
  rs --include '*/' --include '*_params.json' --exclude '*' \
        "${PKG_LOCAL}/results/ls10_mocklrt/" "${HOST}:${PKG_REMOTE}/results/ls10_mocklrt/"
fi

echo
echo "== next, on dahu:"
echo "   cd ~/${BENCH_REMOTE} && oarsub --project \${SMB_PROJECT} -S ./oarsub/build_env.sh"
