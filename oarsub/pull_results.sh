#!/usr/bin/env bash
# Bring the campaign products back.  Runs on YOUR machine.
#   ./oarsub/pull_results.sh <tag> [--dry-run]
#
# Only assembled products come back -- JSON, CSV, PNG and the job logs.  The
# intermediate FITS on /bettik stay there: they are regenerable, and they are
# what makes the difference between a few MB and a few GB.
set -euo pipefail
cd "$(dirname "$0")/.."
source oarsub/site.sh 2>/dev/null || true
HOST="${SMB_SSH_HOST:?set SMB_SSH_HOST in oarsub/site.sh}"
WORK="${SMB_WORK:?set SMB_WORK in oarsub/site.sh (pull runs off-cluster, so it cannot be derived)}"
TAG="${1:?usage: pull_results.sh <tag> [--dry-run]}"; shift
DRY=""; [ "${1:-}" = "--dry-run" ] && DRY="-n"

DEST="results/campaign/${TAG}"
mkdir -p "${DEST}"
echo "== ${HOST}:${WORK}/results  ->  ${DEST}  (tag ${TAG})"
rsync -avz ${DRY} --prune-empty-dirs \
  --include '*/' \
  --include '*.json' --include '*.csv' --include '*.png' --include '*.txt' \
  --exclude '*' \
  "${HOST}:${WORK}/results/" "${DEST}/"

echo "== job logs -> ${DEST}/logs"
rsync -avz ${DRY} --prune-empty-dirs \
  "${HOST}:${SMB_REPO_REMOTE:-software/sys_mapping_benchmark}/oarsub/logs/" "${DEST}/logs/" || true

echo
echo "== landed:"
find "${DEST}" -name '*.csv' -o -name '*.json' | sed "s|^|   |" | head -30
echo "== then: regenerate the docs tables and the paper sections from ${DEST}"
