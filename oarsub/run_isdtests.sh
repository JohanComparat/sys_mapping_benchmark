#!/usr/bin/env bash
#OAR --name smb_isdtests
#OAR -l /nodes=1/core=8,walltime=12:00:00
#OAR --array 25
#OAR --stdout oarsub/logs/%jobid%.isdtests.out
#OAR --stderr oarsub/logs/%jobid%.isdtests.err
#
# FAMILY H -- what ISD-3 is actually for.
#
# Every simulation campaign to date injects `delta_g*(1 + sum b_i t_i) + sum a_i t_i`,
# which is linear in every template.  A linear marginal fit already suffices for that,
# so the grid CANNOT separate ISD-1 from ISD-3 -- and does not: across all 900 NSIDE-64
# cells of 20260907c the two recover the same amplitude to within 1%, and ISD-3's only
# measurable difference is a worse false-positive rate.  That is a property of the
# injection, not of the method.
#
# This family injects `1 + d_obs = (1 + d_true) * prod_i (1 + F_i(t_i))` instead -- the
# selection-efficiency model ISD actually inverts -- with F non-linear.  Three of the
# six shapes lie inside the cubic basis (linear, quadratic, cubic) and three do not
# (tanh, threshold, exp); the last three are there to find where the method stops
# working rather than to confirm that it does.  Every shape is rescaled to the same
# rms(F), or a ranking of shapes is a ranking of amplitudes.
#
# Only two of the five templates are contaminated.  With all five contaminated there
# is no true negative, so greedy SELECTION is untestable -- which is the second reason
# the existing grid says nothing about ISD.
#
# NSIDE 32 only.  NSIDE 64 is a follow-up once these are read: at ns32 MCMC-comb costs
# 156 s per config against ISD's 0.07 s, and it alone is 99.9% of the wall clock.
#
#   MODE=capability  all six methods, 18 configs, the headline comparison
#   MODE=sweep       ISD only, the hyper-parameter grid (poly_order, n_bins, ...)
#
#   oarsub --project <proj> -S "./oarsub/run_isdtests.sh <tag> <mode> <seeds_per_task> [offset]"
set -euo pipefail
cd "$(dirname "$0")/.."
source oarsub/_campaign_env.sh

TAG="${1:-${OAR_ARRAY_ID:-local}}"
MODE="${2:-capability}"
SPT="${3:-2}"
OFF="${4:-0}"
# NSIDE is positional, not an env var: OAR does not propagate the submitting
# shell's environment into the job, so `NSIDE=128 oarsub ...` silently ran at 32.
NSIDE_ARG="${5:-}"
campaign_activate_env
campaign_threads >/dev/null

# Without a mock-calibrated Delta chi^2_68 the threshold S < 2 is in raw units and
# means nothing; the iteration then re-selects templates it has already corrected.
ISD_NMOCK="${ISD_NMOCK:-30}"
NSIDE="${NSIDE_ARG:-${NSIDE:-32}}"
NGLASS="${NGLASS:-500000}"
NCONTAM="${NCONTAM:-2}"

IDX=$(( ${OAR_ARRAY_INDEX:-1} - 1 + OFF ))

case "${MODE}" in
  capability)
    METHODS="OLS ISD-1 ISD-3 ElasticNet MCMC-add MCMC-comb"
    for j in $(seq 0 $(( SPT - 1 ))); do
        SEED=$(( IDX * SPT + j ))
        # No nside in this path: the analysis script creates its own
        # nside%04d directory under --output-dir, so naming it here produced
        # capability/nside0064/seed010/nside0064/... and campaign_status.sh,
        # which looks for capability/seed010/nside0064/..., reported every
        # cell missing while 50 of them sat on disk.
        OUT="${SMB_RESULTS}/isdtests/${TAG}/capability/seed$(printf '%03d' "${SEED}")"
        echo "-- capability NSIDE=${NSIDE} seed=${SEED} -> ${OUT}"
        python "${SMB_PKG}/scripts/run_simulation_tests.py" \
            --nside "${NSIDE}" \
            --n-glass "${NGLASS}" \
            --glass-only \
            --responses \
            --n-contaminated "${NCONTAM}" \
            --methods ${METHODS} \
            --isd-n-mocks "${ISD_NMOCK}" \
            --syst-dir "${SMB_DATA}/systematics" \
            --seed "${SEED}" \
            --output-dir "${OUT}"
    done
    ;;

  sweep)
    # A hyper-parameter setting changes the fitted weight and nothing else -- not
    # the mock, not the calibration null, not the injected contamination, not the
    # truth or contaminated w(theta).  The first version of this branch re-invoked
    # run_simulation_tests.py per setting and paid all of them again: ~14 min of
    # fixed cost against 0.009 s of ISD payload, and every job died at the wall
    # having covered 17 of 47 settings.  The sweep driver pays each once and scores
    # by the residual Delta chi^2 under a probe of fixed order, which is the only
    # metric that compares across poly_order at all.
    for j in $(seq 0 $(( SPT - 1 ))); do
        SEED=$(( IDX * SPT + j ))
        OUT="${SMB_RESULTS}/isdtests/${TAG}/sweep/nside$(printf '%04d' "${NSIDE}")/seed$(printf '%03d' "${SEED}").json"
        echo "-- sweep NSIDE=${NSIDE} seed=${SEED} -> ${OUT}"
        python "${REPO}/characterisation/isd_hyperparameter_sweep.py" \
            --nside "${NSIDE}" \
            --n-glass "${NGLASS}" \
            --n-contaminated "${NCONTAM}" \
            --isd-n-mocks "${ISD_NMOCK}" \
            --seed "${SEED}" \
            --syst-dir "${SMB_DATA}/systematics" \
            --out "${OUT}"
    done
    ;;

  *) echo "!! unknown mode '${MODE}' (expected capability or sweep)" >&2; exit 1 ;;
esac

# A family that can exit 0 having done nothing is worse than one that crashes.
if [ "${MODE}" = "capability" ]; then
    python "${REPO}/oarsub/check_isd_cell.py" \
        "${SMB_RESULTS}/isdtests/${TAG}/capability/nside$(printf '%04d' "${NSIDE}")" "${NSIDE}"
else
    python "${REPO}/oarsub/check_isd_cell.py" --sweep \
        "${SMB_RESULTS}/isdtests/${TAG}/sweep/nside$(printf '%04d' "${NSIDE}")"
fi
echo "== isdtests ${MODE} cell ${IDX} DONE on $(hostname)"
