#!/usr/bin/env bash
# =============================================================================
# Shared environment for every sys_mapping_benchmark job on GRICAD/Dahu.
# Sourced by run_*.sh; not executable on its own.
#
# Convention copied from emu_pk/oarsub/_campaign_env.sh, which already paid for
# the OAR dialect gotchas noted below.
# =============================================================================

if [ -r "$(dirname "${BASH_SOURCE[0]}")/site.sh" ]; then
    # shellcheck disable=SC1091
    source "$(dirname "${BASH_SOURCE[0]}")/site.sh"
fi

# Operate on the tree this file lives in, never a hardcoded path: bash reads a
# job script incrementally, so rewriting a run_*.sh under a running job makes it
# resume at a stale byte offset in a different file.
_CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${SMB_REPO:-$(dirname "${_CAMPAIGN_DIR}")}"

# Not from /applis/cluster_name -- that says "luke" on both dahu and bigfoot.
# Hostname is dahu/bigfoot on frontends, dahu103/bigfoot7 on nodes.
campaign_cluster () {
    case "$(hostname)" in
        bigfoot*) echo bigfoot ;;
        dahu*)    echo dahu ;;
        *)        echo unknown ;;
    esac
}

CONDA_ENV="${SMB_ENV:-sys_map}"

campaign_project () {
    if [ -z "${SMB_PROJECT:-}" ]; then
        echo "!! SMB_PROJECT is not set." >&2
        echo "   cp oarsub/site.sh.example oarsub/site.sh and fill it in." >&2
        return 1
    fi
    echo "${SMB_PROJECT}"
}

# /bettik is the per-project BeeGFS scratch; $HOME is a shared NAS.
WORK="${SMB_WORK:-/bettik/PROJECTS/${SMB_PROJECT:-UNSET}/${USER}/sys_mapping_benchmark}"
export SMB_RESULTS="${WORK}/results"

# Where the staged inputs live on the cluster.
export SMB_DATA="${SMB_DATA:-${HOME}/${SMB_DATA_REMOTE:-data/legacysurvey/dr10}}"
export SMB_PKG="${SMB_PKG:-${HOME}/${SMB_PKG_REMOTE:-software/sys_mapping}}"

campaign_activate_env () {
    local mamba_exe="${MAMBA_EXE:-${HOME}/miniforge3/bin/mamba}"
    if [ -x "${mamba_exe}" ]; then
        export MAMBA_EXE="${mamba_exe}"
        export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/miniforge3}"
        local hook
        if hook="$("${MAMBA_EXE}" shell hook --shell bash --root-prefix "${MAMBA_ROOT_PREFIX}" 2>/dev/null)"; then
            eval "${hook}"
        else
            alias mamba="${MAMBA_EXE}"
        fi
        mamba activate "${CONDA_ENV}"
    elif [ -r /applis/environments/conda.sh ]; then
        # shellcheck disable=SC1091
        source /applis/environments/conda.sh
        conda activate "${CONDA_ENV}"
    else
        echo "!! no mamba at ${mamba_exe} and no /applis/environments/conda.sh." >&2
        exit 1
    fi
    echo "-- cluster=$(campaign_cluster) env=${CONDA_ENV} python=$(command -v python)"
    # Fail here, loudly, rather than every array element failing identically an
    # hour later with an ImportError nobody reads.  blackjax and joblib are the
    # ones that matter: environment.yml omits both, so an env built from it
    # imports fine and then dies in every NUTS/LRT job.
    python - <<'PYEOF' || exit 1
import sys
missing = []
for m in ("numpy", "scipy", "healpy", "jax", "blackjax", "emcee",
          "joblib", "sklearn", "glass", "sys_mapping"):
    try:
        __import__(m)
    except Exception as e:
        missing.append(f"{m} ({type(e).__name__})")
if missing:
    sys.exit("!! environment is missing: " + ", ".join(missing))
print("environment ok")
PYEOF
}

# OAR_RES_NB_CORES does not exist on this OAR build; OAR_NODEFILE has one line
# per allocated core and OAR uses cpusets, so nproc agrees.  Keep both + a floor.
campaign_ncores () {
    local n="${OAR_RES_NB_CORES:-}"
    if [ -z "${n}" ] && [ -r "${OAR_NODEFILE:-/nonexistent}" ]; then
        n="$(wc -l < "${OAR_NODEFILE}")"
    fi
    echo "${n:-$(nproc 2>/dev/null || echo 4)}"
}

campaign_threads () {
    local n; n="$(campaign_ncores)"
    export OMP_NUM_THREADS="${n}" OPENBLAS_NUM_THREADS="${n}" MKL_NUM_THREADS="${n}"
    # float64 is REQUIRED: under float32 the combined model's 1+sum(b.t) goes
    # non-positive on the near-degenerate LS10 basis and NUTS returns nan.
    export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
    if [ -z "${JAX_PLATFORMS:-}" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
        export JAX_PLATFORMS=cpu
    fi
    echo "${n}"
}

# The nine LS10 volume-limited samples, in the order the campaign indexes them.
SMB_SAMPLES=(
  LS10_VLIM_ANY_9.0_Mstar_12.0_0.05_z_0.08_N_0523486
  LS10_VLIM_ANY_9.5_Mstar_12.0_0.05_z_0.12_N_1432502
  LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238
  LS10_VLIM_ANY_10.25_Mstar_12.0_0.05_z_0.22_N_3308841
  LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228
  LS10_VLIM_ANY_10.75_Mstar_12.0_0.05_z_0.31_N_2802710
  LS10_VLIM_ANY_11.0_Mstar_12.0_0.05_z_0.35_N_1619838
  LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855
  LS10_VLIM_ANY_11.5_Mstar_12.0_0.05_z_0.35_N_0120882
)
SMB_NSIDES=(32 64 128 256)
export SMB_SAMPLES SMB_NSIDES
