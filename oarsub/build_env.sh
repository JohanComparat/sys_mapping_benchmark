#!/usr/bin/env bash
#OAR --name smb_build_env
#OAR -l /nodes=1/core=16,walltime=03:00:00
#OAR --stdout oarsub/logs/%jobid%.build_env.out
#OAR --stderr oarsub/logs/%jobid%.build_env.err
#
# Build the sys_map environment ON A COMPUTE NODE, not the frontend: the
# frontend kills a long conda solve outright.
#
# Built from sys_mapping/pyproject.toml, NOT environment.yml -- the latter omits
# blackjax and joblib, so an env built from it imports cleanly and then fails
# every NUTS/LRT job at runtime.
#
#   oarsub --project <proj> -S ./oarsub/build_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source oarsub/_campaign_env.sh

ENVN="${SMB_ENV:-sys_map}"
MAMBA="${HOME}/miniforge3/bin/mamba"
[ -x "${MAMBA}" ] || { echo "!! no mamba at ${MAMBA}"; exit 1; }
export MAMBA_ROOT_PREFIX="${HOME}/miniforge3"
eval "$("${MAMBA}" shell hook --shell bash --root-prefix "${MAMBA_ROOT_PREFIX}")"

if mamba env list | awk '{print $1}' | grep -qx "${ENVN}"; then
    echo "== ${ENVN} exists; updating in place"
else
    echo "== creating ${ENVN}"
    mamba create -y -n "${ENVN}" python=3.11
fi
mamba activate "${ENVN}"

# conda-forge for the compiled stack, pip for the rest: healpy and scikit-learn
# are far more reliable from conda-forge on this site.
mamba install -y -c conda-forge numpy scipy astropy healpy matplotlib scikit-learn pandas

PKG="${SMB_PKG}"
[ -d "${PKG}" ] || { echo "!! sys_mapping checkout not found at ${PKG}"; exit 1; }
python -m pip install --no-input "jax[cpu]>=0.9" "blackjax>=1.2" joblib emcee treecorr
python -m pip install --no-input "glass>=2026.1" || echo "!! glass failed -- GLASS families will not run"
python -m pip install --no-input -e "${PKG}"

echo "== import gate"
campaign_activate_env
echo "== build_env DONE on $(hostname)"
