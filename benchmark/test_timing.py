"""Timing tests: wall-clock time for each method as a function of NSIDE and n_templates.

These tests measure how computation time scales with:
- HEALPix resolution (NSIDE = 8, 16, 32, 64)
- Number of systematic templates (1 to 10)

All six implemented methods are benchmarked:
  OLS, ElasticNet, ISD-1, ISD-3, MCMC-additive, MCMC-combined

Run with::

    pytest tests/test_timing.py -v -s --no-header

Use ``-k "nside32"`` to run only the NSIDE=32 tests, etc.
Slow MCMC tests are marked ``@pytest.mark.slow``; skip them with
``pytest tests/test_timing.py -v -m "not slow"``.
"""

from __future__ import annotations

import time
import warnings

import healpy as hp
import numpy as np
import pytest

import sys_mapping as sm
from sys_mapping.contamination import apply_contamination, unpack_params
from sys_mapping.correction import rotate_templates, transform_params_from_rotated

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NSIDES = [8, 16, 32, 64]
N_TEMPLATES_RANGE = list(range(1, 11))   # 1 to 10
N_MEAN = 50
SIGMA_G = 0.5
AMPLITUDE = 0.10

# Reduced MCMC settings to keep slow tests manageable
MCMC_N_WALKERS = 32
MCMC_N_STEPS = 60
MCMC_N_BURN = 15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_templates(nside: int, n_templates: int, seed: int = 0) -> np.ndarray:
    """Generate n_templates synthetic HEALPix maps at the given NSIDE.

    Uses 5 standard spectral families cycling with two different seeds so
    up to 10 diverse maps can be produced without repetition.
    """
    families = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    seeds = [seed] * 5 + [seed + 7] * 5
    return np.stack([
        sm.generate_systematic_map(nside, families[i], seed=seeds[i])
        for i in range(n_templates)
    ])


def _galactic_mask(nside: int) -> np.ndarray:
    n_pix = hp.nside2npix(nside)
    theta, _ = hp.pix2ang(nside, np.arange(n_pix))
    b_gal = np.abs(np.pi / 2 - theta)
    return b_gal > np.radians(20)


def _make_data(nside: int, n_templates: int, seed: int = 42
               ) -> tuple[np.ndarray, np.ndarray]:
    """Return (delta_g, delta_t) ready for method fitting."""
    templates = _generate_templates(nside, n_templates, seed=0)
    footprint = _galactic_mask(nside)

    rng = np.random.default_rng(seed)
    n_pix = hp.nside2npix(nside)
    lmax = 3 * nside - 1
    ell = np.arange(lmax + 1, dtype=float)
    cl_G = (ell + 1.0) ** (-2)
    cl_G[0] = 0.0
    cl_G *= SIGMA_G**2 / np.sum((2 * ell + 1) / (4 * np.pi) * cl_G)
    G = hp.synfast(cl_G, nside=nside, lmax=lmax)
    delta_true = np.exp(G - 0.5 * SIGMA_G**2) - 1.0

    a_true = np.full(n_templates, AMPLITUDE)
    b_true = np.full(n_templates, AMPLITUDE * 0.5)
    delta_obs = apply_contamination(delta_true, templates, a_true, b_true)

    n_g_mean = N_MEAN * (1.0 + delta_obs)
    n_g_mean = np.where(n_g_mean < 0, 0, n_g_mean)
    gal_counts = rng.poisson(n_g_mean * footprint.astype(float))
    rand_counts = rng.poisson(N_MEAN * 8 * footprint.astype(float))

    delta_g, good_pix = sm.compute_overdensity(gal_counts, rand_counts)
    delta_t = sm.assign_template_values(templates, good_pix)
    return delta_g, delta_t


# ---------------------------------------------------------------------------
# Parametrized fixture: (nside, n_templates)
# ---------------------------------------------------------------------------

@pytest.fixture(
    params=[
        pytest.param((nside, n_tmpl), id=f"nside{nside}_ntmpl{n_tmpl}")
        for nside in NSIDES
        for n_tmpl in N_TEMPLATES_RANGE
    ]
)
def prepared_data(request):
    nside, n_tmpl = request.param
    delta_g, delta_t = _make_data(nside, n_tmpl)
    return nside, n_tmpl, delta_g, delta_t


# ---------------------------------------------------------------------------
# OLS timing
# ---------------------------------------------------------------------------

class TestTimingOLS:
    @pytest.mark.parametrize(
        "nside,n_tmpl",
        [(n, k) for n in NSIDES for k in N_TEMPLATES_RANGE],
        ids=[f"nside{n}_ntmpl{k}" for n in NSIDES for k in N_TEMPLATES_RANGE],
    )
    def test_ols_time(self, nside, n_tmpl):
        delta_g, delta_t = _make_data(nside, n_tmpl)
        t0 = time.perf_counter()
        X = delta_t.T
        a_hat, *_ = np.linalg.lstsq(X, delta_g, rcond=None)
        elapsed = time.perf_counter() - t0
        print(f"\nOLS | nside={nside} | n_tmpl={n_tmpl} | t={elapsed:.4f}s")
        assert elapsed < 30.0, f"OLS too slow: {elapsed:.2f}s"
        assert a_hat.shape == (n_tmpl,)


# ---------------------------------------------------------------------------
# ElasticNet timing
# ---------------------------------------------------------------------------

class TestTimingElasticNet:
    @pytest.mark.parametrize(
        "nside,n_tmpl",
        [(n, k) for n in NSIDES for k in N_TEMPLATES_RANGE],
        ids=[f"nside{n}_ntmpl{k}" for n in NSIDES for k in N_TEMPLATES_RANGE],
    )
    def test_elasticnet_time(self, nside, n_tmpl):
        delta_g, delta_t = _make_data(nside, n_tmpl)
        t0 = time.perf_counter()
        a_hat, _, _ = sm.elasticnet_contamination_fit(delta_g, delta_t, cv_folds=3)
        elapsed = time.perf_counter() - t0
        print(f"\nElasticNet | nside={nside} | n_tmpl={n_tmpl} | t={elapsed:.4f}s")
        # ElasticNet with 5-fold CV can take 5–10 minutes for nside=64 with many
        # templates on a loaded CPU-only machine.  Budget set to 600 s.
        assert elapsed < 600.0, f"ElasticNet too slow: {elapsed:.2f}s"
        assert a_hat.shape == (n_tmpl,)


# ---------------------------------------------------------------------------
# ISD-1 timing
# ---------------------------------------------------------------------------

class TestTimingISD1:
    @pytest.mark.parametrize(
        "nside,n_tmpl",
        [(n, k) for n in NSIDES for k in N_TEMPLATES_RANGE],
        ids=[f"nside{n}_ntmpl{k}" for n in NSIDES for k in N_TEMPLATES_RANGE],
    )
    def test_isd1_time(self, nside, n_tmpl):
        delta_g, delta_t = _make_data(nside, n_tmpl)
        t0 = time.perf_counter()
        _, alpha_all, n_it = sm.iterative_systematics_decontamination(
            delta_g, delta_t, poly_order=1, max_iter=50
        )
        elapsed = time.perf_counter() - t0
        print(f"\nISD-1 | nside={nside} | n_tmpl={n_tmpl} | "
              f"t={elapsed:.4f}s | n_iter={n_it}")
        assert elapsed < 60.0, f"ISD-1 too slow: {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# ISD-3 timing
# ---------------------------------------------------------------------------

class TestTimingISD3:
    @pytest.mark.parametrize(
        "nside,n_tmpl",
        [(n, k) for n in NSIDES for k in N_TEMPLATES_RANGE],
        ids=[f"nside{n}_ntmpl{k}" for n in NSIDES for k in N_TEMPLATES_RANGE],
    )
    def test_isd3_time(self, nside, n_tmpl):
        delta_g, delta_t = _make_data(nside, n_tmpl)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, alpha_all, n_it = sm.iterative_systematics_decontamination(
                delta_g, delta_t, poly_order=3, max_iter=50
            )
        elapsed = time.perf_counter() - t0
        print(f"\nISD-3 | nside={nside} | n_tmpl={n_tmpl} | "
              f"t={elapsed:.4f}s | n_iter={n_it}")
        assert elapsed < 120.0, f"ISD-3 too slow: {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# MCMC-additive timing (slow)
# ---------------------------------------------------------------------------

class TestTimingMCMCAdd:
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "nside,n_tmpl",
        [(n, k) for n in NSIDES for k in N_TEMPLATES_RANGE],
        ids=[f"nside{n}_ntmpl{k}" for n in NSIDES for k in N_TEMPLATES_RANGE],
    )
    def test_mcmc_add_time(self, nside, n_tmpl):
        delta_g, delta_t = _make_data(nside, n_tmpl)
        delta_t_rot, R, _ = rotate_templates(delta_t)
        n_dim = n_tmpl + 1
        nw = max(MCMC_N_WALKERS, 2 * n_dim + 2)
        t0 = time.perf_counter()
        flat, _ = sm.run_mcmc(
            n_sys=n_tmpl, model="additive",
            delta_g_obs=delta_g, delta_t=delta_t_rot,
            n_walkers=nw, n_steps=MCMC_N_STEPS, n_burn=MCMC_N_BURN,
            seed=0, progress=False,
        )
        elapsed = time.perf_counter() - t0
        print(f"\nMCMC-add | nside={nside} | n_tmpl={n_tmpl} | "
              f"t={elapsed:.2f}s | chain={flat.shape}")
        assert elapsed < 600.0, f"MCMC-add too slow: {elapsed:.2f}s"
        assert flat.shape[1] == n_tmpl + 1


# ---------------------------------------------------------------------------
# MCMC-combined timing (slow)
# ---------------------------------------------------------------------------

class TestTimingMCMCComb:
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "nside,n_tmpl",
        [(n, k) for n in NSIDES for k in N_TEMPLATES_RANGE],
        ids=[f"nside{n}_ntmpl{k}" for n in NSIDES for k in N_TEMPLATES_RANGE],
    )
    def test_mcmc_comb_time(self, nside, n_tmpl):
        delta_g, delta_t = _make_data(nside, n_tmpl)
        delta_t_rot, R, _ = rotate_templates(delta_t)
        n_dim = 2 * n_tmpl + 1
        nw = max(MCMC_N_WALKERS, 2 * n_dim + 2)
        t0 = time.perf_counter()
        flat, _ = sm.run_mcmc(
            n_sys=n_tmpl, model="combined",
            delta_g_obs=delta_g, delta_t=delta_t_rot,
            n_walkers=nw, n_steps=MCMC_N_STEPS, n_burn=MCMC_N_BURN,
            seed=0, progress=False,
        )
        elapsed = time.perf_counter() - t0
        print(f"\nMCMC-comb | nside={nside} | n_tmpl={n_tmpl} | "
              f"t={elapsed:.2f}s | chain={flat.shape}")
        assert elapsed < 600.0, f"MCMC-comb too slow: {elapsed:.2f}s"
        assert flat.shape[1] == 2 * n_tmpl + 1
