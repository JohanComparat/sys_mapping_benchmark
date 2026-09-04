"""Timing benchmarks for sys_mapping functions.

Run with::

    pytest tests/test_benchmarks.py -v -s

The ``-s`` flag is required to see timing output printed to stdout.
Each test measures wall-clock time per call using :mod:`timeit` after a
JAX/Python warmup to exclude compilation overhead.

Results are documented in each function's docstring.
"""

from __future__ import annotations

import timeit

import numpy as np
import jax.numpy as jnp
import pytest

# ---------------------------------------------------------------------------
# Benchmark constants and shared data
# ---------------------------------------------------------------------------

N_PIX = 10_000   # representative pixel count for most benchmarks
N_SYS = 5        # number of systematic templates
N_BINS = 15      # angular bins for two-point function
NSIDE_BENCH = 64 # HEALPix resolution (~49 k pixels)

_RNG = np.random.default_rng(42)

_delta_t = _RNG.standard_normal((N_SYS, N_PIX))
_delta_t -= _delta_t.mean(axis=1, keepdims=True)
_delta_t /= _delta_t.std(axis=1, keepdims=True)
_delta_g = _RNG.standard_normal(N_PIX) * 0.1
_a = _RNG.uniform(-0.05, 0.05, N_SYS)
_b = _RNG.uniform(-0.03, 0.03, N_SYS)
_sigma = float(np.std(_delta_g))
_w_obs = _RNG.standard_normal(N_BINS) * 0.01
_tcorr = np.abs(_RNG.standard_normal((N_SYS, N_BINS))) * 0.001


def _bench(fn, n_warmup: int = 3, n_repeat: int = 100) -> float:
    """Return mean wall-clock time in seconds per call."""
    for _ in range(n_warmup):
        fn()
    return timeit.timeit(fn, number=n_repeat) / n_repeat


# ---------------------------------------------------------------------------
# contamination.py
# ---------------------------------------------------------------------------


class TestTimingContamination:
    """Timing for apply_contamination, invert_contamination, compute_two_point_correction."""

    def test_apply_contamination(self):
        from sys_mapping.contamination import apply_contamination

        dg = jnp.asarray(_delta_g)
        dt = jnp.asarray(_delta_t)
        aj, bj = jnp.asarray(_a), jnp.asarray(_b)

        t = _bench(lambda: apply_contamination(dg, dt, aj, bj).block_until_ready())
        print(f"\n  apply_contamination          n_pix={N_PIX}, n_sys={N_SYS}: {t*1e6:6.1f} μs/call")

    def test_invert_contamination(self):
        from sys_mapping.contamination import apply_contamination, invert_contamination

        dg = jnp.asarray(_delta_g)
        dt = jnp.asarray(_delta_t)
        aj, bj = jnp.asarray(_a), jnp.asarray(_b)
        obs = apply_contamination(dg, dt, aj, bj)

        t = _bench(lambda: invert_contamination(obs, dt, aj, bj).block_until_ready())
        print(f"\n  invert_contamination         n_pix={N_PIX}, n_sys={N_SYS}: {t*1e6:6.1f} μs/call")

    def test_compute_two_point_correction(self):
        from sys_mapping.contamination import compute_two_point_correction

        w = jnp.asarray(_w_obs)
        a_sq = jnp.asarray(_a ** 2)
        b_sq = jnp.asarray(_b ** 2)
        tc = jnp.asarray(_tcorr)

        t = _bench(lambda: compute_two_point_correction(w, a_sq, b_sq, tc).block_until_ready())
        print(f"\n  compute_two_point_correction n_sys={N_SYS}, n_bins={N_BINS}: {t*1e6:6.1f} μs/call")


# ---------------------------------------------------------------------------
# maps.py
# ---------------------------------------------------------------------------


class TestTimingMaps:
    """Timing for map generation and catalog pixelization."""

    def test_systematic_power_spectrum(self):
        from sys_mapping.maps import systematic_power_spectrum

        t = _bench(lambda: systematic_power_spectrum(NSIDE_BENCH, 0), n_repeat=500)
        print(f"\n  systematic_power_spectrum    nside={NSIDE_BENCH}: {t*1e6:6.1f} μs/call")

    def test_generate_systematic_map(self):
        from sys_mapping.maps import generate_systematic_map

        t = _bench(lambda: generate_systematic_map(NSIDE_BENCH, 0, seed=42),
                   n_warmup=1, n_repeat=10)
        print(f"\n  generate_systematic_map      nside={NSIDE_BENCH}: {t*1e3:6.1f} ms/call")

    def test_pixelize_catalog(self):
        from sys_mapping.maps import pixelize_catalog

        n_gal = 100_000
        rng = np.random.default_rng(7)
        ra = rng.uniform(0, 360, n_gal)
        dec = rng.uniform(-90, 90, n_gal)

        t = _bench(lambda: pixelize_catalog(ra, dec, NSIDE_BENCH), n_repeat=20)
        print(f"\n  pixelize_catalog             n_gal={n_gal}, nside={NSIDE_BENCH}: {t*1e3:6.1f} ms/call")

    def test_compute_overdensity(self):
        from sys_mapping.maps import pixelize_catalog, compute_overdensity

        rng = np.random.default_rng(7)
        gal = pixelize_catalog(rng.uniform(0, 360, 50_000), rng.uniform(-90, 90, 50_000), NSIDE_BENCH)
        rand = pixelize_catalog(rng.uniform(0, 360, 250_000), rng.uniform(-90, 90, 250_000), NSIDE_BENCH)

        t = _bench(lambda: compute_overdensity(gal, rand), n_repeat=200)
        print(f"\n  compute_overdensity          n_pix={12*NSIDE_BENCH**2}: {t*1e6:6.1f} μs/call")

    def test_assign_template_values(self):
        from sys_mapping.maps import assign_template_values

        rng = np.random.default_rng(0)
        templates = rng.standard_normal((N_SYS, 12 * NSIDE_BENCH ** 2))
        mask = rng.random(12 * NSIDE_BENCH ** 2) > 0.1

        t = _bench(lambda: assign_template_values(templates, mask), n_repeat=500)
        print(f"\n  assign_template_values       n_sys={N_SYS}: {t*1e6:6.1f} μs/call")


# ---------------------------------------------------------------------------
# likelihood.py
# ---------------------------------------------------------------------------


class TestTimingLikelihood:
    """Timing for JIT-compiled log-likelihood evaluation."""

    def test_log_likelihood_jit_warmup(self):
        """First-call compile time."""
        import time
        from sys_mapping.likelihood import make_log_likelihood
        from sys_mapping.contamination import pack_params

        theta = jnp.asarray(pack_params(_a, _b, _sigma, model="combined"))
        dg, dt = jnp.asarray(_delta_g), jnp.asarray(_delta_t)
        log_lik = make_log_likelihood(N_SYS, "combined")

        t0 = time.perf_counter()
        log_lik(theta, dg, dt).block_until_ready()
        t1 = time.perf_counter()
        print(f"\n  log_likelihood JIT compile   first call: {(t1-t0)*1e3:6.0f} ms")

    def test_log_likelihood_after_jit(self):
        """Steady-state time after JIT compilation."""
        from sys_mapping.likelihood import make_log_likelihood
        from sys_mapping.contamination import pack_params

        theta = jnp.asarray(pack_params(_a, _b, _sigma, model="combined"))
        dg, dt = jnp.asarray(_delta_g), jnp.asarray(_delta_t)
        log_lik = make_log_likelihood(N_SYS, "combined")
        log_lik(theta, dg, dt).block_until_ready()  # warmup

        t = _bench(lambda: log_lik(theta, dg, dt).block_until_ready())
        print(f"\n  log_likelihood (post-JIT)    n_pix={N_PIX}, n_sys={N_SYS}: {t*1e6:6.1f} μs/call")

    def test_log_likelihood_skewed_after_jit(self):
        from sys_mapping.likelihood import make_log_likelihood
        from sys_mapping.contamination import pack_params

        theta = jnp.asarray(pack_params(_a, _b, _sigma, gamma=0.5, model="combined"))
        dg, dt = jnp.asarray(_delta_g), jnp.asarray(_delta_t)
        log_lik = make_log_likelihood(N_SYS, "combined", use_skewed=True)
        log_lik(theta, dg, dt).block_until_ready()

        t = _bench(lambda: log_lik(theta, dg, dt).block_until_ready())
        print(f"\n  log_likelihood skew (post-JIT) n_pix={N_PIX}, n_sys={N_SYS}: {t*1e6:6.1f} μs/call")


# ---------------------------------------------------------------------------
# correction.py
# ---------------------------------------------------------------------------


class TestTimingCorrection:
    """Timing for parameter debiasing and template rotation."""

    def test_debias_params(self):
        from sys_mapping.correction import debias_params

        var = np.full(N_SYS, 5e-4)
        t = _bench(lambda: debias_params(_a, _b, var, var), n_repeat=1000)
        print(f"\n  debias_params                n_sys={N_SYS}: {t*1e9:6.0f} ns/call")

    def test_rotate_templates(self):
        from sys_mapping.correction import rotate_templates

        t = _bench(lambda: rotate_templates(_delta_t), n_repeat=200)
        print(f"\n  rotate_templates             n_pix={N_PIX}, n_sys={N_SYS}: {t*1e6:6.1f} μs/call")

    def test_transform_params_from_rotated(self):
        from sys_mapping.correction import rotate_templates, transform_params_from_rotated

        _, R, _ = rotate_templates(_delta_t)
        a_rot = R @ _a
        b_rot = R @ _b

        t = _bench(lambda: transform_params_from_rotated(a_rot, b_rot, R), n_repeat=1000)
        print(f"\n  transform_params_from_rotated n_sys={N_SYS}: {t*1e9:6.0f} ns/call")

    def test_correct_two_point_function(self):
        from sys_mapping.correction import correct_two_point_function

        var = np.full(N_SYS, 5e-4)
        tc = np.abs(_tcorr)

        t = _bench(lambda: correct_two_point_function(_w_obs, _a, _b, var, var, tc),
                   n_repeat=200)
        print(f"\n  correct_two_point_function   n_sys={N_SYS}, n_bins={N_BINS}: {t*1e6:6.1f} μs/call")


# ---------------------------------------------------------------------------
# utils.py
# ---------------------------------------------------------------------------


class TestTimingUtils:
    """Timing for covariance and amplitude bias utilities."""

    def test_compute_covariance_matrix(self):
        from sys_mapping.utils import compute_covariance_matrix

        t = _bench(lambda: compute_covariance_matrix(_delta_t))
        print(f"\n  compute_covariance_matrix    n_pix={N_PIX}, n_sys={N_SYS}: {t*1e6:6.1f} μs/call")

    def test_compute_amplitude_bias(self):
        from sys_mapping.utils import compute_covariance_matrix, compute_amplitude_bias

        cov = compute_covariance_matrix(_delta_t)

        t = _bench(lambda: compute_amplitude_bias(_a ** 2, _b ** 2, cov), n_repeat=1000)
        print(f"\n  compute_amplitude_bias       n_sys={N_SYS}: {t*1e9:6.0f} ns/call")


# ---------------------------------------------------------------------------
# model_selection.py
# ---------------------------------------------------------------------------


class TestTimingModelSelection:
    """Timing for likelihood ratio test."""

    def test_likelihood_ratio_test(self):
        from sys_mapping.model_selection import likelihood_ratio_test
        from sys_mapping.contamination import pack_params

        n_sys = 3
        rng = np.random.default_rng(0)
        dg = rng.standard_normal(N_PIX) * 0.1
        dt = rng.standard_normal((n_sys, N_PIX))
        a = np.zeros(n_sys)
        b = np.zeros(n_sys)
        theta_add = pack_params(a, None, 0.1, model="additive")
        theta_comb = pack_params(a, b, 0.1, model="combined")

        # JIT warmup
        likelihood_ratio_test(dg, dt, theta_add, theta_comb, "additive", "combined")

        t = _bench(
            lambda: likelihood_ratio_test(dg, dt, theta_add, theta_comb, "additive", "combined"),
            n_repeat=50,
        )
        print(f"\n  likelihood_ratio_test        n_pix={N_PIX}, n_sys={n_sys}: {t*1e6:6.1f} μs/call")


# ---------------------------------------------------------------------------
# bootstrap.py
# ---------------------------------------------------------------------------


class TestTimingBootstrap:
    """Timing for block bootstrap variance estimation."""

    def test_block_bootstrap_variance(self):
        from sys_mapping.bootstrap import block_bootstrap_variance
        from sys_mapping.contamination import apply_contamination

        nside = 32
        n_pix_full = 12 * nside ** 2
        rng = np.random.default_rng(0)
        good_pixels = np.ones(n_pix_full, dtype=bool)
        dg = rng.standard_normal(n_pix_full) * 0.1
        n_sys = 3
        dt = rng.standard_normal((n_sys, n_pix_full))

        def estimator(dg_b, dt_b):
            return np.array([np.mean(dg_b)])

        t = _bench(
            lambda: block_bootstrap_variance(dg, dt, good_pixels, nside, estimator,
                                             n_bootstrap=50, n_patches=8),
            n_warmup=1, n_repeat=5,
        )
        print(f"\n  block_bootstrap_variance     n_boot=50, n_patches=8: {t*1e3:6.1f} ms/call")
