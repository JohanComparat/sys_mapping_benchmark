#!/usr/bin/env python3
"""Measure the variance inflation of the iid pixel likelihood on a correlated field.

Motivation
----------
The recorded null test (``docs/_static/results_sim_null_test/summary_null_test.csv``)
reports a 3-sigma false-positive rate of 76-96% on systematics-free simulations, where
0.27% is expected.  That is known, but it is only a *flag*: it does not say by how much
the error bars are wrong, nor how the error depends on resolution, template count, or
template correlation.  Without that, the only calibrated route is to run a mock
ensemble per sample, which is what makes the calibrated path expensive.

This script measures the inflation directly.  On *uncontaminated* mocks (a_true = 0)
it compares

    sigma_emp[i] = std over realisations of  a_hat[i]        (the truth)
    sigma_iid[i] = OLS analytic error  sqrt(sigma^2 (T T^T)^-1_ii)   (what is reported)

and reports the ratio ``kappa[i] = sigma_emp[i] / sigma_iid[i]``, which is the factor
by which a reported significance is too large.  Because a_true = 0, any non-zero
a_hat is pure noise, so sigma_emp is unbiased by construction.

It also records the *field*-level inflation, ``kappa_field``, using
``A = rms(sum_i a_hat_i t_i)``, since the field statistic is the VIF-free one the
detectability law uses.

Grid: NSIDE x n_templates x mock family (lognormal / GLASS), n_real realisations each.

Usage
-----
    python scripts/run_variance_inflation.py --quick
    python scripts/run_variance_inflation.py --nsides 32 64 --n-sys 3 5 11 --n-real 200
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import healpy as hp                                   # noqa: E402
import sys_mapping as sm                              # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "variance_inflation"
SYST = Path.home() / "data" / "legacysurvey" / "dr10" / "systematics"
_ZFILL = {"LS10": 4, "GAIA": 5}

# Real LS10 templates, ordered so the first few are the least mutually degenerate.
REAL_TEMPLATES = [
    "GAIA_nstar_faint", "LS10_EBV", "LS10_PSFSIZE_R", "LS10_GALDEPTH_Z",
    "LS10_NOBS_R", "GAIA_nstar_medium", "LS10_GALDEPTH_G", "LS10_GALDEPTH_R",
    "GAIA_phot_g_mean_flux", "GAIA_phot_bp_mean_flux", "GAIA_phot_rp_mean_flux",
]


def load_real(names, nside):
    maps = []
    for name in names:
        src = name.split("_")[0]
        p = SYST / f"{nside:04d}" / f"{name}_NSIDE_{nside:0{_ZFILL.get(src,4)}d}.fits"
        m = hp.read_map(str(p), verbose=False) if hasattr(hp, "read_map") else None
        maps.append(np.asarray(m, float))
    T = np.vstack(maps)
    T[~np.isfinite(T)] = hp.UNSEEN
    valid = np.all(T != hp.UNSEEN, axis=0)
    return T, valid


def ols_fit(delta_g, T):
    """Return (a_hat, sigma_iid, s2) -- estimate, analytic iid error, residual variance."""
    X = T.T
    a_hat, *_ = np.linalg.lstsq(X, delta_g, rcond=None)
    resid = delta_g - X @ a_hat
    dof = max(X.shape[0] - X.shape[1], 1)
    s2 = float(resid @ resid) / dof
    XtX_inv = np.linalg.pinv(X.T @ X)
    return a_hat, np.sqrt(np.maximum(s2 * np.diag(XtX_inv), 1e-300)), s2


def lognormal_delta(nside, sigma, rng, slope=2.0):
    """Clustered, systematics-free overdensity: the field the iid model gets wrong.

    ``slope`` is the power-law index of C_l ~ (l+1)^-slope.  It matters a great deal:
    the inflation is driven by how much power the field has on the large, smooth
    scales the templates live on, so kappa is a function of the assumed spectrum and
    must be quoted with it.  slope=2 is the package default in
    ``mocks.generate_lognormal_field``.
    """
    lmax = 3 * nside - 1
    ell = np.arange(lmax + 1, dtype=float)
    cl = (ell + 1.0) ** (-slope)
    cl[0] = 0.0
    cl *= sigma**2 / np.sum((2 * ell + 1) / (4 * np.pi) * cl)
    np.random.seed(int(rng.integers(0, 2**31 - 1)))     # hp.synfast uses the global RNG
    G = hp.synfast(cl, nside=nside, lmax=lmax, verbose=False) if "verbose" in \
        hp.synfast.__code__.co_varnames else hp.synfast(cl, nside=nside, lmax=lmax)
    return np.exp(G - 0.5 * sigma**2) - 1.0


def run_cell(nside, n_sys, n_real, n_mean, sigma_clus, seed, shot_only=False,
             slope=2.0):
    T_full, valid = load_real(REAL_TEMPLATES[:n_sys], nside)
    T = T_full[:, valid]
    T = (T - T.mean(1, keepdims=True)) / T.std(1, keepdims=True)
    n_pix = T.shape[1]
    rng = np.random.default_rng(seed)

    a_hats, sig_iids, fields, resid_var = [], [], [], []
    for k in range(n_real):
        if shot_only:
            # Pure Poisson: the regime the iid likelihood is actually valid in.
            counts = rng.poisson(n_mean, size=n_pix).astype(float)
            delta = counts / counts.mean() - 1.0
        else:
            d_true = lognormal_delta(nside, sigma_clus, rng, slope)[valid]
            counts = rng.poisson(np.maximum(n_mean * (1.0 + d_true), 0.0))
            delta = counts / max(counts.mean(), 1e-12) - 1.0
        a_hat, sig, s2 = ols_fit(delta, T)
        a_hats.append(a_hat); sig_iids.append(sig); resid_var.append(s2)
        fields.append(float(np.mean((a_hat @ T) ** 2)))     # A^2, not A

    A = np.vstack(a_hats); S = np.vstack(sig_iids)
    F2 = np.asarray(fields)                                  # per-realisation A^2
    sigma_emp = A.std(axis=0, ddof=1)
    sigma_iid = S.mean(axis=0)
    kappa = sigma_emp / np.maximum(sigma_iid, 1e-300)

    # Field level.  With a_hat ~ N(0, Sigma) the reconstructed field satisfies
    # E[A^2] = tr(C Sigma) for C = T T^T / n_pix.  Under the iid model
    # Sigma = s2 (T T^T)^-1, so E[A^2]_iid = s2 * n_sys / n_pix exactly, whatever the
    # template correlation.  Comparing second moments (not std of A, which is
    # chi-distributed and not directly comparable) gives the field inflation.
    e_a2_iid = float(np.mean(resid_var)) * n_sys / n_pix
    kappa_field = float(np.sqrt(np.mean(F2) / max(e_a2_iid, 1e-300)))

    # 3-sigma false-positive rate on the *reported* (iid) significance
    z = np.abs(A) / np.maximum(S, 1e-300)
    fpr3_per_template = float(np.mean(z > 3.0))
    fpr3_any_template = float(np.mean((z > 3.0).any(axis=1)))

    return {
        "nside": nside, "n_sys": n_sys, "n_pix": n_pix, "n_real": n_real,
        "n_mean": n_mean, "sigma_clus": 0.0 if shot_only else sigma_clus,
        "regime": "shot-only" if shot_only else "clustered",
        "cl_slope": 0.0 if shot_only else slope,
        "kappa_per_template_median": float(np.median(kappa)),
        "kappa_per_template_max": float(kappa.max()),
        "kappa_per_template": [float(x) for x in kappa],
        "kappa_field": kappa_field,
        "fpr3_per_template": fpr3_per_template,
        "fpr3_any_template": fpr3_any_template,
        "sigma_emp_median": float(np.median(sigma_emp)),
        "sigma_iid_median": float(np.median(sigma_iid)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--nsides", type=int, nargs="+", default=None)
    ap.add_argument("--n-sys", type=int, nargs="+", default=None)
    ap.add_argument("--n-real", type=int, default=None)
    ap.add_argument("--n-mean", type=float, default=127.0, help="galaxies per pixel")
    ap.add_argument("--sigma-clus", type=float, default=0.4,
                    help="clustering amplitude of the lognormal field")
    ap.add_argument("--cl-slopes", type=float, nargs="+", default=[2.0],
                    help="power-law indices of C_l ~ (l+1)^-slope to scan")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tag", default=None,
                    help="Run tag. Outputs go to <out-dir>/<tag>/ so a narrow scan can "
                         "never overwrite a wide grid -- which is exactly how the "
                         "original 20-cell (nside x n_sys) grid was lost. Defaults to a "
                         "descriptive tag built from the grid itself.")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    a = ap.parse_args()

    nsides = a.nsides or ([32] if a.quick else [32, 64])
    n_sys_grid = a.n_sys or ([3, 11] if a.quick else [1, 3, 5, 8, 11])
    n_real = a.n_real or (40 if a.quick else 200)

    # Run-tagged output directory.  Without this, two runs with different grids write
    # the same filename and the second silently destroys the first.
    tag = a.tag or (
        "ns" + "-".join(str(n) for n in nsides)
        + "_k" + "-".join(str(k) for k in n_sys_grid)
        + "_slope" + "-".join(f"{s:g}" for s in a.cl_slopes)
        + f"_n{n_real}"
    )
    a.out_dir = a.out_dir / tag
    a.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run tag: {tag}")
    rows = []
    print(f"variance inflation: nside={nsides} n_sys={n_sys_grid} n_real={n_real} "
          f"nbar={a.n_mean} sigma_clus={a.sigma_clus}\n")
    print(f"{'regime':<11}{'slope':>6}{'nside':>6}{'n_sys':>7}{'n_pix':>8}"
          f"{'kappa_tmpl':>12}{'kappa_field':>13}{'FPR3(any)':>11}")
    for shot in (True, False):
        slopes = [0.0] if shot else a.cl_slopes
        for slope in slopes:
            for ns in nsides:
                for k in n_sys_grid:
                    r = run_cell(ns, k, n_real, a.n_mean, a.sigma_clus,
                                 a.seed + 1000 * ns + k, shot_only=shot, slope=slope)
                    rows.append(r)
                    print(f"{r['regime']:<11}{slope:>6.1f}{ns:>6}{k:>7}{r['n_pix']:>8}"
                          f"{r['kappa_per_template_median']:>12.2f}"
                          f"{r['kappa_field']:>13.2f}{r['fpr3_any_template']*100:>10.1f}%")

    (a.out_dir / "variance_inflation.json").write_text(json.dumps(
        {"grid": {"nsides": nsides, "n_sys": n_sys_grid, "n_real": n_real,
                  "cl_slopes": a.cl_slopes, "n_mean": a.n_mean,
                  "sigma_clus": a.sigma_clus, "seed": a.seed, "tag": tag},
         "rows": rows}, indent=2))
    import csv
    keys = [k for k in rows[0] if k != "kappa_per_template"]
    with (a.out_dir / "variance_inflation.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    print(f"\n-> {a.out_dir/'variance_inflation.csv'}")


if __name__ == "__main__":
    main()
