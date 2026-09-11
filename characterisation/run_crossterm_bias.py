#!/usr/bin/env python3
"""Measure what dropping the cross-template terms costs the two-point correction.

``docs/results_algorithm_characterisation.rst`` has long quoted a 7--17 % median
error from the auto-only approximation and attributed it to a script of this name
that was not in the repository, so the number could not be checked.  This is that
script, written against the implemented full-sum path
(:func:`sys_mapping.contamination.compute_two_point_correction` with a
``(n_sys, n_sys, n_bins)`` correlation matrix and
:func:`sys_mapping.correction.debias_params_matrix`).

The quantity reported is

    | (w_obs - w_full) - (w_obs - w_auto) | / | w_obs - w_full |

that is, the fraction of the *correction* that the cross terms carry.  It depends
on the amplitude pattern as well as on the basis, so ``--params-json`` should be
given the amplitudes actually fitted to the data; random amplitudes of the right
size give the right order of magnitude and nothing sharper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import healpy as hp
import numpy as np
from numpy.polynomial.legendre import legval

import sys_mapping as sm
from sys_mapping.correction import rotate_templates


def template_xi_matrix(templates, good, nside, theta_rad, lmax=None):
    """Full ``xi_ij(theta)`` between every template pair, via cross power spectra."""
    n = templates.shape[0]
    lmax = int(lmax or 2 * nside)
    ell = np.arange(lmax + 1)
    npix = hp.nside2npix(nside)
    maps = np.zeros((n, npix))
    maps[:, good] = templates

    def w_of(cl):
        return legval(np.cos(theta_rad), (2 * ell + 1) / (4 * np.pi) * cl)

    xi = np.zeros((n, n, theta_rad.size))
    for i in range(n):
        for j in range(i, n):
            cl = hp.anafast(maps[i], maps[j], lmax=lmax, use_pixel_weights=True)
            xi[i, j] = xi[j, i] = w_of(cl)
    return xi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template-dir", required=True,
                    help="Directory of template FITS for one NSIDE.")
    ap.add_argument("--nside", type=int, required=True)
    ap.add_argument("--params-json", type=Path, default=None,
                    help="A *_params.json from run_ls10_analysis.py; the fitted "
                         "a_hat is read from it. Without it, random amplitudes of "
                         "the typical size are used and the result is indicative.")
    ap.add_argument("--method", default="MCMC-add",
                    help="Which method's a_hat to read from --params-json.")
    ap.add_argument("--theta-min", type=float, default=0.5, help="degrees")
    ap.add_argument("--theta-max", type=float, default=5.0, help="degrees")
    ap.add_argument("--n-theta", type=int, default=10)
    ap.add_argument("--rotate", action="store_true", default=True,
                    help="Work in the PCA-rotated basis the pipeline fits in.")
    ap.add_argument("--no-rotate", dest="rotate", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    import importlib.util
    import sys as _sys
    _ls10 = Path(__file__).resolve().parents[2] / "sys_mapping" / "scripts" / "run_ls10_analysis.py"
    if not _ls10.exists():
        _ls10 = Path.home() / "software" / "sys_mapping" / "scripts" / "run_ls10_analysis.py"
    spec = importlib.util.spec_from_file_location("_ls10", str(_ls10))
    mod = importlib.util.module_from_spec(spec)
    _sys.modules["_ls10"] = mod
    spec.loader.exec_module(mod)

    T = np.atleast_2d(np.asarray(mod.load_templates_from_dir(a.template_dir, a.nside)[0]))
    good = np.all(np.isfinite(T), axis=0) & np.all(T != 0, axis=0)
    Tv = T[:, good]
    Tv = (Tv - Tv.mean(1, keepdims=True)) / Tv.std(1, keepdims=True)
    n_sys = Tv.shape[0]

    basis = Tv
    if a.rotate:
        rot = rotate_templates(Tv)
        basis = rot[0] if isinstance(rot, tuple) else rot

    if a.params_json is not None:
        d = json.loads(Path(a.params_json).read_text())
        # run_ls10_analysis.py stores every method's amplitudes under "methods";
        # the two MCMC vectors are also at the top level for backward compatibility.
        meth = d.get("methods") or {}
        if a.method in meth:
            amps = np.asarray(meth[a.method].get("a_hat", []), dtype=float)
        else:
            amps = np.asarray(d.get("a_hat_add", []), dtype=float)
        if amps.size != n_sys:
            raise SystemExit(
                f"{a.params_json}: {a.method}.a_hat has {amps.size} entries, "
                f"expected {n_sys}")
        source = f"{a.params_json.name}:{a.method}"
    else:
        amps = np.random.default_rng(a.seed).normal(0, 0.03, n_sys)
        source = "random (indicative only)"

    theta = np.radians(np.linspace(a.theta_min, a.theta_max, a.n_theta))
    xi = template_xi_matrix(basis, good, a.nside, theta)
    autos = np.array([xi[i, i] for i in range(n_sys)])

    var = np.full(n_sys, 1e-5)
    w_obs = np.full(theta.size, 1e-3)
    z = np.zeros(n_sys)
    d_full = w_obs - sm.correct_two_point_function(w_obs, amps, z, var, z, xi)
    d_auto = w_obs - sm.correct_two_point_function(w_obs, amps, z, var, z, autos)
    frac = np.abs(d_full - d_auto) / np.abs(d_full)

    print(f"NSIDE {a.nside}  n_sys={n_sys}  basis={'rotated' if a.rotate else 'raw'}  "
          f"amplitudes={source}")
    print(f"  cross-term share of the correction: median {np.median(frac):.1%}, "
          f"range {frac.min():.1%}-{frac.max():.1%}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({
            "nside": a.nside, "n_sys": n_sys, "rotated": bool(a.rotate),
            "amplitude_source": source, "theta_deg": np.degrees(theta).tolist(),
            "frac_cross": frac.tolist(), "median_frac_cross": float(np.median(frac)),
        }, indent=1))
        print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
