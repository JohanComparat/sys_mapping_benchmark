#!/usr/bin/env python3
"""Quantify the bias from dropping cross-template terms in the two-point correction.

The correction implemented in ``contamination.compute_two_point_correction`` is

    w_corr(theta) = [ w_obs(theta) - sum_i a~_i^2 xi_ii(theta) ]
                    / [ 1 + sum_i b~_i^2 xi_ii(theta) ]

i.e. it keeps only template *auto*-correlations.  But the contamination an additive
field f = sum_i a_i t_i actually imprints on w(theta) is its own autocorrelation,

    Delta_w(theta) = xi_ff(theta) = sum_ij a_i a_j xi_ij(theta),

so the terms i != j are dropped.  The PCA rotation diagonalises the template
covariance C = xi(0) at *zero lag* only -- it does not make xi_ij(theta) vanish for
theta > 0 -- so this is an uncontrolled approximation, and its size has never been
measured.  This script measures it.

Two arms
--------
1. *Analytic, on the real fits.*  For the real LS10 template basis and the amplitudes
   actually fitted to each sample (read from ``*_params.json``), form both the exact
   and the auto-only contamination and report the fractional error per angular bin.
   No mock is involved: this is the error the published correction carries.

2. *Rotated basis.*  Repeat in the PCA-rotated basis the pipeline actually fits in,
   to test the implicit assumption that rotation makes the cross terms negligible.

xi_ij(theta) is obtained from the masked cross power spectra via
xi(theta) = sum_l (2l+1)/(4 pi) C_l P_l(cos theta), which is exact for the pseudo-Cl
and identical in construction for auto and cross terms, so the comparison is fair.

Usage
-----
    python scripts/run_crossterm_bias.py
    python scripts/run_crossterm_bias.py --nsides 64 --theta-min 5 --theta-max 300
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import healpy as hp                                    # noqa: E402
from numpy.polynomial.legendre import legval           # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "crossterm_bias"
SYS_MAPPING = Path(
    os.environ.get("SYS_MAPPING_ROOT", Path.home() / "software" / "sys_mapping")
).expanduser()
SYST = Path.home() / "data" / "legacysurvey" / "dr10" / "systematics"
_ZFILL = {"LS10": 4, "GAIA": 5}


def load_templates(names, nside):
    maps = []
    for name in names:
        src = name.split("_")[0]
        stem = re.sub(r"_NSIDE_\d+$", "", name)
        p = SYST / f"{nside:04d}" / f"{stem}_NSIDE_{nside:0{_ZFILL.get(src,4)}d}.fits"
        maps.append(np.asarray(hp.read_map(str(p)), float))
    T = np.vstack(maps)
    T[~np.isfinite(T)] = hp.UNSEEN
    valid = np.all(T != hp.UNSEEN, axis=0)
    return T, valid


def xi_matrix(T_full, valid, nside, theta_deg):
    """Return xi[i,j,theta]: masked template cross-correlation functions.

    Computed from pseudo-C_l via the Legendre sum, identically for i==j and i!=j so
    that the auto/cross comparison is like-for-like.
    """
    n_sys = T_full.shape[0]
    mask = valid.astype(float)
    lmax = 2 * nside
    maps = []
    for i in range(n_sys):
        m = np.zeros(T_full.shape[1])
        v = T_full[i, valid]
        m[valid] = (v - v.mean()) / v.std()
        maps.append(m * mask)
    w2 = float(np.mean(mask**2))                      # first-order mask correction

    ell = np.arange(lmax + 1, dtype=float)
    pref = (2 * ell + 1) / (4 * np.pi)
    x = np.cos(np.radians(theta_deg))

    xi = np.zeros((n_sys, n_sys, len(theta_deg)))
    for i in range(n_sys):
        for j in range(i, n_sys):
            cl = hp.anafast(maps[i], map2=maps[j], lmax=lmax) / max(w2, 1e-12)
            coef = pref * cl
            xi[i, j] = legval(x, coef)
            xi[j, i] = xi[i, j]
    return xi


def contamination_terms(a, xi):
    """Return (exact, auto_only) contamination in w(theta) for amplitudes a."""
    exact = np.einsum("i,j,ijt->t", a, a, xi)
    auto = np.einsum("i,iit->t", a**2, xi)
    return exact, auto


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nsides", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--theta-min", type=float, default=0.5, help="arcmin")
    ap.add_argument("--theta-max", type=float, default=300.0, help="arcmin")
    ap.add_argument("--nbins", type=int, default=12)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    theta_arcmin = np.logspace(np.log10(a.theta_min), np.log10(a.theta_max), a.nbins)
    theta_deg = theta_arcmin / 60.0

    rows = []
    for nside in a.nsides:
        files = sorted(glob.glob(str(SYS_MAPPING / "data" / "sys_weights"
                                      / f"*_NSIDE{nside:04d}_params.json")))
        if not files:
            print(f"  no params.json at NSIDE {nside}")
            continue
        first = json.loads(Path(files[0]).read_text())
        names = first["template_names"]
        T_full, valid = load_templates(names, nside)
        print(f"NSIDE {nside}: {len(names)} templates, {int(valid.sum())} px, "
              f"{len(files)} samples")
        xi = xi_matrix(T_full, valid, nside, theta_deg)

        # PCA rotation of the same basis, to test whether rotating helps
        Tv = T_full[:, valid]
        Tv = (Tv - Tv.mean(1, keepdims=True)) / Tv.std(1, keepdims=True)
        C = Tv @ Tv.T / Tv.shape[1]
        evals, evecs = np.linalg.eigh(C)
        R = evecs[:, np.argsort(evals)[::-1]].T          # rows are eigenvectors
        xi_rot = np.einsum("ai,ijt,bj->abt", R, xi, R)

        for f in files:
            d = json.loads(Path(f).read_text())
            a_hat = np.asarray(d.get("a_hat_add") or [], float)
            if a_hat.size != len(names):
                continue
            mass = float(re.search(r"VLIM_ANY_([0-9.]+)_Mstar", d["sample_id"]).group(1))

            for basis, X in (("original", xi), ("pca-rotated", xi_rot)):
                amp = a_hat if basis == "original" else R @ a_hat
                exact, auto = contamination_terms(amp, X)
                with np.errstate(divide="ignore", invalid="ignore"):
                    frac = np.where(np.abs(exact) > 0, (auto - exact) / exact, np.nan)
                rows.append({
                    "nside": nside, "logM": mass, "basis": basis,
                    "theta_arcmin": theta_arcmin.tolist(),
                    "exact": exact.tolist(), "auto_only": auto.tolist(),
                    "frac_error": frac.tolist(),
                    "median_abs_frac_error": float(np.nanmedian(np.abs(frac))),
                    "max_abs_frac_error": float(np.nanmax(np.abs(frac))),
                })

    if not rows:
        print("nothing computed")
        return

    (a.out_dir / "crossterm_bias.json").write_text(json.dumps(rows, indent=2))

    import csv
    with (a.out_dir / "crossterm_bias.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["nside", "logM", "basis", "theta_arcmin", "exact",
                    "auto_only", "frac_error"])
        for r in rows:
            for t, e, au, fr in zip(r["theta_arcmin"], r["exact"],
                                    r["auto_only"], r["frac_error"]):
                w.writerow([r["nside"], r["logM"], r["basis"],
                            f"{t:.4f}", f"{e:.6e}", f"{au:.6e}", f"{fr:.6f}"])

    print(f"\n{'nside':>6}{'basis':>14}{'median |err|':>14}{'max |err|':>12}")
    for nside in sorted({r["nside"] for r in rows}):
        for basis in ("original", "pca-rotated"):
            sub = [r for r in rows if r["nside"] == nside and r["basis"] == basis]
            if sub:
                med = np.median([r["median_abs_frac_error"] for r in sub])
                mx = np.max([r["max_abs_frac_error"] for r in sub])
                print(f"{nside:>6}{basis:>14}{med*100:>13.1f}%{mx*100:>11.1f}%")
    print(f"\n-> {a.out_dir/'crossterm_bias.csv'}")


if __name__ == "__main__":
    main()
