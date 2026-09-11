#!/usr/bin/env python3
"""Check (and calibrate) the clustering amplitude of the GLASS mocks against real data.

Why this matters
----------------
GLASS mocks are the null hypothesis for three separate calibrations in this package:

* ``diagnostics.isd_template_significance``  -- Stage-1 template p-values,
* ``model_selection.lrt_null_distribution``  -- the mock-calibrated LRT,
* ``covariance.mock_sandwich_covariance``    -- the honest amplitude error bar.

All three are only as good as the mocks.  The footprint-matching rule in
``isd_template_significance`` guarantees the mock reproduces the data's *surface
density*, hence its *shot noise* -- but nothing constrains its *clustering*.  If the
mock is under-clustered, every one of those nulls is too narrow and the "calibrated"
answers remain overconfident.

This script measures the mock's clustering variance and compares it with the data's,
then optionally scans ``cl_amplitude`` to find the value that matches.

    sigma_hat^2 = 1/nbar + sigma_clus^2

so with ``nbar`` matched by construction, any difference in ``sigma_hat`` is
clustering.

Usage
-----
    python scripts/calibrate_glass_clustering.py                     # check default
    python scripts/calibrate_glass_clustering.py --scan 5e-4 5e-3 1e-2 4e-2
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import healpy as hp                    # noqa: E402
import sys_mapping as sm               # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "glass_calibration"


def measure(nside, n_total, z_edges, nz, amp, seed, rand_factor=2):
    """Pixel statistics of one mock realisation at amplitude ``amp``.

    ``sigma_clus`` subtracts the shot noise of *both* catalogues.  The galaxy term
    ``1/nbar`` is the obvious one; the random term ``1/(rand_factor*nbar)`` is not,
    and omitting it was a real defect: at ``rand_factor=2`` it inflates
    ``sigma_clus^2`` by ``1/(2*nbar*sigma_clus^2)``, which is under 9% wherever the
    map holds more than ten galaxies per pixel and 38% for the sparsest LS10 cell at
    NSIDE 64.  Because the data's ``sigma_clus`` is measured against the real, far
    denser random catalogue, the inflation does not cancel in the ratio -- it biases
    the fitted amplitude low, and where it dominates it flattens the objective
    entirely, so a mock with no clustering at all matches the data.  ``shot_gal`` and
    ``shot_rand`` are reported separately so the leak is auditable per cell.
    """
    cat = sm.generate_glass_fullsky_mock(nside, n_total, z_edges, nz,
                                         seed=seed, rand_factor=rand_factor,
                                         cl_amplitude=amp)
    ng = sm.pixelize_catalog(cat["ra"], cat["dec"], nside)
    nr = sm.pixelize_catalog(cat["ra_rand"], cat["dec_rand"], nside)
    d, good = sm.compute_overdensity(ng, nr)
    nbar = float(ng[good].mean())
    nbar_rand = float(nr[good].mean())
    var = float(d.var())
    shot_gal = 1.0 / nbar
    shot_rand = 1.0 / nbar_rand if nbar_rand > 0 else 0.0
    return {"cl_amplitude": amp, "nbar": nbar, "nbar_rand": nbar_rand,
            "sigma_hat": float(np.sqrt(var)),
            "shot": shot_gal, "shot_rand": shot_rand,
            "sigma_clus": float(np.sqrt(max(var - shot_gal - shot_rand, 0.0))),
            "sigma_clus_galshot_only":
                float(np.sqrt(max(var - shot_gal, 0.0)))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--n-gal", type=int, default=2_759_238, help="footprint galaxy count")
    ap.add_argument("--fsky", type=float, default=0.4408)
    ap.add_argument("--sigma-hat-data", type=float, default=0.3969,
                    help="measured sigma_hat of the real sample")
    ap.add_argument("--z-range", type=float, nargs=2, default=[0.05, 0.18])
    ap.add_argument("--scan", type=float, nargs="+", default=None,
                    help="cl_amplitude values to try (default: just the package default)")
    ap.add_argument("--fit", action="store_true",
                    help="root-find the cl_amplitude matching sigma_hat_data, "
                         "instead of only evaluating --scan")
    ap.add_argument("--fit-iters", type=int, default=4,
                    help="refinement steps for --fit (each costs one mock)")
    ap.add_argument("--fit-tol", type=float, default=0.02,
                    help="stop once |clustering ratio - 1| is below this")
    ap.add_argument("--min-probes", type=int, default=2,
                    help="Reject a fit that converged in fewer probes than this, "
                         "i.e. one that never searched at all. Kept as cheap "
                         "insurance only: measured over the 36-cell grid the probe "
                         "count does not separate good fits from bad. Healthy "
                         "cells take 2-7 probes and the cells the shot-leak gate "
                         "rejects take 6-7, because they never converge. A "
                         "threshold of 3 rejected 25 healthy seeds and no bad "
                         "ones. --max-shot-leak is the discriminator.")
    ap.add_argument("--max-shot-leak", type=float, default=0.25,
                    help="Reject a cell whose random-catalogue shot variance exceeds "
                         "this fraction of sigma_clus^2; above it the clustering "
                         "ratio no longer responds to the amplitude.")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    n_total = int(a.n_gal / a.fsky)          # the footprint-matching rule
    nbar_data = a.n_gal / (a.fsky * hp.nside2npix(a.nside))
    sclus_data = float(np.sqrt(max(a.sigma_hat_data**2 - 1.0 / nbar_data, 0.0)))

    print(f"data: nside={a.nside} n_gal={a.n_gal:,} fsky={a.fsky} "
          f"nbar={nbar_data:.2f}")
    print(f"      sigma_hat={a.sigma_hat_data:.4f}  shot={1/nbar_data:.5f}  "
          f"-> sigma_clus={sclus_data:.4f}\n")

    rows = []
    print(f"{'cl_amplitude':>14}{'nbar':>9}{'sigma_hat':>11}{'sigma_clus':>12}"
          f"{'clus ratio':>12}")

    def probe(amp):
        r = measure(a.nside, n_total, np.array(a.z_range),
                    np.array([float(a.n_gal)]), amp, a.seed)
        r["sigma_clus_data"] = sclus_data
        r["clustering_ratio"] = r["sigma_clus"] / sclus_data if sclus_data else np.nan
        rows.append(r)
        print(f"{amp:>14.2e}{r['nbar']:>9.2f}{r['sigma_hat']:>11.4f}"
              f"{r['sigma_clus']:>12.4f}{r['clustering_ratio']:>11.2f}x")
        return r

    converged = None
    if a.fit:
        # sigma_clus is very nearly C * amp^(1/2), so work in log-log: two probes
        # determine the local slope exactly and the third lands on the target.
        # A fixed --scan cannot do this -- the required amplitude varies by an
        # order of magnitude across the nine samples, so any grid wide enough to
        # bracket them all is too coarse to be a fit.
        amp = (a.scan or [5e-4])[0]
        r0 = probe(amp)
        prev = None
        for _ in range(max(a.fit_iters, 1)):
            if not np.isfinite(r0["clustering_ratio"]) or r0["clustering_ratio"] <= 0:
                print("  -> mock has no measurable clustering; cannot fit")
                break
            if abs(r0["clustering_ratio"] - 1.0) <= a.fit_tol:
                converged = r0
                break
            if prev is None or prev["sigma_clus"] <= 0 or \
               abs(np.log(r0["cl_amplitude"] / prev["cl_amplitude"])) < 1e-12:
                slope = 0.5                                  # sqrt scaling
            else:
                slope = (np.log(r0["sigma_clus"] / prev["sigma_clus"])
                         / np.log(r0["cl_amplitude"] / prev["cl_amplitude"]))
                if not np.isfinite(slope) or slope <= 0.05:
                    slope = 0.5
            amp_new = r0["cl_amplitude"] * (1.0 / r0["clustering_ratio"]) ** (1.0 / slope)
            amp_new = float(np.clip(amp_new, 1e-8, 1.0))
            prev, r0 = r0, probe(amp_new)
        if converged is None and abs(r0["clustering_ratio"] - 1.0) <= a.fit_tol:
            converged = r0
    else:
        for amp in (a.scan or [5e-4]):
            probe(amp)

    best = min(rows, key=lambda r: abs(r["clustering_ratio"] - 1.0))
    print(f"\nclosest match: cl_amplitude={best['cl_amplitude']:.3e} "
          f"(clustering ratio {best['clustering_ratio']:.2f})")
    if abs(best["clustering_ratio"] - 1.0) > 0.1:
        print("  -> still not matched; widen the scan or raise --fit-iters")

    # Convergence is not the acceptance test.  The failure this guards against is
    # an objective that cannot see the amplitude at all, because the mock's own
    # random-catalogue shot noise dominates sigma_clus^2 -- see `measure`.  Two
    # checks, of very unequal value:
    #   * the shot leak itself, which is the real discriminator;
    #   * too few probes, which is nearly free but, measured over the full grid,
    #     never fires on a cell the leak gate has not already rejected.  Healthy
    #     fits take 2-7 probes; cells that fail take 6-7, because they iterate to
    #     the cap without converging.  It is kept only to catch a fit that
    #     accepted its starting point without searching.
    leak = (best["shot_rand"] / best["sigma_clus"] ** 2
            if best["sigma_clus"] > 0 else float("inf"))
    flat = []
    if a.fit and len(rows) < a.min_probes:
        flat.append(f"converged in {len(rows)} probe(s) < --min-probes={a.min_probes}")
    if leak > a.max_shot_leak:
        flat.append(f"random-catalogue shot noise is {leak:.0%} of sigma_clus^2 "
                    f"(> --max-shot-leak={a.max_shot_leak:.0%})")
    if flat:
        print("  !! REJECTED -- the objective is flat, not solved:")
        for why in flat:
            print(f"     - {why}")
        print("     the amplitude below is the starting point, not a measurement.")

    (a.out_dir / "glass_calibration.json").write_text(
        json.dumps({"data": {"nside": a.nside, "n_gal": a.n_gal, "fsky": a.fsky,
                             "nbar": nbar_data, "sigma_hat": a.sigma_hat_data,
                             "sigma_clus": sclus_data},
                    "mocks": rows,
                    "fit": {"requested": bool(a.fit),
                            "converged": converged is not None,
                            "accepted": converged is not None and not flat,
                            "rejected_because": flat,
                            "shot_leak_fraction": leak,
                            "tol": a.fit_tol,
                            "n_probes": len(rows),
                            "cl_amplitude": best["cl_amplitude"],
                            "clustering_ratio": best["clustering_ratio"],
                            "sigma_hat_mock": best["sigma_hat"]}}, indent=2))
    print(f"\n-> {a.out_dir/'glass_calibration.json'}")


if __name__ == "__main__":
    main()
