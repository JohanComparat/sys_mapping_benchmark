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
    cat = sm.generate_glass_fullsky_mock(nside, n_total, z_edges, nz,
                                         seed=seed, rand_factor=rand_factor,
                                         cl_amplitude=amp)
    ng = sm.pixelize_catalog(cat["ra"], cat["dec"], nside)
    nr = sm.pixelize_catalog(cat["ra_rand"], cat["dec_rand"], nside)
    d, good = sm.compute_overdensity(ng, nr)
    nbar = float(ng[good].mean())
    var = float(d.var())
    return {"cl_amplitude": amp, "nbar": nbar, "sigma_hat": float(np.sqrt(var)),
            "shot": 1.0 / nbar,
            "sigma_clus": float(np.sqrt(max(var - 1.0 / nbar, 0.0)))}


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

    amps = a.scan or [5e-4]
    rows = []
    print(f"{'cl_amplitude':>14}{'nbar':>9}{'sigma_hat':>11}{'sigma_clus':>12}"
          f"{'clus ratio':>12}")
    for amp in amps:
        r = measure(a.nside, n_total, np.array(a.z_range),
                    np.array([float(a.n_gal)]), amp, a.seed)
        r["sigma_clus_data"] = sclus_data
        r["clustering_ratio"] = r["sigma_clus"] / sclus_data if sclus_data else np.nan
        rows.append(r)
        print(f"{amp:>14.2e}{r['nbar']:>9.2f}{r['sigma_hat']:>11.4f}"
              f"{r['sigma_clus']:>12.4f}{r['clustering_ratio']:>11.2f}x")

    best = min(rows, key=lambda r: abs(r["clustering_ratio"] - 1.0))
    print(f"\nclosest match: cl_amplitude={best['cl_amplitude']:.3e} "
          f"(clustering ratio {best['clustering_ratio']:.2f})")
    if abs(best["clustering_ratio"] - 1.0) > 0.1:
        print("  -> still not matched; widen the scan")

    (a.out_dir / "glass_calibration.json").write_text(
        json.dumps({"data": {"nside": a.nside, "n_gal": a.n_gal, "fsky": a.fsky,
                             "nbar": nbar_data, "sigma_hat": a.sigma_hat_data,
                             "sigma_clus": sclus_data},
                    "mocks": rows}, indent=2))
    print(f"\n-> {a.out_dir/'glass_calibration.json'}")


if __name__ == "__main__":
    main()
