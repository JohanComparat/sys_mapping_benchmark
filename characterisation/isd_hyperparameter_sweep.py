#!/usr/bin/env python3
"""Sweep the ISD hyper-parameters, scoring by the metric that can see them.

Why this is not ``run_simulation_tests.py --isd-kwargs`` in a loop
-----------------------------------------------------------------
It was, and that was a mistake worth recording.  A hyper-parameter setting
changes one thing: the fitted weight.  It does not change the mock, the
calibration null, the injected contamination, or the truth and contaminated
w(theta) curves.  Re-invoking the full runner per setting recomputes all of them,
and the w(theta) measurement dominates: on the first attempt the fixed cost per
setting was ~14 min against 0.009 s of actual ISD payload, a ratio of 10^5, and
the jobs died at the 4 h wall having covered 17 of 47 settings.

This script pays each fixed cost once:

    mock -> calibration -> per-config injection -> [settings loop] -> scores

and scores by the residual ``Delta chi^2`` the corrected field still shows
against the contaminating template, probed at a *common* cubic order for every
setting.  That last point is not a detail: ``ISDResult.significance`` is each
run's own significance at its own ``poly_order``, so comparing ISD-1's residual
to ISD-3's through it compares a linear probe with a cubic one.  ``a_hat`` is
worse still -- it is the linear projection of the fitted curve and is blind to
curvature by construction (docs/methods.rst, "Why ISD-3 is worth having").
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

import sys_mapping as sm
from sys_mapping.contamination import evaluate_response
from sys_mapping.diagnostics import isd_marginal_fit
from sys_mapping.simulation import (
    load_systematic_maps,
    make_response_grid,
)

PROBE_ORDER = 3  # common probe for every setting; see the module docstring


def build_settings() -> list[dict]:
    """The grid: a full cross where the axes interact, one-at-a-time elsewhere."""
    settings = [
        {"isd_poly_order": d, "isd_n_bins": nb, "isd_binning": b}
        for d, nb, b in itertools.product((1, 2, 3, 4, 5), (5, 10, 20, 40),
                                          ("quantile", "width"))
    ]
    settings += [{"isd_threshold": t} for t in (1.0, 3.0, 5.0)]
    settings += [{"isd_max_reuse": m} for m in (1, 10)]
    settings += [{"isd_bad_pixel_frac": f} for f in (0.001, 0.05)]
    settings += [{"isd_w_max": w} for w in (5.0, 100.0)]
    settings += [{"isd_chi2_68": None}]          # the uncalibrated threshold
    return settings


def curve_error(coeffs_by_template, config, delta_t, n_sys) -> float:
    """RMS difference between the fitted response and the injected one.

    Evaluated over the template's own pixel distribution, so the tails count in
    proportion to how much footprint they actually occupy.
    """
    errs = []
    for i in range(n_sys):
        resp = config.responses[i] if config.responses else None
        truth = (evaluate_response(resp, delta_t[i]) if resp is not None
                 else np.zeros(delta_t.shape[1]))
        fitted = np.zeros(delta_t.shape[1])
        for c in coeffs_by_template.get(i, []):
            powers = np.arange(len(c))
            fitted = fitted + (delta_t[i][:, None] ** powers[None, :]) @ np.asarray(c)
        errs.append(np.sqrt(np.mean((fitted - truth) ** 2)))
    return float(np.sqrt(np.mean(np.square(errs))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nside", type=int, default=32)
    ap.add_argument("--n-glass", type=int, default=500_000)
    ap.add_argument("--n-contaminated", type=int, default=2)
    ap.add_argument("--isd-n-mocks", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--syst-dir", type=str, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    t_start = time.perf_counter()
    templates, names, _ = load_systematic_maps(a.syst_dir, a.nside)
    n_sys = templates.shape[0]

    cat = sm.generate_glass_fullsky_mock(
        a.nside, a.n_glass, np.array([0.05, 0.35]), np.array([float(a.n_glass)]),
        seed=a.seed, rand_factor=2)
    gal = sm.pixelize_catalog(cat["ra"], cat["dec"], a.nside)
    ran = sm.pixelize_catalog(cat["ra_rand"], cat["dec_rand"], a.nside)
    delta_clean, good = sm.compute_overdensity(gal, ran)
    delta_t = sm.assign_template_values(templates, good)
    print(f"  mock + templates: {len(cat['ra']):,} galaxies, {good.sum():,} pixels "
          f"({time.perf_counter() - t_start:.1f}s)")

    t0 = time.perf_counter()
    out = sm.isd_template_significance(
        delta_clean, delta_t, good, a.nside, n_total=0,
        z_edges=np.array([0.0, 1.0]), nz=np.array([float(a.n_glass)]),
        n_total_footprint=len(cat["ra"]), n_mocks=a.isd_n_mocks,
        poly_order=PROBE_ORDER, binning="quantile", seed=a.seed, rand_factor=2)
    chi2_68 = np.percentile(out["delta_chi2_mocks"], 68, axis=0)
    print(f"  chi2_68 calibration ({a.isd_n_mocks} mocks): "
          f"{np.array2string(chi2_68, precision=1)} ({time.perf_counter() - t0:.1f}s)")

    configs = make_response_grid(n_sys=n_sys, seed=a.seed,
                                 n_contaminated=a.n_contaminated)
    settings = build_settings()
    print(f"  {len(configs)} configs x {len(settings)} settings = "
          f"{len(configs) * len(settings)} fits")

    # Inject once per config; a setting cannot change the contamination.
    injected = {}
    for cfg in configs:
        obs = sm.apply_nonlinear_contamination(
            delta_clean, delta_t, cfg.responses or [None] * n_sys)
        uncorr = isd_marginal_fit(obs, delta_t, poly_order=PROBE_ORDER)[0]
        injected[(cfg.level, cfg.shape)] = (cfg, obs, uncorr)

    rows = []
    for si, setting in enumerate(settings):
        kw = dict(setting)
        if "isd_chi2_68" in kw and kw["isd_chi2_68"] is None:
            calibrated = False
        else:
            kw.setdefault("isd_chi2_68", chi2_68)
            calibrated = True
        for (level, shape), (cfg, obs, uncorr) in injected.items():
            t0 = time.perf_counter()
            res = sm.run_decontamination("ISD-3", obs, delta_t, **kw)
            elapsed = time.perf_counter() - t0
            corrected = res["weights"] * (1.0 + obs) - 1.0
            resid = isd_marginal_fit(corrected, delta_t, poly_order=PROBE_ORDER)[0]

            picked = sorted({int(s["template"]) for s in res["isd_steps"]})
            truth = set(cfg.contaminated)
            tp = len(truth & set(picked))
            coeffs = {}
            for s in res["isd_steps"]:
                coeffs.setdefault(int(s["template"]), []).append(s["coeffs"])

            rows.append({
                "setting_index": si, "setting": setting, "calibrated": calibrated,
                "level": level, "shape": shape, "seed": a.seed, "nside": a.nside,
                "poly_order": res["isd_poly_order"],
                "residual_dchi2_contaminated": float(np.sum(resid[list(truth)])),
                "uncorrected_dchi2_contaminated": float(np.sum(uncorr[list(truth)])),
                "residual_dchi2_clean": float(
                    np.sum(np.delete(resid, list(truth)))),
                "curve_error": curve_error(coeffs, cfg, delta_t, n_sys),
                "rms_a_hat": float(np.sqrt(np.mean(np.square(res["a_hat"])))),
                "n_steps": int(res["n_iterations"]),
                "stopped_on": res["isd_stopped_on"],
                "n_floored": int(res["isd_n_floored"]),
                "precision": tp / len(picked) if picked else float("nan"),
                "recall": tp / len(truth) if truth else float("nan"),
                "n_picked": len(picked),
                "elapsed_s": elapsed,
            })
        print(f"   [{si + 1:3d}/{len(settings)}] {json.dumps(setting)}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"seed": a.seed, "nside": a.nside, "n_sys": n_sys,
         "template_names": list(names), "probe_order": PROBE_ORDER,
         "n_contaminated": a.n_contaminated, "chi2_68": chi2_68.tolist(),
         "rows": rows}, indent=1))
    print(f"\n  -> {a.out}  ({len(rows)} rows, "
          f"{time.perf_counter() - t_start:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
