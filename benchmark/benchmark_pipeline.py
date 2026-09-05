#!/usr/bin/env python3
"""Benchmark every stage of the sys_mapping pipeline and record it reproducibly.

Writes two files:

* ``results/benchmarks.csv``  — one row per (group, operation, nside, n_sys)
  with ``n_pix``, ``n_repeat``, ``median_s`` and ``mad_s``.
* ``results/machine.json``    — CPU, cores, RAM, library versions, timestamp and
  load average, so every number is attributable to a machine and a date.

These are consumed by the ``sys_mapping_paper`` repository (§7) and by
``docs/results_benchmark.rst`` in ``sys_mapping``.

The undated benchmark table in ``Claude.md`` and the inline microsecond figures in
``docs/methods.rst`` are superseded by this harness.

Data generation reuses ``test_timing.py`` in this directory (``_make_data``,
``_generate_templates``) so the benchmark and the timing tests measure identical
inputs.

Usage
-----
    python benchmark/benchmark_pipeline.py                # full grid
    python benchmark/benchmark_pipeline.py --quick        # fast subset
    python benchmark/benchmark_pipeline.py --groups micro # one group only
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                       # the sys_mapping_benchmark repository
sys.path.insert(0, str(HERE))            # test_timing.py sits alongside this file

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import healpy as hp                                    # noqa: E402
import jax                                             # noqa: E402
import sys_mapping as sm                               # noqa: E402
from sys_mapping.contamination import apply_contamination, invert_contamination, pack_params  # noqa: E402
from sys_mapping.correction import debias_params, rotate_templates  # noqa: E402

from test_timing import (                                   # noqa: E402
    _make_data as _make_data_le10,
    _generate_templates as _generate_templates_le10,
    _galactic_mask,
    N_MEAN, SIGMA_G, AMPLITUDE,
)


def _generate_templates(nside: int, n_templates: int, seed: int = 0) -> np.ndarray:
    """Template generator without the 10-map ceiling of ``tests/test_timing.py``.

    Delegates to the test helper for n <= 10 so those configurations are measured on
    byte-identical inputs; beyond that it continues the same construction (5 spectral
    families cycled against fresh seeds), which is needed because the real LS10 basis
    has 11 templates.
    """
    if n_templates <= 10:
        return _generate_templates_le10(nside, n_templates, seed=seed)
    return np.stack([
        sm.generate_systematic_map(nside, i % 5, seed=seed + 7 * (i // 5))
        for i in range(n_templates)
    ])


def _make_data(nside: int, n_templates: int, seed: int = 42):
    """(delta_g, delta_t) for the benchmark; mirrors ``test_timing._make_data``."""
    if n_templates <= 10:
        return _make_data_le10(nside, n_templates, seed=seed)

    templates = _generate_templates(nside, n_templates, seed=0)
    footprint = _galactic_mask(nside)
    rng = np.random.default_rng(seed)
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

    lam = np.maximum(N_MEAN * (1.0 + delta_obs), 0.0)
    gal = rng.poisson(lam * footprint.astype(float))
    rnd = rng.poisson(N_MEAN * 8 * footprint.astype(float))
    delta_g, good = sm.compute_overdensity(gal, rnd)
    return delta_g, sm.assign_template_values(templates, good)

OUT_DIR = REPO / "results"


# ── timing core ────────────────────────────────────────────────────────────

def _time(fn, n_repeat: int, warmup: int = 1) -> tuple[float, float]:
    """Return (median, MAD) seconds over ``n_repeat`` calls, warm-up excluded.

    Median/MAD rather than mean/std: JIT stragglers and scheduler noise are
    one-sided, so the mean overstates the cost of a warm call.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    arr = np.asarray(samples)
    med = float(np.median(arr))
    return med, float(np.median(np.abs(arr - med)))


class Recorder:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, group, operation, nside, n_sys, n_pix, fn, n_repeat, note=""):
        try:
            med, mad = _time(fn, n_repeat)
        except Exception as exc:                        # keep the grid going
            print(f"    ! {operation}: {type(exc).__name__}: {exc}")
            # Record the failure rather than dropping the row.  A silently
            # absent method makes benchmarks.csv indistinguishable from one
            # where the method was never requested.
            self.rows.append({
                "group": group, "operation": operation, "nside": nside,
                "n_sys": n_sys, "n_pix": n_pix, "n_repeat": n_repeat,
                "median_s": "", "mad_s": "",
                "note": (note + "; " if note else "") + f"FAILED: {type(exc).__name__}",
            })
            return
        self.rows.append({
            "group": group, "operation": operation, "nside": nside,
            "n_sys": n_sys, "n_pix": n_pix, "n_repeat": n_repeat,
            "median_s": f"{med:.9g}", "mad_s": f"{mad:.9g}", "note": note,
        })
        print(f"    {operation:<42s} {med*1e3:10.3f} ms  (± {mad*1e3:.3f})")


# ── group 1: per-function micro-benchmarks ─────────────────────────────────

def _good_pixels(nside: int, n_sys: int, seed: int = 42) -> np.ndarray:
    """Footprint mask matching ``tests/test_timing._make_data`` at this nside.

    ``_make_data`` returns only ``delta_g`` / ``delta_t``; the bootstrap and
    jackknife estimators also need the boolean mask, so rebuild it the same way.
    """
    rng = np.random.default_rng(seed)
    footprint = _galactic_mask(nside)
    n_gal = rng.poisson(50.0 * footprint.astype(float))
    n_rnd = rng.poisson(50.0 * 8 * footprint.astype(float))
    _, good = sm.compute_overdensity(n_gal, n_rnd)
    return good



def bench_micro(rec: Recorder, nside: int, n_sys: int, n_repeat: int) -> None:
    print(f"  [micro] nside={nside} n_sys={n_sys}")
    delta_g, delta_t = _make_data(nside, n_sys)
    n_pix = delta_g.size
    a = np.full(n_sys, 0.05)
    b = np.full(n_sys, 0.02)
    add = lambda op, fn, r=n_repeat: rec.add("micro", op, nside, n_sys, n_pix, fn, r)

    dg_j, dt_j = jax.numpy.asarray(delta_g), jax.numpy.asarray(delta_t)
    a_j, b_j = jax.numpy.asarray(a), jax.numpy.asarray(b)

    add("apply_contamination", lambda: apply_contamination(dg_j, dt_j, a_j, b_j).block_until_ready())
    add("invert_contamination", lambda: invert_contamination(dg_j, dt_j, a_j, b_j).block_until_ready())

    for skew in (False, True):
        ll = sm.make_log_likelihood(n_sys, "combined", use_skewed=skew)
        theta = jax.numpy.asarray(
            pack_params(a, b, 0.3, gamma=0.5 if skew else None, model="combined")
        )
        ll(theta, dg_j, dt_j).block_until_ready()       # compile outside the timer
        add(f"log_likelihood[{'skew' if skew else 'gauss'}]",
            lambda ll=ll, th=theta: ll(th, dg_j, dt_j).block_until_ready())

    add("rotate_templates", lambda: rotate_templates(delta_t))
    var = np.full(n_sys, 1e-4)
    add("debias_params", lambda: debias_params(a, b, var, var))

    tcorr = np.abs(np.random.default_rng(0).standard_normal((n_sys, 15))) * 1e-3
    a_sq, b_sq = debias_params(a, b, var, var)
    tc_j, asq_j, bsq_j = (jax.numpy.asarray(x) for x in (tcorr, a_sq, b_sq))
    w_obs_j = jax.numpy.asarray(np.linspace(1e-3, 1e-4, 15))
    add("compute_two_point_correction",
        lambda: sm.compute_two_point_correction(w_obs_j, asq_j, bsq_j, tc_j).block_until_ready())

    add("compute_covariance_matrix", lambda: sm.compute_covariance_matrix(delta_t))

    th_null = pack_params(a, None, 0.3, model="additive")
    th_alt = pack_params(a, b, 0.3, model="combined")
    add("likelihood_ratio_test",
        lambda: sm.likelihood_ratio_test(delta_g, delta_t, th_null, th_alt,
                                         "additive", "combined"))

    # bootstrap / jackknife need the footprint mask, so rebuild delta_g on a
    # mask we hold on to (same construction as tests/test_timing._make_data).
    good_pix = _good_pixels(nside, n_sys)
    dg_b = delta_g[: int(good_pix.sum())]
    dt_b = delta_t[:, : int(good_pix.sum())]
    ols = lambda dg, dt: np.linalg.lstsq(dt.T, dg, rcond=None)[0]

    add("block_bootstrap_variance[B=20,K=8]",
        lambda: sm.block_bootstrap_variance(dg_b, dt_b, good_pix, nside, ols,
                                            n_bootstrap=20, n_patches=8),
        max(3, n_repeat // 3))
    add("jackknife_covariance[K=8]",
        lambda: sm.jackknife_covariance(dg_b, dt_b, good_pix, nside, ols, n_patches=8),
        max(3, n_repeat // 3))

    full = np.zeros(hp.nside2npix(nside))
    mask = _galactic_mask(nside)
    full[mask] = np.random.default_rng(1).standard_normal(int(mask.sum())) * 0.1
    # use_pixel_weights=False: healpy has no published weight file for small NSIDE and
    # retries a 404 download on *every* call, which would time the network rather than
    # the transform.  The ring-weight path is what matters for the comparison anyway.
    add("measure_pseudo_cl", lambda: sm.measure_pseudo_cl(full, mask,
                                                          use_pixel_weights=False),
        max(3, n_repeat // 3))


# ── group 2: HEALPix map utilities ─────────────────────────────────────────

def bench_maps(rec: Recorder, nside: int, n_sys: int, n_repeat: int) -> None:
    print(f"  [maps] nside={nside} n_sys={n_sys}")
    n_pix = hp.nside2npix(nside)
    rng = np.random.default_rng(0)
    n_gal = 100_000
    ra = rng.uniform(0, 360, n_gal)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n_gal)))
    add = lambda op, fn, r=n_repeat: rec.add("maps", op, nside, n_sys, n_pix, fn, r)

    add("systematic_power_spectrum", lambda: sm.systematic_power_spectrum(nside, 2))
    add("generate_systematic_map", lambda: sm.generate_systematic_map(nside, 2, seed=0),
        max(3, n_repeat // 5))
    add("generate_systematic_maps",
        lambda: sm.generate_systematic_maps(nside, families=[0, 1, 2, 3, 4], seed=0),
        max(3, n_repeat // 10))
    add("pixelize_catalog[1e5 gal]", lambda: sm.pixelize_catalog(ra, dec, nside))

    gal = sm.pixelize_catalog(ra, dec, nside)
    rnd = sm.pixelize_catalog(rng.uniform(0, 360, n_gal * 8),
                              np.degrees(np.arcsin(rng.uniform(-1, 1, n_gal * 8))), nside)
    add("compute_overdensity", lambda: sm.compute_overdensity(gal, rnd))

    _, good = sm.compute_overdensity(gal, rnd)
    templates = _generate_templates(nside, n_sys)
    add("assign_template_values", lambda: sm.assign_template_values(templates, good))


# ── group 3: Stage-1 pre-selection ─────────────────────────────────────────

def bench_stage1(rec: Recorder, nside: int, n_sys: int, n_repeat: int) -> None:
    print(f"  [stage1] nside={nside} n_sys={n_sys}")
    delta_g, delta_t = _make_data(nside, n_sys)
    n_pix = delta_g.size
    for meth in ("template", "data", "peak", "isd"):
        if meth == "peak":
            continue                                    # needs a full-sky map, not a subset
        rec.add("stage1", f"snr_template_ranking[{meth}]", nside, n_sys, n_pix,
                lambda m=meth: sm.snr_template_ranking(delta_g, delta_t, method=m),
                n_repeat)
    rec.add("stage1", "snr_template_ranking[isd,poly3]", nside, n_sys, n_pix,
            lambda: sm.snr_template_ranking(delta_g, delta_t, method="isd", poly_order=3),
            n_repeat)


# ── group 4: Stage-2 decontamination methods ───────────────────────────────

_STAGE2 = ["OLS", "ISD-1", "ElasticNet", "ISD-3", "MCMC-add", "MCMC-comb"]


def bench_methods(rec: Recorder, nside: int, n_sys: int, n_repeat: int,
                  methods: list[str], nuts: int, n_repeat_mcmc: int = 3) -> None:
    print(f"  [stage2] nside={nside} n_sys={n_sys}")
    delta_g, delta_t = _make_data(nside, n_sys)
    n_pix = delta_g.size
    for meth in methods:
        # MCMC used to be pinned at one draw, which is why every MCMC row in the
        # published table has mad = 0: not a stable measurement, no measurement
        # of spread at all.  Two draws is the minimum that yields a MAD.
        reps = (max(2, n_repeat_mcmc) if meth.startswith("MCMC")
                else max(2, n_repeat // 4))
        rec.add("stage2", meth, nside, n_sys, n_pix,
                lambda m=meth: sm.run_decontamination(
                    m, delta_g, delta_t, sampler="auto",
                    nuts_n_warmup=nuts, nuts_n_samples=nuts, n_chains=2, seed=42),
                reps, note=f"nuts={nuts}" if meth.startswith("MCMC") else "")


# ── provenance ─────────────────────────────────────────────────────────────

def machine_info() -> dict:
    def _cpu() -> str:
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
        return platform.processor() or "unknown"

    def _mem_gb() -> float:
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 1024**2, 1)
        except OSError:
            pass
        return float("nan")

    def _ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return "n/a"

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = "unknown"

    import os
    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:
        load1 = load5 = load15 = float("nan")

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Recorded so a contended run is identifiable after the fact: timings taken
        # while the machine is loaded are not comparable with quiet-machine ones.
        "loadavg_at_start": [round(load1, 2), round(load5, 2), round(load15, 2)],
        "git_commit": commit,
        "cpu": _cpu(),
        "n_cores_logical": os.cpu_count(),
        "mem_total_gb": _mem_gb(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "jax_x64": bool(jax.config.jax_enable_x64),
        "versions": {m: _ver(m) for m in
                     ("numpy", "scipy", "jax", "healpy", "blackjax", "emcee",
                      "sklearn", "treecorr")},
        "env": {k: os.environ.get(k, "") for k in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "XLA_FLAGS")},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="small grid, few repeats")
    p.add_argument("--groups", nargs="+",
                   default=["micro", "maps", "stage1", "stage2"],
                   choices=["micro", "maps", "stage1", "stage2"])
    p.add_argument("--nsides", type=int, nargs="+", default=None)
    p.add_argument("--n-sys", type=int, nargs="+", default=None)
    p.add_argument("--n-repeat", type=int, default=None)
    p.add_argument("--nuts", type=int, default=None, help="NUTS warmup=samples")
    p.add_argument("--n-repeat-mcmc", type=int, default=3,
                   help="repeats for the MCMC methods (min 2, so the row has a MAD)")
    p.add_argument("--methods", nargs="+", default=_STAGE2)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    nsides = args.nsides or ([16, 32] if args.quick else [16, 32, 64])
    n_sys_grid = args.n_sys or ([5] if args.quick else [5, 11])
    n_repeat = args.n_repeat or (5 if args.quick else 15)
    nuts = args.nuts or (100 if args.quick else 400)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    info = machine_info()
    print(f"sys_mapping benchmark — {info['cpu']} · {info['n_cores_logical']} cores · "
          f"jax {info['versions']['jax']} ({info['jax_backend']}, x64={info['jax_x64']})")
    print(f"grid: nside={nsides} n_sys={n_sys_grid} repeats={n_repeat} nuts={nuts}")
    _load1 = info["loadavg_at_start"][0]
    _cores = info["n_cores_logical"] or 1
    if _load1 > 0.5 * _cores:
        print(f"\n  WARNING: 1-min load average is {_load1:.1f} on {_cores} cores.")
        print("  Timings taken on a contended machine are not comparable with quiet-machine")
        print("  numbers. Re-run when the machine is idle before quoting these.\n")
    else:
        print()

    rec = Recorder()
    for nside in nsides:
        for n_sys in n_sys_grid:
            if "micro" in args.groups:
                bench_micro(rec, nside, n_sys, n_repeat)
            if "maps" in args.groups:
                bench_maps(rec, nside, n_sys, n_repeat)
            if "stage1" in args.groups:
                bench_stage1(rec, nside, n_sys, n_repeat)
            if "stage2" in args.groups:
                bench_methods(rec, nside, n_sys, n_repeat, args.methods, nuts,
                              args.n_repeat_mcmc)

    csv_path = args.out_dir / "benchmarks.csv"
    fields = ["group", "operation", "nside", "n_sys", "n_pix", "n_repeat",
              "median_s", "mad_s", "note"]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rec.rows)
    (args.out_dir / "machine.json").write_text(json.dumps(info, indent=2))

    print(f"\n{len(rec.rows)} rows → {csv_path}")
    print(f"provenance     → {args.out_dir / 'machine.json'}")


if __name__ == "__main__":
    main()
