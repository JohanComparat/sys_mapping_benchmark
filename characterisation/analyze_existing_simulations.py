#!/usr/bin/env python3
"""Maximise what the *existing* simulation outputs can say about when correcting helps.

Two large campaigns already exist on disk and were never joined:

* ``results/detectability_sweep_*.csv``     -- field-level detection vs amplitude
  (14 400 rows for LS10), columns ``amp``, ``field_snr``, ``a_field_corr``.
* ``data/simulations/nside*/results_summary.json`` -- w(theta) correction quality,
  18 configurations with ground-truth ``a_true``/``b_true``, recovered ``a_hat`` per
  method, and the full ``w_true`` / ``w_contaminated`` / ``w_recovered_*`` curves.

The first says when a systematic is *detectable*; the second says whether correcting
*helped*.  This script joins them through a mechanistic quantity computable from what
is already stored.

The mechanism
-------------
Correcting removes a bias but injects the noise of the fitted amplitudes.  Writing the
true contaminating field and the residual (mis-fit) field as

    A     = rms( sum_i a_true_i t_i )       -- what the correction must remove
    dA    = rms( sum_i (a_hat_i - a_true_i) t_i )   -- what the correction adds back

and noting that a w(theta) contamination enters as the *square* of a field amplitude,
the correction is expected to help when A > dA, i.e. when the recovered field is
closer to the truth than to zero.  ``A/dA`` is therefore the natural predictor of the
improvement factor, and ``A/dA = 1`` is the predicted break-even point.

Both A and dA are computable from stored quantities: ``a_true`` and ``a_hat`` are in
the JSON, and the templates are named there and standardised the same way the
pipeline standardises them.

Outputs
-------
    results/analysis_existing/existing_data_joined.csv
    results/analysis_existing/fig_improvement_vs_AdA.png
    results/analysis_existing/fig_sweep_detection.png
    results/analysis_existing/summary.json
"""

from __future__ import annotations

import glob
import json
import os
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import healpy as hp                                       # noqa: E402
import matplotlib                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                           # noqa: E402
from astropy.io import fits                               # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "analysis_existing"
SYS_MAPPING = Path(
    os.environ.get("SYS_MAPPING_ROOT", Path.home() / "software" / "sys_mapping")
).expanduser()
SYST = Path.home() / "data" / "legacysurvey" / "dr10" / "systematics"
METHODS = ["OLS", "ISD-1", "ElasticNet", "MCMC-add", "MCMC-comb"]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": "--", "axes.spines.top": False, "axes.spines.right": False,
})


# ── template loading ───────────────────────────────────────────────────────

_ZFILL = {"LS10": 4, "GAIA": 5}


def load_named_templates(names: list[str], nside: int) -> np.ndarray:
    """Load templates by the short names stored in the results JSON.

    Standardised over valid pixels exactly as the pipeline does, so the amplitudes
    in the JSON multiply the same maps they were fitted against.
    """
    out = []
    for name in names:
        src = name.split("_")[0]
        path = SYST / f"{nside:04d}" / f"{name}_NSIDE_{nside:0{_ZFILL.get(src, 4)}d}.fits"
        if not path.exists():
            raise FileNotFoundError(path)
        with fits.open(path) as hdul:
            for hdu in hdul[1:]:
                if hasattr(hdu, "columns"):
                    # HEALPix FITS stores the map in 1024-element rows, so the raw
                    # column is (npix/1024, 1024); ravel it back to a flat map.
                    m = np.asarray(hdu.data[hdu.columns.names[0]], dtype=float).ravel()
                    break
        npix = hp.nside2npix(nside)
        if m.size != npix:
            m = hp.ud_grade(m, nside)
        m[~np.isfinite(m)] = hp.UNSEEN
        out.append(m)
    T = np.vstack(out)
    valid = np.all(T != hp.UNSEEN, axis=0) & np.all(np.isfinite(T), axis=0)
    Tv = T[:, valid]
    Tv = (Tv - Tv.mean(1, keepdims=True)) / Tv.std(1, keepdims=True)
    return Tv


# ── metrics ────────────────────────────────────────────────────────────────

def field_rms(coeffs: np.ndarray, T: np.ndarray) -> float:
    """rms over pixels of sum_i c_i t_i -- the field amplitude the coefficients imply."""
    return float(np.sqrt(np.mean((coeffs @ T) ** 2)))


def curve_bias(w: np.ndarray, w_true: np.ndarray) -> float:
    """Fractional L2 distance between two w(theta) curves."""
    w, w_true = np.asarray(w, float), np.asarray(w_true, float)
    ok = np.isfinite(w) & np.isfinite(w_true)
    if ok.sum() == 0:
        return float("nan")
    denom = np.linalg.norm(w_true[ok])
    return float(np.linalg.norm(w[ok] - w_true[ok]) / denom) if denom > 0 else float("nan")


# ── part 1: the join ───────────────────────────────────────────────────────

def analyse_simulations() -> list[dict]:
    rows: list[dict] = []
    pattern = SYS_MAPPING / "data" / "simulations" / "nside*" / "results_summary.json"
    for summary in sorted(glob.glob(str(pattern))):
        nside = int(Path(summary).parent.name.replace("nside", ""))
        entries = json.loads(Path(summary).read_text())
        T = load_named_templates(entries[0]["template_names"], nside)
        print(f"  {Path(summary).parent.name}: {len(entries)} configs, "
              f"{T.shape[0]} templates x {T.shape[1]} px")

        for e in entries:
            cfg = e["config"]
            a_true = np.asarray(cfg["a_true"], float)
            b_true = np.asarray(cfg["b_true"], float)
            w_true = np.asarray(e["w_true"], float)
            w_cont = np.asarray(e["w_contaminated"], float)

            # the additive field the fit is able to see, and the multiplicative one it is not
            A_add = field_rms(a_true, T)
            A_mult = field_rms(b_true, T)
            bias_cont = curve_bias(w_cont, w_true)

            for meth in METHODS:
                key_w, key_p = f"w_recovered_{meth}", f"params_{meth}"
                if key_w not in e or key_p not in e:
                    continue
                a_hat = np.asarray(e[key_p].get("a_hat") or np.zeros_like(a_true), float)
                b_hat = np.asarray(e[key_p].get("b_hat") or np.zeros_like(b_true), float)
                if a_hat.shape != a_true.shape:
                    continue

                # residual mis-fit field: what the correction adds back as noise
                dA_add = field_rms(a_hat - a_true, T)
                dA_mult = field_rms(b_hat - b_true, T)
                # total signal the method could remove vs total residual it leaves
                A_tot = float(np.hypot(A_add, A_mult))
                dA_tot = float(np.hypot(dA_add, dA_mult))

                bias_rec = curve_bias(e[key_w], w_true)
                rows.append({
                    "nside": nside, "source": e["source"], "level": cfg["level"],
                    "scenario": cfg["scenario"], "method": meth,
                    "n_gal": e.get("n_gal"), "n_rand": e.get("n_rand"),
                    "A_add": A_add, "A_mult": A_mult, "A_tot": A_tot,
                    "dA_add": dA_add, "dA_mult": dA_mult, "dA_tot": dA_tot,
                    "A_over_dA": A_tot / dA_tot if dA_tot > 0 else np.inf,
                    "bias_contaminated": bias_cont, "bias_recovered": bias_rec,
                    "improvement": bias_cont / bias_rec if bias_rec > 0 else np.inf,
                })
    return rows


# ── part 2: the detectability sweep ────────────────────────────────────────

def analyse_sweep() -> dict:
    import csv
    from collections import defaultdict
    path = SYS_MAPPING / "results" / "detectability_sweep_ls10.csv"
    if not path.exists():
        return {}
    agg = defaultdict(list)
    with path.open() as fh:
        for r in csv.DictReader(fh):
            try:
                agg[(int(r["nside"]), float(r["amp"]), r["method"])].append(
                    (float(r["field_snr"]), float(r["a_field_corr"])))
            except (ValueError, KeyError):
                continue
    out = {}
    for (ns, amp, meth), vals in agg.items():
        snr = np.array([v[0] for v in vals]); corr = np.array([v[1] for v in vals])
        out[f"{ns}|{amp}|{meth}"] = {
            "nside": ns, "amp": amp, "method": meth, "n_sims": len(vals),
            "field_snr_median": float(np.median(snr)),
            "a_field_corr_median": float(np.median(corr)),
            "frac_corr_above_0p5": float(np.mean(corr > 0.5)),
        }
    return out


# ── plots ──────────────────────────────────────────────────────────────────

def plot_improvement(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, src in zip(axes, ["uchuu", "glass"]):
        sub = [r for r in rows if r["source"] == src and np.isfinite(r["A_over_dA"])
               and np.isfinite(r["improvement"])]
        if not sub:
            continue
        for meth in METHODS:
            m = [r for r in sub if r["method"] == meth]
            if m:
                ax.scatter([r["A_over_dA"] for r in m], [r["improvement"] for r in m],
                           s=26, alpha=0.8, label=meth)
        ax.axhline(1.0, color="k", lw=1, ls="--")
        ax.axvline(1.0, color="crimson", lw=1.2, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$A/\Delta A$  (signal / mis-fit residual)")
        ax.set_title(f"{src}   (n={len(sub)})")
        ax.text(0.03, 0.95, "correction helps", transform=ax.transAxes,
                va="top", fontsize=7, color="green")
        ax.text(0.03, 0.05, "correction hurts", transform=ax.transAxes,
                va="bottom", fontsize=7, color="crimson")
    axes[0].set_ylabel(r"improvement  $\mathcal{B}_{\rm cont}/\mathcal{B}_{\rm corr}$")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(r"Correcting helps only once the fitted field beats its own mis-fit "
                 r"($A/\Delta A \gtrsim 1$)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_improvement_vs_AdA.png")
    plt.close(fig)


def plot_sweep(sweep: dict) -> None:
    if not sweep:
        return
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    from collections import defaultdict
    series = defaultdict(list)
    for v in sweep.values():
        series[(v["nside"], v["method"])].append((v["amp"], v["a_field_corr_median"]))
    for (ns, meth), pts in sorted(series.items()):
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=3, lw=1,
                label=f"nside {ns}, {meth}")
    ax.axhline(0.5, color="crimson", ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("injected per-template amplitude")
    ax.set_ylabel(r"median field recovery  ${\rm corr}(\hat f, f)$")
    ax.set_title("Existing detectability sweep: field recovery vs amplitude")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sweep_detection.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Joining existing simulation outputs ...")
    print(f"  sys_mapping root: {SYS_MAPPING}")
    if not SYS_MAPPING.is_dir():
        print(f"  ! {SYS_MAPPING} not found — set SYS_MAPPING_ROOT to a sys_mapping checkout")
        return
    rows = analyse_simulations()
    sweep = analyse_sweep()

    import csv
    if rows:
        with (OUT / "existing_data_joined.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        plot_improvement(rows)
    plot_sweep(sweep)

    # does A/dA actually predict whether correcting helps?
    fin = [r for r in rows if np.isfinite(r["A_over_dA"]) and np.isfinite(r["improvement"])]
    summary: dict = {"n_rows": len(rows), "n_finite": len(fin)}
    if fin:
        x = np.log10([r["A_over_dA"] for r in fin])
        y = np.log10([r["improvement"] for r in fin])
        summary["spearman_logAdA_vs_logimprovement"] = float(
            np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])
        helped = np.array([r["improvement"] > 1 for r in fin])
        above = np.array([r["A_over_dA"] > 1 for r in fin])
        summary["contingency"] = {
            "A>dA and helped": int((above & helped).sum()),
            "A>dA and hurt": int((above & ~helped).sum()),
            "A<dA and helped": int((~above & helped).sum()),
            "A<dA and hurt": int((~above & ~helped).sum()),
        }
        for src in ("uchuu", "glass"):
            s = [r for r in fin if r["source"] == src]
            if s:
                summary[f"{src}_frac_helped"] = float(np.mean([r["improvement"] > 1 for r in s]))
                summary[f"{src}_median_A_over_dA"] = float(np.median([r["A_over_dA"] for r in s]))

        # Split by scenario. The additive-only rows are the clean discriminator: the
        # `multiplicative` fit/inject mismatch (b=a while retaining a) cannot apply
        # there, so if additive cells still degrade the cause must be the threshold.
        summary["by_scenario"] = {}
        for sc in ("additive", "multiplicative", "combined"):
            s = [r for r in fin if r["scenario"] == sc]
            if not s:
                continue
            helped = np.array([r["improvement"] > 1 for r in s])
            above = np.array([r["A_over_dA"] > 1 for r in s])
            summary["by_scenario"][sc] = {
                "n": len(s),
                "frac_helped": float(helped.mean()),
                "median_A_over_dA": float(np.median([r["A_over_dA"] for r in s])),
                "rule_accuracy": float((helped == above).mean()),
            }
        summary["by_scenario_source_level"] = {}
        for sc in ("additive", "multiplicative", "combined"):
            for src in ("uchuu", "glass"):
                for lv in ("low", "medium", "high"):
                    s = [r for r in fin if r["scenario"] == sc and r["source"] == src
                         and r["level"] == lv]
                    if not s:
                        continue
                    summary["by_scenario_source_level"][f"{sc}|{src}|{lv}"] = {
                        "n": len(s),
                        "frac_helped": float(np.mean([r["improvement"] > 1 for r in s])),
                        "median_A_over_dA": float(np.median([r["A_over_dA"] for r in s])),
                    }
    summary["sweep"] = sweep
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{len(rows)} method-configuration rows -> {OUT/'existing_data_joined.csv'}")
    if "contingency" in summary:
        print("\nDoes A/dA > 1 predict that correcting helps?")
        for k, v in summary["contingency"].items():
            print(f"   {k:<22} {v}")
        print(f"\n  rank corr(log A/dA, log improvement) = "
              f"{summary['spearman_logAdA_vs_logimprovement']:+.3f}")
        for src in ("uchuu", "glass"):
            if f"{src}_frac_helped" in summary:
                print(f"  {src:<6} helped in {summary[f'{src}_frac_helped']*100:5.1f}% of cells, "
                      f"median A/dA = {summary[f'{src}_median_A_over_dA']:.2f}")
        print("\n  by scenario (additive rows cannot be affected by the b=a mismatch):")
        for sc, v in summary.get("by_scenario", {}).items():
            print(f"    {sc:<15} n={v['n']:<4} helped={v['frac_helped']*100:5.1f}%  "
                  f"median A/dA={v['median_A_over_dA']:.2f}  "
                  f"rule acc={v['rule_accuracy']*100:.1f}%")


if __name__ == "__main__":
    main()
