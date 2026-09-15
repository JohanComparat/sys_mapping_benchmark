"""Export the campaign-F break-even join and its summary for docs/_static/characterisation."""
import os
from pathlib import Path as _P
BENCH_ROOT = _P(__file__).resolve().parents[2]
SYS_MAPPING_ROOT = _P(os.environ.get("SYS_MAPPING_ROOT", _P.home() / "software" / "sys_mapping"))
EXPORT = BENCH_ROOT / "results" / "docs_exports"
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = EXPORT
OUT = HERE / "export"
OUT.mkdir(parents=True, exist_ok=True)
n = pd.read_csv(HERE / "breakeven" / "campaignF_20260907c_joined.csv")
old = pd.read_csv(BENCH_ROOT / "results/analysis_existing/existing_data_joined.csv")

cols = ["seed", "nside", "scenario", "level", "method", "A_add", "A_mult", "dA_add", "dA_mult",
        "bias_contaminated", "bias_recovered", "improvement"]
n[cols].to_csv(OUT / "breakeven_20260907c_joined.csv", index=False, float_format="%.6g")


def ratio(A, dA):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(dA > 0, A / dA, np.nan)


n["r"] = ratio(n.A_add.to_numpy(), n.dA_add.to_numpy())
n["helped"] = n.improvement > 1
ac = n[n.scenario != "multiplicative"]
mu = n[n.scenario == "multiplicative"]
above = ac.r > 1


def rho(x, y):
    return float(spearmanr(np.log10(x), np.log10(y)).correlation)


summary = {
    "source": "sys_mapping_benchmark results/campaign/20260907c/simulations/20260907c "
              "(family F, dahu, 2026-09-07/08); 134 simulation cells x 9 configurations x 6 methods",
    "definitions": {
        "A": "rms over template-valid pixels of sum_i a_true_i t_i (additive amplitudes)",
        "dA": "rms of sum_i (a_hat_i - a_true_i) t_i",
        "improvement": "B_cont / B_corr, B = ||w - w_true||_2 / ||w_true||_2 over the theta bins",
        "helped": "improvement > 1",
        "rank_correlation": "Spearman between log10(A/dA) and log10(improvement)",
    },
    "n_rows": int(len(n)),
    "cells_per_nside": {int(k): int(v) for k, v in n.groupby("nside").seed.nunique().items()},
    "additive_and_combined": {
        "n": int(len(ac)),
        "contingency": {
            "above_helped": int((above & ac.helped).sum()), "above_hurt": int((above & ~ac.helped).sum()),
            "below_helped": int((~above & ac.helped).sum()), "below_hurt": int((~above & ~ac.helped).sum())},
        "rule_accuracy": float(((ac.r > 1) == ac.helped).mean()),
        "frac_helped": float(ac.helped.mean()),
        "spearman": rho(ac.r, ac.improvement),
        "median_A_over_dA": float(ac.r.median()),
    },
    "multiplicative": {
        "n": int(len(mu)),
        "n_no_additive_correction": int((mu.dA_add == 0).sum()),
        "no_correction_by_method": {k: int(v) for k, v in mu[mu.dA_add == 0].method.value_counts().items()},
        "n_corrected": int((mu.dA_add > 0).sum()),
        "helped_of_corrected": int(mu[mu.dA_add > 0].helped.sum()),
    },
    "by_method": {}, "by_nside": {}, "by_level": {},
    "total_A_definition_all_rows": {
        "note": "A and dA as the root sum square of the additive and multiplicative fields",
        "n": int(len(n)),
        "spearman": rho(np.hypot(n.A_add, n.A_mult) / np.hypot(n.dA_add, n.dA_mult), n.improvement),
    },
    "v0.9.5_grid_180": {
        "note": "sys_mapping_benchmark results/analysis_existing/existing_data_joined.csv; total-A definition",
        "n": int(len(old)),
        "spearman_total_A": rho(old.A_tot / old.dA_tot, old.improvement),
    },
}
for key, grp, col in (("by_method", ac, "method"), ("by_nside", ac, "nside"), ("by_level", ac, "level")):
    for k, g in grp.groupby(col, sort=False):
        summary[key][str(k)] = {"n": int(len(g)), "median_A_over_dA": float(g.r.median()),
                                "frac_helped": float(g.helped.mean()),
                                "rule_accuracy": float(((g.r > 1) == g.helped).mean())}
(OUT / "breakeven_20260907c_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
print("theta bins:", len(json.load(open(
    BENCH_ROOT / "results/campaign/20260907c/simulations/20260907c/seed000/nside0032/results_summary.json"))[0]["theta"]))
