"""Re-join the break-even quantities on campaign F (20260907c) and recompute statistics on
both datasets.  Uses the benchmark's own template loader and metrics; writes only here."""
import os
from pathlib import Path as _P
BENCH_ROOT = _P(__file__).resolve().parents[2]
SYS_MAPPING_ROOT = _P(os.environ.get("SYS_MAPPING_ROOT", _P.home() / "software" / "sys_mapping"))
EXPORT = BENCH_ROOT / "results" / "docs_exports"
import csv, glob, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(BENCH_ROOT / "characterisation"))
import analyze_existing_simulations as ae  # noqa: E402

OUT = EXPORT / "breakeven"
OUT.mkdir(parents=True, exist_ok=True)
CAMP = BENCH_ROOT / "results/campaign/20260907c/simulations/20260907c"
METHODS = ["OLS", "ISD-1", "ISD-3", "ElasticNet", "MCMC-add", "MCMC-comb"]


def join(files):
    rows, cache = [], {}
    for summary in files:
        p = Path(summary)
        nside = int(p.parent.name.replace("nside", ""))
        seed = int(p.parent.parent.name.replace("seed", ""))
        entries = json.loads(p.read_text())
        key = (nside, tuple(entries[0]["template_names"]))
        if key not in cache:
            cache[key] = ae.load_named_templates(list(key[1]), nside)
        T = cache[key]
        for e in entries:
            cfg = e["config"]
            a_true = np.asarray(cfg["a_true"], float)
            b_true = np.asarray(cfg["b_true"], float)
            w_true = np.asarray(e["w_true"], float)
            A_add, A_mult = ae.field_rms(a_true, T), ae.field_rms(b_true, T)
            bias_cont = ae.curve_bias(e["w_contaminated"], w_true)
            for m in METHODS:
                if f"params_{m}" not in e:
                    continue
                a_hat = np.asarray(e[f"params_{m}"].get("a_hat") or np.zeros_like(a_true), float)
                b_hat = np.asarray(e[f"params_{m}"].get("b_hat") or np.zeros_like(b_true), float)
                dA_add, dA_mult = ae.field_rms(a_hat - a_true, T), ae.field_rms(b_hat - b_true, T)
                bias_rec = ae.curve_bias(e[f"w_recovered_{m}"], w_true)
                rows.append(dict(
                    seed=seed, nside=nside, source=e["source"], level=cfg["level"],
                    scenario=cfg["scenario"], method=m, A_add=A_add, A_mult=A_mult,
                    dA_add=dA_add, dA_mult=dA_mult,
                    A_tot=float(np.hypot(A_add, A_mult)), dA_tot=float(np.hypot(dA_add, dA_mult)),
                    bias_contaminated=bias_cont, bias_recovered=bias_rec,
                    improvement=bias_cont / bias_rec if bias_rec > 0 else np.inf,
                    rms_a_true=float(np.sqrt(np.mean(a_true**2))),
                    rms_a_hat=float(np.sqrt(np.mean(a_hat**2))),
                ))
    return rows


if __name__ == "__main__":
    files = sorted(glob.glob(str(CAMP / "seed*" / "nside*" / "results_summary.json")))
    print(len(files), "simulation cells")
    rows = join(files)
    with (OUT / "campaignF_20260907c_joined.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(len(rows), "rows")
