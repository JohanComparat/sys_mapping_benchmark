"""Parametric versus matched GLASS nulls against the LS10 data, per sample and NSIDE."""
import os
from pathlib import Path as _P
BENCH_ROOT = _P(__file__).resolve().parents[2]
SYS_MAPPING_ROOT = _P(os.environ.get("SYS_MAPPING_ROOT", _P.home() / "software" / "sys_mapping"))
EXPORT = BENCH_ROOT / "results" / "docs_exports"
import csv, glob, json
from pathlib import Path
import numpy as np

BENCH = BENCH_ROOT
B = BENCH / "results/campaign/20260904b/glass_calibration/20260904b"
OUT = EXPORT / "export" / "null_spectra_nside32_64.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)
prov = {r["file"]: r["source_tag"] for r in csv.DictReader(open(BENCH / "matched_spectra/PROVENANCE.tsv"), delimiter="\t")}
# the curated "validated" files are byte-identical to these family-M outputs
tag_of = {}
for f in (BENCH / "matched_spectra").glob("*_match.json"):
    for t in ("20260907m", "20260906m"):
        g = BENCH / "results/campaign/20260908i/glass_match" / t / f.name
        if g.exists() and g.read_bytes() == f.read_bytes():
            tag_of[f.name] = t
            break

rows = []
for cell in sorted(B.iterdir()):
    ns = int(cell.name.split("NSIDE")[1])
    if ns > 64:
        continue
    sample = cell.name.rsplit("_NSIDE", 1)[0]
    data = [json.loads(p.read_text())["data"] for p in sorted(cell.glob("seed*/glass_calibration.json"))]
    nbar_d = float(np.median([d["nbar"] for d in data])); sh_d = float(np.median([d["sigma_hat"] for d in data]))
    row = dict(log_mstar_min=float(sample.split("_")[3]), nside=ns, nbar_data=round(nbar_d, 1),
               sigma_hat_data=round(sh_d, 4), sigma_clus_data=round(np.sqrt(sh_d**2 - 1 / nbar_d), 4))
    mf = BENCH / "matched_spectra" / f"{sample}_NSIDE{ns:04d}_match.json"
    if mf.exists():
        m = json.loads(mf.read_text()); v = m["validation"]; h0 = m["history"][0]
        sc_p = np.sqrt(max(h0["sigma_hat_mock"] ** 2 - 1 / h0["nbar_mock"], 0.0))
        sc_m = np.sqrt(max(v["sigma_hat_mock"] ** 2 - 1 / m["nbar_data"], 0.0))
        row.update(sigma_hat_powerlaw=round(h0["sigma_hat_mock"], 4), sigma_clus_powerlaw=round(sc_p, 4),
                   sigma_hat_matched=round(v["sigma_hat_mock"], 4), sigma_clus_matched=round(sc_m, 4),
                   large_scale_ratio=round(v["large_scale_ratio"], 4), tol=v["tol"],
                   ell_min=v["l_range"][0], ell_max=v["l_range"][1],
                   rp_min_mpch=round(v["rp_mpch_covered"][0], 1), rp_max_mpch=round(v["rp_mpch_covered"][1], 1),
                   n_seeds=v["n_seeds"], density_ratio_err=round(v["density_ratio_err"], 4),
                   passed=v["passed"], match_tag=tag_of.get(mf.name, prov.get(mf.name)))
    rows.append(row)

rows.sort(key=lambda r: (r["nside"], r["log_mstar_min"]))
keys = ["log_mstar_min", "nside", "nbar_data", "sigma_hat_data", "sigma_clus_data", "sigma_hat_powerlaw",
        "sigma_clus_powerlaw", "sigma_hat_matched", "sigma_clus_matched", "large_scale_ratio", "tol",
        "ell_min", "ell_max", "rp_min_mpch", "rp_max_mpch", "n_seeds", "density_ratio_err", "passed", "match_tag"]
with OUT.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in keys})
print(OUT.read_text())
m = [r for r in rows if "large_scale_ratio" in r]
rp = np.array([r["sigma_clus_powerlaw"] / r["sigma_clus_data"] for r in m])
rm = np.array([r["sigma_clus_matched"] / r["sigma_clus_data"] for r in m])
hp_ = np.array([r["sigma_hat_powerlaw"] / r["sigma_hat_data"] for r in m])
hm = np.array([r["sigma_hat_matched"] / r["sigma_hat_data"] for r in m])
lsr = np.array([r["large_scale_ratio"] for r in m])
print("n", len(m))
print("powerlaw sigma_clus ratio", rp.min(), np.median(rp), rp.max(), "variance factor", 1/np.median(rp)**2, 1/rp.max()**2, 1/rp.min()**2)
print("matched sigma_clus ratio", rm.min(), np.median(rm), rm.max())
print("powerlaw sigma_hat ratio", hp_.min(), np.median(hp_), hp_.max())
print("matched sigma_hat ratio", hm.min(), np.median(hm), hm.max())
print("LSR", lsr.min(), np.median(lsr), lsr.max(), "dens err max", max(r["density_ratio_err"] for r in m))
print("tags", {r["match_tag"] for r in m})
