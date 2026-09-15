"""Cross-term share of the two-point correction in the basis the LS10 products use.

For each sample and NSIDE: footprint from the sample's randoms (compute_overdensity rule),
templates standardised on that footprint (checked against the rms recorded in the
product's params.json), xi_ij from the benchmark's pixel estimator on the footprint,
PCA rotation of the footprint basis, and the amplitudes transformed with it (a_rot = R a).
theta grid, debias variance and w_obs as in run_crossterm_bias.py.
"""
import os
from pathlib import Path as _P
BENCH_ROOT = _P(__file__).resolve().parents[2]
SYS_MAPPING_ROOT = _P(os.environ.get("SYS_MAPPING_ROOT", _P.home() / "software" / "sys_mapping"))
EXPORT = BENCH_ROOT / "results" / "docs_exports"
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import healpy as hp
from astropy.io import fits
import sys_mapping as sm
from sys_mapping.correction import rotate_templates

sys.path.insert(0, str(BENCH_ROOT / "characterisation"))
from run_crossterm_bias import template_xi_matrix  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "_ls10", str(SYS_MAPPING_ROOT / "scripts/run_ls10_analysis.py"))
mod = importlib.util.module_from_spec(spec); sys.modules["_ls10"] = mod; spec.loader.exec_module(mod)

PARAMS = SYS_MAPPING_ROOT / "data/sys_weights"
RAND = Path.home() / "data/legacysurvey/dr10/sweep/BGS_VLIM_Mstar"
SYST = Path.home() / "data/legacysurvey/dr10/systematics"
theta = np.radians(np.linspace(0.5, 5.0, 10))
METHODS = ("OLS", "ElasticNet", "ISD-1", "ISD-3", "MCMC-add", "MCMC-comb")


def parts(amps, xi):
    n = amps.size
    autos = np.array([xi[i, i] for i in range(n)])
    var = np.full(n, 1e-5); w_obs = np.full(theta.size, 1e-3); z = np.zeros(n)
    d_full = w_obs - sm.correct_two_point_function(w_obs, amps, z, var, z, xi)
    d_auto = w_obs - sm.correct_two_point_function(w_obs, amps, z, var, z, autos)
    return d_full, d_auto


rows = []
t0 = time.time()
tmpl = {ns: mod.load_templates_from_dir(str(SYST / f"{ns:04d}"), ns) for ns in (32, 64)}
for ns in (32, 64):
    T, names = tmpl[ns]
    for pf in sorted(PARAMS.glob(f"*_NSIDE{ns:04d}_params.json")):
        d = json.loads(pf.read_text())
        sid = d["sample_id"]
        assert [n for n in names] == d["template_names"], (names, d["template_names"])
        with fits.open(RAND / f"{sid}_RAND.fits", memmap=True) as h:
            r = h[1].data
            rc = sm.pixelize_catalog(np.asarray(r["RA"], float), np.asarray(r["DEC"], float), ns)
        good = rc >= 0.1 * rc.max()
        dt = sm.assign_template_values(T, good)
        dt, mean_b, rms_b = sm.standardise_on_footprint(dt, return_scales=True)
        rms_rec = np.asarray(d["template_basis"]["rms_before"])
        basis_err = float(np.max(np.abs(rms_b / rms_rec - 1)))
        xi = template_xi_matrix(dt, good, ns, theta)
        _, R, _ = rotate_templates(dt)
        xi_rot = np.einsum("ia,abk,jb->ijk", R, xi, R)
        for m in METHODS:
            a = np.asarray(d["methods"][m]["a_hat"], float)
            if not np.any(a):
                rows.append(dict(nside=ns, logM=float(sid.split("_")[3]), method=m, n_good=int(good.sum()),
                                 n_good_rec=d["n_good_pix"], basis_rms_err=basis_err, norm=0.0))
                continue
            f_o, a_o = parts(a, xi)
            f_r, a_r = parts(R @ a, xi_rot)
            rows.append(dict(
                nside=ns, logM=float(sid.split("_")[3]), method=m, n_good=int(good.sum()),
                n_good_rec=d["n_good_pix"], basis_rms_err=basis_err, norm=float(np.linalg.norm(a)),
                invariance=float(np.max(np.abs(f_o - f_r) / np.abs(f_o))),
                share_orig_median=float(np.median(np.abs(f_o - a_o) / np.abs(f_o))),
                share_rot_median=float(np.median(np.abs(f_r - a_r) / np.abs(f_r))),
                share_rot_max=float(np.max(np.abs(f_r - a_r) / np.abs(f_r))),
                share_rot_per_theta=(np.abs(f_r - a_r) / np.abs(f_r)).tolist(),
                share_orig_per_theta=(np.abs(f_o - a_o) / np.abs(f_o)).tolist(),
            ))
        print(f"{ns} {sid.split('_')[3]:>6} good={good.sum()} (rec {d['n_good_pix']}) basis_err={basis_err:.2e} "
              f"t={time.time()-t0:.0f}s", flush=True)

out = EXPORT / "crossterm" / "crossterm_share_footprint_20260912p.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"theta_deg": np.degrees(theta).tolist(), "rows": rows}, indent=1))
import pandas as pd
df = pd.DataFrame([{k: v for k, v in r.items() if not k.endswith("per_theta")} for r in rows])
pd.set_option("display.width", 220)
print(df.round(4).to_string())
g = df.dropna(subset=["share_rot_median"]).groupby(["nside", "method"])
print((g[["share_orig_median", "share_rot_median"]].median() * 100).round(1))
print((g[["share_orig_median", "share_rot_median"]].min() * 100).round(1))
print((g[["share_orig_median", "share_rot_median", "share_rot_max"]].max() * 100).round(1))
