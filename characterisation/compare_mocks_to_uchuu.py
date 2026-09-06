#!/usr/bin/env python3
"""Compare data, matched GLASS and Uchuu on large-scale clustering.

Why
---
The GLASS mock is matched to the *data's* measured spectrum, and the data's
spectrum contains whatever residual systematic power the survey carries --
largest exactly where the match matters, at low l.  So agreeing with the data is
necessary but not sufficient: it cannot distinguish "the mock now has the right
clustering" from "the mock now has the right clustering plus the survey's
gradients".

Uchuu breaks that degeneracy.  It is an N-body/HOD realisation of the same
sample, density-matched to it, with genuine two-halo clustering and no survey
systematics at all.  Three-way:

    data ~ GLASS ~ Uchuu   the match is physical; the data's low-l power is
                           clustering, and the null is calibrated.
    data ~ GLASS >> Uchuu  the match absorbed survey systematics into the null.
                           The null is then too WIDE, which is conservative for
                           detection but wrong for error bars.
    GLASS ~ Uchuu << data  the matcher failed and the data's excess is real.

Footprints differ -- LS10 covers ~18200 deg^2, the Uchuu mocks an octant
(~5160 deg^2) -- and that difference cannot be corrected away by dividing each
pseudo-C_l by its own f_sky.  That leading-order correction is worst exactly at
the low multipoles this comparison is about, and on this pair it is wrong by 35
per cent: it made Uchuu look 33 per cent short of the data (ratio 0.673) when on
a common mask the two agree to 3 per cent (1.029).

So the comparison is made on the INTERSECTION of the two footprints.  Both
fields then see the same mask, the same mode coupling and the same estimator, and
whatever the mask does to the spectrum divides out of the ratio instead of being
modelled.  The price is area -- the overlap is ~5 per cent of sky against LS10's
46 -- and correspondingly larger cosmic variance, which is the honest trade.

Usage
-----
    python characterisation/compare_mocks_to_uchuu.py \
        --sample LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238 \
        --match results/glass_match/<tag> --nside 32
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import sys                                       # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sys_mapping as sm                         # noqa: E402
from match_glass_to_data import (                # noqa: E402
    measure_signal_cl, uniform_randoms_on)

UCHUU = Path.home() / "data/Uchuu/FullSky/mock_catalogues"


def find_uchuu(sample: str) -> Path | None:
    """The Uchuu mock built for this LS10 sample, via its SUMMARY mapping."""
    for f in UCHUU.glob(f"SUMMARY_{sample}_to_MOCK_*.txt"):
        m = re.search(r"_to_(MOCK_\S+)\.txt$", f.name)
        if m and (UCHUU / m.group(1)).is_dir():
            return UCHUU / m.group(1)
    return None


def large_scale_power(cl, fsky, lo=2, hi=32):
    """Mode-weighted mean C_l over the large-scale band, scaled by 1/f_sky.

    The 1/f_sky factor is the leading-order mask correction and is only good when
    the masks being compared are similar.  Use :func:`on_common_mask` for the
    data-vs-Uchuu comparison, where they are not.
    """
    hi = int(min(hi, len(cl) - 1))
    w = 2 * np.arange(lo, hi + 1) + 1.0
    return float(np.sum(w * cl[lo:hi + 1]) / w.sum() / max(fsky, 1e-12))


def rp_to_ell(rp_mpch, z_eff, cosmo=None):
    """Multipole corresponding to a projected separation rp, at redshift z_eff.

    wp(rp) is measured in Mpc/h of transverse separation; C_l is measured in
    multipoles.  A transverse comoving separation rp subtends
    theta = rp / D_C(z_eff), and a feature of angular size theta appears near
    l ~ pi / theta.  So the SAME physical scale is a different multipole for
    every sample, because the samples sit at different redshifts: rp = 10 Mpc/h
    is l = 63 for the logM >= 9.0 sample at z_eff 0.07 and l = 241 for
    logM >= 11.5 at z_eff 0.27.  Comparing all samples in one fixed l band would
    compare different physics in each.
    """
    from astropy.cosmology import FlatLambdaCDM
    cosmo = cosmo or FlatLambdaCDM(H0=100, Om0=0.31)   # H0=100 => Mpc/h
    d_c = float(cosmo.comoving_distance(z_eff).value)
    theta = float(rp_mpch) / d_c                        # radians
    return np.pi / theta, d_c


def on_common_mask(n_gal, n_rand, mask, nside, lmax, lo=2, hi=32):
    """Large-scale power measured through one shared mask.

    Restricting both catalogues to the same pixels makes the mode coupling
    identical, so it cancels in the ratio rather than having to be deconvolved.
    """
    ng = np.where(mask, n_gal, 0)
    nr = np.where(mask, n_rand, 0)
    cl, nbar, fsky, _ = measure_signal_cl(ng, nr, nside, lmax)
    hi = int(min(hi, len(cl) - 1))
    w = 2 * np.arange(lo, hi + 1) + 1.0
    return float(np.sum(w * cl[lo:hi + 1]) / w.sum() / max(fsky, 1e-12)), nbar, fsky


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--catalog-dir",
                    default=str(Path.home() / "data/legacysurvey/dr10/sweep/BGS_VLIM_Mstar"))
    ap.add_argument("--match", default=None,
                    help="directory of *_match.json from match_glass_to_data.py")
    ap.add_argument("--nside", type=int, default=32)
    ap.add_argument("--lmax", type=int, default=None)
    ap.add_argument("--rp-mpch", type=float, nargs=2, default=[5.0, 20.0],
                    metavar=("RP_MIN", "RP_MAX"),
                    help="projected separation range to compare over, in Mpc/h "
                         "(the wp(rp) scale).  Converted per sample to a "
                         "multipole band through the sample's own z_eff, since "
                         "the same physical scale is a different l at each "
                         "redshift.  Default 5 20.")
    ap.add_argument("--l-large", type=int, default=32,
                    help="fallback band when --rp-mpch cannot be reached at this NSIDE")
    ap.add_argument("--n-seeds", type=int, default=6)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "results" / "mock_comparison")
    a = ap.parse_args()

    from astropy.table import Table
    lmax = a.lmax or 2 * a.nside
    ns = a.nside

    def field(dat, rnd):
        ng = sm.pixelize_catalog(np.asarray(dat["RA"]), np.asarray(dat["DEC"]), ns)
        nr = sm.pixelize_catalog(np.asarray(rnd["RA"]), np.asarray(rnd["DEC"]), ns)
        cl, nbar, fsky, good = measure_signal_cl(ng, nr, ns, lmax)
        return cl, nbar, fsky, good, nr

    base = Path(a.catalog_dir) / a.sample
    d = Table.read(f"{base}_DATA.fits"); r = Table.read(f"{base}_RAND.fits")
    cl_d, nbar_d, fsky_d, good_d, nr_d = field(d, r)

    # The comparison band comes from the physical scale, per sample.
    zc0 = next(c for c in ("BEST_Z", "Z", "z") if c in d.colnames)
    z_eff = float(np.median(np.asarray(d[zc0], float)))
    rp_hi, rp_lo = min(a.rp_mpch), max(a.rp_mpch)     # small rp -> large l
    l_hi, d_c = rp_to_ell(rp_hi, z_eff)
    l_lo, _ = rp_to_ell(rp_lo, z_eff)
    band_lo, band_hi = int(max(2, round(l_lo))), int(round(l_hi))
    reachable = band_hi <= lmax
    print(f"sample {a.sample}  NSIDE={ns}  z_eff={z_eff:.3f}  D_C={d_c:.0f} Mpc/h")
    print(f"  rp = {rp_hi:g}-{rp_lo:g} Mpc/h  ->  l = {band_lo}-{band_hi}"
          f"   (lmax here is {lmax})")
    if not reachable:
        need = int(2 ** np.ceil(np.log2(max(band_hi / 2.0, 1))))
        print(f"  !! this NSIDE cannot reach l={band_hi}: the pixel window "
              f"suppresses power above l~2*NSIDE={2*ns}.  Re-run at NSIDE >= "
              f"{need} for the full rp range, or the comparison is only valid "
              f"over l = {band_lo}-{min(band_hi, lmax)} "
              f"(rp >= {np.pi / (lmax / d_c) if lmax else 0:.1f} Mpc/h).")
        band_hi = lmax
    p_data = large_scale_power(cl_d, fsky_d, lo=band_lo, hi=band_hi)
    print(f"  DATA   fsky {fsky_d:.4f}  nbar {nbar_d:8.2f}  "
          f"large-scale C_l {p_data:.4e}")

    rows = {"data": {"fsky": fsky_d, "nbar": nbar_d, "cl_large": p_data}}

    # --- Uchuu: its own octant footprint, its own randoms ---
    common = None
    udir = find_uchuu(a.sample)
    if udir is None:
        print("  UCHUU  not found for this sample")
    else:
        ud = Table.read(next(udir.glob("*_DATA.fits.gz")))
        ur = Table.read(next(udir.glob("*_RAND.fits.gz")))
        ung = sm.pixelize_catalog(np.asarray(ud["RA"]), np.asarray(ud["DEC"]), ns)
        unr = sm.pixelize_catalog(np.asarray(ur["RA"]), np.asarray(ur["DEC"]), ns)
        overlap = (np.asarray(nr_d) > 0) & (unr > 0)
        n_ov = int(overlap.sum())
        if n_ov < 200:
            print(f"  UCHUU  only {n_ov} pixels overlap the data footprint -- "
                  f"too few to compare on a common mask")
        else:
            ngd = sm.pixelize_catalog(np.asarray(d["RA"]), np.asarray(d["DEC"]), ns)
            p_dc, nbd_c, f_ov = on_common_mask(ngd, np.asarray(nr_d), overlap,
                                               ns, lmax, lo=band_lo, hi=band_hi)
            p_uc, nbu_c, _ = on_common_mask(ung, unr, overlap, ns, lmax, lo=band_lo, hi=band_hi)
            print(f"  common mask: {n_ov} pixels (fsky {f_ov:.4f}); both fields "
                  f"see the same mode coupling")
            print(f"    DATA  on common mask  C_l {p_dc:.4e}  nbar {nbd_c:7.2f}")
            print(f"    UCHUU on common mask  C_l {p_uc:.4e}  nbar {nbu_c:7.2f}"
                  f"   ratio {p_uc / p_dc:6.3f}")
            rows["uchuu"] = {"n_overlap_pix": n_ov, "fsky_overlap": f_ov,
                             "cl_large_common": p_uc, "nbar": nbu_c,
                             "ratio_to_data_common": p_uc / p_dc, "dir": str(udir)}
            rows["data_common_mask"] = {"cl_large_common": p_dc, "nbar": nbd_c}
            common = {"mask": overlap, "p_data": p_dc}

    # --- GLASS, matched and default, on the data's footprint ---
    zc = next(c for c in ("BEST_Z", "Z", "z") if c in d.colnames)
    z = np.asarray(d[zc], float)
    ze, nz = sm.measure_nz(z, z.min(), z.max(), n_bins=20)
    n_total = int(len(d) / fsky_d)
    mock_rand = uniform_randoms_on(good_d, float(np.asarray(nr_d)[good_d].mean()))

    cl_fit = None
    if a.match:
        cl_fit = sm.load_matched_cl(a.match, a.sample, ns)

    for label, kw in (("GLASS default", dict(cl_amplitude=5e-4)),
                      ("GLASS matched", dict(cl_input=cl_fit))):
        if kw.get("cl_input") is None and label.endswith("matched"):
            print("  GLASS matched  no validated spectrum supplied")
            continue
        cls, nbars, ngs = [], [], []
        for k in range(a.n_seeds):
            cat = sm.generate_glass_fullsky_mock(ns, n_total, ze, nz, seed=430001 + k,
                                                 rand_factor=2, **kw)
            ng = sm.pixelize_catalog(cat["ra"], cat["dec"], ns)
            cl_m, nbar_m, _, _ = measure_signal_cl(ng, mock_rand, ns, lmax)
            cls.append(cl_m); nbars.append(nbar_m); ngs.append(np.asarray(ng, float))
        p_g = large_scale_power(np.mean(cls, axis=0), fsky_d, lo=band_lo, hi=band_hi)
        rec = {"cl_large": p_g, "ratio_to_data": p_g / p_data,
               "nbar": float(np.mean(nbars))}
        line = (f"  {label:<14} fsky {fsky_d:.4f}  nbar {np.mean(nbars):8.2f}  "
                f"C_l {p_g:.4e}   ratio to data {p_g / p_data:6.3f}")
        # Also on the Uchuu overlap, so all three are compared through one mask.
        # Ratios taken through different masks are not comparable to each other,
        # even when each is individually correct.
        if common is not None:
            # The mock carries no angular selection, so it is normalised against
            # a UNIFORM expectation on the common mask -- not against the data's
            # randoms, which encode the survey selection and would imprint it on
            # the mock as ~5x spurious power.  And the average is taken over
            # per-seed spectra, not over count maps: averaging the maps first
            # averages away the very fluctuations being measured.
            cm = common["mask"]
            unif_c = uniform_randoms_on(cm, float(np.asarray(nr_d)[cm].mean()))
            pgs = [on_common_mask(g, unif_c, cm, ns, lmax,
                                  lo=band_lo, hi=band_hi)[0] for g in ngs]
            p_gc = float(np.mean(pgs))
            rec["cl_large_common"] = p_gc
            rec["ratio_to_data_common"] = p_gc / common["p_data"]
            line += f"   | common mask {p_gc / common['p_data']:6.3f}"
        print(line)
        rows[label.replace(" ", "_").lower()] = rec

    if "uchuu" in rows and "glass_matched" in rows and \
            "ratio_to_data_common" in rows["uchuu"]:
        ru = rows["uchuu"]["ratio_to_data_common"]
        rg = rows["glass_matched"].get("ratio_to_data_common",
                                       rows["glass_matched"]["ratio_to_data"])
        print(f"\n  on the common mask: Uchuu/data = {ru:.3f}, "
              f"matched GLASS/data = {rg:.3f}")
        if abs(ru - 1) <= 0.25 and abs(rg - 1) <= 0.25:
            print("  -> data, GLASS and Uchuu agree: the match is physical, and the "
                  "data's large-scale power is clustering rather than systematics.")
        elif ru < 0.75 <= rg:
            print("  -> GLASS tracks the data but Uchuu does not.  The data carries "
                  "large-scale power Uchuu has no mechanism for, so the match has "
                  "absorbed survey systematics into the null: conservative for "
                  "detection, but the null is wider than the true clustering.")
        else:
            print("  -> the three disagree; read the numbers rather than a verdict.")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"{a.sample}_NSIDE{ns:04d}_mockcomparison.json"
    out.write_text(json.dumps({"sample": a.sample, "nside": ns, "z_eff": z_eff,
                               "comoving_distance_mpch": d_c,
                               "rp_mpch": [rp_hi, rp_lo],
                               "ell_band": [band_lo, band_hi],
                               "ell_band_reachable": bool(reachable),
                               "rows": rows}, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
