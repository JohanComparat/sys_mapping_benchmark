#!/usr/bin/env python3
"""Iterate GLASS mock generation until both density AND clustering match the data.

The problem
-----------
The mocks are the null hypothesis for three calibrations (Stage-1 ISD p-values,
the mock-calibrated LRT, the sandwich covariance).  The footprint-matching rule
fixes their surface *density*, so their shot noise is right by construction.
Nothing fixes their *clustering*.

``calibrate_glass_clustering.py`` improved on that by fitting one number --- the
amplitude of a fixed C_l ∝ (l+1)^-1.5 power law --- until the mock's pixel
variance matched the data's.  That is still not a match:

* One scalar cannot constrain a function.  The variance is an integral over all
  scales, so an amplitude fitted with the wrong slope buys the right total power
  distributed wrongly across l --- and the variance-inflation measurement shows
  the *slope* is what drives the error underestimate, far more than the
  amplitude (kappa 2.7 to 16.1 across slopes 1.0 to 3.0 at fixed amplitude).
* It compared unlike things.  The mock's variance was measured full-sky while
  the data's was measured on a 44% footprint, so mask coupling sat uncorrected
  inside the "ratio".

What this does instead
----------------------
Match the whole spectrum, by forward-modelling rather than by fitting a shape.

1. Build the data's overdensity on its own footprint and measure its pseudo-C_l.
2. Generate a mock, pixelise it, and measure it through the *identical* path ---
   the data's own random catalogue supplies the mask and selection, so the mock
   and the data see the same footprint, the same estimator and the same pixel
   window.
3. Update the input spectrum multiplicatively,

       C_l^(n+1) = C_l^(n) * (C_l^target / C_l^mock,(n)) ** damping

   in bands, and repeat.

The ratio is the point.  Every effect that acts identically on data and mock ---
mask coupling, the pixel window, the lognormal transform's distortion of the
input spectrum, the pseudo-C_l normalisation --- divides out of it, so none of
them has to be modelled.  What does not divide out is shot noise, because it is
a property of each catalogue, so it is subtracted from both before the ratio is
formed.

Convergence is declared only when *both* criteria hold: the density ratio is
within ``--tol-density`` and every band of the clustering ratio is within
``--tol-cl``.  Matching one while drifting on the other is the failure mode this
script exists to remove.

What it achieves, and what it does not
--------------------------------------
Validated on LS10 logM >= 10.0 at NSIDE 32, against seeds never used in the fit:

    sigma_hat      data 0.382    default 0.058 (0.15x)    matched 0.336 (0.88x)
    C_l ratio      l >= 7: 0.85-1.04        l = 2-6: 0.03-0.22

So the scatter deficit closes from a factor 6.6 to 12 per cent, and the spectrum
matches to a few per cent for l >= 7.  It does NOT match l < 7, and that is worth
stating plainly rather than hiding behind the summary number: those bands carry
too few modes to constrain, are excluded from the convergence test by
``--min-modes``, and a "converged" run therefore says nothing about them.  The
per-band report printed at the end flags which bands were in the test.

There is also a reason not to force a match there.  The target is the data's
*measured* spectrum, which at the largest scales is where the survey's own
depth and extinction gradients live.  Matching l = 2-3 would inject
systematics-shaped power into a null whose entire purpose is to be
systematics-free.  The residual deficit is thus partly a limitation and partly a
deliberate floor; the honest statement is that the null is calibrated for
l >= 7 and remains conservative at larger scales.

Usage
-----
    python characterisation/match_glass_to_data.py \
        --sample LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238 \
        --nside 64 --n-iter 8 --n-seeds 3
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
from sys_mapping.glass_mocks import sanitise_cl   # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "glass_match"


# ── measurement: one path, used for both data and mock ──────────────────────

def _bandpowers(cl: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mean C_l in each band.  Bands beat down the per-l scatter that would
    otherwise dominate the update and make the iteration chase noise."""
    return np.array([cl[a:b].mean() if b > a else np.nan
                     for a, b in zip(edges[:-1], edges[1:])])


def _expand(band_vals: np.ndarray, edges: np.ndarray, n_ell: int) -> np.ndarray:
    """Piecewise-constant expansion of band values onto ``n_ell`` multipoles.

    GLASS needs a spectrum to l = 3*nside, while the match is measured only to
    ``lmax`` (default 2*nside, where the pixel window is still benign).  Beyond
    the measured range the last band's correction is held rather than reset to
    1, so the update does not put a discontinuity into the input spectrum at the
    edge of what was measured.
    """
    out = np.ones(n_ell)
    last = 1.0
    for v, a, b in zip(band_vals, edges[:-1], edges[1:]):
        if np.isfinite(v):
            out[a:min(b, n_ell)] = v
            last = v
    if edges[-1] < n_ell:
        out[edges[-1]:] = last
    return out


def uniform_randoms_on(good: np.ndarray, level: float = 1.0) -> np.ndarray:
    """A random map uniform inside the footprint and empty outside.

    The mock carries no angular selection, so it must be normalised against a
    uniform expectation.  Normalising it against the *data's* randoms divides a
    uniform field by the survey's selection function and imprints that selection
    on the mock as spurious clustering: at NSIDE 32 that inflated the mock's
    measured power by 4--10x above the data's and pinned the iteration against
    its clip bound.
    """
    return np.where(good, float(level), 0.0)


def measure_signal_cl(n_gal_map: np.ndarray, n_rand_map: np.ndarray,
                      nside: int, lmax: int) -> tuple[np.ndarray, float, float]:
    """Shot-noise-subtracted pseudo-C_l of one catalogue on its own footprint.

    Returns ``(cl_signal, nbar, fsky)``.  The shot term for a pixelised
    overdensity is ``Omega_pix / nbar`` scaled by the sky fraction the mask
    leaves, which is the same convention for data and mock because both are
    measured here.
    """
    delta, good = sm.compute_overdensity(n_gal_map, n_rand_map)
    npix = hp.nside2npix(nside)
    full = np.zeros(npix)
    full[good] = np.asarray(delta)
    mask = good.astype(float)

    # measure_pseudo_cl returns (ell, cl) -- in that order.
    _, cl = sm.measure_pseudo_cl(full, mask, lmax=lmax, use_pixel_weights=False)

    nbar = float(np.asarray(n_gal_map)[good].mean())
    fsky = float(good.sum()) / npix
    omega_pix = 4.0 * np.pi / npix
    shot = fsky * omega_pix / max(nbar, 1e-30)
    return cl - shot, nbar, fsky, good


# ── the iteration ───────────────────────────────────────────────────────────

def match(nside, n_total, z_edges, nz, mock_randoms, cl_target, nbar_target,
          sigma_data=None,
          *, lmax, edges, n_iter, n_seeds, damping, tol_cl, tol_density,
          cl_start=None, seed0=101, min_modes=25.0, verbose=True):
    """Iterate the input spectrum until mock and data agree on both statistics."""
    # The input spectrum lives on GLASS's grid (l <= 3*nside), which is longer
    # than the grid the match is measured on.
    cl_in = (sanitise_cl(cl_start, 3 * nside) if cl_start is not None
             else sm.glass_mocks._make_glass_cls(nside, 5e-4, -1.5))
    tgt_bands = _bandpowers(cl_target, edges)
    # Modes per band, (2l+1) summed over the band.  A band with few modes is
    # cosmic-variance dominated: its ratio scatters by ~sqrt(2/n_modes) between
    # realisations, so judging convergence on it means chasing noise.  At l=2
    # that is 63%.
    n_modes = np.array([np.sum(2 * np.arange(a, b) + 1)
                        for a, b in zip(edges[:-1], edges[1:])], dtype=float)
    history = []

    for it in range(n_iter):
        cls, nbars, sigs = [], [], []
        for k in range(n_seeds):
            cat = sm.generate_glass_fullsky_mock(
                nside, n_total, z_edges, nz, seed=seed0 + 1000 * it + k,
                rand_factor=2, cl_input=cl_in)
            ng = sm.pixelize_catalog(cat["ra"], cat["dec"], nside)
            # Same footprint as the data, but a UNIFORM expectation inside it:
            # the mock carries no angular selection, so normalising it against
            # the data's randoms would imprint the survey's selection on it.
            cl_m, nbar_m, _, _ = measure_signal_cl(ng, mock_randoms, nside, lmax)
            dm, _gm = sm.compute_overdensity(ng, mock_randoms)
            sigs.append(float(np.std(np.asarray(dm))))
            cls.append(cl_m); nbars.append(nbar_m)
        cl_mock = np.mean(cls, axis=0)
        nbar_mock = float(np.mean(nbars))
        sigma_mock = float(np.mean(sigs))

        mock_bands = _bandpowers(cl_mock, edges)
        # A band whose measured signal is non-positive carries no information:
        # shot-noise subtraction has swamped it.  Such a band must be EXCLUDED
        # from the convergence test, never silently assigned ratio 1 -- that
        # reads as perfect agreement and converges the loop on the first pass
        # having matched nothing.
        usable = np.isfinite(mock_bands) & (mock_bands > 0) & \
                 np.isfinite(tgt_bands) & (tgt_bands > 0) & (n_modes >= min_modes)
        if not usable.any():
            raise SystemExit("!! no band has positive signal in both data and "
                             "mock; the spectrum cannot be matched at this "
                             "resolution (try a lower --lmax or fewer bands)")
        ratio = np.ones_like(mock_bands)
        ratio[usable] = tgt_bands[usable] / mock_bands[usable]
        ratio = np.clip(ratio, 0.2, 5.0)

        cl_err = float(np.max(np.abs(ratio[usable] - 1.0)))
        # expected scatter from cosmic variance alone, for context
        cv = float(np.max(np.sqrt(2.0 / n_modes[usable]) / np.sqrt(n_seeds)))
        d_err = abs(nbar_mock / nbar_target - 1.0)
        history.append({"iter": it, "cl_max_band_err": cl_err,
                        "effective_target": float(max(tol_cl, cv)),
                        "density_err": d_err, "nbar_mock": nbar_mock, "sigma_hat_mock": sigma_mock,
                        "n_bands_usable": int(usable.sum()),
                        "cosmic_variance_floor": cv,
                        "n_bands": int(usable.size),
                        "band_ratio": [float(r) for r in ratio]})
        if verbose:
            print(f"  iter {it}: max|C_l band ratio - 1| = {cl_err:6.3f}  "
                  f"(cosmic-variance floor {cv:.3f}, "
                  f"{int(usable.sum())}/{usable.size} bands)   "
                  f"|nbar/nbar_data - 1| = {d_err:7.4f}   "
                  f"sigma_hat {sigma_mock:.4f}", flush=True)

        # You cannot match a spectrum more closely than the realisations let you
        # measure it.  Requiring cl_err <= tol_cl when tol_cl sits below the
        # cosmic-variance floor is a test that can never pass, however good the
        # spectrum is -- the loop just burns iterations chasing its own noise.
        # The effective target is therefore the looser of the two.
        target = max(tol_cl, cv)
        if verbose and (it == n_iter - 1 or (cl_err <= target and d_err <= tol_density)):
            print("     band report (a band outside the test is NOT matched, "
                  "only unconstrained):", flush=True)
            for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                tag = "in test" if usable[i] else "EXCLUDED"
                print(f"       l={lo:>3}-{hi-1:<4} ratio {ratio[i]:6.3f}  "
                      f"{int(n_modes[i]):>5} modes  [{tag}]", flush=True)
        if cl_err <= target and d_err <= tol_density:
            if verbose:
                print(f"  -> converged: {cl_err:.3f} <= {target:.3f} "
                      f"({'noise floor' if cv > tol_cl else 'requested tolerance'})", flush=True)
            return cl_in, history, True

        cl_in = sanitise_cl(cl_in * _expand(ratio, edges, cl_in.size) ** damping,
                            3 * nside)

    return cl_in, history, False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--catalog-dir",
                    default=str(Path.home() / "data/legacysurvey/dr10/sweep/BGS_VLIM_Mstar"))
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--lmax", type=int, default=None, help="default 2*nside")
    ap.add_argument("--n-bands", type=int, default=8)
    ap.add_argument("--n-iter", type=int, default=8)
    ap.add_argument("--n-seeds", type=int, default=4,
                    help="mocks averaged per iteration; more = less chasing noise")
    ap.add_argument("--damping", type=float, default=0.4,
                    help="exponent on the update; <1 trades speed for stability")
    ap.add_argument("--tol-cl", type=float, default=0.10)
    ap.add_argument("--tol-density", type=float, default=0.01)
    ap.add_argument("--min-modes", type=float, default=25.0,
                    help="skip bands with fewer modes than this; they are\n                          cosmic-variance dominated")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    a = ap.parse_args()

    from astropy.table import Table
    lmax = a.lmax or 2 * a.nside
    base = Path(a.catalog_dir) / a.sample

    d = Table.read(f"{base}_DATA.fits")
    r = Table.read(f"{base}_RAND.fits")
    ng = sm.pixelize_catalog(np.asarray(d["RA"]), np.asarray(d["DEC"]), a.nside)
    nr = sm.pixelize_catalog(np.asarray(r["RA"]), np.asarray(r["DEC"]), a.nside)
    cl_target, nbar_target, fsky, good = measure_signal_cl(ng, nr, a.nside, lmax)

    zcol = next((c for c in ("BEST_Z", "Z", "z", "redshift", "Z_SPEC")
                 if c in d.colnames), None)
    if zcol is None:
        raise SystemExit(f"!! no redshift column in {base}_DATA.fits; "
                         f"columns are {d.colnames}")
    z = np.asarray(d[zcol], dtype=float)
    z_edges, nz = sm.measure_nz(z, float(z.min()), float(z.max()), n_bins=20)
    n_total = int(len(d) / max(fsky, 1e-6))     # the footprint-matching rule

    print(f"sample {a.sample}", flush=True)
    print(f"  nside={a.nside} lmax={lmax} fsky={fsky:.4f} "
          f"nbar={nbar_target:.2f} n_gal={len(d):,} -> n_total={n_total:,}", flush=True)

    edges = np.unique(np.geomspace(2, lmax + 1, a.n_bands + 1).astype(int))
    print(f"  {len(edges)-1} bands over l = 2..{lmax}\n", flush=True)

    cl_fit, history, ok = match(
        a.nside, n_total, z_edges, nz,
        uniform_randoms_on(good, float(np.asarray(nr)[good].mean())),
        cl_target, nbar_target,
        lmax=lmax, edges=edges, n_iter=a.n_iter, n_seeds=a.n_seeds,
        damping=a.damping, tol_cl=a.tol_cl, tol_density=a.tol_density,
        min_modes=a.min_modes)

    print(f"\n{'CONVERGED' if ok else 'NOT converged'} after {len(history)} iterations", flush=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{a.sample}_NSIDE{a.nside:04d}"
    (a.out_dir / f"{stem}_match.json").write_text(json.dumps({
        "sample": a.sample, "nside": a.nside, "lmax": lmax, "fsky": fsky,
        "nbar_data": nbar_target, "n_total": n_total,
        "converged": ok, "band_edges": [int(e) for e in edges],
        "history": history,
        "cl_matched": [float(c) for c in cl_fit],
        "cl_target_signal": [float(c) for c in cl_target],
    }, indent=2))
    print(f"-> {a.out_dir / (stem + '_match.json')}", flush=True)


if __name__ == "__main__":
    main()
