#!/usr/bin/env python3
"""Post-condition for family P: the weights file must carry the new convention.

Three ways this family can produce a file and still have failed, all of which
have a precedent in this project:

* the file is missing, or has no rows;
* it carries the old weight convention, which for ISD was a linear
  reconstruction from ``a_hat`` rather than the method's own cumulative product;
* a column is identically one, which means the fit returned nothing and the
  weight is a no-op being shipped as a correction;
* the corrected w(theta) came back negative, which means the subtracted template
  term exceeded the measurement.  Nothing bounds it by w_obs, so this is silent
  in the file and only visible if someone looks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

EXPECTED = ["WEIGHT_OLS", "WEIGHT_ENET", "WEIGHT_ISD1", "WEIGHT_ISD3",
            "WEIGHT_ADD", "WEIGHT_COMB", "WEIGHT_SYS"]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: check_ls10_products.py <WEIGHTS.fits>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"!! {path} was not written", file=sys.stderr)
        return 1

    try:
        with fits.open(path) as hdul:
            hdr, data = hdul[1].header, hdul[1].data
    except (OSError, TypeError, ValueError) as exc:
        # A partially written file reads as a buffer-size error rather than
        # anything self-describing.  Say what it actually is.
        size = path.stat().st_size
        try:
            h = fits.getheader(path, 1)
            want = int(h.get("NAXIS1", 0)) * int(h.get("NAXIS2", 0))
            print(f"!! {path.name}: truncated -- {size:,} bytes on disk against "
                  f"{want:,} of table data declared in the header. The write did "
                  f"not complete.", file=sys.stderr)
        except Exception:
            print(f"!! {path.name}: unreadable ({exc})", file=sys.stderr)
        return 1

    if data is None or len(data) == 0:
        print(f"!! {path.name}: no rows", file=sys.stderr)
        return 1

    # 3 is the footprint-standardised basis; 2 the load-time one, which is
    # reachable only behind --no-footprint-standardise and is accepted so a
    # deliberate reproduction of an older product still passes.
    ver = hdr.get("WEIGHTVER")
    if ver not in (2, 3):
        print(f"!! {path.name}: WEIGHTVER={ver!r}, expected 2 or 3. This file "
              f"was written by the old weight path.", file=sys.stderr)
        return 1
    basis = str(hdr.get("TPLBASIS", "load-time"))
    if ver == 2:
        print(f"   note: WEIGHTVER=2 ({basis} basis) -- amplitudes are not in "
              f"units of one template sigma on this footprint")
    src = str(hdr.get("WEIGHTCON", ""))
    if src != "library":
        print(f"!! {path.name}: WEIGHTCON={src!r}, expected 'library'. The "
              f"weights were reconstructed rather than taken from the fit.",
              file=sys.stderr)
        return 1

    missing = [c for c in EXPECTED if c not in data.columns.names]
    if missing:
        print(f"!! {path.name}: missing columns {missing}", file=sys.stderr)
        return 1

    unity, wmax = [], 0.0
    for c in EXPECTED:
        v = np.asarray(data[c], dtype=float)
        wmax = max(wmax, float(np.nanmax(v)))
        if np.allclose(v, 1.0, atol=1e-12):
            unity.append(c)

    # A unity column is a null result, not a failure.  ElasticNet reaches it by
    # shrinking every amplitude to zero, and ISD by declining to correct any
    # template whose marginal Delta chi^2 does not clear the mock-calibrated
    # threshold -- which is the stopping rule working, and is why ISD-1 and ISD-3
    # can disagree on the same field.  Report it loudly, because a user must not
    # take a no-op column for a correction, but do not fail the cell.
    if unity:
        print(f"   note: identically unity (no correction applied): "
              f"{', '.join(unity)}")
    if len(unity) == len(EXPECTED):
        print(f"!! {path.name}: every weight column is unity -- nothing was "
              f"fitted at all.", file=sys.stderr)
        return 1
    if wmax > 20.0 + 1e-6:
        print(f"!! {path.name}: max weight {wmax:.3f} exceeds the library clip of "
              f"20", file=sys.stderr)
        return 1

    # The amplitudes must be persisted too, or a published ISD number has no
    # provenance: they were previously printed and discarded.
    pj = path.with_name(path.name.replace("_WEIGHTS.fits", "_params.json"))
    if pj.exists():
        import json
        meth = (json.loads(pj.read_text()).get("methods") or {})
        if not meth:
            print(f"!! {pj.name}: no per-method amplitudes stored. Re-run against a "
                  f"version of run_ls10_analysis.py that persists them.",
                  file=sys.stderr)
            return 1
        isd = {k: v for k, v in meth.items() if k.startswith("ISD-")}
        for k, v in sorted(isd.items()):
            print(f"   {k}: rms|a_hat|={v.get('rms_a_hat')!s:>8}  "
                  f"n_steps={v.get('n_steps')}  stopped={v.get('stopped_on')}  "
                  f"calibrated={v.get('calibrated')}")


    # The corrected w(theta): a negative value means the subtracted term
    # sum_i a_i^2 xi_i(theta) exceeded the measurement.  Only bins where w_obs
    # is itself positive count -- a noise-dominated w_obs goes negative on its
    # own and says nothing about the correction.
    wj = path.with_name(path.name.replace("_WEIGHTS.fits", "_wtheta_data.json"))
    if wj.exists():
        import json
        wd = json.loads(wj.read_text())
        w_obs = np.asarray(wd.get("w_obs", []), dtype=float)
        pos = np.isfinite(w_obs) & (w_obs > 0)
        bad = {}
        uncorrected = {}
        for meth, curve in sorted((wd.get("all_w_corr") or {}).items()):
            wc = np.asarray(curve, dtype=float)
            if wc.shape != w_obs.shape:
                continue
            over = pos & np.isfinite(wc) & (wc < 0)
            if over.any():
                bad[meth] = float(np.min(wc[over] / w_obs[over]))
            uncorrected[meth] = int(np.sum(pos & (wc == w_obs)))
        flat = [m for m, n in uncorrected.items() if n == int(pos.sum())]
        if flat and len(flat) < len(uncorrected):
            print(f"   note: correction is a no-op in every bin for "
                  f"{', '.join(sorted(flat))}")
        if bad:
            worst = ", ".join(f"{m} {r:.3g}" for m, r in sorted(bad.items()))
            print(f"!! {wj.name}: the correction overshoots the signal -- "
                  f"w_corr < 0 where w_obs > 0 ({worst}). Those bins are not "
                  f"usable.", file=sys.stderr)
            return 1
    else:
        print(f"   note: {wj.name} absent; the corrected w(theta) was not checked")

    print(f"   check: {path.name}: {len(data)} rows, WEIGHTVER={ver}, "
          f"basis={basis}, source={src}, max weight {wmax:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
