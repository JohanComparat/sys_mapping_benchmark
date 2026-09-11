#!/usr/bin/env python3
"""Post-condition for family H: refuse to exit 0 on an empty or useless cell.

A job that produces a file is not a job that produced a measurement.  The three
ways this family can silently fail are all cheap to detect and all have happened
to some family here before:

* no ``results_summary.json`` at all (the run died after its last print);
* the ISD entries carry no ``steps`` record, which means the runner was the old
  one that dropped them and the cell cannot answer the question it was run for;
* ``calibrated`` is false, which means the stopping threshold was in raw
  Delta chi^2 units and the amplitudes overshoot by 25--40%.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _check_sweep(root: Path) -> int:
    """The sweep writes one JSON per seed, not a results_summary tree."""
    files = sorted(root.glob("seed*.json"))
    if not files:
        print(f"!! no sweep output under {root}", file=sys.stderr)
        return 1
    for f in files:
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError) as exc:
            print(f"!! {f}: unreadable ({exc})", file=sys.stderr)
            return 1
        rows = d.get("rows") or []
        if not rows:
            print(f"!! {f}: no rows", file=sys.stderr)
            return 1
        n_set = len({r["setting_index"] for r in rows})
        if n_set < 2:
            print(f"!! {f}: only {n_set} setting(s) -- died early?", file=sys.stderr)
            return 1
        print(f"   check: {f.name}: {len(rows)} rows over {n_set} settings, "
              f"probe order {d.get('probe_order')}")
    return 0


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--sweep":
        if len(sys.argv) < 3:
            print("usage: check_isd_cell.py --sweep <sweep_dir>", file=sys.stderr)
            return 2
        return _check_sweep(Path(sys.argv[2]))
    if len(sys.argv) < 3:
        print("usage: check_isd_cell.py <mode_dir> <nside>", file=sys.stderr)
        return 2
    root, nside = Path(sys.argv[1]), int(sys.argv[2])

    summaries = sorted(root.glob(f"**/nside{nside:04d}/results_summary.json"))
    if not summaries:
        print(f"!! no results_summary.json under {root}", file=sys.stderr)
        return 1

    bad = []
    for f in summaries:
        try:
            entries = json.loads(f.read_text())
        except (OSError, ValueError) as exc:
            bad.append(f"{f}: unreadable ({exc})")
            continue
        if not entries:
            bad.append(f"{f}: empty")
            continue
        for e in entries:
            isd = [k for k in e if k.startswith("params_ISD-")]
            if not isd:
                bad.append(f"{f}: no ISD column")
                break
            for k in isd:
                p = e[k]
                if "steps" not in p:
                    bad.append(f"{f}: {k} has no step record (stale runner?)")
                    break
                if not p.get("calibrated"):
                    bad.append(f"{f}: {k} ran with an uncalibrated threshold")
                    break
            else:
                continue
            break

    if bad:
        print(f"!! {len(bad)} problem(s):", file=sys.stderr)
        for b in bad[:10]:
            print(f"   {b}", file=sys.stderr)
        return 1

    n_cfg = sum(len(json.loads(f.read_text())) for f in summaries)
    print(f"   check: {len(summaries)} summary file(s), {n_cfg} configurations, "
          f"ISD steps recorded, threshold calibrated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
