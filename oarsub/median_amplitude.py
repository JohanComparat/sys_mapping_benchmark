"""Median converged cl_amplitude across family B's seeds for one cell.

Prints nothing when no seed converged, so the caller can fall back or fail
loudly rather than silently reusing the 5e-4 default the campaign exists to
replace.
"""
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
vals = []
for p in root.rglob("glass_calibration.json"):
    try:
        fit = json.load(p.open()).get("fit") or {}
    except (OSError, ValueError):
        continue
    if fit.get("converged") and isinstance(fit.get("cl_amplitude"), (int, float)):
        vals.append(fit["cl_amplitude"])
print(f"{statistics.median(vals):.6g}" if vals else "")
