"""Post-condition for one family-M cell: was a usable spectrum produced?

The matcher legitimately ends "NOT converged" when a band sits inside its own
cosmic-variance floor but outside the requested tolerance --- at l=2, on five
modes, that floor is ~26 per cent.  So convergence is the wrong thing to assert.
What must be true is that a spectrum was written, that it is positive and finite,
and that the large scales actually moved away from the parametric starting point,
which is the whole purpose of the run.
"""
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    sys.exit(f"!! no match written at {path}")

d = json.loads(path.read_text())
cl = d.get("cl_matched") or []
hist = d.get("history") or []
problems = []
if len(cl) < 8:
    problems.append(f"cl_matched has {len(cl)} multipoles")
if any((c is None) or (c != c) or (c < 0) for c in cl):
    problems.append("cl_matched contains negative or non-finite values")
if not hist:
    problems.append("no iteration history")
else:
    last = hist[-1]
    if last.get("density_err", 1.0) > 0.05:
        problems.append(f"density still off by {last['density_err']:.3f}")
# The gate: the spectrum must have been shown, on seeds it was not fitted to, to
# reproduce the data's large-scale clustering.  Without this a cell can "succeed"
# by writing a file, which is how an uncalibrated null gets used.
val = d.get("validation")
if val is None:
    problems.append("no validation block -- the large-scale check did not run")
elif not val.get("passed"):
    problems.append(f"FAILED the large-scale check: power ratio "
                    f"{val.get('large_scale_ratio', float('nan')):.3f} over "
                    f"l={val.get('l_range')}, tol {val.get('tol')}")

if problems:
    sys.exit("!! " + path.name + ": " + "; ".join(problems))

last = hist[-1]
print(f"-- verified: {len(cl)} multipoles, {len(hist)} iterations, "
      f"density err {last.get('density_err', float('nan')):.4f}; "
      f"large-scale power ratio {val['large_scale_ratio']:.3f} "
      f"(tol {val['tol']}, {val['n_seeds']} unseen seeds) PASSED")
