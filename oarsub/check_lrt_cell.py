"""Post-condition for one family-E cell: did it really produce a mock null?

The first pass of E exited 0 in under a minute having computed nothing, because
--resume-null found no file to resume and skipped the work.  A campaign whose
jobs can succeed without doing anything cannot be trusted at the end, so each
cell now asserts its own output.
"""
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
want = int(sys.argv[2])

if not path.exists():
    sys.exit(f"!! no params.json at {path}")

lrt = (json.loads(path.read_text()).get("lrt") or {})
calib = lrt.get("calibration")
null = lrt.get("null_lambda") or []

problems = []
if calib != "mock":
    problems.append(f"calibration is {calib!r}, expected 'mock'")
if len(null) < want:
    problems.append(f"null_lambda has {len(null)} draws, expected >= {want}")
if problems:
    sys.exit("!! " + path.name + ": " + "; ".join(problems))

print(f"-- verified: {len(null)} mock draws, calibration={calib}, "
      f"p={lrt.get('p_value_mock', lrt.get('p_value'))}")
