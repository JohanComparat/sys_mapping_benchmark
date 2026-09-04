# OAR campaign

Job scripts for running the benchmark's five measurement families on a GRICAD
cluster. **Nothing here names an account, an allocation or a host** — those
three values live in `oarsub/site.sh`, which is git-ignored:

```bash
cp oarsub/site.sh.example oarsub/site.sh && $EDITOR oarsub/site.sh
```

The scheduler is **OAR, not SLURM**. The dialect differences that actually bite:
`--name` rather than `-n`; the submitting environment is *not* propagated, so
every setting is passed positionally; `$OAR_ARRAY_INDEX` is 1-based;
`OAR_RES_NB_CORES` does not exist (read `$OAR_NODEFILE`); walltime caps at 48 h;
`-t devel` caps at 30 min; and GRICAD refuses a submission that would leave more
than 100 jobs waiting, so no array here exceeds 94.

## Why these five families

Each one closes a gap where the published result is a single point or is
under-sampled to the edge of meaning:

| | Family | Gap it closes |
|---|---|---|
| B | GLASS calibration | 1 of 36 cells was measured, single seed, 3-point scan — not a fit |
| C | variance cube | FPR(3σ) rested on 3 events in 150; the NSIDE × `n_sys` × slope cube was never run jointly |
| D | benchmark timings | stops at NSIDE 64; every MCMC row is one draw, so `mad = 0` |
| E | mock-calibrated LRT | the null used mocks ~25× under-clustered (mock σ̂ 0.117 vs data 0.397); `p` floored at 0.032 by N=31 |
| F | simulation tests | one realisation per config — every `frac_helped` is a fraction of ten |

## Order

```bash
# on your machine
./oarsub/rsync_to_dahu.sh code tier0 tier1     # ~92 MB
./oarsub/rsync_to_dahu.sh tier2                # +2.4 GB, only family E needs it

# on the frontend
TAG=$(date +%Y%m%d)
./oarsub/submit_campaign.sh $TAG A             # build the env, alone, first
./oarsub/campaign_status.sh $TAG               # check the import gate passed
./oarsub/submit_campaign.sh $TAG B C D F       # independent, run concurrently
./oarsub/submit_campaign.sh $TAG E 3.8e-2      # LRT, with B's fitted amplitude

# back on your machine
./oarsub/pull_results.sh $TAG
```

`E` deliberately refuses to start without an explicit `cl_amplitude`: running it
with the 5 × 10⁻⁴ default would reproduce the very bias it exists to remove.
`campaign_status.sh B` prints the median and scatter over B's runs and the exact
command to submit `E` with it.

## Outputs

Every family writes under
`$SMB_WORK/results/<family>/<tag>/…`, run-tagged so a later scan can never
overwrite an earlier grid — which is exactly how the 20-cell variance grid was
lost once already. A partial campaign therefore stays readable.

`campaign_status.sh` names the *missing cells* rather than counting files: a
count that says 34/36 does not tell you which two to resubmit.

## Environment

Build it with `build_env.sh`, **as a job** — the frontend kills a long solve.
It builds from `sys_mapping/pyproject.toml`, not `environment.yml`: the latter
omits `blackjax` and `joblib`, so an env built from it imports cleanly and then
fails every NUTS and LRT job at runtime. `_campaign_env.sh` gates on those
imports before any job does work, so a bad env fails once and loudly rather than
identically in 36 array elements an hour later.
