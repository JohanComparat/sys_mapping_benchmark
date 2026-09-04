# sys_mapping_benchmark

Timing benchmarks and algorithm characterisation for
[`sys_mapping`](https://github.com/JohanComparat/sys_mapping).

Kept separate from the package so that `sys_mapping`'s CI does not carry 259
timing cases, and so these measurements can be run and cited independently.

## Install

```bash
pip install -e ".[dev]"          # pulls sys-mapping from PyPI
```

Some characterisation scripts read pipeline outputs from a `sys_mapping` checkout.
Point them at it with `SYS_MAPPING_ROOT` (default `~/software/sys_mapping`).

## Timing

```bash
python benchmark/benchmark_pipeline.py --quick     # fast subset
python benchmark/benchmark_pipeline.py             # full grid
pytest benchmark/ -m "not slow"                    # timing tests with budget asserts
```

Writes `results/benchmarks.csv` and `results/machine.json` — the latter records CPU,
cores, RAM, library versions, git commit, timestamp **and load average**, because a
timing taken on a busy machine is not comparable with one taken on an idle machine.
The harness warns if the 1-minute load exceeds half the core count.

Measured on an i9-11900H (16 cores, idle, load 1.32), NSIDE 64 with 11 templates:

| Stage | Time |
|---|---|
| `OLS` | 2.9 ms |
| `ISD-1` | 28.5 ms |
| `ElasticNet` | 123 ms |
| `MCMC-add` (analytic) | 72 ms |
| `ISD-3` | 2.26 s |
| `MCMC-comb` (NUTS, 400+400) | 169 s |

Five orders of magnitude between the fastest and slowest method.

## Characterisation

```bash
python characterisation/analyze_existing_simulations.py   # break-even condition
python characterisation/run_variance_inflation.py         # how wrong the iid errors are
python characterisation/run_crossterm_bias.py             # dropped cross-terms
python characterisation/calibrate_glass_clustering.py --scan 5e-4 1e-2 3.8e-2
```

Headline results, with their caveats:

- **Break-even.** Correcting helps only when `A/ΔA > 1`, where `A` is the contaminating
  field and `ΔA` the mis-fit field the correction adds back. Rank correlation with the
  improvement factor is **+0.73** over 180 cells; the rule is right 79 % of the time.
  Below break-even, applying weights makes `w(θ)` *worse*.
- **Variance inflation.** The iid pixel error is **3–18× too tight** on a clustered
  field, giving a 3σ false-positive rate of 76–100 % where 0.27 % is expected. A
  pure-Poisson control returns κ ≈ 1, confirming the measurement. *Caveat:* κ depends
  strongly on the assumed `C_ℓ` slope (3.2 at −1.5, 13.1 at −4.0), so it must always be
  quoted with the spectrum it was measured on.
- **Cross-terms.** The auto-only two-point correction carries a 7–17 % error in the
  PCA-rotated basis the pipeline uses — but a factor **14–20** in the unrotated basis.
  The rotation is load-bearing, not just an MCMC-mixing convenience.
- **GLASS mocks are under-clustered.** The default `cl_amplitude=5e-4` gives
  σ_clus = 0.078 against LS10's 0.387 — **25× too little clustering variance**, while
  matching the shot noise exactly. Since those mocks are the null for the ISD
  p-values, the mock-calibrated LRT *and* the sandwich covariance, all three remain
  overconfident. `cl_amplitude = 3.8e-2` reproduces the LS10 field to better than 1 %.

Results are written under `results/` and written up in
[`sys_mapping`'s documentation](https://sys-mapping.readthedocs.io).

## Related repositories

- [`sys_mapping`](https://github.com/JohanComparat/sys_mapping) — the package and pipeline
- `sys_mapping_paper` — the pipeline document (private)
