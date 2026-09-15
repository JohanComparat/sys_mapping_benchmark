# Data behind docs/results_algorithm_characterisation.rst

Each script reads pulled campaign results under `results/campaign/` (and, for the
cross-term share, the LS10 products and catalogues) and writes to `results/docs_exports/`.
The files copied to `sys_mapping/docs/_static/characterisation/` come from here.
Set `SYS_MAPPING_ROOT` when the package checkout is not `~/software/sys_mapping`.

| Script | Output | Page section |
|---|---|---|
| `breakeven_join.py`, then `breakeven_export.py` | `breakeven_20260907c_joined.csv`, `breakeven_20260907c_summary.json` | when correcting helps (campaign F, 20260907c) |
| `null_spectra_export.py` | `null_spectra_nside32_64.csv` | parametric against matched GLASS nulls (campaigns B and M) |
| `crossterm_footprint.py` | `crossterm_share_footprint_20260912p.json` | cross-term share of the correction, nine samples, six methods |

`../run_crossterm_bias.py --rand-file` computes the cross-term share of one sample in the
same basis.
