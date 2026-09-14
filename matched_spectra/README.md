# Matched null spectra

One validated angular power spectrum per LS10 sample and resolution, the input every
GLASS null is drawn from. `run_ls10_analysis.py` refuses to build a null without one:
the default power law under-clusters these samples about 25 times in variance, so a
null built on it is too narrow and its p-values are anticonservative.

Each file is a `*_match.json` written by `characterisation/match_glass_to_data.py`: a
spectrum fitted band by band to the sample's own measured clustering on its footprint,
then validated on seeds the fit never saw. Only files whose `validation.passed` is true
are kept here. `load_matched_cl` reads the directory directly.

## Coverage

| Sample, log M* ≥ | NSIDE 32 | NSIDE 64 | NSIDE 128 |
|---|---|---|---|
| 9.0, 9.5 | yes | yes | from NSIDE 64 |
| 10.0 to 11.25 | yes | yes | yes |
| 11.5 | yes | from NSIDE 128 | yes |

Three cells have no validated spectrum at their own resolution, and `load_matched_cl`
serves the finest validated one for the sample instead. The spectrum belongs to the
sample and its footprint; resolution limits what can be checked, not what can be used.
A coarser spectrum used at a finer resolution is extended flat past its last measured
multipole by `sanitise_cl`, and a finer one is truncated.

## Provenance

`PROVENANCE.tsv` names the campaign tag each file came from and its validated
large-scale ratio. Where several tags validated the same cell, the curated `validated`
set is preferred, then the most recent tag.
