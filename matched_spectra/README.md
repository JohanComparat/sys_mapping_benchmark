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

| Sample, log M* ≥ | NSIDE 32 | NSIDE 64 |
|---|---|---|
| 9.0 to 11.25 | yes | yes |
| 11.5 | yes | failed its large-scale check |

`load_matched_cl` serves the exact resolution when it passed, otherwise the validated fit
nearest at or above the requested resolution, otherwise the finest below. So an NSIDE 128
analysis uses the sample's NSIDE 64 fit, extended flat past its last multipole by
`sanitise_cl`, NSIDE 16 uses NSIDE 32, and log M* ≥ 11.5 at NSIDE 64 uses NSIDE 32. The
spectrum belongs to the sample and its footprint; resolution limits what can be checked,
not what can be used.

The seven NSIDE 128 fits are in `../matched_spectra_withdrawn/`. Their large-scale power
passes (ratios 0.945 to 1.013), but their `validation.passed` was written by a density
gate evaluated on the NSIDE 128 map, where the sparse samples hold one to six galaxies
per pixel and the density ratio is mostly shot noise. Measured on a map of at most NSIDE
64 their density errs by about 7% against a 3.3% tolerance, so they are not used for
nulls. They remain usable for clustering-only comparisons.

## Provenance

`PROVENANCE.tsv` names the campaign tag each file came from and its validated
large-scale ratio. Where several tags validated the same cell, the curated `validated`
set is preferred, then the most recent tag.
