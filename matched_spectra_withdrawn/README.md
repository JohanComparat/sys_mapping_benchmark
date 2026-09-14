# Withdrawn matched spectra

The NSIDE 128 fits from campaign `20260907m`. Each file's `validation.passed` is true,
but it was written by a density gate measured on the NSIDE 128 map itself, where the
sparser LS10 samples hold one to six galaxies per pixel. Measured on a map of at most
NSIDE 64 their mean density errs by about 7% against a tolerance of 3.3%, so no null is
drawn from them. Their large-scale power ratios, 0.945 to 1.013, pass, and they are used
for the clustering-only comparison against Uchuu.

The directory sits outside `matched_spectra/` so that `load_matched_cl`, which globs that
directory, cannot select them.
