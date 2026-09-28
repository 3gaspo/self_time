# Experiment catalog

## Self-augmentation comparison

The implemented experiment evaluates `chronos2` and `ts_icl` with z-score
instance normalization. It compares vanilla inference with six declared
past-only augmentation configurations: normalized square-root magnitude,
normalized sign, raw visible target, masked zero, masked one, and the sum of
normalized square-root magnitude and sign. Each augmentation list is ordered
and recorded in the run manifest.

The default configuration evaluates every task in the project TIME catalog
with seed 1. Model context lengths are 8192 for Chronos-2 and 4096 for TS-ICL.
The scheduler submits both model jobs and then a dependent report. No experiment
has yet been executed.
