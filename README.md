# Self TIME

Self TIME evaluates causal transformations of a target's visible history as
past covariates for frozen time-series foundation models. The implemented first
experiment compares the same task with and without each configured
self-augmentation while keeping the target, test windows, model alias, and
evaluation grid fixed.

## Status

| Implementation | Experiments | Next milestone |
| --- | --- | --- |
| First experiment implemented | None run | Execute the Chronos-2 and TS-ICL comparison |

No Self TIME result is available yet.

## First experiment

The supported backbones are the canonical `chronos2` and `ts_icl` aliases,
which both accept covariates. Chronos-Bolt is excluded because it cannot accept
the required past covariates. Each run uses z-score instance normalization and
one seed for every stochastic choice.

The configured matched conditions are vanilla plus:

- `sqrt_abs_x_norm`;
- `sign_x_norm`;
- the visible unnormalized target `x`;
- masked constants zero and one;
- `sqrt_abs_x_norm_plus_sign_x_norm`.

Normalized transformations are computed only from the visible normalized
history. The `x` channel uses the visible raw history. Missing positions remain
missing, no forecast target enters a covariate, and unsupported non-empty
covariates fail explicitly.

Run the complete Selena workflow with:

```bash
bash scripts/submit_self_time.sh selena
```

The launcher submits one model job per supported backbone and an `afterok`
summary job. Use `dgx` instead of `selena` for the DGX fronts.

## Outputs

Scientific tasks live below `<O>/self_augmentation/tasks/<model>/.../run_n/`
and reports below `<O>/self_augmentation/reports/`. Here `<O>` is
`outputs/dgx` for DGX/local execution, Selena's scratch output root during
execution, or `outputs/selena` after synchronization. Each
`run_n/manifest.json` is the authoritative scientific configuration and
lifecycle record.
The report reads the common baseline from the independent Seasonal checkout's
`outputs/seasonal_naive/evaluations/` tree through
`TIME_SEASONAL_EVALUATIONS_ROOT`.

Every Slurm stream, Hydra directory, stage log, and workflow status is grouped
below `logs/<surface>/self_augmentation/`. Launch IDs and timestamps remain in
manifests and logs rather than directory names.

## Source tree

- `src/timebench/model_loading/`: canonical Chronos-2 and TS-ICL adapters.
- `src/timebench/proposal/`: causal self-augmentation construction.
- `src/scripts/`: Hydra experiment and report entry points.
- `src/slurm/`: scheduler workflow implementations.
- `src/timebench/results/`: matched task aggregation and reporting.
- `src/conf/self_time.yaml`: experiment configuration.

See [architecture](docs/architecture.md),
[experiment catalog](docs/experiment_catalog.md), and
[results recap](docs/results_recap.md).
