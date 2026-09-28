# Architecture

`src/scripts/run_self_time.py` resolves the Hydra sweep, task catalog, one run
seed, and supported backbone. The canonical adapters in
`src/timebench/model_loading/` own foundation inference. Before inference,
`src/timebench/proposal/self_augmentations.py` builds the requested ordered
past-only channels from the visible raw and z-score-normalized target history.

The inherited TIME evaluation layer owns saved-Arrow loading, official windows,
the Seasonal-defined finite grid, metrics, and authoritative run manifests.
`src/scripts/report.py` selects completed task manifests and writes the matched
comparison below the experiment's `reports/` directory. Root submission scripts
compose concise Slurm fronts whose workflow implementations live in
`src/slurm/`.

Dependencies point inward: the proposal package depends only on NumPy and does
not import Hydra, Slurm, manifests, reporting, or plotting.
