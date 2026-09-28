"""
TS-ICL model experiments for time series forecasting.

Requires:
    tsicl

Self TIME calls :func:`run_tsicl_experiment` through its Hydra entry point.
The inference path is source-adapted from Evaluating TSFMs revision 1266509b;
Self TIME adds project-owned normalization and self-augmentation construction.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

import torch

from gluonts.time_feature import get_seasonality

from timebench.evaluation.saver import save_window_predictions
from timebench.evaluation.grid import EVALUATION_GRID_DEFINITION
from timebench.evaluation.timing import EvaluationTimer
from timebench.evaluation.utils import (
    get_available_terms,
    normalize_tsicl_quantiles,
    patch_tsicl_covariate_rollout,
)
from timebench.evaluation.normalization import normalize_instance
from timebench.evaluation.data import (
    DEFAULT_CONFIG_PATH,
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)
from timebench.paths import (
    foundation_experiment_axis,
    foundation_experiment_name,
    foundation_experiment_root,
    foundation_identity_root,
    foundation_weight_path,
)
from timebench.pipeline import (
    allocate_run,
    resolve_shared_evaluation_grid,
    resolve_target_mode,
)
from timebench.pipeline.inference_cache import (
    dependency_reference, load_raw_inference, save_raw_inference,
)
from timebench.proposal import (
    SELF_AUGMENTATIONS,
    build_past_self_covariates,
    validate_self_augmentations,
)

from tsicl import TSICL


DEFAULT_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SUPPORTS_COVARIATES = True


def run_tsicl_experiment(
    dataset_name: str,
    terms: list[str] | None = None,
    model_size: str = "tsicl-v1",
    output_dir: str | None = None,
    batch_size: int = 32,
    context_length: int = 2048,
    config_path: Path | None = None,
    quantile_levels: list[float] | None = None,
    model_path: str | Path | None = None,
    self_augmentations: list[str] | tuple[str, ...] | None = None,
    augmentation_id: str = "vanilla",
    seed: int = 1,
    target_mode: str = "auto",
    instance_normalization: str = "zscore",
):
    """ Run TS_ICL experiments."""

    self_augmentations = validate_self_augmentations(self_augmentations)
    if instance_normalization != "zscore":
        raise ValueError("Self TIME requires instance_normalization='zscore'")
    covariate_mode = "past_only" if self_augmentations else "none"

    # Set CUDA device
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    from timebench.pipeline.runtime_resources import log_selected_device
    log_selected_device(device_map, stage="forecast", model="ts_icl")

    # Load dataset configuration
    print("Loading configuration...")
    config_path = Path(config_path or DEFAULT_CONFIG_PATH).resolve()
    config = load_dataset_config(config_path)
    print(f"Dataset config: {config_path}")

    # Auto-detect available terms from config if not specified
    if terms is None:
        terms = get_available_terms(dataset_name, config)
        if not terms:
            raise ValueError(f"No terms defined for {dataset_name=} in config")

    if quantile_levels is None:
        quantile_levels = DEFAULT_QUANTILE_LEVELS

    if output_dir is None:
        output_dir = str(foundation_experiment_root())
    Path(output_dir).mkdir(exist_ok=True,parents=True)
    experiment = foundation_experiment_name()
    checkpoint_path = foundation_weight_path(
        "tsicl/tsicl-v1.ckpt",
        explicit=model_path,
        directory=False,
    )

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Terms: {terms}")
    print(f"Covariate mode: {covariate_mode}")
    print(f"Self-augmentations: {list(self_augmentations)}")
    print(f"Instance normalization: {instance_normalization}")
    print(f"{'='*60}")

    for term in terms:
        print(f"\n--- Term: {term} ---")

        # Get settings from config
        settings          = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings["prediction_length"]
        test_length       = settings["test_length"]
        val_length        = settings["val_length"]
        print(f"  Config: {prediction_length=}, {test_length=}, {val_length=}")

        # Dataset Initialization
        dataset = Dataset(
            name              = dataset_name,
            term              = term,
            to_univariate     = False,
            prediction_length = prediction_length,
            test_length       = test_length,
            val_length        = val_length,
        )
        resolved_target_mode = resolve_target_mode(
            target_mode,
            target_dim=dataset.target_dim,
            # TS-ICL moves multiple target variables into its batch dimension;
            # cross-channel mixing is available only through covariates.
            supports_multivariate=False,
        )
        if resolved_target_mode == "univariate" and dataset.target_dim > 1:
            dataset = Dataset(
                name=dataset_name,
                term=term,
                to_univariate=True,
                prediction_length=prediction_length,
                test_length=test_length,
                val_length=val_length,
            )
        season_length = get_seasonality(dataset.freq)
        covariate_channels = len(self_augmentations)
        evaluation_grid_path = resolve_shared_evaluation_grid(
            dataset_name, term, resolved_target_mode
        )
        identity_root = foundation_identity_root(
            output_dir,
            "ts_icl",
            resolved_target_mode,
            dataset_name,
            term,
            experiment_axis=foundation_experiment_axis(
                experiment,
                context_length=context_length,
                instance_normalization=instance_normalization,
            ),
        )
        identity = {
                "model": "ts_icl",
                "target_mode": resolved_target_mode,
                "dataset": dataset_name.rpartition("/")[0] or dataset_name,
                "frequency": dataset.freq,
                "term": term,
            }
        scientific_model = {
                "model_size": model_size,
                "context_length": context_length,
                "quantile_levels": quantile_levels,
                "covariate_capability": "past_only",
            }
        scientific_experiment = {
            "instance_normalization": instance_normalization,
            "augmentation_id": augmentation_id,
            "self_augmentations": list(self_augmentations),
            "seed": int(seed),
            "covariate_mode": covariate_mode,
            "covariate_channels": covariate_channels,
            "covariate_source": "target_history" if self_augmentations else "none",
            "covariate_time_span": "L" if self_augmentations else "none",
        }
        inference_root = foundation_identity_root(
            Path(output_dir).parent / "inference",
            "ts_icl", resolved_target_mode, dataset_name, term,
            experiment_axis=foundation_experiment_axis(
                experiment, context_length=context_length,
                instance_normalization=instance_normalization))
        inference_run = allocate_run(
            inference_root,
            experiment=f"{experiment}_raw_inference",
            identity=identity, model_config=scientific_model,
            pipeline_config={
                "prediction_length": prediction_length,
                "test_length": test_length,
                "windows": dataset.windows,
                "target_mode": resolved_target_mode,
                "covariate_mode": covariate_mode,
            },
            runtime_config={
                "batch_size": batch_size,
                "device": device_map,
                "checkpoint_path": str(checkpoint_path),
            },
            experiment_config=scientific_experiment,
            provenance={
                "dataset_config_path": str(config_path),
                "dataset_config_keys": ["prediction_length", "test_length", "val_length"],
            },
        )

        model_hyperparams = {"model": "ts_icl", **scientific_model,
            **scientific_experiment, "experiment": experiment,
            "target_mode": resolved_target_mode}

        def complete_evaluation(forecasts, levels, seconds):
            run = allocate_run(identity_root, experiment=experiment,
                identity=identity, model_config=scientific_model,
                pipeline_config={"prediction_length": prediction_length,
                    "test_length": test_length, "windows": dataset.windows,
                    "seasonality": season_length,
                    "nan_policy": "omit_nan_predictions_report_counts_reject_infinity",
                    "raw_inference": dependency_reference(inference_run.run_dir),
                    "evaluation_grid": {"definition": EVALUATION_GRID_DEFINITION,
                        "producer": dependency_reference(evaluation_grid_path.parent)}},
                runtime_config={"batch_size": batch_size, "device": device_map,
                    "checkpoint_path": str(checkpoint_path)},
                experiment_config=scientific_experiment,
                provenance={"dataset_config_path": str(config_path),
                    "dataset_config_keys": ["prediction_length", "test_length", "val_length"],
                    "evaluation_grid": str(evaluation_grid_path),
                    "raw_inference_manifest": str(inference_run.run_dir / "manifest.json")})
            if not run.should_run:
                return None, run
            with run:
                metadata = save_window_predictions(dataset=dataset, fc_quantiles=forecasts,
                    ds_config=f"{dataset_name}/{term}", output_base_dir=output_dir,
                    seasonality=season_length, model_hyperparams=model_hyperparams,
                    quantile_levels=levels, inference_seconds=seconds,
                    task_output_dir=str(run.run_dir), evaluation_grid_path=str(evaluation_grid_path))
                run.complete(["predictions.npz", "metrics.npz", "metrics_summary.json"], artifact_metadata={"evaluation": metadata})
            return metadata, run

        if inference_run.action == "finalize":
            inference_run.complete()
        if not inference_run.should_run:
            fc_quantiles, quantile_levels, inference_seconds = load_raw_inference(inference_run.run_dir)
            metadata, run = complete_evaluation(fc_quantiles, quantile_levels, inference_seconds)
            print(f"  Reused raw inference: {inference_run.run_dir}")
            continue

        print(f"  Initializing model pipeline...")
        model = TSICL(
            model_path=str(checkpoint_path),
            allow_auto_download=False,
        )
        patch_tsicl_covariate_rollout(model)

        # Determine split
        data_length = test_length
        num_windows = dataset.windows
        split_name = "Test split"
        eval_data = dataset.test_data

        print("  Dataset info:")
        print(f"    - Frequency: {dataset.freq}")
        print(f"    - Num series: {len(dataset.hf_dataset)}")
        print(f"    - Target dim: {dataset.target_dim}")
        print(f"    - Target mode: {resolved_target_mode}")
        print(f"    - Covariate channels: {covariate_channels}")
        print(f"    - Series length: min={dataset._min_series_length}, \
              max={dataset._max_series_length}, avg={dataset._avg_series_length:.1f}")
        print(f"    - {split_name}: {data_length} steps")
        print(f"    - Prediction length: {dataset.prediction_length}")
        print(f"    - Windows: {num_windows}")

        # Helper function to prepare a single context
        def _prepare_context(d):
            target = np.asarray(d["target"])

            # Manually truncate context
            seq_len = target.shape[-1]
            if seq_len > context_length:
                target = target[..., -context_length:]

            if target.ndim == 1:
                target = target[np.newaxis, :]

            raw_target = np.asarray(target, dtype=np.float32)
            target, normalizer = normalize_instance(raw_target, instance_normalization)
            covariates = build_past_self_covariates(
                raw_target, target, self_augmentations
            )
            context = torch.tensor(target).permute(1, 0)
            covariate_tensor = (
                None if covariates is None else torch.from_numpy(covariates).permute(1, 0)
            )
            return context, covariate_tensor, normalizer

        # Batch Inference with lazy loading
        fc_quantiles_batches = []
        timer = EvaluationTimer()
        timer.start()
        eval_items = list(eval_data)
        total_items = len(eval_items)
        for start in range(0, total_items, batch_size):
            end = min(start + batch_size, total_items)
            batch_items = eval_items[start:end]
            prepared_contexts = [
                _prepare_context(input_entry)
                for input_entry, _ in batch_items
            ]
            batch_contexts = [item[0] for item in prepared_contexts]
            batch_covariates = [item[1] for item in prepared_contexts]
            batch_normalizers = [item[2] for item in prepared_contexts]
            if not self_augmentations:
                batch_covariates = None

            if batch_covariates is None:
                with torch.no_grad():
                    _, batch_q = model.forecast(
                        inputs            = batch_contexts,
                        prediction_length = prediction_length,
                        batch_size        = batch_size,
                        quantile_levels   = quantile_levels,
                        context_length    = context_length,
                        device            = torch.device(device_map),
                        denormalize       = True,
                        squeeze_output    = False,
                    )
                # TS-ICL returns a tensor for stackable contexts and a list for
                # variable-length contexts. Normalize both documented forms.
                batch_q_array = normalize_tsicl_quantiles(batch_q)
            else:
                grouped_indices = {}
                for index, (context, covariates) in enumerate(
                    zip(batch_contexts, batch_covariates)
                ):
                    grouped_indices.setdefault(
                        (tuple(context.shape), tuple(covariates.shape)), []
                    ).append(index)

                ordered_quantiles = [None] * len(batch_contexts)
                for indices in grouped_indices.values():
                    grouped_contexts = torch.stack(
                        [batch_contexts[index] for index in indices]
                    )
                    grouped_covariates = torch.stack(
                        [batch_covariates[index] for index in indices]
                    )
                    with torch.no_grad():
                        _, group_q = model.forecast(
                            inputs=grouped_contexts,
                            covars=grouped_covariates,
                            prediction_length=prediction_length,
                            batch_size=len(indices),
                            quantile_levels=quantile_levels,
                            context_length=int(grouped_contexts.shape[1]),
                            device=torch.device(device_map),
                            denormalize=True,
                            squeeze_output=False,
                            allow_auto_complete=False,
                            allow_covar_forecast=True,
                        )
                    group_array = normalize_tsicl_quantiles(group_q)
                    for group_index, original_index in enumerate(indices):
                        ordered_quantiles[original_index] = group_array[group_index]
                batch_q_array = np.stack(ordered_quantiles, axis=0)
            for index, normalizer in enumerate(batch_normalizers):
                if normalizer is not None:
                    batch_q_array[index] = normalizer.inverse_quantiles(
                        batch_q_array[index]
                    )
            if resolved_target_mode == "univariate" and batch_q_array.ndim == 4:
                batch_q_array = batch_q_array[:, :, 0, :]
            fc_quantiles_batches.append(batch_q_array)

            # Optional progress logging
            if (start // batch_size + 1) % 10 == 0:
                print(f"    Processed {min(start + batch_size, total_items)}/{total_items}...")

        # Concatenate all batches into a single array
        # Shape: (num_total_instances, num_quantiles, num_variates, prediction_length)
        fc_quantiles = np.concatenate(fc_quantiles_batches, axis=0)
        inference_seconds = timer.stop()

        with inference_run:
            inference_run.complete(save_raw_inference(
                inference_run.run_dir, fc_quantiles, quantile_levels, inference_seconds))
        metadata, run = complete_evaluation(fc_quantiles, quantile_levels, inference_seconds)

        print(f"  Completed: {metadata["num_series"]} series × {metadata["num_windows"]} windows")
        print(f"  Output: {run.run_dir}")
        print(f"  Inference time: {inference_seconds:.3f}s")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)
    return


def main():
    parser = argparse.ArgumentParser(description="Run TS-ICL experiments")
    parser.add_argument("--dataset", type=str, nargs="+", default=["Water_Quality_Darwin/15T"],
                        help="Dataset name(s). Can be a single dataset, multiple datasets, or 'all_datasets'")
    parser.add_argument("--terms", type=str, nargs="+", default=None,
                        choices=["short", "medium", "long"],
                        help="Terms to evaluate. If not specified, auto-detect from config.")
    parser.add_argument("--model-size", type=str, default="tsicl-v1",
                            choices=["tsicl-v1"], help="TS-ICL model size")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Task root; defaults to the selected experiment's tasks directory")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for prediction")
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Quantile levels to predict",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=int(os.environ.get("TIME_CONTEXT_LENGTH", "4096")),
        help="Maximum context length",
    )
    parser.add_argument(
        "--instance-normalization",
        choices=("zscore",),
        default="zscore",
        help="Self TIME always applies per-window z-score normalization",
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to datasets.yaml config file")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Local TS-ICL checkpoint file")
    parser.add_argument("--self-augmentations", nargs="*", choices=SELF_AUGMENTATIONS,
                        default=[], help="Ordered past-only covariate channels")
    parser.add_argument("--augmentation-id", default="manual")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--target-mode",
        choices=("auto", "univariate", "multivariate"),
        default=os.environ.get("TIME_TARGET_MODE", "auto"),
        help="Target representation; auto uses native multivariate input when available",
    )

    args = parser.parse_args()

    # Handle dataset list or 'all_datasets'
    config_path = Path(args.config) if args.config else None

    if len(args.dataset) == 1 and args.dataset[0] == "all_datasets":
        config = load_dataset_config(config_path)
        datasets = list(config.get("datasets", {}).keys())
        print(f"Running all {len(datasets)} datasets from config:")
    else:
        datasets = args.dataset

    # Iterate over all datasets
    total_datasets = len(datasets)
    for idx, dataset_name in enumerate(datasets, 1):
        print(f"\n{'#'*60}")
        print(f"# Dataset {idx}/{total_datasets}: {dataset_name}")
        print(f"{'#'*60}")

        run_tsicl_experiment(
            dataset_name=dataset_name,
            terms=args.terms,
            model_size=args.model_size,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            context_length=args.context_length,
            config_path=config_path,
            quantile_levels=args.quantiles,
            model_path=args.model_path,
            self_augmentations=args.self_augmentations,
            augmentation_id=args.augmentation_id,
            seed=args.seed,
            target_mode=args.target_mode,
            instance_normalization=args.instance_normalization,
        )

    print(f"\n{'#'*60}")
    print(f"# All {total_datasets} dataset(s) completed!")
    print(f"{'#'*60}")



if __name__ == "__main__":
    main()
