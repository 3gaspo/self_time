"""
Chronos-2 model experiments for time series forecasting.

Self TIME calls :func:`run_chronos2_experiment` through its Hydra entry point.
The inference path is source-adapted from Evaluating TSFMs revision 1266509b;
Self TIME adds project-owned normalization and self-augmentation construction.
"""

import argparse
import os
import sys
import logging
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from chronos import BaseChronosPipeline
from gluonts.time_feature import get_seasonality

from timebench.evaluation.saver import save_window_predictions
from timebench.evaluation.grid import EVALUATION_GRID_DEFINITION
from timebench.evaluation.timing import EvaluationTimer
from timebench.evaluation.utils import get_available_terms
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
    dependency_reference,
    load_raw_inference,
    save_raw_inference,
)
from timebench.proposal import (
    SELF_AUGMENTATIONS,
    build_past_self_covariates,
    validate_self_augmentations,
)

# Load environment variables
load_dotenv()

logging.getLogger("chronos").setLevel(logging.ERROR)

SUPPORTS_COVARIATES = True


def run_chronos2_experiment(
    dataset_name: str,
    terms: list[str] = None,
    model_size: str = "chronos2",
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
    """
    Run Chronos-2 model experiments.
    """
    self_augmentations = validate_self_augmentations(self_augmentations)
    if instance_normalization != "zscore":
        raise ValueError("Self TIME requires instance_normalization='zscore'")
    covariate_mode = "past_only" if self_augmentations else "none"

    # Set CUDA device
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    from timebench.pipeline.runtime_resources import log_selected_device
    log_selected_device(device_map, stage="forecast", model="chronos2")

    # Load dataset configuration
    print("Loading configuration...")
    config_path = Path(config_path or DEFAULT_CONFIG_PATH).resolve()
    config = load_dataset_config(config_path)
    print(f"Dataset config: {config_path}")

    # Auto-detect available terms from config if not specified
    if terms is None:
        terms = get_available_terms(dataset_name, config)
        if not terms:
            raise ValueError(f"No terms defined for dataset '{dataset_name}' in config")

    if quantile_levels is None:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    if output_dir is None:
        output_dir = str(foundation_experiment_root())

    os.makedirs(output_dir, exist_ok=True)
    experiment = foundation_experiment_name()

    if model_size != "chronos2":
        raise ValueError(f"Unsupported Chronos-2 model size: {model_size}")
    checkpoint_path = foundation_weight_path(
        "chronos2",
        explicit=model_path,
        directory=True,
    )

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print("Model: amazon/chronos-2")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Terms: {terms}")
    print(f"Covariate mode: {covariate_mode}")
    print(f"Self-augmentations: {list(self_augmentations)}")
    print(f"Instance normalization: {instance_normalization}")
    print(f"{'='*60}")

    for term in terms:
        print(f"\n--- Term: {term} ---")

        # Get settings from config
        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings.get("prediction_length")
        test_length = settings.get("test_length")
        val_length = settings.get("val_length")

        print(f"  Config: prediction_length={prediction_length}, test_length={test_length}, val_length={val_length}")

        # Dataset Initialization
        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=False,
            prediction_length=prediction_length,
            test_length=test_length,
            val_length=val_length,
        )
        resolved_target_mode = resolve_target_mode(
            target_mode,
            target_dim=dataset.target_dim,
            supports_multivariate=True,
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
        target_channels = 1 if resolved_target_mode == "univariate" else dataset.target_dim
        covariate_channels = len(self_augmentations) * target_channels
        evaluation_grid_path = resolve_shared_evaluation_grid(
            dataset_name, term, resolved_target_mode
        )
        identity_root = foundation_identity_root(
            output_dir,
            "chronos2",
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
            "model": "chronos2",
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
            "chronos2", resolved_target_mode, dataset_name, term,
            experiment_axis=foundation_experiment_axis(
                experiment, context_length=context_length,
                instance_normalization=instance_normalization,
            ),
        )
        inference_run = allocate_run(
            inference_root,
            experiment=f"{experiment}_raw_inference",
            identity=identity,
            model_config=scientific_model,
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

        model_hyperparams = {"model": "chronos2", **scientific_model, **scientific_experiment,
            "experiment": experiment, "target_mode": resolved_target_mode}

        def complete_evaluation(forecasts, levels, seconds):
            run = allocate_run(
            identity_root, experiment=experiment,
            identity=identity,
            model_config=scientific_model,
            pipeline_config={
                "prediction_length": prediction_length,
                "test_length": test_length,
                "windows": dataset.windows,
                "seasonality": season_length,
                "nan_policy": "omit_nan_predictions_report_counts_reject_infinity",
                "raw_inference": dependency_reference(inference_run.run_dir),
                "evaluation_grid": {"definition": EVALUATION_GRID_DEFINITION,
                    "producer": dependency_reference(evaluation_grid_path.parent)},
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
                "evaluation_grid": str(evaluation_grid_path),
                "raw_inference_manifest": str(inference_run.run_dir / "manifest.json"),
            },
            )
            if not run.should_run:
                return None, run
            with run:
                metadata = save_window_predictions(
                    dataset=dataset, fc_quantiles=forecasts, ds_config=f"{dataset_name}/{term}",
                    output_base_dir=output_dir, seasonality=season_length,
                    model_hyperparams=model_hyperparams, quantile_levels=levels,
                    inference_seconds=seconds, task_output_dir=str(run.run_dir),
                    evaluation_grid_path=str(evaluation_grid_path),
                )
                run.complete(["predictions.npz", "metrics.npz", "metrics_summary.json"], artifact_metadata={"evaluation": metadata})
            return metadata, run

        if inference_run.action == "finalize":
            inference_run.complete()
        if not inference_run.should_run:
            fc_quantiles, quantile_levels, inference_seconds = load_raw_inference(inference_run.run_dir)
            metadata, run = complete_evaluation(fc_quantiles, quantile_levels, inference_seconds)
            print(f"  Reused raw inference: {inference_run.run_dir}")
            if metadata is not None:
                print(f"  Completed: {metadata['num_series']} series × {metadata['num_windows']} windows")
            continue

        # Initialize Chronos only after the dataset capability check.
        print(f"  Initializing Chronos pipeline ({checkpoint_path})...")
        pipeline = BaseChronosPipeline.from_pretrained(
            str(checkpoint_path),
            device_map=device_map,
            local_files_only=True,
        )

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
        print(f"    - Series length: min={dataset._min_series_length}, max={dataset._max_series_length}, avg={dataset._avg_series_length:.1f}")
        print(f"    - {split_name}: {data_length} steps")
        print(f"    - Prediction length: {dataset.prediction_length}")
        print(f"    - Windows: {num_windows}")

        timer = EvaluationTimer()
        timer.start()

        # ---------------------------------------------------------
        # 1. Running Inference (Chronos-2 Specific Logic)
        # ---------------------------------------------------------
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
            target_tensor = torch.tensor(target)
            covariates = build_past_self_covariates(
                raw_target, target, self_augmentations
            )
            if covariates is None:
                return target_tensor, normalizer
            item = {
                "target": target_tensor,
                "past_covariates": {
                    f"covariate_{channel}": torch.from_numpy(values)
                    for channel, values in enumerate(covariates)
                },
            }
            return item, normalizer

        # Batch Inference with lazy loading
        fc_quantiles_batches = []
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
            batch_normalizers = [item[1] for item in prepared_contexts]

            # Filter out verbose warnings from Chronos-2 during prediction to keep output clean
            class ContentFilterStderr:
                def __init__(self, original_stream):
                    self.original_stream = original_stream

                def write(self, data):
                    if "Quantiles to be predicted" in data and "Chronos-2" in data:
                        return
                    self.original_stream.write(data)

                def flush(self):
                    self.original_stream.flush()

            original_stderr = sys.stderr
            sys.stderr = ContentFilterStderr(original_stderr)
            try:
                with torch.no_grad():
                    batch_q, batch_m = pipeline.predict_quantiles(
                        inputs=batch_contexts,
                        prediction_length=prediction_length,
                        quantile_levels=quantile_levels,
                    )
            finally:
                sys.stderr = original_stderr

            batch_quantiles_list = []
            for q, normalizer in zip(batch_q, batch_normalizers):
                if isinstance(q, torch.Tensor):
                    if q.ndim == 3 and q.shape[-1] == len(quantile_levels):
                        # Shape: (num_variates, pred_len, num_quantiles) -> (num_quantiles, num_variates, pred_len)
                        q = q.permute(2, 0, 1)
                    q = q.cpu().float().numpy()
                if normalizer is not None:
                    q = normalizer.inverse_quantiles(q)
                # q shape: (num_quantiles, num_variates, prediction_length)
                # Add batch dimension: (1, num_quantiles, num_variates, prediction_length)
                batch_quantiles_list.append(q[np.newaxis, ...])


            # Stack into batch: (batch_size, num_quantiles, num_variates, prediction_length)
            batch_q_array = np.concatenate(batch_quantiles_list, axis=0)
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
                inference_run.run_dir, fc_quantiles, quantile_levels, inference_seconds
            ))
        metadata, run = complete_evaluation(fc_quantiles, quantile_levels, inference_seconds)

        if metadata is not None:
            print(f"  Completed: {metadata['num_series']} series × {metadata['num_windows']} windows")
        print(f"  Output: {run.run_dir}")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run Chronos experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["Global_Influenza/W"],
        help=(
            "Dataset names, all_datasets, or all_multivariate_datasets "
            "for the shared D>1 comparison subset"
        ),
    )
    parser.add_argument("--terms", type=str, nargs="+", default=None,
                        choices=["short", "medium", "long"],
                        help="Terms to evaluate. If not specified, auto-detect from config.")
    parser.add_argument("--model-size", type=str, default="chronos2",
                        help="Chronos model size (use 'chronos2' for amazon/chronos-2)")
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
        "--context-length", type=int,
        default=int(os.environ.get("TIME_CONTEXT_LENGTH", "8192")),
                        help="Maximum context length")
    parser.add_argument(
        "--instance-normalization",
        choices=("zscore",),
        default="zscore",
        help="Self TIME always applies per-window, per-variate z-score normalization",
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to datasets.yaml config file")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Local Chronos-2 checkpoint directory")
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

    if len(args.dataset) == 1 and args.dataset[0] in {
        "all_datasets",
        "all_multivariate_datasets",
    }:
        # Load all datasets from config
        config = load_dataset_config(config_path)
        datasets = list(config.get("datasets", {}).keys())
        if args.dataset[0] == "all_multivariate_datasets":
            datasets = [
                name
                for name in datasets
                if Dataset(name=name).target_dim > 1
            ]
        print(f"Running all {len(datasets)} datasets from config:")
        for ds in datasets:
            print(f"  - {ds}")
    else:
        datasets = args.dataset

    # Iterate over all datasets with progress logging
    total_datasets = len(datasets)
    for idx, dataset_name in enumerate(datasets, 1):
        print(f"\n{'#'*60}")
        print(f"# Dataset {idx}/{total_datasets}: {dataset_name}")
        print(f"{'#'*60}")

        run_chronos2_experiment(
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
