"""
Utility functions.
"""
import yaml
import numpy as np


def patch_tsicl_covariate_rollout(pipeline):
    """Apply TS-ICL's upstream long-horizon covariate rollout correction.

    The published ``tsicl==0.2.1`` release concatenates rollout covariates on
    the feature axis and slices later horizon chunks from the wrong offset.
    This is the corrected upstream implementation, limited to long forecasts
    with covariates; every other call keeps the installed package behavior.
    """
    from types import MethodType

    import torch

    if getattr(pipeline, "_timebench_covariate_rollout_fixed", False):
        return pipeline
    original_rollout = pipeline._rollout_f

    def corrected_rollout(
        self,
        grid,
        series_c,
        covariates,
        has_covar,
        prediction_length,
        denormalize,
        allow_auto_complete=False,
        allow_covar_forecast=False,
    ):
        if (
            prediction_length <= self.max_target_length
            or not isinstance(covariates, torch.Tensor)
        ):
            return original_rollout(
                grid=grid,
                series_c=series_c,
                covariates=covariates,
                has_covar=has_covar,
                prediction_length=prediction_length,
                denormalize=denormalize,
                allow_auto_complete=allow_auto_complete,
                allow_covar_forecast=allow_covar_forecast,
            )

        rollouts = (
            prediction_length // self.max_target_length
            + int(prediction_length % self.max_target_length > 0)
        )
        covariate_length = covariates.shape[-2]
        covariates_cover_future = covariate_length > series_c.shape[-2]
        if covariates_cover_future:
            expected = prediction_length + series_c.shape[-2]
            if covariate_length != expected:
                raise ValueError(
                    f"Covariate sequence length {covariate_length} does not match "
                    f"context+horizon length {expected}"
                )
        covariate_past = covariates[..., : series_c.shape[-2], :]
        if covariates_cover_future:
            covariate_horizon = covariates[..., series_c.shape[-2] :, :]
        else:
            covariate_horizon = torch.full(
                (*covariates.shape[:-2], prediction_length, 1),
                torch.nan,
                device=covariates.device,
                dtype=covariates.dtype,
            )

        forecasts = []
        for rollout_index in range(rollouts):
            start = rollout_index * self.max_target_length
            chunk_length = min(
                self.max_target_length, prediction_length - start
            )
            if covariates_cover_future:
                chunk_covariates = torch.cat(
                    [
                        covariate_past,
                        covariate_horizon[..., start : start + chunk_length, :],
                    ],
                    dim=-2,
                )
            else:
                chunk_covariates = covariate_past
            chunk = self._run_forward(
                grid=grid,
                series_c=series_c,
                covariates=chunk_covariates,
                has_covar=has_covar,
                prediction_length=chunk_length,
                setting="forecasting",
                denormalize=True,
                save_scaler=rollout_index == 0,
                allow_auto_complete=allow_auto_complete,
                allow_covar_forecast=allow_covar_forecast,
            )
            forecasts.append(chunk)
            series_c = torch.cat(
                [series_c, chunk.mean(dim=-1, keepdim=True)], dim=1
            )
            covariate_past = torch.cat(
                [
                    covariate_past,
                    covariate_horizon[..., start : start + chunk_length, :],
                ],
                dim=-2,
            )

        result = torch.cat(forecasts, dim=1)
        return result if denormalize else self.scaler.transform(result)

    pipeline._rollout_f = MethodType(corrected_rollout, pipeline)
    pipeline._timebench_covariate_rollout_fixed = True
    return pipeline

def get_available_terms(dataset_name: str, config: dict) -> list[str]:
    """Get the terms that are actually defined in the config for a dataset."""
    datasets_config = config.get("datasets", {})
    if dataset_name not in datasets_config:
        return []
    dataset_config = datasets_config[dataset_name]
    available_terms = []
    for term in ["short", "medium", "long"]:
        if term in dataset_config and dataset_config[term].get("prediction_length") is not None:
            available_terms.append(term)
    return available_terms


def normalize_tsicl_quantiles(quantiles) -> np.ndarray:
    """Convert either documented TS-ICL forecast form to TIME's batch layout.

    TS-ICL returns one tensor for stackable contexts and a list of tensors for
    variable-length contexts. Tensor output is ``(batch, variate, horizon,
    quantile)``; each list item is ``(variate, horizon, quantile)``. TIME uses
    ``(batch, quantile, variate, horizon)`` for both cases.
    """

    def to_numpy(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return np.asarray(value)

    if isinstance(quantiles, list):
        if not quantiles:
            raise ValueError("TS-ICL returned an empty quantile list")
        normalized = []
        for index, item in enumerate(quantiles):
            array = to_numpy(item)
            if array.ndim != 3:
                raise ValueError(
                    "TS-ICL list quantiles must have shape "
                    f"(variate, horizon, quantile); item {index} has {array.shape}"
                )
            normalized.append(array.transpose(2, 0, 1))
        try:
            return np.stack(normalized, axis=0)
        except ValueError as error:
            raise ValueError("TS-ICL list quantiles have inconsistent output shapes") from error

    array = to_numpy(quantiles)
    if array.ndim != 4:
        raise ValueError(
            "TS-ICL tensor quantiles must have shape "
            f"(batch, variate, horizon, quantile); received {array.shape}"
        )
    return array.transpose(0, 3, 1, 2)


def impute_nans_1d(series: np.ndarray) -> np.ndarray:
    series = series.astype(np.float32, copy=False)
    if not np.isnan(series).any():
        return series
    idx = np.arange(series.shape[0])
    mask = np.isfinite(series)
    if mask.sum() == 0:
        return np.nan_to_num(series, nan=0.0)
    series[~mask] = np.interp(idx[~mask], idx[mask], series[mask])
    return series


def clean_nan_target(series: np.ndarray) -> np.ndarray:
    if series.ndim == 1:
        return impute_nans_1d(series)
    if series.ndim == 2:
        cleaned = np.empty_like(series, dtype=np.float32)
        for i in range(series.shape[0]):
            cleaned[i] = impute_nans_1d(series[i])
        return cleaned
    return np.nan_to_num(series, nan=0.0)


def parse_dataset_key(dataset_key: str) -> tuple[str, str]:
    """
    Parse dataset key format '{dataset}/{freq}' in datasets.yaml

    Args:
        dataset_key: e.g., 'exchange_rate/D', 'ETTh1/H', 'bitbrains_rnd/5T'

    Returns:
        (dataset_name, freq): e.g., ('exchange_rate', 'D')
    """
    parts = dataset_key.split('/')
    if len(parts) != 2:
        raise ValueError(f"Invalid dataset key format: {dataset_key}. Expected 'dataset/freq'")
    return parts[0], parts[1]


def find_dataset_config(datasets_config: dict, dataset_key: str) -> tuple[str, str, dict]:
    """
    Find dataset configuration in datasets.yaml

    Args:
        datasets_config: 'datasets' dictionary in datasets.yaml
        dataset_key: dataset key, format as '{dataset_name}/{freq}' (e.g., 'IMOS/15T')

    Returns:
        (dataset_key, freq, config)
    """
    if dataset_key in datasets_config:
        name, freq = parse_dataset_key(dataset_key)
        return dataset_key, freq, datasets_config[dataset_key]

    # Compatible with old case of only passing dataset_name
    for key, config in datasets_config.items():
        name, freq = parse_dataset_key(key)
        if name == dataset_key:
            return key, freq, config

    raise ValueError(f"Dataset '{dataset_key}' not found in config")


def load_datasets_config(config_path: str) -> dict:
    """Load datasets.yaml configuration file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_test_length(dataset_config: dict) -> int | None:
    """
    Get test_length value for a dataset

    Args:
        dataset_config: dataset specific configuration

    Returns:
        test_length value, if not configured, return None
    """
    if dataset_config and "test_length" in dataset_config:
        return dataset_config["test_length"]
    return None
