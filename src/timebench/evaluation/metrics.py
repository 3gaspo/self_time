"""
Per-window metrics computation for time series forecasting evaluation.
Aligned with GluonTS implementation.

Supported metrics:
- MSE: Mean Squared Error (using median forecast, aligned with GluonTS MSE[0.5])
- MAE: Mean Absolute Error (using median forecast)
- RMSE: Root Mean Squared Error (using median forecast)
- MAPE: Mean Absolute Percentage Error (using median forecast, returns fraction)
- sMAPE: Symmetric Mean Absolute Percentage Error (using median forecast, range [0, 2])
- MASE: Mean Absolute Scaled Error (using median forecast)
- ND: Normalized Deviation (using median forecast)
- CRPS: Continuous Ranked Probability Score (MeanWeightedSumQuantileLoss)

"""

import numpy as np


def summarize_metric_values(values: np.ndarray, evaluation_values: int) -> dict:
    """Aggregate finite metric cells; dispersion is population (ddof=0)."""
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return {
        "mean": float(np.mean(finite)) if finite.size else None,
        "std": float(np.std(finite, dtype=np.float64, ddof=0)) if finite.size else None,
        "variance": float(np.var(finite, dtype=np.float64, ddof=0)) if finite.size else None,
        "dispersion_ddof": 0,
        "finite_values": int(finite.size),
        "evaluation_values": int(evaluation_values),
        "total_values": int(values.size),
    }


def fill_missing_history(context: np.ndarray) -> np.ndarray:
    """Apply the maintained TIME forward-fill policy along the time axis."""

    values = np.asarray(context).copy()
    for history in values.reshape(-1, values.shape[-1]):
        if not np.isnan(history).any():
            continue
        missing = np.isnan(history)
        indexes = np.where(~missing, np.arange(len(missing)), 0)
        np.maximum.accumulate(indexes, out=indexes)
        history[:] = history[indexes]
        if np.isnan(history).any():
            observed = history[~np.isnan(history)]
            first_observed = observed[0] if len(observed) else 0
            history[:] = np.nan_to_num(history, nan=first_observed)
    return values


def seasonal_naive_point_forecast(
    context: np.ndarray,
    prediction_length: int,
    seasonality: int,
) -> np.ndarray:
    """Repeat the last season after Improved's missing-history preparation."""

    values = fill_missing_history(context)
    period = min(int(seasonality), values.shape[-1])
    if period <= 0:
        raise ValueError("seasonality and context length must be positive")
    repeats = int(np.ceil(int(prediction_length) / period))
    return np.tile(values[..., -period:], (*([1] * (values.ndim - 1)), repeats))[
        ..., : int(prediction_length)
    ]


def seasonal_naive_scale(
    context: np.ndarray,
    seasonality: int,
    *,
    squared: bool = False,
) -> np.ndarray:
    """Return a finite-pair seasonal scale without collapsing missing dates.

    The final axis is time. A seasonal difference contributes only when both
    observations exist at their original timestamps. ``squared=False`` returns
    the MASE denominator; ``squared=True`` returns its RMS analogue for MSSE.
    """

    values = np.asarray(context, dtype=np.float64)
    period = int(seasonality)
    if period <= 0:
        raise ValueError("seasonality must be positive")
    result_shape = values.shape[:-1]
    if values.shape[-1] <= period:
        return np.full(result_shape, np.nan, dtype=np.float64)
    left = values[..., :-period]
    right = values[..., period:]
    valid = np.isfinite(left) & np.isfinite(right)
    differences = np.where(valid, right - left, 0.0)
    contributions = np.square(differences) if squared else np.abs(differences)
    counts = valid.sum(axis=-1)
    mean = np.divide(
        contributions.sum(axis=-1, dtype=np.float64),
        counts,
        out=np.full(result_shape, np.nan, dtype=np.float64),
        where=counts > 0,
    )
    scale = np.sqrt(mean) if squared else mean
    return np.where(scale > 0, scale, np.nan)


# Default quantile levels for CRPS computation (aligned with GluonTS)
DEFAULT_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def compute_per_window_metrics_from_quantiles(
    predictions_quantiles: np.ndarray,
    ground_truth: np.ndarray,
    context: np.ndarray,
    seasonality: int = 1,
    quantile_levels: list[float] = None,
    target_mask: np.ndarray | None = None,
    evaluation_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Compute evaluation metrics for each prediction window from quantile forecasts.

    Args:
        predictions_quantiles: Quantile forecasts with shape
            (num_series, num_windows, num_quantiles, num_variates, pred_len)
        ground_truth: Ground truth values with shape
            (num_series, num_windows, num_variates, pred_len)
        context: Historical context with shape
            (num_series, num_windows, num_variates, max_ctx_len)
            Note: Shorter contexts are NaN-padded
        seasonality: Seasonal period length for MASE computation
        quantile_levels: Quantile levels corresponding to predictions_quantiles.
            Defaults to [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    Returns:
        Dictionary of metric arrays, each with shape (num_series, num_windows, num_variates)
        Keys: MSE, MAE, RMSE, MAPE, sMAPE, MASE, ND, CRPS
    """
    if quantile_levels is None:
        quantile_levels = DEFAULT_QUANTILE_LEVELS

    (
        num_series,
        num_windows,
        num_quantiles,
        num_variates,
        pred_len,
    ) = predictions_quantiles.shape

    if len(quantile_levels) != num_quantiles:
        raise ValueError(
            "quantile_levels length must match predictions_quantiles' quantile dimension"
        )
    if 0.5 not in quantile_levels:
        raise ValueError("quantile_levels must include 0.5 for median-based metrics")
    median_idx = quantile_levels.index(0.5)

    target_mask = (
        np.isfinite(ground_truth)
        if target_mask is None
        else np.asarray(target_mask, dtype=bool)
    )
    if target_mask.shape != ground_truth.shape:
        raise ValueError("target_mask must have the same shape as ground_truth")
    if np.any(target_mask & ~np.isfinite(ground_truth)):
        raise ValueError("target_mask includes non-finite ground-truth values")
    metric_shape = (num_series, num_windows, num_variates)
    evaluation_mask = (
        target_mask.any(axis=-1)
        if evaluation_mask is None
        else np.asarray(evaluation_mask, dtype=bool)
    )
    if evaluation_mask.shape != metric_shape:
        raise ValueError("evaluation_mask has the wrong series/window/variate shape")
    if np.any(evaluation_mask & ~target_mask.any(axis=-1)):
        raise ValueError("evaluation_mask includes cells without finite ground truth")

    # Cells outside the shared grid remain absent for every method and metric.
    mse = np.full(metric_shape, np.nan)
    mae = np.full(metric_shape, np.nan)
    rmse = np.full(metric_shape, np.nan)
    mape = np.full(metric_shape, np.nan)
    smape = np.full(metric_shape, np.nan)
    mase = np.full(metric_shape, np.nan)
    nd = np.full(metric_shape, np.nan)

    # CRPS (Continuous Ranked Probability Score) - using weighted quantile loss
    crps = np.full(metric_shape, np.nan)

    for s in range(num_series):
        for w in range(num_windows):
            for v in range(num_variates):
                if not evaluation_mask[s, w, v]:
                    continue
                q_preds = predictions_quantiles[s, w, :, v]  # (num_quantiles, pred_len)
                gt = ground_truth[s, w, v]  # (pred_len,)
                ctx = context[s, w, v]  # (max_ctx_len,)

                # Compute median (0.5 quantile) forecast - used for most metrics
                median_pred = q_preds[median_idx]  # (pred_len,)

                valid_mask = target_mask[s, w, v]

                # Filter to valid timesteps only
                gt = np.asarray(gt[valid_mask], dtype=np.float64)
                median_pred = np.asarray(median_pred[valid_mask], dtype=np.float64)
                q_preds = np.asarray(
                    q_preds[:, valid_mask], dtype=np.float64
                )  # (num_quantiles, valid_len)
                if not np.isfinite(q_preds).all():
                    raise ValueError(
                        "non-finite forecast on shared evaluation grid at "
                        f"series={s}, window={w}, variate={v}"
                    )

                # Compute error using median forecast
                error = gt - median_pred
                abs_error = np.abs(error)

                # MSE (using median forecast, aligned with GluonTS MSE[0.5])
                mse[s, w, v] = np.mean(error ** 2)

                # MAE (using median forecast)
                mae[s, w, v] = np.mean(abs_error)

                # RMSE (sqrt of MSE)
                rmse[s, w, v] = np.sqrt(mse[s, w, v])

                # MAPE (using median forecast, returns fraction not percentage)
                nonzero_target = np.abs(gt) > 0
                if np.any(nonzero_target):
                    mape[s, w, v] = np.mean(
                        abs_error[nonzero_target] / np.abs(gt[nonzero_target])
                    )

                # sMAPE (using median forecast, range [0, 2])
                smape_denominator = np.abs(gt) + np.abs(median_pred)
                smape_vals = np.divide(
                    2 * abs_error,
                    smape_denominator,
                    out=np.zeros_like(abs_error),
                    where=smape_denominator > 0,
                )
                smape[s, w, v] = np.mean(smape_vals)

                # MASE (Mean Absolute Scaled Error, using median forecast)
                seasonal_error = seasonal_naive_scale(ctx, seasonality)
                mase[s, w, v] = mae[s, w, v] / seasonal_error

                # ND (Normalized Deviation, using median forecast)
                abs_label_sum = np.sum(np.abs(gt))
                if abs_label_sum > 0:
                    nd[s, w, v] = np.sum(abs_error) / abs_label_sum
                else:
                    nd[s, w, v] = np.nan

                # CRPS (MeanWeightedSumQuantileLoss) from provided quantiles
                if abs_label_sum > 0:
                    weighted_quantile_losses = []
                    for q, q_pred in zip(quantile_levels, q_preds):
                        q_error = gt - q_pred
                        indicator = (q_pred >= gt).astype(float)
                        q_loss = 2 * np.abs(q_error * (indicator - q))
                        weighted_ql = np.sum(q_loss) / abs_label_sum
                        weighted_quantile_losses.append(weighted_ql)
                    crps[s, w, v] = np.mean(weighted_quantile_losses)
                else:
                    crps[s, w, v] = np.nan

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "sMAPE": smape,
        "MASE": mase,
        "ND": nd,
        "CRPS": crps,
    }
