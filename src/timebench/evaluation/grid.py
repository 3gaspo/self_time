"""Model-independent evaluation support defined by Seasonal Naive MASE."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from timebench.evaluation.metrics import seasonal_naive_scale


EVALUATION_GRID_SCHEMA = 1
EVALUATION_GRID_FILE = "evaluation_grid.npz"
EVALUATION_GRID_DEFINITION = "finite_ground_truth_and_seasonal_naive_mase"


def build_evaluation_grid(
    seasonal_quantiles: np.ndarray,
    ground_truth: np.ndarray,
    context: np.ndarray,
    seasonality: int,
    quantile_levels: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed target-step and metric-cell masks for one TIME task."""

    predictions = np.asarray(seasonal_quantiles)
    targets = np.asarray(ground_truth)
    if predictions.ndim != 5 or targets.ndim != 4:
        raise ValueError("evaluation-grid arrays have unexpected dimensions")
    if predictions.shape[:2] != targets.shape[:2] or predictions.shape[3:] != targets.shape[2:]:
        raise ValueError("Seasonal Naive predictions and ground truth do not align")
    levels = [float(value) for value in quantile_levels]
    if 0.5 not in levels:
        raise ValueError("Seasonal Naive grid construction requires a median forecast")

    target_mask = np.isfinite(targets)
    median = predictions[:, :, levels.index(0.5)]
    prediction_finite = np.all(~target_mask | np.isfinite(median), axis=-1)
    scales = seasonal_naive_scale(context, seasonality)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        absolute_error = np.where(target_mask, np.abs(targets - median), 0.0)
        target_count = target_mask.sum(axis=-1)
        mae = np.divide(
            absolute_error.sum(axis=-1, dtype=np.float64),
            target_count,
            out=np.full(target_count.shape, np.nan, dtype=np.float64),
            where=target_count > 0,
        )
        seasonal_mase = mae / scales
    evaluation_mask = prediction_finite & np.isfinite(seasonal_mase)
    return target_mask, evaluation_mask


def save_evaluation_grid(
    path: str | Path,
    target_mask: np.ndarray,
    evaluation_mask: np.ndarray,
) -> Path:
    """Write the compact shared grid once beside the Seasonal Naive result."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.asarray(EVALUATION_GRID_SCHEMA, dtype=np.int64),
            target_mask=np.asarray(target_mask, dtype=bool),
            evaluation_mask=np.asarray(evaluation_mask, dtype=bool),
        )
    temporary.replace(destination)
    return destination


def load_evaluation_grid(
    path: str | Path,
    *,
    ground_truth: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a shared grid and optionally verify its model-independent target mask."""

    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as payload:
        if int(payload["schema_version"]) != EVALUATION_GRID_SCHEMA:
            raise ValueError(f"unsupported evaluation grid schema: {source}")
        target_mask = np.asarray(payload["target_mask"], dtype=bool)
        evaluation_mask = np.asarray(payload["evaluation_mask"], dtype=bool)
    if target_mask.shape[:-1] != evaluation_mask.shape:
        raise ValueError(f"evaluation grid arrays do not align: {source}")
    if ground_truth is not None:
        expected = np.isfinite(np.asarray(ground_truth))
        if target_mask.shape != expected.shape or not np.array_equal(target_mask, expected):
            raise ValueError(
                "shared Seasonal Naive grid and current ground truth have different finite support"
            )
    return target_mask, evaluation_mask


def flatten_univariate_grid(
    target_mask: np.ndarray, evaluation_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map saver order (series, window, variate) to TIME univariate row order."""

    targets = np.asarray(target_mask, dtype=bool)
    cells = np.asarray(evaluation_mask, dtype=bool)
    return (
        targets.transpose(0, 2, 1, 3).reshape(-1, targets.shape[-1]),
        cells.transpose(0, 2, 1).reshape(-1),
    )
