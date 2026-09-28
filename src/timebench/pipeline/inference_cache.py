"""Raw test-forecast caches independent of evaluation-grid reductions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .runs import manifest_reference


def dependency_reference(run_dir: str | Path) -> dict:
    return manifest_reference(run_dir)


def save_raw_inference(run_dir: str | Path, forecasts: np.ndarray,
    quantile_levels, inference_seconds: float) -> list[str]:
    destination = Path(run_dir)
    np.savez_compressed(destination / "raw_predictions.npz",
        forecasts=np.asarray(forecasts),
        quantile_levels=np.asarray(quantile_levels, dtype=np.float64))
    (destination / "inference.json").write_text(json.dumps({
        "schema_version": 1,
        "inference_seconds": float(inference_seconds),
        "forecast_shape": list(np.asarray(forecasts).shape),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ["raw_predictions.npz", "inference.json"]


def load_raw_inference(run_dir: str | Path):
    source = Path(run_dir)
    with np.load(source / "raw_predictions.npz", allow_pickle=False) as payload:
        forecasts = payload["forecasts"]
        quantile_levels = payload["quantile_levels"].astype(float).tolist()
    metadata = json.loads((source / "inference.json").read_text(encoding="utf-8"))
    return forecasts, quantile_levels, float(metadata["inference_seconds"])
