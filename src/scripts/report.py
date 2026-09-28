"""Build Self TIME MASE, inference-time, and heatmap report artifacts."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _foundation_reporter():
    path = PROJECT_ROOT / "scripts" / "compute_foundation_summary.py"
    spec = importlib.util.spec_from_file_location("self_time_foundation_reporter", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load inherited reporter from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_summary(rows: list[dict], path: Path) -> None:
    fields = [
        "base_model",
        "augmentation_id",
        "self_augmentations",
        "scaled_MASE",
        "inference_seconds",
        "datasets",
        "tasks",
        "timed_tasks",
        "prediction_nan_values",
        "prediction_values",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)

    lines = [
        "# Self TIME summary",
        "",
        "MASE is divided task-by-task by the matching Seasonal Naive MASE and "
        "combined with the TIME geometric mean. Inference seconds are summed "
        "only when every selected task has timing metadata.",
        "",
        "| Backbone | Augmentation set | Channels | Scaled MASE | Inference seconds | Tasks |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        mase = row.get("scaled_MASE")
        seconds = row.get("inference_seconds")
        lines.append(
            f"| {row['base_model']} | {row['augmentation_id']} | "
            f"`{row['self_augmentations']}` | "
            f"{'' if mase is None else f'{mase:.6f}'} | "
            f"{'' if seconds is None else f'{seconds:.3f}'} | {row['tasks']} |"
        )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_matrix(rows: list[dict], value: str, destination: Path, label: str) -> list[Path]:
    import matplotlib.pyplot as plt

    models = list(dict.fromkeys(row["base_model"] for row in rows))
    augmentations = list(dict.fromkeys(row["augmentation_id"] for row in rows))
    lookup = {(row["augmentation_id"], row["base_model"]): row.get(value) for row in rows}
    matrix = np.array(
        [[lookup.get((augmentation, model), np.nan) for model in models]
         for augmentation in augmentations],
        dtype=float,
    )

    csv_path = destination.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["augmentation_id", *models])
        for augmentation, values in zip(augmentations, matrix):
            writer.writerow([augmentation, *values])

    finite = matrix[np.isfinite(matrix)]
    figure, axis = plt.subplots(
        figsize=(max(5.5, 1.8 * len(models)), max(4.0, 0.6 * len(augmentations))),
        constrained_layout=True,
    )
    try:
        masked = np.ma.masked_invalid(matrix)
        cmap = plt.get_cmap("YlOrRd").copy()
        cmap.set_bad("#dddddd")
        artist = axis.imshow(masked, aspect="auto", cmap=cmap)
        axis.set_xticks(range(len(models)), labels=models)
        axis.set_yticks(range(len(augmentations)), labels=augmentations)
        axis.set_xlabel("Foundation backbone")
        axis.set_ylabel("Self-augmentation configuration")
        for (row, column), number in np.ndenumerate(matrix):
            if np.isfinite(number):
                axis.text(column, row, f"{number:.4g}", ha="center", va="center", fontsize=8)
        if len(finite):
            figure.colorbar(artist, ax=axis, label=label)
        png_path = destination.with_suffix(".png")
        pdf_path = destination.with_suffix(".pdf")
        figure.savefig(png_path, dpi=220)
        figure.savefig(pdf_path, bbox_inches="tight")
    finally:
        plt.close(figure)
    return [csv_path, png_path, pdf_path]


def main() -> None:
    from timebench.paths import outputs_root
    from timebench.pipeline.runtime_resources import log_selected_device

    log_selected_device("cpu", stage="report", component="self_time_summary")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=outputs_root() / "self_augmentation" / "tasks")
    parser.add_argument("--seasonal-naive-results-dir", type=Path,
                        default=os.environ.get("TIME_SEASONAL_TASKS_ROOT"))
    parser.add_argument("--launch-id", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.seasonal_naive_results_dir is None:
        raise ValueError("Set TIME_SEASONAL_TASKS_ROOT or pass --seasonal-naive-results-dir")

    reporter = _foundation_reporter()
    cells = reporter.load_result_cells(
        args.results_dir,
        {"chronos2", "ts_icl"},
        launch_id=args.launch_id,
        config_policy="distinct",
        repeat_policy="latest",
    )
    seasonal = reporter.load_result_cells(
        args.seasonal_naive_results_dir,
        {"seasonal_naive"},
        config_policy="latest",
        repeat_policy="latest",
        task_specific_model_fields={"season_length"},
    )
    if not cells:
        raise ValueError(f"No completed Self TIME tasks found below {args.results_dir}")

    configurations: dict[str, tuple[str, str]] = {}
    for cell in cells:
        experiment = cell["scientific_config"]["experiment_config"]
        augmentation_id = str(experiment["augmentation_id"])
        channels = list(experiment["self_augmentations"])
        label = f"{cell['base_model']}__{augmentation_id}"
        cell["model"] = label
        configurations[label] = (augmentation_id, json.dumps(channels))

    rows = reporter.summarize_cells(cells, seasonal)
    for row in rows:
        augmentation_id, channels = configurations[row["model"]]
        row["augmentation_id"] = augmentation_id
        row["self_augmentations"] = channels
    rows.sort(key=lambda row: (row["base_model"], row["augmentation_id"]))

    destination = args.output or (
        outputs_root() / "self_augmentation" / "reports"
    )
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "self_time_summary.csv"
    _write_summary(rows, summary_path)
    artifacts = [summary_path, summary_path.with_suffix(".md")]
    artifacts.extend(
        reporter.write_performance_artifacts(cells, seasonal, destination / "performance")
    )
    artifacts.extend(_write_matrix(
        rows,
        "scaled_MASE",
        destination / "performance" / "augmentation_backbone_scaled_MASE",
        "Seasonal-scaled MASE (lower is better)",
    ))
    artifacts.extend(_write_matrix(
        rows,
        "inference_seconds",
        destination / "performance" / "augmentation_backbone_inference_seconds",
        "Total inference time (seconds)",
    ))

    manifest = {
        "schema_version": 1,
        "report": "self_time_summary",
        "selection": {
            "launch_id": args.launch_id,
            "models": ["chronos2", "ts_icl"],
            "config_policy": "distinct",
            "repeat_policy": "latest",
        },
        "input_manifests": [cell["manifest_path"] for cell in cells],
        "seasonal_naive_input_manifests": [cell["manifest_path"] for cell in seasonal],
        "artifacts": [str(path.relative_to(destination)) for path in artifacts],
    }
    (destination / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Self TIME report written to {destination}")


if __name__ == "__main__":
    main()
