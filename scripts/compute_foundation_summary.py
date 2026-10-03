#!/usr/bin/env python3
"""Create a Seasonal-Naive-scaled MASE and inference-time TIME table."""

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from timebench.paths import foundation_experiment_root, outputs_root, foundation_experiment_name
from timebench.pipeline import manifest_reference, parse_config_filters, select_completed_runs
from timebench.pipeline.report_transaction import ReportTransaction

DEFAULT_MODELS = (
    "chronos2",
    "ts_icl",
    "seasonal_naive",
)


def load_result_cells(
    root: Path,
    models: set[str] | None = None,
    launch_id: str | None = None,
    target_modes: set[str] | None = None,
    config_filters: dict | None = None,
    config_policy: str = "error",
    repeat_policy: str = "latest",
    task_specific_model_fields: set[str] | None = None,
    config_axis_fields: list[str] | None = None,
) -> list[dict]:
    """Load selected completed manifests for dataset/frequency/horizon cells."""
    cells = []
    selected = select_completed_runs(
        root,
        models=models,
        target_modes=target_modes,
        launch_id=launch_id,
        config_filters=config_filters,
        config_policy=config_policy,
        repeat_policy=repeat_policy,
        task_specific_model_fields=task_specific_model_fields,
        config_axis_fields=config_axis_fields,
    )
    for run_dir, manifest in selected:
        identity = manifest["identity"]
        summary_path = run_dir / "metrics_summary.json"
        config = manifest.get("artifact_metadata", {}).get("evaluation")
        if not isinstance(config, dict):
            raise ValueError(f"Run manifest lacks evaluation artifact metadata: {run_dir}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        mase_summary = summary.get("metrics", {}).get("MASE", {})
        mase = mase_summary.get("mean")
        if mase is None:
            continue
        mase = float(mase)
        inference_seconds = config.get("inference_seconds")
        if inference_seconds is not None:
            inference_seconds = float(inference_seconds)
        selection = manifest.get("selection", {})

        cells.append(
            {
                "model": selection.get("model_label", identity["model"]),
                "base_model": identity["model"],
                "target_mode": identity["target_mode"],
                "dataset_id": f"{identity['dataset']}/{identity['frequency']}",
                "horizon": identity["term"],
                "MASE": mase,
                "MASE_std": mase_summary.get("std", np.nan),
                "MASE_variance": mase_summary.get("variance", np.nan),
                "MASE_finite_values": int(mase_summary["finite_values"]),
                "MASE_evaluation_values": int(mase_summary["evaluation_values"]),
                "MASE_total_values": int(mase_summary["total_values"]),
                "evaluation_grid": dict(summary["evaluation_grid"]),
                "inference_seconds": inference_seconds,
                "prediction_length": int(config["prediction_length"]),
                "manifest_path": str(run_dir / "manifest.json"),
                "scientific_config": selection.get(
                    "scientific_config",
                    {
                        "model_config": manifest.get("model_config", {}),
                        "pipeline_config": manifest.get("pipeline_config", {}),
                        "experiment_config": manifest.get("experiment_config", {}),
                    },
                ),
            }
        )
    return cells


def _effective_cells(cells: list[dict]) -> list[dict]:
    """Apply configured repeat/config averaging to task-level raw MASE."""
    by_exact_config = defaultdict(list)
    for cell in cells:
        key = (
            cell["model"],
            cell.get("base_model", cell["model"]),
            cell["target_mode"],
            cell["dataset_id"],
            cell["horizon"],
            json.dumps(cell.get("scientific_config", {}), sort_keys=True),
        )
        by_exact_config[key].append(cell)

    config_means = []
    for key, repeats in by_exact_config.items():
        grids = {
            json.dumps(cell["evaluation_grid"], sort_keys=True) for cell in repeats
        }
        if len(grids) != 1:
            raise ValueError("exact repeats use different evaluation grids")
        timed = [
            cell["inference_seconds"]
            for cell in repeats
            if cell["inference_seconds"] is not None
            and np.isfinite(cell["inference_seconds"])
        ]
        config_means.append(
            {
                "model": key[0],
                "base_model": key[1],
                "target_mode": key[2],
                "dataset_id": key[3],
                "horizon": key[4],
                "MASE": float(np.mean([cell["MASE"] for cell in repeats])),
                "MASE_std": float(np.mean([cell.get("MASE_std", np.nan) for cell in repeats])),
                "MASE_variance": float(np.mean([cell.get("MASE_variance", np.nan) for cell in repeats])),
                "MASE_finite_values": sum(
                    cell["MASE_finite_values"] for cell in repeats
                ),
                "MASE_evaluation_values": sum(
                    cell["MASE_evaluation_values"] for cell in repeats
                ),
                "MASE_total_values": sum(
                    cell["MASE_total_values"] for cell in repeats
                ),
                "evaluation_grid": repeats[0]["evaluation_grid"],
                "inference_seconds": (
                    float(np.mean(timed)) if len(timed) == len(repeats) else None
                ),
            }
        )

    by_task = defaultdict(list)
    for cell in config_means:
        by_task[
            (
                cell["model"],
                cell["base_model"],
                cell["target_mode"],
                cell["dataset_id"],
                cell["horizon"],
            )
        ].append(cell)
    effective_cells = []
    for key, configs in by_task.items():
        grids = {
            json.dumps(cell["evaluation_grid"], sort_keys=True) for cell in configs
        }
        if len(grids) != 1:
            raise ValueError("selected configurations use different evaluation grids")
        timed = [cell["inference_seconds"] for cell in configs if cell["inference_seconds"] is not None]
        effective_cells.append(
            {
                "model": key[0],
                "base_model": key[1],
                "target_mode": key[2],
                "dataset_id": key[3],
                "horizon": key[4],
                "MASE": float(np.mean([cell["MASE"] for cell in configs])),
                "MASE_std": float(np.mean([cell.get("MASE_std", np.nan) for cell in configs])),
                "MASE_variance": float(np.mean([cell.get("MASE_variance", np.nan) for cell in configs])),
                "MASE_finite_values": sum(
                    cell["MASE_finite_values"] for cell in configs
                ),
                "MASE_evaluation_values": sum(
                    cell["MASE_evaluation_values"] for cell in configs
                ),
                "MASE_total_values": sum(
                    cell["MASE_total_values"] for cell in configs
                ),
                "evaluation_grid": configs[0]["evaluation_grid"],
                "inference_seconds": (
                    float(np.mean(timed)) if len(timed) == len(configs) else None
                ),
            }
        )

    return effective_cells


def _geometric_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("scaled MASE requires finite non-negative task values")
    if np.any(array == 0):
        return 0.0
    return float(np.exp(np.mean(np.log(array))))


def summarize_cells(cells: list[dict], seasonal_naive_cells: list[dict]) -> list[dict]:
    """Normalize each task by Seasonal Naive and geometrically average tasks."""

    effective_cells = _effective_cells(cells)
    baseline_cells = _effective_cells(seasonal_naive_cells)
    baseline_by_task: dict[tuple[str, str], list[float]] = defaultdict(list)
    for cell in baseline_cells:
        baseline_by_task[(cell["dataset_id"], cell["horizon"])].append(cell["MASE"])
    baseline = {
        key: float(np.mean(values)) for key, values in baseline_by_task.items()
    }
    baseline_grids = {
        (cell["dataset_id"], cell["horizon"]): cell["evaluation_grid"]
        for cell in baseline_cells
    }
    for cell in effective_cells:
        key = (cell["dataset_id"], cell["horizon"])
        denominator = baseline.get(key)
        if denominator is None or not np.isfinite(denominator) or denominator <= 0:
            raise ValueError(
                f"missing positive Seasonal Naive MASE for {cell['dataset_id']}/{cell['horizon']}"
            )
        if cell["evaluation_grid"] != baseline_grids[key]:
            raise ValueError(
                f"evaluation grid differs from Seasonal Naive for "
                f"{cell['dataset_id']}/{cell['horizon']}"
            )
        cell["scaled_MASE"] = float(cell["MASE"] / denominator)

    by_model = defaultdict(list)
    for cell in effective_cells:
        by_model[cell["model"]].append(cell)

    rows = []
    for model, model_cells in by_model.items():
        datasets = {cell["dataset_id"] for cell in model_cells}
        timed = [
            cell["inference_seconds"]
            for cell in model_cells
            if cell["inference_seconds"] is not None
            and np.isfinite(cell["inference_seconds"])
        ]
        all_tasks_timed = len(timed) == len(model_cells)
        rows.append(
            {
                "model": model,
                "base_model": model_cells[0].get("base_model", model),
                "target_modes": ",".join(
                    sorted({cell["target_mode"] for cell in model_cells})
                ),
                "scaled_MASE": _geometric_mean(
                    [cell["scaled_MASE"] for cell in model_cells]
                ),
                "inference_seconds": float(sum(timed)) if all_tasks_timed else None,
                "datasets": len(datasets),
                "tasks": len(model_cells),
                "timed_tasks": len(timed),
                "MASE_finite_values": sum(
                    cell["MASE_finite_values"] for cell in model_cells
                ),
                "MASE_evaluation_values": sum(
                    cell["MASE_evaluation_values"] for cell in model_cells
                ),
                "MASE_total_values": sum(
                    cell["MASE_total_values"] for cell in model_cells
                ),
            }
        )

    return sorted(rows, key=lambda row: (row["scaled_MASE"], row["model"]))



def write_performance_artifacts(cells: list[dict], seasonal_cells: list[dict], destination: Path):
    """Adapt the existing selected/reduced foundation task contract."""
    from timebench.results.performance import write_performance_report

    effective = _effective_cells([cell for cell in cells if cell.get("base_model") != "seasonal_naive"])
    if not effective:
        return []
    horizons = {}
    for cell in [*cells, *seasonal_cells]:
        key = (cell["dataset_id"], cell["horizon"])
        horizon = int(cell["prediction_length"])
        if key in horizons and horizons[key] != horizon:
            raise ValueError(f"Different forecast horizons for {key}")
        horizons[key] = horizon
    grouped = defaultdict(list)
    for cell in _effective_cells(seasonal_cells):
        grouped[(cell["dataset_id"], cell["horizon"])].append(cell)
    wanted = {(cell["dataset_id"], cell["horizon"]) for cell in effective}
    baselines = {}
    for key in wanted:
        selected = grouped[key]
        times = [cell["inference_seconds"] for cell in selected]
        baselines[key] = {
            "model": "seasonal_naive", "dataset_id": key[0], "horizon": key[1],
            "MASE": float(np.mean([cell["MASE"] for cell in selected])),
            "MASE_std": float(np.mean([cell.get("MASE_std", np.nan) for cell in selected])),
            "MASE_variance": float(np.mean([cell.get("MASE_variance", np.nan) for cell in selected])),
            "inference_seconds": float(np.mean(times)) if all(value is not None for value in times) else None,
        }
    tasks = []
    for cell in [*effective, *baselines.values()]:
        key = (cell["dataset_id"], cell["horizon"])
        dataset, frequency = cell["dataset_id"].rsplit("/", 1)
        tasks.append({
            "model": cell["model"], "dataset": dataset, "frequency": frequency,
            "term": cell["horizon"], "horizon_steps": horizons[key],
            "MASE": cell["MASE"], "scaled_MASE": cell["MASE"] / baselines[key]["MASE"],
            "MASE_std": cell.get("MASE_std", np.nan),
            "MASE_variance": cell.get("MASE_variance", np.nan),
            "seasonal_MASE_variance": baselines[key].get("MASE_variance", np.nan),
            "inference_seconds": cell["inference_seconds"],
        })
    return write_performance_report(
        tasks, destination, reference="seasonal_naive", scaled_aggregation="geometric",
        inputs={
            "model_dependencies": [
                manifest_reference(cell["manifest_path"]) for cell in cells
            ],
            "seasonal_dependencies": [
                manifest_reference(cell["manifest_path"]) for cell in seasonal_cells
            ],
        })



def load_model_statuses(
    status_dir: Path | None, launch_id: str | None
) -> dict[str, dict[str, str]]:
    """Load terminal per-model workflow status for one launch when available."""
    if status_dir is None or not status_dir.is_dir():
        return {}

    statuses = {}
    launch_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", launch_id) if launch_id else None
    pattern = f"{launch_name}__*.status" if launch_name else "*.status"
    for status_path in sorted(status_dir.glob(pattern)):
        values = {}
        for line in status_path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key] = value
        statuses[status_path.stem.partition("__")[2] or status_path.stem] = values
    return statuses


def parse_model_statuses(values: list[str]) -> dict[str, dict[str, str]]:
    """Parse explicit MODEL=STATE,EXIT_CODE evaluation statuses."""
    statuses = {}
    for value in values:
        model, separator, status = value.partition("=")
        state, comma, exit_code = status.partition(",")
        if not separator or not model or not comma or not state or not exit_code:
            raise ValueError(
                f"Invalid model status {value!r}; expected MODEL=STATE,EXIT_CODE"
            )
        statuses[model] = {"state": state, "exit_code": exit_code}
    return statuses


def add_model_status(
    metric_rows: list[dict],
    models: list[str] | tuple[str, ...],
    statuses: dict[str, dict[str, str]],
    launch_id: str | None,
) -> list[dict]:
    """Attach launch status and retain failed models with no metric cells."""
    rows = [row.copy() for row in metric_rows]
    if statuses:
        present = {row.get("base_model", row["model"]) for row in rows}
        for model in models:
            if model in present:
                continue
            rows.append(
                {
                    "model": model,
                    "base_model": model,
                    "target_modes": "",
                    "scaled_MASE": None,
                    "inference_seconds": None,
                    "datasets": 0,
                    "tasks": 0,
                    "timed_tasks": 0,
                    "MASE_finite_values": 0,
                    "MASE_evaluation_values": 0,
                    "MASE_total_values": 0,
                }
            )
    for row in rows:
        model = row.get("base_model", row["model"])
        status = statuses.get(model, {})
        row["launch_id"] = launch_id or status.get("launch_id", "")
        row["state"] = status.get("state", "")
        row["exit_code"] = status.get("exit_code", "")
    return sorted(
        rows,
        key=lambda row: (
            row["scaled_MASE"] is None,
            float("inf") if row["scaled_MASE"] is None else row["scaled_MASE"],
            row["model"],
            row["target_modes"],
        ),
    )


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "launch_id",
        "model",
        "base_model",
        "target_modes",
        "state",
        "exit_code",
        "scaled_MASE",
        "inference_seconds",
        "datasets",
        "tasks",
        "timed_tasks",
        "MASE_finite_values",
        "MASE_evaluation_values",
        "MASE_total_values",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Foundation-model benchmark summary",
        "",
        "Each task MASE is divided by the matching Seasonal Naive MASE, then "
        "task ratios are combined with the TIME leaderboard geometric mean. "
        "Inference seconds are summed over the same test forecasting tasks; "
        "a blank total means at least one task lacks timing metadata.",
        "",
        "| Model | Target mode | State | Exit | Scaled MASE (GM) | Inference seconds | Datasets | Tasks | Timed tasks | MASE finite/grid/total |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        seconds = row["inference_seconds"]
        mase = row["scaled_MASE"]
        lines.append(
            f"| {row['model']} | {row['target_modes']} | {row['state']} | {row['exit_code']} | "
            f"{'' if mase is None else f'{mase:.6f}'} | "
            f"{'' if seconds is None else f'{seconds:.3f}'} | "
            f"{row['datasets']} | {row['tasks']} | {row['timed_tasks']} | "
            f"{row['MASE_finite_values']}/{row['MASE_evaluation_values']}/{row['MASE_total_values']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_manifest(
    cells: list[dict],
    seasonal_naive_cells: list[dict],
    path: Path,
    *,
    results_dir: Path,
    seasonal_naive_results_dir: Path,
    models: list[str],
    target_modes: set[str] | None,
    launch_id: str | None,
    config_filters: dict,
    config_policy: str,
    repeat_policy: str,
    artifacts: list[Path],
) -> None:
    """Record the exact run manifests selected for this aggregate."""
    payload = {
        "schema_version": 1,
        "report": "foundation_model_summary",
        "results_dir": str(results_dir),
        "metric": "task_MASE_divided_by_matching_Seasonal_Naive_MASE",
        "aggregation": "geometric_mean_over_tasks",
        "seasonal_naive_results_dir": str(seasonal_naive_results_dir),
        "selection": {
            "models": models,
            "target_modes": sorted(target_modes or []),
            "launch_id": launch_id,
            "config_filters": config_filters,
            "config_policy": config_policy,
            "repeat_policy": repeat_policy,
        },
        "input_dependencies": [
            manifest_reference(cell["manifest_path"]) for cell in cells
        ],
        "seasonal_naive_input_dependencies": [
            manifest_reference(cell["manifest_path"])
            for cell in seasonal_naive_cells
        ],
        "artifacts": [
            artifact.resolve().relative_to(path.parent.resolve()).as_posix()
            if artifact.resolve().is_relative_to(path.parent.resolve())
            else str(artifact)
            for artifact in artifacts
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    from timebench.pipeline.runtime_resources import log_selected_device
    log_selected_device("cpu", stage="report", component="foundation_summary")
    parser = argparse.ArgumentParser(
        description="Summarize foundation-model MASE and test inference time."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=foundation_experiment_root(),
        help="One experiment task root (default: outputs/self_augmentation/tasks)",
    )
    parser.add_argument(
        "--seasonal-naive-results-dir",
        type=Path,
        default=None,
        help=(
            "Task root containing the matching Seasonal Naive baseline; "
            "defaults to --results-dir"
        ),
    )
    parser.add_argument(
        "--seasonal-naive-launch-id",
        default=None,
        help="Optional launch filter for Seasonal Naive baseline tasks",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV table (default: outputs/<experiment>/reports/foundation_model_summary.csv)",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Markdown table (default: beside the CSV table)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Canonical model result directories to summarize",
    )
    parser.add_argument(
        "--launch-id",
        default=None,
        help="Include only task artifacts stamped with this launch ID",
    )
    parser.add_argument(
        "--target-mode",
        nargs="+",
        choices=("univariate", "multivariate"),
        default=None,
        help="Optional target-representation filter",
    )
    parser.add_argument(
        "--run-config",
        action="append",
        default=[],
        help="Manifest filter FIELD=JSON, for example model_config.context_length=2048",
    )
    parser.add_argument(
        "--config-policy",
        choices=("error", "distinct", "latest", "average"),
        default="error",
        help="How to handle different matching scientific configs",
    )
    parser.add_argument(
        "--config-axis",
        action="append",
        default=[],
        help="Dotted scientific field used as an explicit distinct-report axis",
    )
    parser.add_argument(
        "--repeat-policy",
        choices=("selected", "latest", "distinct", "average"),
        default="latest",
        help="How to select or aggregate exact repeated configurations",
    )
    parser.add_argument(
        "--status-dir",
        type=Path,
        default=None,
        help="Per-model workflow status directory for the selected launch",
    )
    parser.add_argument(
        "--model-status",
        action="append",
        default=[],
        metavar="MODEL=STATE,EXIT_CODE",
        help=(
            "Explicit evaluation status, used when the workflow status filename "
            "is a comparison mode rather than a model alias"
        ),
    )
    args = parser.parse_args()

    report_root = outputs_root() / foundation_experiment_name() / "reports"
    final_csv = args.csv or report_root / "foundation_model_summary.csv"
    final_markdown = args.markdown or final_csv.parent / "foundation_model_summary.md"
    report_transaction = ReportTransaction(final_csv.parent)
    args.csv = report_transaction.path(final_csv)
    args.markdown = report_transaction.path(final_markdown)
    staged_extra_artifacts = []
    for artifact in args.extra_artifact:
        artifact = artifact.expanduser().resolve()
        if artifact.is_relative_to(report_transaction.destination):
            staged = report_transaction.path(artifact)
        else:
            staged = report_transaction.staging / "performance" / artifact.name
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact, staged)
        staged_extra_artifacts.append(staged)
    args.extra_artifact = staged_extra_artifacts
    baseline_root = args.seasonal_naive_results_dir or args.results_dir
    baseline_launch_id = args.seasonal_naive_launch_id
    if baseline_launch_id is None and baseline_root.resolve() == args.results_dir.resolve():
        baseline_launch_id = args.launch_id
    models = set(args.models)
    config_filters = parse_config_filters(args.run_config)
    cells = load_result_cells(
        args.results_dir,
        models,
        launch_id=args.launch_id,
        target_modes=None if args.target_mode is None else set(args.target_mode),
        config_filters=config_filters,
        config_policy=args.config_policy,
        repeat_policy=args.repeat_policy,
        config_axis_fields=args.config_axis,
    )
    seasonal_naive_cells = load_result_cells(
        baseline_root,
        {"seasonal_naive"},
        launch_id=baseline_launch_id,
        target_modes={"univariate"},
        config_policy="error",
        repeat_policy=args.repeat_policy,
        task_specific_model_fields={"season_length"},
    )
    summary_cells = [
        cell for cell in cells if cell.get("base_model") != "seasonal_naive"
    ]
    if "seasonal_naive" in models:
        summary_cells.extend(seasonal_naive_cells)
    metric_rows = summarize_cells(summary_cells, seasonal_naive_cells)
    statuses = load_model_statuses(args.status_dir, args.launch_id)
    statuses.update(parse_model_statuses(args.model_status))
    rows = add_model_status(metric_rows, args.models, statuses, args.launch_id)
    if not rows:
        raise SystemExit(
            f"No aggregate metric cells or model statuses found below {args.results_dir}"
        )

    write_csv(rows, args.csv)
    write_markdown(rows, args.markdown)
    performance_artifacts = write_performance_artifacts(
        summary_cells, seasonal_naive_cells, args.csv.parent / "performance")
    write_report_manifest(
        cells,
        seasonal_naive_cells,
        args.csv.with_name("foundation_model_report_manifest.json"),
        results_dir=args.results_dir,
        seasonal_naive_results_dir=baseline_root,
        models=args.models,
        target_modes=None if args.target_mode is None else set(args.target_mode),
        launch_id=args.launch_id,
        config_filters=config_filters,
        config_policy=args.config_policy,
        repeat_policy=args.repeat_policy,
        artifacts=[args.csv, args.markdown, *performance_artifacts],
    )
    report_transaction.commit()
    print(f"Foundation-model summary written to {final_csv} and {final_markdown}")
    print()
    for row in rows:
        seconds = row["inference_seconds"]
        seconds_text = "incomplete" if seconds is None else f"{seconds:.3f}s"
        mase = row["scaled_MASE"]
        mase_text = "incomplete" if mase is None else f"{mase:.6f}"
        print(
            f"{(row['model'] + ('/' + row['target_modes'] if row['target_modes'] else '')):<28} "
            f"state={row['state'] or 'unknown'}  "
            f"scaled_MASE={mase_text}  "
            f"inference={seconds_text}  "
            f"coverage={row['timed_tasks']}/{row['tasks']} timed tasks, "
            f"{row['MASE_finite_values']}/{row['MASE_evaluation_values']}/"
            f"{row['MASE_total_values']} finite/grid/total MASE values"
        )


if __name__ == "__main__":
    main()
