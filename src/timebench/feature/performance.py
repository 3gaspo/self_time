"""Dataset-feature joins and dependency-free SVG performance plots."""

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from timebench.pipeline import select_completed_runs


NON_FEATURE_COLUMNS = {
    "dataset_id",
    "series_count",
    "variate_count",
    "model",
    "scaled_MASE",
}


def load_dataset_features(root: Path, split: str = "full") -> pd.DataFrame:
    """Load one dataset-level feature row per dataset/frequency."""
    paths = sorted(root.glob(f"*/*/{split}_dataset.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No {split}_dataset.csv feature summaries found below {root}"
        )
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    duplicates = frame[frame["dataset_id"].duplicated()]["dataset_id"].tolist()
    if duplicates:
        raise ValueError(f"Duplicate dataset feature rows: {sorted(set(duplicates))}")
    return frame


def load_dataset_mase(
    results_root: Path,
    seasonal_naive_results_root: Path | None = None,
    seasonal_naive_launch_id: str | None = None,
    models: set[str] | None = None,
    launch_id: str | None = None,
    target_modes: set[str] | None = None,
    config_filters: dict | None = None,
    config_policy: str = "latest",
    repeat_policy: str = "latest",
) -> pd.DataFrame:
    """Geometrically average Seasonal-Naive-scaled task MASE per dataset."""
    rows = []
    selected = select_completed_runs(
        results_root,
        models=models,
        target_modes=target_modes,
        launch_id=launch_id,
        config_filters=config_filters,
        config_policy=config_policy,
        repeat_policy=repeat_policy,
    )
    for run_dir, manifest in selected:
        identity = manifest["identity"]
        path = run_dir / "metrics_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        metrics = summary.get("metrics", {})
        mase = metrics.get("MASE", {}).get("mean")
        if mase is not None and np.isfinite(float(mase)):
            selection = manifest.get("selection", {})
            rows.append(
                {
                    "model": selection.get("model_label", identity["model"]),
                    "base_model": identity["model"],
                    "dataset_id": f"{identity['dataset']}/{identity['frequency']}",
                    "horizon": identity["term"],
                    "MASE": float(mase),
                    "scientific_config": json.dumps(
                        selection.get(
                            "scientific_config",
                            {
                                "model_config": manifest.get("model_config", {}),
                                "pipeline_config": manifest.get("pipeline_config", {}),
                                "experiment_config": manifest.get("experiment_config", {}),
                            },
                        ),
                        sort_keys=True,
                    ),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No finite task MASE summaries found below {results_root}")
    frame = pd.DataFrame(rows)
    baseline_root = seasonal_naive_results_root or results_root
    if (
        seasonal_naive_launch_id is None
        and baseline_root.resolve() == results_root.resolve()
    ):
        seasonal_naive_launch_id = launch_id
    baseline_selected = select_completed_runs(
        baseline_root,
        models={"seasonal_naive"},
        target_modes={"univariate"},
        launch_id=seasonal_naive_launch_id,
        config_policy=config_policy,
        repeat_policy=repeat_policy,
        task_specific_model_fields={"season_length"},
    )
    baseline_rows = []
    for run_dir, manifest in baseline_selected:
        summary = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
        mase = summary.get("metrics", {}).get("MASE", {}).get("mean")
        if mase is not None and np.isfinite(float(mase)) and float(mase) > 0:
            identity = manifest["identity"]
            baseline_rows.append(
                {
                    "dataset_id": f"{identity['dataset']}/{identity['frequency']}",
                    "horizon": identity["term"],
                    "baseline_MASE": float(mase),
                }
            )
    if not baseline_rows:
        raise FileNotFoundError("No finite Seasonal Naive MASE baselines found")
    baseline = pd.DataFrame(baseline_rows).groupby(
        ["dataset_id", "horizon"], as_index=False
    )["baseline_MASE"].mean()
    frame = frame.merge(baseline, on=["dataset_id", "horizon"], validate="many_to_one")
    frame["scaled_MASE"] = frame["MASE"] / frame["baseline_MASE"]
    per_config = frame.groupby(
        ["model", "base_model", "dataset_id", "horizon", "scientific_config"],
        as_index=False,
    )["scaled_MASE"].mean()
    per_task = per_config.groupby(
        ["model", "base_model", "dataset_id", "horizon"], as_index=False
    )["scaled_MASE"].mean()
    def geometric_mean(values: pd.Series) -> float:
        array = values.to_numpy(dtype=np.float64)
        if np.any(array == 0):
            return 0.0
        return float(np.exp(np.log(array).mean()))

    return per_task.groupby(
        ["model", "base_model", "dataset_id"], as_index=False
    )["scaled_MASE"].agg(geometric_mean)


def join_features_and_mase(
    features_root: Path,
    results_root: Path,
    seasonal_naive_results_root: Path | None = None,
    seasonal_naive_launch_id: str | None = None,
    split: str = "full",
    models: set[str] | None = None,
    launch_id: str | None = None,
    target_modes: set[str] | None = None,
    config_filters: dict | None = None,
    config_policy: str = "latest",
    repeat_policy: str = "latest",
) -> pd.DataFrame:
    """Join dataset features to Seasonal-Naive-scaled model performance."""
    features = load_dataset_features(features_root, split=split)
    mase = load_dataset_mase(
        results_root,
        seasonal_naive_results_root=seasonal_naive_results_root,
        seasonal_naive_launch_id=seasonal_naive_launch_id,
        models=models,
        launch_id=launch_id,
        target_modes=target_modes,
        config_filters=config_filters,
        config_policy=config_policy,
        repeat_policy=repeat_policy,
    )
    joined = mase.merge(features, on="dataset_id", how="inner", validate="many_to_one")
    if joined.empty:
        raise ValueError("Feature summaries and model results have no dataset IDs in common")
    return joined


def feature_correlations(
    frame: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Compute Spearman correlations within each model and a selection score."""
    if features is None:
        features = [
            column
            for column in frame.select_dtypes(include=[np.number]).columns
            if column not in NON_FEATURE_COLUMNS
        ]
    unknown = sorted(set(features) - set(frame.columns))
    if unknown:
        raise ValueError(f"Unknown feature columns: {unknown}")

    rows = []
    for feature in features:
        correlations = []
        for model, group in frame.groupby("model"):
            values = group[[feature, "scaled_MASE"]].replace([np.inf, -np.inf], np.nan).dropna()
            correlation = np.nan
            if (
                len(values) >= 3
                and values[feature].nunique() > 1
                and values["scaled_MASE"].nunique() > 1
            ):
                correlation = float(values[feature].corr(values["scaled_MASE"], method="spearman"))
                if np.isfinite(correlation):
                    correlations.append(abs(correlation))
            rows.append(
                {
                    "feature": feature,
                    "model": model,
                    "spearman_rho": correlation,
                    "datasets": len(values),
                }
            )
        rows.append(
            {
                "feature": feature,
                "model": "mean_absolute",
                "spearman_rho": float(np.mean(correlations)) if correlations else np.nan,
                "datasets": int(frame["dataset_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _scale(values: np.ndarray, low: float, high: float) -> tuple[np.ndarray, float, float]:
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    if maximum <= minimum:
        maximum = minimum + 1.0
    padding = 0.04 * (maximum - minimum)
    minimum -= padding
    maximum += padding
    scaled = low + (values - minimum) * (high - low) / (maximum - minimum)
    return scaled, minimum, maximum


def write_feature_svg(
    frame: pd.DataFrame,
    correlations: pd.DataFrame,
    features: list[str],
    path: Path,
) -> None:
    """Write a multi-panel scaled-MASE-versus-feature scatter plot as plain SVG."""
    width, height = 1280, 1160
    panel_width, panel_height = 600, 320
    left_margin, top_margin = 50, 105
    colors = ["#3366cc", "#dc3912", "#109618", "#990099", "#ff9900"]
    models = sorted(frame["model"].unique())
    model_colors = {model: colors[index % len(colors)] for index, model in enumerate(models)}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#555;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.point{fill-opacity:.68}.trend{fill:none;stroke-width:2}</style>',
        '<text x="50" y="42" font-size="25" font-weight="bold">Dataset scaled MASE versus top correlated TIME features</text>',
        '<text x="50" y="70" font-size="14">Features selected by mean absolute within-model Spearman correlation; task MASE is Seasonal-Naive-scaled and geometrically averaged.</text>',
    ]

    for model_index, model in enumerate(models):
        x = 640 + model_index * 145
        lines.append(
            f'<circle cx="{x}" cy="43" r="6" fill="{model_colors[model]}"/>'
            f'<text x="{x + 10}" y="48" font-size="13">{escape(model)}</text>'
        )

    for index, feature in enumerate(features):
        column, row = index % 2, index // 2
        panel_x = left_margin + column * 625
        panel_y = top_margin + row * 345
        plot_left, plot_right = panel_x + 70, panel_x + panel_width - 20
        plot_top, plot_bottom = panel_y + 40, panel_y + panel_height - 55
        panel = frame[["model", feature, "scaled_MASE"]].replace([np.inf, -np.inf], np.nan).dropna()
        x_values = panel[feature].to_numpy(dtype=float)
        y_values = panel["scaled_MASE"].to_numpy(dtype=float)
        x_scaled, x_min, x_max = _scale(x_values, plot_left, plot_right)
        y_scaled, y_min, y_max = _scale(y_values, plot_bottom, plot_top)
        score_row = correlations[
            (correlations["feature"] == feature)
            & (correlations["model"] == "mean_absolute")
        ]
        score = float(score_row["spearman_rho"].iloc[0])

        lines.append(
            f'<text x="{panel_x + 10}" y="{panel_y + 20}" font-size="16" font-weight="bold">'
            f'{escape(feature)}  (mean |rho|={score:.3f})</text>'
        )
        for tick in range(5):
            fraction = tick / 4
            y = plot_bottom - fraction * (plot_bottom - plot_top)
            value = y_min + fraction * (y_max - y_min)
            lines.append(f'<line class="grid" x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}"/>')
            lines.append(f'<text x="{plot_left - 8}" y="{y + 4:.1f}" font-size="11" text-anchor="end">{value:.2f}</text>')
        lines.append(f'<line class="axis" x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}"/>')
        lines.append(f'<line class="axis" x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}"/>')
        lines.append(f'<text x="{panel_x + 17}" y="{(plot_top + plot_bottom) / 2}" font-size="12" transform="rotate(-90 {panel_x + 17} {(plot_top + plot_bottom) / 2})" text-anchor="middle">Scaled MASE</text>')
        lines.append(f'<text x="{(plot_left + plot_right) / 2}" y="{plot_bottom + 39}" font-size="12" text-anchor="middle">{escape(feature)}</text>')
        lines.append(f'<text x="{plot_left}" y="{plot_bottom + 17}" font-size="11" text-anchor="middle">{x_min:.3g}</text>')
        lines.append(f'<text x="{plot_right}" y="{plot_bottom + 17}" font-size="11" text-anchor="middle">{x_max:.3g}</text>')

        for point_index, (_, point) in enumerate(panel.iterrows()):
            lines.append(
                f'<circle class="point" cx="{x_scaled[point_index]:.2f}" cy="{y_scaled[point_index]:.2f}" '
                f'r="4" fill="{model_colors[point["model"]]}"/>'
            )

        for model, group in panel.groupby("model"):
            if len(group) < 2 or group[feature].nunique() < 2:
                continue
            slope, intercept = np.polyfit(group[feature], group["scaled_MASE"], 1)
            trend_x = np.asarray([group[feature].min(), group[feature].max()], dtype=float)
            trend_y = slope * trend_x + intercept
            trend_x_scaled = plot_left + (trend_x - x_min) * (plot_right - plot_left) / (x_max - x_min)
            trend_y_scaled = plot_bottom - (trend_y - y_min) * (plot_bottom - plot_top) / (y_max - y_min)
            lines.append(
                f'<path class="trend" stroke="{model_colors[model]}" '
                f'd="M {trend_x_scaled[0]:.2f} {trend_y_scaled[0]:.2f} L {trend_x_scaled[1]:.2f} {trend_y_scaled[1]:.2f}"/>'
            )

    lines.append(
        f'<text x="50" y="{height - 25}" font-size="12">Each point is one model/dataset pair; lines are descriptive least-squares fits, not significance claims.</text>'
    )
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_feature_performance(
    features_root: Path,
    results_root: Path,
    output_svg: Path,
    seasonal_naive_results_root: Path | None = None,
    seasonal_naive_launch_id: str | None = None,
    split: str = "full",
    models: set[str] | None = None,
    launch_id: str | None = None,
    target_modes: set[str] | None = None,
    config_filters: dict | None = None,
    config_policy: str = "latest",
    repeat_policy: str = "latest",
    features: list[str] | None = None,
    top: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Join artifacts, rank correlations, and write the requested SVG plot."""
    joined = join_features_and_mase(
        features_root,
        results_root,
        seasonal_naive_results_root,
        seasonal_naive_launch_id,
        split,
        models,
        launch_id,
        target_modes,
        config_filters,
        config_policy,
        repeat_policy,
    )
    correlations = feature_correlations(joined, features=features)
    scores = correlations[correlations["model"] == "mean_absolute"].dropna(
        subset=["spearman_rho"]
    )
    selected = (
        scores.sort_values("spearman_rho", ascending=False)["feature"]
        .head(top)
        .tolist()
    )
    if not selected:
        raise ValueError("No feature has enough finite variation for correlation analysis")
    write_feature_svg(joined, correlations, selected, output_svg)
    return joined, correlations, selected
