"""Audit shared TIME source series and configured evaluation windows."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from timebench.evaluation.data import (
    DEFAULT_CONFIG_PATH,
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)
from timebench.evaluation.utils import get_available_terms
from timebench.paths import dataset_metadata_root


DEFAULT_CONTEXT_PROFILES = {
    "full": None,
    "seasonal_naive": None,
    "chronos2": 8192,
    "ts_icl": 4096,
    "chronos_bolt": 2048,
}

COUNT_FIELDS = (
    "generated_queries",
    "channel_windows",
    "values",
    "finite_values",
    "nan_values",
    "positive_infinity_values",
    "negative_infinity_values",
    "nonfinite_values",
    "nonfinite_channel_windows",
    "constant_channel_windows",
    "all_nonfinite_channel_windows",
)

SUMMARY_FIELDS = (
    "dataset",
    "frequency",
    "term",
    "scope",
    "window_config",
    "context_limit",
    "prediction_length",
    *COUNT_FIELDS,
)

EVENT_FIELDS = (
    "dataset",
    "frequency",
    "term",
    "item_id",
    "window_index",
    "scope",
    "window_config",
    "context_limit",
    "raw_context_length",
    "window_length",
    "prediction_length",
    "field",
    "channel_index",
    "channel_name",
    "source_start_position",
    "source_end_position",
    "source_start_timestamp",
    "source_end_timestamp",
    "finite_values",
    "nan_values",
    "positive_infinity_values",
    "negative_infinity_values",
    "nonfinite_values",
    "constant",
    "all_nonfinite",
    "constant_value",
)

POSITION_FIELDS = (
    "dataset",
    "frequency",
    "item_id",
    "field",
    "channel_index",
    "channel_name",
    "source_position",
    "timestamp",
    "value_kind",
)

MODEL_CONTEXT_FIELDS = (
    "profile_name",
    "context_limit",
    "window_config_prefix",
)


def _log(message: str) -> None:
    print(f"{datetime.now().astimezone().isoformat(timespec='seconds')} | {message}")


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_context_profiles(values: Iterable[str] | None) -> dict[str, int | None]:
    if not values:
        return dict(DEFAULT_CONTEXT_PROFILES)

    profiles: dict[str, int | None] = {}
    for value in values:
        name, separator, raw_limit = value.partition("=")
        name = name.strip()
        if not name or name in profiles:
            raise ValueError(f"Invalid or duplicate context profile: {value!r}")
        if not separator:
            if name != "full":
                raise ValueError(
                    f"Context profile {value!r} needs NAME=LENGTH; only full has no limit"
                )
            profiles[name] = None
            continue
        limit = int(raw_limit)
        if limit <= 0:
            raise ValueError(f"Context length must be positive: {value!r}")
        profiles[name] = limit
    return profiles


def _channels(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2:
        raise ValueError(f"Expected target shape (channels, time), received {array.shape}")
    return array


def _channel_names(entry: dict, channels: int) -> list[str]:
    names = entry.get("variate_names")
    if names is not None and len(names) == channels:
        return [str(name) for name in names]
    if channels == 1:
        return ["target"]
    return [f"dim_{index}" for index in range(channels)]


def _timestamp(start, frequency: str, source_position: int) -> str:
    try:
        if isinstance(start, pd.Period):
            return str(start + source_position)
        timestamp = pd.Timestamp(start)
        return str(timestamp + source_position * pd.tseries.frequencies.to_offset(frequency))
    except (TypeError, ValueError, OverflowError):
        return ""


def _empty_counts() -> dict[str, int]:
    return {field: 0 for field in COUNT_FIELDS}


def _add_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for field in COUNT_FIELDS:
        target[field] += int(source[field])


def _prefix(mask: np.ndarray) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))


def _channel_state(values: np.ndarray) -> dict[str, np.ndarray]:
    nan = np.isnan(values)
    positive_infinity = np.isposinf(values)
    negative_infinity = np.isneginf(values)
    finite = np.isfinite(values)
    breaks = np.zeros(len(values), dtype=bool)
    if len(values) > 1:
        breaks[1:] = (~finite[:-1]) | (~finite[1:]) | (values[:-1] != values[1:])
    return {
        "values": values,
        "nan": _prefix(nan),
        "positive_infinity": _prefix(positive_infinity),
        "negative_infinity": _prefix(negative_infinity),
        "breaks": _prefix(breaks),
    }


def _interval_counts(
    state: dict[str, np.ndarray], start: int, end: int
) -> tuple[dict[str, int], bool, bool, float | None]:
    length = end - start
    nan_values = int(state["nan"][end] - state["nan"][start])
    positive_infinity_values = int(
        state["positive_infinity"][end] - state["positive_infinity"][start]
    )
    negative_infinity_values = int(
        state["negative_infinity"][end] - state["negative_infinity"][start]
    )
    nonfinite_values = nan_values + positive_infinity_values + negative_infinity_values
    finite_values = length - nonfinite_values
    transitions = (
        int(state["breaks"][end] - state["breaks"][start + 1])
        if length > 1
        else 0
    )
    constant = bool(length > 0 and nonfinite_values == 0 and transitions == 0)
    all_nonfinite = bool(length > 0 and finite_values == 0)
    counts = _empty_counts()
    counts.update(
        {
            "channel_windows": 1,
            "values": length,
            "finite_values": finite_values,
            "nan_values": nan_values,
            "positive_infinity_values": positive_infinity_values,
            "negative_infinity_values": negative_infinity_values,
            "nonfinite_values": nonfinite_values,
            "nonfinite_channel_windows": int(nonfinite_values > 0),
            "constant_channel_windows": int(constant),
            "all_nonfinite_channel_windows": int(all_nonfinite),
        }
    )
    constant_value = float(state["values"][start]) if constant else None
    return counts, constant, all_nonfinite, constant_value


def _window_config(scope: str, context_limit: int | None, horizon: int) -> str:
    if scope == "query":
        return f"H={horizon}"
    length = "full" if context_limit is None else str(context_limit)
    return f"L={length},H={horizon}"


def _model_context_rows(context_profiles: dict[str, int | None]) -> list[dict]:
    return [
        {
            "profile_name": name,
            "context_limit": "" if limit is None else limit,
            "window_config_prefix": "L=full" if limit is None else f"L={limit}",
        }
        for name, limit in context_profiles.items()
    ]


def audit_time_windows(
    *,
    config_path: Path | None,
    output_dir: Path,
    context_profiles: dict[str, int | None],
    selected_datasets: set[str] | None = None,
) -> dict:
    """Audit each source once and each distinct ``(L, H)`` window once."""
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "window_events.csv"
    positions_path = output_dir / "nonfinite_positions.csv"
    task_summary_path = output_dir / "task_summary.csv"
    dataset_summary_path = output_dir / "dataset_summary.csv"
    model_contexts_path = output_dir / "model_contexts.csv"
    manifest_path = output_dir / "audit_manifest.json"

    resolved_config_path = config_path or DEFAULT_CONFIG_PATH
    config = load_dataset_config(resolved_config_path)
    datasets_config = config.get("datasets", {})
    dataset_keys = [
        key for key in datasets_config if selected_datasets is None or key in selected_datasets
    ]
    if selected_datasets is not None:
        missing = sorted(selected_datasets.difference(dataset_keys))
        if missing:
            raise ValueError(f"Datasets are absent from the TIME config: {missing}")

    distinct_context_limits = list(dict.fromkeys(context_profiles.values()))
    model_context_rows = _model_context_rows(context_profiles)
    _write_csv(model_contexts_path, MODEL_CONTEXT_FIELDS, model_context_rows)

    launch_id = os.environ.get("TIME_LAUNCH_ID")
    started_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1,
        "report": "time_window_audit",
        "status": "running",
        "launch_id": launch_id,
        "started_at": started_at,
        "updated_at": started_at,
        "config_path": str(resolved_config_path.resolve()),
        "datasets": dataset_keys,
        "context_profiles": context_profiles,
        "distinct_context_limits": distinct_context_limits,
        "definitions": {
            "query": "Configured TIME test window with source interval [context_end, context_end + H).",
            "context": "Source interval [max(0, context_end - L), context_end), with full starting at zero.",
            "constant": "A non-empty channel window whose values are all finite and exactly equal.",
            "position": "Zero-based position in the original saved-Arrow target series.",
            "deduplication": "Source positions occur once and model profiles sharing L reuse one L-H window row.",
        },
        "artifacts": [
            events_path.name,
            positions_path.name,
            task_summary_path.name,
            dataset_summary_path.name,
            model_contexts_path.name,
        ],
    }
    _write_json(manifest_path, manifest)

    summaries: dict[tuple, dict[str, int]] = defaultdict(_empty_counts)
    summary_metadata: dict[tuple, dict] = {}

    try:
        with events_path.open("w", newline="", encoding="utf-8") as events_file, positions_path.open(
            "w", newline="", encoding="utf-8"
        ) as positions_file:
            event_writer = csv.DictWriter(events_file, fieldnames=EVENT_FIELDS)
            position_writer = csv.DictWriter(positions_file, fieldnames=POSITION_FIELDS)
            event_writer.writeheader()
            position_writer.writeheader()

            for dataset_number, dataset_key in enumerate(dataset_keys, start=1):
                terms = get_available_terms(dataset_key, config)
                first_settings = get_dataset_settings(dataset_key, terms[0], config)
                dataset = Dataset(
                    name=dataset_key,
                    term=terms[0],
                    to_univariate=False,
                    prediction_length=int(first_settings["prediction_length"]),
                    test_length=int(first_settings["test_length"]),
                    val_length=int(first_settings["val_length"]),
                )
                _log(f"dataset {dataset_number}/{len(dataset_keys)} {dataset_key} terms={terms}")

                term_settings = []
                for term in terms:
                    settings = get_dataset_settings(dataset_key, term, config)
                    horizon = int(settings["prediction_length"])
                    test_length = int(settings["test_length"])
                    if horizon <= 0 or test_length < horizon:
                        raise ValueError(
                            f"Invalid H={horizon}, test_length={test_length} for {dataset_key}/{term}"
                        )
                    term_settings.append((term, horizon, test_length, test_length // horizon))

                dataset_query_counts = {term: 0 for term in terms}
                for item_index, entry in enumerate(dataset.hf_dataset):
                    target = _channels(entry["target"])
                    names = _channel_names(entry, target.shape[0])
                    item_id = str(entry.get("item_id", f"item_{item_index}"))
                    start = entry.get("start")
                    frequency = str(entry.get("freq", dataset.freq))
                    channel_states = [_channel_state(values) for values in target]

                    for channel_index, values in enumerate(target):
                        for source_position in np.flatnonzero(~np.isfinite(values)):
                            value = values[source_position]
                            value_kind = (
                                "nan"
                                if np.isnan(value)
                                else "positive_infinity"
                                if np.isposinf(value)
                                else "negative_infinity"
                            )
                            position_writer.writerow(
                                {
                                    "dataset": dataset_key.rpartition("/")[0],
                                    "frequency": frequency,
                                    "item_id": item_id,
                                    "field": "target",
                                    "channel_index": channel_index,
                                    "channel_name": names[channel_index],
                                    "source_position": int(source_position),
                                    "timestamp": _timestamp(start, frequency, int(source_position)),
                                    "value_kind": value_kind,
                                }
                            )

                    for term, horizon, test_length, windows in term_settings:
                        if test_length > target.shape[-1]:
                            raise ValueError(
                                f"test_length={test_length} exceeds series length={target.shape[-1]} "
                                f"for {dataset_key}/{term}/{item_id}"
                            )
                        test_start = target.shape[-1] - test_length
                        for window_index in range(windows):
                            context_end = test_start + window_index * horizon
                            intervals = [("query", None, context_end, context_end + horizon)]
                            intervals.extend(
                                (
                                    "context",
                                    limit,
                                    0 if limit is None else max(0, context_end - limit),
                                    context_end,
                                )
                                for limit in distinct_context_limits
                            )

                            for scope, limit, source_start, source_end in intervals:
                                window_config = _window_config(scope, limit, horizon)
                                key = (dataset_key, term, scope, window_config)
                                if key not in summary_metadata:
                                    summary_metadata[key] = {
                                        "dataset": dataset_key.rpartition("/")[0],
                                        "frequency": frequency,
                                        "term": term,
                                        "scope": scope,
                                        "window_config": window_config,
                                        "context_limit": "" if limit is None else limit,
                                        "prediction_length": horizon,
                                    }
                                summaries[key]["generated_queries"] += 1

                                for channel_index, state in enumerate(channel_states):
                                    counts, constant, all_nonfinite, constant_value = _interval_counts(
                                        state, source_start, source_end
                                    )
                                    _add_counts(summaries[key], counts)
                                    if counts["nonfinite_values"] == 0 and not constant:
                                        continue
                                    event_writer.writerow(
                                        {
                                            **summary_metadata[key],
                                            "item_id": item_id,
                                            "window_index": window_index,
                                            "raw_context_length": context_end,
                                            "window_length": source_end - source_start,
                                            "field": "target",
                                            "channel_index": channel_index,
                                            "channel_name": names[channel_index],
                                            "source_start_position": source_start,
                                            "source_end_position": source_end - 1,
                                            "source_start_timestamp": _timestamp(start, frequency, source_start),
                                            "source_end_timestamp": _timestamp(start, frequency, source_end - 1),
                                            "finite_values": counts["finite_values"],
                                            "nan_values": counts["nan_values"],
                                            "positive_infinity_values": counts["positive_infinity_values"],
                                            "negative_infinity_values": counts["negative_infinity_values"],
                                            "nonfinite_values": counts["nonfinite_values"],
                                            "constant": str(constant).lower(),
                                            "all_nonfinite": str(all_nonfinite).lower(),
                                            "constant_value": "" if constant_value is None else constant_value,
                                        }
                                    )
                            dataset_query_counts[term] += 1

                for term in terms:
                    _log(f"task {dataset_key}/{term} audited queries={dataset_query_counts[term]}")

        task_rows = [
            {**summary_metadata[key], **summaries[key]} for key in sorted(summaries)
        ]
        _write_csv(task_summary_path, SUMMARY_FIELDS, task_rows)

        dataset_counts: dict[tuple, dict[str, int]] = defaultdict(_empty_counts)
        dataset_metadata: dict[tuple, dict] = {}
        for row in task_rows:
            key = (
                row["dataset"],
                row["frequency"],
                row["scope"],
                row["window_config"],
                row["context_limit"],
                row["prediction_length"],
            )
            dataset_metadata[key] = {
                "dataset": row["dataset"],
                "frequency": row["frequency"],
                "term": "all",
                "scope": row["scope"],
                "window_config": row["window_config"],
                "context_limit": row["context_limit"],
                "prediction_length": row["prediction_length"],
            }
            _add_counts(dataset_counts[key], row)
        dataset_rows = [
            {**dataset_metadata[key], **dataset_counts[key]}
            for key in sorted(dataset_counts)
        ]
        _write_csv(dataset_summary_path, SUMMARY_FIELDS, dataset_rows)

        totals = _empty_counts()
        for row in task_rows:
            _add_counts(totals, row)
        completed_at = datetime.now(timezone.utc).isoformat()
        manifest.update(
            {
                "status": "completed",
                "updated_at": completed_at,
                "completed_at": completed_at,
                "task_summary_rows": len(task_rows),
                "dataset_summary_rows": len(dataset_rows),
                "window_configuration_totals": totals,
            }
        )
        _write_json(manifest_path, manifest)
        _log(
            f"audit completed output={output_dir} nonfinite_values="
            f"{totals['nonfinite_values']} constant_channel_windows="
            f"{totals['constant_channel_windows']}"
        )
        return manifest
    except Exception as error:
        failed_at = datetime.now(timezone.utc).isoformat()
        manifest.update(
            {
                "status": "interrupted",
                "updated_at": failed_at,
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )
        _write_json(manifest_path, manifest)
        raise


def main() -> None:
    from timebench.pipeline.runtime_resources import log_selected_device
    log_selected_device("cpu", stage="diagnostics", component="window_audit")
    parser = argparse.ArgumentParser(
        description="Audit shared TIME source series and configured L-H windows."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=dataset_metadata_root() / "window_audit",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Audit one configured dataset/frequency key; repeat to select several.",
    )
    parser.add_argument(
        "--context-profile",
        action="append",
        default=None,
        help="Context profile NAME=LENGTH; use full without a length.",
    )
    args = parser.parse_args()
    audit_time_windows(
        config_path=args.config,
        output_dir=args.output_dir,
        context_profiles=_parse_context_profiles(args.context_profile),
        selected_datasets=None if args.dataset is None else set(args.dataset),
    )


if __name__ == "__main__":
    main()
