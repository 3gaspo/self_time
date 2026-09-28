"""
Feature extraction runner for batch processing datasets.

Usage:
    python -m timebench.feature.features_runner --dataset Water_Quality_Darwin/15T
    python -m timebench.feature.features_runner --all

Input formats:
    Preprocessed CSV files below ${TIME_DATA_ROOT}/processed_csv, or TIME's
    saved Arrow datasets below ${TIME_DATASET} with --input-format hf.

    Each CSV file has format:
    - First column: timestamp
    - Other columns: variates
"""

import argparse
import glob
import os
import time
from multiprocessing import Pool
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
from tqdm import tqdm

from timebench.feature.features import (
    dataset_feature_summary,
    extended_mstl_features,
    extended_stl_features,
    preprocess_for_tsfeatures,
    safe_parse_datetime,
    temporal_heterogeneity_frame,
    tsfeatures_with_uid_freq_map,
)

from timebench.evaluation.utils import (
    load_datasets_config,
    parse_dataset_key,
    get_test_length,
    find_dataset_config,
)
from timebench.paths import data_root, dataset_metadata_root, dataset_storage_root

# Default config path relative to this module
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "datasets.yaml"


# Define the desired output column order
FEATURE_COLUMNS_ORDER = [
    # Identifiers (for joining with results)
    "dataset_id",
    "series_name",
    "variate_name",
    "unique_id",
    # Meta features (from preprocess tags)
    "stationarity",
    "x_entropy",  # Entropy of raw series (predictability/signal-to-noise)
    # Distribution changes across chronological blocks
    "temporal_location_heterogeneity",
    "temporal_scale_heterogeneity",
    "temporal_frequency_heterogeneity",
    "temporal_heterogeneity",
    # Trend features (from STL)
    "trend_strength",
    "trend_stability",
    "trend_hurst",
    "trend_nonlinearity",
    "linearity",
    "curvature",
    # Seasonal features (from STL) - before residual features
    "seasonal_strength",
    "seasonal_corr",
    "seasonal_lumpiness",
    "seasonal_entropy",
    # Residual features (from STL)
    "e_acf1",
    "e_acf10",
    "e_diff1_acf1",
    "e_entropy",
    "e_kurtosis",
    "e_shapiro_w",
    "e_arch_lm",
    "spike",
    # Statistics (from preprocessing)
    "mean",
    "std",
    "missing_rate",
    "length",
    "period1",
    "period2",
    "period3",
    "p_strength1",
    "p_strength2",
    "p_strength3",
]


def _mstl_worker(args):
    """Top-level worker for Pool.imap — computes MSTL features for one variate."""
    uid, ts_values, primary_freq, periods = args
    feats = extended_mstl_features(ts_values, freq=primary_freq, periods=periods)
    return pd.DataFrame(feats, index=[uid])


def convert_multi_csv_to_panel(
    csv_dir: str,
    test_length: int | None = None,
    mode: str = "test"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert multiple CSV files from preprocess.py output to tsfeatures panel format.

    Args:
        csv_dir: Directory containing *.csv files
        test_length: Number of timesteps for test portion (required if mode="test")
        mode: Which portion to compute features on:
            - "full": Use entire series
            - "test": Use only the last `test_length` timesteps

    Returns:
        panel_df: DataFrame with columns ['unique_id', 'ds', 'y']
                  unique_id format: "{series_name}_{variate_name}"
        uid_info_df: DataFrame with columns ['unique_id', 'series_name', 'variate_name']
                     for later joining with features
    """
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))

    if not csv_files:
        raise ValueError(f"No *.csv files found in {csv_dir}")

    all_records = []
    uid_info_records = []

    for csv_path in csv_files:
        series_name = os.path.splitext(os.path.basename(csv_path))[0]  # e.g., "item_0" or any filename

        df = pd.read_csv(csv_path, parse_dates=[0])
        time_col = df.columns[0]
        var_cols = df.columns[1:].tolist()

        # Ensure time is sorted
        df = df.sort_values(time_col).reset_index(drop=True)

        # Filter based on mode
        if mode == "test":
            if test_length is None:
                raise ValueError("test_length must be provided when mode='test'")
            # Keep only the last test_length rows
            df = df.iloc[-test_length:].reset_index(drop=True)

        # Convert each variate to panel format
        for var in var_cols:
            unique_id = f"{series_name}_{var}"
            temp = pd.DataFrame({
                "unique_id": unique_id,
                "ds": safe_parse_datetime(df[time_col]),
                "y": df[var],
            })
            all_records.append(temp)

            # Record series_name and variate_name for this unique_id
            uid_info_records.append({
                "unique_id": unique_id,
                "series_name": series_name,
                "variate_name": var,
            })

    panel_df = pd.concat(all_records, ignore_index=True)
    uid_info_df = pd.DataFrame(uid_info_records)
    return panel_df, uid_info_df


def convert_hf_dataset_to_panel(
    dataset_dir: str,
    test_length: int | None = None,
    mode: str = "test",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert TIME's saved Arrow dataset directly to the feature panel format."""
    hf_dataset = datasets.load_from_disk(dataset_dir)
    all_records = []
    uid_info_records = []

    for item_index, item in enumerate(hf_dataset):
        target = np.asarray(item["target"], dtype=float)
        if target.ndim == 1:
            target = target[np.newaxis, :]
        if target.ndim != 2:
            raise ValueError(
                f"Expected one- or two-dimensional target, got shape {target.shape} "
                f"in {dataset_dir}"
            )
        if mode == "test":
            if test_length is None:
                raise ValueError("test_length must be provided when mode='test'")
            target = target[:, -test_length:]

        series_name = str(item.get("item_id", f"item_{item_index}"))
        variate_names = item.get("variate_names")
        if variate_names is not None:
            variate_names = list(variate_names)

        for variate_index, values in enumerate(target):
            if variate_names and variate_index < len(variate_names):
                variate_name = str(variate_names[variate_index])
            elif target.shape[0] == 1:
                variate_name = "target"
            else:
                variate_name = f"dim_{variate_index}"
            unique_id = f"{series_name}_{variate_name}"
            all_records.append(
                pd.DataFrame(
                    {
                        "unique_id": unique_id,
                        "ds": np.arange(len(values)),
                        "y": values,
                    }
                )
            )
            uid_info_records.append(
                {
                    "unique_id": unique_id,
                    "series_name": series_name,
                    "variate_name": variate_name,
                }
            )

    if not all_records:
        raise ValueError(f"No target series found in {dataset_dir}")
    return pd.concat(all_records, ignore_index=True), pd.DataFrame(uid_info_records)


def source_uid_info(input_dir: str, input_format: str) -> pd.DataFrame:
    """Read the expected variate identities without constructing a time panel."""
    records = []
    if input_format == "hf":
        hf_dataset = datasets.load_from_disk(input_dir)
        for item_index, item in enumerate(hf_dataset):
            target = np.asarray(item["target"])
            channel_count = 1 if target.ndim == 1 else target.shape[0]
            series_name = str(item.get("item_id", f"item_{item_index}"))
            variate_names = item.get("variate_names")
            if variate_names is not None:
                variate_names = list(variate_names)
            for variate_index in range(channel_count):
                if variate_names and variate_index < len(variate_names):
                    variate_name = str(variate_names[variate_index])
                elif channel_count == 1:
                    variate_name = "target"
                else:
                    variate_name = f"dim_{variate_index}"
                records.append(
                    {
                        "unique_id": f"{series_name}_{variate_name}",
                        "series_name": series_name,
                        "variate_name": variate_name,
                    }
                )
    else:
        csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
        if not csv_files:
            raise ValueError(f"No *.csv files found in {input_dir}")
        for csv_path in csv_files:
            series_name = Path(csv_path).stem
            for variate_name in pd.read_csv(csv_path, nrows=0).columns[1:]:
                records.append(
                    {
                        "unique_id": f"{series_name}_{variate_name}",
                        "series_name": series_name,
                        "variate_name": variate_name,
                    }
                )
    return pd.DataFrame(records)


def missing_feature_ids(
    expected_ids: set[str],
    existing_features: pd.DataFrame,
) -> set[str] | None:
    """Return missing source IDs, or ``None`` when the artifact is incompatible."""
    if "unique_id" not in existing_features:
        return None
    existing_ids = set(existing_features["unique_id"])
    if len(existing_features) != len(existing_ids) or not existing_ids <= expected_ids:
        return None
    return expected_ids - existing_ids


def compute_dataset_features(
    dataset_name: str,
    freq: str,
    input_dir: str,
    output_dir: str | None = None,
    test_length: int | None = None,
    split_mode: str = "test",
    decomp_method: str = "stl",
    input_format: str = "processed_csv",
    force: bool = False,
) -> Path:
    """
    Compute and save the full set of time series features for a given dataset.

    Pipeline:
        1. Load CSV files and convert to panel format.
        2. Filter data: if split_mode="test", keep only the last test_length timesteps per series.
        3. Preprocess the time series (interpolation, scaling, frequency analysis).
        4. Compute statistical and STL-based time series features.
        5. Merge all features with dataset statistics.
        6. Save the resulting feature DataFrame into the output directory.

    Args:
        dataset_name: The name of the dataset (e.g., "Water_Quality_Darwin", "ETTh1").
        freq: Frequency string (e.g., "H", "D", "15T").
        input_dir: Processed CSV directory or saved Arrow dataset directory.
        output_dir: Base directory for output files.
        test_length: Number of timesteps for test portion (required if split_mode="test").
        split_mode: Which portion to compute features on:
            - "full": full series
            - "test": test split only
        decomp_method: Seasonal-trend decomposition method:
            - "stl": single-period STL using the strongest FFT period (default)
            - "mstl": multi-period MSTL using the top-3 FFT periods
        input_format: "processed_csv" or "hf".
        force: Recompute the feature files even when both outputs already exist.

    Returns:
        Path to the dataset-level feature CSV saved below
        {output_dir}/{decomp_method}_features/{dataset}/{freq}/.
    """
    start = time.time()
    if output_dir is None:
        output_dir = str(dataset_metadata_root())

    # dataset_id for joining with results (format: "{dataset_name}/{freq}")
    dataset_id = f"{dataset_name}/{freq}"

    # Set the directories & paths (format: {decomp_method}_features/{dataset}/{freq}/)
    feature_dir = os.path.join(output_dir, f"{decomp_method}_features", dataset_name, freq)
    os.makedirs(feature_dir, exist_ok=True)

    output_csv_path = os.path.join(feature_dir, f'{split_mode}.csv')
    dataset_csv_path = os.path.join(feature_dir, f'{split_mode}_dataset.csv')

    existing_features = None
    repair_ids = None

    # Reuse complete artifacts and repair only source variates missing from a
    # partial artifact, such as rows previously removed for undefined optional
    # features.
    if not force and os.path.exists(output_csv_path) and os.path.exists(dataset_csv_path):
        expected_info = source_uid_info(input_dir, input_format)
        expected_ids = set(expected_info["unique_id"])
        existing_features = pd.read_csv(output_csv_path)
        repair_ids = missing_feature_ids(expected_ids, existing_features)
        if repair_ids == set():
            print(
                f"[Skip] Features for {dataset_name}/{freq} ({split_mode}) "
                f"already cover all {len(expected_ids)} source variates"
            )
            return Path(dataset_csv_path)
        if repair_ids is not None:
            print(
                f"[Repair] Features for {dataset_name}/{freq} ({split_mode}) are "
                f"missing {len(repair_ids)} of {len(expected_ids)} source variates"
            )
        else:
            print(
                f"[Recompute] Features for {dataset_name}/{freq} ({split_mode}) "
                "do not match the current source identities"
            )
            existing_features = None

    print(f"[Start] Processing {dataset_name}/{freq} ({split_mode}) from {input_dir}")
    if split_mode == "test":
        print(f"        test_length={test_length}")

    # Generate panel from CSV directory with appropriate filtering
    print("Loading CSV files and converting to panel format...")
    if input_format == "hf":
        panel, uid_info_df = convert_hf_dataset_to_panel(
            input_dir, test_length=test_length, mode=split_mode
        )
    else:
        panel, uid_info_df = convert_multi_csv_to_panel(
            input_dir, test_length=test_length, mode=split_mode
        )
    print(f"Loaded panel: {len(panel)} rows, {panel['unique_id'].nunique()} unique_ids")

    feature_panel = (
        panel[panel["unique_id"].isin(repair_ids)]
        if repair_ids is not None
        else panel
    )
    feature_uid_info = (
        uid_info_df[uid_info_df["unique_id"].isin(repair_ids)]
        if repair_ids is not None
        else uid_info_df
    )
    temporal_df = temporal_heterogeneity_frame(feature_panel)

    # Interpolate, Scale, Freq_analysis
    print("Running preprocessing...")
    series, stats_df = preprocess_for_tsfeatures(feature_panel, freq=freq)
    assert series['y'].isna().sum() == 0, "There are still NaNs in preprocessed series!"

    # Compute seasonal-trend decomposition features (trend, seasonal, residual)
    if decomp_method == "stl":
        # [STL] Single-period decomposition using the strongest FFT period (period1)
        uid_freq_map = stats_df.set_index('unique_id')['period1'].to_dict()
        features_df = tsfeatures_with_uid_freq_map(
            series,
            uid_freq_map=uid_freq_map,
            features=[extended_stl_features],
            scale=False,
        )
    else:
        # [MSTL] Multi-period decomposition using the top-3 FFT periods per variate
        uid_periods_map = {}
        for _, row in stats_df.iterrows():
            uid = row['unique_id']
            periods = []
            for col in ['period1', 'period2', 'period3']:
                if col in row.index and pd.notna(row[col]):
                    p = int(row[col])
                    if p > 1:
                        periods.append(p)
            uid_periods_map[uid] = periods

        mstl_args = []
        for uid, group in series.groupby('unique_id'):
            periods = uid_periods_map.get(uid, [])
            primary_freq = periods[0] if periods else 1
            mstl_args.append((uid, group['y'].values, primary_freq, periods))

        with Pool() as pool:
            feature_rows = list(tqdm(
                pool.imap(_mstl_worker, mstl_args),
                total=len(mstl_args),
                desc="Computing MSTL features",
            ))
        features_df = pd.concat(feature_rows).rename_axis('unique_id').reset_index()

    # Merge all features
    features_df = features_df.merge(stats_df, on='unique_id', how='left')
    features_df = features_df.merge(temporal_df, on='unique_id', how='left')

    # Add identifier columns (dataset_id, series_name, variate_name)
    features_df = features_df.merge(feature_uid_info, on='unique_id', how='left')
    features_df['dataset_id'] = dataset_id

    # Reorder columns according to FEATURE_COLUMNS_ORDER
    ordered_cols = [col for col in FEATURE_COLUMNS_ORDER if col in features_df.columns]
    # Add any remaining columns not in the order list
    remaining_cols = [col for col in features_df.columns if col not in ordered_cols]
    features_df = features_df[ordered_cols + remaining_cols]

    # Check for NaN values and remove rows with failed required features.
    # Some periods are unavailable at coarse frequencies, while seasonal
    # correlation is undefined when fewer than two nonconstant cycles exist.
    exclude_cols = [
        'period2',
        'period3',
        'p_strength2',
        'p_strength3',
        'seasonal_corr',
    ]
    check_cols = [c for c in features_df.columns if c not in exclude_cols and c != 'unique_id']

    nan_rows = features_df[check_cols].isna().any(axis=1)
    if nan_rows.sum() > 0:
        nan_uids = features_df.loc[nan_rows, 'unique_id'].tolist()
        # Find which features have NaN for each row
        nan_features = features_df.loc[nan_rows, check_cols].apply(
            lambda row: [c for c in check_cols if pd.isna(row[c])], axis=1
        )
        # Get unique NaN features across all rows
        all_nan_features = set()
        for feats in nan_features:
            all_nan_features.update(feats)
        print(f"[Warning] Removing {nan_rows.sum()} rows with NaN values: {nan_uids[:10]}{'...' if len(nan_uids) > 5 else ''}")
        print(f"          NaN features: {sorted(all_nan_features)}")
        features_df = features_df[~nan_rows]

    if existing_features is not None:
        features_df = pd.concat([existing_features, features_df], ignore_index=True)
        features_df = features_df.sort_values("unique_id")

    # Save all features
    features_df.to_csv(output_csv_path, index=False)
    dataset_feature_summary(features_df, panel).to_csv(dataset_csv_path, index=False)
    print(
        f"[Done] {dataset_name}/{freq} ({split_mode}): Saved {len(features_df)} "
        f"per-variate rows to {output_csv_path} and one dataset row to "
        f"{dataset_csv_path} (elapsed {time.time() - start:.2f}s)"
    )
    return Path(dataset_csv_path)


def write_dataset_feature_index(paths: list[Path], output_path: Path) -> None:
    """Combine dataset summaries and add descending heterogeneity ranks."""
    rows = []
    for path in paths:
        row = pd.read_csv(path)
        row["feature_split"] = path.stem.removesuffix("_dataset")
        rows.append(row)
    if not rows:
        return
    index = pd.concat(rows, ignore_index=True)
    for feature in ("temporal_heterogeneity", "spatial_heterogeneity"):
        index[f"{feature}_rank"] = index[feature].rank(
            method="min", ascending=False
        ).astype("Int64")
    index = index.sort_values("dataset_id")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(output_path, index=False)
    print(f"[Done] Dataset feature index: {output_path}")


def main():
    from timebench.pipeline.runtime_resources import log_selected_device
    log_selected_device("cpu", stage="diagnostics", component="feature_extraction")
    parser = argparse.ArgumentParser(
        description="Run tsfeatures extraction on preprocessed datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process one dataset from TIME_DATA_ROOT/processed_csv
    # Uses test_length from config/datasets.yaml
    python -m timebench.feature.features_runner --dataset Water_Quality_Darwin/15T

    # Process all datasets in config
    python -m timebench.feature.features_runner --all

    # Compute features on full series instead of last test_length timesteps
    python -m timebench.feature.features_runner --dataset Water_Quality_Darwin/15T --split full

    # Use multi-period MSTL decomposition instead of the default single-period STL
    python -m timebench.feature.features_runner --dataset Water_Quality_Darwin/15T --decomp mstl

    # Read the downloaded TIME Arrow datasets directly
    python -m timebench.feature.features_runner --all --input-format hf --split full
        """
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset key in format '{name}/{freq}' (e.g., 'Water_Quality_Darwin/15T')"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all datasets in the config file"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["full", "test"],
        help="Which portion to compute features on: 'full', 'test'"
    )
    parser.add_argument(
        "--decomp",
        type=str,
        default="stl",
        choices=["stl", "mstl"],
        help="Seasonal-trend decomposition method: 'stl' (single period, default) or 'mstl' (multiple periods)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to datasets.yaml config file (default: src/timebench/config/datasets.yaml)"
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        default=str(data_root() / "processed_csv"),
        help="Base directory for processed CSV files"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=str(dataset_storage_root()),
        help="Base directory for saved Arrow datasets"
    )
    parser.add_argument(
        "--input-format",
        choices=["processed_csv", "hf"],
        default="processed_csv",
        help="Read preprocessed CSV files or TIME's saved Arrow datasets"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(dataset_metadata_root()),
        help="Base directory for output files"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and replace existing feature artifacts",
    )

    args = parser.parse_args()

    # Load config
    config = load_datasets_config(args.config)
    datasets_config = config.get("datasets", {})

    if not datasets_config:
        raise ValueError(f"No datasets found in config file: {args.config}")

    if args.all:
        # Process all datasets in config
        dataset_summary_paths = []

        for dataset_key in tqdm(datasets_config.keys(), desc="Processing datasets", unit="dataset"):
            dataset_name, freq = parse_dataset_key(dataset_key)
            input_root = args.dataset_dir if args.input_format == "hf" else args.csv_dir
            dataset_input_dir = os.path.join(input_root, dataset_name, freq)

            if not os.path.isdir(dataset_input_dir):
                print(f"[Warning] Directory not found: {dataset_input_dir}, skipping {dataset_key}")
                continue

            # Get test_length from config
            dataset_cfg = datasets_config.get(dataset_key, {})
            test_length = get_test_length(dataset_cfg)

            # Validate test_length when mode is "test"
            if args.split == "test" and test_length is None:
                print(f"[Warning] test_length not found in config for {dataset_key}, skipping")
                continue

            # If test_length < 500, use full series instead
            effective_split_mode = args.split
            if args.split == "test" and test_length is not None and test_length < 500:
                print(f"[Info] test_length={test_length} < 500 for {dataset_key}, using full series instead")
                effective_split_mode = "full"

            dataset_summary_paths.append(compute_dataset_features(
                dataset_name=dataset_name,
                freq=freq,
                input_dir=dataset_input_dir,
                output_dir=args.output_dir,
                test_length=test_length,
                split_mode=effective_split_mode,
                decomp_method=args.decomp,
                input_format=args.input_format,
                force=args.force,
            ))

        write_dataset_feature_index(
            dataset_summary_paths,
            Path(args.output_dir)
            / f"{args.decomp}_features"
            / f"dataset_features_{args.split}.csv",
        )


    elif args.dataset:
        # Process single dataset
        dataset_key, freq, dataset_cfg = find_dataset_config(datasets_config, args.dataset)
        dataset_name, _ = parse_dataset_key(dataset_key)

        input_root = args.dataset_dir if args.input_format == "hf" else args.csv_dir
        dataset_input_dir = os.path.join(input_root, dataset_name, freq)

        if not os.path.isdir(dataset_input_dir):
            expected = (
                "saved Arrow dataset"
                if args.input_format == "hf"
                else "processed CSV files (*.csv)"
            )
            raise FileNotFoundError(
                f"Dataset directory not found: {dataset_input_dir}\n"
                f"Expected {expected} below the selected input root."
            )

        # Get test_length from config
        test_length = get_test_length(dataset_cfg)

        # Validate test_length when mode is "test"
        if args.split == "test" and test_length is None:
            raise ValueError(
                f"test_length not found in config for {dataset_key}. "
                f"Please add test_length to the config or use --split full."
            )

        # If test_length < 500, use full series instead
        effective_split_mode = args.split
        if args.split == "test" and test_length is not None and test_length < 500:
            print(f"[Info] test_length={test_length} < 500, using full series instead")
            effective_split_mode = "full"

        compute_dataset_features(
            dataset_name=dataset_name,
            freq=freq,
            input_dir=dataset_input_dir,
            output_dir=args.output_dir,
            test_length=test_length,
            split_mode=effective_split_mode,
            decomp_method=args.decomp,
            input_format=args.input_format,
            force=args.force,
        )
    else:
        parser.print_help()
        raise ValueError("You must provide either --dataset or --all")


if __name__ == "__main__":
    main()
