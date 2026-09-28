"""
StatsForecast model wrappers for TIME benchmark.
Adapted from gift_eval implementation.

"""

from typing import Iterator, List, Optional

import numpy as np
import pandas as pd

from timebench.evaluation.metrics import fill_missing_history

from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive


# Default quantile levels aligned with GluonTS/gift_eval
DEFAULT_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def quantile_levels_to_intervals(quantile_levels: List[float]) -> List[int]:
    """
    Convert quantile levels to statsforecast interval levels.

    statsforecast uses symmetric prediction intervals specified by 'level'.
    For example, level=80 gives the 10th and 90th percentiles (lo-80, hi-80).

    Args:
        quantile_levels: List of quantile levels (e.g., [0.1, 0.2, ..., 0.9])

    Returns:
        List of unique interval levels for statsforecast
    """
    intervals = set()
    for q in quantile_levels:
        # Convert quantile to interval: q=0.1 or q=0.9 -> interval=80
        interval = round(200 * (max(q, 1 - q) - 0.5))
        if interval > 0:  # Skip 0 (mean)
            intervals.add(interval)
    return sorted(intervals)


def get_quantile_column_name(model_name: str, quantile_level: float) -> str:
    """
    Get the statsforecast column name for a given quantile level.

    Args:
        model_name: Name of the model (e.g., 'SeasonalNaive')
        quantile_level: Quantile level (e.g., 0.1, 0.9)

    Returns:
        Column name (e.g., 'SeasonalNaive-lo-80' for q=0.1)
    """
    if quantile_level == 0.5:
        return model_name  # Mean/median

    interval = round(200 * (max(quantile_level, 1 - quantile_level) - 0.5))
    side = "hi" if quantile_level > 0.5 else "lo"
    return f"{model_name}-{side}-{interval}"


class SeasonalNaiveForecast:
    """
    Wrapper for SeasonalNaive quantile forecasts.

    StatsForecast already provides the point forecast and prediction-interval
    bounds needed by TIME's quantile metrics, so retain those quantiles
    directly without Monte Carlo resampling.

    Attributes:
        quantiles: Forecasts with shape (num_quantiles, num_variates, pred_len)
    """

    def __init__(
        self,
        quantile_forecasts: dict,
        quantile_levels: List[float],
    ):
        """
        Initialize forecast wrapper from quantile predictions.

        Args:
            quantile_forecasts: Dict mapping quantile levels to forecast arrays
                               Each array has shape (num_variates, pred_len)
            quantile_levels: List of quantile levels used
        """
        self.quantile_levels = sorted(quantile_levels)
        self.quantile_forecasts = quantile_forecasts

        quantile_values = []
        for q in self.quantile_levels:
            qf = quantile_forecasts[q]
            if qf.ndim == 1:
                qf = qf[np.newaxis, :]
            quantile_values.append(qf)
        self._quantiles = np.stack(quantile_values, axis=0)

    @property
    def quantiles(self) -> np.ndarray:
        """Return direct quantile forecasts in ascending quantile order."""
        return self._quantiles


class SeasonalNaivePredictor:
    """
    Predictor wrapping statsforecast SeasonalNaive model with prediction intervals.

    Unlike the simple point forecast replication, this predictor uses
    statsforecast's built-in prediction interval estimation to generate
    true probabilistic forecasts.

    The Seasonal Naive method forecasts the value from the same season
    in the previous seasonal cycle. Prediction intervals are estimated
    based on historical seasonal errors.
    """

    def __init__(
        self,
        prediction_length: int,
        season_length: int,
        freq: str,
        quantile_levels: Optional[List[float]] = None,
    ):
        """
        Initialize SeasonalNaive predictor.

        Args:
            prediction_length: Number of time steps to forecast
            season_length: Seasonal period length (e.g., 7 for weekly, 24 for daily)
            freq: Frequency string (e.g., 'D', 'H', '15T')
            quantile_levels: List of quantile levels for prediction intervals.
                           Defaults to [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        """
        self.prediction_length = prediction_length
        self.season_length = season_length
        self.freq = freq
        self.quantile_levels = quantile_levels or DEFAULT_QUANTILE_LEVELS

        # Convert quantile levels to statsforecast intervals
        self.intervals = quantile_levels_to_intervals(self.quantile_levels)

        # Model name for column lookup
        self.model_name = "SeasonalNaive"

    def _create_dataframe(self, history: np.ndarray, unique_id: str) -> pd.DataFrame:
        """
        Create a pandas DataFrame in statsforecast format.

        Args:
            history: Historical values with shape (context_len,)
            unique_id: Unique identifier for this series

        Returns:
            DataFrame with columns: unique_id, ds, y
        """
        n = len(history)
        # Create a simple datetime index
        dates = pd.date_range(start='2000-01-01', periods=n, freq=self.freq)

        return pd.DataFrame({
            'unique_id': unique_id,
            'ds': dates,
            'y': history.astype(np.float32),
        })

    def _extract_quantile_forecasts(
        self,
        result_df: pd.DataFrame,
    ) -> dict:
        """
        Extract quantile forecasts from statsforecast result DataFrame.

        Args:
            result_df: DataFrame with forecast results from statsforecast

        Returns:
            Dict mapping quantile levels to forecast arrays
        """
        quantile_forecasts = {}

        for q in self.quantile_levels:
            col_name = get_quantile_column_name(self.model_name, q)
            if col_name in result_df.columns:
                quantile_forecasts[q] = result_df[col_name].values
            elif q == 0.5 and self.model_name in result_df.columns:
                # Use mean as 0.5 quantile
                quantile_forecasts[q] = result_df[self.model_name].values

        return quantile_forecasts

    def _forecast_single_variate(self, history: np.ndarray, unique_id: str) -> dict:
        """
        Generate quantile forecasts for a single univariate time series.

        Args:
            history: Historical values with shape (context_len,)
            unique_id: Unique identifier for this series

        Returns:
            Dict mapping quantile levels to forecast arrays of shape (pred_len,)
        """
        history = fill_missing_history(history)

        # Create DataFrame
        df = self._create_dataframe(history, unique_id)

        # Create StatsForecast model
        sf = StatsForecast(
            models=[SeasonalNaive(season_length=self.season_length)],
            freq=self.freq,
            n_jobs=1,
        )

        # Generate forecasts with prediction intervals
        result = sf.forecast(
            df=df,
            h=self.prediction_length,
            level=self.intervals,
        )

        # Extract quantile forecasts
        return self._extract_quantile_forecasts(result)

    def predict(self, dataset_input) -> Iterator[SeasonalNaiveForecast]:
        """
        Generate forecasts for all series in the dataset.

        Args:
            dataset_input: Iterable of data entries, each containing:
                - "target": Historical values with shape (context_len,) for univariate
                           or (num_variates, context_len) for multivariate

        Yields:
            SeasonalNaiveForecast objects with probabilistic predictions
        """
        for idx, entry in enumerate(dataset_input):
            target = entry["target"]

            # Handle univariate case
            if target.ndim == 1:
                quantile_forecasts = self._forecast_single_variate(
                    target, f"series_{idx}"
                )
                yield SeasonalNaiveForecast(
                    quantile_forecasts,
                    self.quantile_levels,
                )

            # Handle multivariate case - process each variate independently
            else:
                num_variates = target.shape[0]

                # Collect quantile forecasts for all variates
                all_quantile_forecasts = {q: [] for q in self.quantile_levels}

                for v in range(num_variates):
                    qf = self._forecast_single_variate(
                        target[v], f"series_{idx}_var_{v}"
                    )
                    for q in self.quantile_levels:
                        if q in qf:
                            all_quantile_forecasts[q].append(qf[q])

                # Stack variates: (num_variates, pred_len)
                combined_qf = {}
                for q in self.quantile_levels:
                    if all_quantile_forecasts[q]:
                        combined_qf[q] = np.stack(all_quantile_forecasts[q], axis=0)

                yield SeasonalNaiveForecast(
                    combined_qf,
                    self.quantile_levels,
                )
