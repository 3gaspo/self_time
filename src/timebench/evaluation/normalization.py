"""Per-instance target normalization for foundation-model ablations."""

from dataclasses import dataclass

import numpy as np


INSTANCE_NORMALIZATION_MODES = ("none", "zscore")


@dataclass(frozen=True)
class InstanceNormalizer:
    """Z-score one forecast input independently along its time axis."""

    location: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, context: np.ndarray) -> "InstanceNormalizer":
        values = np.asarray(context)
        location = np.nanmean(values, axis=-1, keepdims=True)
        scale = np.nanstd(values, axis=-1, keepdims=True, ddof=0)
        scale = np.where(scale > 0, scale, 1.0)
        return cls(location=location, scale=scale)

    def transform(self, context: np.ndarray) -> np.ndarray:
        values = np.asarray(context)
        dtype = np.result_type(values.dtype, np.float32)
        return ((values - self.location) / self.scale).astype(dtype, copy=False)

    def inverse_quantiles(self, quantiles: np.ndarray) -> np.ndarray:
        values = np.asarray(quantiles)
        location = self.location
        scale = self.scale
        while location.ndim < values.ndim:
            location = np.expand_dims(location, axis=0)
            scale = np.expand_dims(scale, axis=0)
        return values * scale + location


def normalize_instance(
    context: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, InstanceNormalizer | None]:
    """Normalize one context and return the inverse transform owner."""

    if mode not in INSTANCE_NORMALIZATION_MODES:
        raise ValueError(
            f"instance_normalization must be one of {INSTANCE_NORMALIZATION_MODES}"
        )
    values = np.asarray(context)
    if mode == "none":
        return values, None
    normalizer = InstanceNormalizer.fit(values)
    return normalizer.transform(values), normalizer
