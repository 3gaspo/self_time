"""Past-only transformations of a target history used as covariate channels."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


SELF_AUGMENTATIONS = (
    "sqrt_abs_x_norm",
    "sign_x_norm",
    "x",
    "constant_0",
    "constant_1",
    "sqrt_abs_x_norm_plus_sign_x_norm",
)


def validate_self_augmentations(names: Sequence[str] | None) -> tuple[str, ...]:
    """Return one ordered augmentation list using only canonical names."""

    selected = tuple(str(name) for name in (names or ()))
    unknown = [name for name in selected if name not in SELF_AUGMENTATIONS]
    if unknown:
        raise ValueError(
            f"Unknown self-augmentations {unknown}; expected values from "
            f"{SELF_AUGMENTATIONS}"
        )
    return selected


def _masked_constant(reference: np.ndarray, value: float) -> np.ndarray:
    channel = np.full(reference.shape, value, dtype=np.float32)
    channel[np.isnan(reference)] = np.nan
    return channel


def build_past_self_covariates(
    x: np.ndarray,
    x_norm: np.ndarray,
    names: Sequence[str] | None,
) -> np.ndarray | None:
    """Build ordered past-only channels from one visible model context.

    ``x`` and ``x_norm`` are shaped ``(target_variates, history)``. Each
    requested transformation contributes one channel per target variate, in
    augmentation-major then variate-major order. No future value is produced.
    """

    selected = validate_self_augmentations(names)
    if not selected:
        return None

    raw = np.asarray(x, dtype=np.float32)
    normalized = np.asarray(x_norm, dtype=np.float32)
    if raw.ndim != 2 or normalized.shape != raw.shape:
        raise ValueError(
            "x and x_norm must share shape (target_variates, history)"
        )
    if np.isinf(raw).any() or np.isinf(normalized).any():
        raise ValueError("Self-augmentation inputs must not contain infinity")

    channels: list[np.ndarray] = []
    for name in selected:
        if name == "sqrt_abs_x_norm":
            values = np.sqrt(np.abs(normalized))
        elif name == "sign_x_norm":
            values = np.sign(normalized)
        elif name == "x":
            values = raw.copy()
        elif name == "constant_0":
            values = _masked_constant(raw, 0.0)
        elif name == "constant_1":
            values = _masked_constant(raw, 1.0)
        else:
            values = np.sqrt(np.abs(normalized)) + np.sign(normalized)
        channels.append(np.asarray(values, dtype=np.float32))

    result = np.concatenate(channels, axis=0)
    if np.isinf(result).any():
        raise ValueError("Self-augmentation channels must not contain infinity")
    return result
