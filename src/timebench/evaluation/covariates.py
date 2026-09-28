"""Known-covariate contract shared by TIME foundation-model runners."""

from dataclasses import dataclass

import numpy as np


COVARIATE_MODES = ("none", "future_included", "past_targets")
COVARIATE_FIELDS = ("feat_dynamic_real", "past_feat_dynamic_real")


@dataclass(frozen=True)
class CovariateWindow:
    """One known-covariate block split at the forecast boundary."""

    full: np.ndarray
    past: np.ndarray
    future: np.ndarray

    @property
    def channels(self) -> int:
        return int(self.full.shape[0])


def validate_covariate_mode(
    model: str,
    mode: str,
    *,
    supports_covariates: bool,
    supported_modes: tuple[str, ...] | None = None,
) -> str:
    """Validate a runner's explicit covariate mode and model capability."""
    normalized = str(mode).lower()
    if normalized not in COVARIATE_MODES:
        raise ValueError(
            f"Unknown covariate mode {mode!r}; expected one of {COVARIATE_MODES}"
        )
    allowed = (
        supported_modes
        if supported_modes is not None
        else (("none", "future_included") if supports_covariates else ("none",))
    )
    if normalized not in allowed:
        if not supports_covariates:
            raise ValueError(f"{model} does not consume covariates")
        raise ValueError(
            f"{model} does not support covariate_mode={normalized!r}; "
            f"expected one of {allowed}"
        )
    return normalized


def _covariate_field(entry: dict) -> str | None:
    present = [field for field in COVARIATE_FIELDS if field in entry]
    if len(present) > 1:
        raise ValueError(
            "Provide known covariates in exactly one field, not both "
            f"{COVARIATE_FIELDS}"
        )
    return present[0] if present else None


def _as_channels_by_time(values, field: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(
            f"{field} must have shape (channels, time), received {array.shape}"
        )
    return array


def extract_covariate_window(
    input_entry: dict,
    label_entry: dict,
    *,
    context_length: int,
    prediction_length: int,
    require_future: bool = True,
) -> CovariateWindow:
    """Extract an ``L+H`` known-covariate window or an ``L`` past window.

    GluonTS may retain a known dynamic feature as one complete block in the
    input entry or split it between the input and label entries. Both forms
    describe the same contract and are normalized here before context
    truncation. Non-finite observations become NaNs so capable backbones can
    apply the same missing-value mask used for target contexts.
    """
    field = _covariate_field(input_entry)
    if field is None:
        raise ValueError(
            "A covariate mode requires feat_dynamic_real or "
            "past_feat_dynamic_real on every evaluation input"
        )
    label_field = _covariate_field(label_entry)
    if label_field is not None and label_field != field:
        raise ValueError(
            f"Covariate field changes across the forecast boundary: {field} -> "
            f"{label_field}"
        )

    target = np.asarray(input_entry["target"])
    raw_context_length = int(target.shape[-1])
    requested_length = raw_context_length + int(prediction_length)
    input_covariates = _as_channels_by_time(input_entry[field], field)

    if require_future and input_covariates.shape[-1] == requested_length:
        complete = input_covariates
    elif (
        require_future
        and input_covariates.shape[-1] == raw_context_length
        and label_field is not None
    ):
        future = _as_channels_by_time(label_entry[field], field)
        if future.shape[0] != input_covariates.shape[0]:
            raise ValueError("Past and future covariate channel counts do not match")
        if future.shape[-1] != prediction_length:
            raise ValueError(
                f"Future covariates must have H={prediction_length} values, "
                f"received {future.shape[-1]}"
            )
        complete = np.concatenate([input_covariates, future], axis=-1)
    elif input_covariates.shape[-1] == raw_context_length and not require_future:
        complete = input_covariates
    else:
        expected = (
            f"exactly L+H={requested_length} values, or L={raw_context_length} "
            f"input values plus H={prediction_length} label values"
            if require_future
            else f"exactly L={raw_context_length} past values"
        )
        raise ValueError(
            f"{field} must contain {expected}; received "
            f"{input_covariates.shape[-1]} input values"
        )

    if not np.isfinite(complete).all():
        complete = np.where(np.isfinite(complete), complete, np.nan).astype(
            np.float32, copy=False
        )

    effective_context = min(raw_context_length, int(context_length))
    past = complete[:, :raw_context_length][:, -effective_context:]
    future = (
        complete[:, raw_context_length:requested_length]
        if require_future
        else np.empty((complete.shape[0], 0), dtype=complete.dtype)
    )
    full = np.concatenate([past, future], axis=-1)
    expected_length = effective_context + (prediction_length if require_future else 0)
    if full.shape[-1] != expected_length:
        raise ValueError("Internal covariate slicing did not preserve its time span")
    return CovariateWindow(full=full, past=past, future=future)


def validate_covariate_channels(
    windows: list[CovariateWindow],
    *,
    expected: int | None = None,
) -> int:
    """Require one channel schema within and across inference batches."""
    counts = {window.channels for window in windows}
    if len(counts) != 1:
        raise ValueError(f"Covariate channel count varies within a batch: {sorted(counts)}")
    channels = counts.pop()
    if expected is not None and channels != expected:
        raise ValueError(
            f"Covariate channel count changed across batches: {expected} -> {channels}"
        )
    return channels
