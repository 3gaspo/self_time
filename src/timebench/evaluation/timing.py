"""Shared timing for complete TIME test forecasting loops."""

from time import perf_counter


def _synchronize_accelerator() -> None:
    """Wait for queued CUDA work when the active model uses an accelerator."""
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.synchronize()


class EvaluationTimer:
    """Measure one test forecasting loop with accelerator synchronization."""

    def __init__(self) -> None:
        self._started: float | None = None

    def start(self) -> None:
        if self._started is not None:
            raise RuntimeError("evaluation timer is already running")
        _synchronize_accelerator()
        self._started = perf_counter()

    def stop(self) -> float:
        if self._started is None:
            raise RuntimeError("evaluation timer has not been started")
        _synchronize_accelerator()
        seconds = perf_counter() - self._started
        self._started = None
        return seconds
