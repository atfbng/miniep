"""Reference latency measurement for CPU and CUDA teaching experiments.

CPU operations are measured with ``time.perf_counter_ns``. CUDA operations
are measured with CUDA Events because kernel launches are asynchronous with
respect to the host. Synchronization happens only at measurement boundaries;
it is intentionally not inserted into the measured operation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import statistics
import time

import torch


@dataclass(frozen=True)
class LatencyStats:
    """Latency samples and summary statistics, all expressed in milliseconds."""

    samples_ms: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.samples_ms:
            raise ValueError("samples_ms must contain at least one sample")
        if any(
            not math.isfinite(sample) or sample < 0.0
            for sample in self.samples_ms
        ):
            raise ValueError("latency samples must be finite and non-negative")

    @property
    def repeats(self) -> int:
        """Return the number of measured repetitions."""

        return len(self.samples_ms)

    @property
    def median_ms(self) -> float:
        """Return the median latency in milliseconds."""

        return float(statistics.median(self.samples_ms))

    @property
    def mean_ms(self) -> float:
        """Return the arithmetic mean latency in milliseconds."""

        return float(statistics.fmean(self.samples_ms))

    @property
    def min_ms(self) -> float:
        """Return the minimum latency in milliseconds."""

        return min(self.samples_ms)

    @property
    def max_ms(self) -> float:
        """Return the maximum latency in milliseconds."""

        return max(self.samples_ms)


def _validate_measurement(
    device: torch.device,
    warmup: int,
    repeats: int,
) -> None:
    if device.type not in ("cpu", "cuda"):
        raise ValueError(
            f"device must be CPU or CUDA, got device type {device.type!r}"
        )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA timing was requested, but CUDA is unavailable")
        if (
            device.index is not None
            and not 0 <= device.index < torch.cuda.device_count()
        ):
            raise ValueError(
                f"CUDA device index must be in [0, {torch.cuda.device_count()}), "
                f"got {device.index}"
            )
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError(f"warmup must be a non-negative integer, got {warmup!r}")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError(f"repeats must be a positive integer, got {repeats!r}")


def measure_latency(
    operation: Callable[[], torch.Tensor],
    *,
    device: torch.device | str,
    warmup: int = 10,
    repeats: int = 100,
) -> LatencyStats:
    """Measure one Tensor operation after warmup.

    Args:
        operation: Zero-argument callable that launches the operation and
            returns its output Tensor. Inputs and outputs remain owned by the
            caller; this helper does not mutate or reuse their storage.
        device: CPU or CUDA device on which ``operation`` runs.
        warmup: Number of unmeasured calls. Warmup keeps initialization and
            first-use effects out of the steady-state samples.
        repeats: Number of measured calls.

    Returns:
        Per-call latency samples. CUDA samples measure GPU elapsed time between
        Events on the current stream, while CPU samples measure host wall time.

    Raises:
        ValueError: If the device type or repetition counts are invalid.
        RuntimeError: If CUDA timing is requested without an available GPU.
    """

    resolved_device = torch.device(device)
    _validate_measurement(resolved_device, warmup, repeats)

    if resolved_device.type == "cpu":
        with torch.inference_mode():
            for _ in range(warmup):
                operation()
            samples_ms = []
            for _ in range(repeats):
                started_ns = time.perf_counter_ns()
                operation()
                ended_ns = time.perf_counter_ns()
                samples_ms.append((ended_ns - started_ns) / 1_000_000.0)
        return LatencyStats(tuple(samples_ms))

    # Select the requested CUDA device explicitly. Otherwise Event.record()
    # would use the current device even when the operands live on cuda:1.
    with torch.cuda.device(resolved_device), torch.inference_mode():
        for _ in range(warmup):
            operation()

        # Synchronize once before recording so unrelated queued work does not
        # leak into the first sample. Each Event pair remains on the current
        # stream and therefore brackets exactly one operation launch.
        torch.cuda.synchronize(resolved_device)
        event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        for _ in range(repeats):
            started = torch.cuda.Event(enable_timing=True)
            ended = torch.cuda.Event(enable_timing=True)
            started.record()
            operation()
            ended.record()
            event_pairs.append((started, ended))

        # One synchronization drains all recorded work without adding a
        # device-wide synchronization to every measured operation.
        torch.cuda.synchronize(resolved_device)
        samples_ms = tuple(
            started.elapsed_time(ended) for started, ended in event_pairs
        )
    return LatencyStats(samples_ms)
