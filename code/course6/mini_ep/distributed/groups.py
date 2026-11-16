"""Process-group setup helpers for the Mini-EP NCCL lessons.

Importing this module has no distributed side effects.  A process group is
created only when :func:`nccl_distributed` is entered, and only from a process
launched with ``torchrun``-style environment variables.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
import os

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class TorchrunEnvironment:
    """Rank metadata supplied to one worker process by ``torchrun``."""

    global_rank: int
    local_rank: int
    world_size: int


@dataclass(frozen=True)
class DistributedContext:
    """Initialized default Process Group metadata for one NCCL worker.

    ``global_rank`` identifies the process within the default Process Group.
    ``local_rank`` selects this process's CUDA device on the current node.
    ``world_size`` is the number of processes participating in every lesson-4
    collective.  Lesson 4 uses the default group as the EP group; subgroup
    construction and a distinct EP rank/size are introduced later.
    """

    global_rank: int
    local_rank: int
    world_size: int
    device: torch.device
    group: dist.ProcessGroup


def _read_required_int(
    environment: Mapping[str, str],
    name: str,
) -> int:
    try:
        raw_value = environment[name]
    except KeyError as error:
        raise RuntimeError(
            f"missing {name}; launch this program with torchrun"
        ) from error

    try:
        return int(raw_value)
    except ValueError as error:
        raise ValueError(
            f"{name} must be an integer, got {raw_value!r}"
        ) from error


def read_torchrun_environment(
    environment: Mapping[str, str] | None = None,
) -> TorchrunEnvironment:
    """Parse and validate ``RANK``, ``LOCAL_RANK``, and ``WORLD_SIZE``.

    Passing an explicit mapping makes the parser independently unit-testable.
    Omitting it reads the current process environment.
    """

    source = os.environ if environment is None else environment
    global_rank = _read_required_int(source, "RANK")
    local_rank = _read_required_int(source, "LOCAL_RANK")
    world_size = _read_required_int(source, "WORLD_SIZE")

    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if not 0 <= global_rank < world_size:
        raise ValueError(
            f"RANK must be in [0, {world_size}), got {global_rank}"
        )
    if local_rank < 0:
        raise ValueError(f"LOCAL_RANK must be non-negative, got {local_rank}")

    return TorchrunEnvironment(
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def _validate_nccl_runtime(settings: TorchrunEnvironment) -> torch.device:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable in this PyTorch build")
    if not dist.is_nccl_available():
        raise RuntimeError("the NCCL backend is unavailable in this PyTorch build")
    if not torch.cuda.is_available():
        raise RuntimeError("NCCL lesson requires CUDA, but CUDA is unavailable")

    device_count = torch.cuda.device_count()
    if settings.local_rank >= device_count:
        raise RuntimeError(
            "LOCAL_RANK must select a visible CUDA device, got "
            f"LOCAL_RANK={settings.local_rank} with {device_count} visible device(s)"
        )
    return torch.device("cuda", settings.local_rank)


@contextmanager
def nccl_distributed(
    *,
    timeout_seconds: int = 120,
) -> Iterator[DistributedContext]:
    """Initialize and later destroy the default NCCL Process Group.

    The rendezvous uses ``env://``.  ``torchrun`` supplies the rank, world-size,
    and rendezvous environment for each process.  Each process selects exactly
    one CUDA device using ``LOCAL_RANK`` before creating the Process Group.

    Args:
        timeout_seconds: Positive collective timeout in seconds.  The timeout
            bounds failures; it is not part of a performance measurement.

    Yields:
        Rank metadata and the default Process Group.  Every Tensor used with
        this NCCL group must live on ``context.device``.
    """

    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int):
        raise ValueError(
            "timeout_seconds must be a positive integer, got "
            f"{timeout_seconds!r}"
        )
    if timeout_seconds <= 0:
        raise ValueError(
            f"timeout_seconds must be positive, got {timeout_seconds}"
        )
    if dist.is_initialized():
        raise RuntimeError("the default Process Group is already initialized")

    settings = read_torchrun_environment()
    device = _validate_nccl_runtime(settings)
    torch.cuda.set_device(device)

    initialized_here = False
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=timeout_seconds),
            device_id=device,
        )
        initialized_here = True

        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        if global_rank != settings.global_rank or world_size != settings.world_size:
            raise RuntimeError(
                "initialized Process Group metadata does not match torchrun: "
                f"rank {global_rank} vs {settings.global_rank}, "
                f"world_size {world_size} vs {settings.world_size}"
            )

        yield DistributedContext(
            global_rank=global_rank,
            local_rank=settings.local_rank,
            world_size=world_size,
            device=device,
            group=dist.group.WORLD,
        )
    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()
