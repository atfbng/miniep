"""NVTX range helper for Mini-EP profiling lessons.

The range names used across the course are ``ROUTER``, ``DISPATCH``,
``A2A_FORWARD``, ``EXPERT``, ``A2A_RETURN`` and ``COMBINE``; chunked
pipelines add ``CHUNK_n_COMM`` and ``CHUNK_n_COMPUTE``.  Keeping the helper
here means a CPU-only run stays silent instead of failing on a machine without
CUDA.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    """Emit one NVTX range while the wrapped block runs.

    Args:
        name: Range name shown in Nsight Systems.

    Yields:
        ``None``.  Without CUDA the block runs unchanged and no NVTX push or
        pop happens, so the same code path is usable in CPU unit tests.
    """

    if not torch.cuda.is_available():
        yield
        return

    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
