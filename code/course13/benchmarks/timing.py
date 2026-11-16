"""CUDA events collected only by the benchmark driver."""

from contextlib import contextmanager
import torch
from mini_ep.profiler.stages import Stages


class TimedStages(Stages):
    def __init__(self):
        super().__init__(timings=False)
        self.events: dict = {}

    @contextmanager
    def stage(self, name: str):
        with super().stage(name):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            try:
                yield
            finally:
                end.record()
                self.events[name] = (begin, end)

    def milliseconds(self) -> dict[str, float]:
        """Call after synchronization at the measurement boundary."""
        return {
            name: begin.elapsed_time(end) for name, (begin, end) in self.events.items()
        }
