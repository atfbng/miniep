"""供 Benchmark 驱动使用的 CUDA Event 计时工具。"""

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
        """在测量边界同步之后调用，返回各阶段毫秒数。"""
        return {
            name: begin.elapsed_time(end) for name, (begin, end) in self.events.items()
        }
