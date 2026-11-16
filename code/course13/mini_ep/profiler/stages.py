"""核心 NVTX 阶段标签，并提供可选的 CUDA Event 计时。"""

from contextlib import contextmanager
import os
import torch


class Stages:
    """作用：按阶段记录 NVTX range，并可选记录 CUDA Event 时间。

    参数：
        timings: 是否启用 CUDA Event 计时。``None`` 时读取环境变量
            ``MINIEP_STAGE_TIMING=1``；显式传入布尔值可覆盖环境变量。

    正常推理只推入/弹出 NVTX range；只有显式开启计时时才创建 Event，
    避免在热路径增加额外的 CUDA 事件和同步。阶段名称与课程主线一致，
    例如 ``ROUTER``、``DISPATCH``、``A2A_FORWARD``、``EXPERT``、
    ``A2A_RETURN`` 和 ``COMBINE``。
    """

    def __init__(self, timings: bool | None = None):
        """作用：初始化阶段计时器，并建立阶段名到 Event 对的存储。

        参数：
            timings: 是否启用 CUDA Event 计时。``None`` 表示读取
                ``MINIEP_STAGE_TIMING``；True/False 可显式覆盖环境变量。
        """
        # 正常路径只推 NVTX range。只有显式打开 profiling 时才记录 Event，
        # 避免给教学实现和正常推理增加额外事件与同步。
        self.timings = (
            os.environ.get("MINIEP_STAGE_TIMING") == "1" if timings is None else timings
        )
        self.events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

    @contextmanager
    def stage(self, name: str):
        """作用：用 ``with`` 包围一个可观测阶段。

        参数：
            name: NVTX range 和计时结果使用的阶段名称。

        产出：
            无。进入上下文时推入 NVTX range，并在开启计时时记录起始
            Event；退出时记录结束 Event 并弹出 range。

        即使阶段内部抛出异常，``finally`` 也会成对退出 NVTX range，
        避免后续嵌套层级错位。本函数不主动同步 GPU。
        """
        # NVTX 仍作为时间线标签；可选 Event 记录用于跨 Rank 阶段汇总。
        cuda = torch.cuda.is_available()
        if cuda:
            # NVTX 与 Event 都只在 CUDA 可用时创建。
            torch.cuda.nvtx.range_push(name)
            begin = torch.cuda.Event(enable_timing=True) if self.timings else None
            end = torch.cuda.Event(enable_timing=True) if self.timings else None
            if begin is not None:
                begin.record()
        else:
            begin = end = None
        try:
            # 执行阶段内的 Dispatch/All-to-All/Expert 等实际工作。
            yield
        finally:
            if end is not None:
                # Event 的 record 是异步的；真正的耗时读取在 milliseconds()。
                end.record()
                self.events.setdefault(name, []).append((begin, end))
            # 即使阶段内抛异常也成对退出 range，避免后续 NVTX 嵌套错位。
            if cuda:
                torch.cuda.nvtx.range_pop()

    def milliseconds(self) -> dict[str, list[float]]:
        """作用：返回各阶段每次执行的毫秒耗时。

        返回：
            ``{阶段名: [耗时, ...]}``。未开启计时或没有记录时返回空字典。
            多个同名阶段（例如逐 Chunk 调用）会按发生顺序保留多条记录。

        本方法会调用 ``torch.cuda.synchronize()``，因此不能在热路径中
        反复调用；只应在测量边界或 profiling 结束时使用。
        """
        if not self.timings or not self.events:
            return {}
        # 同步一次，确保所有 Event 已完成，之后逐对计算 elapsed_time。
        torch.cuda.synchronize()
        return {
            name: [begin.elapsed_time(end) for begin, end in rows]
            for name, rows in self.events.items()
        }
