"""描述 Grouped GEMM 的设备端 tile 元数据。

计划把“Expert-major 行”切成固定大小的行 tile，每个 tile 只属于一个专家。
GPU 规划器直接生成这些 int64 表，后续 Kernel 读取时不需要 Host 回读计数。
"""

from dataclasses import dataclass
import torch


@dataclass(frozen=True, slots=True)
class GroupedGemmPlan:
    """Expert-major 计划：设备端连续的专家边界和 tile 映射。

    字段：
        offsets: int64 [L+1]，每个专家在打包行中的半开区间；可选 padding 前缀
            也会被跳过。
        tile_experts: int64 [Q]，每个行 tile 所属的专家编号；无效尾部为 -1。
        tile_rows: int64 [Q]，tile 在专家区间内的行偏移。
        block_m: 行 tile 大小。
        num_rows: 包含 padding 的总行数。

    这些字段都是设备端张量，Kernel 可直接使用，不需要读取 GPU 标量。
    """

    offsets: torch.Tensor
    tile_experts: torch.Tensor
    tile_rows: torch.Tensor
    block_m: int
    num_rows: int
