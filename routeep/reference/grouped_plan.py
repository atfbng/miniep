"""可独立阅读的 Grouped GEMM Host 规划 Reference。"""

import torch


def build_grouped_plan_reference(
    counts: list[int], device: torch.device, block_m: int, *, padding_rows: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """稳定 Expert 顺序的计数 [L] -> offsets [L+1] 和 tile 映射。

    参数：
        counts: 每个专家的行数，非负。
        block_m: 每个 tile 的行数，必须为正。
        device: 返回张量所在设备。

    返回：
        连续的 int64 张量。tile_experts 保存本地专家编号，tile_rows 保存 tile 在
        专家区间内的行偏移。可选的 padding_rows 会把所有 offset 向后平移，tile
        行偏移仍相对专家起点。空专家不产生 tile，输入没有 Top-K 维度。

    本 Reference 只使用 Python/PyTorch，不导入优化 Kernel。
    """
    if block_m < 1 or not counts or any(count < 0 for count in counts):
        raise ValueError(
            f"expected nonempty nonnegative counts and block_m>0, "
            f"got {counts=}, {block_m=}"
        )
    if not isinstance(padding_rows, int) or padding_rows < 0:
        raise ValueError(
            f"padding_rows must be a nonnegative integer, got {padding_rows!r}"
        )
    expert_offsets = [padding_rows]
    tile_experts: list[int] = []
    tile_rows: list[int] = []
    for expert, count in enumerate(counts):
        expert_offsets.append(expert_offsets[-1] + count)
        for tile in range((count + block_m - 1) // block_m):
            tile_experts.append(expert)
            tile_rows.append(tile * block_m)

    def tensor(values: list[int]) -> torch.Tensor:
        """把 Python 整数列表转成指定设备上的连续 int64 张量。"""
        return torch.tensor(values, dtype=torch.int64, device=device)

    return tensor(expert_offsets), tensor(tile_experts), tensor(tile_rows)
