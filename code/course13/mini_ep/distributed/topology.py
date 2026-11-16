"""TP/EP 边界上的均匀专家放置与归属映射。"""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ExpertParallelTopology:
    """作用：描述一个 EP 组内的均匀专家放置。

    参数：
        num_experts: 全局专家总数 E，必须能被 ``ep_size`` 整除。
        ep_size: EP 组大小 P，即参与专家并行的 Rank 数。
        ep_rank: 当前 Rank 在 EP 组内的编号，范围 ``[0,P)``；它不是
            全局 rank，也不是 CUDA 设备编号。

    全局专家 ``e`` 属于 Rank ``e // local_experts``，本地编号为
    ``e % local_experts``。构造时立即检查 ``E % P == 0``，避免后续
    除法在错误配置下静默产生不均匀布局。
    """

    num_experts: int
    ep_size: int
    ep_rank: int

    def __post_init__(self) -> None:
        """作用：校验专家总数与 EP 大小，并检查组内 Rank 编号是否合法。"""
        if self.ep_size < 1 or self.num_experts < 1 or self.num_experts % self.ep_size:
            raise ValueError(
                f"experts={self.num_experts} must be positive and divisible by ep_size={self.ep_size}"
            )
        if not 0 <= self.ep_rank < self.ep_size:
            raise ValueError(f"ep_rank={self.ep_rank} outside [0,{self.ep_size})")

    @property
    def num_local_experts(self) -> int:
        """作用：返回每个 EP Rank 拥有的完整专家数 ``L = E // P``。"""

        return self.num_experts // self.ep_size

    def destination_rank(self, expert_ids: torch.Tensor) -> torch.Tensor:
        """作用：把全局专家编号映射为目的 EP Rank。

        参数：
            expert_ids: 任意形状的整数 Tensor，元素为全局专家编号
                ``[0,E)``。

        返回：
            与输入同形状的整数 Tensor，每个元素是负责该专家的 EP Rank，
            即 ``expert_id // L``。本函数不发起通信。
        """

        if expert_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"expert_ids must be int32/int64, got {expert_ids.dtype}")
        # 全局专家编号 → 所属 EP rank，例如 E=128、P=4 时每卡 32 个专家。
        return expert_ids.to(torch.int64) // self.num_local_experts

    def local_expert_id(
        self, expert_ids: torch.Tensor, *, padding: bool = False
    ) -> torch.Tensor:
        """作用：把全局专家编号映射为本地权重栈下标。

        参数：
            expert_ids: 已按 EP Rank 接收到的全局专家编号 Tensor。
            padding: 固定容量 A2A 路径下设为 True。此时 ``-1`` 是填充
                标记，必须保持为 ``-1``，不能映射成有效的本地专家 0。

        返回：
            与输入同形状的本地专家编号；有效编号范围为 ``[0,L)``。
            当 ``padding=True`` 时，填充行保持 ``-1``。

        实现通过取模得到本地编号。``L=1`` 时 ``fmod(-1,1)=0`` 会丢失
        哨兵，因此单独使用 ``clamp_max(0)``；``L>1`` 时带符号取余可以
        保留 ``-1``。
        """

        # 普通变长路径无需 padding mask；仅固定容量需要保留 -1。
        if expert_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"expert_ids must be int32/int64, got {expert_ids.dtype}")
        expert_ids = expert_ids.to(torch.int64)
        local_experts = self.num_local_experts
        if not padding:
            return expert_ids % local_experts
        # L=1 需要单独处理：fmod(-1,1)=0 会把 padding 哨兵误当成专家 0。
        if local_experts == 1:
            return expert_ids.clamp_max(0)
        # L>1 时，带符号取余会保留 -1 这个 padding 哨兵。
        return torch.fmod(expert_ids, local_experts)

    def token_range(self, tokens: int) -> tuple[int, int]:
        """作用：返回当前 EP Rank 负责的源 Token 左闭右开区间 ``[start,end)``。

        参数：
            tokens: 需要划分的源 Token 总数 N。

        返回：
            ``(start, end)``。本方法在 EP 维度均匀切分输入行；即使 N
            不能被 P 整除，也保证每行恰好属于一个 Rank。
        """

        return (
            tokens * self.ep_rank // self.ep_size,
            tokens * (self.ep_rank + 1) // self.ep_size,
        )
