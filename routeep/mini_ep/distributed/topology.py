"""定义 EP 组内的均匀专家放置，以及“全局专家编号 -> Rank/本地编号”的映射。

数据流中的位置：Router 输出全局专家编号后，Dispatch 先用本文件映射目标 Rank，
Expert 计算再用本文件把收到的全局编号换成当前 Rank 的本地权重下标。
"""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ExpertParallelTopology:
    """描述一个 EP 组内“每个 Rank 拥有连续一段专家”的放置方式。

    参数：
        num_experts: 全局专家总数 E，必须能被 ep_size 整除。
        ep_size: EP 组大小 P，即参与专家并行的 Rank 数。
        ep_rank: 当前 Rank 在 EP 组内的编号，范围 [0,P)；它不是全局 rank，
            也不是 CUDA 设备编号。

    映射规则：
        全局专家 e 属于 Rank e // L，本地编号为 e % L，其中 L = E // P。
        例如 E=128、P=4 时，每张卡拥有 32 个连续专家。

    构造时立即检查 E % P == 0，避免错误配置在后续除法中产生不均匀布局。
    """

    num_experts: int
    ep_size: int
    ep_rank: int

    def __post_init__(self) -> None:
        """校验专家总数、EP 大小和组内 Rank 编号是否合法。"""
        if self.ep_size < 1 or self.num_experts < 1 or self.num_experts % self.ep_size:
            raise ValueError(
                f"experts={self.num_experts} must be positive and divisible by ep_size={self.ep_size}"
            )
        if not 0 <= self.ep_rank < self.ep_size:
            raise ValueError(f"ep_rank={self.ep_rank} outside [0,{self.ep_size})")

    @property
    def num_local_experts(self) -> int:
        """每个 EP Rank 拥有的专家数 L = E // P。"""

        return self.num_experts // self.ep_size

    def destination_rank(self, expert_ids: torch.Tensor) -> torch.Tensor:
        """把全局专家编号映射为负责它的 EP Rank。

        参数：
            expert_ids: 任意形状的整数 Tensor，元素为全局专家编号 [0,E)。

        返回：
            与输入同形状的 int64 Tensor，值为 expert_id // L。本函数不发起通信。
        """

        if expert_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"expert_ids must be int32/int64, got {expert_ids.dtype}")
        # 全局专家编号 → 所属 EP rank，例如 E=128、P=4 时每卡 32 个专家。
        return expert_ids.to(torch.int64) // self.num_local_experts

    def local_expert_id(
        self, expert_ids: torch.Tensor, *, padding: bool = False
    ) -> torch.Tensor:
        """把全局专家编号映射为当前 Rank 权重栈中的本地下标。

        参数：
            expert_ids: 已到达当前 Rank 的全局专家编号。
            padding: 固定容量 A2A 路径设为 True。此时 -1 是填充标记，必须继续
                保持为 -1，不能被映射成有效的本地专家 0。

        返回：
            与输入同形状的本地专家编号，有效范围 [0,L)；padding=True 时填充行仍为 -1。

        实现说明：L>1 时使用带符号取余，fmod(-1,L) 仍为 -1；L=1 时取余会把 -1
        变成 0，因此单独使用 clamp_max(0) 保留哨兵。
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
        """返回当前 EP Rank 负责的源 Token 半开区间 [start,end)。

        参数：
            tokens: 源 Token 总数 N。

        返回：
            (start, end)。按 EP 维度均匀切分；即使 N 不能被 P 整除，也保证每一行
            恰好属于一个 Rank。该函数不发起通信。
        """

        return (
            tokens * self.ep_rank // self.ep_size,
            tokens * (self.ep_rank + 1) // self.ep_size,
        )
