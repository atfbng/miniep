"""把原始 Token/Top-K 分支按“目标 EP Rank”重新打包。

本文件只改变行顺序，不做通信、不计算专家。输入是原始 Token 及其全局专家编号，
输出已经按目的 Rank 连续分段，后续可以直接交给 All-to-All。这样每个 Rank 只需
按 split 发送连续行，不必在通信前再做一次 Expert-major 复制。
"""

from dataclasses import dataclass
import torch
from mini_ep.distributed.topology import ExpertParallelTopology


@dataclass(frozen=True, slots=True)
class RankPackedTensors:
    """变长路径的打包结果：每个目的 Rank 的行连续存放。

    字段含义（N 为 Token 数，K 为 Top-K，H 为隐藏维，P 为 EP 大小）：
        packed_tokens: [N*K,H]，已经展开成“一个分支一行”，按目标 Rank 分组。
        packed_expert_ids: [N*K]，与 packed_tokens 同序的全局专家编号。
        packed_routing_weights: [N*K]，FP32 路由权重，与 packed_tokens 同序。
        pack_index: [N*K]，pack_index[p] 是打包行 p 在原始记录中的编号；
            原始记录按 token-major/slot-minor 排列，即记录 b = token*(K)+slot。
        restore_index: [N*K]，pack_index 的逆置换；通信回来后用它恢复原始顺序。
        send_splits: [P]，每个目的 Rank 收到的行数，单位是行，不是字节。

    所有字段与输入位于同一设备。
    """

    packed_tokens: torch.Tensor
    packed_expert_ids: torch.Tensor
    packed_routing_weights: torch.Tensor
    pack_index: torch.Tensor
    restore_index: torch.Tensor
    send_splits: torch.Tensor


@dataclass(frozen=True, slots=True)
class FixedRankPackedTensors:
    """固定容量路径的打包结果：每个目的 Rank 预留 C 行。

    字段含义（C 为 rows_per_rank）：
        packed_tokens: [P*C,H]，每个目的 Rank 一段；有效行之后补零。
        packed_expert_ids: [P*C]，填充行写 -1，表示“不是有效专家分支”。
        packed_routing_weights: [P*C]，填充行写 0。
        restore_index: [N*K]，原始记录 -> 返回缓冲中的固定行号。
        valid_rows: [P]，各目的 Rank 的有效行数，仅用于诊断。
        rows_per_rank: 整数 C。

    固定容量路径保证每次向每个 Rank 发送相同行数，因此通信前不需要读取计数。
    接收方必须用 expert_id=-1 屏蔽填充行；不能把填充行当作真实专家 0。
    """

    packed_tokens: torch.Tensor
    packed_expert_ids: torch.Tensor
    packed_routing_weights: torch.Tensor
    restore_index: torch.Tensor
    valid_rows: torch.Tensor
    rows_per_rank: int


def _validate(tokens, expert_ids, routing_weights):
    """检查打包函数共用的形状、dtype 和设备前提。

    只校验布局：tokens 为 [N,H]，expert_ids/routing_weights 为 [N,K]，权重为
    FP32。专家编号是否越界、权重是否有限由上层 MoE 负责。
    """
    if (
        tokens.ndim != 2
        or expert_ids.ndim != 2
        or tokens.shape[0] != expert_ids.shape[0]
    ):
        raise ValueError("expected tokens [N,H] and expert_ids [N,K]")
    if tokens.shape[1] < 1 or expert_ids.shape[1] < 1:
        raise ValueError("H and Top-K must be positive")
    if tokens.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError(f"packing requires floating tokens, got {tokens.dtype}")
    if expert_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("expert_ids must be int32/int64")
    if (
        routing_weights.shape != expert_ids.shape
        or routing_weights.dtype != torch.float32
    ):
        raise TypeError("routing_weights must be FP32 [N,K]")
    if expert_ids.device != tokens.device or routing_weights.device != tokens.device:
        raise ValueError("tokens, IDs and weights must share a device")


def pack_by_destination_rank(
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    topology: ExpertParallelTopology,
) -> RankPackedTensors:
    """把 [N,H] Token 和 [N,K] 路由打包成 Rank-major 的展开记录。

    参数：
        tokens: 原始 Token [N,H]，允许非连续。
        expert_ids: 全局专家编号 [N,K]，整数类型。
        routing_weights: FP32 路由权重 [N,K]。
        topology: 均匀专家放置，用于把专家编号映射到目标 Rank。

    返回：
        RankPackedTensors。同一目标 Rank 内保持原顺序：先按 Token，再按 slot。
        载荷 dtype 跟随输入，专家编号转为 int64，权重保持 FP32。

    本函数只做布局重排，不校验专家编号范围，也不做浮点归一化；这些由外层模型负责。
    """
    _validate(tokens, expert_ids, routing_weights)
    flat = expert_ids.reshape(-1).long()
    destinations = topology.destination_rank(flat)
    pack_index = torch.argsort(destinations, stable=True)
    return RankPackedTensors(
        tokens.index_select(0, pack_index // expert_ids.shape[1]).contiguous(),
        flat.index_select(0, pack_index),
        routing_weights.reshape(-1).index_select(0, pack_index),
        pack_index,
        torch.argsort(pack_index),
        torch.bincount(destinations, minlength=topology.ep_size),
    )


def pack_by_destination_rank_fixed(
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    topology: ExpertParallelTopology,
    rows_per_rank: int,
) -> FixedRankPackedTensors:
    """按固定容量 C 打包，每个目的 Rank 都得到 C 行。

    参数：
        tokens/expert_ids/routing_weights: 与变长路径相同。
        rows_per_rank: 每个目的 Rank 预留的行数 C，必须覆盖 max(1,N*K)。

    返回：
        FixedRankPackedTensors。有效行按目标 Rank 连续排列，其余位置为
        Token=0、权重=0、expert_id=-1 的填充。restore_index 把每个原始记录
        映射到回传缓冲中的固定行。

    计数和置换都在设备上完成，不读取 GPU 标量。逐记录循环版本见
    reference/fixed_dispatch.py；小批量解码还可使用对应的 Triton 打包器。
    """
    _validate(tokens, expert_ids, routing_weights)
    if rows_per_rank < max(1, expert_ids.numel()):
        raise ValueError(
            f"C={rows_per_rank} must cover max(1,N*K)={max(1,expert_ids.numel())}"
        )
    branches = expert_ids.numel()
    k = expert_ids.shape[1]
    flat = expert_ids.reshape(-1).long()
    destinations = topology.destination_rank(flat)
    counts = torch.bincount(destinations, minlength=topology.ep_size)
    order = torch.argsort(destinations, stable=True)
    sorted_destinations = destinations.index_select(0, order)
    starts = torch.cumsum(counts, dim=0) - counts
    within_sorted = torch.arange(
        branches, device=expert_ids.device
    ) - starts.index_select(0, sorted_destinations)
    within = torch.empty(branches, device=expert_ids.device, dtype=torch.int64)
    within.index_copy_(0, order, within_sorted)
    restore_index = destinations * rows_per_rank + within
    rows = topology.ep_size * rows_per_rank
    packed_tokens = tokens.new_zeros((rows, tokens.shape[1]))
    packed_ids = torch.full((rows,), -1, device=expert_ids.device, dtype=torch.int64)
    packed_weights = routing_weights.new_zeros(rows)
    destination_rows = sorted_destinations * rows_per_rank + within_sorted
    packed_tokens.index_copy_(
        0, destination_rows, tokens.index_select(0, order // k).contiguous()
    )
    packed_ids.index_copy_(0, destination_rows, flat.index_select(0, order))
    packed_weights.index_copy_(0, restore_index, routing_weights.reshape(-1))
    return FixedRankPackedTensors(
        packed_tokens.contiguous(),
        packed_ids.contiguous(),
        packed_weights,
        restore_index.contiguous(),
        counts,
        rows_per_rank,
    )
