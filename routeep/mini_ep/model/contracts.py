"""定义一次 EP 前向所需的 Tensor 形状/layout 契约。

本文件只做校验，不依赖 Triton 或任何 Kernel；在进入集合通信前快速失败，
避免错误布局在多个 Rank 之间传播。
"""

import torch
from mini_ep.distributed.topology import ExpertParallelTopology


def validate_inputs(
    tokens: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topology: ExpertParallelTopology,
) -> None:
    """作用：校验一次 EP 前向所需的形状、dtype、设备和权重布局。

    参数：
        tokens: 本来源 Rank 的 Token ``[N,H]``，支持 FP32/BF16。
        ids: 全局专家编号 ``[N,K]``，dtype 为 int32/int64。
        weights: FP32 路由权重 ``[N,K]``。
        w13: 本地 gate/up 权重 ``[L,2F,H]``。
        w2: 本地 down 权重 ``[L,H,F]``。
        topology: 均匀专家放置，提供全局专家数和本地专家数。

    返回：
        无返回值；所有检查通过时正常返回。

    所有 Tensor 必须位于同一设备。校验失败时抛出带实际值和期望条件的
    ``ValueError``/``TypeError``，从而在进入集合通信前快速失败。
    """
    # Token 与 Top-K 元数据必须是二维，且每个 Token 具有相同数量的分支。
    if (
        tokens.ndim != 2
        or ids.ndim != 2
        or ids.shape != weights.shape
        or ids.shape[0] != tokens.shape[0]
    ):
        raise ValueError(
            f"expected [N,H], [N,K], [N,K], got {tokens.shape}, {ids.shape}, {weights.shape}"
        )
    # Top-K 至少为 1，且不能超过全局专家数（允许路由到全部专家）。
    if not 1 <= ids.shape[1] <= topology.num_experts:
        raise ValueError(f"K={ids.shape[1]} must be in [1,{topology.num_experts}]")
    # 专家编号使用整数；权重固定为 FP32，避免 Combine 的累加精度被输入改变。
    if ids.dtype not in (torch.int32, torch.int64) or weights.dtype != torch.float32:
        raise TypeError("IDs must be int32/int64; routing weights must be FP32")
    # 当前教学实现只覆盖 FP32 和 BF16 激活。
    if tokens.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(f"only FP32/BF16 supported, got {tokens.dtype}")
    # 分布式集合通信要求所有参与张量位于同一设备。
    if any(t.device != tokens.device for t in (ids, weights, w13, w2)):
        raise ValueError("all tensors must share a device")
    # Expert 权重与输入激活使用相同 dtype，GEMM 中不做隐式类型转换。
    if w13.dtype != tokens.dtype or w2.dtype != tokens.dtype:
        raise TypeError("weights and tokens must share dtype")
    l, h = topology.num_local_experts, tokens.shape[1]
    # w13 的 [2F] 必须能对半拆成 gate/up；w2 的形状由 H 和 F 唯一确定。
    if (
        w13.ndim != 3
        or w2.ndim != 3
        or h < 1
        or w13.shape[1] < 2
        or w13.shape[0] != l
        or w13.shape[2] != h
        or w13.shape[1] % 2
        or w2.shape != (l, h, w13.shape[1] // 2)
    ):
        raise ValueError(
            f"expected w13 [L,2F,H], w2 [L,H,F]; L={l}, H={h}, got {w13.shape}, {w2.shape}"
        )
    # 全局专家编号必须落在 [0, E)。空输入不触发 min/max，避免对空 Tensor 归约。
    if ids.numel() and (
        ids.min().item() < 0 or ids.max().item() >= topology.num_experts
    ):
        raise ValueError(f"expert IDs outside [0,{topology.num_experts})")
    # 权重需要可有限求和；负权重会使 Combine 失去凸组合语义。
    if not torch.isfinite(weights).all().item() or (weights < 0).any().item():
        raise ValueError("routing weights must be finite and nonnegative")
