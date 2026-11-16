"""Combine Kernel：把 Rank-major 的加权分支结果恢复成原始 Token 顺序。

这里不发起 NCCL 归约，也不再次乘路由权重；输入分支已经加权。实现保留
“先模型 dtype、再 FP32 累加、最后模型 dtype”的舍入顺序。
"""

import torch
import triton
import triton.language as tl
from mini_ep.distributed.topology import ExpertParallelTopology


@triton.jit
def combine_kernel(
    Returned,
    Restore,
    Ids,
    Output,
    N,
    H: tl.constexpr,
    K: tl.constexpr,
    P: tl.constexpr,
    LOCAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """一个 Combine Triton program：为 BLOCK 个输出元素恢复原 Token 顺序。

    参数：
        Returned: [R,H] 已加权分支结果。
        Restore: int64 [N*K]，原始记录 -> 返回行号。
        Ids: [N,K] 全局专家编号，用于判断分支属于哪个 owner。
        Output: [N,H] 输出。
        N/H/K/P/LOCAL/BLOCK: Token 数、隐藏维、Top-K、EP 大小、本地专家数、块大小。

    每个 program 按 owner Rank 升序、slot 升序累加；分支先舍入到输出 dtype，
    owner 部分和再舍入一次，最后做 FP32 合并。
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token, col = offsets // H, offsets % H
    valid = offsets < N * H
    total = tl.full((BLOCK,), 0.0, tl.float32)
    # 分支已经乘过路由权重。这里保留分支/owner 的模型 dtype 舍入，
    # 再按 EP Rank 升序做 FP32 合并，不使用 NCCL 归约。
    for owner in tl.static_range(P):
        partial = tl.full((BLOCK,), 0.0, tl.float32)
        for slot in tl.static_range(K):
            branch = token * K + slot
            packed = tl.load(Restore + branch, valid, 0)
            expert = tl.load(Ids + branch, valid, 0)
            value = tl.load(Returned + packed * H + col, valid, 0)
            rounded = value.to(Output.dtype.element_ty).to(tl.float32)
            partial = partial + tl.where(expert // LOCAL == owner, rounded, 0.0)
        total = total + partial.to(Output.dtype.element_ty).to(tl.float32)
    tl.store(Output + offsets, total, valid)


def combine_triton(
    returned: torch.Tensor,
    restore_index: torch.Tensor,
    expert_ids: torch.Tensor,
    topology: ExpertParallelTopology,
    *,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    """把 Rank-major 的 [R,H] 加权结果恢复为原始 [N,H]。

    参数：
        returned: 加权分支结果 [R,H]；变长路径 R=N*K，固定容量路径 R=P*C。
        restore_index: int64 [N*K]，原始记录 -> 返回行号。
        expert_ids: int64 [N,K]，用于确定每个 Token 的 Top-K owner 顺序。
        topology: 提供 EP 大小，用于按 owner Rank 升序合并。
        model_dtype: 最终输出 dtype。

    返回：
        原顺序 [N,H]，与输入同设备。

    所有张量必须位于同一 CUDA 设备；非连续输入会先物化。路由权重已经应用，
    本 Kernel 不会再次相乘。
    """
    if model_dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(f"model dtype must be FP32/BF16, got {model_dtype}")
    if returned.ndim != 2 or expert_ids.ndim != 2 or returned.shape[1] < 1:
        raise ValueError("expected returned [R,H], expert_ids [N,K], H>0")
    n, k = expert_ids.shape
    if k < 1 or restore_index.shape != (n * k,) or restore_index.dtype != torch.int64:
        raise ValueError("restore_index must be int64 [N*K], K>0")
    if expert_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("expert IDs must be integers")
    if returned.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError("unsupported return transport dtype")
    if not returned.is_cuda or any(
        t.device != returned.device for t in (restore_index, expert_ids)
    ):
        raise ValueError("returned values and metadata must share a CUDA device")
    output = torch.empty(
        (n, returned.shape[1]), device=returned.device, dtype=model_dtype
    )
    if n:
        combine_kernel[(triton.cdiv(n * returned.shape[1], 256),)](
            returned.contiguous(),
            restore_index.contiguous(),
            expert_ids.contiguous(),
            output,
            n,
            returned.shape[1],
            k,
            topology.ep_size,
            topology.num_local_experts,
            256,
            enable_fp_fusion=False,
        )
    return output
