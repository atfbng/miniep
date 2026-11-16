"""去重 EP 链路 EXPERT/COMBINE 阶段的 Triton 搬运与有序归约。

对应的 PyTorch Reference 位于 mini_ep/distributed/dedup.py：
    expand_owner_branches + compute_grouped_local_experts + reduce_owner_branches
        -> compute_owner_partials_triton
    combine_owner_partials -> combine_owner_partials_triton

Reference 先用 nonzero 物化 source-major 分支 [R,H]，再排序拷贝成 Expert-major，
算完写回 source-major，最后对 K 个 slot 各做一遍 FP32 index_select/where/add。
这里改为：
    1. 对 [U*K] 扁平 slot 做稳定排序（无效 slot 记 -1，作为 padding 排在最前），
       直接从接收到的唯一行 gather 成 Expert-major 输入；
    2. 复用原 Grouped GEMM / SwiGLU，padding 行不进入任何 tile；
    3. 一个 Kernel 按 slot 升序读取 Expert-major 输出，完成 owner 归约。
有效分支在每个专家内的相对顺序与 Reference 相同，加法顺序和舍入边界也相同，
因此与 Reference 逐元素一致。全程不读回 Host 计数（没有 nonzero）。

所有 Kernel 在调用方当前 CUDA Stream 上按顺序执行，不跨 Stream 复用缓冲。
"""

import torch
import triton
import triton.language as tl

from mini_ep.kernels.grouped_gemm.triton_impl import (
    _tile_config, build_device_plan, grouped_expert_mlp,
)


@triton.jit
def _gather_rows_kernel(
    Received, Origin, ValidStart, Packed, RECV_STRIDE,
    H: tl.constexpr, K: tl.constexpr, BH: tl.constexpr,
):
    """Packed[p] = Received[Origin[p] // K]；p < ValidStart[0] 的 padding 行不写。"""
    row = tl.program_id(0)
    if row >= tl.load(ValidStart):
        cols = tl.program_id(1) * BH + tl.arange(0, BH)
        source = tl.load(Origin + row) // K
        values = tl.load(Received + source * RECV_STRIDE + cols, cols < H)
        tl.store(Packed + row.to(tl.int64) * H + cols, values, cols < H)


@triton.jit
def _reduce_owner_kernel(
    Packed, Position, Ids, Output,
    H: tl.constexpr, K: tl.constexpr, BH: tl.constexpr,
):
    """每个唯一行按 slot 升序：分支舍入到模型 dtype -> FP32 累加 -> 舍入一次。"""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BH + tl.arange(0, BH)
    mask = cols < H
    total = tl.zeros((BH,), tl.float32)
    for slot in tl.static_range(K):
        flat = row * K + slot
        valid = tl.load(Ids + flat) >= 0
        position = tl.load(Position + flat)
        # 无效 slot 对应 padding 行（未初始化），用掩码读 0，与 Reference 的 where 相同。
        value = tl.load(Packed + position * H + cols, mask & valid, other=0)
        total = total + value.to(Output.dtype.element_ty).to(tl.float32)
    tl.store(Output + row * H + cols, total, mask)


@triton.jit
def _combine_owner_kernel(
    Returned, Restore, Output, N,
    H: tl.constexpr, P: tl.constexpr, BH: tl.constexpr,
):
    """每个原 Token 按 owner 升序做 FP32 累加，最后写回模型 dtype。"""
    token = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BH + tl.arange(0, BH)
    mask = cols < H
    total = tl.zeros((BH,), tl.float32)
    for owner in tl.static_range(P):
        row = tl.load(Restore + owner * N + token)
        value = tl.load(Returned + row * H + cols, mask & (row >= 0), other=0)
        total = total + value.to(tl.float32)
    tl.store(Output + token * H + cols, total, mask)


def _block_h(h: int) -> int:
    return min(triton.next_power_of_2(h), 1024)


def compute_owner_partials_triton(
    received: torch.Tensor, expert_ids: torch.Tensor, routing_weights: torch.Tensor,
    gate_up_weights: torch.Tensor, down_weights: torch.Tensor, *,
    return_dtype: torch.dtype,
) -> torch.Tensor:
    """目标卡：接收的唯一行 -> 本地 Expert -> 每个唯一行一个 owner 部分和。

    参数：
        received: source-major 唯一行 [U,H]，CUDA BF16/FP32；最后一维连续，
            行跨度可大于 H（例如 split_owner_rows 返回的字节缓冲视图）。
        expert_ids: [U,K] 全局专家编号，不属于本卡的 slot 为 -1。
        routing_weights: [U,K] FP32 权重，无效 slot 为 0。
        gate_up_weights/down_weights: 本卡权重 [L,2F,H]/[L,H,F]，dtype 同 received。
        return_dtype: down 投影加权后分支的舍入 dtype（FP32/BF16/FP16）。

    返回：
        [U,H] owner 部分和，dtype 同 received，行顺序与 received 相同（source-major），
        可直接作为返回 All-to-All 的发送缓冲。

    中间布局：
        local_ids [U*K]：扁平 slot 的本地专家号，无效 slot 为 -1；
        origin [U*K]：Expert-major 行 -> 扁平 slot，-1 padding 排在最前；
        packed [U*K,H]：只有 offsets[0] 之后的行被写入和计算。
    """
    u, k = expert_ids.shape
    h = received.shape[1]
    num_local = gate_up_weights.shape[0]
    model_dtype = received.dtype
    if u == 0:
        return received.new_empty((0, h))
    if received.stride(1) != 1:
        raise ValueError("received rows must have unit stride in the hidden dimension")
    flat_ids = expert_ids.reshape(-1).to(torch.int64)
    # 无效 slot 置 -1：稳定排序后成为 padding 前缀，Grouped 计划会跳过它们。
    local_ids = torch.where(flat_ids >= 0, flat_ids % num_local, -1)
    origin = torch.argsort(local_ids, stable=True)
    plan = build_device_plan(local_ids, origin, num_local, _tile_config(model_dtype, u * k)[0])
    packed = torch.empty((u * k, h), device=received.device, dtype=model_dtype)
    bh = _block_h(h)
    _gather_rows_kernel[(u * k, triton.cdiv(h, bh))](
        received, origin, plan.offsets, packed, received.stride(0), h, k, bh, num_warps=4,
    )
    packed_weights = routing_weights.reshape(-1).index_select(0, origin).contiguous()
    packed_output = grouped_expert_mlp(
        packed, gate_up_weights, down_weights, plan,
        row_weights=packed_weights, output_dtype=return_dtype, padding=True,
    )
    # position[u*K+slot]：该 slot 在 Expert-major 输出中的行号（origin 的逆置换）。
    position = torch.empty_like(origin)
    position[origin] = torch.arange(origin.numel(), device=origin.device)
    output = torch.empty((u, h), device=received.device, dtype=model_dtype)
    _reduce_owner_kernel[(u, triton.cdiv(h, bh))](
        packed_output, position, expert_ids.contiguous(), output, h, k, bh,
        num_warps=4, enable_fp_fusion=False,
    )
    return output


def combine_owner_partials_triton(
    returned: torch.Tensor, restore_index: torch.Tensor,
) -> torch.Tensor:
    """返回的 [U,H] owner 部分和 + [P,N] restore_index -> 原始顺序 [N,H]。

    与 combine_owner_partials 相同：按 owner 升序 FP32 累加，缺失 owner 贡献 0，
    最后转回 returned 的 dtype。
    """
    p, n = restore_index.shape
    h = returned.shape[1]
    output = torch.empty((n, h), device=returned.device, dtype=returned.dtype)
    if n == 0:
        return output
    if returned.shape[0] == 0:
        return output.zero_()
    bh = _block_h(h)
    _combine_owner_kernel[(n, triton.cdiv(h, bh))](
        returned.contiguous(), restore_index.contiguous(), output, n, h, p, bh,
        num_warps=4, enable_fp_fusion=False,
    )
    return output


# ---------------------------------------------------------------------------
# 固定容量去重 Dispatch：一个 Kernel 生成单一字节发送缓冲。
# ---------------------------------------------------------------------------

#: Triton 固定容量打包一次在单个 program 内扫描全部来源 Token，只用于短批次。
FIXED_OWNER_TRITON_MAX_TOKENS = 256


def owner_row_bytes(hidden_size: int, top_k: int, element_size: int) -> int:
    """字节缓冲每行长度：H 个激活 + K 个 int32 专家编号 + K 个 FP32 权重。"""
    return hidden_size * element_size + 8 * top_k


@triton.jit
def _fixed_owner_pack_kernel(
    X, Ids, Weights, TokenOut, IdOut, WeightOut, Restore, Counts,
    N, C, ROW_X, ROW_W,
    H: tl.constexpr, K: tl.constexpr, LOCAL: tl.constexpr,
    BN: tl.constexpr, BK: tl.constexpr, BH: tl.constexpr, BC: tl.constexpr,
):
    """program (owner r, 列块 j)：把选中 r 的 Token 按原顺序写到 r*C+位置。

    TokenOut/IdOut/WeightOut 是同一字节缓冲按 dtype 重新解释的视图，
    行跨度分别为 ROW_X（激活元素）和 ROW_W（4 字节元素）。j==0 的 program
    额外写 ID、权重、restore、计数和 padding 行的 ID=-1/权重=0。
    """
    owner = tl.program_id(0)
    block = tl.program_id(1)
    tok = tl.arange(0, BN)
    slot = tl.arange(0, BK)
    tmask = tok < N
    m2 = tmask[:, None] & (slot[None, :] < K)
    ids = tl.load(Ids + tok[:, None] * K + slot[None, :], m2, other=-1)
    owned = m2 & (ids >= 0) & (ids // LOCAL == owner)
    present = tl.max(owned.to(tl.int32), 1) > 0
    # 段内位置 = 原 Token 顺序上的前缀计数，与变长布局的段内顺序相同。
    pos = tl.cumsum(present.to(tl.int32), 0) - 1
    rows = (owner * C + pos).to(tl.int64)
    cols = block * BH + tl.arange(0, BH)
    cmask = cols < H
    xv = tl.load(X + tok[:, None] * H + cols[None, :], tmask[:, None] & cmask[None, :])
    tl.store(TokenOut + rows[:, None] * ROW_X + cols[None, :], xv,
             present[:, None] & cmask[None, :])
    if block == 0:
        count = tl.sum(present.to(tl.int32), 0)
        tl.store(Counts + owner, count)
        tl.store(Restore + owner * N + tok, tl.where(present, rows, -1), tmask)
        wv = tl.load(Weights + tok[:, None] * K + slot[None, :], m2, other=0.0)
        smask = present[:, None] & (slot[None, :] < K)
        tl.store(IdOut + rows[:, None] * ROW_W + slot[None, :],
                 tl.where(owned, ids, -1).to(tl.int32), smask)
        tl.store(WeightOut + rows[:, None] * ROW_W + slot[None, :],
                 tl.where(owned, wv, 0.0), smask)
        # padding 行：段内 [count, C) 的 ID=-1、权重=0；激活不初始化。
        for start in range(0, C, BC):
            pad = start + tl.arange(0, BC)
            pmask = (pad >= count) & (pad < C)
            prow = (owner * C + pad).to(tl.int64)
            pm2 = pmask[:, None] & (slot[None, :] < K)
            tl.store(IdOut + prow[:, None] * ROW_W + slot[None, :],
                     tl.full((BC, BK), -1, tl.int32), pm2)
            tl.store(WeightOut + prow[:, None] * ROW_W + slot[None, :],
                     tl.zeros((BC, BK), tl.float32), pm2)


def split_owner_rows(buffer: torch.Tensor, hidden_size: int, top_k: int, dtype: torch.dtype):
    """把 [R,row_bytes] uint8 缓冲解释为 (tokens [R,H], ids int32 [R,K], weights FP32 [R,K])。

    返回的是共享存储的跨步视图（行跨度为 row_bytes），不复制数据。
    """
    hb = hidden_size * torch.empty((), dtype=dtype).element_size()
    tokens = buffer[:, :hb].view(dtype)
    ids = buffer[:, hb:hb + 4 * top_k].view(torch.int32)
    weights = buffer[:, hb + 4 * top_k:hb + 8 * top_k].view(torch.float32)
    return tokens, ids, weights


def pack_token_owners_fixed_triton(
    tokens: torch.Tensor, expert_ids: torch.Tensor, routing_weights: torch.Tensor,
    num_local_experts: int, ep_size: int, capacity: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """固定容量去重打包的 Triton 版本，语义与 distributed.dedup.pack_token_owners_fixed 相同。

    参数：
        tokens: 连续 CUDA [N,H]，BF16/FP32/FP16。
        expert_ids: [N,K] int32/int64 全局专家编号；routing_weights: FP32 [N,K]。
        capacity: 每个目标的固定行数 C >= N，所有 Rank 相同。

    返回 (send_buffer, restore_index, counts)：
        send_buffer: uint8 [P*C, H*elem+8K]，一行 = 激活 | int32 ID | FP32 权重；
            用 split_owner_rows 取视图。padding 行激活未初始化，ID=-1、权重=0。
        restore_index: int64 [P,N]，(owner, Token) -> 行号 r*C+位置，缺失为 -1。
        counts: int32 [P] 每个目标的有效行数，仅诊断用。
    合成一个缓冲后，分发只需一次 all_to_all_single。
    """
    n, k = expert_ids.shape
    h = tokens.shape[1]
    if n > FIXED_OWNER_TRITON_MAX_TOKENS:
        raise ValueError(f"Triton fixed owner pack supports N<={FIXED_OWNER_TRITON_MAX_TOKENS}, got {n}")
    if capacity < max(n, 1):
        raise ValueError(f"fixed owner capacity={capacity} must be >= max(local tokens={n}, 1)")
    if not tokens.is_cuda or not tokens.is_contiguous() or tokens.ndim != 2:
        raise ValueError("tokens must be contiguous CUDA [N,H]")
    if routing_weights.dtype != torch.float32 or expert_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("expert_ids must be int32/int64 and routing_weights FP32")
    es = tokens.element_size()
    row_bytes = owner_row_bytes(h, k, es)
    if (h * es) % 4:
        raise ValueError(f"H*element_size={h * es} must be 4-byte aligned for the ID/weight fields")
    rows = ep_size * capacity
    buffer = torch.empty((rows, row_bytes), dtype=torch.uint8, device=tokens.device)
    x_out, id_out, w_out = split_owner_rows(buffer, h, k, tokens.dtype)
    restore = torch.empty((ep_size, n), dtype=torch.int64, device=tokens.device)
    counts = torch.empty(ep_size, dtype=torch.int32, device=tokens.device)
    bn = triton.next_power_of_2(max(n, 1))
    bh = max(min(triton.next_power_of_2(h), 8192 // bn), 16)
    _fixed_owner_pack_kernel[(ep_size, triton.cdiv(h, bh))](
        tokens, expert_ids.contiguous(), routing_weights.contiguous(),
        x_out, id_out, w_out, restore, counts,
        n, capacity, row_bytes // es, row_bytes // 4,
        h, k, num_local_experts, bn, triton.next_power_of_2(k), bh, 64,
        num_warps=4,
    )
    return buffer, restore, counts
