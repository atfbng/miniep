"""稳定专家打包、tile 元数据、两次投影，以及接收顺序恢复。

与第十一课一致，``origin`` 把打包行映射回 source-major 接收行；
``expert_offsets``、``tile_experts`` 和 ``tile_rows`` 描述 Grouped GEMM。
元数据在稳定排序后完全于 CUDA 上构造，延续第九课“整数元数据与浮点
Token 搬运分离”的做法。Qwen 使用 SwiGLU 替代教学版 GELU；路由权重
仍然在本地专家计算之后应用。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from mini_ep.kernels.grouped_gemm.plan import GroupedGemmPlan

from mini_ep.kernels.swiglu_triton import swiglu_activation_triton


@triton.jit
def _plan_kernel(
    Ids,
    Order,
    Offsets,
    TileExperts,
    TileRows,
    R,
    L: tl.constexpr,
    ID_STRIDE: tl.constexpr,
    BM: tl.constexpr,
    CAPACITY,
    SEARCH_STEPS: tl.constexpr,
    BE: tl.constexpr,
    BT: tl.constexpr,
):
    """作用：由单个 CTA 根据稳定 argsort 构造专家边界和有界 tile 映射。

    参数：
        Ids: source-major 本地专家编号 ``[R]``，可能包含 -1 padding。
        Order: 稳定 argsort 得到的置换 ``[R]``。
        Offsets: 输出专家边界 ``[L+1]``。
        TileExperts: 输出每个行 tile 的专家编号 ``[CAPACITY]``。
        TileRows: 输出每个行 tile 在专家区间内的行偏移 ``[CAPACITY]``。
        R: 有效接收行数。
        L: 本地专家数，编译期常量。
        ID_STRIDE: Ids 第 0 维的 stride，支持非连续编号。
        BM: 行 tile 大小。
        CAPACITY: tile 表容量。
        SEARCH_STEPS: 专家边界二分搜索步数。
        BE: 专家向量容量（2 的幂）。
        BT: tile 向量容量（2 的幂）。
    """
    experts = tl.arange(0, BE)
    # 1. 按专家求 Expert-major 边界；offsets[e] 是排序序列中第一个
    #    ID>=e 的位置，offsets[L] 即有效行总数。
    #    这里直接在稳定置换 Order 上二分，不生成排序后的 ID 缓冲，
    #    也不使用 histogram 原子操作或 device->Host 计数拷贝。
    lo = tl.full((BE,), 0, tl.int32)
    hi = tl.full((BE,), R, tl.int32)
    for _ in range(SEARCH_STEPS):
        mid = (lo + hi) // 2
        row = tl.load(Order + mid, mid < R, other=0)
        value = tl.load(Ids + row * ID_STRIDE, mid < R, other=L)
        smaller = value < experts
        lo = tl.where(smaller, mid + 1, lo)
        hi = tl.where(smaller, hi, mid)
    offsets = lo
    tl.store(Offsets + experts, offsets, experts <= L)
    # 2. 专家 e 贡献 ceil(count[e]/BM) 个输出行 tile。
    next_offset = tl.gather(offsets, tl.minimum(experts + 1, BE - 1), 0)
    tiles = tl.where(experts < L, tl.cdiv(next_offset - offsets, BM), 0)
    # tile_ends[e] 是专家 e 最后一个 tile 之后的后缀和边界。
    tile_ends = tl.cumsum(tiles)
    # 所有专家的 tile 数满足 sum ceil(count[e]/BM) <= ceil(R/BM)+L-1，
    # 因此 CAPACITY 足以容纳有效 tile；尾部多余 tile 标记 expert=-1，
    # GEMM 会在读取权重或激活之前跳过它们。
    # 3. 用二分查找定位每个 tile 所属专家，并生成 Expert/行偏移表。
    tile = tl.arange(0, BT)
    left = tl.full((BT,), 0, tl.int32)
    right = tl.full((BT,), L, tl.int32)
    # 找到第一个满足 tile_ends[e] > tile 的专家 e。
    for _ in range((L + 1).bit_length()):
        mid = (left + right) // 2
        end = tl.gather(tile_ends, mid, 0)
        smaller = end <= tile
        left = tl.where(smaller, mid + 1, left)
        right = tl.where(smaller, right, mid)
    expert = left
    # 该专家 tile 区间的起始 tile 编号；再乘 BM 得到行内偏移。
    previous = tl.gather(tile_ends, tl.maximum(expert - 1, 0), 0)
    start = tl.where(expert == 0, 0, previous)
    tl.store(TileExperts + tile, tl.where(expert < L, expert, -1), tile < CAPACITY)
    tl.store(TileRows + tile, (tile - start) * BM, tile < CAPACITY)


def build_device_plan(
    local_ids: torch.Tensor, order: torch.Tensor, experts: int, block_m: int
) -> GroupedGemmPlan:
    """作用：为 R>0 的接收行构造纯 GPU 端 Grouped GEMM 元数据。

    参数：
        local_ids: 本地专家编号 ``[R]``，取值 ``[-1,L)``；其中 ``-1``
            表示固定 A2A 的 padding 行。
        order: 连续 int64 置换 ``[R]``，来自 ``argsort(local_ids,
            stable=True)``，也就是打包 Token 时使用的 ``origin``。
        experts: 本地专家数 L。
        block_m: Grouped GEMM 的行 tile 大小 BM。

    返回：
        ``GroupedGemmPlan``，字段布局如下：

        - ``offsets``：int64 ``[L+1]``，每个专家在打包行中的
          左闭右开边界；
        - ``tile_experts``：int64 ``[capacity]``，每个 BM 行 tile 所属
          专家；尾部无效 tile 为 ``-1``；
        - ``tile_rows``：int64 ``[capacity]``，tile 在专家区间内的行偏移。

        所有输出都在 ``local_ids`` 所在 CUDA 设备上连续。

    这里不再保留 Top-K 维度，每行已经是一个分支。``-1`` padding 会在
    稳定排序中排在专家 0 之前，因此专家边界自然把它们排除在每个有效
    专家区间之外。空专家、倾斜路由都由同一套边界/tile 规则处理。
    所有 Kernel 使用调用方当前 Stream，不产生 Host 回读。
    """
    rows = local_ids.numel()
    # 有效 tile 数上界：R 行最多产生 ceil(R/BM) 个行 tile，
    # 每个非空专家最多额外多出一个未对齐 tile。
    capacity = triton.cdiv(rows, block_m) + experts - 1
    expert_offsets = torch.empty(
        experts + 1, device=local_ids.device, dtype=torch.int64
    )
    tile_experts = torch.empty(capacity, device=local_ids.device, dtype=torch.int64)
    tile_rows = torch.empty_like(tile_experts)
    _plan_kernel[(1,)](
        local_ids,
        order,
        expert_offsets,
        tile_experts,
        tile_rows,
        # rows/capacity 作为运行时参数传入，避免每个 A2A 接收数都触发
        # 重新特化；只有搜索深度和向量容量桶是编译期常量。
        rows,
        experts,
        local_ids.stride(0),
        block_m,
        capacity,
        (rows + 1).bit_length(),
        triton.next_power_of_2(experts + 1),
        triton.next_power_of_2(capacity),
        num_warps=4,
    )
    return GroupedGemmPlan(expert_offsets, tile_experts, tile_rows, block_m, rows)


@triton.jit
def grouped_gemm_kernel(
    X,
    W,
    Y,
    Offsets,
    TileExperts,
    TileRows,
    RowWeights,
    PREWEIGHTED: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """作用：计算 Expert-major Grouped 投影中的一个 BM×BN 输出 tile。

    参数：
        X: Expert-major 输入 ``[R,K]``。
        W: 权重栈 ``[L,N,K]``。
        Y: 输出 ``[R,N]``。
        Offsets: 专家边界 ``[L+1]``。
        TileExperts: 每个 tile 的专家编号 ``[capacity]``。
        TileRows: 每个 tile 的行偏移 ``[capacity]``。
        RowWeights: 可选的连续 Expert-major FP32 ``[R]``，每行一个路由权重。
        PREWEIGHTED: 为 True 时，FP32 累加后乘权重再转换到 Y 的 dtype。
        K: 输入维度，编译期常量。
        N: 输出维度，编译期常量。
        BM: 行 tile 大小。
        BN: 列 tile 大小。
        BK: K 维分块大小。
    """
    tile = tl.program_id(0)
    expert = tl.load(TileExperts + tile)
    # GPU 规划器分配的是有界表；尾部 -1 项不拥有任何工作。
    if expert < 0:
        return
    # first/end 是该专家在打包行上的半开区间；rows 是当前 tile 的 BM 行。
    first = tl.load(Offsets + expert)
    end = tl.load(Offsets + expert + 1)
    rows = first + tl.load(TileRows + tile) + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)

    for block in range(tl.cdiv(K, BK)):
        ks = block * BK + tl.arange(0, BK)
        # 行方向只读取当前专家区间内的行，列方向按 BK 分块。
        a = tl.load(
            X + rows[:, None] * K + ks[None, :],
            (rows[:, None] < end) & (ks[None, :] < K),
            other=0,
        )
        # W 存储为 [L,N,K]；下面的地址选择专家 e 并计算 X @ W[e].T。
        b = tl.load(
            W + expert * N * K + cols[None, :] * K + ks[:, None],
            (cols[None, :] < N) & (ks[:, None] < K),
            other=0,
        )
        acc = tl.dot(a, b, acc, input_precision="ieee")

    if PREWEIGHTED:
        routing = tl.load(RowWeights + rows, rows < end, other=0)
        acc = acc * routing[:, None]
    tl.store(
        Y + rows[:, None] * N + cols[None, :],
        acc,
        (rows[:, None] < end) & (cols[None, :] < N),
    )


def _tile_config(dtype: torch.dtype) -> tuple[int, int, int, int]:
    """作用：返回 BF16/FP32 测试使用的小型、显式 H100 tile 策略。

    参数：
        dtype: 输入激活 dtype，只支持 BF16 或 FP32。

    返回：
        ``(BM, BN, BK, num_warps)``。BF16 使用 128x128x64、8 warps；
        FP32 使用 16x32x32、4 warps。该策略固定，不做自动调优。
    """
    if dtype == torch.bfloat16:
        # BM=128 减少预填充阶段的行 tile 数，同时也能覆盖单个短解码批次。
        # BK 保持 64，使 K 维累加顺序（也就是已验证的 BF16 舍入路径）不变。
        return 128, 128, 64, 8
    if dtype == torch.float32:
        return 16, 32, 32, 4
    raise TypeError(f"grouped SwiGLU supports BF16/FP32, got {dtype}")


def grouped_gemm(
    x: torch.Tensor,
    weights: torch.Tensor,
    plan: GroupedGemmPlan,
    *,
    output_dtype: torch.dtype,
    row_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """作用：用 ``[L,N,K]`` 权重栈计算连续的 Expert-major 行。

    参数：
        x: 连续 Expert-major 输入 ``[R,K]``，dtype 为 BF16/FP32。
        weights: 权重栈 ``[L,N,K]``，dtype 与 ``x`` 相同。
        plan: GroupedGemmPlan，包含同设备连续 int64 专家边界和 tile 布局。
        output_dtype: 输出 dtype，由这次投影的舍入边界决定。
        row_weights: 可选同设备连续 FP32 [R]，Expert-major 顺序；
            FP32 GEMM 累加后乘权重，再写入 output_dtype。

    返回：
        连续 ``[R,N]`` 输出。每个 Kernel program 只处理一个 BM×BN tile；
        空专家和尾部无效 tile 由 ``tile_experts=-1`` 直接跳过。
    """
    if x.ndim != 2 or weights.ndim != 3 or x.shape[1] != weights.shape[2]:
        raise ValueError("grouped GEMM expects [R,K] and [L,N,K]")
    if (
        x.dtype != weights.dtype
        or x.device != weights.device
        or not x.is_contiguous()
        or not weights.is_contiguous()
    ):
        raise ValueError(
            "grouped GEMM requires same dtype/device and contiguous row-major inputs"
        )
    if row_weights is not None and (
        row_weights.shape != (x.shape[0],)
        or row_weights.dtype != torch.float32
        or row_weights.device != x.device
        or not row_weights.is_contiguous()
    ):
        raise ValueError("row_weights must be contiguous FP32 [R] on input device")
    offsets, tile_experts, tile_rows = plan.offsets, plan.tile_experts, plan.tile_rows
    if x.shape[0] != plan.num_rows:
        raise ValueError(f"plan rows={plan.num_rows}, input rows={x.shape[0]}")
    if plan.block_m != _tile_config(x.dtype)[0] or plan.offsets.shape != (
        weights.shape[0] + 1,
    ):
        raise ValueError("plan tile size/Expert count differs from grouped GEMM")
    n = weights.shape[1]
    out = torch.empty((x.shape[0], n), device=x.device, dtype=output_dtype)
    if x.shape[0] == 0:
        return out

    block_m, block_n, block_k, warps = _tile_config(x.dtype)
    if tile_experts.numel():
        # 网格第一维覆盖所有 tile（包括尾部无效项），第二维覆盖 N 列块。
        grouped_gemm_kernel[(tile_experts.numel(), triton.cdiv(n, block_n))](
            x,
            weights,
            out,
            offsets,
            tile_experts,
            tile_rows,
            row_weights if row_weights is not None else x,
            row_weights is not None,
            x.shape[1],
            n,
            block_m,
            block_n,
            block_k,
            num_warps=warps,
            num_stages=3,
            enable_fp_fusion=row_weights is None,
        )
    return out


@triton.jit
def _restore_received_kernel(
    Packed,
    Origin,
    ValidStart,
    Received,
    H: tl.constexpr,
    BH: tl.constexpr,
):
    """作用：把 FP32 Expert-major ``[R,H]`` 写回接收顺序，每行只有一个写者。

    参数：
        Packed: Expert-major 输出 ``[R,H]``。
        Origin: 打包行 -> 接收行 ``[R]``。
        ValidStart: 首个有效专家行位置；其前为 padding。
        Received: source-major 输出 ``[R,H]``。
        H: 隐藏维度。
        BH: 列块大小。
    """
    row = tl.program_id(0)
    cols = tl.program_id(1) * BH + tl.arange(0, BH)
    received_row = tl.load(Origin + row)
    valid = row >= tl.load(ValidStart)
    # 地址映射与 index_copy_(0, origin, packed_output) 完全相同。
    # 与第十课 Combine 的加载一样，使用掩码避免额外的零填充中间缓冲。
    values = tl.load(Packed + row * H + cols, valid & (cols < H), other=0)
    tl.store(Received + received_row * H + cols, values, cols < H)


def compute_grouped_local_experts(
    tokens: torch.Tensor,
    local_ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    *,
    padding: bool = False,
    branch_weights: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """作用：完成 source-major 输入 -> Expert-major 计算 -> source-major 输出。

    参数：
        tokens: 第一次 All-to-All 收到的 source-major 行 ``[R,H]``，
            每一行是一个 Top-K 分支。
        local_ids: 每行的本地专家编号 ``[R]``。普通路径取值 ``[0,L)``；
            ``padding=True`` 时允许 ``-1`` 表示固定 A2A 填充。
        w13: 本地 gate/up 权重 ``[L,2F,H]``。
        w2: 本地 down 权重 ``[L,H,F]``。
        padding: 是否处理固定容量路径的 ``-1`` 填充。为 True 时，
            填充行会从专家 tile 计划中排除，并以 0 写回接收缓冲，
            因此不会对 Combine 产生贡献。

    返回：
        同设备、source-major 顺序的连续 ``[R,H]``，dtype 为 output_dtype。
        branch_weights 为可选同设备连续 FP32 [R]，source-major 顺序；
        在 down 投影 FP32 累加后乘一次权重，再写入 BF16/FP16/FP32。

    输入 Token 和权重必须是连续行主序 BF16/FP32，dtype 与设备一致；
    专家编号允许非连续 int32/int64。Top-K 已经展开，每行只对应一个
    分支。专家打包使用稳定排序，因此相同专家内的行保持接收顺序。

    GPU 规划与投影都在调用方当前 Stream 上执行，按 Stream 顺序消费
    元数据，不需要 Host 回读，也不跨 Stream 复用缓冲。
    """
    # 先校验形状、dtype 和连续性；任何布局错误都应在这里快速失败。
    if tokens.ndim != 2 or local_ids.ndim != 1:
        raise ValueError(
            f"expected tokens [R,H] and local_ids [R], got "
            f"{tokens.shape} and {local_ids.shape}"
        )
    if local_ids.shape[0] != tokens.shape[0]:
        raise ValueError(
            f"local_ids must have one entry per token row, got "
            f"{local_ids.shape[0]} IDs for {tokens.shape[0]} rows"
        )
    if local_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"local_ids must be int32/int64, got {local_ids.dtype}")
    if w13.ndim != 3 or w2.ndim != 3:
        raise ValueError(
            f"expected weights [L,2F,H] and [L,H,F], got {w13.shape} and {w2.shape}"
        )
    if (
        tokens.dtype not in (torch.bfloat16, torch.float32)
        or w13.dtype != tokens.dtype
        or w2.dtype != tokens.dtype
    ):
        raise TypeError("tokens and both weight stacks must share BF16/FP32 dtype")
    if (
        tokens.device != local_ids.device
        or tokens.device != w13.device
        or tokens.device != w2.device
    ):
        raise ValueError("tokens, local_ids and weights must share a device")
    if not tokens.is_contiguous() or not w13.is_contiguous() or not w2.is_contiguous():
        raise ValueError("grouped SwiGLU requires contiguous row-major tensors")
    if w13.shape[0] != w2.shape[0] or w13.shape[2] != tokens.shape[1]:
        raise ValueError(
            f"incompatible Expert weights: tokens={tokens.shape}, "
            f"w13={w13.shape}, w2={w2.shape}"
        )
    if w13.shape[1] % 2 or w2.shape != (
        w13.shape[0],
        tokens.shape[1],
        w13.shape[1] // 2,
    ):
        raise ValueError(
            f"expected w13 [L,2F,H] and w2 [L,H,F], got " f"{w13.shape} and {w2.shape}"
        )
    if output_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError(f"output_dtype must be FP32/BF16/FP16, got {output_dtype}")
    if branch_weights is not None and (
        branch_weights.shape != (tokens.shape[0],)
        or branch_weights.dtype != torch.float32
        or branch_weights.device != tokens.device
        or not branch_weights.is_contiguous()
    ):
        raise ValueError("branch_weights must be contiguous FP32 [R] on tokens.device")
    if tokens.shape[0] == 0:
        return torch.empty_like(tokens, dtype=output_dtype)

    # 1. 稳定打包：origin[p] 是打包行 p 对应的接收行。
    #    固定 A2A 中，-1 padding 会排在所有有效本地专家之前。
    origin = torch.argsort(local_ids.to(torch.int64), stable=True)
    packed_tokens = tokens.index_select(0, origin).contiguous()
    # 2. 在 GPU 上构造第十一课风格的 offsets/tile 表，不读取 Host 计数。
    block_m, _, _, _ = _tile_config(tokens.dtype)
    plan = build_device_plan(local_ids, origin, w13.shape[0], block_m)

    packed_weights = (
        branch_weights.index_select(0, origin) if branch_weights is not None else None
    )
    packed_output = grouped_expert_mlp(
        packed_tokens,
        w13,
        w2,
        plan,
        row_weights=packed_weights,
        output_dtype=output_dtype,
        padding=padding,
    )
    # 4. 恢复与第十一课 index_copy_ 相同的 origin 地址。
    #    返回方向的 A2A 需要 source-major 行，才能继续使用原来的 splits。
    result = torch.empty_like(tokens, dtype=output_dtype)
    if padding:
        # Offsets 和 Origin 都是设备端连续 int64，并且已在当前 Stream 就绪。
        # 掩码加载跳过 PackedOutput 中未写入的 padding；置换保证每个接收
        # 行恰好写一次，包括值为 0 的填充行。
        _restore_received_kernel[(tokens.shape[0], triton.cdiv(tokens.shape[1], 128))](
            packed_output,
            origin,
            plan.offsets,
            result,
            tokens.shape[1],
            128,
            num_warps=4,
        )
    else:
        # 普通变长路径没有 padding，可以直接使用 PyTorch 的索引写回。
        result.index_copy_(0, origin, packed_output)
    return result


def grouped_expert_mlp(
    packed_tokens: torch.Tensor,
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    plan: GroupedGemmPlan,
    *,
    row_weights: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float32,
    padding: bool = False,
) -> torch.Tensor:
    """Two shared-plan projections; Qwen SwiGLU replaces the lesson GELU.

    Rows are Expert-major and Top-K-expanded. Gate/up [L,2F,H] stores model
    dtype. Down [L,H,F] accumulates FP32, multiplies optional FP32 [R] weights,
    then stores output_dtype. Do not cast unweighted down values prematurely.
    """
    gate_up = grouped_gemm(
        packed_tokens, gate_up_weights, plan, output_dtype=packed_tokens.dtype
    )
    activated = swiglu_activation_triton(
        gate_up, valid_start=plan.offsets if padding else None
    )
    return grouped_gemm(
        activated,
        down_weights,
        plan,
        output_dtype=output_dtype,
        row_weights=row_weights,
    )
