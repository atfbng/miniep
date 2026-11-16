"""Peer 复制、设备信号、地址规划及 TMA SwiGLU Up Kernel。

指针由 PeerSwiGLU 提供；载荷 BF16，计数/偏移/分支身份 int64，就绪与 epoch
为 int32。原输入保持 [N,H]，只有 _send 按已排序分支身份展开；接收布局为
Expert-major、Expert 内 source-major。Kernel 不分配工作区或创建进程组。"""
import triton
import triton.language as tl


@triton.jit
def _wait(ptr, threshold):
    """在本 GPU 用系统级 acquire 轮询 int32 计数，达到阈值后消费数据。"""
    while tl.atomic_add(ptr, 0, sem="acquire", scope="sys") < threshold:
        tl.inline_asm_elementwise("nanosleep.u32 128; mov.u32 $0, 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _signal(Ptrs, epoch, P: tl.constexpr, RANK: tl.constexpr, FINISH: tl.constexpr):
    """uint64 指针表 [P] 指向各卡 int32 flags[2P]，发布并等待本轮 epoch。"""
    # flags 前/后 P 项分别用于开始/结束；每个来源只写自己的 epoch 槽。
    for peer in range(P):
        ptr = tl.load(Ptrs + peer).to(tl.pointer_type(tl.int32))
        tl.atomic_xchg(ptr + FINISH * P + RANK, epoch, sem="release", scope="sys")
    own = tl.load(Ptrs + RANK).to(tl.pointer_type(tl.int32))
    for peer in range(P):
        _wait(own + FINISH * P + peer, epoch)


@triton.jit
def _layout(Counts, SendStarts, Targets, SourceRows, TileStarts, Offsets,
            E: tl.constexpr, P: tl.constexpr, L: tl.constexpr,
            RANK: tl.constexpr, BM: tl.constexpr, BE: tl.constexpr, BP: tl.constexpr):
    """设备 int64 counts [P,E] → 来源/目标偏移 [E] 及本地 offsets [L+1]。

    所有张量连续。Starts 定位来源 Expert 排序后的分支；Targets 包含同一
    Expert 中更早来源的行数；SourceRows 是该前缀，TileStarts 是 Expert 的 tile 基址。"""
    e = tl.arange(0, BE)
    source = tl.arange(0, BP)
    counts = tl.load(Counts + source[:, None] * E + e[None, :],
                     (source[:, None] < P) & (e[None, :] < E), 0)
    total = tl.sum(counts, 0)
    # 在每个 owner rank 边界重新开始累计专家计数。
    totals = tl.reshape(total, (P, BE // P))
    within = tl.cumsum(totals, 1) - totals
    tiles = tl.cdiv(totals, BM)
    tile_starts = tl.cumsum(tiles, 1) - tiles
    local = tl.load(Counts + RANK * E + e, e < E, 0)
    prior_sources = tl.sum(tl.where(source[:, None] < RANK, counts, 0), 0)
    tl.store(SendStarts + e, tl.cumsum(local) - local, e < E)
    tl.store(Targets + e, tl.reshape(within, (BE,)) + prior_sources, e < E)
    tl.store(SourceRows + e, prior_sources, e < E)
    tl.store(TileStarts + e, tl.reshape(tile_starts, (BE,)), e < E)
    begin = RANK * L
    local_mask = (e >= begin) & (e < begin + L)
    tl.store(Offsets + e - begin, tl.reshape(within, (BE,)), local_mask)
    tl.store(Offsets + L, tl.sum(tl.where(local_mask, total, 0), 0))


@triton.jit
def _plan(Offsets, Experts, Rows, Expected, L: tl.constexpr, BM: tl.constexpr,
          CAP: tl.constexpr, BL: tl.constexpr, BT: tl.constexpr, HC: tl.constexpr):
    """int64 offsets [L+1] → Expert/相对行/预期复制计数 [CAP]。

    CAP 覆盖各 Expert 的 ceil(rows/BM) 之和，尾部无效 Expert 标记 -1。
    有效 tile 需要 valid_rows*HC 次发布，HC=ceil(H/复制列块宽度)。"""
    e = tl.arange(0, BL)
    first = tl.load(Offsets + e, e < L, 0)
    end = tl.load(Offsets + e + 1, e < L, 0)
    tile_ends = tl.cumsum(tl.cdiv(end - first, BM))
    tile = tl.arange(0, BT)
    lo = tl.full((BT,), 0, tl.int32)
    hi = tl.full((BT,), L, tl.int32)
    for _ in range((L + 1).bit_length()):
        mid = (lo + hi) // 2
        value = tl.gather(tile_ends, tl.minimum(mid, BL - 1), 0)
        # 稀疏 Expert 会留下容量尾部；mid==L 必须保持哨兵，不能越界定位 Expert。
        smaller = (mid < L) & (value <= tile)
        lo = tl.where(smaller, mid + 1, lo)
        hi = tl.where(smaller, hi, mid)
    expert = lo
    previous = tl.gather(tile_ends, tl.maximum(expert - 1, 0), 0)
    relative = (tile - tl.where(expert == 0, 0, previous)) * BM
    count = tl.gather(end - first, tl.minimum(expert, BL - 1), 0)
    tl.store(Expected + tile, tl.minimum(BM, count - relative) * HC, (tile < CAP) & (expert < L))
    tl.store(Experts + tile, tl.where(expert < L, expert, -1), tile < CAP)
    tl.store(Rows + tile, relative, tile < CAP)


@triton.jit
def _send(X, Ids, Weights, Order, Starts, Targets, SourceRows, TileStarts,
          DataPtrs, MetaPtrs, WeightPtrs, ReadyPtrs,
          B, H: tl.constexpr, K: tl.constexpr, L: tl.constexpr,
          RANK: tl.constexpr, BM: tl.constexpr, BH: tl.constexpr):
    """原 BF16 X[N,H]、展平 ID/FP32 权重 [B]、排序置换 Order[B]，B=N*K。

    uint64 指针表 [P] 指向紧凑 Expert-major 接收向量、int64 元数据 [R,2]
    (source,branch)、FP32 权重 [R] 和 int32 tile 就绪计数。各来源写入互不
    重叠的行；这里只用整数就绪原子，不做浮点原子归约。"""
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    col = tl.program_id(1) * BH + tl.arange(0, BH)
    valid = row < B
    branch = tl.load(Order + row, valid, 0)
    expert = tl.load(Ids + branch, valid, 0)
    owner = expert // L
    within = row - tl.load(Starts + expert)
    destination = tl.load(Targets + expert) + within
    data = tl.load(DataPtrs + owner).to(tl.pointer_type(X.dtype.element_ty))
    meta = tl.load(MetaPtrs + owner).to(tl.pointer_type(tl.int64))
    weight_ptr = tl.load(WeightPtrs + owner).to(tl.pointer_type(tl.float32))
    value = tl.load(X + (branch[:, None] // K) * H + col[None, :],
                    valid[:, None] & (col[None, :] < H), 0)
    tl.store(data[:, None] + destination[:, None] * H + col[None, :], value,
             valid[:, None] & (col[None, :] < H))
    if tl.program_id(1) == 0:
        tl.store(meta + destination * 2, RANK, valid)
        tl.store(meta + destination * 2 + 1, branch, valid)
        tl.store(weight_ptr + destination, tl.load(Weights + branch, valid, 0), valid)
    # 整数 acq_rel 把各复制 CTA 的写入串到就绪计数上；Up acquire 到阈值才读取。
    tl.debug_barrier()
    ready = tl.load(ReadyPtrs + owner).to(tl.pointer_type(tl.int32))
    expert_row = tl.load(SourceRows + expert) + within
    tile = tl.load(TileStarts + expert) + expert_row // BM
    tl.atomic_add(ready + tile, 1, mask=valid, sem="acq_rel", scope="sys")


@triton.jit
def _up(InputDesc, WeightDesc, Activation, Offsets, Experts, Rows, Ready, Expected,
        F: tl.constexpr, H: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
        BK: tl.constexpr):
    """BF16 TMA 描述符 [capacity,H]/[L*2F,H] → BF16 激活 [capacity,F]。

    Expert 偏移与 tile 映射为设备 int64；仅写有效行。Down 复用同一计划，
    不会消费容量尾部尚未初始化的行。"""
    tile = tl.program_id(0)
    expert = tl.load(Experts + tile).to(tl.int32)
    if expert < 0:
        return
    first = tl.load(Offsets + expert)
    end = tl.load(Offsets + expert + 1)
    start = (first + tl.load(Rows + tile)).to(tl.int32)
    rows = start + tl.arange(0, BM)
    # 仅等待当前 Expert tile 的所有来源/列块复制完成，不等待其他无关 tile。
    _wait(Ready + tile, tl.load(Expected + tile))
    # Peer 写入走 generic proxy，TMA 读取走 async proxy。系统级 acquire 之后，
    # 仍需 proxy fence 让这些写入对 TMA 可见，不能只凭就绪计数就发描述符读取。
    tl.inline_asm_elementwise(
        "fence.proxy.async.global; mov.u32 $0, 0;", "=r", [],
        dtype=tl.int32, is_pure=False, pack=1,
    )
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    gate_acc = tl.zeros((BM, BN), tl.float32)
    up_acc = tl.zeros((BM, BN), tl.float32)
    for step in range(tl.cdiv(H, BK)):
        x = InputDesc.load([start, step * BK])
        gate = WeightDesc.load([expert * (2 * F) + tl.program_id(1) * BN, step * BK]).T
        up = WeightDesc.load([expert * (2 * F) + F + tl.program_id(1) * BN, step * BK]).T
        gate_acc = tl.dot(x, gate, gate_acc)
        up_acc = tl.dot(x, up, up_acc)
    # Gate/up、SiLU、激活乘法分别保留 BF16 舍入边界，与独立 SwiGLU 计算对齐。
    gate_value = gate_acc.to(Activation.dtype.element_ty).to(tl.float32)
    up_value = up_acc.to(Activation.dtype.element_ty).to(tl.float32)
    silu = (gate_value / (1.0 + tl.exp(-gate_value))).to(Activation.dtype.element_ty).to(tl.float32)
    tl.store(Activation + rows[:, None] * F + cols[None, :], silu * up_value,
             (rows[:, None] < end) & (cols[None, :] < F))


@triton.jit
def _push(Outputs, Meta, Offsets, ReturnedPtrs, H: tl.constexpr,
          L: tl.constexpr, BH: tl.constexpr):
    """已加权 BF16 输出 [R,H] → 来源卡 returned[原分支号,H]。

    int64 Meta[R,2] 提供 EP 来源 rank 与 Token-major/slot-minor 分支号；
    Offsets[L] 是有效接收行数，跳过容量尾部。此函数供保留的逐分支对照使用。"""
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    col = tl.program_id(1) * BH + tl.arange(0, BH)
    valid = row < tl.load(Offsets + L)
    source = tl.load(Meta + row * 2, valid, 0)
    branch = tl.load(Meta + row * 2 + 1, valid, 0)
    ptr = tl.load(ReturnedPtrs + source).to(tl.pointer_type(Outputs.dtype.element_ty))
    value = tl.load(Outputs + row[:, None] * H + col[None, :],
                    valid[:, None] & (col[None, :] < H), 0)
    tl.store(ptr[:, None] + branch[:, None] * H + col[None, :], value,
             valid[:, None] & (col[None, :] < H))
