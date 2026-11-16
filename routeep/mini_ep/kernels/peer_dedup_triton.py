"""去重 Peer 大向量 Kernel；Expert 路由元数据仍逐分支传输。

分配与指针表归 distributed.peer_dedup 管理。BF16 向量为连续 [row,H]，
元数据 int64，路由权重 FP32。来源身份使用 (EP rank, 原 Token, slot)，
不能用 global rank/CUDA 序号索引指针表；不使用无序浮点原子归约。"""
import triton
import triton.language as tl
from mini_ep.kernels.peer_triton import _wait


@triton.jit
def send_unique_kernel(X, Ids, InputPtrs, N, H: tl.constexpr, K: tl.constexpr,
                       L: tl.constexpr, C: tl.constexpr, RANK: tl.constexpr,
                       BK: tl.constexpr, BH: tl.constexpr):
    """原输入 X[N,H]/IDs[N,K] → 每个 Token/owner 只写一份远端输入。

    目标输入缓冲 [P*C,H] 按来源/原 Token 排列，每个 CTA 复制16行。
    K 只用于判断目标归属，不把向量按 K 展开后再跨卡发送。"""
    token = tl.program_id(0) * 16 + tl.arange(0, 16)
    owner = tl.program_id(1)
    col = tl.program_id(2) * BH + tl.arange(0, BH)
    slots = tl.arange(0, BK)
    expert = tl.load(Ids + token[:, None] * K + slots[None, :],
                     (token[:, None] < N) & (slots[None, :] < K), -1)
    # 只判断该 Token 是否至少有一个本 owner 分支，不按命中 Expert 数重复复制向量。
    owns = tl.sum(((expert // L == owner) & (expert >= 0)).to(tl.int32), 1) > 0
    ptr = tl.load(InputPtrs + owner).to(tl.pointer_type(X.dtype.element_ty))
    valid = (token[:, None] < N) & owns[:, None] & (col[None, :] < H)
    value = tl.load(X + token[:, None] * H + col[None, :], valid, 0)
    tl.store(ptr + (RANK * C + token[:, None]) * H + col[None, :], value, valid)


@triton.jit
def send_metadata_kernel(Ids, Weights, Order, Starts, Targets,
                         MetaPtrs, WeightPtrs, LookupPtrs, ReadyPtrs, SourceRows, TileStarts, B,
                         K: tl.constexpr, L: tl.constexpr, C: tl.constexpr,
                         RANK: tl.constexpr, BM: tl.constexpr, BLOCK: tl.constexpr):
    """排序分支身份 → 目标 Expert-major 元数据及原 slot 的逆映射。

    远端 Meta[R,2]=(来源 EP rank, 原分支号)，权重 [R] FP32；
    Lookup[P*C,K] int64 把来源 Token/slot 映射为接收 Expert 行。
    开始屏障前已填 -1，只写本 owner 的 slot；其他 slot 和未使用容量保持 -1。"""
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = row < B
    branch = tl.load(Order + row, valid, 0)
    expert = tl.load(Ids + branch, valid, 0)
    owner = expert // L
    target = tl.load(Targets + expert) + row - tl.load(Starts + expert)
    meta = tl.load(MetaPtrs + owner).to(tl.pointer_type(tl.int64))
    routing = tl.load(WeightPtrs + owner).to(tl.pointer_type(tl.float32))
    lookup = tl.load(LookupPtrs + owner).to(tl.pointer_type(tl.int64))
    tl.store(meta + target * 2, RANK, valid)
    tl.store(meta + target * 2 + 1, branch, valid)
    tl.store(routing + target, tl.load(Weights + branch, valid, 0), valid)
    tl.store(lookup + RANK * C * K + branch, target, valid)
    # 唯一输入复制是同一来源 stream 的前一个 Kernel；此处发布 metadata，
    # 同时交接先前输入写入和当前分支元数据，目标卡 acquire 后才能读取。
    tl.debug_barrier()
    ready = tl.load(ReadyPtrs + owner).to(tl.pointer_type(tl.int32))
    expert_row = tl.load(SourceRows + expert) + row - tl.load(Starts + expert)
    tile = tl.load(TileStarts + expert) + expert_row // BM
    tl.atomic_add(ready + tile, 1, mask=valid, sem="acq_rel", scope="sys")


@triton.jit
def expand_local_kernel(Unique, Meta, Data, Offsets, Experts, Rows, Ready, MetaReady, Expected,
                        H: tl.constexpr, K: tl.constexpr, C: tl.constexpr,
                        BM: tl.constexpr, BH: tl.constexpr):
    """本卡唯一输入 [P*C,H] → 紧凑 Expert-major Data[R,H]。

    acquire 当前 tile 的 metadata 计数，同时保证先前唯一输入写入可见。
    后续 Top-K 展开只发生在本卡，不增加跨卡大向量；复制后发布供 TMA Up
    消费的 tile-ready 计数。"""
    tile = tl.program_id(0)
    expert = tl.load(Experts + tile)
    if expert < 0:
        return
    # metadata 每分支只发布一次；Expected 含列块数，因此先除去列块因子。
    _wait(MetaReady + tile, tl.load(Expected + tile) // tl.cdiv(H, BH))
    row = tl.load(Offsets + expert) + tl.load(Rows + tile) + tl.program_id(1) * 16 + tl.arange(0, 16)
    end = tl.load(Offsets + expert + 1)
    col = tl.program_id(2) * BH + tl.arange(0, BH)
    source = tl.load(Meta + row * 2, row < end, 0)
    branch = tl.load(Meta + row * 2 + 1, row < end, 0)
    valid = (row[:, None] < end) & (col[None, :] < H)
    value = tl.load(Unique + (source[:, None] * C + branch[:, None] // K) * H + col[None, :], valid, 0)
    tl.store(Data + row[:, None] * H + col[None, :], value, valid)
    tl.debug_barrier()
    tl.atomic_add(Ready + tile, tl.sum((row < end).to(tl.int32), 0), sem="acq_rel", scope="sys")


@triton.jit
def push_owner_partials_kernel(Outputs, Lookup, ReturnPtrs,
                               H: tl.constexpr, K: tl.constexpr, C: tl.constexpr,
                               P: tl.constexpr, RANK: tl.constexpr, BH: tl.constexpr):
    """已加权 BF16 Expert 行 → 每个来源 Token/owner 一个 BF16 部分和。

    Lookup[P*C,K] 提供本卡 Expert 行，其他 owner 的 slot 为 -1。
    按原 slot 升序以 FP32 累加已舍入分支，在 owner 边界只舍入一次为 BF16。
    这是把原 Combine 的同一规则移到目标卡；不能逐次用 BF16 累加。"""
    pair = tl.program_id(0) * 16 + tl.arange(0, 16)
    source = pair // C
    token = pair % C
    col = tl.program_id(1) * BH + tl.arange(0, BH)
    partial = tl.full((16, BH), 0.0, tl.float32)
    any_owned = tl.full((16,), False, tl.int1)
    # 不能改成 BF16 递增求和或无序原子加法，例如 [256,1,-256,1] 必须得到2。
    for slot in tl.static_range(K):
        row = tl.load(Lookup + pair * K + slot, pair < P * C, -1)
        valid = (pair[:, None] < P * C) & (row[:, None] >= 0) & (col[None, :] < H)
        value = tl.load(Outputs + row[:, None] * H + col[None, :], valid, 0).to(tl.float32)
        partial = partial + value
        any_owned = any_owned | (row >= 0)
    # 指针按来源选择；返回缓冲的行按当前 owner rank 排列，便于来源按 owner 升序合并。
    ptr = tl.load(ReturnPtrs + source, pair < P * C, 0).to(tl.pointer_type(Outputs.dtype.element_ty))
    tl.store(ptr[:, None] + (RANK * C + token[:, None]) * H + col[None, :], partial,
             (pair[:, None] < P * C) & any_owned[:, None] & (col[None, :] < H))


@triton.jit
def combine_owner_partials_kernel(Returned, Ids, Output, N,
                                  H: tl.constexpr, K: tl.constexpr,
                                  C: tl.constexpr, P: tl.constexpr,
                                  L: tl.constexpr, BLOCK: tl.constexpr):
    """BF16 Returned[P*C,H] → 原顺序 BF16 Output[N,H]。

    只读取原 IDs[N,K] 选中的 owner，其他缓冲行可能未初始化，不得读取。
    按 EP rank 升序以 FP32 累加部分和，最后仅做一次模型 dtype 输出舍入。"""
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token, col = idx // H, idx % H
    valid = idx < N * H
    total = tl.full((BLOCK,), 0.0, tl.float32)
    for owner in tl.static_range(P):
        owns = tl.full((BLOCK,), False, tl.int1)
        for slot in tl.static_range(K):
            expert = tl.load(Ids + token * K + slot, valid, 0)
            owns = owns | (expert // L == owner)
        value = tl.load(Returned + (owner * C + token) * H + col, valid & owns, 0)
        total = total + value.to(tl.float32)
    tl.store(Output + idx, total, valid)
