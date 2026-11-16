"""Single-node EP through mapped peer memory and a cooperative Triton MLP.

The CUDA symmetric-memory backend maps NVLink/P2P-accessible peer allocations.
The main kernel pushes packed input tiles while peers multiply Up, and pushes Down tiles
back to their source. FP32 weighted combine is a separate kernel. No AllGather
or Tensor Parallel operation is used.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _clock():
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
    )


@triton.jit
def _wait(Ptr, epoch, SYSTEM: tl.constexpr):
    # The scope is resolved at compile time: peer signals use system scope.
    scope: tl.constexpr = "sys" if SYSTEM else "gpu"
    while tl.atomic_add(Ptr, 0, sem="acquire", scope=scope) < epoch:
        tl.inline_asm_elementwise(
            "nanosleep.u32 128; mov.u32 $0, 0;",
            "=r",
            [],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )


@triton.jit
def _gelu_fp16(acc):
    # Match the Reference rounding before GELU and before the Down projection.
    value = acc.to(tl.float16).to(tl.float32)
    return (0.5 * value * (1.0 + libdevice.erf(value * 0.7071067811865475))).to(tl.float16)


@triton.jit
def _grid_barrier(Counters, slot: tl.constexpr, epoch, GRID: tl.constexpr):
    # A cooperative grid guarantees residency of all participants. Every lane's
    # global stores precede the block's release; acquire completes the handoff.
    tl.debug_barrier()
    tl.atomic_add(Counters + slot, 1, sem="release", scope="gpu")
    _wait(Counters + slot, epoch * GRID, False)
    tl.debug_barrier()


@triton.jit
def peer_mlp_kernel(
    InputPtrs,
    OutputPtrs,
    FlagPtrs,
    Up,
    Down,
    Activation,
    LocalInput,
    TileReady,
    RowSources,
    RowOffsets,
    ExpertRows,
    Counters,
    LocalPtrs,
    ReadyPtrs,
    Segments,
    Expected,
    InputDesc,
    UpDesc,
    ActivationDesc,
    DownDesc,
    Times,
    epoch,
    H: tl.constexpr,
    F: tl.constexpr,
    Q: tl.constexpr,
    P: tl.constexpr,
    RANK: tl.constexpr,
    M: tl.constexpr,
    GRID: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    C: tl.constexpr,
    SEGMENTS: tl.constexpr,
    COMM: tl.constexpr,
    COPY_N: tl.constexpr,
    PROFILE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Stage 1: protect the previous output and publish the new packed input.
    if pid == 0:
        # Publish local input production AND completion of the previous combine.
        # All ranks participate once per invocation, in the same call order.
        for peer in range(P):
            flag = tl.load(FlagPtrs + peer).to(tl.pointer_type(tl.int32))
            tl.atomic_xchg(flag + RANK, epoch, sem="release", scope="sys")
        own = tl.load(FlagPtrs + RANK).to(tl.pointer_type(tl.int32))
        for peer in range(P):
            _wait(own + peer, epoch, True)
    _grid_barrier(Counters, 0, epoch, GRID)

    if PROFILE:
        tl.store(Times + pid * 5, _clock())
    mt: tl.constexpr = tl.cdiv(M, BM)
    up_nt: tl.constexpr = tl.cdiv(F, BN)
    copy_nt: tl.constexpr = tl.cdiv(H, COPY_N)
    # Stage 2: dedicated communication CTAs feed the resident Up CTAs.
    if pid < COMM:
        # Dedicated resident communication CTAs push coalesced tiles. The
        # source segments split exactly at the target Expert's M-tile boundary.
        for task in range(pid, SEGMENTS * copy_nt, COMM):
            seg = task // copy_nt
            expert = tl.load(Segments + seg * 4)
            src_row = tl.load(Segments + seg * 4 + 1)
            length = tl.load(Segments + seg * 4 + 2)
            dst_row = tl.load(Segments + seg * 4 + 3)
            rows = tl.arange(0, BM)
            cols = task % copy_nt * COPY_N + tl.arange(0, COPY_N)
            destination = expert // Q
            local_expert = expert % Q
            local = tl.load(InputPtrs + RANK).to(tl.pointer_type(tl.float16))
            remote = tl.multiple_of(tl.load(LocalPtrs + destination), 16).to(
                tl.pointer_type(tl.float16)
            )
            value = tl.load(
                local + (expert * C + src_row + rows[:, None]) * H + cols[None, :],
                (rows[:, None] < length) & (cols[None, :] < H),
                0,
            )
            tl.store(
                remote + (local_expert * M + dst_row + rows[:, None]) * H + cols[None, :],
                value,
                (rows[:, None] < length) & (cols[None, :] < H),
            )
            tl.debug_barrier()
            ready = tl.load(ReadyPtrs + destination).to(tl.pointer_type(tl.int32))
            tl.atomic_add(ready + local_expert * mt + dst_row // BM, 1, sem="release", scope="sys")
    else:
        for task in range(pid - COMM, Q * mt * up_nt, GRID - COMM):
            expert = task // (mt * up_nt)
            row = (task % mt) * BM + tl.arange(0, BM)
            col = (task // mt % up_nt) * BN + tl.arange(0, BN)
            count = tl.load(ExpertRows + expert)
            if task % mt * BM < count:
                tile_row = expert * mt + task % mt
                expected = tl.load(Expected + tile_row)
                _wait(TileReady + tile_row, epoch * expected, True)
                acc = tl.zeros((BM, BN), tl.float32)
                for step in range(tl.cdiv(H, BK)):
                    ks = step * BK + tl.arange(0, BK)
                    a = InputDesc.load([expert * M + (task % mt) * BM, step * BK])
                    b = UpDesc.load([expert * F + (task // mt % up_nt) * BN, step * BK]).T
                    acc = tl.dot(a, b, acc)
                left, right = tl.split(tl.reshape(acc, (BM, 2, BN // 2)).permute(0, 2, 1))
                half_col = (task // mt % up_nt) * BN + tl.arange(0, BN // 2)
                activated = _gelu_fp16(left)
                tl.store(
                    Activation + (expert * M + row[:, None]) * F + half_col[None, :],
                    activated,
                    (row[:, None] < count) & (half_col[None, :] < F),
                )
                activated = _gelu_fp16(right)
                tl.store(
                    Activation + (expert * M + row[:, None]) * F + (half_col[None, :] + BN // 2),
                    activated,
                    (row[:, None] < count) & (half_col[None, :] + BN // 2 < F),
                )
    if PROFILE:
        tl.store(Times + pid * 5 + 1, _clock())
    _grid_barrier(Counters, 1, epoch, GRID)
    if PROFILE:
        tl.store(Times + pid * 5 + 2, _clock())

    # Stage 3: all CTAs compute Down and write each row back to its source.
    down_nt: tl.constexpr = tl.cdiv(H, BN)
    for task in range(pid, Q * mt * down_nt, GRID):
        expert = task // (mt * down_nt)
        row = (task % mt) * BM + tl.arange(0, BM)
        col = (task // mt % down_nt) * BN + tl.arange(0, BN)
        count = tl.load(ExpertRows + expert)
        if task % mt * BM < count:
            tl.inline_asm_elementwise(
                "fence.proxy.async.global; mov.u32 $0, 0;",
                "=r",
                [],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
            acc = tl.zeros((BM, BN), tl.float32)
            for step in range(tl.cdiv(F, BK)):
                ks = step * BK + tl.arange(0, BK)
                a = ActivationDesc.load([expert * M + (task % mt) * BM, step * BK])
                b = DownDesc.load([expert * H + (task // mt % down_nt) * BN, step * BK]).T
                acc = tl.dot(a, b, acc)
            sources = tl.load(RowSources + expert * M + row, row < count, 0)
            offsets = tl.load(RowOffsets + expert * M + row, row < count, 0)
            base = tl.multiple_of(tl.load(OutputPtrs + sources), 16).to(tl.pointer_type(tl.float16))
            tl.store(
                base[:, None] + offsets[:, None] * H + col[None, :],
                acc.to(tl.float16),
                (row[:, None] < count) & (col[None, :] < H),
            )
    if PROFILE:
        tl.store(Times + pid * 5 + 3, _clock())
    _grid_barrier(Counters, 2, epoch, GRID)
    if PROFILE:
        tl.store(Times + pid * 5 + 4, _clock())
    # Stage 4: release the returned values to the source-side Combine.
    if pid == 0:
        for peer in range(P):
            flag = tl.load(FlagPtrs + peer).to(tl.pointer_type(tl.int32))
            tl.atomic_xchg(flag + P + RANK, epoch, sem="release", scope="sys")


@triton.jit
def peer_combine_kernel(
    Returned,
    Restore,
    Weights,
    Output,
    OwnFlags,
    epoch,
    N: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    P: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for peer in range(P):
        _wait(OwnFlags + P + peer, epoch, True)
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token, col = idx // H, idx % H
    value = tl.full((BLOCK,), 0, tl.float32)
    for slot in range(K):
        row = tl.load(Restore + token * K + slot, token < N, 0)
        w = tl.load(Weights + token * K + slot, token < N, 0)
        v = tl.load(Returned + row * H + col, token < N, 0, cache_modifier=".cg").to(tl.float32)
        value += v * w
    tl.store(Output + idx, value, idx < N * H)


class TritonPeerMLP:
    """Prepared pure-EP inference on one node, one process per CUDA GPU.

    Inputs: contiguous CUDA FP16 X[N,H], Up[Q,F,H], Down[Q,H,F]; int64
    expert_ids[N,K], FP32 routing_weights[N,K]. E=P*Q uniformly placed Experts.
    Symmetric packed/returned [E,N*K,H] have capacity N*K per global Expert;
    only counted rows are accessed. restore_index[N,K] addresses this layout.
    Internal activation [Q,M,F] is Expert-major; M is the global maximum Expert
    receive count rounded up to BM. local_inputs [Q,M,H] uses symmetric memory.
    RowSources/RowOffsets int32[Q,M] identify source EP rank and that source
    packed/returned row. Segments int32[L,4] contain global Expert, source row,
    segment length and target Expert row; counts/offsets use token rows.
    TileReady int32[Q,ceil(M/BM)] counts release publications; Expected counts
    the source segments times H-column tiles that cover each input M-tile.
    Metadata preparation and symmetric rendezvous are outside the execution.
    All ranks must call run() in the same order. Start/done epoch signals protect
    input readiness, remote output completion and reuse after prior combine.
    """

    def __init__(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
        up: torch.Tensor,
        down: torch.Tensor,
        *,
        bm=128,
        bn=256,
        bk=64,
        warps=8,
        stages=3,
        grid=None,
        comm=16,
        copy_n=256,
    ) -> None:
        if not dist.is_initialized() or x.device.type != "cuda":
            raise ValueError("TritonPeerMLP requires an initialized CUDA process group")
        self.rank, self.p = dist.get_rank(), dist.get_world_size()
        self.n, self.h = x.shape
        self.q, self.f, wh = up.shape
        self.e, self.k = self.q * self.p, expert_ids.shape[1]
        if (
            wh != self.h
            or down.shape != (self.q, self.h, self.f)
            or expert_ids.shape != (self.n, self.k)
            or routing_weights.shape != expert_ids.shape
            or self.k not in (1, 2)
            or self.h % 8
            or self.f % 8
        ):
            raise ValueError("inconsistent shapes; require Top-1/2 and H,F divisible by 8")
        if any(
            t.dtype != torch.float16 or t.device != x.device or not t.is_contiguous()
            for t in (x, up, down)
        ):
            raise ValueError("X/Up/Down must be contiguous FP16 tensors on the same CUDA device")
        if expert_ids.dtype != torch.int64 or expert_ids.device != x.device:
            raise ValueError("expert_ids must be CUDA int64 on the input device")
        if (
            routing_weights.dtype != torch.float32
            or routing_weights.device != x.device
            or not routing_weights.is_contiguous()
        ):
            raise ValueError("routing_weights must be contiguous CUDA FP32")
        if expert_ids.numel() and (
            expert_ids.min().item() < 0 or expert_ids.max().item() >= self.e
        ):
            raise ValueError(f"expert IDs must be in [0,{self.e})")
        if (
            self.n < 1
            or self.q < 1
            or bm not in (16, 32, 64, 128)
            or bn not in (32, 64, 128, 256)
            or bk not in (32, 64, 128)
        ):
            raise ValueError("invalid dimensions or tile configuration")
        if warps not in (4, 8, 16) or stages not in (1, 2, 3, 4) or copy_n not in (64, 128, 256):
            raise ValueError("warps=4/8/16, stages=1..4, copy_n=64/128/256 are required")
        sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
        if (grid is not None and grid < 2) or comm < 1 or comm >= (grid or sm_count):
            raise ValueError("require grid>=2 and 1<=comm<grid")
        if torch.cuda.get_device_capability(x.device)[0] < 9:
            raise ValueError("this TMA implementation requires CUDA compute capability >=9.0")
        if self.e * self.n * self.k >= 2**31:
            raise ValueError("packed row metadata must fit int32")
        device = x.device
        self.capacity = self.n * self.k
        self.packed = symm_mem.empty(self.e, self.capacity, self.h, dtype=x.dtype, device=device)
        self.returned = symm_mem.empty(self.e, self.capacity, self.h, dtype=x.dtype, device=device)
        self.flags = symm_mem.empty(2 * self.p, dtype=torch.int32, device=device)
        self.flags.zero_()
        self.handles = [
            symm_mem.rendezvous(t, dist.group.WORLD)
            for t in (self.packed, self.returned, self.flags)
        ]
        self.ptrs = [
            torch.tensor(h.buffer_ptrs, dtype=torch.uint64, device=device) for h in self.handles
        ]
        self.up, self.down, self.weights = up, down, routing_weights
        ids = expert_ids.flatten()
        order = torch.argsort(ids, stable=True)
        counts = torch.bincount(ids, minlength=self.e)
        count_list = counts.tolist()
        restore = torch.empty_like(ids)
        position = 0
        for expert, count in enumerate(count_list):
            selected = order[position : position + count]
            self.packed[expert, :count].copy_(x[selected // self.k])
            restore[selected] = expert * self.capacity + torch.arange(count, device=device)
            position += count
        self.restore = restore.reshape(self.n, self.k)
        self._input_order, self._count_list = order, count_list
        received = torch.empty_like(counts)
        dist.all_to_all_single(received, counts)
        counts_sq = received.reshape(self.p, self.q).cpu()
        sizes = counts_sq.sum(0).tolist()
        max_rows = torch.tensor(max(1, max(sizes)), device=device, dtype=torch.int64)
        dist.all_reduce(max_rows, op=dist.ReduceOp.MAX)
        self.m = max_rows.item()
        # All row maps include padding to avoid speculative OOB metadata reads.
        self.m = triton.cdiv(self.m, bm) * bm
        sources = torch.zeros((self.q, self.m), dtype=torch.int32)
        offsets = torch.zeros_like(sources)
        for expert in range(self.q):
            cursor = 0
            for source in range(self.p):
                count = counts_sq[source, expert].item()
                sources[expert, cursor : cursor + count] = source
                offsets[expert, cursor : cursor + count] = (
                    self.rank * self.q + expert
                ) * self.capacity + torch.arange(count)
                cursor += count
        self.sources, self.offsets = sources.to(device), offsets.to(device)
        self.rows = torch.tensor(sizes, dtype=torch.int32, device=device)
        self.activation = torch.empty((self.q, self.m, self.f), dtype=x.dtype, device=device)
        self.output = torch.empty((self.n, self.h), dtype=torch.float32, device=device)
        self.local_inputs = symm_mem.empty(self.q, self.m, self.h, dtype=x.dtype, device=device)
        self.tile_ready = symm_mem.empty(
            self.q * triton.cdiv(self.m, bm), dtype=torch.int32, device=device
        )
        self.tile_ready.zero_()
        more_handles = [
            symm_mem.rendezvous(t, dist.group.WORLD) for t in (self.local_inputs, self.tile_ready)
        ]
        self.handles.extend(more_handles)
        self.local_ptrs, self.ready_ptrs = [
            torch.tensor(h.buffer_ptrs, dtype=torch.uint64, device=device) for h in more_handles
        ]
        all_counts = [torch.empty_like(counts) for _ in range(self.p)]
        dist.all_gather(all_counts, counts)
        all_counts = torch.stack(all_counts).cpu()
        prefixes = all_counts.cumsum(0) - all_counts
        segments = []
        for expert, count in enumerate(count_list):
            row = 0
            while row < count:
                target_row = prefixes[self.rank, expert].item() + row
                length = min(count - row, bm - target_row % bm)
                segments.append((expert, row, length, target_row))
                row += length
        # Deliver local Expert 0 to every destination before local Expert 1.
        # This matches the GEMM task order and avoids making the last rank wait
        # until all Experts of earlier ranks have been transmitted.
        segments.sort(key=lambda seg: (seg[0] % self.q, seg[1] // bm, seg[0] // self.q))
        self.segments = torch.tensor(segments, dtype=torch.int32, device=device).reshape(-1, 4)
        self.expected = torch.zeros((self.q, triton.cdiv(self.m, bm)), dtype=torch.int32)
        for expert in range(self.q):
            for source in range(self.p):
                first = prefixes[source, self.rank * self.q + expert].item()
                end = first + all_counts[source, self.rank * self.q + expert].item()
                if first < end:
                    for tile in range(first // bm, triton.cdiv(end, bm)):
                        self.expected[expert, tile] += triton.cdiv(self.h, copy_n)
        self.max_expected = max(1, self.expected.max().item())
        self.expected = self.expected.to(device)
        self.copy_n = copy_n
        self.comm = min(
            comm,
            max(1, (grid or torch.cuda.get_device_properties(device).multi_processor_count) - 1),
        )
        self.counters = torch.zeros(3, dtype=torch.int32, device=device)
        self.grid = grid or torch.cuda.get_device_properties(device).multi_processor_count
        self.bm, self.bn, self.bk, self.warps, self.stages = bm, bn, bk, warps, stages
        self.descriptors = [
            TensorDescriptor.from_tensor(t.reshape(-1, t.shape[-1]), block)
            for t, block in (
                (self.local_inputs, [bm, bk]),
                (self.up, [bn, bk]),
                (self.activation, [bm, bk]),
                (self.down, [bn, bk]),
            )
        ]
        self.times = torch.zeros((self.grid, 5), device=device, dtype=torch.int64)
        self.epoch = 0
        torch.cuda.synchronize()
        dist.barrier()

    def update_input(self, x: torch.Tensor) -> None:
        """Replace prepared values on the caller stream without changing routing."""
        if (
            x.shape != (self.n, self.h)
            or x.dtype != torch.float16
            or x.device != self.packed.device
        ):
            raise ValueError("replacement input shape/dtype/device must match")
        pos = 0
        for expert, count in enumerate(self._count_list):
            self.packed[expert, :count].copy_(x[self._input_order[pos : pos + count] // self.k])
            pos += count

    def run(self, profile=False) -> torch.Tensor:
        """Push/ready → Up/GELU → Down/push in one cooperative kernel, then combine."""
        self.epoch += 1
        if self.epoch * max(self.grid, self.max_expected) >= 2**31:
            raise RuntimeError("epoch counter exhausted; recreate the prepared pipeline")
        self.last_kernel = peer_mlp_kernel[(self.grid,)](
            *self.ptrs,
            self.up,
            self.down,
            self.activation,
            self.local_inputs,
            self.tile_ready,
            self.sources,
            self.offsets,
            self.rows,
            self.counters,
            self.local_ptrs,
            self.ready_ptrs,
            self.segments,
            self.expected,
            *self.descriptors,
            self.times,
            self.epoch,
            self.h,
            self.f,
            self.q,
            self.p,
            self.rank,
            self.m,
            self.grid,
            self.bm,
            self.bn,
            self.bk,
            self.capacity,
            len(self.segments),
            self.comm,
            self.copy_n,
            profile,
            num_warps=self.warps,
            num_stages=self.stages,
            launch_cooperative_grid=True,
        )
        peer_combine_kernel[(triton.cdiv(self.n * self.h, 1024),)](
            self.returned,
            self.restore,
            self.weights,
            self.output,
            self.flags,
            self.epoch,
            self.n,
            self.h,
            self.k,
            self.p,
            1024,
        )
        return self.output


@triton.jit
def grouped_projection_kernel(
    X,
    W,
    Y,
    Rows,
    Ready,
    Expected,
    ADesc,
    BDesc,
    epoch,
    IN: tl.constexpr,
    OUT: tl.constexpr,
    M: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GELU: tl.constexpr,
    WAIT: tl.constexpr,
):
    """Independent local-memory grouped GEMM with the same precision boundaries."""
    expert = tl.program_id(0)
    row = tl.program_id(1) * BM + tl.arange(0, BM)
    col = tl.program_id(2) * BN + tl.arange(0, BN)
    count = tl.load(Rows + expert)
    if tl.program_id(1) * BM < count:
        if WAIT:
            tile = expert * tl.cdiv(M, BM) + tl.program_id(1)
            expected = tl.load(Expected + tile)
            _wait(Ready + tile, epoch * expected, True)
        tl.inline_asm_elementwise(
            "fence.proxy.async.global; mov.u32 $0, 0;",
            "=r",
            [],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        acc = tl.zeros((BM, BN), tl.float32)
        for step in range(tl.cdiv(IN, BK)):
            ks = step * BK + tl.arange(0, BK)
            a = ADesc.load([expert * M + tl.program_id(1) * BM, step * BK])
            b = BDesc.load([expert * OUT + tl.program_id(2) * BN, step * BK]).T
            acc = tl.dot(a, b, acc)
        if GELU:
            left, right = tl.split(tl.reshape(acc, (BM, 2, BN // 2)).permute(0, 2, 1))
            half_col = tl.program_id(2) * BN + tl.arange(0, BN // 2)
            v = left.to(tl.float16).to(tl.float32)
            value = (0.5 * v * (1.0 + libdevice.erf(v * 0.7071067811865475))).to(tl.float16)
            tl.store(
                Y + (expert * M + row[:, None]) * OUT + half_col[None, :],
                value,
                (row[:, None] < count) & (half_col[None, :] < OUT),
            )
            v = right.to(tl.float16).to(tl.float32)
            value = (0.5 * v * (1.0 + libdevice.erf(v * 0.7071067811865475))).to(tl.float16)
            tl.store(
                Y + (expert * M + row[:, None]) * OUT + half_col[None, :] + BN // 2,
                value,
                (row[:, None] < count) & (half_col[None, :] + BN // 2 < OUT),
            )
        else:
            tl.store(
                Y + (expert * M + row[:, None]) * OUT + col[None, :],
                acc.to(tl.float16),
                (row[:, None] < count) & (col[None, :] < OUT),
            )


@triton.jit
def peer_copy_kernel(
    PeerPtrs,
    Local,
    Sources,
    Offsets,
    Rows,
    H: tl.constexpr,
    M: tl.constexpr,
    Q: tl.constexpr,
    PUSH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = idx // H, idx % H
    expert = row // M
    count = tl.load(Rows + expert, expert < Q, 0)
    mask = (expert < Q) & (row % M < count)
    source = tl.load(Sources + row, mask, 0)
    offset = tl.load(Offsets + row, mask, 0)
    peer = tl.multiple_of(tl.load(PeerPtrs + source), 16).to(tl.pointer_type(tl.float16))
    if PUSH:
        v = tl.load(Local + idx, mask, 0)
        tl.store(peer + offset * H + col, v, mask)
    else:
        v = tl.load(peer + offset * H + col, mask, 0, cache_modifier=".cg")
        tl.store(Local + idx, v, mask)


@triton.jit
def peer_signal_kernel(FlagPtrs, epoch, P: tl.constexpr, RANK: tl.constexpr, START: tl.constexpr):
    for peer in range(P):
        ptr = tl.load(FlagPtrs + peer).to(tl.pointer_type(tl.int32))
        slot = RANK if START else P + RANK
        tl.atomic_xchg(ptr + slot, epoch, sem="release", scope="sys")
    if START:
        own = tl.load(FlagPtrs + RANK).to(tl.pointer_type(tl.int32))
        for peer in range(P):
            _wait(own + peer, epoch, True)


class TritonPeerSeparated(TritonPeerMLP):
    """Same peer protocol, explicit copy/compute stages; independent comparator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_outputs = torch.empty_like(self.local_inputs)

    def pull(self) -> None:
        """Diagnostic pull of immutable prepared inputs; synchronization is external."""
        peer_copy_kernel[(triton.cdiv(self.q * self.m * self.h, 8192),)](
            self.ptrs[0],
            self.local_inputs,
            self.sources,
            self.offsets,
            self.rows,
            self.h,
            self.m,
            self.q,
            False,
            8192,
        )

    def compute(self, wait: bool = False) -> torch.Tensor:
        """Two local TMA GEMMs; wait=True acquires the distributed input tiles."""
        grouped_projection_kernel[
            (self.q, triton.cdiv(self.m, self.bm), triton.cdiv(self.f, self.bn))
        ](
            self.local_inputs,
            self.up,
            self.activation,
            self.rows,
            self.tile_ready,
            self.expected,
            *self.descriptors[:2],
            self.epoch,
            self.h,
            self.f,
            self.m,
            self.bm,
            self.bn,
            self.bk,
            True,
            wait,
            num_warps=self.warps,
            num_stages=self.stages,
        )
        grouped_projection_kernel[
            (self.q, triton.cdiv(self.m, self.bm), triton.cdiv(self.h, self.bn))
        ](
            self.activation,
            self.down,
            self.local_outputs,
            self.rows,
            self.tile_ready,
            self.expected,
            *self.descriptors[2:],
            self.epoch,
            self.f,
            self.h,
            self.m,
            self.bm,
            self.bn,
            self.bk,
            False,
            False,
            num_warps=self.warps,
            num_stages=self.stages,
        )
        return self.local_outputs

    def send_input(self, signal: bool = True) -> None:
        """Publish input tiles; signal=False is an isolated immutable-data benchmark."""
        if len(self.segments):
            peer_input_push_kernel[(len(self.segments) * triton.cdiv(self.h, self.copy_n),)](
                self.packed,
                self.local_ptrs,
                self.ready_ptrs,
                self.segments,
                self.h,
                self.m,
                self.q,
                self.capacity,
                self.bm,
                self.copy_n,
                signal,
                num_warps=self.warps,
            )

    def push(self) -> None:
        """Write computed rows to their source GPUs; run() then publishes done."""
        peer_copy_kernel[(triton.cdiv(self.q * self.m * self.h, 8192),)](
            self.ptrs[1],
            self.local_outputs,
            self.sources,
            self.offsets,
            self.rows,
            self.h,
            self.m,
            self.q,
            True,
            8192,
        )

    def run(self) -> torch.Tensor:
        self.epoch += 1
        if self.epoch * max(self.grid, self.max_expected) >= 2**31:
            raise RuntimeError("epoch counter exhausted; recreate the prepared pipeline")
        peer_signal_kernel[(1,)](self.ptrs[2], self.epoch, self.p, self.rank, True, num_warps=4)
        self.send_input()
        self.compute(wait=True)
        self.push()
        peer_signal_kernel[(1,)](self.ptrs[2], self.epoch, self.p, self.rank, False, num_warps=4)
        peer_combine_kernel[(triton.cdiv(self.n * self.h, 1024),)](
            self.returned,
            self.restore,
            self.weights,
            self.output,
            self.flags,
            self.epoch,
            self.n,
            self.h,
            self.k,
            self.p,
            1024,
        )
        return self.output


@triton.jit
def peer_input_push_kernel(
    Input,
    LocalPtrs,
    ReadyPtrs,
    Segments,
    H: tl.constexpr,
    M: tl.constexpr,
    Q: tl.constexpr,
    C: tl.constexpr,
    BM: tl.constexpr,
    COPY_N: tl.constexpr,
    SIGNAL: tl.constexpr,
):
    nt: tl.constexpr = tl.cdiv(H, COPY_N)
    task = tl.program_id(0)
    seg = task // nt
    expert = tl.load(Segments + seg * 4)
    src = tl.load(Segments + seg * 4 + 1)
    count = tl.load(Segments + seg * 4 + 2)
    dst = tl.load(Segments + seg * 4 + 3)
    rows = tl.arange(0, BM)
    cols = task % nt * COPY_N + tl.arange(0, COPY_N)
    peer = expert // Q
    remote = tl.multiple_of(tl.load(LocalPtrs + peer), 16).to(tl.pointer_type(tl.float16))
    v = tl.load(
        Input + (expert * C + src + rows[:, None]) * H + cols[None, :],
        (rows[:, None] < count) & (cols[None, :] < H),
        0,
    )
    tl.store(
        remote + ((expert % Q) * M + dst + rows[:, None]) * H + cols[None, :],
        v,
        (rows[:, None] < count) & (cols[None, :] < H),
    )
    if SIGNAL:
        tl.debug_barrier()
        ready = tl.load(ReadyPtrs + peer).to(tl.pointer_type(tl.int32))
        tl.atomic_add(
            ready + (expert % Q) * tl.cdiv(M, BM) + dst // BM, 1, sem="release", scope="sys"
        )
