"""基于单机 Peer 显存和 TMA 的 SwiGLU EP 工作区（逐分支实现）。

本文件保留“每个 Top-K 分支都跨卡传输”的对照实现，同时作为去重 Peer 路径的
基类。它适配第 12 课的分离执行协议，支持 BF16 SwiGLU、动态 Top-K 和设备端
元数据。导入本模块不会创建模型、进程组或 CUDA 工作区。

生产 Runner 默认使用 NCCL 路径；只有显式选择 peer 传输时才使用本实现。
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm
import triton
from triton.tools.tensor_descriptor import TensorDescriptor

from mini_ep.distributed.topology import ExpertParallelTopology
from mini_ep.model.contracts import validate_inputs
from mini_ep.kernels.grouped_gemm.plan import GroupedGemmPlan
from mini_ep.kernels.grouped_gemm.triton_impl import grouped_gemm
from mini_ep.kernels.combine_triton import combine_triton
from mini_ep.profiler.stages import Stages
from mini_ep.kernels.peer_triton import _layout, _plan, _send, _signal, _up, _push


class PeerSwiGLU:
    """可复用的单机 EP 工作区：一个 EP 组、一条 CUDA Stream 对应一份缓冲。

    布局约定：
        - Token [N,H]、本地权重 [L,2F,H]/[L,H,F] 为连续 CUDA BF16；
        - 未展开的专家编号 [N,K] 为 int64，路由权重 [N,K] 为 FP32；
        - 专家均匀放置 E=P*L，每个 Rank 的缓冲形状相同。

    通信方式：只 AllGather 整数计数 [P,E]；Token 大向量通过 Peer 写直接进入
    目标卡的紧凑 Expert-major 缓冲，每个 Expert 内按来源 Rank 排序。返回元数据
    [R,2] 保存 (来源 EP rank, 原分支号)。所有 Rank 必须以相同顺序调用，空来源
    也必须参与。CUDA 设备由 local rank 选择，指针表按 EP rank 排列。

    同步方式：开始信号保护上一轮 Combine 的读取；复制 Kernel 发布 tile 就绪计数；
    Up Kernel 等到当前 tile 的计数后再读；Down/回传在同一 Stream 后续执行；
    结束信号保护返回缓冲的读取。完成由 GPU 信号保证，不是 Python 返回或
    CPU Work handle。销毁工作区/进程组前必须 drain，禁止无保护地跨 Stream 复用。

    容量：输入缓冲按 P*max(source_rows)*K 行准备并向上对齐，覆盖极端倾斜路由；
    返回缓冲按 max(source_rows)*K 行准备。
    """
    def __init__(self, group: dist.ProcessGroup, hidden: int, local_experts: int, top_k: int,
                 source_rows: list[int], device: torch.device) -> None:
        """创建一份可跨层复用的 Peer 工作区。

        参数：
            group: 当前单机 EP 进程组。
            hidden: 隐藏维 H。
            local_experts: 本 Rank 的专家数 L。
            top_k: 每 Token 的 Top-K 分支数 K。
            source_rows: 每个 EP Rank 的 Token 上限，按 EP rank 排列。
            device: 当前 Rank 使用的 CUDA 设备。

        构造阶段会分配对称缓冲、建立 Peer 映射并同步一次。这些属于准备开销，
        不能算进稳态推理时间。
        """
        self.group = group
        self.p, self.rank = dist.get_world_size(group), dist.get_rank(group)
        self.topology = ExpertParallelTopology(local_experts * self.p, self.p, self.rank)
        self.e, self.l, self.h, self.k = local_experts * self.p, local_experts, hidden, top_k
        if len(source_rows) != self.p or any(n < 0 for n in source_rows):
            raise ValueError(f"source_rows={source_rows} must have {self.p} nonnegative entries")
        if min(hidden, local_experts, top_k) < 1 or top_k > self.e:
            raise ValueError(f"require H,L,K>0 and K<=E, got {hidden},{local_experts},{top_k}")
        if self.e & (self.e - 1):
            raise ValueError(f"experimental peer layout requires power-of-two E, got {self.e}")
        if not torch.cuda.is_available() or device.type != "cuda":
            raise ValueError("PeerSwiGLU requires CUDA")
        if torch.cuda.get_device_capability(device)[0] < 9 or hidden % 8:
            raise ValueError("TMA peer path requires SM>=90 and H divisible by 8; use MINIEP_TRANSPORT=nccl on unsupported GPUs")
        self.device, self.stream = device, torch.cuda.current_stream(device).cuda_stream
        self.source_limit = max(1, max(source_rows))
        self.capacity = triton.cdiv(self.source_limit * self.p * top_k, 128) * 128
        self.return_capacity = max(1, self.source_limit * top_k)
        self.epoch = 0
        # 缓冲张量与跨卡映射句柄必须一起保活，直到所有已提交 GPU 工作结束。
        specs = [((self.capacity, hidden), torch.bfloat16),
                 ((self.capacity, 2), torch.int64), ((self.capacity,), torch.float32),
                 ((self.capacity,), torch.int32),
                 ((self.return_capacity, hidden), torch.bfloat16),
                 ((2 * self.p,), torch.int32)]
        self.buffers = [symm.empty(*shape, dtype=dtype, device=device) for shape, dtype in specs]
        self.data, self.meta, self.weights, self.ready, self.returned, self.flags = self.buffers
        self.ready.zero_()
        self.flags.zero_()
        self.handles = [symm.rendezvous(t, group) for t in self.buffers]
        self.ptrs = [torch.tensor(h.buffer_ptrs, dtype=torch.uint64, device=device) for h in self.handles]
        self.counts = torch.empty((self.p, self.e), dtype=torch.int64, device=device)
        self.starts = torch.empty(self.e, dtype=torch.int64, device=device)
        self.targets = torch.empty_like(self.starts)
        self.source_prefixes = torch.empty_like(self.starts)
        self.tile_starts = torch.empty_like(self.starts)
        self.offsets = torch.empty(self.l + 1, dtype=torch.int64, device=device)
        tiles = triton.cdiv(self.capacity, 128) + self.l - 1
        self.experts = torch.empty(tiles, dtype=torch.int64, device=device)
        self.rows = torch.empty_like(self.experts)
        self.expected = torch.empty_like(self.experts)
        self.plan = GroupedGemmPlan(self.offsets, self.experts, self.rows, 128, self.capacity)
        # 分配/rendezvous 属于准备阶段；只在这里同步，不把每轮稳态执行强制同步。
        torch.cuda.synchronize(device)
        dist.barrier(group)

    def forward(
        self, tokens: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
        w13: torch.Tensor, w2: torch.Tensor, *, stages: Stages | None = None,
        validate: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """原顺序 [N,H] -> 原顺序 BF16 [N,H]。

        输入、权重、布局、Stream 和容量约束见类注释。本方法只负责一次前向，
        不创建进程组，也不改变调用方持有的 group。
        """
        if validate:
            validate_inputs(tokens, ids, weights, w13, w2, self.topology)
        if (tokens.dtype != torch.bfloat16 or tokens.device != self.device
            or ids.dtype != torch.int64 or ids.shape[1] != self.k
            or tokens.shape[1] != self.h or w13.shape[0] != self.l
            or tokens.shape[0] > self.source_limit
            or weights.dtype != torch.float32 or weights.shape != ids.shape
            or w13.dtype != tokens.dtype or w2.dtype != tokens.dtype
            or any(t.device != tokens.device for t in (ids, weights, w13, w2))
            or not all(t.is_contiguous() for t in (tokens, ids, weights, w13, w2))):
            raise ValueError("peer expects contiguous BF16 tokens/weights, int64 IDs, FP32 routing weights within prepared H/L/K/N capacity")
        if w2.shape[2] % 128:
            raise ValueError(f"TMA fused Up requires F divisible by 128, got {w2.shape[2]}")
        if torch.cuda.current_stream(self.device).cuda_stream != self.stream:
            raise RuntimeError("peer workspace calls must use their construction CUDA stream")
        if self.epoch >= 2**31 - 1:
            raise RuntimeError("peer epoch exhausted; drain and recreate workspace")
        stages = stages or Stages()
        self.epoch += 1
        with stages.stage("DISPATCH"):
            flat = ids.view(-1)
            order = torch.argsort(flat, stable=True)
            counts = torch.bincount(flat, minlength=self.e)
            # 只收集整数路由计数，大向量仍由 Peer 写入；这不是用 Token AllGather 模拟 EP。
            dist.all_gather_into_tensor(self.counts.view(-1), counts, group=self.group)
            _layout[(1,)](self.counts, self.starts, self.targets, self.source_prefixes, self.tile_starts, self.offsets,
                self.e, self.p, self.l, self.rank, 128, self.e, triton.next_power_of_2(self.p))
            _plan[(1,)](self.offsets, self.experts, self.rows, self.expected, self.l, 128,
                self.experts.numel(), triton.next_power_of_2(self.l),
                triton.next_power_of_2(self.experts.numel()), triton.cdiv(self.h, 256))
            self.ready.zero_()
            _signal[(1,)](self.ptrs[5], self.epoch, self.p, self.rank, False)
        with stages.stage("A2A_FORWARD"):
            if flat.numel():
                _send[(triton.cdiv(flat.numel(), 16), triton.cdiv(self.h, 256))](tokens, flat, weights, order,
                    self.starts, self.targets, self.source_prefixes, self.tile_starts, *self.ptrs[:4],
                    flat.numel(), self.h, self.k, self.l, self.rank, 128, 256)
        with stages.stage("EXPERT"):
            f = w2.shape[2]
            activated = torch.empty((self.capacity, f), dtype=tokens.dtype, device=self.device)
            input_desc = TensorDescriptor(self.data, [self.capacity, self.h], [self.h, 1], [128, 64])
            weight_desc = TensorDescriptor(w13, [self.l * 2 * f, self.h], [self.h, 1], [128, 64])
            _up[(self.experts.numel(), triton.cdiv(f, 128))](input_desc, weight_desc, activated,
                self.offsets, self.experts, self.rows, self.ready, self.expected, f, self.h, 128, 128, 64,
                num_warps=8, num_stages=3, enable_fp_fusion=False)
            output = grouped_gemm(activated, w2, self.plan, output_dtype=tokens.dtype, row_weights=self.weights)
        with stages.stage("A2A_RETURN"):
            _push[(triton.cdiv(self.capacity, 16), triton.cdiv(self.h, 256))](output, self.meta, self.offsets,
                self.ptrs[4], self.h, self.l, 256)
            _signal[(1,)](self.ptrs[5], self.epoch, self.p, self.rank, True)
        with stages.stage("COMBINE"):
            restore = torch.arange(flat.numel(), dtype=torch.int64, device=self.device)
            result = combine_triton(self.returned, restore, ids, self.topology, model_dtype=tokens.dtype)
        return result, dict(transport="peer", layout="compact-expert-major", branches=flat.numel(),
            source_tokens=tokens.shape[0], input_capacity=self.capacity, epoch=self.epoch,
            return_dtype=str(tokens.dtype), preweighted=True, expert_backend="peer_tma_swiglu",
            fixed_a2a=False, ep_size=self.p)


_WORKSPACES: dict[tuple, PeerSwiGLU] = {}


def peer_forward(
    tokens: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
    w13: torch.Tensor, w2: torch.Tensor, group: dist.ProcessGroup, *,
    source_rows: list[int], stages: Stages | None = None, validate: bool = True,
) -> tuple[torch.Tensor, dict]:
    """动态路由入口：按 Token 数选择容量桶，跨模型层复用工作区。

    Host 侧 source_rows 必须等于各 EP Rank 的实际 Token 数，并且所有 Rank
    看到同一列表。vLLM 可以在不读取 GPU 计数的情况下提供它。扩容或 rendezvous
    属于准备开销，不能从端到端计时中隐藏。缓存只在当前进程内有效，仅用于推理，
    不可重入。
    """
    p = dist.get_world_size(group)
    rank = dist.get_rank(group)
    if (len(source_rows) != p or any(n < 0 for n in source_rows)
        or source_rows[rank] != tokens.shape[0]):
        raise ValueError(f"source_rows={source_rows} must match EP={p} and local N={tokens.shape[0]}")
    limit = triton.next_power_of_2(max(1, max(source_rows)))
    # decode 若复用最大 prefill 容量，会启动大量空 tile/复制 CTA。
    # 按当前 Token 数取2次幂容量桶，避免大 prefill 放大后续小 decode 的工作量。
    # 容量桶按2次幂增长；实际显存还包含行对齐和每桶元数据的额外开销。
    key = (group, tokens.device, tokens.shape[1], w13.shape[0], ids.shape[1], limit)
    workspace = _WORKSPACES.get(key)
    if workspace is None:
        replacement = PeerSwiGLU(group, tokens.shape[1], w13.shape[0], ids.shape[1],
                                [limit] * p, tokens.device)
        _WORKSPACES[key] = replacement
        workspace = replacement
    return workspace.forward(tokens, ids, weights, w13, w2, stages=stages, validate=validate)


def clear_peer_workspaces() -> None:
    """释放 Peer 映射前，先等待所有工作区已提交的 GPU 工作结束。"""
    for workspace in _WORKSPACES.values():
        torch.cuda.synchronize(workspace.device)
    _WORKSPACES.clear()
