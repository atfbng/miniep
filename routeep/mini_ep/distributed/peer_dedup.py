"""Peer EP 去重路径：每个 (Token, 目标卡) 只传一份大向量，在目标卡先归约输出。

与逐分支 Peer 实现的区别：
    - Token 大向量只在来源卡写一次；
    - 专家编号和 FP32 路由权重仍逐分支发送；
    - 接收卡把分支展开给 Expert，并在回传前合并同一 Token 的本卡输出。

独立精度对照见 reference.moe.moe_reference 和 owner_partials_reference。
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm
import triton
from triton.tools.tensor_descriptor import TensorDescriptor

from mini_ep.distributed.peer import PeerSwiGLU
from mini_ep.model.contracts import validate_inputs
from mini_ep.profiler.stages import Stages
from mini_ep.kernels.peer_triton import _layout, _plan, _signal, _up
from mini_ep.kernels.grouped_gemm.plan import GroupedGemmPlan
from mini_ep.kernels.grouped_gemm.triton_impl import _tile_config, grouped_gemm
from mini_ep.kernels.peer_dedup_triton import (
    send_unique_kernel, send_metadata_kernel, expand_local_kernel,
    push_owner_partials_kernel, combine_owner_partials_kernel,
)


class DeduplicatedPeerSwiGLU(PeerSwiGLU):
    """单机 BF16 去重 EP 工作区，继承基类的 EP 组、单 Stream 和布局约定。

    新增对称缓冲（C 为每个来源预留的 Token 上限）：
        unique_inputs [P*C,H] BF16：按来源/原 Token 排列的唯一输入。
        row_lookup [P*C,K] int64：把来源/Token/slot 映射到本 owner 的 Expert 行；
            不属于本卡的 slot 为 -1。
        owner_returned [P*C,H] BF16：按 owner/原 Token 排列的部分和。
        metadata_ready [tile_capacity] int32：每个 Expert tile 的就绪计数。

    发送与计算：来源卡先复制唯一输入，再发 metadata；同一 Stream 的 release
    让目标卡 acquire 后能看到输入。目标卡每个 tile 等自己的 metadata 就绪后，
    才在本卡展开给 Expert，不额外加“全卡输入完成”屏障。空来源也参与信号。

    精度：Expert 输出已经加权并舍入到 BF16；目标卡按原 slot 顺序用 FP32 累加，
    每个 owner 只舍入一次后回传；来源再按 owner EP rank 升序用 FP32 累加。
    不能改成逐次 BF16 累加、无序原子求和或提前加权。

    生命周期：每轮先清计数，再发布开始 epoch 信号；结束信号排在回传之后。
    释放映射前必须 drain，禁止跨 Stream 复用。
    """
    def __init__(self, group: dist.ProcessGroup, hidden: int, local_experts: int,
                 top_k: int, source_rows: list[int], device: torch.device) -> None:
        """在基类工作区上再分配去重路径需要的对称缓冲。

        参数与 PeerSwiGLU.__init__ 相同。额外缓冲用于唯一输入、行查找、owner
        部分和以及 metadata 就绪计数；构造完成后同样只做一次准备同步。
        """
        super().__init__(group, hidden, local_experts, top_k, source_rows, device)
        self.c = self.source_limit
        specs = [((self.p * self.c, self.h), torch.bfloat16),
                 ((self.p * self.c, self.k), torch.int64),
                 ((self.p * self.c, self.h), torch.bfloat16),
                 ((self.experts.numel(),), torch.int32)]
        self.dedup_buffers = [symm.empty(*shape, dtype=dtype, device=device) for shape,dtype in specs]
        self.unique_inputs, self.row_lookup, self.owner_returned, self.metadata_ready = self.dedup_buffers
        self.dedup_handles = [symm.rendezvous(t, group) for t in self.dedup_buffers]
        self.dedup_ptrs = [torch.tensor(h.buffer_ptrs, dtype=torch.uint64, device=device) for h in self.dedup_handles]
        # Down 投影按容量选择 tile（与 NCCL 路径同一张表）。TMA Up 与就绪计数仍绑定
        # BM=128 的计划；BM 不同时另建一份只供 Down 使用的计划，共用同一份 offsets。
        down_bm = _tile_config(torch.bfloat16, self.capacity)[0]
        if down_bm == 128:
            self.down_plan = self.plan
        else:
            tiles = triton.cdiv(self.capacity, down_bm) + self.l - 1
            self.down_experts = torch.empty(tiles, dtype=torch.int64, device=device)
            self.down_rows = torch.empty_like(self.down_experts)
            # _plan 也会写每个 tile 的就绪阈值；Down 不等待就绪，写入这块临时缓冲即可。
            self.down_expected = torch.empty_like(self.down_experts)
            self.down_plan = GroupedGemmPlan(self.offsets, self.down_experts, self.down_rows,
                                             down_bm, self.capacity)
        torch.cuda.synchronize(device)
        dist.barrier(group)

    def forward(self, tokens: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
                w13: torch.Tensor, w2: torch.Tensor, *, stages: Stages | None = None,
                validate: bool = True) -> tuple[torch.Tensor, dict]:
        """执行一次去重 Peer 前向：原顺序 [N,H] -> 原顺序 BF16 [N,H]。

        布局、Stream、容量和精度约定见类注释。输入必须落在构造时预留的容量内，
        且所有 EP Rank 必须以相同顺序调用。
        """
        if validate:
            validate_inputs(tokens, ids, weights, w13, w2, self.topology)
        if (tokens.dtype != torch.bfloat16 or tokens.device != self.device
            or ids.dtype != torch.int64 or ids.shape[1] != self.k
            or tokens.shape[1] != self.h or w13.shape[0] != self.l
            or tokens.shape[0] > self.source_limit
            or weights.dtype != torch.float32 or weights.shape != ids.shape
            or w13.dtype != tokens.dtype or w2.dtype != tokens.dtype
            or any(t.device != tokens.device for t in (ids,weights,w13,w2))
            or not all(t.is_contiguous() for t in (tokens,ids,weights,w13,w2))):
            raise ValueError('deduplicated peer requires contiguous CUDA BF16 payload/weights, int64 IDs, FP32 routing weights within prepared H/L/K/N')
        if w2.shape[2] % 128:
            raise ValueError(f'TMA fused Up requires F divisible by 128, got {w2.shape[2]}')
        if torch.cuda.current_stream(self.device).cuda_stream != self.stream:
            raise RuntimeError('deduplicated peer calls must use the construction CUDA stream')
        if self.epoch >= 2**31 - 1:
            raise RuntimeError('peer phase counter exhausted; drain and recreate workspace')
        stages = stages or Stages()
        self.epoch += 1
        with stages.stage('DISPATCH'):
            flat = ids.view(-1)
            order = torch.argsort(flat, stable=True)
            # 固定长度 E 的整数 scatter_add：结果与 bincount 相同，但不需要读回最大值，
            # 因此没有 Host 同步（CUDA bincount 会同步）。
            counts = torch.zeros(self.e, dtype=torch.int64, device=self.device)
            counts.scatter_add_(0, flat, torch.ones_like(flat))
            dist.all_gather_into_tensor(self.counts.view(-1), counts, group=self.group)
            _layout[(1,)](self.counts,self.starts,self.targets,self.source_prefixes,self.tile_starts,self.offsets,
                self.e,self.p,self.l,self.rank,128,self.e,triton.next_power_of_2(self.p))
            _plan[(1,)](self.offsets,self.experts,self.rows,self.expected,self.l,128,
                self.experts.numel(),triton.next_power_of_2(self.l),
                triton.next_power_of_2(self.experts.numel()),triton.cdiv(self.h,256))
            if self.down_plan is not self.plan:
                _plan[(1,)](self.offsets,self.down_experts,self.down_rows,self.down_expected,
                    self.l,self.down_plan.block_m,self.down_experts.numel(),triton.next_power_of_2(self.l),
                    triton.next_power_of_2(self.down_experts.numel()),triton.cdiv(self.h,256))
            self.ready.zero_()
            # 旧路由不得残留到本轮：未被本 owner 选中的 slot 必须保持 -1。
            self.row_lookup.fill_(-1)
            self.metadata_ready.zero_()
            _signal[(1,)](self.ptrs[5],self.epoch,self.p,self.rank,False)
        with stages.stage('A2A_FORWARD'):
            if flat.numel():
                # 同一 stream 先复制唯一输入、后发 metadata；后者的 release 发布前者写入。
                send_unique_kernel[(triton.cdiv(tokens.shape[0],16),self.p,triton.cdiv(self.h,256))](
                    tokens,flat,self.dedup_ptrs[0],tokens.shape[0],self.h,self.k,self.l,
                    self.c,self.rank,triton.next_power_of_2(self.k),256)
                send_metadata_kernel[(triton.cdiv(flat.numel(),256),)](
                    flat,weights,order,self.starts,self.targets,self.ptrs[1],self.ptrs[2],
                    self.dedup_ptrs[1],self.dedup_ptrs[3],self.source_prefixes,self.tile_starts,
                    flat.numel(),self.k,self.l,self.c,self.rank,128,256)
            # 每个 Expert tile 等自己的 metadata 就绪后在本卡展开，不加全局输入完成屏障。
            expand_local_kernel[(self.experts.numel(),8,triton.cdiv(self.h,256))](
                self.unique_inputs,self.meta,self.data,self.offsets,self.experts,self.rows,
                self.ready,self.metadata_ready,self.expected,self.h,self.k,self.c,128,256)
        with stages.stage('EXPERT'):
            f = w2.shape[2]
            activated = torch.empty((self.capacity,f),dtype=tokens.dtype,device=self.device)
            input_desc = TensorDescriptor(self.data,[self.capacity,self.h],[self.h,1],[128,64])
            weight_desc = TensorDescriptor(w13,[self.l*2*f,self.h],[self.h,1],[128,64])
            _up[(self.experts.numel(),triton.cdiv(f,128))](input_desc,weight_desc,activated,
                self.offsets,self.experts,self.rows,self.ready,self.expected,f,self.h,128,128,64,
                num_warps=8,num_stages=3,enable_fp_fusion=False)
            expert_outputs = grouped_gemm(activated,w2,self.down_plan,output_dtype=tokens.dtype,row_weights=self.weights)
        with stages.stage('A2A_RETURN'):
            # 在目标卡按原 slot 顺序 FP32 归约，舍入一次后每个 owner 只回一份。
            push_owner_partials_kernel[(triton.cdiv(self.p*self.c,16),triton.cdiv(self.h,256))](
                expert_outputs,self.row_lookup,self.dedup_ptrs[2],self.h,self.k,self.c,
                self.p,self.rank,256,enable_fp_fusion=False)
            # 结束信号必须排在回传之后，来源 acquire 后才可读取本轮部分和。
            _signal[(1,)](self.ptrs[5],self.epoch,self.p,self.rank,True)
        with stages.stage('COMBINE'):
            result = torch.empty_like(tokens)
            if tokens.numel():
                combine_owner_partials_kernel[(triton.cdiv(tokens.numel(),256),)](
                    self.owner_returned,ids,result,tokens.shape[0],self.h,self.k,self.c,
                    self.p,self.l,256,enable_fp_fusion=False)
        return result,dict(transport='peer',deduplicated=True,layout='unique-token-owner / compact-expert-major',
            branches=flat.numel(),source_tokens=tokens.shape[0],input_capacity=self.capacity,
            unique_capacity=self.p*self.c,epoch=self.epoch,return_dtype=str(tokens.dtype),
            # Peer 只支持 BF16 回传：owner 部分和缓冲与模型 dtype 相同，权重在 Down 中应用。
            model_dtype=str(tokens.dtype),return_buffer_dtype=str(self.owner_returned.dtype),
            return_dtype_requested='auto',fuse_return_weight=True,
            preweighted=True,expert_backend='peer_tma_swiglu',owner_reduction='slot_order_fp32_then_bf16',
            fixed_a2a=False,ep_size=self.p)


_WORKSPACES: dict[tuple,DeduplicatedPeerSwiGLU] = {}


def deduplicated_peer_forward(tokens: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
                              w13: torch.Tensor, w2: torch.Tensor, group: dist.ProcessGroup, *,
                              source_rows: list[int], stages: Stages | None = None,
                              validate: bool = True) -> tuple[torch.Tensor,dict]:
    """动态路由入口：按来源容量桶复用对称工作区，不缓存具体路由内容。"""
    p,rank = dist.get_world_size(group),dist.get_rank(group)
    if len(source_rows)!=p or any(n<0 for n in source_rows) or source_rows[rank]!=tokens.shape[0]:
        raise ValueError(f'source_rows={source_rows} must match EP={p} and local N={tokens.shape[0]}')
    # 每层可复用相同容量桶，但不缓存 IDs/权重；它们每轮随真实路由更新。
    limit = triton.next_power_of_2(max(1,max(source_rows)))
    key = (group,tokens.device,tokens.shape[1],w13.shape[0],ids.shape[1],limit)
    workspace = _WORKSPACES.get(key)
    if workspace is None:
        workspace = DeduplicatedPeerSwiGLU(group,tokens.shape[1],w13.shape[0],ids.shape[1],
                                         [limit]*p,tokens.device)
        _WORKSPACES[key] = workspace
    return workspace.forward(tokens,ids,weights,w13,w2,stages=stages,validate=validate)


def clear_deduplicated_peer_workspaces() -> None:
    """释放 Peer 映射/进程组前，先等待已提交的 GPU 工作全部结束。"""
    for workspace in _WORKSPACES.values():
        torch.cuda.synchronize(workspace.device)
    _WORKSPACES.clear()
