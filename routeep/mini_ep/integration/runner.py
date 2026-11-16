"""TP 局部 Token 归属、跨 DP 专家交换和 TP 局部重组。"""

import os

import torch
import torch.distributed as dist
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.distributed import get_ep_group, get_tp_group, get_dp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from mini_ep.distributed.all_to_all import FIXED_OWNER_MAX_TOKENS, fixed_owner_capacity
from mini_ep.distributed.topology import ExpertParallelTopology
from mini_ep.model.deduplicated_moe import ep_forward
from mini_ep.model.contracts import validate_inputs
from mini_ep.model.router import RouterOutput
from mini_ep.profiler.stages import Stages
from mini_ep.integration.input_layout import token_range, gather_tp_rows
from mini_ep.integration.transport import get_peer_forward, resolve_transport

logger = init_logger("vllm.mini_ep")


class MiniEPRunner(MoERunner):
    """作用：用 RouteEP 执行 Qwen MoE 层，其余模型部分仍由 vLLM 管理。

    输入归属随并行模式变化：

    - DP=1：两个 TP Rank 收到相同的复制行，使用 :func:`token_range`
      把每行确定性地分配给唯一来源 Rank；
    - DP=2、TP=2：Qwen 已把 DP 行切分到 TP Rank，直接消费本地行。
    - DP=2、TP=1：每行已归属于当前 DP 来源，无需 TP 切分或重组。

    专家权重和 EP/TP/DP 进程组均由 vLLM 创建，本类只读取和使用，
    不会重新初始化或销毁它们。

    构造参数通过 ``*args, **kwargs`` 原样传给 ``MoERunner``，通常包括
    层配置、Gate、Router 和专家权重对象。构造完成后还会校验课程要求的
    TP/DP/EP 拓扑、eager 执行、BF16、Triton 权重布局等约束。
    """

    def __init__(self, *args, **kwargs):
        """作用：初始化 Runner 并校验 vLLM 提供的并行配置。

        参数：
            *args: 原样传给 ``MoERunner.__init__`` 的位置参数。
            **kwargs: 原样传给 ``MoERunner.__init__`` 的关键字参数，
                通常包含 MoE 层配置和权重对象。
        """
        # 父类提供 Gate、Router 和专家权重对象，本类接管 MoE 的前向执行。
        super().__init__(*args, **kwargs)
        cfg = get_current_vllm_config()
        return_name = os.environ.get("MINIEP_RETURN_DTYPE", "auto")
        return_types = {
            "auto": None,
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }
        if return_name not in return_types:
            raise ValueError(
                f"MINIEP_RETURN_DTYPE must be auto/fp32/bf16/fp16, got {return_name!r}"
            )
        self.return_dtype = return_types[return_name]
        # NCCL 去重短批次默认用固定容量：无计数交换、无 Host 同步、分发一次 A2A。
        self.owner_fixed = os.environ.get("MINIEP_OWNER_FIXED", "auto")
        fixed_owner_capacity(self.owner_fixed, [0])  # 尽早校验取值
        # Qwen3.5 共享专家可在旁路 Stream 上与路由专家链路并行；只在 TP=1 时可用，
        # 此时共享专家是本地 MLP，不含 TP collective，不会与 EP A2A 交错进入不同通信组。
        # 默认关闭：2026-10-27 H100 EP2 实测 auto 使 decode 每步变慢约 9 ms，原因待查。
        self.shared_stream_mode = os.environ.get("MINIEP_SHARED_STREAM", "off")
        if self.shared_stream_mode not in ("auto", "off"):
            raise ValueError(
                f"MINIEP_SHARED_STREAM must be auto/off, got {self.shared_stream_mode!r}"
            )
        p = cfg.parallel_config
        # 显式校验四卡教学拓扑或双卡 Pure EP 拓扑，避免运行到
        # 集合通信才发现配置与输入归属不匹配。
        required = dict(
            pipeline_parallel_size=1,
            enable_expert_parallel=True,
            enable_eplb=False,
            enable_dbo=False,
            expert_placement_strategy="linear",
        )
        for name, expected in required.items():
            actual = getattr(p, name)
            if actual != expected:
                raise ValueError(
                    f"RouteEP requires {name}={expected!r}, got {actual!r}"
                )
        if p.tensor_parallel_size not in (1, 2) or (
            p.tensor_parallel_size == 1 and p.data_parallel_size != 2
        ):
            raise ValueError("RouteEP requires TP2/DP1,2 or TP1/DP2")
        if p.all2all_backend != "deepep_high_throughput":
            raise ValueError(f"unsupported RouteEP framework backend {p.all2all_backend!r}")
        if p.data_parallel_size not in (1, 2):
            raise ValueError(f"RouteEP requires DP=1/2, got {p.data_parallel_size}")
        # 默认 eager + 同步调度，保证每个阶段按代码顺序执行，便于观察通信。
        # 唯一允许的 graph 配置是 engine_options(cudagraph_capture_sizes=...) 生成的
        # FULL_DECODE_ONLY + 编译模式 NONE：只有 decode 批次被 capture，prefill 仍 eager。
        from vllm.config import CompilationMode, CUDAGraphMode

        comp = cfg.compilation_config
        self.cudagraph = comp.cudagraph_mode != CUDAGraphMode.NONE
        if self.cudagraph:
            sizes = comp.cudagraph_capture_sizes or []
            if (
                comp.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY
                or comp.mode != CompilationMode.NONE
            ):
                raise ValueError(
                    f"RouteEP graph requires FULL_DECODE_ONLY with compilation mode NONE, "
                    f"got cudagraph_mode={comp.cudagraph_mode}, mode={comp.mode}"
                )
            if not sizes or max(sizes) > FIXED_OWNER_MAX_TOKENS:
                raise ValueError(
                    f"RouteEP graph capture sizes must be <= {FIXED_OWNER_MAX_TOKENS}, got {sizes}"
                )
            if self.owner_fixed != "auto":
                raise ValueError("RouteEP graph requires MINIEP_OWNER_FIXED=auto")
        elif not cfg.model_config.enforce_eager:
            raise ValueError("requires eager execution or RouteEP FULL_DECODE_ONLY graph")
        if cfg.quant_config is not None or cfg.speculative_config is not None:
            raise ValueError("requires nonquantized, nonspeculative execution")
        if cfg.scheduler_config.async_scheduling:
            raise ValueError("requires synchronous scheduling (--no-async-scheduling)")
        if cfg.model_config.dtype != torch.bfloat16:
            raise ValueError(
                f"vLLM integration requires BF16, got {cfg.model_config.dtype}"
            )
        if cfg.kernel_config.moe_backend != "triton":
            raise ValueError("requires moe_backend=triton for canonical weights")
        # Qwen3 和 Qwen3.5 的文本 MoE 共用这里的 Expert 权重布局。
        family = cfg.model_config.hf_text_config.model_type
        if family not in ("qwen3_moe", "qwen3_5_moe_text"):
            raise ValueError(f"supports Qwen3/Qwen3.5 MoE, got {family!r}")
        if (self._shared_experts is not None) != (family == "qwen3_5_moe_text"):
            raise ValueError(
                "Qwen3.5 requires a separate native shared expert; fused shared experts are unsupported"
            )
        # 复用 vLLM 的通信组，不额外创建进程组；组内 rank 不等同于全局 rank。
        self.input_is_tp_sharded = p.use_sequence_parallel_moe
        # TP1/DP2 has no TP row split: every row already belongs to this DP
        # source. Attention/shared experts are replicated; routed experts are
        # sharded across EP=2. Both transport choices always deduplicate.
        self.input_is_dp_local = self.input_is_tp_sharded or (
            p.tensor_parallel_size == 1 and p.data_parallel_size == 2
        )
        if self._shared_experts is not None and not self.input_is_dp_local:
            raise ValueError(
                "Qwen3.5 DP=1 shared/routed joint TP rounding is outside the DeepEP-only path; use DP=2"
            )
        self.ep_group = get_ep_group().device_group
        self.tp_group = get_tp_group().device_group
        # tp_rank 是 TP 组内的局部编号，不是全局 rank。
        self.tp_rank, self.tp_size = dist.get_rank(self.tp_group), dist.get_world_size(
            self.tp_group
        )
        self.dp_rank = get_dp_group().rank_in_group
        self.topology = ExpertParallelTopology(
            self.moe_config.num_experts,
            dist.get_world_size(self.ep_group),
            dist.get_rank(self.ep_group),
        )
        # 四卡示例：EP=[0,1,2,3]，两个 DP 副本的 TP 组分别为 [0,1]、[2,3]。
        # 专家跨四卡放置，每个 TP 组内的 Attention 仍协作处理同一批请求。
        members = dist.get_process_group_ranks(self.ep_group)
        tp_members = dist.get_process_group_ranks(self.tp_group)
        begin = self.dp_rank * self.tp_size
        # 要求 EP 组按 DP-major/TP-minor 排列，保证本地专家连续归属
        # 到当前 DP 副本对应的 TP Rank。
        if (
            self.topology.ep_size != self.tp_size * p.data_parallel_size
            or tp_members != members[begin : begin + self.tp_size]
        ):
            raise ValueError(
                f"expected DP-major/TP-minor EP ordering; EP={members}, TP={tp_members}, DP rank={self.dp_rank}"
            )
        # MINIEP_TRANSPORT=auto（默认）：SM90+、单机、组内 CUDA P2P 全通且形状满足时用 Peer/TMA，
        # 否则用 NCCL。判定在 EP 组的 CPU 组上 all_gather 一次硬件信息，所有 Rank 结果相同；
        # 各层按同样顺序构造，首层之后命中缓存，不再通信。
        if self.cudagraph:
            # Peer 路径的 epoch 是 Host 标量、工作区按需分配，还不能 capture；graph 模式
            # 统一走 NCCL 固定容量路径（capture size <= 256 保证 decode 批次都是固定容量）。
            # 所有 Rank 配置相同，都会跳过 resolve_transport 的 all_gather，顺序仍一致。
            if os.environ.get("MINIEP_TRANSPORT", "auto") == "peer":
                raise ValueError("MINIEP_TRANSPORT=peer cannot be captured; use auto/nccl with graph")
            self.transport, self.transport_reason = "nccl", "CUDA graph requires fixed-capacity NCCL"
        else:
            self.transport, self.transport_reason = resolve_transport(
                get_ep_group().cpu_group,
                hidden=self.moe_config.hidden_dim,
                num_experts=self.moe_config.num_experts,
                intermediate=self.moe_config.intermediate_size_per_partition,
                dtype=cfg.model_config.dtype,
                return_dtype=self.return_dtype,
            )
        self._checked = False
        self._shared_stream = None
        if (
            self._shared_experts is not None
            and self.shared_stream_mode == "auto"
            and self.tp_size == 1
            and torch.cuda.is_available()
        ):
            # 每层一个 Stream 和一对 Event；层按顺序执行，同一层不会并发复用。
            self._shared_stream = torch.cuda.Stream()
            self._shared_input_ready = torch.cuda.Event()
            self._shared_output_ready = torch.cuda.Event()
        if self.layer_name.endswith("layers.0.mlp.experts"):
            logger.info(
                "RouteEP TP=%s DP=%s EP=%s; transport=%s (%s); deduplicated dispatch; destination FP32 owner merge",
                self.tp_size,
                p.data_parallel_size,
                self.topology.ep_size,
                self.transport,
                self.transport_reason,
            )

    def _ep_local_sizes(self, tokens: int) -> list[int]:
        """作用：返回按 DP-major/TP-minor 排列的各 EP Rank Token 行数。

        参数：
            tokens: 当前 Rank 实际持有的 Token 行数 N。

        返回：
            长度等于 EP 大小的 Python 整数列表，按 EP rank 顺序给出每个
            来源 Rank 将发送的 Token 行数。

        DP=1 时，各 TP Rank 收到复制输入，因此按 :func:`token_range`
        的确定性半开区间计算每行归属。DP=2 时优先读取 vLLM 的 Host 侧
        ``dp_metadata.num_tokens_across_dp_cpu``，再按 TP 大小做
        ``ceil(n/sp_size)`` 切分；这与 vLLM 序列并行 MoE 的分配一致。
        全程使用 Host 元数据，不读取 GPU 计数。
        """
        if not self.input_is_dp_local:
            # DP=1 时每个 Rank 收到相同的 TP 行；仍沿用确定性半开区间切分，
            # 使每行只由一个来源 Rank 负责。
            return [
                token_range(tokens, rank, self.tp_size)[1]
                - token_range(tokens, rank, self.tp_size)[0]
                for rank in range(self.tp_size)
            ]
        context = get_forward_context()
        dp_metadata = getattr(context, "dp_metadata", None)
        if dp_metadata is None:
            raise ValueError("DP=2 requires vLLM Host-side dp_metadata")
        # num_tokens_across_dp_cpu 是每个 DP 副本的 Host 计数，
        # 不含 GPU 同步；每个副本再等分给它的 TP Rank。
        counts = dp_metadata.num_tokens_across_dp_cpu
        per_tp = (counts + self.tp_size - 1) // self.tp_size
        sizes = per_tp.repeat_interleave(self.tp_size).tolist()
        if len(sizes) != self.topology.ep_size:
            raise ValueError(
                f"DP metadata has {len(sizes)} EP-local sizes, "
                f"expected {self.topology.ep_size}"
            )
        if sizes[self.topology.ep_rank] != tokens:
            raise ValueError(
                f"DP metadata local size={sizes[self.topology.ep_rank]}, "
                f"runner rows={tokens}"
            )
        return sizes

    def route_tokens(
        self,
        hidden_states: torch.Tensor,
        stages: Stages,
        router_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RouterOutput]:
        """Native Top-K -> familiar RouterOutput for this source's [n_local,H].

        DP=2 rows are source-local (SP-sharded only when TP>1). DP=1 partitions the
        replicated TP rows. Select before slicing to preserve native gate/Top-K
        arithmetic and slot tie order. In vLLM 0.29 Qwen passes hidden_states
        through the router_logits argument when the Runner owns the Gate;
        match native _forward_impl by executing that Gate before selection.
        """
        n = hidden_states.shape[0]
        start, end = (
            (0, n)
            if self.input_is_dp_local
            else token_range(n, self.tp_rank, self.tp_size)
        )
        local = hidden_states[start:end].contiguous()
        with stages.stage("ROUTER"):
            logits = router_logits
            if self.gate is not None:
                logits = (
                    self.gate(hidden_states)[0]
                    if n
                    else hidden_states.new_empty((0, self.moe_config.num_experts))
                )
            if logits is None:
                raise ValueError("routing requires a Runner Gate or explicit logits")
            if (
                logits.shape != (n, self.moe_config.num_experts)
                or logits.device != hidden_states.device
            ):
                raise ValueError(
                    f"router logits must be [{n},{self.moe_config.num_experts}] on {hidden_states.device}"
                )
            if n:
                weights, ids = self.router.select_experts(
                    hidden_states,
                    logits,
                    topk_indices_dtype=self._quant_method.topk_indices_dtype
                    or torch.int32,
                )
                weights = weights[start:end].contiguous()
                ids = ids[start:end].contiguous()
            else:
                ids = torch.empty(
                    (0, self.routed_experts.top_k),
                    device=local.device,
                    dtype=self._quant_method.topk_indices_dtype or torch.int32,
                )
                weights = torch.empty(
                    ids.shape, device=local.device, dtype=torch.float32
                )
        return local, RouterOutput(logits[start:end], ids, weights)

    def _check_layout(
        self,
        local: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
    ) -> None:
        """作用：每个 worker 只检查一次 vLLM 加载的权重和线性专家映射。

        参数：
            local: 本来源 Rank 的 Token 行 ``[n_local,H]``。
            ids: 全局专家编号 ``[n_local,K]``。
            weights: FP32 路由权重 ``[n_local,K]``。
            w13: vLLM 加载的本地 gate/up 权重 ``[L,2F,H]``。
            w2: vLLM 加载的本地 down 权重 ``[L,H,F]``。

        返回：
            无返回值。首次调用执行完整校验，后续调用直接返回。

        除了数值/形状校验，还要求 ``expert_map`` 是线性放置：当前 Rank
        的全局专家映射到 ``0..L-1``，其他专家映射到 ``-1``。这样
        RouteEP 的 ``expert_id // L`` 与 vLLM 的权重布局保持一致。
        """
        if self._checked:
            return
        if torch.cuda.is_current_stream_capturing():
            # torch.equal 需要 Host 同步；vLLM 的 profile run 是 eager，会先完成这次检查。
            raise RuntimeError("RouteEP layout check must run eagerly before CUDA graph capture")
        # 先复用 EP 前向的通用输入校验，再检查 vLLM 特有的专家映射。
        validate_inputs(local, ids, weights, w13, w2, self.topology)
        # expert_map：当前 Rank 的全局专家 -> 0..L-1，其他专家 -> -1。
        expected = torch.full(
            (self.topology.num_experts,),
            -1,
            device=local.device,
            dtype=torch.int32,
        )
        offset = self.topology.ep_rank * self.topology.num_local_experts
        # 线性放置意味着当前 Rank 拥有连续的一段全局专家编号。
        expected[offset : offset + self.topology.num_local_experts] = torch.arange(
            self.topology.num_local_experts,
            device=local.device,
            dtype=torch.int32,
        )
        if (
            not w13.is_contiguous()
            or not w2.is_contiguous()
            or not torch.equal(self.routed_experts.expert_map, expected)
        ):
            # 非连续权重会破坏 Grouped GEMM 的地址计算；非线性映射会让
            # 本地专家编号与权重栈下标不一致。
            raise ValueError(
                "requires contiguous Triton weights and linear expert placement"
            )
        self._checked = True

    def forward_ep(
        self,
        hidden_states: torch.Tensor,
        *,
        router_logits: torch.Tensor | None = None,
        stages: Stages | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Adapt framework rows/weights around the generic synchronous EP chain.

        Four-rank DP=2 uses the same local FP32 owner merge as DeepEP. The
        framework owns its SP chunk/all-gather boundary. DP=1 standalone RouteEP
        still needs TP row reassembly; it is not a native DeepEP reference in
        vLLM 0.29. Shared experts use the framework's SP path for TP2/DP2,
        or a complete local MLP for TP1/DP2; no extra TP reduction belongs here.
        """
        stages = stages or Stages()
        shared_output = None
        shared_async = False
        if self._shared_experts is not None:
            self.routed_experts._ensure_moe_quant_config_init()
            if self._shared_stream is not None and not stages.timings:
                # 生产者/消费者：主 Stream 生产 hidden_states，旁路 Stream 计算共享专家；
                # 主 Stream 在 SHARED_COMBINE 前等待 output_ready。CPU 不等待任何一方。
                # 分阶段计时时退回串行，避免 Event 区间把并行部分算进别的阶段。
                main = torch.cuda.current_stream()
                self._shared_input_ready.record(main)
                with torch.cuda.stream(self._shared_stream):
                    self._shared_stream.wait_event(self._shared_input_ready)
                    with stages.stage("SHARED_EXPERT"):
                        shared_output = self._shared_experts._layer(hidden_states)
                    self._shared_output_ready.record(self._shared_stream)
                # hidden_states 在旁路 Stream 上被读取；输出在主 Stream 上被消费。
                # record_stream 让缓存分配器在两侧工作完成前都不复用这两块显存。
                hidden_states.record_stream(self._shared_stream)
                shared_output.record_stream(main)
                shared_async = True
            else:
                with stages.stage("SHARED_EXPERT"):
                    shared_output = self._shared_experts._layer(hidden_states)
        local, routing = self.route_tokens(hidden_states, stages, router_logits)
        n = hidden_states.shape[0]
        # 两种传输都始终去重。Host 侧各来源行数：Peer 用于工作区，NCCL 用于选择
        # 固定容量 C=max(sizes)；所有 EP Rank 由同一份 Host 元数据得到相同的 C。
        ep_sizes = self._ep_local_sizes(n)
        owner_capacity = (
            fixed_owner_capacity(self.owner_fixed, ep_sizes)
            if self.transport == "nccl" else None
        )
        gate_up_weights = self.routed_experts.w13_weight
        down_weights = self.routed_experts.w2_weight
        self._check_layout(
            local,
            routing.expert_ids,
            routing.routing_weights,
            gate_up_weights,
            down_weights,
        )
        if torch.cuda.is_current_stream_capturing() and (
            self.transport != "nccl" or owner_capacity is None
        ):
            # 变长路径要交换计数并在 Host 读取，capture 时会失败或把一次性的 split 固化进图。
            raise RuntimeError(
                f"RouteEP capture needs fixed-capacity NCCL; transport={self.transport}, sizes={ep_sizes}"
            )
        if owner_capacity is not None and not getattr(MiniEPRunner, "_logged_fixed", False):
            MiniEPRunner._logged_fixed = True
            logger.info("RouteEP fixed owner A2A active: capacity=%s sizes=%s", owner_capacity, ep_sizes)
        if self.transport == "peer":
            peer_forward = get_peer_forward(self.transport)

            result, metadata = peer_forward(
                local, routing.expert_ids, routing.routing_weights,
                gate_up_weights, down_weights, self.ep_group,
                source_rows=ep_sizes, stages=stages, validate=False,
            )
        else:
            result, metadata = ep_forward(
                local,
                routing.expert_ids,
                routing.routing_weights,
                gate_up_weights,
                down_weights,
                self.ep_group,
                stages=stages,
                validate=False,
                return_dtype=self.return_dtype,
                fixed_capacity=owner_capacity,
            )
        if not self.input_is_dp_local:
            with stages.stage("TP_REASSEMBLE"):
                result = gather_tp_rows(result, n, self.tp_group)
        if shared_output is not None:
            if shared_async:
                torch.cuda.current_stream().wait_event(self._shared_output_ready)
            with stages.stage("SHARED_COMBINE"):
                result = shared_output + result
        metadata.update(
            tokens=n,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            dp_rank=self.dp_rank,
            ep_rank=self.topology.ep_rank,
            ep_size=self.topology.ep_size,
            fixed_a2a=owner_capacity is not None,
            deduplicated=True,
            transport=self.transport,
            shared_expert=shared_output is not None,
            shared_expert_stream="side" if shared_async else "main",
        )
        if stages.timings:
            metadata["_stage_events"] = stages.events
        return result, metadata

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Native framework signature; route supplied logits once, return [N,H]."""
        result, _ = self.forward_ep(hidden_states, router_logits=router_logits)
        return result
