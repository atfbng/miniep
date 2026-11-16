"""TP 局部 Token 归属、跨 DP 专家交换和 TP 局部重组。"""

import os

import torch
import torch.distributed as dist
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.distributed import get_ep_group, get_tp_group, get_dp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from mini_ep.distributed.topology import ExpertParallelTopology
from mini_ep.distributed.all_to_all import fixed_rows_per_rank
from mini_ep.model.moe import ep_forward
from mini_ep.model.contracts import validate_inputs
from mini_ep.model.router import RouterOutput
from mini_ep.profiler.stages import Stages
from mini_ep.integration.input_layout import token_range, gather_tp_rows

logger = init_logger("vllm.mini_ep")


class MiniEPRunner(MoERunner):
    """作用：用 Mini-EP 执行 Qwen MoE 层，其余模型部分仍由 vLLM 管理。

    输入归属随并行模式变化：

    - DP=1：两个 TP Rank 收到相同的复制行，使用 :func:`token_range`
      把每行确定性地分配给唯一来源 Rank；
    - DP=2：Qwen 已经把 DP 行切分到 TP Rank，本 Runner 直接消费本地行，
      不再二次切分或额外 gather。

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
        p = cfg.parallel_config
        # 课程实现依赖固定拓扑和 DeepEP 高吞吐配置；
        # 任何不匹配都在构造阶段失败，避免运行到集合通信才发现。
        required = dict(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            enable_expert_parallel=True,
            enable_eplb=False,
            enable_dbo=False,
            expert_placement_strategy="linear",
            all2all_backend="deepep_high_throughput",
        )
        for name, expected in required.items():
            actual = getattr(p, name)
            if actual != expected:
                raise ValueError(
                    f"course13 requires {name}={expected!r}, got {actual!r}"
                )
        if p.data_parallel_size not in (1, 2):
            raise ValueError(f"course13 requires DP=1/2, got {p.data_parallel_size}")
        # eager + 同步调度保证每个阶段按代码顺序执行，便于观察通信。
        if (
            not cfg.model_config.enforce_eager
            or cfg.quant_config is not None
            or cfg.speculative_config is not None
        ):
            raise ValueError("requires eager, nonquantized, nonspeculative execution")
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
        if self._shared_experts is not None and not self.input_is_tp_sharded:
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
        self._checked = False
        if self.layer_name.endswith("layers.0.mlp.experts"):
            logger.info(
                "Mini-EP TP=%s DP=%s EP=%s; weighted input-dtype A2A return; local FP32 owner merge",
                self.tp_size,
                p.data_parallel_size,
                self.topology.ep_size,
            )
        fixed_mode = os.environ.get("MINIEP_FIXED_A2A", "auto")
        if fixed_mode not in ("auto", "on", "off"):
            raise ValueError(
                f"MINIEP_FIXED_A2A must be auto/on/off, got {fixed_mode!r}"
            )
        self.fixed_a2a_mode = fixed_mode
        # 小批量固定容量打包可选择 Triton 或 PyTorch 参考实现。
        self.fixed_pack_backend = os.environ.get("MINIEP_FIXED_PACK", "triton")
        if self.fixed_pack_backend not in ("torch", "triton"):
            raise ValueError(
                f"MINIEP_FIXED_PACK must be torch/triton, got {self.fixed_pack_backend!r}"
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
        if not self.input_is_tp_sharded:
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

        DP=2/SP arrives already sharded. DP=1 deterministically partitions the
        replicated TP rows. Select before slicing to preserve native gate/Top-K
        arithmetic and slot tie order. In vLLM 0.29 Qwen passes hidden_states
        through the router_logits argument when the Runner owns the Gate;
        match native _forward_impl by executing that Gate before selection.
        """
        n = hidden_states.shape[0]
        start, end = (
            (0, n)
            if self.input_is_tp_sharded
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
        Mini-EP 的 ``expert_id // L`` 与 vLLM 的权重布局保持一致。
        """
        if self._checked:
            return
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
        framework owns its SP chunk/all-gather boundary. DP=1 standalone Mini-EP
        still needs TP row reassembly; it is not a native DeepEP reference in
        vLLM 0.29. Shared experts are supported only on the framework's SP path,
        as checked at initialization; no extra TP reduction belongs here.
        """
        stages = stages or Stages()
        shared_output = None
        if self._shared_experts is not None:
            self.routed_experts._ensure_moe_quant_config_init()
            with stages.stage("SHARED_EXPERT"):
                shared_output = self._shared_experts._layer(hidden_states)
        local, routing = self.route_tokens(hidden_states, stages, router_logits)
        n = hidden_states.shape[0]
        ep_sizes = self._ep_local_sizes(n) if self.fixed_a2a_mode != "off" else None
        capacity = (
            fixed_rows_per_rank(
                self.fixed_a2a_mode, ep_sizes, routing.expert_ids.shape[1]
            )
            if ep_sizes is not None
            else None
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
        result, metadata = ep_forward(
            local,
            routing.expert_ids,
            routing.routing_weights,
            gate_up_weights,
            down_weights,
            self.ep_group,
            stages=stages,
            validate=False,
            fixed_rows_per_rank=capacity,
            source_rows=ep_sizes,
            fixed_pack_backend=self.fixed_pack_backend,
            return_dtype=self.return_dtype,
        )
        if not self.input_is_tp_sharded:
            with stages.stage("TP_REASSEMBLE"):
                result = gather_tp_rows(result, n, self.tp_group)
        if shared_output is not None:
            with stages.stage("SHARED_COMBINE"):
                result = shared_output + result
        metadata.update(
            tokens=n,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            dp_rank=self.dp_rank,
            ep_rank=self.topology.ep_rank,
            ep_size=self.topology.ep_size,
            fixed_a2a=capacity is not None,
            shared_expert=shared_output is not None,
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
