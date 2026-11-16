"""TP2/DP1,2 默认配置，以及扩展实验的 TP1/DP2/EP2 配置。"""

import os
import json
from pathlib import Path


def model_metadata(model: str) -> dict:
    """作用：读取文本塔的模型族和层数，不加载模型权重。

    参数：
        model: 本地模型目录或 Hugging Face 模型标识。若目录中存在
            ``config.json`` 则直接读取，否则回退到 ``AutoConfig``。

    返回：
        ``{"model_family": ..., "num_hidden_layers": ...}``。

    仅支持 ``qwen3_moe`` 和 ``qwen3_5_moe``；Qwen3.5 的多模态配置会把
    文本参数放在 ``text_config`` 中，本函数会自动下钻到该字段。
    """
    path = Path(model) / "config.json"
    # 优先读取本地 config.json，避免联网或加载完整权重。
    if path.is_file():
        config = json.loads(path.read_text())
    else:
        # 远程模型只获取配置元数据，不构造模型。
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model).to_dict()
    family = config.get("model_type")
    # 本课程只验证 Qwen3/Qwen3.5 MoE 的文本推理路径。
    if family not in ("qwen3_moe", "qwen3_5_moe"):
        raise ValueError(f"requires qwen3_moe/qwen3_5_moe, got {family!r}")
    # Qwen3.5 可能把文本塔超参数嵌套在 text_config 下。
    text = config.get("text_config", config)
    layers = text.get("num_hidden_layers")
    if not isinstance(layers, int) or layers < 1:
        raise ValueError(f"num_hidden_layers must be positive, got {layers!r}")
    return dict(model_family=family, num_hidden_layers=layers)


def configure(backend: str) -> None:
    """作用：设置 RouteEP 插件与 vLLM 启动所需的环境变量。

    参数：
        backend: ``"native"`` 使用 vLLM 原生 MoE；``"miniep"`` 安装本课程
            的 Runner。必须在构造 ``LLM`` 或启动 worker 之前调用。

    返回：
        无返回值。环境变量会由后续启动的 worker 子进程继承。
    """
    if backend not in ("native", "miniep"):
        raise ValueError(f"backend must be native/miniep, got {backend!r}")
    # VLLM_PLUGINS 控制加载哪个插件；MINIEP_BACKEND 控制插件是否替换 Runner。
    # 必须在构建 LLM / 启动 worker 前设置，使子进程继承相同配置。
    os.environ["MINIEP_BACKEND"] = backend
    os.environ["VLLM_PLUGINS"] = "routeep"
    # spawn 保证每个 worker 使用干净的 Python 解释器，避免插件状态泄漏。
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # 课程要求保留 vLLM 的 padding 行语义，固定容量 A2A 依赖该约定。
    os.environ["VLLM_MOE_SKIP_PADDING"] = "0"
    # 每次启动都清理旧的报告目录，防止误读上一次实验数据。
    os.environ.pop("MINIEP_REPORT_DIR", None)


def engine_options(
    model: str, dp_size: int = 1, *, external: bool = False, tp_size: int = 2,
    cudagraph_capture_sizes: list[int] | None = None,
) -> dict:
    """作用：生成本课程验证用的 vLLM ``LLM`` 构造参数。

    参数：
        model: 本地模型目录或 Hugging Face 模型标识。
        dp_size: 数据并行副本数，只支持 1 或 2。
        tp_size: 默认 2；H100 EP2 扩展实验使用 TP=1、DP=2。
        external: 是否由外部 ``torchrun`` 启动。离线 DP=2 必须为 True，
            进程数为 TP×DP。
        cudagraph_capture_sizes: None（默认）为 eager；给出列表时 decode 批次用 FULL CUDA Graph
            （``FULL_DECODE_ONLY``），含 prefill 的批次仍 eager。每个值是一个 DP 副本的 Token 数，
            必须在 1..FIXED_OWNER_MAX_TOKENS 内，保证 capture 时走无 Host 同步的 NCCL 固定容量路径。
            编译模式固定为 NONE（不做 torch.compile），因此 vLLM 不会因 HT 后端名而关闭 graph。

    返回：
        可直接传给 vLLM ``LLM(...)`` 的关键字参数字典。

    配置默认 Attention TP=2，固定 BF16、eager、同步调度和 Triton MoE 后端，
    以便通信顺序和权重布局在教学中保持可观察。
    """
    if tp_size not in (1, 2) or (tp_size == 1 and dp_size != 2):
        raise ValueError(f"requires TP=2/DP=1,2 or TP=1/DP=2, got TP={tp_size}/DP={dp_size}")
    if dp_size not in (1, 2):
        raise ValueError(f"DP must be 1 or 2, got {dp_size}")
    if dp_size > 1 and not external:
        # 单进程无法创建 DP=2 所需的完整 TP×DP 进程拓扑。
        raise ValueError(
            "offline DP=2 requires torchrun with external launcher; use test/deepep_alignment.py"
        )
    # Attention 使用 TP=1/2；启用 EP 后，MoE 专家分布在 TP×DP 个 rank 上。
    # eager + 同步调度保持通信顺序清晰；离线 DP=2 由 torchrun 启动 TP×2 个进程。
    metadata = model_metadata(model)
    # 插件在 worker 内读取该变量，决定替换 Qwen3 还是 Qwen3.5 的工厂。
    os.environ["MINIEP_MODEL_FAMILY"] = metadata["model_family"]
    options = dict(
        model=model,
        dtype="bfloat16",
        tensor_parallel_size=tp_size,
        data_parallel_size=dp_size,
        enable_expert_parallel=True,
        enforce_eager=True,
        async_scheduling=False,
        seed=13,
        max_model_len=4096,
        max_num_seqs=8,
        max_num_batched_tokens=16384,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.80,
        # RouteEP 直接消费 vLLM 的 Triton 规范权重布局。
        kernel_config={"moe_backend": "triton"},
        all2all_backend="deepep_high_throughput",
        # external_launcher 由 torchrun 提供进程和进程组；mp 适合单机 DP=1。
        distributed_executor_backend="external_launcher" if external else "mp",
    )
    if cudagraph_capture_sizes is not None:
        from mini_ep.distributed.all_to_all import FIXED_OWNER_MAX_TOKENS

        sizes = sorted(set(cudagraph_capture_sizes))
        if not sizes or sizes[0] < 1 or sizes[-1] > FIXED_OWNER_MAX_TOKENS:
            raise ValueError(
                f"capture sizes must be within 1..{FIXED_OWNER_MAX_TOKENS}, got {cudagraph_capture_sizes}"
            )
        # mode=0 即 CompilationMode.NONE：模型不经 torch.compile，整模型由 CUDAGraphWrapper
        # 在 decode 批次上 capture/replay；set_splitting_ops_for_v1 在非 VLLM_COMPILE 下提前返回，
        # 不会走“HT 后端 + DP>1 关闭 graph”的规则。
        options.update(
            enforce_eager=False,
            compilation_config={
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": sizes,
            },
        )
    if metadata["model_family"] == "qwen3_5_moe":
        # 本扩展只验证文本推理；视觉/视频输入关闭，循环状态仍由 vLLM 管理。
        options["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    return options
