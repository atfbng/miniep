"""精简的 vLLM 插件入口：把 Qwen MoE 层的 Runner 换成 RouteEP。

加载顺序：
    1. vLLM 调用 register()；
    2. native 模式直接返回；miniep 模式延迟导入并安装 Runner；
    3. 后续构造 Qwen3/Qwen3.5 MoE 层时，用注入的工厂选择 MiniEPRunner。

本文件不包含测试钩子或 Reference 依赖，正常推理不会加载验证代码。
"""

import importlib.metadata
import os
from functools import wraps


def prepare_runtime():
    """作用：锁定本课程依赖的私有工厂 API 和 BF16 原生算术配置。

    返回：
        对应的 vLLM 模型模块（``qwen3_moe`` 或 ``qwen3_next``），后续
        通过替换其 ``FusedMoEFactory`` 注入 RouteEP Runner。

    本函数校验 vLLM 版本、padding 环境和模型族，并关闭 BF16 归约的
    硬件加速路径，保证跨实验的数值行为一致。若版本或配置不符，直接
    抛错而不是在内部 API 变化后静默接错。
    """
    # 工厂和 Runner 属于 vLLM 内部 API；固定版本，避免接口变化后静默接错。
    version = importlib.metadata.version("vllm")
    if version.split("+")[0] != "0.29.0":
        raise RuntimeError(f"RouteEP requires vllm==0.29.0, got {version}")
    if os.environ.get("VLLM_MOE_SKIP_PADDING") != "0":
        raise ValueError("RouteEP requires VLLM_MOE_SKIP_PADDING=0")
    import torch

    # 禁止 reduced-precision BF16 归约，使原生 collective 的舍入路径可复现。
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    family = os.environ.get("MINIEP_MODEL_FAMILY", "qwen3_moe")
    if family == "qwen3_moe":
        from vllm.model_executor.models import qwen3_moe

        return qwen3_moe
    if family == "qwen3_5_moe":
        # Qwen3.5 通过 qwen3_next 模块构造 routed/shared MoE。
        from vllm.model_executor.models import qwen3_next

        return qwen3_next
    raise ValueError(f"unsupported MINIEP_MODEL_FAMILY={family!r}")


def install_runner(runner_cls: type) -> None:
    """作用：通过 Qwen3 现有工厂注入一个 MoE Runner，不接管模型加载。

    参数：
        runner_cls: 要注入的 Runner 类，本课程使用 ``MiniEPRunner``。

    返回：
        无返回值。函数把 Qwen3 模块的 ``FusedMoEFactory`` 替换为只负责
        填入 ``runner_cls`` 的包装函数，并以 ``_miniep_runner_cls`` 防止
        同一 worker 中重复选择后端。
    """
    model = prepare_runtime()
    # 同一进程重复安装同一 Runner 是幂等的；安装不同 Runner 则必须失败。
    if getattr(model, "_miniep_runner_cls", None) is runner_cls:
        return
    if getattr(model, "_miniep_runner_cls", None) is not None:
        raise RuntimeError("choose a backend in a fresh worker process")
    # 保存原工厂：包装函数仍调用它创建层，权重加载继续由 vLLM 负责。
    original = model.FusedMoEFactory

    # 此处只定义闭包；Qwen3 后续创建 MoE 层时才执行 factory。
    # 闭包保留 original 和 runner_cls，因此不会递归调用替换后的入口。
    @wraps(original)
    def factory(*args, **kwargs):
        """作用：在原 Qwen3 MoE 工厂基础上注入 RouteEP Runner。

        参数：
            *args: 原工厂的位置参数，原样转发。
            **kwargs: 原工厂的关键字参数；会额外写入 ``runner_cls``。

        返回：
            原工厂创建的 MoE 层对象。
        """
        # 不改变原工厂的参数语义，只额外指定 Runner 实现。
        kwargs["runner_cls"] = runner_cls
        return original(*args, **kwargs)

    # 只替换 Qwen3 模块中的工厂入口，不修改 vLLM 安装目录的源码。
    model.FusedMoEFactory = factory
    model._miniep_runner_cls = runner_cls


def register() -> None:
    """作用：作为 vLLM 插件入口，根据 ``MINIEP_BACKEND`` 安装或跳过 RouteEP。

    该函数由 ``pyproject.toml`` 注册的插件机制调用一次。``native`` 模式
    直接返回；``miniep`` 模式延迟导入 Runner，避免原生路径加载 Triton
    或 RouteEP 模块。
    """
    # pyproject.toml 注册的插件入口：vLLM 加载插件时调用一次安装逻辑。
    # native 直接返回；miniep 经 install_runner 为后续模型构建指定 Runner。
    mode = os.environ.get("MINIEP_BACKEND", "native")
    if mode == "native":
        return
    if mode != "miniep":
        raise ValueError(f"production backend must be native/miniep, got {mode!r}")
    from mini_ep.integration.runner import MiniEPRunner

    install_runner(MiniEPRunner)
