"""仅供测试和 Benchmark 使用的后端选择器。

reference 走独立 PyTorch 实现，triton 走 mini_ep 生产实现；生产代码不会导入
本文件，避免测试依赖进入部署路径。
"""


def ep_forward(
    *args,
    backend="reference",
    expert_backend="grouped",
    **kwargs,
):
    if backend == "reference":
        from reference.moe import ep_forward_reference

        return ep_forward_reference(*args, **kwargs)
    if backend == "triton":
        from mini_ep.model.moe import ep_forward as core_forward

        return core_forward(
            *args,
            expert_backend=expert_backend,
            **kwargs,
        )
    raise ValueError(f"unknown backend {backend!r}")
