"""Test/benchmark-only backend switch."""


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
