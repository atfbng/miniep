"""Failure-only probes. All ranks replay a frozen layer before terminating.

No native/mini kernel math is replaced. Buffer clones are made only during the
failure replay, before workspace reuse. Native down pre-weight FP32 registers
are not observable; the offline PyTorch replay labels them as a reference.
"""

from contextlib import contextmanager
from functools import wraps
import inspect
import torch


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, (list, tuple)):
        return [clone_tree(x) for x in value]
    if isinstance(value, dict):
        return {str(k): clone_tree(v) for k, v in value.items()}
    return (
        value
        if isinstance(value, (str, int, float, bool, type(None)))
        else repr(type(value))
    )


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [cpu_tree(x) for x in value]
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    return value


@contextmanager
def capture_stages(runner):
    import mini_ep.model.moe as mini
    import vllm.model_executor.layers.fused_moe.experts.triton_moe as native

    data = {}
    originals = []

    def patch(obj, name, wrapper):
        original = getattr(obj, name)
        originals.append((obj, name, original))
        setattr(obj, name, wrapper(original))

    def expert_wrapper(original):
        signature = inspect.signature(original)

        @wraps(original)
        def call(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            data["mini_expert_inputs"] = clone_tree(dict(bound.arguments))
            result = original(*args, **kwargs)
            data["mini_weighted_branches_received_order"] = result.detach().clone()
            return result

        return call

    def combine_wrapper(original):
        signature = inspect.signature(original)

        @wraps(original)
        def call(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            data["mini_returned_source_order"] = clone_tree(dict(bound.arguments))
            result = original(*args, **kwargs)
            from reference.dispatch import owner_partials_reference

            data["mini_owner_partials_replay"] = (
                owner_partials_reference(**bound.arguments).detach().clone()
            )
            return result

        return call

    def native_expert_wrapper(original):
        signature = inspect.signature(original)

        @wraps(original)
        def call(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            data["native_expert_inputs"] = clone_tree(
                {
                    k: v
                    for k, v in bound.arguments.items()
                    if k
                    in (
                        "hidden_states",
                        "topk_weights",
                        "topk_ids",
                        "expert_map",
                        "global_num_experts",
                    )
                }
            )
            result = original(*args, **kwargs)
            data["native_owner_partials_received_order"] = (
                bound.arguments["output"].detach().clone()
            )
            return result

        return call

    def native_linear_wrapper(original):
        signature = inspect.signature(original)

        @wraps(original)
        def call(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            result = original(*args, **kwargs)
            label = (
                "down_weighted" if bound.arguments["mul_routed_weight"] else "gate_up"
            )
            data["native_" + label] = bound.arguments["C"].detach().clone()
            return result

        return call

    try:
        patch(mini, "compute_grouped_local_experts", expert_wrapper)
        patch(mini, "combine_triton", combine_wrapper)
        patch(native.TritonExperts, "apply", native_expert_wrapper)
        patch(native, "invoke_fused_moe_triton_kernel", native_linear_wrapper)
        yield data
        pf = runner._quant_method.moe_kernel.prepare_finalize
        data["native_dispatch_handles"] = clone_tree(getattr(pf, "handles", None))
    finally:
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)
