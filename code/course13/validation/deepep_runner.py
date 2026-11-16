"""Exact native/mini shadow execution; validation only, no production changes."""

import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from mini_ep.integration.runner import MiniEPRunner
from validation.deepep_backend import verify_backend
from validation.exact import exact_tensor_metrics, exact_tensor_flag
from validation.plugin import read_control, append_record
from validation.runner import record_call


class DeepEPExactRunner(MiniEPRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.return_dtype is not None:
            raise RuntimeError(
                "exact default validation requires auto/expert/auto policy"
            )

    def _capture_route(self, fn):
        original = self.router.select_experts
        captured = []

        def select(*args, **kwargs):
            result = original(*args, **kwargs)
            captured.append(tuple(x.detach().clone() for x in result))
            return result

        self.router.select_experts = select
        try:
            return fn(), captured
        finally:
            self.router.select_experts = original

    def forward(
        self, hidden_states, router_logits, input_ids=None, shared_experts_input=None
    ):
        control = read_control()
        mode = control.get("matrix_mode", "native")
        native = lambda: MoERunner.forward(
            self,
            hidden_states.clone(),
            router_logits.clone(),
            input_ids,
            shared_experts_input,
        )
        if mode == "native":
            result = native()
            verify_backend(self)
            return result
        if mode != "compare":
            output, record = self.forward_ep(hidden_states, router_logits=router_logits)
            record["logical_requests"] = control["logical_requests"]
            record["framework_rows"] = hidden_states.shape[0]
            record["phase"] = control["phase"]
            record_call(self, record)
            return output
        expected, native_routes = self._capture_route(lambda: native().clone())
        verify_backend(self)
        (actual, record), mini_routes = self._capture_route(
            lambda: self.forward_ep(hidden_states, router_logits=router_logits)
        )
        output_ok = exact_tensor_flag(actual, expected)
        route_ok = torch.ones((), device=hidden_states.device, dtype=torch.bool)
        if len(native_routes) != len(mini_routes) or (
            hidden_states.shape[0] and len(native_routes) != 1
        ):
            route_ok.fill_(False)
        for a, b in zip(native_routes, mini_routes):
            if len(a) != len(b):
                route_ok.fill_(False)
            for x, y in zip(a, b):
                route_ok = route_ok & exact_tensor_flag(x, y)
        bad = (~(output_ok & route_ok)).to(torch.int32).reshape(1)
        # One host decision per layer; every EP rank decides together whether
        # to replay this same frozen layer before the next model collective.
        dist.all_reduce(bad, op=dist.ReduceOp.MAX, group=self.ep_group)
        failed = bool(bad.item())
        stats = (
            exact_tensor_metrics(actual, expected)
            if failed
            else dict(
                exact_equal=True,
                finite=True,
                contract_match=True,
                mismatch_elements=0,
                max_abs=0.0,
                first_difference=None,
            )
        )
        routes_equal = bool(route_ok.item()) if failed else True
        record.update(
            stats,
            router_exact=routes_equal,
            logical_requests=control["logical_requests"],
            framework_rows=hidden_states.shape[0],
            phase=control["phase"],
        )
        record_call(self, record)
        if failed:
            from validation.plugin import flush_records

            flush_records()
            root = Path(os.environ["MINIEP_REPORT_DIR"]) / "first-failure"
            root.mkdir(exist_ok=True)
            summary = dict(
                layer=self.layer_name,
                rank=dist.get_rank(),
                key=control["key"],
                tensor=stats,
                router_exact=routes_equal,
                ep_size=self.topology.ep_size,
                local_experts=self.topology.num_local_experts,
                ep_rank=self.topology.ep_rank,
            )
            (root / "summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=False)
            )
            from validation.deepep_probe import capture_stages, cpu_tree

            with capture_stages(self) as stages:
                replay_native = native().clone()
                replay_mini, _ = self.forward_ep(
                    hidden_states, router_logits=router_logits
                )
                stages.update(
                    input=hidden_states,
                    router_logits=router_logits,
                    source_native_routes=native_routes,
                    source_mini_routes=mini_routes,
                    native_output=replay_native,
                    mini_output=replay_mini,
                    w13=self.routed_experts.w13_weight,
                    w2=self.routed_experts.w2_weight,
                )
            torch.save(cpu_tree(stages), root / "tensors.pt")
            # Drain diagnostic writes on every rank before torchrun can kill
            # peers after the first exception. This is failure-only coordination.
            ack = torch.ones(1, device=hidden_states.device)
            dist.all_reduce(ack, group=self.ep_group)
            ack.item()
            raise RuntimeError(
                f"exact DeepEP mismatch rank={dist.get_rank()} layer={self.layer_name}; {root}"
            )
        return expected
