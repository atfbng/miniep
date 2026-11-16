"""可选验证插件，刻意放在生产包之外。

只有在 MINIEP_REPORT_DIR 和 MINIEP_VALIDATION_MODE 都设置时才安装，
因此正常服务启动不会加载任何验证钩子。
"""

import json
import os
from functools import wraps
from pathlib import Path


def register() -> None:
    if not os.environ.get("MINIEP_REPORT_DIR") or not os.environ.get(
        "MINIEP_VALIDATION_MODE"
    ):
        return
    from mini_ep.integration.plugin import prepare_runtime, install_runner

    model = prepare_runtime()
    if getattr(model, "_routeep_validation_installed", False):
        return
    mode = os.environ.get("MINIEP_VALIDATION_MODE", "native")
    if mode == "deepep_native":
        from validation.deepep_backend import install_native_observer

        install_native_observer()
    elif mode == "deepep_exact":
        from validation.deepep_runner import DeepEPExactRunner

        install_runner(DeepEPExactRunner)
    else:
        raise ValueError(
            f"exact validation mode must be deepep_native/deepep_exact, got {mode!r}"
        )
    if os.environ.get("MINIEP_MODEL_FAMILY", "qwen3_moe") == "qwen3_5_moe":
        from vllm.model_executor.models.qwen3_5 import (
            Qwen3_5MoeForConditionalGeneration,
        )

        install_logits_capture(Qwen3_5MoeForConditionalGeneration)
    else:
        install_logits_capture(model.Qwen3MoeForCausalLM)
    model._routeep_validation_installed = True


_CONTROL = None
_RECORDS = {}


def set_control(state: dict | None) -> None:
    """ExternalLauncher 或进程内测试驱动写入一个不可变的 case 状态。

    文件仍是可复现的实验日志；直接传状态可避免每个 MoE 层都读共享文件系统。
    其他启动方式继续读文件。case 结束和抛错前会 flush 已缓冲记录。
    """
    global _CONTROL
    _CONTROL = state


def read_control() -> dict:
    if _CONTROL is not None:
        return _CONTROL
    root = os.environ.get("MINIEP_REPORT_DIR")
    if not root:
        return {}
    path = Path(root) / "control.json"
    return json.loads(path.read_text()) if path.exists() else {}


def flush_records() -> None:
    root = os.environ.get("MINIEP_REPORT_DIR")
    if not root:
        return
    for name, records in _RECORDS.items():
        with (Path(root) / name).open("a") as stream:
            stream.writelines(
                json.dumps(record, allow_nan=False) + "\n" for record in records
            )
    _RECORDS.clear()


def append_record(name: str, record: dict) -> None:
    root = os.environ.get("MINIEP_REPORT_DIR")
    if not root:
        return
    if _CONTROL is not None:
        _RECORDS.setdefault(name, []).append(record.copy())
    else:
        with (Path(root) / name).open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")


def install_logits_capture(model_class: type) -> None:
    """测试专用的全词表捕获，并在采样前执行 teacher forcing。

    基线 Tensor 只放在磁盘，不放进提交的模型。teacher forcing 时先计算指标，
    再把 logits 替换为基线的 argmax；返回的 masked logits 只用于让后续前缀一致。
    没有 control 文件的 warmup 会被忽略。所有 logits 非 None 的 Rank 遵守同一
    规则，各自写自己的文件和计数。
    """
    import torch
    import torch.distributed as dist

    original = model_class.compute_logits

    @wraps(original)
    def compute_logits(self, *args, **kwargs):
        logits = original(self, *args, **kwargs)
        if os.environ.get("MINIEP_VALIDATION_MODE") == "deepep_native":
            from validation.deepep_backend import flush_native_observer

            flush_native_observer()
        control = read_control()
        if (
            logits is None
            or not control
            or control.get("skip_logits", False)
            or control["phase"] not in ("native", "teacher", "native_check")
        ):
            return logits
        key = control["key"]
        if getattr(self, "_miniep_capture_key", None) != key:
            self._miniep_capture_key, self._miniep_step = key, 0
        step = self._miniep_step
        self._miniep_step += 1
        if step >= control.get("max_steps", float("inf")):
            return logits
        rank = dist.get_rank()
        root = Path(control["baseline_dir"])
        path = root / f"{control['batch_key']}-rank{rank}-step{step:03d}.pt"
        if control["phase"] == "native":
            root.mkdir(parents=True, exist_ok=True)
            torch.save(logits.detach().cpu(), path)
            if control.get("acceptance") == "exact":
                finite = bool(torch.isfinite(logits).all())
                append_record(
                    f"native-logits-rank{rank}.jsonl",
                    dict(key=key, step=step, rows=logits.shape[0], finite=finite),
                )
                if not finite:
                    raise RuntimeError(
                        f"nonfinite native reference rank={rank} key={key} step={step}"
                    )
        elif control["phase"] in ("teacher", "native_check"):
            baseline = torch.load(path, map_location=logits.device, weights_only=True)
            if control.get("acceptance") == "exact":
                from validation.exact import exact_logits_metrics

                stats = exact_logits_metrics(logits, baseline)
                name = (
                    "native-check" if control["phase"] == "native_check" else "logits"
                )
                append_record(
                    f"{name}-rank{rank}.jsonl", dict(key=key, step=step, **stats)
                )
                if not stats["exact_equal"]:
                    failure = Path(os.environ["MINIEP_REPORT_DIR"]) / "first-failure"
                    failure.mkdir(exist_ok=True)
                    (failure / "summary.json").write_text(
                        json.dumps(
                            dict(kind=name, rank=rank, key=key, step=step, **stats),
                            indent=2,
                            allow_nan=False,
                        )
                    )
                    torch.save(
                        dict(
                            actual=logits.detach().cpu(),
                            expected=baseline.detach().cpu(),
                        ),
                        failure / "logits.pt",
                    )
                    flush_records()
                    raise RuntimeError(
                        f"exact {name} mismatch rank={rank} key={key} step={step}; {failure}"
                    )
                if control["phase"] == "native_check":
                    return logits
                forced = baseline.argmax(-1, keepdim=True)
                return torch.full_like(logits, -float("inf")).scatter_(-1, forced, 0.0)
            raise RuntimeError("full logits capture requires exact acceptance")
        return logits

    model_class.compute_logits = compute_logits
