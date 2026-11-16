"""选择 RouteEP 的跨卡通信实现；去重分发始终开启，没有关闭开关。

MINIEP_TRANSPORT 取值：
    auto（默认）：硬件和模型形状都满足 Peer/TMA 条件时用 peer，否则用 nccl；
    nccl：强制 NCCL variable-split All-to-All，适配绝大多数 GPU；
    peer：强制 Peer/TMA，不满足条件时直接报错，不静默退回。
历史值 peer_dedup 仍可读取，等价于 peer。

auto 的判定需要 EP 组内一次 all_gather_object，所有 Rank 必须同时调用
resolve_transport，并得到同一个结果；结果按进程组缓存，只判定一次。
"""
import importlib
import os
import socket
from typing import Callable

import torch

TRANSPORTS = ("auto", "nccl", "peer")


def get_transport() -> str:
    """读取 MINIEP_TRANSPORT 请求值（auto/nccl/peer），不做硬件判定。

    历史值 peer_dedup 兼容映射为 peer，不再代表单独算法；不提供去重开关。"""
    name = os.environ.get("MINIEP_TRANSPORT", "auto")
    if name == "peer_dedup":
        return "peer"
    if name not in TRANSPORTS:
        raise ValueError(f"MINIEP_TRANSPORT must be auto/nccl/peer, got {name!r}")
    return name


def peer_shape_problems(
    *, hidden: int, num_experts: int, intermediate: int,
    dtype: torch.dtype, return_dtype: torch.dtype | None,
) -> list[str]:
    """返回模型形状/精度不满足 Peer/TMA 路径的原因；空列表表示满足。

    条件来自 distributed/peer.py 与 peer_dedup.py 的入口检查：BF16 激活与回传、
    H 能被 8 整除（TMA 对齐）、E 为 2 的幂（布局 Kernel）、F 能被 128 整除（TMA Up tile）。
    """
    problems = []
    if dtype != torch.bfloat16:
        problems.append(f"activation dtype {dtype} is not BF16")
    if return_dtype not in (None, torch.bfloat16):
        problems.append(f"return dtype {return_dtype} is not BF16")
    if hidden % 8:
        problems.append(f"H={hidden} not divisible by 8")
    if num_experts < 1 or num_experts & (num_experts - 1):
        problems.append(f"E={num_experts} is not a power of two")
    if intermediate % 128:
        problems.append(f"F={intermediate} not divisible by 128")
    return problems


def _local_probe() -> dict:
    """本 Rank 的硬件信息：主机名、当前 CUDA 设备、算力、可见设备的 Peer 访问表。"""
    probe = dict(host=socket.gethostname(), device=None, capability=None,
                 peer=[], symmetric_memory=False)
    if not torch.cuda.is_available():
        return probe
    device = torch.cuda.current_device()
    probe["device"] = device
    probe["capability"] = list(torch.cuda.get_device_capability(device))
    probe["peer"] = [
        j == device or torch.cuda.can_device_access_peer(device, j)
        for j in range(torch.cuda.device_count())
    ]
    # 用 importlib 而不是 `import torch.distributed...`：后者会把 torch 绑定成本函数的
    # 局部变量，导致上面的 torch.cuda 引用报 UnboundLocalError。
    try:
        importlib.import_module("torch.distributed._symmetric_memory")
        probe["symmetric_memory"] = True
    except ImportError:
        pass
    return probe


def decide_transport(probes: list[dict], shape_problems: list[str]) -> tuple[str, str]:
    """根据全组的硬件探测结果和形状检查给出 (transport, 原因)。纯函数，便于单元测试。

    选 peer 的条件：所有 Rank 同一主机、都有 CUDA 设备且算力 >= SM90、
    symmetric memory 可用、每个 Rank 都能 Peer 访问组内其他 Rank 的设备，且形状满足。
    """
    if shape_problems:
        return "nccl", "model shape: " + "; ".join(shape_problems)
    if len({q["host"] for q in probes}) != 1:
        return "nccl", "EP group spans multiple hosts"
    if any(q["device"] is None for q in probes):
        return "nccl", "CUDA unavailable on some rank"
    if any(q["capability"][0] < 9 for q in probes):
        caps = sorted({tuple(q["capability"]) for q in probes})
        return "nccl", f"compute capability {caps} below SM90"
    if not all(q["symmetric_memory"] for q in probes):
        return "nccl", "torch symmetric memory unavailable"
    devices = [q["device"] for q in probes]
    if len(set(devices)) != len(devices):
        return "nccl", f"ranks share CUDA device indices {devices}"
    for q in probes:
        for other in devices:
            if other >= len(q["peer"]) or not q["peer"][other]:
                return "nccl", f"device {q['device']} cannot peer-access device {other}"
    return "peer", "SM90+ single host with full CUDA peer access"


_RESOLVED: dict = {}


def resolve_transport(
    group, *, hidden: int, num_experts: int, intermediate: int,
    dtype: torch.dtype, return_dtype: torch.dtype | None,
) -> tuple[str, str]:
    """把请求值解析成实际 transport，返回 (transport, 原因)。

    auto 时在 EP 组内 all_gather_object 硬件探测结果；所有 Rank 必须同时调用。
    显式 peer 但条件不满足时抛错。结果按 (group, 请求值, 形状) 缓存。
    """
    import torch.distributed as dist

    requested = get_transport()
    key = (id(group), requested, hidden, num_experts, intermediate, dtype, return_dtype)
    if key in _RESOLVED:
        return _RESOLVED[key]
    problems = peer_shape_problems(hidden=hidden, num_experts=num_experts,
                                   intermediate=intermediate, dtype=dtype,
                                   return_dtype=return_dtype)
    if requested == "nccl":
        result = ("nccl", "requested by MINIEP_TRANSPORT=nccl")
    else:
        probes = [None] * dist.get_world_size(group)
        dist.all_gather_object(probes, _local_probe(), group=group)
        name, reason = decide_transport(probes, problems)
        if requested == "peer" and name != "peer":
            raise ValueError(f"MINIEP_TRANSPORT=peer is unsupported here: {reason}")
        result = (name, reason if requested == "auto" else "requested by MINIEP_TRANSPORT=peer")
    _RESOLVED[key] = result
    return result


def get_peer_forward(transport: str) -> Callable | None:
    """NCCL 不导入 Peer 模块；选择 peer 时始终返回去重实现。"""
    if transport == "nccl":
        return None
    if transport in ("peer", "peer_dedup"):
        from mini_ep.distributed.peer_dedup import deduplicated_peer_forward

        return deduplicated_peer_forward
    raise ValueError(f"unknown transport {transport!r}")
