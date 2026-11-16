"""单个 Expert 的稠密投影与无偏置 MLP Reference。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def validate_gemm_inputs(x: torch.Tensor, weight: torch.Tensor) -> tuple[int, int, int]:
    """检查连续行主序 X[M,K]、W[N,K]，返回 M、N、K。"""
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError(f"需要 X[M,K]、W[N,K]，实际为 {tuple(x.shape)}、{tuple(weight.shape)}")
    m, k = x.shape
    n, weight_k = weight.shape
    if k <= 0 or n <= 0 or k != weight_k:
        raise ValueError(f"需要 N、K>0 且输入宽度一致，实际 X={tuple(x.shape)}, W={tuple(weight.shape)}")
    if x.device != weight.device:
        raise ValueError(f"需要相同 device，实际为 {x.device}、{weight.device}")
    if x.dtype != weight.dtype or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"需要相同 FP16/BF16/FP32 dtype，实际为 {x.dtype}、{weight.dtype}")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("X 与 W 需要连续行主序布局")
    return m, n, k


def validate_mlp_inputs(x: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> None:
    """X[M,H]、up[F,H]、down[H,F]；各 Tensor 的设备、dtype 相同。"""
    _, f, h = validate_gemm_inputs(x, up)
    if tuple(down.shape) != (h, f):
        raise ValueError(f"下投影权重需要 [{h},{f}]，实际为 {tuple(down.shape)}")
    if down.device != x.device:
        raise ValueError(f"下投影权重需要 device={x.device}，实际为 {down.device}")
    if down.dtype != x.dtype:
        raise TypeError(f"下投影权重需要 dtype={x.dtype}，实际为 {down.dtype}")
    if not down.is_contiguous():
        raise ValueError("下投影权重需要连续行主序布局")


def gemm_reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """CPU/CUDA X[M,K] @ W[N,K].T -> Y[M,N]，保留行序、dtype 和 device。

    每行是当前 Expert 的一条输入记录；Top-K 若已展开，M 即分支行数。
    函数只执行稠密投影，不改变打包顺序，也不应用路由权重。
    """
    validate_gemm_inputs(x, weight)
    return x @ weight.T


def expert_mlp_reference(x: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    """连续 X[M,H] -> GEMM -> GELU -> GEMM -> Y[M,H]，无偏置、保持行序。

    up[F,H]、down[H,F] 与 X 同 dtype/device，支持 CPU 或 CUDA。
    """
    validate_mlp_inputs(x, up, down)
    return F.gelu(x @ up.T) @ down.T
