"""课件 3.8 的固定分块与按形状调优 GEMM。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from mini_ep.model.expert import validate_gemm_inputs, validate_mlp_inputs


@triton.jit
def gemm_kernel(X, W, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # program (pid_m,pid_n) 只负责 Y 中一个 [BM,BN] 输出块。
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for block in range(tl.cdiv(K, BK)):
        ks = block * BK + tl.arange(0, BK)
        a = tl.load(X + rows[:, None] * K + ks[None, :],
                    mask=(rows[:, None] < M) & (ks[None, :] < K), other=0)
        # W 仍保存为 [N,K]；按 b[k,n] 排列地址，直接读取逻辑转置。
        b = tl.load(W + cols[None, :] * K + ks[:, None],
                    mask=(cols[None, :] < N) & (ks[:, None] < K), other=0)
        acc += tl.dot(a, b, input_precision="ieee")
    # 写回时转成 Y 的 dtype；越界行列不写入。
    tl.store(Y + rows[:, None] * N + cols[None, :], acc,
             mask=(rows[:, None] < M) & (cols[None, :] < N))


# BM、BN、BK、Warp 数、流水级数，与课件中的候选集合一致。
CANDIDATES = [
    (16, 32, 32, 4, 3), (16, 64, 32, 4, 3),
    (16, 128, 64, 4, 3), (32, 64, 64, 4, 3),
    (32, 128, 32, 4, 4), (32, 128, 64, 4, 3),
    (64, 64, 32, 4, 4), (64, 128, 32, 4, 4),
    (64, 128, 64, 4, 3), (64, 256, 64, 8, 3),
    (128, 64, 32, 4, 4), (128, 128, 32, 4, 4),
    (128, 128, 64, 4, 3), (128, 256, 64, 8, 3),
]
_autotuned_kernel = triton.autotune(
    configs=[triton.Config({"BM": bm, "BN": bn, "BK": bk},
                           num_warps=warps, num_stages=stages)
             for bm, bn, bk, warps, stages in CANDIDATES],
    key=["M", "N", "K"], use_cuda_graph=True,
)(gemm_kernel)
_selected: dict[tuple[int, int, int, torch.dtype, int], triton.Config] = {}


def _key(x: torch.Tensor, weight: torch.Tensor) -> tuple[int, int, int, torch.dtype, int]:
    return x.shape[0], weight.shape[0], x.shape[1], x.dtype, x.device.index


def _project(x: torch.Tensor, weight: torch.Tensor, *, tuned: bool) -> torch.Tensor:
    m, n, k = validate_gemm_inputs(x, weight)
    if not x.is_cuda:
        raise ValueError(f"Triton GEMM 需要 CUDA Tensor，实际 device={x.device}")
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m == 0:
        return out
    with torch.cuda.device(x.device):
        if tuned and _key(x, weight) not in _selected:
            grid = lambda meta: (triton.cdiv(m, meta["BM"]), triton.cdiv(n, meta["BN"]))
            _autotuned_kernel[grid](x, weight, out, m, n, k)
            _selected[_key(x, weight)] = _autotuned_kernel.best_config
        else:
            config = _selected[_key(x, weight)] if tuned else triton.Config(
                {"BM": 16, "BN": 32, "BK": 32}, num_warps=4, num_stages=3)
            grid = (triton.cdiv(m, config.kwargs["BM"]), triton.cdiv(n, config.kwargs["BN"]))
            gemm_kernel[grid](x, weight, out, m, n, k, **config.kwargs,
                              num_warps=config.num_warps, num_stages=config.num_stages)
    return out


def triton_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """连续 CUDA X[M,K]、W[N,K] -> Y[M,N]，固定 16×32×32 分块。

    dtype 为相同的 FP16/BF16/FP32，内部 FP32 累加；输出连续且保持
    输入行序、dtype、device。M 可为 0，投影不处理路由或 Top-K 合并。
    """
    return _project(x, weight, tuned=False)


def triton_gemm_tuned(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """同 triton_gemm 的布局；首次调用按 M/N/K 调优，随后直接用缓存参数。"""
    return _project(x, weight, tuned=True)


def selected_config(x: torch.Tensor, weight: torch.Tensor) -> dict[str, int] | None:
    """返回已选分块、Warp 数和流水级数；未调优或空输入返回 None。"""
    config = _selected.get(_key(x, weight))
    if config is None:
        return None
    return {**config.kwargs, "num_warps": config.num_warps, "num_stages": config.num_stages}


def expert_mlp_triton(x: torch.Tensor, up: torch.Tensor, down: torch.Tensor,
                      *, tuned: bool = False) -> torch.Tensor:
    """CUDA X[M,H] -> Y[M,H]；up[F,H]、down[H,F] 连续且与 X 同 dtype/device。

    两次 GEMM 使用同一个 PyTorch GELU。tuned=True 时上下投影分别选择
    配置；输出仍按输入行序排列，无偏置、不应用路由权重。
    """
    validate_mlp_inputs(x, up, down)
    project = triton_gemm_tuned if tuned else triton_gemm
    hidden = project(x, up)
    activated = F.gelu(hidden)
    return project(activated, down)
