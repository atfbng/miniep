"""扩大下投影的流水线候选，保留课件中的原始调优路径供对照。"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import triton

from mini_ep.kernels.gemm_triton import CANDIDATES, gemm_kernel, triton_gemm_tuned
from mini_ep.model.expert import validate_gemm_inputs, validate_mlp_inputs

# 保留原候选，并联合搜索分块、Warp 数和流水深度；包含用户建议的
# 128×128×64 / 4 / 4。资源不足的配置由 AutoTune 跳过。
DOWN_CANDIDATES = list(dict.fromkeys(CANDIDATES + [
    (bm, bn, bk, warps, stages)
    for bm in (64, 128) for bn in (128, 256) for bk in (32, 64, 128)
    for warps in (4, 8) for stages in (2, 3, 4)
]))
_autotuned = triton.autotune(
    configs=[triton.Config(dict(BM=bm, BN=bn, BK=bk), num_warps=warps,
                           num_stages=stages)
             for bm, bn, bk, warps, stages in DOWN_CANDIDATES],
    key=['M', 'N', 'K'], use_cuda_graph=True,
)(gemm_kernel)
_selected: dict[tuple, triton.Config] = {}


def _key(x: torch.Tensor, weight: torch.Tensor) -> tuple:
    return (*x.shape, weight.shape[0], x.dtype, x.device.index)


def down_projection_tuned(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """连续 CUDA X[M,F]、W[H,F] -> Y[M,H]，FP16/BF16/FP32，FP32 累加。

    输入可以是展开后的 Token 分支，输出保持行序、dtype 和 device。
    M=0 返回空张量。首次调用搜索分块和流水参数，后续直接使用缓存。
    """
    m, n, k = validate_gemm_inputs(x, weight)
    if not x.is_cuda:
        raise ValueError(f'下投影需要 CUDA Tensor，实际 device={x.device}')
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if not m:
        return y
    key = _key(x, weight)
    with torch.cuda.device(x.device):
        if key not in _selected:
            grid = lambda meta: (triton.cdiv(m, meta['BM']), triton.cdiv(n, meta['BN']))
            _autotuned[grid](x, weight, y, m, n, k)
            _selected[key] = _autotuned.best_config
        else:
            cfg = _selected[key]
            gemm_kernel[(triton.cdiv(m, cfg.kwargs['BM']), triton.cdiv(n, cfg.kwargs['BN']))](
                x, weight, y, m, n, k, **cfg.kwargs,
                num_warps=cfg.num_warps, num_stages=cfg.num_stages)
    return y


def down_selected_config(x: torch.Tensor, weight: torch.Tensor) -> dict | None:
    """已完成调优时返回分块、Warp 数、流水级数；否则返回 None。"""
    cfg = _selected.get(_key(x, weight))
    return None if cfg is None else dict(cfg.kwargs, num_warps=cfg.num_warps,
                                         num_stages=cfg.num_stages)


def expert_mlp_down_tuned(x: torch.Tensor, up: torch.Tensor,
                          down: torch.Tensor) -> torch.Tensor:
    """连续 CUDA X[M,H]、up[F,H]、down[H,F] -> Y[M,H]；同 dtype/device。

    上投影沿用原调优版本，中间使用 PyTorch GELU，只扩大下投影候选。
    返回连续输出，保持原输入行序，不应用路由权重。
    """
    validate_mlp_inputs(x, up, down)
    return down_projection_tuned(F.gelu(triton_gemm_tuned(x, up)), down)
