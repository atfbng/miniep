"""查看 PyTorch/CUDA 环境；--nccl 在 torchrun 下检查一项集合通信。"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
import platform

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nccl", action="store_true", help="在 torchrun 下验证 NCCL AllReduce")
    args = parser.parse_args()
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "visible_gpu_count": torch.cuda.device_count(),
        "nccl_available": dist.is_nccl_available(),
    }
    if args.nccl:
        required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
        missing = [key for key in required if key not in os.environ]
        if missing:
            raise RuntimeError(f"--nccl 需要 torchrun，缺少环境变量：{missing}")
        local_rank = int(os.environ["LOCAL_RANK"])
        if not torch.cuda.is_available() or not dist.is_nccl_available():
            raise RuntimeError("--nccl 需要可用 CUDA GPU 与支持 NCCL 的 PyTorch")
        if not 0 <= local_rank < torch.cuda.device_count():
            raise ValueError(f"LOCAL_RANK={local_rank} 超出可见设备范围")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=timedelta(seconds=60))
        try:
            rank, size = dist.get_rank(), dist.get_world_size()
            # 每个 Rank 提供一个 FP32 CUDA 标量，归约后各 Rank 都应收到同一个和。
            value = torch.tensor([rank + 1.0], device=f"cuda:{local_rank}")
            dist.all_reduce(value)
            expected = torch.full_like(value, size * (size + 1) / 2)
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
            report.update(rank=rank, local_rank=local_rank, world_size=size, all_reduce=value.item())
        finally:
            dist.destroy_process_group()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
