#!/usr/bin/env python3
"""Lesson 4: verify torchrun, NCCL, and three fixed-size collectives."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path

import torch
import torch.distributed as dist

# 将本课目录加入导入路径，直接运行示例时也能找到本课的 mini_ep。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_ep.distributed import (  # noqa: E402
    DistributedContext,
    equal_split_all_to_all,
    nccl_distributed,
)


@dataclass(frozen=True)
class CollectiveResults:
    """Outputs retained by the lesson's correctness checks.

    All Tensors are contiguous ``torch.int64`` CUDA Tensors on the current
    rank's device.  All-to-All rows have payload ``[source_rank,
    destination_rank]``.
    """

    all_reduce_output: torch.Tensor
    all_gather_output: torch.Tensor
    all_to_all_send: torch.Tensor
    all_to_all_recv: torch.Tensor


def run_collectives(context: DistributedContext) -> CollectiveResults:
    """Run collectives in one identical order on every rank and verify values."""

    # context 保存当前 Worker 的身份、GPU 和默认通信组。
    # 本课使用默认组，因此组内 Rank 等于 Global Rank。
    rank = context.global_rank
    world_size = context.world_size
    device = context.device
    group = context.group

    # 所有 Rank 都按 AllReduce → AllGather → All-to-All 的顺序调用通信。
    # 1) AllReduce：Rank 0 提供 [1, 10]，Rank 1 提供 [2, 20]。
    # SUM 将同一位置相加，原地更新各端缓冲区；两端都得到 [3, 30]。
    all_reduce_output = torch.tensor(
        [rank + 1, 10 * (rank + 1)],
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(all_reduce_output, op=dist.ReduceOp.SUM, group=group)

    # 2) AllGather：Rank 0 提供 [0, 100]，Rank 1 提供 [1, 101]。
    # 每端都收齐所有来源的数据，列表第 s 项保存来源 Rank s 的输入。
    all_gather_input = torch.tensor(
        [rank, rank + 100],
        dtype=torch.int64,
        device=device,
    )
    # empty_like 只分配同 shape、dtype、device 的接收空间，数值由通信填入。
    gathered_rows = [torch.empty_like(all_gather_input) for _ in range(world_size)]
    dist.all_gather(gathered_rows, all_gather_input, group=group)
    # stack 新增来源维；两卡时 shape 为 [2, 2]，结果为 [[0, 100], [1, 101]]。
    all_gather_output = torch.stack(gathered_rows)

    # 3) 等长 All-to-All：每行的两个数分别标记来源 Rank 和目标 Rank。
    # source_ranks 是来源列，destination_ranks 是目标列；stack 将两列配成行。
    # Rank 0 发送 [[0, 0], [0, 1]]，Rank 1 发送 [[1, 0], [1, 1]]。
    source_ranks = torch.full(
        (world_size,), rank, dtype=torch.int64, device=device
    )
    destination_ranks = torch.arange(
        world_size, dtype=torch.int64, device=device
    )
    all_to_all_send = torch.stack((source_ranks, destination_ranks), dim=1)
    # 发送第 d 行发给 Rank d，接收第 s 行来自 Rank s；标签便于追踪数据。
    # Rank 0 收到 [[0, 0], [1, 0]]，Rank 1 收到 [[0, 1], [1, 1]]。
    # 封装内分配接收缓冲区，并调用 dist.all_to_all_single(async_op=False)。
    all_to_all_recv = equal_split_all_to_all(all_to_all_send, group=group)

    # 同步接口建立当前 CUDA 流读取结果所需的依赖，返回不等于 GPU 已完成。
    # 三次通信后统一等待本设备完成，再检查结果并交给 CPU 打印。
    torch.cuda.synchronize(device)

    # 按求和公式和来源顺序构造预期值，可用于任意正整数 world_size。
    rank_sum = world_size * (world_size + 1) // 2
    expected_all_reduce = torch.tensor(
        [rank_sum, 10 * rank_sum], dtype=torch.int64, device=device
    )
    expected_all_gather = torch.stack(
        (
            torch.arange(world_size, dtype=torch.int64, device=device),
            torch.arange(world_size, dtype=torch.int64, device=device) + 100,
        ),
        dim=1,
    )
    expected_all_to_all = torch.stack(
        (
            torch.arange(world_size, dtype=torch.int64, device=device),
            torch.full(
                (world_size,), rank, dtype=torch.int64, device=device
            ),
        ),
        dim=1,
    )

    # 先核对数值，再逐项检查 shape、dtype、device 和连续性。
    torch.testing.assert_close(all_reduce_output, expected_all_reduce)
    torch.testing.assert_close(all_gather_output, expected_all_gather)
    torch.testing.assert_close(all_to_all_recv, expected_all_to_all)

    for name, tensor, expected_shape in (
        ("all_reduce_output", all_reduce_output, (2,)),
        ("all_gather_output", all_gather_output, (world_size, 2)),
        ("all_to_all_send", all_to_all_send, (world_size, 2)),
        ("all_to_all_recv", all_to_all_recv, (world_size, 2)),
    ):
        if tensor.shape != expected_shape:
            raise AssertionError(
                f"{name} shape must be {expected_shape}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.int64:
            raise AssertionError(f"{name} dtype must be int64, got {tensor.dtype}")
        if tensor.device != device:
            raise AssertionError(
                f"{name} device must be {device}, got {tensor.device}"
            )
        if not tensor.is_contiguous():
            raise AssertionError(f"{name} must be contiguous")

    # 保留四份张量，供 main 打印当前 Rank 的通信结果。
    return CollectiveResults(
        all_reduce_output=all_reduce_output,
        all_gather_output=all_gather_output,
        all_to_all_send=all_to_all_send,
        all_to_all_recv=all_to_all_recv,
    )


def parse_args() -> argparse.Namespace:
    # 调试时可调大超时，给断点暂停和变量查看留出时间。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-world-size",
        type=int,
        default=None,
        help="fail if torchrun created a different number of processes",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="Process Group collective timeout",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_world_size is not None and args.expected_world_size <= 0:
        raise ValueError(
            "--expected-world-size must be positive, got "
            f"{args.expected_world_size}"
        )

    # 进入 with：读取 torchrun 环境、按 Local Rank 绑定 GPU，并建立 NCCL 组。
    # 退出 with：上下文管理器通过 finally 释放本次创建的进程组。
    with nccl_distributed(timeout_seconds=args.timeout_seconds) as context:
        if (
            args.expected_world_size is not None
            and context.world_size != args.expected_world_size
        ):
            raise RuntimeError(
                "torchrun world size mismatch: expected "
                f"{args.expected_world_size}, got {context.world_size}"
            )

        results = run_collectives(context)
        # 每个 Worker 分别输出日志；Rank 标签用于区分两个进程。
        print(
            f"[global_rank={context.global_rank} "
            f"local_rank={context.local_rank} "
            f"world_size={context.world_size} "
            f"device={context.device}] "
            f"all_reduce={results.all_reduce_output.cpu().tolist()} "
            f"all_gather={results.all_gather_output.cpu().tolist()} "
            f"a2a_send={results.all_to_all_send.cpu().tolist()} "
            f"a2a_recv={results.all_to_all_recv.cpu().tolist()} PASS",
            flush=True,
        )


if __name__ == "__main__":
    main()
