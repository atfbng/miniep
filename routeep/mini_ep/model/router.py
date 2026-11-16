"""框架适配层使用的 RouterOutput 数据对象。

本文件只定义路由结果的契约，不实现路由算法。算法可以来自独立 Reference，
也可以来自 vLLM 原生 selector；这个对象不引入额外 GPU 计算。
"""

from dataclasses import dataclass
import torch


@dataclass(slots=True)
class RouterOutput:
    """当前来源 Rank 的路由结果：[N,K] 全局专家编号和 FP32 权重。

    字段：
        logits: [N,E] gate 输出，模型 dtype；E 是全局专家数。
        expert_ids: [N,K] 全局专家编号，位于输入设备。
        routing_weights: [N,K] FP32 权重，与 expert_ids 同形状。

    专家计数不在源端附加，而是在 Dispatch 之后由 GPU 规划器统计。
    """

    logits: torch.Tensor
    expert_ids: torch.Tensor
    routing_weights: torch.Tensor
