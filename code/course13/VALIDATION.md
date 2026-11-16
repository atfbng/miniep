# Course13 验证入口

当前唯一正式验收为 [EXACT_ALIGNMENT.md](EXACT_ALIGNMENT.md) 的 DeepEP 零误差检查。旧 `strict_passed`、允许误差的层输出统计以及只比较生成文本均不作为正式通过证据。旧双卡/四卡 AGRS 专用工具和归约策略已删除，原始结果留在历史 artifacts。

独立 PyTorch Reference 验证路由、rank packing、专家和逐 owner 舍入；Triton/NCCL 核心测试中独立 GEMM 对照使用各 dtype 合理容差，这些通过不替代整模型严格逐元素相等。EP=1/2/4 可独立验证；实际空来源由核心 NCCL 回归覆盖。

运行命令、模型约束、真实后端 proof、四 Rank 完整性和失败重放见 EXACT_ALIGNMENT。性能实验必须独立运行，见 PERFORMANCE。
