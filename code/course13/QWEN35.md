# Qwen3.5 的适用范围

本轮唯一精度验收模型是 Qwen3-30B-A3B BF16。Qwen3.5 的旧逐层、生成、服务与性能记录见 [历史资料](history/README.md)，不能用于证明整理后版本与 DeepEP 零误差。

仍保留框架配置识别和 Shared Expert 接口。DP=2/SP 的共享分支在本地计算，与路由结果相加；核心 EP 不接管线性 Attention、模型状态或 vLLM 生命周期。DP=1 的 shared/routed 联合 TP 舍入依赖已删除的旧数值结构，当前显式拒绝该组合，避免静默改变精度。没有删除 Attention 或模型正确性自身需要的 TP/DP/SP 通信。

完整 FP16 模型、Qwen3.5 新版本完整 DeepEP 验收以及这些配置的性能测量不在本轮已通过范围中。
