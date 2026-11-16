# 四 Rank DeepEP 验收

固定 Attention TP=2、DP=2、MoE EP=4。两个 TP group 为 `[0,1]`、`[2,3]`，EP group 按 DP-major/TP-minor 排列。每 Rank 拥有 32 个完整 Qwen3 Expert。真实 Token redistribution 使用 Mini-EP 的两次 NCCL A2A。

正式计划、通过范围和未完成项统一维护在 [EXACT_ALIGNMENT.md](EXACT_ALIGNMENT.md) 与 [EXACT_ALIGNMENT_RESULTS.json](EXACT_ALIGNMENT_RESULTS.json)，避免保留互相冲突的策略说明。

单侧空闲请求测试不自动意味着框架真的传入零行：vLLM 可能执行一行 dummy padding。报告分别记录逻辑请求数和实际 `framework_rows`。核心 NCCL 回归用真实零行来源验证收发和恢复，不能混淆两者。
