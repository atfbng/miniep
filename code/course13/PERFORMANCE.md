# 最终 Mini-EP 与 vLLM DeepEP 的性能测量

唯一对照为真实 vLLM DeepEP HT。旧原生 AGRS、FP32 默认及旧 owner 兼容路径的计时仅在 [history](history/README.md) 中保留，不用于最终差距或简化收益结论。最终六配置计时和独立采样已完成，详见 [DEEPEP_REPORT.md](DEEPEP_REPORT.md) 和原始 samples。

后续核心死代码清理的清理前后增量测量另见 [CORE_CLEANUP.md](CORE_CLEANUP.md)，不混入本页原有六配置 DeepEP 差距统计。

## 公平配置

同一 H100 节点、同一组四 GPU、Qwen3-30B-A3B BF16 权重/Tokenizer、TP2/DP2/EP4、literal 输入与种子。两者使用 eager、同步调度、Triton experts、相同 KV、8192-token 调度预算、32 请求槽，无 prefix cache、DBO/EPLB/量化。固定 CPU 配额/内存和软件环境随各 Rank 保存。

Native 测量检查真实 manager/prepare-finalize/expert/Runner 类；Mini-EP 启动核对实际 NCCL send/recv buffer dtype、shape、元素数、字节。启动检查包装在 warmup 前移除；普通计时不保留 shadow、逐层验证、张量 clone/落盘或测试日志。生产路径本身的 Gate、计数交换/Host split、通信、恢复、模型调度和输出处理仍计入耗时。

整模型精度先通过相同工作负载和调度预算的严格验收，再对整理后最终代码作性能结论。整理前快照只选用已验证的 DeepEP local FP32 数值分支；快照外的构造选择器不改计算/通信/forward，源码清单和选择器一同存档，不进入正式核心。

## 工作负载

| 用例 | 每 DP 请求数 | prompt Tokens | output Tokens |
|---|---:|---:|---:|
| decode_b1 | 1 | 128 | 128 |
| decode_b4 | 4 | 128 | 128 |
| decode_b32 | 32 | 128 | 128 |
| prefill_b1_long | 1 | 2048 | 1 |
| prefill_b4_short | 4 | 128 | 1 |
| prefill_b4_long | 4 | 2048 | 1 |

DP 两侧输入内容不同，TP peers 输入相同。源码 `benchmarks/deepep_workloads.py` 与额外精度检查共用 literal 工作负载，不能只选有利的小 batch。

## 测量口径

每个实现独立启动三次，按 Native/Mini、Mini/Native、Native/Mini 交替；每用例 warmup=3、测量=5，共 15 个正式样本。排除模型加载、初始化、JIT、输入构建、边界同步和结果序列化，完整保留原始样本、p10/p90、min/max 和中位数。

- 端到端：提交 literal IDs 到获得完整输出，包含正常调度、模型及输出处理；墙钟和同一 ROI 的 CUDA Event 间隔同时记录。
- Prefill：vLLM engine-core monotonic 时间，从批次最早 scheduling 到最后 first-token；包含生成首 Token。报告 batch 延迟和输入 Token 吞吐。
- Decode：从批次最早 first-token 到最后 last-token，除以 output_tokens−1；报告每步延迟。单输出用例无 decode 值。
- 各重复先对四 Rank 同口径取最大值；吞吐只统计两个 DP 的唯一 Token，不重复计算 TP。

时间戳需要开启 vLLM 请求统计（`disable_log_stats=False`），两者一致；它属于本测量口径，统计本身的额外开销未单独消融。单独的 Event profiling 扰动与正式无逐层采样轮次分开报告。各阶段分别取最慢 Rank，因此不能相加当作同一 Rank 的端到端时间。

最终报告同时给 Mini-EP/DeepEP 延迟比、吞吐比和简化前/后变化。若样本分布或 CPU 抖动不足以支持稳定收益，明确记录不确定性。

## 命令

```bash
cd code/course13
python3 benchmarks/deepep_latency.py --model "$MODEL_PATH" \
  --checkpoint-manifest results/exact-deep4/checkpoint.json \
  --output results/latency-final --label final \
  --trials 3 --warmup 3 --repeats 5 --profile-stages
```

`--profile-stages` 在正式无钩子轮次结束后另起 profiling 进程，不把其耗时混入正式统计。Mini-EP 对 0/24/47 层采样阶段 CUDA Events、实际 padding/通信字节；Native 只记录能可靠观测的完整 MoE，无法可靠拆分其 Dispatch/A2A/Expert/Combine 时明确标注未观测。

当前没有通信计算 overlap 结论。若后续提出重叠收益，需额外 Nsight Systems Timeline，并报告 exposed communication time 与 overlap ratio；仅 async 或 CUDA Event 不能证明重叠。

## 数据保留与版本

每 Rank 保存 environment、manifest、backend proof、Mini-EP transport proof、samples、completed。控制器保存每个命令、PID、时间和 exit code；失败/缺 Rank/缺样本/输入不一致/后端回退均不能作为有效比较。

源码 `SOURCE_MAP.json`、独立 source SHA 清单和冻结源码归档对应最终代码；文档更新不能把旧实现结果改名成新版本通过。返回 BF16 相比 FP32 少一半每元素字节，仅为载荷事实，不能代替延迟测量。

## 本轮结果

最终Mini-EP/DeepEP端到端延迟比为1.041～1.258，生成吞吐比0.795～0.961。decode batch1/4/32延迟比分别1.073/1.057/1.258。简化前后Mini-EP中位数变化−2.25%～+1.26%，所有p10/p90区间重叠；未见稳定退化，也没有稳定提速证据。

完整prefill/decode、输入/生成吞吐、统计分布、源版本、实际返回字节和阶段瓶颈在 [完整报告](DEEPEP_REPORT.md)。正式比较通过全部GPU UUID、模型、软件、推理选项、literal输入和输出Token一致性核对。Native长prefill基线有波动，不能把归一化比值下降直接归因为代码简化。
