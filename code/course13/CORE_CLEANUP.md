# Course13 核心死代码清理（2026-10-21）

本轮只清理 `mini_ep/` 中能够证明不参与有效执行的内容。此前未提交的实现、Reference、实验记录和用户草稿保留。本轮证据单独写入 `artifacts/course13-core-cleanup-20261021/`；上一轮完整 DeepEP 验收和性能报告仍保留原冻结源码版本。

## 实际删除

| 位置 | 删除内容及依据 |
|---|---|
| `model/router.py::RouterOutput` | 无读写方的可选 `tokens_per_expert` 字段；专家计数仍由接收端 GPU plan 负责 |
| `integration/runner.py` | `_fixed_rows_per_rank` 单纯转发包装；直接调用原函数，参数、容量策略不变 |
| `integration/runner.py::forward_ep` | 无消费者的 `router_logits_reused` 调试字段 |
| `integration/runner.py::forward_ep` | shared expert 的旧 TP all-reduce 分支；构造函数已拒绝 shared expert 且非 SP 的配置，该分支在所有受支持配置下不可达 |
| `model/expert.py` | 仅使用一次的 `compute = swiglu_triton` 别名；直接调用原函数 |
| `model/moe.py::ep_forward` | 无消费者的 `fixed_pack_backend_requested` 字段；仍记录实际采用的 `fixed_pack_backend` |

同时修正外围四卡测试 `checks/tp_dp.py` 的旧 `token_range` 导入，使用实际所属的 `integration/input_layout.py`。未恢复已删除的旧模块导出。

实际核心为 24 个 Python 文件，2592 → 2575 行。未以删行数代替调用关系和语义检查。

## 保留与不变的行为

独立 PyTorch Reference、Expert loop/Grouped、融合/非融合 down 对照、显式传输 dtype 诊断、运行前配置校验、固定/变长 A2A、必要 TP 输入输出重组、NVTX 和按需 CUDA Event 测量都有现存调用方，继续保留。固定打包的 GPU 布局 metadata 用于 Reference/Kernel 正确性检查，不能因为正常推理未读取就删除。

默认仍为 down FP32 累加 → 乘一次 FP32 routing weight → 转 `tokens.dtype` → 返回 A2A → 唯一的 DeepEP owner 合并顺序。BF16 模型回传 BF16；传输诊断不代表支持完整 FP16 模型。Qwen3.5 DP1 仍明确拒绝，未移除任何有效配置所需的 TP collective。

所有 GPU Kernel、tile 参数、加权/舍入、发送缓冲和通信调用逐字节不变。`source-audit.json` 使用窄范围 AST 规则，只允许表中列举的删除与调用内联，要求剩余全部核心执行 AST 相等，并检查 shared/SP 初始化约束。审计不替代运行和性能测量。

## 本轮验证与性能

H100 新执行目录为 `/experiment/core-cleanup-20261021`，通过已有 qemu-host/h100-jump 跳板连接；没有覆盖旧结果。

| 本轮新增执行 | 结果 |
|---|---|
| 本地 unittest | 42 项中 34 通过、8 项 CUDA 明确 skip；保留 NumPy ABI 环境警告日志 |
| H100 unittest | 42/42 通过，无 skip |
| 真实四卡 TP2/DP2 回归 | 各 Rank 72 例通过；独立 DP 输入、空/不均衡 Token、固定/变长 A2A |
| 真实四卡回传诊断 | 各 Rank 432 例通过；FP32/BF16 模型核心，FP32/FP16/BF16 传输，loop/Grouped、融合/非融合、固定/变长 |
| 真实整模型 DeepEP 严格对照 | `candidate-rotated-13-r1` 通过，`errors=[]`；32 条 prompt，batch 1/4，64 个生成 Token，DP 输入不同 |
| 源码/布局检查 | 24 个核心文件的限定 AST 审计、课程隔离、compileall、双 SOURCE_MAP、git diff --check 通过 |

整模型对照为 Qwen3-30B-A3B BF16、vLLM 0.29.0、真正的 `deepep_high_throughput`、TP2/DP2/EP4。只复用未修改原生环境下的独立 stock MoERunner golden：当前原生源码/二进制指纹、模型、配置、输入和完整四 Rank 覆盖通过复用审计，并在本轮重新执行 native_check。没有沿用旧 Mini-EP 候选通过结果来代替本轮执行。

本轮 native_check / compare / teacher / free 均完成 40 个批次；全 Rank、48 层的专家 ID、slot/FP32 权重、MoE 输出、完整词表 logits 和全部生成 Token 严格相等。共 128 次请求序列执行、8192 个唯一生成 Token；teacher 和 native_check 各 16384 个含 TP 副本的位置，完整词表 151936 列。数值必须 finite；检查 shape/dtype/device、实际返回 buffer dtype 为 BF16、auto 返回策略和发送字节，未使用容差通过作为零误差证明。

回传诊断最初一次启动漏传必需的 `--output`，被 argparse 拒绝，未进入计算；原失败日志保留，补齐参数后四 Rank 均通过。没有数值失败或降低阈值。以上小张量/Reference 回归与真实整模型严格零误差验收分别报告，不混用其容差口径。

### 清理前后性能

在同一 gpu3 H100 四卡、同一模型/权重、BF16、TP2/DP2/EP4、相同输入、seed=13、8192-token 调度预算和32请求槽上运行。顺序为 before → after → after → before，每版本两次独立引擎启动；每个场景、每次启动先 warmup 3 次，再测量5次，共每版本每场景10个正式样本。每次取最慢 Rank 的端到端延迟，再报告 median / p10 / p90；初始化、加载、JIT和预热不计入正式样本。

关闭 shadow、逐层诊断/落盘和阶段 Event profiling。双方保留相同的正常 vLLM 请求统计以获取 prefill/decode 时间；该统计及偶发进度读取的扰动未单独消融，不宣称测量工具零开销。必要的 Gate、metadata 构造、计数交换/Host split、两次 A2A、恢复和合并均计入生产路径。最终统计器检查四 Rank 完整性、启动检查已移除、model/options/versions/GPU UUID/driver 一致、literal 输入和全部生成 Token 一致；启动 A2A trace 及除明确删除字段之外的布局 metadata 逐项相等。

| 场景（每 DP） | 清理前 e2e median ms | 清理后 e2e median ms | 变化 | p10/p90 前 / 后 ms |
|---|---:|---:|---:|---|
| decode B1，输入128/输出128 | 11236.175 | 11146.187 | −0.80% | 10973.435–11614.649 / 10900.343–11405.869 |
| decode B32，输入128/输出128 | 13308.345 | 13226.794 | −0.61% | 13095.728–13516.646 / 13170.873–13759.166 |
| prefill B4，输入2048/输出1 | 226.052 | 225.656 | −0.18% | 225.233–229.179 / 224.193–227.690 |

| 场景 | prefill median 前 → 后 ms | decode 每步 median 前 → 后 ms |
|---|---:|---:|
| decode B1 | 104.097 → 103.776 | 87.638 → 86.841 |
| decode B32 | 127.572 → 127.341 | 103.722 → 103.096 |
| prefill B4 长输入 | 221.455 → 221.197 | 无后续 decode |

三个场景的 e2e、prefill 和可用 decode 指标，其 p10/p90 区间均重叠。**本轮测量未见性能退化；不把小幅中位数下降宣称为稳定加速，也不能证明任意配置严格零开销。** 本轮没有减少 A2A 载荷、改变 Kernel 或引入重叠；未采集 Nsight Timeline，不作 overlap 结论。上一轮六配置 Mini-EP/原生 DeepEP 差距仍保留其原始源码/日期，本轮不新增一个“已重测全部 DeepEP 性能”的结论。

### 可复现证据与命令

[原始 archive](../../artifacts/course13-core-cleanup-20261021/remote-raw.tgz) 保存四 Rank 的完整 JSON/JSONL、完成状态、源文件 SHA、原生命中后端、模型清单、环境、实际 argv、每次计时及 Token。整模型原生 logits golden 继续保留在原 H100 持久目录，没有覆盖或重复下载。

- [本轮汇总](../../artifacts/course13-core-cleanup-20261021/results.json)：精度通过及四次计时完整性、公平性检查 `errors=[]`；含全部统计。
- [源码等价审计](../../artifacts/course13-core-cleanup-20261021/source-audit.json) 与 [仅本轮核心差异](../../artifacts/course13-core-cleanup-20261021/cleanup-only.patch)。清理前核心在 `before.tgz`，最终实体文件冻结在 `final-course13.tgz`。
- [原始执行记录](../../artifacts/course13-core-cleanup-20261021/remote-raw/progress.json) 和 `remote-raw/run.py` / `remote-raw/resume.py`：含失败启动及修正后的调用，保留两轮顺序。机器环境和GPU UUID见各 `rankN/environment.json`。
- [本地测试日志](../../artifacts/course13-core-cleanup-20261021/local-unit.log)；GPU日志为 `remote-raw/unit.log`、`tp-dp.log`、`return-transport-retry1.log`；精度逐层/全词表日志为 `remote-raw/accuracy/candidate-rotated-13-r1/rankN/`。

主要命令（本课根目录，GPU四卡；输出目录需选新的名称）：

```bash
python3 -m unittest discover -s checks -p 'test_*.py'
NCCL_NVLS_ENABLE=0 torchrun --standalone --nproc-per-node=4 checks/tp_dp.py
NCCL_NVLS_ENABLE=0 torchrun --standalone --nproc-per-node=4 checks/return_transport.py --output "$NEW_OUTPUT/return-transport"
python3 test/deepep_alignment.py --model /model --output "$NEW_OUTPUT/accuracy" \
  --checkpoint-manifest /experiment/default-exact-v3/deep4-full/checkpoint.json \
  --jobs candidate-rotated-13-r1 --reuse-completed /experiment/default-exact-v3/deep4-full \
  --audit-native-reuse
NCCL_NVLS_ENABLE=0 torchrun --standalone --nproc-per-node=4 benchmarks/deepep_latency_run.py \
  --model /model --output "$NEW_OUTPUT/timing" --checkpoint-manifest /experiment/default-exact-v3/deep4-full/checkpoint.json \
  --backend miniep --label after --warmup 3 --repeats 5 \
  --cases decode_b1 decode_b32 prefill_b4_long
```

性能命令在清理前后两个独立实体快照中按上述顺序执行；整个编排显式设置 `OMP_NUM_THREADS=1`、`MINIEP_STAGE_TIMING=0`、`NCCL_NVLS_ENABLE=0`，不是通信后端回退。实际完整 argv 及路径以原始执行记录为准。Git HEAD `64c26ea` 只是未提交工作树的历史基点，不能代替本轮逐文件 SHA/冻结 archive。

本轮控制 golden 为独立原生 `MoERunner`。当前外围 `DeepEPExactRunner` 在 native_check/compare 阶段显式调用未修改的 `MoERunner.forward`，实际 manager/prepare-finalize/experts 为 `DeepEPHTAll2AllManager / DeepEPHTPrepareAndFinalize / TritonExperts`，四 Rank 均保存 proof，未回退后端；包装只负责记录与比较，未修改原生专家计算。

本轮不宣称重跑上一轮九种配置全部矩阵，不把自由生成一致单独视为精度验收，也不根据代码减少宣称加速。
