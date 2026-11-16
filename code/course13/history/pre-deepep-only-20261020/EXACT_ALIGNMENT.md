# 默认返回 dtype 与两个原生精度基准

`ep_forward` 的推荐路径现在是：输入激活 → Expert down 的 FP32 累加 → 乘 FP32 routing weight → 转为输入 `tokens.dtype` → 返回 A2A → 恢复顺序与合并。分支只加权一次，Combine 使用 `preweighted=True`。默认 Grouped GEMM 在 down 写出前融合乘法与转换，避免未加权结果提前舍入；`fuse_return_weight=False` 保留 FP32 中间结果，供非融合对照。

核心默认参数为 `return_dtype=None, weight_location="expert", fuse_return_weight=None`。None 传输类型取输入 dtype，None 融合开关在 grouped/expert 时启用，在 loop 对照中使用独立加权。独立 PyTorch Reference 位于 `reference/moe.py`，不依赖 Triton 计算。

核心模型激活仅支持 FP32/BF16，即使 `validate=False` 也拒绝 FP16 激活。vLLM 接入仍只支持 BF16 模型；显式 FP16 传输并不意味着支持 FP16 模型。正常 BF16 激活回传 BF16，FP32 激活回传 FP32。传输策略在所有 EP Rank 上必须一致。

| 入口 | 默认值 | 诊断选项 |
|---|---|---|
| `MINIEP_RETURN_DTYPE` | `auto`，跟随 tokens.dtype | `fp32 / fp16 / bf16` |
| `MINIEP_WEIGHT_LOCATION` | `expert` | `source`，旧加权位置 |
| `MINIEP_FUSE_RETURN_WEIGHT` | `auto` | `0` 独立 FP32 加权；`1` 强制融合 |

旧 FP32 未加权回传对照须同时指定 `fp32 / source / 0`，不能再用省略参数代表旧默认。metadata 记录模型 dtype、请求传输 dtype、实际 return buffer.dtype、发送字节数和实际归约策略。两个方向的行 splits 按 EP rank 排列；变长返回交换前向 send/recv splits，固定容量保持等长分段。FP32 权重向量与展开分支使用相同打包、padding 和恢复索引。

## 独立的两个基准

两种原生后端自身可能不同，验收分别对照各自独立启动的原生 Runner，不要求两个基准输出相等。不修改 vLLM/DeepEP 的专家计算，也不允许切换后端后继续记为通过。

| 拓扑及基准 | 实际 Prepare/Finalize | Mini-EP owner 合并 |
|---|---|---|
| 双卡 TP=2、DP=1、EP=2 原生对照 | `MoEPrepareAndFinalizeNoDPEPModular`，manager 为 None | 原有本地 FP32 合并并恢复 TP 行 |
| 四卡 TP=2、DP=2、EP=4，allgather_reducescatter | `MoEPrepareAndFinalizeNaiveDPEPModular`，`AgRsAll2AllManager` | 模型 dtype owner 部分和及原有兼容 NCCL 归约边界 |
| 同四卡拓扑，deepep_high_throughput | `DeepEPHTPrepareAndFinalize`，`DeepEPHTAll2AllManager` | 模型 dtype owner 部分和后按 owner 顺序 local_fp32 合并，待全量验收 |

vLLM 0.29.0 的双卡 DP=1 不会因设置后端名称而执行 DP AllGather/ReduceScatter；因此它是单独的原生双卡控制组。验收脚本显式校验这个已知路径，不将其记作 DeepEP 回退。四卡基准必须观察到对应 manager、Prepare/Finalize 和 TritonExperts 的实际类。生产 Runner 保留旧 owner 兼容归约；DeepEP 合并规则由外围验证 Runner 显式选择，未将短样本规则推广为生产默认。

## 零误差验收与复现

环境固定真实 Qwen3-30B-A3B BF16、48 层、128 Experts/Top-8、vLLM 0.29.0，eager、同步调度、TP=2、关闭 prefix caching、量化、EPLB 和 DBO；两侧均关闭 BF16 reduced-precision reduction，控制器固定 `NCCL_NVLS_ENABLE=0`。模型和 tokenizer 文件 SHA256、全部运行源码 SHA256、版本、GPU 快照、配置与命令保存在独立实验目录。独立原生参考的兼容性比较冻结模型/版本/配置/工作负载及全部 `mini_ep` 源码，观察器源码单独保留；仅调整统计和失败保存逻辑时，可以复用原生结果，不能复用不同核心计算或不同配置的参考。新目录必须不存在；脚本不删除旧数据。可复用先前哈希清单，但会检查所有文件大小与 mtime。

从本课目录运行，`$MODEL_PATH` 为已下载的真实模型路径：

```bash
# 双卡原生控制组（NoDPEP + TP reduction）
CUDA_VISIBLE_DEVICES=0,1 python3 test/deepep_alignment.py \
  --backend allgather_reducescatter --world-size 2 \
  --model "$MODEL_PATH" --output results/exact-native-ep2

# 四卡旧原生基准
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 test/deepep_alignment.py \
  --backend allgather_reducescatter --world-size 4 \
  --model "$MODEL_PATH" --output results/exact-agrs-ep4

# 四卡 DeepEP 基准（要求已安装兼容 DeepEP）
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 test/deepep_alignment.py \
  --backend deepep_high_throughput --world-size 4 \
  --model "$MODEL_PATH" --output results/exact-deepep-ep4
```

控制器按顺序先启动三次独立原生进程，逐层输出 SHA256、完整 logits 与生成序列检查自身可重复；原生基准保留原 `MoERunner` 类。候选使用真实新默认值，而非显式 BF16 替代 auto。每个候选独立启动，先检查其原生路径与独立参考一致，再逐层 shadow 比较、teacher-forced 全词表比较和自由生成。所有 Rank 按同一顺序参加验证 collective；首个层差异通知所有 Rank 重放该层并保存现场，之后退出，由 torchrun 终止其他进程。每个 job 有超时与进程组清理。无差异路径将精确且有限的比较保留在 GPU 上，每层只读取一次全 EP 失败标记；失败后才计算首个坐标和误差统计，门槛没有改变。

原生观察器把每层输出复制到 pinned CPU buffer，copy 排在结果生产流上、先于同流 buffer 复用，并记录 CUDA Event。队列持有 CPU buffer，CPU 哈希消费前等待对应 Event，compute_logits 和每个 case 末尾 drain 队列。没有把这些测试侧复制当作通信/计算 overlap。失败现场写盘后所有 Rank 再确认一次完成，避免 torchrun 因首个异常过早杀掉尚未保存的 Rank。

完整矩阵包含：32 条 prompt，batch size 1/4，生成 64 Token；DP 相同及 rotated 输入；DP 不同请求数、输入长度及输出步数、两侧分别空闲、不同请求先后完成、长 prefill、最长 prefill、长 decode；长序列子集生成 256 Token；seed=13 候选三次独立启动，seed=37/123 各一次。四卡全量 corpus 每轮去除 TP 副本后是 128 条序列、8192 个生成 Token；双卡是 64 条、4096 Token。`--reuse-completed EARLIER_SUITE` 只复用通过当前完整性复查、模型哈希/核心源码/配置一致的已完成独立 native job；所有候选仍重新启动，旧目录不删除或覆盖。`--dry-run` 输出 job 顺序；`--smoke` 仅检查少量请求，状态为 smoke_passed，不能代替全量验收。

验收检查路由 ID、slot 顺序与 FP32 权重，所有 Rank 和所有层的 MoE 输出，teacher-forced 全词表 logits，自由生成全部 Token，真实返回 dtype 和字节数。NaN/Inf、任意元素差异、缺失 Rank、缺失后续 decode、错误后端、缺失阶段和失败现场均失败。`validation/exact.py` 不调用旧容差指标；旧 `strict_passed` 不是新验收依据。

失败现场保存各 Rank 的首个差异坐标、输入、路由、打包/恢复 metadata、本地专家权重、加权分支、owner 部分和与最终输出。`test/replay_alignment_failure.py --run FAILED_JOB --output diagnosis.json` 离线检查直接可观察的加权分支；DeepEP 去重行与 Mini-EP 展开分支按输入值、expert 和 FP32 weight 匹配。原生 down 的 FP32 寄存器不可直接读取，不把 Reference 推导当作原生测量。未覆盖的诊断边界必须明确记录，不能推定其已一致。

## 本次运行状态

本次新增证据写入 `/experiment/default-exact-v1/v2/v3/v4` 等独立目录（gpu5 四卡组位于 `/workspace/default-exact-v3-results`、`default-exact-v4-results`），历史 `return-20261020` 数据不覆盖。最终配置、结果、失败与未执行项记入 [EXACT_ALIGNMENT_RESULTS.json](EXACT_ALIGNMENT_RESULTS.json)。旧短样本和容差结果保留在 [RETURN_TRANSPORT.md](RETURN_TRANSPORT.md)，不能充当本次完整通过证明。

精度验证与性能测量分开。BF16 激活的返回分支数据从 4 字节/值降为 2 字节/值，同时新增正向 FP32 权重通信；这不构成总延迟下降的证据。本次没有据此宣称加速。
