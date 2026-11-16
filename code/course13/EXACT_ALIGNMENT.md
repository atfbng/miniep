# vLLM DeepEP 严格零误差验收

唯一目标是 `vllm==0.29.0`、真实 Qwen3-30B-A3B BF16、TP=2/DP=2/EP=4 的 `deepep_high_throughput`。其他通信后端不是本课当前精度验收目标。最终范围、结果与尚未执行项维护在 `EXACT_ALIGNMENT_RESULTS.json` 和可复现报告中；smoke 或部分任务状态不能算全量通过。

## 冻结实际执行路径

- 模型 48 层、hidden=2048、128 Experts、Top-K=8、expert intermediate=768；真实权重/配置/Tokenizer 文件保存 SHA256 清单，后续启动检查不可变性。
- BF16、eager、同步调度、无量化、无 EPLB/DBO、无 prefix cache、无 chunked prefill；种子与完整 LLM options 写入各 Rank 的 `environment.json`，输入保存为 literal Token IDs。
- 原生基准保留真正的 `MoERunner`，只加观察钩子；实际检查 `DeepEPHTAll2AllManager`、`DeepEPHTPrepareAndFinalize`、`TritonExperts`、EP=4。后端不符直接失败，不修改 vLLM/DeepEP 计算，不允许静默回退。
- Mini-EP 用正式默认核心：down FP32 累加 → FP32 加权 → 输入 dtype 返回。Grouped 默认融合；显式 F32/F16/BF16 transport 和非融合仅供诊断。每层记录实际发送及返回 buffer dtype/字节；门禁要求 `return_dtype_requested=auto`，显式 BF16 诊断即使 dtype 相同也不作为默认推导的证据；没有通过重设 FP32 默认取巧。
- Qwen3 在 vLLM 0.29 的 `router_logits` 参数传入 hidden_states 占位，原生 Runner 自己执行 Gate；Mini-EP 与此一致。不是把参数名当已计算 logits。

## 固定计划

全量语料为 32 条 prompt、batch=1/4、生成 64 Token；同 DP 输入与旋转后的不同 DP 输入分别验证。原生旋转输入种子 13 独立启动三次，逐层 MoE tensor SHA256、完整 logits tensor 和全部生成 Token 均需相同。候选旋转输入种子 13 也独立启动三次；种子 37、123 另起进程验证。

边界覆盖 balanced、不同请求数/长度/生成长度、idle_dp0、idle_dp1、single decode、请求先后完成、wide prefill、128-token decode、长 prefill 和最大 2048 prompt。补充长测试用 batch=1/4、四条长短 prompt、256-token decode，max model len=2304。

每个候选任务依次执行：

1. `native_check`：当前独立原生路径与保存的完整 logits/Token 对照。
2. `compare`：相同原生前缀上，每层先执行原生再执行 Mini-EP，检查路由 Expert IDs、slot 顺序、routing weights 和 MoE 输出逐元素相等；之后继续原生前缀。
3. `teacher`：Mini-EP 整模型，对完整词表 logits 做逐元素检查，记录后才固定下一 Token 前缀。
4. `free`：Mini-EP 不固定前缀，全部生成 Token 序列必须一致。

所有 Rank/层/模型步/请求/生成位置必须完整；同值 Inf/NaN 也失败，shape/dtype/device 不同失败。缺少 Rank、实际后端 proof、返回 buffer 元信息、任何 phase 或最后几层执行均不能通过。门禁不读取允许误差的 `strict_passed`。

## 实际空输入与计时工作负载

vLLM 离线调度器可能为无请求 DP 执行一行 dummy。请求空闲测试分别保存 `logical_requests` 和 `framework_rows`，不会把逻辑空闲描述成物理零行。

`checks/deepep_empty_rank.py` 直接在真实模型的 48 个原生 MoERunner/DeepEP 层上执行物理零行，对面 DP 有 2/16 行，轮换两侧，种子 13/37、固定/变长。独立 Mini-EP 实例共享同一组已加载权重，使用不同注册名，不覆盖原生层；检查每层实际输出、路由、模型/返回 dtype 和字节。该结果是层 API 验证，不能描述为完整调度器的零行执行。

性能命令使用不同调度预算（8192 Tokens、32 请求槽）。对应六个 literal 工作负载通过 `--suite performance` 另做完整四阶段零误差检查，尤其 batch32 decode 和 batch4×2048 prefill，不能用 max-seqs=4 的结果替代。

## 可复现命令

在本课已安装插件、四张可见 GPU 的环境中执行。结果目录必须新建。节点若遇 NVLS 初始化错误，应在所有对照中一致设置 `NCCL_NVLS_ENABLE=0` 并记录；它不是降低精度阈值或通信后端回退。

```bash
cd code/course13
python3 test/deepep_alignment.py --model "$MODEL_PATH" --output results/deepep-full
# 可选工具链检查，不能用于最终精度结论：
python3 test/deepep_alignment.py --model "$MODEL_PATH" --output results/deepep-smoke --smoke
# 真实零行层 API：
torchrun --standalone --nproc-per-node=4 checks/deepep_empty_rank.py \
  --model "$MODEL_PATH" --output results/deepep-empty
```

原生 golden 可经 `--reuse-completed OLD_SUITE --audit-native-reuse` 复用：只接受独立 stock MoERunner、相同权重/options/版本/工作负载/种子、完整四 Rank 原始证据；记录不可变镜像里的原生 Python/扩展源码指纹及文件时间审计。不同 Mini-EP 版本的候选通过结果一律不能复用，当前候选仍执行完整 native_check。没有原生不可变环境依据时，应运行新原生基准。

`--jobs NAME...` 仅用于分节点补齐指定任务及原生前置项，其终态明确为 `selected_jobs_passed`，必须汇总全部必需任务才能得出全量通过。完整命令和具体复用路径随报告保存。

## 首次差异

每层通过单个 all-rank failure flag 决定是否冻结重放；错误 Rank 不得先退出导致现场缺失。失败保存原始输入、router logits/IDs/weights、local expert 权重、mini 接收/返回/restore 布局、native 加权分支、owner 部分和与输出，再让所有 Rank 确认写入完成后报错。

```bash
python3 test/replay_alignment_failure.py --run FAILED_JOB --output diagnosis.json
```

重放按输入字节、local expert、FP32 weight 联结 DeepEP 去重行和 Mini-EP 展开分支；匹配不完整时不做数值结论。原生 down FP32 寄存器无法从该二进制直接观察，不把 PyTorch 推导当作原生测量。先区分路由/专家及加权舍入/通信恢复/owner 累加/跨 owner 合并，保留首个差异，不放宽阈值。

## 归约规则的证据边界

正式 Combine 显式实现：分支模型 dtype 舍入；owner 内 slot 升序 FP32 累加；owner 部分和模型 dtype 舍入；owner rank 升序 FP32 合并。独立 Reference 和 midpoint/order 回归检验这些边界，真实 DeepEP 完整验收检验其适用范围。

实际 vLLM Python/Triton 路径和 DeepEP 扩展有源码/二进制指纹；镜像未附 DeepEP CUDA 源码，不能凭当前实验推断任意版本、拓扑或模型的底层归约实现。仅对报告中通过的二进制、模型、硬件和配置作精度结论。
