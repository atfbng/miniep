# 源码与执行校验清单

正式核心只保留一套 DeepEP 合并实现；历史文档和不可变快照不参与运行。结果报告必须关联冻结源码 archive、实际 Python 文件 SHA、模型 SHA 清单、四 Rank 实际后端/传输 proof，而非仅 Git HEAD。

| 检查 | 实际位置/命令 | 证明范围 |
|---|---|---|
| 课程独立、导入、路径、依赖和双 SOURCE_MAP | `python3 tools/check_course_layout.py --courses 13` | 静态源码完整性；不等于 GPU 验证 |
| 输入 dtype 默认与 FP16 模型拒绝 | `model/moe.py::ep_forward`、`model/contracts.py`、`checks/test_return_transport.py` | FP32/BF16 核心；集成 BF16；传输诊断不扩展模型 dtype |
| FP32 down / 权重 / 转换顺序 | `kernels/grouped_gemm/triton_impl.py::grouped_gemm_kernel`、`reference/moe.py` | 原生 FP32 accumulator、FP32 权重、最终 store；没有先舍入未加权 down |
| 每分支一次加权 | `ep_forward` / `combine_triton` / `combine_reference` 接口 | Combine 不接收权重；无 source-weighting 分支 |
| 原生兼容路径移除 | `mini_ep/integration/reduction.py` 删除；核心无旧后端 selector/owner-partial 输出/兼容 collective | 必要 TP 输入输出重组仍位于 integration |
| 布局与权重身份 | CPU Reference、`test_reference`、`test_kernels` | int32/int64 IDs、strided、stable packing、fixed padding、restore_index |
| Grouped 算术保持 | 与整理前快照比较 `tl.dot(a,b,acc)`、tile 配置、3 stages、fusion 选项 | 函数提取/plan 封装不授权改变累加顺序 |
| 四 Rank 实际零行 | `checks/deepep_empty_rank.py` | 真实 48 层 DeepEP API，而非调度器 dummy 行推断 |
| 全模型路由/层/logits/token exact | `test/deepep_alignment.py` 和所有 Rank acceptance | 只接受逐元素相等、finite、完整执行；独立原生重复运行 |
| 最终返回 buffer | `return_buffer_dtype` 元信息和 benchmark 启动 NCCL trace | 实际 buffer，非环境变量配置值 |
| 性能测量污染 | `benchmarks/deepep_plugin.py`、environment/transport/backend proofs | 启动钩子移除；profiling 与主要计时独立 |

`SOURCE_MAP.json` 与 `code/source_map.json["13"]` 对当前实体文件记录 SHA；历史文件只做历史文本校验。课程衔接见 TEACHING，逐项实际执行结果见最终可复现报告，不把本清单本身当作通过记录。

上一轮完整 GPU 数值矩阵使用不可变 v6 快照。当轮交付源码在其后只修正了两个核心 wrapper 的 docstring（fixed restore 索引名称、GroupedGemmPlan 返回说明）；24 个核心 Python 文件去除 docstring 后的执行 AST 全部一致，函数签名、默认值、常量、Kernel 和通信算法均未改变。两套逐文件 SHA 与 AST 审计保存在报告 artifacts，不把文档字符串变更冒充新的数值实验。外围清理删除旧计时入口、直接使用 DeepEP 验证模式，并要求 auto 返回请求；两种模式的实际启动环境已与受验版本逐字段核对一致，最终入口另做启动回归。

本轮继续删除已证明未使用的字段、转发包装和初始化约束下不可达的分支，详见 [CORE_CLEANUP.md](CORE_CLEANUP.md)。其源码审计逐项约束允许的变化，不将本轮改动误标为仅 docstring 变更；新增 GPU/精度/计时结果与原始矩阵分别保存。
