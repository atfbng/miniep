# 第 12 课可读性复核与代码整理

实际重读了 [ai-infra-book 第 5 章与第 9 章](https://github.com/bojieli/ai-infra-book)，以及[飞书参考资料](https://tvle9mq8jh.feishu.cn/wiki/IvB3wekGMirjPYkNGNAcBbSvnxG)中的第 14 章 Triton 编程模型、分块 GEMM 与生产者/消费者相关段落。文章采用具体等待问题引入、数值算例推导、主题式标题、代码前后的数据流解释；图示就近承接正文，核心机制用连贯段落讲清楚。

课程补充双卡四 Expert、Top-2 的完整分支往返例子，给出 packed/returned、restore、来源行偏移与加权结果；以此衔接四卡大矩阵、Tile 就绪计数、TMA 与 Up/Down 阶段。新增三张 Mermaid 图，与原有两张合计五张；原两张画板 token 保留。运行与验收命令集中在独立说明中，正文按知识依赖组织。飞书中的五个 Python 代码块与本地课程完全一致，均通过 AST 语法检查，关键 Triton 函数摘录与实际源码一致。

## 代码修改

`_wait()` 将 system/gpu 作用域作为编译期选择，共用一个等待循环；`_gelu_fp16()` 提取两处相同的精度转换与精确 GELU，编译时展开。主 Kernel 增加四个阶段的注释，源码统一格式、参数换行和索引运算间距。数学运算、通信顺序、缓冲区生命周期、Tile 配置和启动参数保持一致。其余函数与类的 AST 与原源码相同。

## H100 验证

重新执行 7 组性能配置与 10 组边界配置，共 50 个 Rank 配置，同步基线、Triton 分离与融合全部对齐 Reference。覆盖 EP=1/2/4、Top-1/2、空 Expert、单 Token、倾斜路由、非整 Tile、非均匀权重、连续排队调用和输入更新。最大绝对误差 0.000251770，容差沿用 atol=0.003、rtol=0.02。执行脚本为附件 sweep.sh、edges.sh；完整输出为 revalidation.log。

双卡、E=8、N=1024、H=4096、F=14336 的同输入整理前后对照，各取 60 个最大 Rank CUDA Event 样本的中位数：原版 0.906064 ms，整理后 0.911568 ms，相差 0.61%。两 Rank 的主 Kernel PTX 在排除调试行号、文件声明与调试标签后完全一致；寄存器数 255、共享内存 147480 bytes、spill 数 182 均一致。这次整理没有增加主 Kernel 的可执行 PTX 指令或编译资源开销；本次单独对照的延迟差为约 0.6%，不能用这两个顺序运行的样本组断言严格零性能差。

| EP / E | 原测融合 / ms | 整理后融合 / ms |
|---|---:|---:|
| 2 / 8 | 0.897 | 0.906 |
| 4 / 8 | 1.013 | 0.997 |
| 4 / 16 | 1.017 | 1.003 |
| 4 / 32 | 1.126 | 1.128 |


上表给出原性能实测与本次回归值，两次测量发生在共享服务器的不同时间，适合检查数量级与回归情况。课程保留原有完整测量表，未把本次的单次更快结果写成额外加速结论。PTX 等价检查覆盖上述双卡大矩阵特化；多卡及边界特化通过 Reference 和性能回归执行验证。

软件与 GPU 条件沿用原报告。本次日志中的 multicast 初始化警告随后回退到对端内存路径，全部配置均完成数值验证。未重新采集 Nsight 时间线；Kernel 数与通信调度未改变，原时间线作为原版证据保留。内部通信/计算重叠率仍为 N/A，原因见原报告的 GPU Metrics 权限说明。

## Stream 示例运行入口

以下为从课程正文移入的运行命令，输入及启动方式与原已验证版本一致。本次未修改或重新执行 Stream 实现。

```bash
NCCL_NVLS_ENABLE=0 torchrun --standalone --nproc-per-node=2 \
  examples/12_stream_pipeline.py --tokens 2048 --hidden-size 512 \
  --expert-hidden-size 1024 --chunk-size 256 --profile \
  --output-dir results/pipeline

NCCL_NVLS_ENABLE=0 nsys profile --sample=none --cpuctxsw=none \
  --trace=cuda,nvtx,osrt --output=results/lesson12-pipeline \
  torchrun --standalone --nproc-per-node=2 examples/12_stream_pipeline.py \
  --tokens 2048 --hidden-size 512 --expert-hidden-size 1024 \
  --chunk-size 256 --profile --output-dir results/pipeline-profile

```

## 证据与源码

附件包含本次实际运行的 Triton 源码、基准依赖、课程正文、脚本、原始 JSON、整理前后 PTX、编译资源记录与 SHA256 清单。原始性能证据和当前可读性版本分别保留，避免将旧源码哈希用于描述新文件。源码 SHA256：`5fb4d18c3638673fd233be242821a0513ced5ac02058cb31f94b5d3e5bd2fb3a`。
