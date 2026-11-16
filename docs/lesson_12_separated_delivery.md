# 第 12 课 Triton 分离版交付说明

## 修改范围

原知识库节点已改为《第 12 课：Triton 分离执行——从 Token 通信到 Expert 计算》：
https://tvle9mq8jh.feishu.cn/wiki/WCOQw8Oe8iDvQUkl5gjcXS1Onke

课程首页的标题与简介同步更新。正文按执行边界、Token 布局、输入发送与就绪、Grouped GEMM、回传与合并、轮次与复用、完整前向、性能组织，共八节。融合版的常驻通信 CTA、Cooperative Grid 和核内全网格栅栏说明已移除。五个原画板保留资源 token，内容改为分离版流程、Top-2 往返、Expert 拼接、Tile 就绪和缓冲区生命周期。

本地正文保持原文件路径 `docs/lesson_12_communication_compute_fusion.md`，便于已有引用继续访问；标题和内容均已替换。可执行库、测试和 benchmark 源码本次未改动。八段代码只增加中文注释，AST 与对应源码完全一致。

本次实际复读了已有缓存的 GitHub ai-infra-book 第五章相关内容，以及飞书风格参考的 Triton 入门和进阶生产消费相关段落，按具体问题、数据布局、机制、源码和系统影响组织文章。

## 本次验证

执行命令：

```bash
python3 artifacts/lesson12_separated_20261007/build.py
python3 artifacts/lesson12_separated_20261007/verify.py
```

结果：八段摘录均通过 Python 语法解析，且与 H100 已验证版本的函数或方法 AST 一致。模块 SHA256 保持为 `5fb4d18c3638673fd233be242821a0513ced5ac02058cb31f94b5d3e5bd2fb3a`。四组性能值按各 Rank 原始样本逐次取最大值，再计算中位数；表内延迟及降幅均匹配。飞书回读检查通过：八个一级章节、八个代码块、三张表、五个原画板；表宽统一为 820，标题与正文一致。

已下载检查画板。第五幅轮次图的首次时序图预览未正确显示，改为流程图后重新验证。代码标题包含主要功能及源码路径，源码中保留函数名或类名及必要中文注释。

## H100 验证来源

本次为课程改写，未重新运行 GPU benchmark。使用同一源码 SHA256 对应的 H100 验证结果：17 组配置、50 个 Rank 配置，覆盖 EP=1/2/4、Top-1/2、空 Expert、单 Token、倾斜路由、非均匀权重、H=136/F=264、跨来源 Tile、连续排队调用及更新输入。记录的最大误差为 0.00025177001953125，容差 atol=0.003、rtol=0.02。

证据位置：

- `artifacts/lesson12_reader_review_20261007/verification.json`：源码 SHA256 与 H100 汇总。
- `artifacts/lesson12_triton_20261007/results/triton-fusion/readability-final/`：本课性能表的原始 Rank 样本及运行配置。
- `artifacts/lesson12_triton_20261007/results/triton-fusion/profile/ep4-e16.nsys-rep`：已有 Timeline，常规分离路径每次七个 Kernel。
- `artifacts/lesson12_separated_20261007/verification.json`：本次摘录、结构与性能表复核结果。

不提供本次新增的 exposed communication time 或 overlap ratio：本次没有新增 Timeline 测量。课程性能结论对应已测量的完整前向区间，并注明准备、路由和打包不在区间内。没有需要用户补充的信息。
