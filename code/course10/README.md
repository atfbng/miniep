# Course 10：Triton 融合 Combine

Python 3.10+、CUDA 版 PyTorch、Triton，以及一张可见 NVIDIA GPU。

本目录包含本课完整实现和本地依赖副本。可以将整个 `course10/` 复制到任意位置，再从该目录启动；无需安装仓库根目录的包或保留其他课的目录。PyTorch、Triton 等第三方依赖由当前 Python 环境提供。

## 代码与示例

`test/` 保存本课可启动示例；`mini_ep/` 保存本地实现；`benchmarks/` 保存示例使用的基准与分析函数；`checks/` 保存正确性测试。实际存在的子目录按本课需要提供。`SOURCE_MAP.json` 记录代码迁移来源。

## 启动

从仓库根目录进入本课；复制出去时，直接进入复制后的本课目录。使用与驱动匹配的 CUDA 版 PyTorch 环境，再补齐本课第三方依赖。

```bash
cd code/course10
python3 -m pip install -r requirements.txt
python3 test/demo.py --tokens 257 --hidden-size 128 --num-experts 8 --top-k 2
```

正确性测试也从本课目录启动：

```bash
python3 -m unittest discover -s checks -v
```
