# Course 0：环境与启动方式

Python 3.10+ 与 PyTorch；查看 CPU 信息无需 GPU。NCCL 检查需要两张可见 NVIDIA GPU。

本目录包含本课完整实现和本地依赖副本。可以将整个 `course0/` 复制到任意位置，再从该目录启动；无需安装仓库根目录的包或保留其他课的目录。PyTorch、Triton 等第三方依赖由当前 Python 环境提供。

## 代码与示例

环境检查实现位于 `test/check_environment.py`，第三方依赖记录在 `requirements.txt`。本课的示例直接调用 PyTorch，无需额外的本地库。

## 启动

从仓库根目录进入本课；复制出去时，直接进入复制后的本课目录。使用与驱动匹配的 CUDA 版 PyTorch 环境，再补齐本课第三方依赖。

```bash
cd code/course0
python3 -m pip install -r requirements.txt
python3 test/check_environment.py
```

两张 GPU 的 NCCL 检查在本目录执行：

```bash
torchrun --standalone --nproc-per-node=2 test/check_environment.py --nccl
```
