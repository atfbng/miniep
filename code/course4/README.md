# Course 4：NCCL 集合通信

Python 3.10+、支持 CUDA/NCCL 的 PyTorch，以及两张可见 NVIDIA GPU。

本目录包含本课完整实现和本地依赖副本。可以将整个 `course4/` 复制到任意位置，再从该目录启动；无需安装仓库根目录的包或保留其他课的目录。PyTorch、Triton 等第三方依赖由当前 Python 环境提供。

## 代码与示例

`test/` 保存本课可启动示例；`mini_ep/` 保存本地实现；`benchmarks/` 保存示例使用的基准与分析函数；`checks/` 保存正确性测试。实际存在的子目录按本课需要提供。`SOURCE_MAP.json` 记录代码迁移来源。

## 启动

从仓库根目录进入本课；复制出去时，直接进入复制后的本课目录。使用与驱动匹配的 CUDA 版 PyTorch 环境，再补齐本课第三方依赖。

```bash
cd code/course4
python3 -m pip install -r requirements.txt
torchrun --standalone --nproc-per-node=2 test/demo.py
```

## VS Code 调试

以仓库的 `code/` 目录打开 VS Code（部署到 `/work` 时打开 `/work/code`），选择装有 CUDA/NCCL 版 PyTorch 的 Python 解释器，并安装 Python Debugger 扩展。在“运行和调试”中选择“第4课：双 GPU NCCL 调试”，按 F5 启动。配置位于 `code/.vscode/launch.json`，通过 `torch.distributed.run` 启动两个 Worker，并开启子进程调试；需要两张可见 NVIDIA GPU。

可在 `run_collectives()` 内设置断点，查看当前进程的 `rank`、`all_reduce_output`、`gathered_rows` 和 `all_to_all_send/recv`。调用堆栈中可切换 Worker；两个 Rank 都需要继续运行，集合通信才能完成。只观察 Rank 0 时，可以给断点设置条件 `rank == 0`（放在 `rank` 赋值之后）。调试配置将通信超时设为 1800 秒，普通命令行运行仍使用默认的 120 秒。

正确性测试也从本课目录启动：

```bash
python3 -m unittest discover -s checks -v
```
