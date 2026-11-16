# Course 12：Triton 通信与计算

CUDA 版 PyTorch、Triton、NumPy，单机至少两张支持 Peer 访问与跨 GPU 原子操作的 NVIDIA GPU。PyTorch 需提供 torch.distributed._symmetric_memory，Triton 需提供 triton.tools.tensor_descriptor.TensorDescriptor；TMA 路径要求计算能力至少为 9.0。使用相互兼容的 PyTorch/CUDA/Triton 构建。

本目录包含本课完整实现和本地依赖副本。可以将整个 `course12/` 复制到任意位置，再从该目录启动；无需安装仓库根目录的包或保留其他课的目录。PyTorch、Triton 等第三方依赖由当前 Python 环境提供。

## 代码与示例

`test/` 保存本课可启动示例；`mini_ep/` 保存本地实现；`benchmarks/` 保存示例使用的基准与分析函数；`checks/` 保存正确性测试。实际存在的子目录按本课需要提供。`SOURCE_MAP.json` 记录代码迁移来源。

## 启动

从仓库根目录进入本课；复制出去时，直接进入复制后的本课目录。使用与驱动匹配的 CUDA 版 PyTorch 环境，再补齐本课第三方依赖。

```bash
cd code/course12
python3 -m pip install -r requirements.txt
torchrun --standalone --nproc-per-node=2 test/demo.py --tokens 17 --hidden-size 128 --expert-hidden-size 256 --experts 8 --weighted --output results/peer
```

正确性测试也从本课目录启动：

```bash
python3 -m unittest discover -s checks -v
```

主示例在内存中生成输入、路由和权重，校对独立 Reference，再比较同步 NCCL、Triton 分离执行与融合路径。`--tokens` 是每 Rank 输入行数，`--experts` 是全组 Expert 数；性能结论需结合生成的各 Rank JSON 和实际设备分析。

Stream Pipeline 扩展示例位于 `test/stream_pipeline.py`：

```bash
torchrun --standalone --nproc-per-node=2 test/stream_pipeline.py --chunk-size 64
```

`extensions/nvshmem/` 保存已有 NVSHMEM/CUDA 扩展源码，`test/nvshmem_reference.py` 和对应基准用于比较这些扩展生成的结果文件。该扩展另需 NVSHMEM、MPI、CUDA Toolkit 和 CUTLASS 开发环境；这些第三方库由外部安装，不从其他 course 目录导入。Triton 主示例无需这些扩展或预生成数据文件。
