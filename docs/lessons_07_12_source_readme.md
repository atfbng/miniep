# 第 1～12 课配套源码

解压后在仓库根目录执行课程命令。正文中的源码路径均相对于此目录；完整实现、Reference、基准与测试共同提供。

Python 路线使用 CUDA 版 PyTorch、NCCL 和 Triton。实际验收为 Python 3.12.3、PyTorch 2.12.0a0、CUDA 13.2、Triton 3.6.0、NCCL 2.29.7。`tests/test_advanced_lessons.py` 可在 CPU 上验证 Reference，GPU 子用例需要 CUDA/Triton。

```bash
python -m unittest tests.test_advanced_lessons -v
NCCL_NVLS_ENABLE=0 torchrun --standalone --nproc-per-node=2 examples/11_grouped_ep.py
NCCL_NVLS_ENABLE=0 torchrun --standalone --nproc-per-node=2 examples/12_stream_pipeline.py
```

第七、八课支持 EP=1/2/4；第九至十一课的独立算子示例使用一张 GPU；完整 EP 对照支持 1/2/4 GPU。FP32 比较关闭 TF32，基准脚本已经设置。GPU 运行时、首轮编译与随机种子按各课说明配置。

设备路线使用 MPI、CUDA Toolkit、NVSHMEM 3.6.5 和 CUTLASS 3.9.2。NVSHMEM 需要完整 include/lib，含 `libnvshmem_device.a` 与 Host 动态库。依赖可从 NVIDIA 官方分发获取：

- NVSHMEM：https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/libnvshmem-linux-x86_64-3.6.5_cuda13-archive.tar.xz
- SHA256：`c06d44aa15eed01299366509788daf174456d492df94c8a616766c43b819afab`
- CUTLASS：https://github.com/NVIDIA/cutlass/releases/tag/v3.9.2

第三方依赖保留在独立目录，未包含在本源码附件中。设置 `NVSHMEM_HOME`、`CUTLASS_HOME`，并把 `$NVSHMEM_HOME/lib` 加入 `LD_LIBRARY_PATH`。第十二课给出编译和启动命令。CUDA 编译目标 `sm_90` 对应 H100；Root 容器运行 MPI 时按 Open MPI 约定设置 `OMPI_ALLOW_RUN_AS_ROOT=1` 和 `OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1`。

`token_exchange.cu` 默认重复同一输入，适合准备后执行基准。`MINIEP_VARY_ROUNDS=1` 每轮改变输入幅度，验证双缓冲不会返回旧数据；结果文件包含 `input_scale`、计数和 origin。`MINIEP_DEVICE_TIMESTAMPS=1` 记录本 Block 内的四个阶段时钟，用同输入的开关对照测量采样影响。

```bash
python benchmarks/analyze_overlap.py results/lesson12-pipeline.sqlite --output results/overlap.json
```

该分析器读取 Nsight Systems 导出的 SQLite，统计同步结束的 `PIPELINE_PROFILE` 范围内 NCCL 与 GEMM 时间并集及交集。独立验收说明列出实际命令、配置、结果与适用范围。


第十二课的设备入口为 `exchange_compute_return()`，缓冲区类型为 `TokenExchangeBuffers`；`expert_mlp_row()` 计算单行 MLP。设备路径直接执行两投影 GELU MLP，结果 JSON 使用 `expert="mlp"` 标识计算类型。正文代码块与本附件源码保持一致。
