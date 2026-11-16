# 去重固定启用：NCCL 与 Peer/TMA 解耦验收

Runner 默认使用 NCCL 去重。`MINIEP_TRANSPORT=peer` 仅切换传输方式，仍固定去重；不提供去重开关。旧 `peer_dedup` 值兼容映射为 `peer`。NCCL 去重不导入 symmetric memory、Peer 或 TMA 模块；本地计算复用原 Grouped GEMM / SwiGLU。

| 验证 | rank级记录 | 结果 |
|---|---:|---|
| CPU/Gloo EP2 | 180 | 与独立 PyTorch Reference 完全一致 |
| CPU/Gloo EP4，含交错rank子组 | 432 | 与独立 PyTorch Reference 完全一致 |
| H100 NCCL EP2，`NCCL_P2P_DISABLE=1` | 360 | 与旧逐分支 NCCL 实现逐元素一致 |
| H100 NCCL EP4，含交错rank子组 | 864 | 与旧逐分支 NCCL 实现逐元素一致 |
| H100 NCCL vs Peer/TMA 去重 | 12 | 逐元素一致，含空来源和全部为空 |

GPU 数值测试与独立 PyTorch Reference 的最大绝对误差为0.000244140625，在检查设定的 FP32/BF16/返回 dtype 容差内；同一 Grouped GEMM 的旧逐分支实现对照全部逐元素相等。GPU合计1236个rank级记录，CPU语义合计612个。重复配置、不同rank不是独立统计实验，不用于计算性能显著性。

另有13个CPU单元检查通过：去重打包与独立循环Reference、dtype/device/contiguous/slot恢复、FP32/BF16/FP16分支舍入与owner顺序、空输入、传输默认值/兼容别名/延迟导入、EP2/EP4配置和核心依赖边界。默认 NCCL 模块在阻断 Peer/TMA 导入的检查中可正常导入。

GPU使用同节点四张H100 80GB HBM3，镜像 `sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b`。框架环境沿用之前的 Torch2.13.0+cu130、Triton3.7.1；CPU语义测试为本机Torch2.2.1+cpu。随机种子、配置和容差在检查脚本内固定。该轮只验证正确性与实现依赖，没有新测性能，也没有在实体3090/4090上运行。

覆盖 FP32/BF16，H=32/128、F=48/128、E=8、Top-1/2/8，uneven/empty_source/all_empty/single/hot_owner路由，以及auto/FP32/FP16分支返回舍入；EP4中额外创建[0,2]与[1,3]子组，验证EP rank与global rank不同的情况。Peer精确对照使用BF16/H128/F128/Top8。

## 命令

从routeep目录运行：

```bash
NCCL_P2P_DISABLE=1 torchrun --standalone --nproc-per-node=2 checks/nccl_dedup.py --output results/no-p2p-ep2
torchrun --standalone --nproc-per-node=4 checks/nccl_dedup.py --peer-smoke --output results/nccl-peer-ep4
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=2 checks/nccl_dedup.py --cpu --output results/cpu-ep2
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=4 checks/nccl_dedup.py --cpu --output results/cpu-ep4
```

原始逐rank记录见 `raw-correctness.tgz`，机器可读汇总见 `summary.json`，本轮GPU执行时的源码快照见 `measured-source.tgz`。快照用于对应这次核心数值检查，后续文档及测试整理以当前仓库为准。旧报告中的“原NCCL”“仅Peer”测量仍对应当时冻结代码，不能当作新NCCL去重路径的性能。
