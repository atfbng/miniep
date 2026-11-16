# Mini-EP 按课独立源码

`course0` 是环境检查，`course1～course12` 与原课程编号一致，`course13` 增加 vLLM/Qwen MoE 接入扩展。每课有完整的本地实现副本，`test/` 是本课示例，`checks/` 是正确性测试；各课之间没有源码导入或符号链接。

| 目录 | 主题 | 主要示例 |
| --- | --- | --- |
| [course0](course0/README.md) | 环境与启动方式 | `course0/test/check_environment.py` |
| [course1](course1/README.md) | 单设备 MoE Reference | `course1/test/demo.py` |
| [course2](course2/README.md) | Router、Dispatch 与 Combine | `course2/test/demo.py` |
| [course3](course3/README.md) | GPU 执行与 GEMM 计时 | `course3/test/demo.py` |
| [course4](course4/README.md) | NCCL 集合通信 | `course4/test/demo.py` |
| [course5](course5/README.md) | 变长 All-to-All | `course5/test/demo.py` |
| [course6](course6/README.md) | Mini-EP V1 | `course6/test/demo.py` |
| [course7](course7/README.md) | EP 扩展与固定全局负载 | `course7/test/demo.py` |
| [course8](course8/README.md) | 负载分析与 Profiling | `course8/test/demo.py` |
| [course9](course9/README.md) | Triton 稳定 Dispatch | `course9/test/demo.py` |
| [course10](course10/README.md) | Triton 融合 Combine | `course10/test/demo.py` |
| [course11](course11/README.md) | Grouped GEMM 与 Expert MLP | `course11/test/demo.py` |
| [course12](course12/README.md) | Triton 通信与计算 | `course12/test/demo.py` |
| [course13](course13/README.md) | vLLM 接入与 Qwen MoE 精度对齐 | `course13/test/demo.py` |

进入一课后，按该课 README 安装第三方依赖并启动。每次在独立 Python 进程中运行一课。`mini_ep` 和 `benchmarks` 都解析到当前课的本地包。
