# 第二次 All-to-All 的返回类型与加权

`model/moe.py::ep_forward(..., return_dtype=None)` 默认令返回 buffer dtype 跟随 `tokens.dtype`，FP32 输入回传 FP32，BF16 输入回传 BF16。集成 BF16 模型正常路径回传 BF16，没有用 FP32 默认换取精度。

`DISPATCH` 把原始 `[N,K]` 的 FP32 routing weights 与 Token/Expert ID 一同稳定打包；A2A1 发送三个同序 payload。接收端为 source-major 展开的分支，Expert-major 排列同时用于 Token 和权重。

Grouped `grouped_expert_mlp` 的 down GEMM 用 FP32 accumulator；写出前乘 FP32 权重，再转换成传输 dtype。`fuse_return_weight=False` 先写独立 FP32 down tensor，再由 PyTorch 乘 FP32 权重和转换；逐专家 loop 同样保留该顺序。独立 PyTorch 实现在 `reference/expert.py`、`reference/moe.py`，不调用 Triton。

禁止先把未加权 down 结果转低精度。Combine 仅接收已加权分支，接口没有权重参数，不能重复乘权重。

```python
weighted = (down_fp32 * routing_weight_fp32).to(return_dtype)
# 不同于 (down_fp32.to(return_dtype).float() * weight).to(return_dtype)
```

`return_dtype=torch.float16/torch.bfloat16/torch.float32` 和集成环境变量 `MINIEP_RETURN_DTYPE=fp16/bf16/fp32` 仅用于诊断。FP16 回传可能溢出；它与完整 FP16 模型是不同实验。核心仍明确拒绝 FP16 模型激活，即使显式指定 FP32 回传、即使关闭深入输入校验。

第二次通信的 send buffer 是 `[R,H]`、接收 buffer 为原 rank-packed 行布局，split 单位是行并反转 A2A1 的 send/recv。`return_send_bytes` 来自实际发送 tensor 的 `numel()*element_size()`；`return_buffer_dtype` 来自实际接收 tensor。固定容量 padding 也计入实际发送字节，额外 FP32 权重通信单独记录，不能只用有效 Token 数推算性能。

正式验收对照仅为 DeepEP。分支和 owner 部分和保留模型 dtype 舍入，owner 内按 slot 顺序 FP32 累加，最终按 EP rank 升序 FP32 合并。该规则的适用范围以完整验收记录为准，不推广到其他 DeepEP 版本、量化格式或硬件。

独立回归与消融：

```bash
torchrun --standalone --nproc-per-node=4 checks/return_transport.py --output results/return-ep4
torchrun --standalone --nproc-per-node=4 benchmarks/benchmark_ep.py   --output results/core-stages.json --return-dtype auto
# 同一核心计算的非融合诊断：添加 --no-fuse-return-weight。
```

原先 source 加权及原生 owner 兼容策略已删除；历史矩阵和对应源码快照仅在历史资料及原始 artifacts 中保留。[RETURN_TRANSPORT_RESULTS.json](RETURN_TRANSPORT_RESULTS.json) 是旧实验数据，不是当前整模型验收。
