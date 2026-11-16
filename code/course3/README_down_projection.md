# 下投影调优

`mini_ep/kernels/down_projection.py` 保留原有 14 组候选，再联合搜索
`BM=64/128、BN=128/256、BK=32/64/128、warps=4/8、stages=2/3/4`，
去重后按当前设备和形状选择。包括 `128×128×64 / 4 warps / 4 stages`。
上投影和 PyTorch GELU 沿用原版，只改变下投影的候选集合。

输入是连续 CUDA 张量 `X[M,F]`、`W[H,F]`，输出为连续 `[M,H]`，
保持输入的行序、dtype 和 device。支持 FP16、BF16、FP32，内部 FP32 累加。
原实现仍可通过 `triton_gemm_tuned()` 和 `expert_mlp_triton(..., tuned=True)` 对照。

在本课目录中用一张 GPU 运行：

```bash
python3 test/demo_down_projection.py --tokens 512 2048 \
  --hidden-size 1024 --expert-hidden-size 2048 --dtype float16 \
  --warmup 100 --repeats 100 --rounds 3 --graph-calls 64 \
  --output results/down_projection.json
python3 -m unittest discover -s checks -p 'test_down_projection.py' -v

# 逐项测量所有候选，比较建议配置、原配置和 PyTorch。
python3 benchmarks/benchmark_down_search.py --tokens 512 2048 \
  --hidden-size 1024 --expert-hidden-size 2048 --dtype float16 \
  --warmup 100 --repeats 50 --rounds 3 --graph-calls 64 \
  --output results/down_search.json
```

三个版本使用相同输入，先验证数值与输出属性，再捕获各自的 CUDA Graph。
每次回放执行 64 次操作，CUDA Event 耗时除以 64；三轮交替测量，报告中位数。
单独下投影与完整 MLP 均比较 PyTorch、原候选和扩展候选。JSON 保存实际选定参数、
环境和所有样本。不同形状会选择不同参数。

逐项扫描报告额外保存每个候选的寄存器、共享内存、溢出数和输出 Tile 数；
资源不足的配置记录原因并跳过，各轮随机测量次序。扫描固定复用输出缓冲区，
用于比较 Kernel 执行；完整 MLP 的收益由 `demo_down_projection.py` 再验证。

## 先前的小范围扫描

先前 H100 的 FP16 `M=2048、N=1024、K=2048` 扫描中，
原先选定的 `64×256×64 / 8 / 3` 为 15.53 μs，
`128×128×64 / 4 / 4` 为 15.13 μs，耗时降低约 2.6%；
同轮 PyTorch 为 14.89 μs。共享内存由 120 KiB 增至 128 KiB，
这次提升来自分块与执行配置的变化，不能归因于共享内存减少。

重新自动调优后，M=2048 的完整 MLP 为原候选 83.50 μs、扩展候选 80.11 μs；
M=512 没有获得提升。测量时 H100 有其他负载，保留原实现和可选路径，
实际使用时可用上面的命令对当前批次进行比较。

## 扩展到 80 组配置，再复测前几名

联合搜索 80 组配置；M=512 和 2048 均有 70 组通过数值与输出属性检查，
10 组因共享内存不足跳过。每个形状取前 5 名，加上原配置与建议配置，再做短复测。
各轮随机测量次序，参数按形状选择并缓存。

```bash
python3 benchmarks/benchmark_down_search.py --tokens 512 2048 \
  --output results/down_search.json
python3 benchmarks/benchmark_down_search.py --tokens 512 2048 \
  --shortlist-from results/down_search.json --top-k 5 \
  --warmup 100 --repeats 100 --rounds 3 --graph-calls 64 \
  --output results/down_shortlist.json
```

短复测使用物理 GPU 2，UUID 为 `d0ff3a32-2144-34e3-0edb-60994decdec3`，
监控期间仅有测量进程。每个配置预热 100 次，64 次运算捕获到 Graph，
三轮各测量 100 次回放，报告 300 个样本的中位数。

| M | 配置 BM×BN×BK / warps / stages | 下投影耗时 |
|---|---|---:|
| 2048 | PyTorch | 14.67 μs |
| 2048 | 原配置 64×256×64 / 8 / 3 | 15.44 μs |
| 2048 | 建议配置 128×128×64 / 4 / 4 | 15.42 μs |
| 2048 | 本轮最快候选 64×256×64 / 4 / 3 | 15.04 μs |
| 512 | PyTorch | 6.87 μs |
| 512 | 原配置 64×256×64 / 8 / 3 | 14.61 μs |
| 512 | 建议配置 128×128×64 / 4 / 4 | 14.11 μs |
| 512 | 本轮最快候选 64×128×128 / 4 / 3 | 9.79 μs |

M=2048 的建议配置与原配置基本持平；最快候选降低约 2.6%，但各轮仍有波动。
M=512 的最快候选相对原配置降低约 33%，需要另一组 Tile 与 BK。
因此，分块、Warp 数和流水深度一起搜索，分别为每个形状选参数。
建议配置的共享内存为 128 KiB，原配置为 120 KiB，两者均无寄存器溢出。
本表比较独立下投影；扩展候选的完整 MLP 数值与输出属性检查也已通过，
完整 MLP 的性能收益需独立测量。
