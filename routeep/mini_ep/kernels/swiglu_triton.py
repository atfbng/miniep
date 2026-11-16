"""课程风格的分块 GEMM 与 SwiGLU，不依赖 vLLM，也不做自动调优。

PyTorch 对照实现位于 ``reference/expert.py::swiglu_reference``。两者使用
相同的存储舍入边界：gate/up 投影、SiLU 和乘法按模型 dtype（BF16/FP32）
舍入，down 在 FP32 中累加。路由与通信都在这些 Kernel 之外完成。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _linear(
    X,
    W,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """作用：计算一个 BM×BN 的 ``X @ W.T`` 输出块。

    参数：
        X: 连续输入 ``[M,K]``。
        W: 连续权重 ``[N,K]``。
        Y: 连续输出 ``[M,N]``。
        M: 输入行数，运行时标量。
        N: 输出列数，编译期常量。
        K: 归约维长度，编译期常量。
        BM: 行 tile 大小。
        BN: 列 tile 大小。
        BK: K 维分块大小。
    """
    # X [M,K]，W [N,K]，Y [M,N]；三者都是连续的行主序 Tensor。
    # 每个 program 计算 Y 的一个 BM×BN 块；沿 K 分块累加 X @ W.T。
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + kk
        # a 为 X 的 [BM,BK] 分块；越界行和列用 0 填充。
        a = tl.load(
            X + rows[:, None] * K + k[None, :],
            (rows[:, None] < M) & (k[None, :] < K),
            0,
        )
        # W 存储为 [N,K]；这里按 [K,N] 组织指针，等价于读取 W.T，
        # 因此不需要额外生成转置副本。
        b = tl.load(
            W + cols[None, :] * K + k[:, None],
            (cols[None, :] < N) & (k[:, None] < K),
            0,
        )
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(
        Y + rows[:, None] * N + cols[None, :],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _swiglu(
    GateUp,
    Activated,
    ValidStart,
    M,
    F: tl.constexpr,
    BLOCK: tl.constexpr,
    MASK_PADDING: tl.constexpr,
):
    """作用：对 gate/up 缓冲执行 SwiGLU 激活。

    参数：
        GateUp: 连续输入 ``[M,2F]``，前 F 列为 gate，后 F 列为 up。
        Activated: 连续输出 ``[M,F]``。
        ValidStart: 可选设备端 int64 向量；MASK_PADDING=True 时其首元素
            是首个有效行。
        M: 行数，运行时标量。
        F: 中间维度，编译期常量。
        BLOCK: 每个 program 处理的扁平元素数。
        MASK_PADDING: 是否跳过首个有效行之前的 padding。
    """
    # 输入 [M,2F] 的前半为 gate、后半为 up；输出 [M,F]。
    # 保持 SiLU 的 BF16 舍入点，避免编译器把激活与乘法错误融合。
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offsets // F, offsets % F
    valid = offsets < M * F
    if MASK_PADDING:
        # 稳定专家排序会把 -1 padding 放在 offsets[0] 之前。这里绝不读取
        # 那些未初始化的 GEMM 行，而是通过掩码加载直接产生 0。
        valid = valid & (row >= tl.load(ValidStart))
    # gate/up 分别位于每行 [0,F) 和 [F,2F) 区间。
    gate = tl.load(GateUp + row * (2 * F) + col, valid, 0).to(tl.float32)
    up = tl.load(GateUp + row * (2 * F) + F + col, valid, 0).to(tl.float32)
    # 先将 SiLU 舍入到输入 dtype，再做乘法；避免融合后改变 BF16 舍入点。
    silu = (gate / (1.0 + tl.exp(-gate))).to(GateUp.dtype.element_ty).to(tl.float32)
    activated = silu * up
    tl.store(Activated + offsets, activated, offsets < M * F)


def linear_triton(
    x: torch.Tensor, weight: torch.Tensor, *, output_dtype: torch.dtype
) -> torch.Tensor:
    """作用：执行一次 ``X @ W.T`` 的 Triton 分块 GEMM。

    参数：
        x: 连续 CUDA 输入 ``[M,K]``，dtype 为 BF16/FP32。
        weight: 连续 CUDA 权重 ``[N,K]``，dtype 必须与 ``x`` 相同。
        output_dtype: 输出 dtype，只支持 FP32/BF16；它决定写回时的
            存储舍入精度。

    返回：
        连续张量 ``[M,N]``。矩阵乘法在 FP32 中累加；固定使用
        16x64x32 分块，保持索引映射清晰，不包含自动调优策略。

    本函数不做路由、通信或进程组管理，只计算单个专家的一次投影。
    """
    # 校验矩阵维度：x 和 weight 的 K 维必须一致。
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError(f"incompatible linear shapes {x.shape}, {weight.shape}")
    if (
        x.shape[1] < 1
        or weight.shape[0] < 1
        or output_dtype not in (torch.float32, torch.bfloat16)
    ):
        raise ValueError(
            "linear dimensions must be positive and output dtype FP32/BF16"
        )
    if (
        not x.is_cuda
        or weight.device != x.device
        or x.dtype != weight.dtype
        or x.dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError("linear_triton requires same-device CUDA BF16/FP32 operands")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("linear_triton requires contiguous row-major operands")
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=output_dtype)
    if m:
        # 网格第一维覆盖 M 方向的 16 行块，第二维覆盖 N 方向的 64 列块。
        _linear[(triton.cdiv(m, 16), triton.cdiv(n, 64))](
            x, weight, out, m, n, k, 16, 64, 32, num_warps=4
        )
    return out


def swiglu_triton(x: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """作用：对单个本地专家执行 SwiGLU MLP。

    参数：
        x: 该专家负责的 Token 行 ``[M,H]``，dtype 为 BF16/FP32。
        w13: gate/up 合并权重 ``[2F,H]``；前 F 行为 gate，后 F 行为 up。
        w2: down 权重 ``[H,F]``。

    返回：
        FP32 ``[M,H]``。返回 FP32 是为了让后续 Combine 在应用路由权重
        前保留更高精度的专家结果。

    Top-K 已经展开，每一行只属于当前专家。本函数不应用路由权重、不
    发起集合通信、不持有持久缓冲，也不初始化进程组。
    """
    if (
        x.ndim != 2
        or w13.ndim != 2
        or w2.ndim != 2
        or w13.shape[0] % 2
        or w2.shape != (x.shape[1], w13.shape[0] // 2)
    ):
        raise ValueError("expected expert weights [2F,H] and [H,F] for tokens [M,H]")
    # 两次 GEMM：H->2F，SwiGLU 后 F->H；最后保留 FP32，交给 Combine 加权。
    gate_up = linear_triton(x, w13, output_dtype=x.dtype)
    activated = swiglu_activation_triton(gate_up)
    return linear_triton(activated, w2, output_dtype=torch.float32)


def swiglu_activation_triton(
    gate_up: torch.Tensor, valid_start: torch.Tensor | None = None
) -> torch.Tensor:
    """作用：对连续的 ``[M,2F]`` gate/up 缓冲应用 SwiGLU。

    参数：
        gate_up: 连续 CUDA 张量 ``[M,2F]``，dtype 为 BF16/FP32；
            每行前 F 列为 gate，后 F 列为 up。
        valid_start: 可选的同设备连续 int64 向量，其首元素位于 ``[0,M]``。
            Grouped 路径传入 ``expert_offsets``；首元素之前的打包行是
            padding，应输出 0 且不读取未初始化的 GEMM 结果。

    返回：
        模型 dtype 的连续张量 ``[M,F]``。输入输出都保持模型 dtype；
        SiLU 在 gate/up 相乘前先舍入，以保持与 Reference 一致的数值行为。

    该函数与 :func:`swiglu_triton` 使用同一个激活实现，单独暴露后可以让
    Grouped 路径一次启动处理所有专家行。``valid_start`` 由调用方的 GPU
    规划器在同一 Stream 上先生成，不产生 Host 回读。
    """
    # 检查 gate/up 的二维形状，确保 2F 能对半拆分。
    if gate_up.ndim != 2 or gate_up.shape[1] < 2 or gate_up.shape[1] % 2:
        raise ValueError(f"gate_up must have shape [M,2F], got {tuple(gate_up.shape)}")
    if (
        not gate_up.is_cuda
        or not gate_up.is_contiguous()
        or gate_up.dtype not in (torch.float32, torch.bfloat16)
    ):
        raise ValueError("activation requires contiguous CUDA BF16/FP32 gate/up rows")
    # valid_start 必须是小型的设备端 int64 向量，才能直接传给 Triton Kernel。
    if valid_start is not None and (
        valid_start.device != gate_up.device
        or valid_start.dtype != torch.int64
        or valid_start.ndim != 1
        or valid_start.numel() < 1
        or not valid_start.is_contiguous()
    ):
        raise ValueError(
            "valid_start must be a nonempty contiguous int64 vector on the input device"
        )
    m, twice_f = gate_up.shape
    f = twice_f // 2
    activated = torch.empty((m, f), device=gate_up.device, dtype=gate_up.dtype)
    if m:
        # 输出元素总数为 M*F，每个 program 处理 256 个扁平元素。
        _swiglu[(triton.cdiv(m * f, 256),)](
            gate_up,
            activated,
            valid_start if valid_start is not None else gate_up,
            m,
            f,
            256,
            valid_start is not None,
        )
    return activated
