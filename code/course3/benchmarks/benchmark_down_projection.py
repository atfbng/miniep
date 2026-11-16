"""单 GPU：原调优、扩大候选的下投影与 PyTorch，使用相同 CUDA Graph 计时。"""
from __future__ import annotations
import argparse
import json
import platform
import statistics
from pathlib import Path
import torch


def capture(operation, calls: int, warmup: int):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            operation()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(calls):
            operation()
    torch.cuda.current_stream().wait_stream(stream)
    return graph


def measure(graph, calls: int, repeats: int):
    for _ in range(10):
        graph.replay()
    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
             for _ in range(repeats)]
    for start, end in pairs:
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) * 1000 / calls for start, end in pairs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', nargs='+', type=int, default=[2048])
    parser.add_argument('--hidden-size', type=int, default=1024)
    parser.add_argument('--expert-hidden-size', type=int, default=2048)
    parser.add_argument('--dtype', choices=['float16', 'bfloat16', 'float32'], default='float16')
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--graph-calls', type=int, default=64)
    parser.add_argument('--seed', type=int, default=31)
    parser.add_argument('--output', type=Path, default=Path('results/down_projection.json'))
    args = parser.parse_args()
    if min(args.tokens + [args.hidden_size, args.expert_hidden_size, args.repeats,
                          args.rounds, args.graph_calls]) <= 0 or args.warmup < 0:
        parser.error('尺寸、repeats、rounds、graph-calls 需大于 0，warmup 需非负')
    if not torch.cuda.is_available():
        parser.error('需要一张 CUDA GPU')
    import triton
    from mini_ep.kernels.down_projection import (
        down_projection_tuned, down_selected_config, expert_mlp_down_tuned)
    from mini_ep.kernels.gemm_triton import (
        expert_mlp_triton, selected_config, triton_gemm_tuned)
    from mini_ep.model.expert import expert_mlp_reference, gemm_reference
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    dtype = getattr(torch, args.dtype)
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        parser.error('GPU 不支持 BF16')
    report = {'config': vars(args) | {'output': str(args.output), 'batch_size': None,
              'seq_len': None, 'num_experts': 1, 'top_k': 1, 'ep_size': 1, 'chunk_size': None},
              'environment': {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'gpu_uuid': str(getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), 'uuid', 'unavailable')),
              'triton': triton.__version__, 'cuda': torch.version.cuda, 'python': platform.python_version()},
              'timing': '单 GPU；64 次运算/Graph（可配置）；10 次回放预热；交替顺序多轮 CUDA Event；回放耗时除以 graph-calls；报告中位数；编译调优捕获不计时',
              'results': [], 'checks': [], 'selected_configs': []}
    tol = {torch.float32: 1e-4, torch.float16: 1e-2, torch.bfloat16: 2e-2}[dtype]
    with torch.inference_mode():
        for m in args.tokens:
            h, f = args.hidden_size, args.expert_hidden_size
            x = torch.randn(m, h, dtype=dtype, device='cuda')
            up = (torch.randn(f, h, dtype=dtype, device='cuda') / h**.5).contiguous()
            down = (torch.randn(h, f, dtype=dtype, device='cuda') / f**.5).contiguous()
            activated = torch.nn.functional.gelu(x @ up.T)
            operations = [
                ('down', 'torch', lambda: gemm_reference(activated, down)),
                ('down', 'original', lambda: triton_gemm_tuned(activated, down)),
                ('down', 'expanded', lambda: down_projection_tuned(activated, down)),
                ('mlp', 'torch', lambda: expert_mlp_reference(x, up, down)),
                ('mlp', 'original', lambda: expert_mlp_triton(x, up, down, tuned=True)),
                ('mlp', 'expanded', lambda: expert_mlp_down_tuned(x, up, down)),
            ]
            expected = {'down': gemm_reference(activated, down), 'mlp': expert_mlp_reference(x, up, down)}
            for kind, name, op in operations:
                actual = op()
                torch.testing.assert_close(actual, expected[kind], rtol=tol, atol=tol)
                assert actual.shape == expected[kind].shape and actual.dtype == dtype
                assert actual.device == x.device and actual.is_contiguous()
                report['checks'].append(dict(tokens=m, operation=kind, backend=name, passed=True,
                                            max_abs_error=float((actual-expected[kind]).abs().max())))
            report['selected_configs'].append(dict(tokens=m, up=selected_config(x, up),
                original_down=selected_config(activated, down), expanded_down=down_selected_config(activated, down)))
            graphs = [(kind, name, capture(op, args.graph_calls, args.warmup)) for kind, name, op in operations]
            samples = {(kind, name): [] for kind, name, _ in graphs}
            for rnd in range(args.rounds):
                for kind, name, graph in (graphs if rnd % 2 == 0 else graphs[::-1]):
                    values = measure(graph, args.graph_calls, args.repeats)
                    samples[kind, name].extend(values)
                    print(f'M={m} round={rnd} {kind}/{name}: {statistics.median(values):.3f} us', flush=True)
            for (kind, name), values in samples.items():
                med = statistics.median(values)
                report['results'].append(dict(tokens=m, operation=kind, backend=name,
                    median_us=med, samples_us=values, tflops=2*m*h*f*(2 if kind=='mlp' else 1)/med/1e6))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(f'PASS: {len(report["checks"])} 项数值与输出属性检查；报告 {args.output}', flush=True)

if __name__ == '__main__':
    main()
