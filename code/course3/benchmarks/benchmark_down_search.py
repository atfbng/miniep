"""逐项扫描下投影分块；数值对照后，以相同 CUDA Graph 测量各配置。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import random
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from benchmarks.benchmark_down_projection import capture, measure
from mini_ep.model.expert import gemm_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', nargs='+', type=int, default=[2048])
    parser.add_argument('--hidden-size', type=int, default=1024)
    parser.add_argument('--expert-hidden-size', type=int, default=2048)
    parser.add_argument('--dtype', choices=('float16', 'bfloat16', 'float32'), default='float16')
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--graph-calls', type=int, default=64)
    parser.add_argument('--seed', type=int, default=31)
    parser.add_argument('--shortlist-from', type=Path, help='读取完整扫描报告，复测每个形状的前几名与两组指定配置')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--output', type=Path, default=Path('results/down_search.json'))
    args = parser.parse_args()
    if min(args.tokens + [args.hidden_size, args.expert_hidden_size, args.repeats,
                          args.rounds, args.graph_calls, args.warmup, args.top_k]) <= 0:
        parser.error('尺寸、warmup、repeats、rounds、graph-calls 必须大于 0')
    if not torch.cuda.is_available():
        parser.error('需要一张 CUDA GPU')
    import triton
    from triton.runtime.errors import OutOfResources
    from mini_ep.kernels.down_projection import DOWN_CANDIDATES
    from mini_ep.kernels.gemm_triton import gemm_kernel

    if args.dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
        parser.error('GPU 不支持 BF16')
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    dtype = getattr(torch, args.dtype)
    tolerance = {torch.float16: 1e-2, torch.bfloat16: 2e-2, torch.float32: 1e-4}[dtype]
    previous = None
    if args.shortlist_from:
        previous = json.loads(args.shortlist_from.read_text())
        for name in ('hidden_size', 'expert_hidden_size', 'dtype'):
            if previous['config'][name] != getattr(args, name):
                parser.error(f'复测需使用相同 {name}，扫描报告为 {previous["config"][name]}')
    report = {'config': vars(args) | {'output': str(args.output), 'num_experts': 1,
              'shortlist_from': str(args.shortlist_from) if args.shortlist_from else None,
              'top_k': 1, 'ep_size': 1, 'batch_size': None, 'seq_len': None, 'chunk_size': None},
              'environment': {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'gpu_uuid': str(getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), 'uuid', 'unavailable')),
              'triton': triton.__version__, 'cuda': torch.version.cuda, 'python': platform.python_version()},
              'candidate_count': len(DOWN_CANDIDATES), 'results': [], 'skipped': [],
              'timing': '单 GPU；相同输入、CUDA Graph；每回放 graph-calls 次；多轮随机次序；CUDA Event 耗时 / graph-calls；报告中位数；编译、数值检查和捕获在计时前完成'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    with torch.inference_mode():
        for m in args.tokens:
            h, f = args.hidden_size, args.expert_hidden_size
            # 与 MLP 相同，输入是上投影经过 GELU 后的连续 [M,F]。
            x = torch.randn((m, h), device='cuda', dtype=dtype)
            up = (torch.randn((f, h), device='cuda', dtype=dtype) / h**0.5).contiguous()
            weight = (torch.randn((h, f), device='cuda', dtype=dtype) / f**0.5).contiguous()
            activated = torch.nn.functional.gelu(x @ up.T)
            expected = gemm_reference(activated, weight)
            entries = [{'backend': 'torch', 'config': None,
                        'graph': capture(lambda: gemm_reference(activated, weight), args.graph_calls, args.warmup),
                        'samples_us': [], 'round_median_us': [], 'checked': True}]
            candidates = DOWN_CANDIDATES
            if previous:
                ranked = sorted((r for r in previous['results'] if r['tokens'] == m and r['backend'] == 'triton'),
                                key=lambda r: r['median_us'])
                if not ranked:
                    parser.error(f'扫描报告中没有 M={m} 的候选结果')
                selected = {tuple(r['config']) for r in ranked[:args.top_k]}
                selected.update({(64, 256, 64, 8, 3), (128, 128, 64, 4, 4)})
                candidates = [cfg for cfg in DOWN_CANDIDATES if cfg in selected]
            for cfg in candidates:
                bm, bn, bk, warps, stages = cfg
                output = torch.empty_like(expected)
                grid = (triton.cdiv(m, bm), triton.cdiv(h, bn))

                def operation(grid=grid, output=output, bm=bm, bn=bn, bk=bk, warps=warps, stages=stages):
                    return gemm_kernel[grid](activated, weight, output, m, h, f,
                        BM=bm, BN=bn, BK=bk, num_warps=warps, num_stages=stages)

                try:
                    compiled = operation()
                    torch.testing.assert_close(output, expected, rtol=tolerance, atol=tolerance)
                    assert output.shape == expected.shape and output.dtype == dtype
                    assert output.device == expected.device and output.is_contiguous()
                    graph = capture(operation, args.graph_calls, args.warmup)
                except OutOfResources as error:
                    report['skipped'].append({'tokens': m, 'config': cfg, 'reason': str(error)})
                    continue
                entries.append({'backend': 'triton', 'config': cfg, 'graph': graph,
                                'samples_us': [], 'round_median_us': [], 'checked': True,
                                'max_abs_error': float((output - expected).abs().max()),
                                'registers_per_thread': compiled.n_regs, 'spills': compiled.n_spills,
                                'shared_bytes': compiled.metadata.shared,
                                'output_tiles': triton.cdiv(m, bm) * triton.cdiv(h, bn)})
            torch.cuda.synchronize()
            print(f'M={m}: {len(entries)-1} 个候选数值检查通过，开始测量', flush=True)
            for rnd in range(args.rounds):
                order = entries.copy()
                rng.shuffle(order)
                for entry in order:
                    values = measure(entry['graph'], args.graph_calls, args.repeats)
                    entry['samples_us'].extend(values)
                    entry['round_median_us'].append(statistics.median(values))
                print(f'M={m}: round {rnd+1}/{args.rounds} 完成', flush=True)
            for entry in entries:
                record = {k: v for k, v in entry.items() if k != 'graph'}
                record.update(tokens=m, median_us=statistics.median(entry['samples_us']),
                              mean_us=statistics.fmean(entry['samples_us']))
                record['tflops'] = 2 * m * h * f / record['median_us'] / 1e6
                report['results'].append(record)
            ranked = sorted(entries, key=lambda e: statistics.median(e['samples_us']))
            for entry in ranked[:8]:
                print(entry['backend'], entry['config'], f'{statistics.median(entry["samples_us"]):.3f} us', flush=True)
            save()
            # 释放图及其私有内存池，再测量下一个 M。
            del entries, order, ranked, graph
    print(f'报告：{args.output}', flush=True)


if __name__ == '__main__':
    main()
