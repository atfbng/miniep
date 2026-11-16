"""检查通信方式可以独立选择，且 NCCL/Peer 两条路径都保持去重。

重点验证：默认 auto；auto 只在 SM90+、单机、组内 CUDA P2P 全通且形状满足时
选 peer，其余情况退回 nccl；显式 peer 与历史别名 peer_dedup 映射到同一个去重
实现；选择 NCCL 时不会导入可选的 Peer/TMA 模块。
"""
import builtins
import sys
import types
import unittest
from unittest.mock import patch
import torch
from mini_ep.integration.transport import (
    decide_transport, get_peer_forward, get_transport, peer_shape_problems)


class TransportTests(unittest.TestCase):
    def test_default_is_auto(self):
        with patch.dict('os.environ', {}, clear=True):
            self.assertEqual(get_transport(), 'auto')

    def test_explicit_transport_and_legacy_alias(self):
        for name, canonical in (('auto','auto'), ('nccl','nccl'), ('peer','peer'), ('peer_dedup','peer')):
            with self.subTest(name=name), patch.dict('os.environ', {'MINIEP_TRANSPORT': name}):
                self.assertEqual(get_transport(), canonical)

    def test_invalid_selection(self):
        with patch.dict('os.environ', {'MINIEP_TRANSPORT': 'tma'}):
            with self.assertRaisesRegex(ValueError, 'tma'):get_transport()
        with self.assertRaisesRegex(ValueError, 'invalid'):get_peer_forward('invalid')

    def test_nccl_has_no_peer_import(self):
        with patch.object(builtins, '__import__', side_effect=ImportError('no peer support')):
            self.assertIsNone(get_peer_forward('nccl'))

    def test_peer_always_deduplicates(self):
        selected = types.ModuleType('mini_ep.distributed.peer_dedup')
        sentinel = object();selected.deduplicated_peer_forward = sentinel
        with patch.dict(sys.modules, {selected.__name__: selected}):
            self.assertIs(get_peer_forward('peer'), sentinel)
            self.assertIs(get_peer_forward('peer_dedup'), sentinel)


if __name__ == '__main__':unittest.main()


def _probe(device, capability=(9, 0), peers=(True,) * 4, host='h', symm=True):
    return dict(host=host, device=device, capability=list(capability), peer=list(peers),
                symmetric_memory=symm)


QWEN = dict(hidden=2048, num_experts=256, intermediate=512, dtype=torch.bfloat16, return_dtype=None)


class AutoTransportTests(unittest.TestCase):
    def test_h100_style_group_selects_peer(self):
        probes = [_probe(i) for i in range(4)]
        self.assertEqual(decide_transport(probes, peer_shape_problems(**QWEN))[0], 'peer')

    def test_falls_back_to_nccl(self):
        cases = {
            'pre-SM90': [_probe(i, capability=(8, 9)) for i in range(2)],
            'no peer access': [_probe(0, peers=(True, False)), _probe(1, peers=(False, True))],
            'multi host': [_probe(0, host='a'), _probe(1, host='b')],
            'no cuda': [dict(host='h', device=None, capability=None, peer=[], symmetric_memory=False)] * 2,
            'no symmetric memory': [_probe(i, symm=False) for i in range(2)],
            'shared device index': [_probe(0), _probe(0)],
        }
        for name, probes in cases.items():
            with self.subTest(name):
                self.assertEqual(decide_transport(probes, [])[0], 'nccl')

    def test_shape_problems(self):
        self.assertEqual(peer_shape_problems(**QWEN), [])
        self.assertEqual(peer_shape_problems(**dict(QWEN, intermediate=768)), [])
        for change in (dict(dtype=torch.float32), dict(return_dtype=torch.float32),
                       dict(hidden=2047), dict(num_experts=96), dict(intermediate=500)):
            with self.subTest(change):
                problems = peer_shape_problems(**dict(QWEN, **change))
                self.assertTrue(problems)
                self.assertEqual(decide_transport([_probe(0), _probe(1)], problems)[0], 'nccl')


class LocalProbeTests(unittest.TestCase):
    def test_local_probe_runs_and_has_expected_fields(self):
        """直接调用真实探测函数（回归：函数内局部 import 曾遮蔽 torch 名字）。"""
        from mini_ep.integration.transport import _local_probe
        probe = _local_probe()
        self.assertEqual(set(probe), {'host', 'device', 'capability', 'peer', 'symmetric_memory'})
        if torch.cuda.is_available():
            self.assertEqual(len(probe['peer']), torch.cuda.device_count())
            self.assertTrue(probe['peer'][probe['device']])
        else:
            self.assertIsNone(probe['device'])
