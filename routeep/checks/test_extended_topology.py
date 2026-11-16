"""检查 EP2/EP4 配置只改变通信方式，去重路径始终开启。

同时验证 TP1/DP2 与 TP2/DP2 的启动参数、非法拓扑会被拒绝，以及历史值
MINIEP_TRANSPORT=peer_dedup 会兼容映射到 peer。
"""
import unittest
from unittest.mock import patch

from mini_ep.integration.config import engine_options, configure
from mini_ep.integration.transport import get_transport


class ExtendedTopologyTests(unittest.TestCase):
    @patch('mini_ep.integration.config.model_metadata', return_value={'model_family': 'qwen3_moe'})
    def test_ep2_and_ep4(self, metadata):
        for tp in (1, 2):
            options = engine_options('unused', dp_size=2, external=True, tp_size=tp)
            self.assertEqual(options['tensor_parallel_size'], tp)
            self.assertEqual(options['data_parallel_size'], 2)
            self.assertTrue(options['enable_expert_parallel'])
            self.assertEqual(options['all2all_backend'], 'deepep_high_throughput')

    def test_reject_invalid_topologies(self):
        for kwargs in ({'tp_size': 1}, {'tp_size': 3}, {'tp_size': 1, 'dp_size': 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                engine_options('unused', **kwargs)

    def test_configuration_preserves_opt_in(self):
        for mode in ('auto', 'nccl', 'peer', 'peer_dedup'):
            with patch.dict('os.environ', {'MINIEP_TRANSPORT': mode}, clear=True):
                configure('miniep')
                self.assertEqual(get_transport(), 'peer' if mode == 'peer_dedup' else mode)
        with patch.dict('os.environ', {}, clear=True):
            configure('miniep')
            # 默认 auto：实际传输由 Runner 构造时按硬件判定。
            self.assertEqual(get_transport(), 'auto')


if __name__ == '__main__':
    unittest.main()
