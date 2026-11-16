"""Regression for cuBLAS Hopper GEMMs whose kernel names start with nvjet_."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from benchmarks.analyze_overlap import analyze


class OverlapAnalysisTests(unittest.TestCase):
    def test_gemm_and_nvjet_have_the_same_overlap_semantics(self):
        for name in ('sm80_xmma_gemm_half', 'nvjet_sm90_hsh_128x256_64x4_TNN'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'trace.sqlite'
                db = sqlite3.connect(path)
                db.executescript('''
                    CREATE TABLE StringIds(id INTEGER, value TEXT);
                    CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT,
                        textId INTEGER, globalTid INTEGER);
                    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,
                        end INTEGER, globalPid INTEGER, demangledName INTEGER);
                ''')
                pid = 42 << 24
                db.executemany('INSERT INTO StringIds VALUES(?,?)',
                               [(1, 'ncclDevKernel_SendRecv'), (2, name), (3, 'elementwise')])
                db.execute('INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,?)',
                           (0, 20_000_000, 'PIPELINE_PROFILE', None, pid + 7))
                db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?)', [
                    (0, 10_000_000, pid, 1), (5_000_000, 15_000_000, pid, 2),
                    (0, 20_000_000, pid, 3),  # Elementwise ops are outside GEMM overlap.
                    (0, 20_000_000, pid + (1 << 24), 2),  # Another process.
                    (30_000_000, 40_000_000, pid, 2),  # Outside this NVTX scope.
                ])
                db.commit()
                db.close()
                result = analyze(str(path))[0]
                self.assertEqual(result['communication_ms'], 10.0)
                self.assertEqual(result['gemm_ms'], 10.0)
                self.assertEqual(result['overlap_ms'], 5.0)
                self.assertEqual(result['exposed_communication_ms'], 5.0)
                self.assertEqual(result['overlap_ratio'], 0.5)


if __name__ == '__main__':
    unittest.main()
