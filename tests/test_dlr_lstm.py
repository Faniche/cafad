"""test_dlr_lstm.py | 2026-11-13 | DLR 阶段识别适配器的最小集成测试 | Codex."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.modules.segment.dlr_lstm import DLR_PHASES, run_dlr_lstm
from src.modules.segment.api import segment_observations


def _observations(points: int = 200) -> pd.DataFrame:
    """生成一条满足 DLR 最小 160 秒要求的规范航班。"""
    seconds = np.arange(points, dtype=np.int64)
    return pd.DataFrame({
        'record_id': [f'record-{index}' for index in seconds], 'flight_id': 'flight-1',
        'icao': 'ABC123', 'call': 'TEST1', 'time_ns': seconds * 1_000_000_000,
        'longitude_deg': 120.0 + seconds * 0.001, 'latitude_deg': 30.0 + seconds * 0.001,
        'geometric_altitude_m': 1000.0 + seconds * 3.0, 'barometric_altitude_m': 990.0 + seconds * 3.0,
        'heading_deg': np.full(points, 45.0), 'speed_mps': 100.0 + seconds * 0.02,
        'injected_label': np.zeros(points, dtype=np.int8), 'split': 'train',
    })


class DlrLstmTests(unittest.TestCase):
    """预训练模型作为运行资源存在时，验证其与正式段表契约的兼容性。"""

    project_root = Path(__file__).resolve().parents[1]

    @unittest.skipUnless((project_root / 'models/dlr_flight_phase_lstm.pt').is_file(), '缺少本地 DLR 模型资源')
    def test_writes_standard_segment_intermediates(self) -> None:
        """DLR 适配器须保持点守恒、写入可供正式段表转换的标准文件。"""
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            summary = run_dlr_lstm(_observations(), target / 'phase', target / 'progress.sqlite',
                                   self.project_root, None, 8, 1, logging.getLogger('cafad.test.dlr'))
            points = pd.read_csv(target / 'phase' / 'points.csv')
            segments = pd.read_csv(target / 'phase' / 'segments.csv')
        self.assertTrue(summary['point_conservation_verified'])
        self.assertEqual(len(points), 200)
        self.assertTrue(set(points['phase']).issubset(DLR_PHASES))
        self.assertTrue(set(segments['phase']).issubset(DLR_PHASES))
        self.assertTrue(points['segment_id'].notna().all())

    @unittest.skipUnless((project_root / 'models/dlr_flight_phase_lstm.pt').is_file(), '缺少本地 DLR 模型资源')
    def test_rejects_flights_shorter_than_160_seconds(self) -> None:
        """不对短轨迹补造观测；应作为可追溯的不可识别航班输出。"""
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            summary = run_dlr_lstm(_observations(159), target / 'phase', target / 'progress.sqlite',
                                   self.project_root, None, 8, 1, logging.getLogger('cafad.test.dlr'))
            rejected = pd.read_csv(target / 'phase' / 'unrecognized_flights.csv')
        self.assertEqual(summary['processed_flights'], 0)
        self.assertEqual(summary['unrecognized_flights'], 1)
        self.assertIn('至少 160', rejected.loc[0, 'reason'])

    @unittest.skipUnless((project_root / 'models/dlr_flight_phase_lstm.pt').is_file(), '缺少本地 DLR 模型资源')
    def test_segment_api_uses_sun_for_dlr_short_flight(self) -> None:
        """DLR 不能满足 160 秒前提时，Sun 回退须覆盖原始点且保留独立标签来源。"""
        arguments = argparse.Namespace(
            method='dlr_lstm', dlr_model=None, dlr_inference_batch_size=8,
            chunk_rows=1_000, buckets=2, window_seconds=60.0,
        )
        observations = _observations(159)
        stable_flight_id = hashlib.sha256(json.dumps(['ABC123', 'TEST1', 0]).encode('utf-8')).hexdigest()[:24]
        observations['flight_id'] = stable_flight_id
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'data'
            summary = segment_observations(observations, target, arguments, 1,
                                           logging.getLogger('cafad.test.dlr'), 1800.0, self.project_root)
            mapping = pd.read_parquet(target / 'point_segment_map.parquet')
            audit = pd.read_parquet(target / 'dlr_sun_fallback_audit.parquet')
            unrecognized = pd.read_parquet(target / 'unrecognized_trajectories.parquet')
        self.assertEqual(summary['sun_fallback_flights'], 1)
        self.assertEqual(summary['sun_fallback_recognized_flights'], 1)
        self.assertEqual(len(mapping), 159)
        self.assertTrue(mapping['phase'].str.startswith('sun_fallback_').all())
        self.assertFalse(audit.empty)
        self.assertTrue(unrecognized.empty)

    @unittest.skipUnless((project_root / 'models/dlr_flight_phase_lstm.pt').is_file(), '缺少本地 DLR 模型资源')
    def test_cpu_preparation_uses_bounded_process_pool(self) -> None:
        """两个独立航班可由两个 CPU 工作进程准备，SQLite 仍只由父进程写入。"""
        first = _observations()
        second = _observations()
        second['record_id'] = ['second-' + value for value in second['record_id']]
        second['flight_id'] = 'flight-2'
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            summary = run_dlr_lstm(pd.concat((first, second), ignore_index=True), target / 'phase',
                                   target / 'progress.sqlite', self.project_root, None, 8, 2,
                                   logging.getLogger('cafad.test.dlr'))
        self.assertEqual(summary['processed_flights'], 2)
        self.assertEqual(summary['configured_workers'], 2)
        self.assertGreaterEqual(summary['observed_worker_processes'], 1)
