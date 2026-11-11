"""test_inject_events.py | 2026-11-06 | 事件化 ADS-B 异常注入单元测试 | Codex."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.modules.inject.corruption import corrupt_flight, load_protocol, minimum_required_points
from src.modules.inject.api import _inject_observations
from src.modules.inject.provenance import (EVENT_COLUMNS, IMPACT_COLUMNS, LABEL_COLUMNS, event_rows, final_labels,
                                           impact_rows, record_labels, validate_provenance)


ROOT = Path(__file__).resolve().parents[1]
EVENT_SEMANTICS = {
    'gaussian_point': 'reported_data_distortion',
    'reported_fixed_offset': 'reported_data_distortion',
    'reported_progressive_drift': 'reported_data_distortion',
    'reported_local_perturbation': 'reported_data_distortion',
    'true_local_maneuver': 'true_motion_anomaly',
    'motion_transfer': 'true_motion_anomaly',
}


def flight_frame(flight_id: str, *, points: int = 50) -> pd.DataFrame:
    """构造严格递增、字段完备的单航班规范观测。"""
    positions = np.arange(points, dtype=np.int64)
    return pd.DataFrame({
        'record_id': [f'{flight_id}-r{position:03d}' for position in positions],
        'source_row_index': positions,
        'flight_id': flight_id,
        'icao': 'ABC123', 'call': 'CAF123',
        'time_ns': positions * 1_000_000_000,
        'longitude_deg': 120.0 + positions * 0.0001,
        'latitude_deg': 30.0 + positions * 0.0001,
        'geometric_altitude_m': 1000.0 + positions,
        'barometric_altitude_m': 990.0 + positions,
        'heading_deg': np.full(points, 45.0),
        'speed_mps': np.full(points, 20.0),
        'injected_label': np.zeros(points, dtype=np.int8),
        'split': 'train',
    })


def donor_pool() -> pd.DataFrame:
    """构造不同来源、同分区的真实运动供体。"""
    donor = flight_frame('donor-flight')
    donor['longitude_deg'] = 121.0 + np.arange(len(donor)) * 0.0002
    donor['latitude_deg'] = 31.0 + np.arange(len(donor)) * 0.00015
    donor['geometric_altitude_m'] = 2000.0 + np.arange(len(donor)) * 2.0
    donor['barometric_altitude_m'] = 1990.0 + np.arange(len(donor)) * 2.0
    return donor


class InjectionEventTests(unittest.TestCase):
    """核对每类事件的语义、边界、稳定 ID 和逐点账本。"""

    def setUp(self) -> None:
        self.protocol = load_protocol(ROOT / 'configs' / 'injection_protocol_example.yaml')
        self.target = flight_frame('target-flight')

    def _only_event_protocol(self, event_type: str) -> dict[str, object]:
        protocol = deepcopy(self.protocol)
        protocol['event_type_probabilities'] = {
            name: 1.0 if name == event_type else 0.0
            for name in EVENT_SEMANTICS
        }
        return protocol

    def test_example_protocol_derives_the_configured_40_point_minimum(self) -> None:
        """40 来自最长 30 点事件与两侧各 5 点上下文，不继承论文的 600 点实验长度。"""
        self.assertEqual(minimum_required_points(self.protocol), 40)

    def test_all_event_types_preserve_provenance_and_semantics(self) -> None:
        """六类事件均产生实际改动，且航班边界与最终标签可交叉核验。"""
        for seed, (event_type, semantics) in enumerate(EVENT_SEMANTICS.items(), start=1):
            with self.subTest(event_type=event_type):
                before = self.target.copy()
                after, events = corrupt_flight(before.copy(), donor_pool(), self._only_event_protocol(event_type),
                                                np.random.default_rng(seed))
                self.assertEqual(len(events), 1)
                event = events[0]
                self.assertEqual(event['event_type'], event_type)
                self.assertEqual(event['semantics'], semantics)
                self.assertEqual(event['support_scope'], 'whole_flight')
                self.assertGreater(len(event['changed_positions']), 0)
                self.assertGreaterEqual(int(np.min(event['planned_positions'])), 5)
                self.assertLessEqual(int(np.max(event['planned_positions'])), len(before) - 6)
                self.assertTrue(np.isfinite(after.select_dtypes(include='number').to_numpy()).all())

                rows = event_rows(events, after, flight_id='target-flight', split='train', seed=seed)
                labels = record_labels(before, after, events)
                impacts = impact_rows(events, flight_id='target-flight', flight_points=len(after))
                validate_provenance(
                    before, after, labels,
                    pd.DataFrame(rows, columns=EVENT_COLUMNS),
                    pd.DataFrame(impacts, columns=IMPACT_COLUMNS),
                )
                self.assertEqual(tuple(labels.columns), LABEL_COLUMNS)
                row = rows[0]
                self.assertEqual(row['interval_convention'], '[start,end)')
                self.assertEqual(row['end_index_exclusive'] - row['start_index'], row['n_target_points'])
                self.assertEqual(int(labels['injected_label'].sum()), row['n_changed_points'])
                if event_type == 'motion_transfer':
                    self.assertEqual(row['donor_flight_id'], 'donor-flight')
                    self.assertNotEqual(row['donor_flight_id'], row['flight_id'])

    def test_local_perturbation_avoids_multiplicative_speed_on_stationary_records(self) -> None:
        """地面等待的零速度窗口改选可修改字段，不能以零改变量中止整个注入任务。"""
        protocol = self._only_event_protocol('reported_local_perturbation')
        protocol['reported_local_perturbation']['field_probabilities'] = {
            'speed': 0.5, 'lateral_position': 0.5, 'altitude': 0.0,
        }
        stationary = self.target.copy()
        stationary['speed_mps'] = 0.0
        after, events = corrupt_flight(stationary, donor_pool(), protocol, np.random.default_rng(37))
        self.assertEqual(events[0]['parameters']['field'], 'lateral_position')
        self.assertGreater(len(events[0]['changed_positions']), 0)
        self.assertTrue(final_labels(stationary, after).any())

    def test_all_normal_plan_does_not_create_an_idle_process_pool(self) -> None:
        """没有合格异常航班时 workers=0 仍应输出正常副本和完整账本。"""
        source = pd.concat((flight_frame('normal-a'), flight_frame('normal-b')), ignore_index=True)
        splits = pd.DataFrame({
            'flight_id': ['normal-a', 'normal-b'],
            'split': ['train', 'val'],
            'injection_plan': ['normal', 'normal'],
            'injection_eligible': [False, False],
            'injection_skip_reason': ['', ''],
        })
        final, labels, events, impacts, outcomes = _inject_observations(
            source, splits, self.protocol, root_seed=42, workers=0,
        )
        self.assertEqual(len(final), len(source))
        self.assertTrue(final['injected_label'].eq(0).all())
        self.assertTrue(labels['injected_label'].eq(0).all())
        self.assertTrue(events.empty)
        self.assertTrue(impacts.empty)
        self.assertEqual(set(outcomes['injection_outcome']), {'normal_planned'})


if __name__ == '__main__':
    unittest.main()
