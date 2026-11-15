"""test_group_train.py | 2026-11-15 | 硬时空切分与确定性分组测试 | Codex."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.app.tables import read_table
from src.modules.group.api import _assign, _references, build_parser
from src.modules.segment.hard_rules import HardSegmentConfig, assign_cells, segment_observations


class HardSegmentGroupTests(unittest.TestCase):
    """验证硬边界可重放、片段唯一单元和同源参考去重。"""

    def test_cell_boundaries_and_contiguous_segments(self) -> None:
        observations = _observations()
        config = HardSegmentConfig()
        cells = assign_cells(observations, config, 0)
        self.assertEqual(cells['height_band'].tolist(), [0, 0, 1, 2])
        self.assertEqual(cells['time_bucket'].tolist(), [0, 0, 1, 1])
        self.assertEqual(cells['longitude_cell'].tolist(), [600, 600, 100, 25])
        with tempfile.TemporaryDirectory() as directory:
            result = segment_observations(observations, Path(directory), config, 0, _Logger())
            segments = read_table(Path(directory), 'segments')
            mapping = read_table(Path(directory), 'point_segment_map')
        self.assertEqual(result['segments'], 3)
        self.assertEqual(len(mapping), len(observations))
        self.assertTrue(mapping.groupby('segment_id')['cell_key'].nunique().eq(1).all())
        self.assertEqual(set(segments['height_band']), {0, 1, 2})

    def test_group_uses_one_reference_per_flight_and_exact_routes(self) -> None:
        normal = pd.DataFrame({
            'segment_id': ['a0', 'a1', 'b0'], 'flight_id': ['A', 'A', 'B'], 'ordinal': [0, 1, 0],
            'cell_key': ['h2_t0_x1_y1'] * 3,
        })
        references = _references(normal)
        self.assertEqual(set(references['flight_id']), {'A', 'B'})
        groups = pd.DataFrame({'group_id': ['h2_t0_x1_y1'], 'expert_id': [0], 'cell_key': ['h2_t0_x1_y1']})
        targets = normal.assign(segment_label=[0, 1, 0])
        assigned, excluded = _assign(targets, groups, 'hard_exact')
        self.assertEqual(len(assigned), 3); self.assertTrue(excluded.empty)
        missing, exclusions = _assign(targets.assign(cell_key='h2_t1_x1_y1'), groups, 'hard_exact')
        self.assertTrue(missing.empty); self.assertEqual(set(exclusions['reason']), {'hard_cell_without_normal_reference'})

    def test_group_parser_exposes_only_reference_constraints(self) -> None:
        values = build_parser().parse_args(['--input', 'segment.json'])
        self.assertEqual((values.min_group_members, values.min_reference_flights), (2, 2))
        self.assertFalse(hasattr(values, 'min_cluster_sizes'))


def _observations() -> pd.DataFrame:
    return pd.DataFrame({
        'record_id': ['r0', 'r1', 'r2', 'r3'], 'flight_id': ['F'] * 4, 'icao': ['ICAO'] * 4, 'call': ['CALL'] * 4,
        'time_ns': [0, 1_000_000_000, 1_801_000_000_000, 7_201_000_000_000],
        'longitude_deg': [120.0, 120.1, 120.1, 120.1], 'latitude_deg': [30.0] * 4,
        'geometric_altitude_m': [100.0, 2999.0, 3000.0, 6000.0], 'injected_label': [0] * 4,
        'split': ['train'] * 4, 'injection_plan': ['normal'] * 4,
    })


class _Logger:
    def info(self, *_: object) -> None:
        return None
