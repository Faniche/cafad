"""test_local_plot.py | 2026-11-11 | 本地分组绘图最小数据回归测试 | Codex."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.web_console.local_plot import _finalize_paths, load_group_index, render_plot, segment_choices


class LocalPlotTests(unittest.TestCase):
    """验证本地数据目录内的分组成员能被 Python 图形后端读取。"""

    def test_renders_segment_and_whole_flight_views(self) -> None:
        """routing 元数据、点映射和规范观测共同生成 SVG，而不依赖浏览器绘图。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observations = root / 'input_observations.csv'
            mapping = root / 'point_segment_map.csv'
            routing = root / 'routing_segments.csv'
            pd.DataFrame({
                'record_id': ['r1', 'r2', 'r3', 'r4'], 'flight_id': ['f1', 'f1', 'f2', 'f2'],
                'time_ns': [1, 2, 1, 2], 'longitude_deg': [120.0, 120.1, 121.0, 121.1],
                'latitude_deg': [30.0, 30.1, 31.0, 31.1], 'geometric_altitude_m': [100.0, 200.0, 300.0, 400.0],
            }).to_csv(observations, index=False)
            pd.DataFrame({'record_id': ['r1', 'r2', 'r3', 'r4'], 'segment_id': ['s1', 's1', 's2', 's2']}).to_csv(mapping, index=False)
            pd.DataFrame({'segment_id': ['s1', 's2'], 'group_id': ['g1', 'g2'], 'flight_id': ['f1', 'f2']}).to_csv(routing, index=False)
            index = load_group_index(observations, routing)
            self.assertEqual(index.group_ids, ('g1', 'g2'))
            self.assertEqual(segment_choices(index, 'g1'), ('s1',))
            image, summary = render_plot(observations, index, 'g1', 's1', False, workers=1)
            self.assertIn(b'<svg', image)
            self.assertEqual(summary['path_count'], 1)
            image, summary = render_plot(observations, index, 'g1', None, True, workers=1)
            self.assertIn(b'<svg', image)
            self.assertEqual(summary['flight_count'], 1)

    def test_parallel_path_finalization_respects_worker_limit(self) -> None:
        """路径足够多时可通过统一进程池并行排序，不重新读取 CSV。"""
        paths = {
            f'segment-{index}': [np.array([[2.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]])]
            for index in range(32)
        }
        result = _finalize_paths(paths, workers=2)
        self.assertEqual(len(result), 32)
        self.assertTrue((result['segment-0'][:, 0] == [1.0, 2.0]).all())
