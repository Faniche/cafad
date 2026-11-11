"""test_group_train.py | 2026-11-08 | group 与 train 的持久化链路测试 | Codex."""
from __future__ import annotations

import json
import argparse
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from src.app.cli import main
from src.modules.train.api import _train_expert
from src.utils.strings import PathUtils
from tests.test_data_module_pipeline import write_adsb_fixture


class GroupTrainTests(unittest.TestCase):
    """验证正常建组、异常训练路由和直接上游 train 输入。"""

    def test_group_floor_replaces_phase_candidate_when_needed(self) -> None:
        """最长正常训练航班的片段数应成为所有阶段有效组数的硬下限。"""
        import numpy as np
        from src.modules.group.api import HdbscanCandidate, _select_candidates_with_group_floor

        def candidate(phase: str, groups: int, coverage: float) -> HdbscanCandidate:
            effective = tuple((index, np.asarray([index]), np.asarray([f'{phase}-{index}'])) for index in range(groups))
            return HdbscanCandidate(phase, 4, 1, np.asarray([0]), np.asarray([1.0]), effective, groups,
                                    coverage, coverage, 1.0)

        preferred = candidate('CR', 1, 1.0)
        split = candidate('CR', 2, 0.95)
        selected = _select_candidates_with_group_floor([('CR', [preferred, split])], 1.0, 2)
        self.assertIs(selected[0], split)
        self.assertGreaterEqual(sum(len(item.effective_clusters) for item in selected), 2)

    def test_weighted_feature_distance_matches_manual_definition(self) -> None:
        import numpy as np
        from src.modules.group.features import weighted_distance_matrix
        vectors = np.asarray([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
        matrix = weighted_distance_matrix(vectors, 2)
        np.testing.assert_allclose(matrix, [[0.0, 5.0, 10.0], [5.0, 0.0, 5.0], [10.0, 5.0, 0.0]], atol=1e-12)

    def test_segment_center_time_uses_supplied_dataset_origin(self) -> None:
        """被排除 NA 片段也不能悄然改变其余片段的全数据集时间原点。"""
        from src.modules.group.features import build_segment_features
        segments = pd.DataFrame({'segment_id': ['s0']})
        values = pd.DataFrame({
            'segment_id': ['s0', 's0'], 'time_ns': [1_000_000_000_000, 1_004_000_000_000],
            'longitude_deg': [120.0, 120.1], 'latitude_deg': [30.0, 30.1],
            'geometric_altitude_m': [100.0, 120.0], 'heading_deg': [359.0, 1.0], 'speed_mps': [40.0, 42.0],
        })
        features = build_segment_features(segments, values, time_origin_ns=998_000_000_000)
        self.assertAlmostEqual(float(features.loc[0, 'center_time_s']), 4.0)
        self.assertAlmostEqual(float(features.loc[0, 'duration_s']), 4.0)

    def test_spawn_pool_trains_independent_experts(self) -> None:
        from src.modules.train.parallel import train_pool
        from src.modules.train.api import build_parser
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'data'; data.mkdir()
            frames = [_sequence(i) for i in range(3)]
            pd.concat(frames).to_csv(data / 'input_observations.csv', index=False)
            pd.DataFrame([{'segment_id': f's{i}', 'record_id': record}
                          for i, frame in enumerate(frames) for record in frame.record_id]).to_csv(data / 'point_segment_map.csv', index=False)
            pd.DataFrame([{'segment_id': f's{i}', 'flight_id': f'F{i}'} for i in range(3)]).to_csv(data / 'segments.csv', index=False)
            assignments = pd.DataFrame([{'group_id': g, 'segment_id': 's2', 'flight_id': 'F2', 'segment_label': 1} for g in ('g0', 'g1')])
            assignments.to_csv(data / 'train_assignments.csv', index=False)
            assignments.to_csv(data / 'validation_assignments.csv', index=False)
            pd.DataFrame([{'group_id': g, 'segment_id': f's{i}', 'flight_id': f'F{i}'}
                          for g in ('g0', 'g1') for i in (0, 1)]).to_csv(data / 'references.csv', index=False)
            args = build_parser().parse_args(['--input', str(root / 'manifest.json'), '--d-model', '8', '--epochs', '1',
                                              '--precision', 'fp32'])
            logger = logging.getLogger('test.pool'); logger.handlers = [logging.NullHandler()]
            result = train_pool([('g0', 0), ('g1', 1)], [torch.device('cpu'), torch.device('cpu')],
                                root / 'manifest.json', root / 'manifest.json', args, 42, root / 'models', logger, 2)
            self.assertEqual({item['group_id'] for item in result}, {'g0', 'g1'})
            progress = json.loads((root / 'training_progress.json').read_text(encoding='utf-8'))
            self.assertEqual({row['state'] for row in progress['experts']}, {'completed'})
            for group in ('g0', 'g1'):
                self.assertTrue((root / 'models' / group / 'best.pt').is_file())

    def test_group_parser_has_visible_startup_defaults(self) -> None:
        """网页 schema 应从 argparse 取得可运行的 HDBSCAN 候选初始值。"""
        from src.modules.group.api import build_parser
        values = build_parser().parse_args(['--input', 'segment.json'])
        self.assertEqual(values.min_cluster_sizes, (4, 8, 12, 16, 24, 32, 48, 64))
        self.assertEqual(values.min_samples_values, (1, 2, 4, 8, 12, 16))
        self.assertEqual(values.weight_center_time, 1.0)
        self.assertEqual(values.weight_turn_rate, 1.0)

    def _run(self, base: Path, values: list[str]) -> None:
        with patch.object(PathUtils, 'get_project_root', return_value=base):
            self.assertEqual(main(values), 0)

    def test_group_and_train_use_only_direct_manifest_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / 'fixture.csv'; write_adsb_fixture(source)
            (base / '.env').write_text('nproc=2\ndata_proc_usage=100%\ninjection_minimum_trajectory_points=5\n', encoding='utf-8')
            self._run(base, ['-m', 'preprocess', '--format', 'adsb-csv', '--input', str(source), '--altitude-unit', 'm',
                             '--speed-unit', 'm/s', '--heading-unit', 'deg', '--flight-gap-seconds', '90'])
            pre = next((base / 'data' / '0_preprocess' / 'fixture').iterdir())
            self._run(base, ['--seed', '10', '-m', 'split', '--input', str(pre / 'manifest.json')])
            split = next((base / 'data' / '1_split' / 'fixture').iterdir())
            protocol = base / 'protocol.json'
            protocol.write_text(json.dumps(_protocol()), encoding='utf-8')
            self._run(base, ['--seed', '10', '-m', 'inject', '--input', str(pre / 'manifest.json'), '--split-manifest',
                             str(split / 'manifest.json'), '--protocol', str(protocol)])
            injected = next((base / 'data' / '2_inject' / 'fixture').iterdir())
            self._run(base, ['-m', 'segment', '--input', str(injected / 'manifest.json'), '--window-seconds', '60', '--buckets', '2', '--chunk-rows', '10'])
            segmented = next((base / 'data' / '3_segment' / 'fixture').iterdir())
            self._run(base, ['-m', 'group', '--input', str(segmented / 'manifest.json'),
                             '--min-cluster-sizes', '2', '--min-samples-values', '1', '--min-group-members', '2', '--max-group-fraction', '1'])
            grouped = next((base / 'data' / '4_group' / 'fixture').iterdir())
            groups = pd.read_csv(grouped / 'data' / 'groups.csv')
            assignments = pd.read_csv(grouped / 'data' / 'train_assignments.csv')
            selection = pd.read_csv(grouped / 'data' / 'hdbscan_selection.csv')
            feature_scaler = json.loads((grouped / 'data' / 'feature_scaler.json').read_text(encoding='utf-8'))
            self.assertFalse(groups.empty); self.assertFalse(assignments.empty)
            self.assertEqual(len(feature_scaler['feature_names']), 13)
            self.assertEqual(len(feature_scaler['weights']), 13)
            self.assertTrue((grouped / 'data' / 'segment_features.csv').is_file())
            self.assertEqual(int(selection['selected'].sum()), selection['phase'].nunique())
            self.assertTrue(assignments['segment_id'].is_unique)
            self.assertTrue(set(assignments['segment_label']).issubset({0, 1}))
            self._run(base, ['-m', 'train', '--input', str(grouped / 'manifest.json'), '--gpus', 'cpu', '--d-model', '8',
                             '--branch-heads', '2', '--group-heads', '2', '--pool-heads', '2', '--memory-slots', '2',
                             '--ref-limit', '2', '--batch-size', '2', '--epochs', '1', '--patience', '1', '--precision', 'fp32'])
            trained = next((base / 'data' / '5_train' / 'fixture').iterdir())
            summary = pd.read_csv(trained / 'expert_summary.csv')
            self.assertEqual(len(summary), len(groups))
            self.assertTrue(summary['checkpoint_selection'].isin({'validation_loss', 'training_loss_no_validation'}).all())
            for group_id in summary['group_id']:
                self.assertTrue((trained / 'models' / group_id / 'memory.npy').is_file())
            from src.modules.detect.inference import score_partition
            scores, pairs, _ = score_partition(trained / 'manifest.json', 'val', 3, 1.0, 'cpu', logging.getLogger('test.detect'))
            self.assertNotIn('segment_label', scores.columns)
            self.assertTrue(scores['score'].between(0, 1).all())
            self.assertTrue(pairs['score'].between(0, 1).all())

    def test_train_expert_falls_back_to_training_loss_without_validation_targets(self) -> None:
        """小组缺少验证路由时仍应产出可追溯专家，而不是让整个训练任务失败。"""
        with tempfile.TemporaryDirectory() as temporary:
            sequences = {name: _sequence(index) for index, name in enumerate(('s0', 's1', 's2'))}
            assignments = pd.DataFrame([{'group_id': 'g0000', 'segment_id': 's2', 'flight_id': 'F2', 'segment_label': 1}])
            validation = pd.DataFrame(columns=assignments.columns)
            references = pd.DataFrame([
                {'group_id': 'g0000', 'segment_id': 's0', 'flight_id': 'F0'},
                {'group_id': 'g0000', 'segment_id': 's1', 'flight_id': 'F1'},
            ])
            arguments = argparse.Namespace(d_model=8, branch_layers=1, branch_heads=2, group_layers=1, group_heads=2,
                                           pool_heads=2, memory_slots=2, dropout=0.0, learning_rate=0.0003,
                                           weight_decay=0.0, epochs=1, patience=1, min_delta=0.0001,
                                           batch_size=1, micro_batch_size=4, grad_clip=1.0, ref_limit=2, ablation='none',
                                           precision='fp32')
            logger = logging.getLogger('test.train.empty_validation'); logger.handlers = [logging.NullHandler()]
            summary = _train_expert('g0000', 0, assignments, validation, references, sequences, arguments, 42,
                                    Path(temporary), logger, torch.device('cpu'))
            self.assertEqual(summary['checkpoint_selection'], 'training_loss_no_validation')
            self.assertTrue((Path(temporary) / 'memory.npy').is_file())


def _protocol() -> dict[str, object]:
    return {
        'schema_version': 2, 'event_length_points': [3, 3], 'context_points_each_side': 1,
        'event_type_probabilities': {'gaussian_point': 0.0, 'reported_fixed_offset': 0.0, 'reported_progressive_drift': 0.0,
                                     'reported_local_perturbation': 0.0, 'true_local_maneuver': 1.0, 'motion_transfer': 0.0},
        'gaussian_point': {'speed_probability': 0.5, 'speed_std_fraction': 0.01, 'speed_std_floor_mps': 0.1, 'speed_std_cap_mps': 1.0,
                           'altitude_std_fraction': 0.01, 'altitude_std_floor_m': 0.1, 'altitude_std_cap_m': 1.0},
        'reported_fixed_offset': {'field_probabilities': {'lateral_position': 1.0, 'altitude': 0.0, 'speed': 0.0},
                                  'lateral_offset_m_range': [1.0, 2.0], 'altitude_offset_m_range': [1.0, 2.0], 'speed_offset_mps_range': [1.0, 2.0]},
        'reported_progressive_drift': {'field_probabilities': {'altitude': 1.0, 'speed': 0.0},
                                       'altitude_step_m_range': [1.0, 2.0], 'speed_step_mps_range': [1.0, 2.0]},
        'reported_local_perturbation': {'field_probabilities': {'speed': 0.0, 'lateral_position': 1.0, 'altitude': 0.0},
                                        'speed_factor_range': [0.8, 1.2], 'lateral_offset_m_range': [1.0, 2.0],
                                        'altitude_amplitude_m_range': [1.0, 2.0], 'smoothing_points': 1},
        'true_local_maneuver': {'field_probabilities': {'lateral_position': 1.0, 'altitude': 0.0},
                                'lateral_offset_m_range': [1.0, 2.0], 'altitude_amplitude_m_range': [1.0, 2.0],
                                'speed_factor_range': [0.8, 1.2], 'smoothing_points': 1},
        'motion_transfer': {'scale_range': [0.8, 1.2]},
    }


def _sequence(offset: int) -> pd.DataFrame:
    """构造保持训练所需字段的极小变长片段。"""
    return pd.DataFrame({
        'record_id': [f'{offset}-0', f'{offset}-1', f'{offset}-2'],
        'time_ns': [0, 1_000_000_000, 2_000_000_000],
        'longitude_deg': [120.0 + offset, 120.01 + offset, 120.02 + offset],
        'latitude_deg': [30.0, 30.01, 30.02],
        'geometric_altitude_m': [100.0, 110.0, 120.0],
        'barometric_altitude_m': [95.0, 105.0, 115.0],
        'heading_deg': [90.0, 90.0, 90.0],
        'speed_mps': [40.0, 42.0, 44.0],
    })
