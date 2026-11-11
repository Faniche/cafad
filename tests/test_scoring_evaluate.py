"""test_scoring_evaluate.py | 2026-11-09 | 多进程评分一致性与评价覆盖检查 | Codex."""
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from src.modules.detect.scoring import score_pairs
from src.modules.train.model import ExpertModel
from src.modules.train.api import _stats
from src.modules.evaluate.metrics import binary_metrics, align_segments
from tests.test_group_train import _sequence


class ScoringEvaluationTests(unittest.TestCase):
    def test_single_and_multiple_devices_agree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            architecture = dict(d_model=8, branch_layers=1, branch_heads=2, group_layers=1, group_heads=2,
                                pool_heads=2, memory_slots=2, dropout=.4)
            sequences = {'s0': _sequence(0), 's1': _sequence(1)}
            for group in ('g0', 'g1'):
                directory = root / 'models' / group; directory.mkdir(parents=True)
                model = ExpertModel(**architecture)
                torch.save({'architecture': architecture, 'state_dict': model.state_dict()}, directory / 'best.pt')
                np.save(directory / 'memory.npy', np.zeros((2, 16), dtype=np.float32))
                (directory / 'stats.json').write_text(json.dumps(_stats(['s0', 's1'], sequences)), encoding='utf-8')
            pairs = pd.DataFrame([{'group_id': group, 'segment_id': f's{i % 2}'} for group in ('g0', 'g1') for i in range(64)])
            devices = [torch.device(f'cuda:{i}') for i in range(min(4, torch.cuda.device_count()))] if torch.cuda.is_available() else [torch.device('cpu')] * 2
            logger = logging.getLogger('test.scoring'); logger.setLevel(logging.INFO)
            logger.handlers = [logging.StreamHandler()]
            single = score_pairs(pairs, sequences, root / 'manifest.json', devices[:1], 4, 8, logger)
            multiple = score_pairs(pairs, sequences, root / 'manifest.json', devices, 4, 2, logger)
            np.testing.assert_allclose(single.score, multiple.score, atol=1e-6, rtol=1e-5)
            self.assertTrue(single.index.equals(multiple.index))

    def test_metrics_and_missing_ids(self) -> None:
        result = binary_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
        self.assertEqual((result['tp'], result['tn'], result['fp'], result['fn']), (1, 1, 1, 1))
        self.assertEqual(result['f1'], .5)
        self.assertIsNone(binary_metrics(np.array([]), np.array([]))['f1'])
        self.assertIn('recall', binary_metrics(np.array([0]), np.array([0]))['undefined_metrics'])
        predicted = pd.DataFrame({'segment_id': ['a'], 'flight_id': ['f'], 'phase': ['CR'], 'split': ['test'], 'score': [.2]})
        truth = predicted.drop(columns='score').assign(segment_label=0)
        self.assertEqual(len(align_segments(predicted, truth)), 1)
        with self.assertRaises(ValueError):
            align_segments(predicted, pd.concat([truth, truth.assign(segment_id='b')]))

    def test_evaluate_cli(self) -> None:
        from src.app.cli import main
        from src.utils.strings import PathUtils
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / '.env').write_text('nproc=2\ndata_proc_usage=100%\n', encoding='utf-8')
            source = root / 'detect.json'
            source.write_text(json.dumps({'module': 'detect', 'status': 'SUCCEEDED', 'dataset_id': 'fixture',
                                           'outputs': [], 'dependencies': []}), encoding='utf-8')
            segments = pd.DataFrame({'segment_id': ['a', 'b'], 'segment_label': [0, 1], 'prediction': [0, 1],
                                      'phase': ['CR', 'NA'], 'route_mode': ['same_phase', 'fallback']})
            flights = pd.DataFrame({'flight_id': ['f'], 'label': [1], 'prediction': [1]})
            with patch.object(PathUtils, 'get_project_root', return_value=root), patch(
                    'src.modules.evaluate.api.read_evaluation', return_value=(segments, flights, 'test')):
                self.assertEqual(main(['-m', 'evaluate', '--input', str(source)]), 0)
            output = next((root / 'data/8_evaluate/fixture').iterdir())
            metrics = json.loads((output / 'data/metrics.json').read_text(encoding='utf-8'))
            self.assertEqual(metrics['segment']['f1'], 1)
