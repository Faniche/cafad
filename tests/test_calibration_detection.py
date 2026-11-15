"""test_calibration_detection.py | 2026-11-09 | 阈值、融合与无标签检测接口 | Codex."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import json
import numpy as np
import pandas as pd
from src.app.tables import read_table, write_table
from src.modules.calibrate.threshold import select_segment_threshold
from src.modules.detect.fusion import fallback_score, fuse_candidates, make_predictions


class DetectionTests(unittest.TestCase):
    def test_calibrate_detect_manifest_pipeline(self) -> None:
        from src.app.cli import main
        from src.utils.strings import PathUtils
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / '.env').write_text('nproc=2\ndata_proc_usage=100%\n', encoding='utf-8')
            trained = root / 'train.json'
            trained.write_text(json.dumps({'status': 'SUCCEEDED', 'module': 'train', 'dataset_id': 'fixture',
                                           'dataset_name': 'fixture', 'outputs': [], 'dependencies': [{'manifest': str(root / 'inject.json')}]}), encoding='utf-8')
            for module, parent in (('inject', 'group'), ('group', 'segment'), ('segment', 'split'), ('split', 'preprocess'), ('preprocess', None)):
                (root / f'{module}.json').write_text(json.dumps({'status': 'SUCCEEDED', 'module': module,
                    'outputs': [], 'dependencies': [{'manifest': str(root / f'{parent}.json')}] if parent else []}), encoding='utf-8')
            (root / 'data').mkdir()
            write_table(pd.DataFrame({'segment_id': ['a', 'b'], 'segment_label': [0, 1], 'flight_id': ['f1', 'f1'],
                                      'phase': ['CR', 'CR'], 'split': ['test', 'test']}), root / 'data', 'segments')
            scores = pd.DataFrame({'segment_id': ['a', 'b'], 'flight_id': ['f1', 'f1'], 'phase': ['CR', 'CR'],
                                   'split': ['val', 'val'], 'segment_label': [0, 1], 'score': [.2, .8],
                                   'route_mode': ['same_phase', 'same_phase']})
            pairs = pd.DataFrame({'segment_id': ['a', 'b'], 'group_id': ['g0', 'g0'], 'score': [.2, .8]})
            with patch.object(PathUtils, 'get_project_root', return_value=root):
                with patch('src.modules.calibrate.api.score_partition', return_value=(scores, pairs, root / 'segment.json')):
                    self.assertEqual(main(['-m', 'calibrate', '--input', str(trained)]), 0)
                calibrated = next((root / 'data/6_calibrate/fixture').iterdir()) / 'manifest.json'
                scores['split'] = 'test'
                with patch('src.modules.detect.api._test_segmentation', return_value={'segments': 2}), \
                     patch('src.modules.detect.api.score_partition', return_value=(scores, pairs, root / 'segment.json')):
                    self.assertEqual(main(['-m', 'detect', '--input', str(calibrated)]), 0)
                detected_manifest = next((root / 'data/7_detect/fixture').iterdir()) / 'manifest.json'
                self.assertEqual(main(['-m', 'evaluate', '--input', str(detected_manifest)]), 0)
            detected = next((root / 'data/7_detect/fixture').iterdir())
            result = read_table(detected / 'data', 'flight_predictions')
            self.assertEqual(result.prediction.tolist(), [1])

    def test_threshold_and_single_class(self) -> None:
        threshold, scan = select_segment_threshold(np.array([.1, .4, .5, .9]), np.array([0, 0, 1, 1]))
        self.assertEqual(threshold, .5)
        self.assertEqual(scan['f1'].max(), 1)
        with self.assertRaises(ValueError):
            select_segment_threshold(np.array([.2]), np.array([0]))

    def test_fusion_and_flight_or(self) -> None:
        self.assertEqual(fallback_score(np.array([.1, .4, .5, .6, .8])), .5)
        self.assertAlmostEqual(fuse_candidates(np.array([.2, .8]), np.array([10000., 10000.]), 1), .5)
        segments, flights = make_predictions(pd.DataFrame({'segment_id': ['s1', 's2'], 'flight_id': ['f1', 'f1'],
                                                          'score': [.2, .5]}), .5)
        self.assertEqual(segments['prediction'].tolist(), [0, 1])
        self.assertEqual(flights['prediction'].tolist(), [1])
