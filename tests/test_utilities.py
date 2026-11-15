"""Small mechanism checks using artificial fixtures, never research data."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.data_processing.anchors import anchor_distance, cluster_anchors, wrap_heading
from src.data_processing.legacy_plane.phase_segmentation import phase_ranges_vertical_rate
from src.data_processing.noise_provenance import (
    label_group, load_records, normalize_times, validate_records,
)
from src.data_processing.sampling import sample_observed_times


class SamplingTests(unittest.TestCase):
    def test_physical_time_sampling_prefers_earlier_tie(self):
        actual = sample_observed_times(np.array([0., 2., 8., 10.]), 3)
        np.testing.assert_array_equal(actual, [0, 1, 3])

    def test_nearest_observation_duplicates_are_removed(self):
        actual = sample_observed_times(np.array([0., .01, .02, .03, 100.]), 4)
        np.testing.assert_array_equal(actual, [0, 3, 4])

    def test_oversized_request_keeps_original_observations(self):
        np.testing.assert_array_equal(sample_observed_times(np.array([3., 7.]), 8), [0, 1])


class AnchorTests(unittest.TestCase):
    def test_heading_wrap_avoids_full_circle_discontinuity(self):
        a = np.array([0., 0., 0., np.deg2rad(359.)])
        b = np.array([0., 0., 0., np.deg2rad(1.)])
        self.assertAlmostEqual(float(anchor_distance(a, b)), 2. / 180.)
        self.assertAlmostEqual(float(wrap_heading(np.deg2rad(358.))), np.deg2rad(-2.))

    def test_distance_preserves_existing_scales_and_broadcasting(self):
        origin = np.zeros(4)
        others = np.array([[10., 10000., 0., np.pi], [0., 0., 0., 0.]])
        np.testing.assert_allclose(anchor_distance(origin, others), [np.sqrt(3.), 0.])

    @unittest.skipUnless(importlib.util.find_spec('sklearn'), 'Optional scikit-learn dependency is not installed')
    def test_sparse_clustering_matches_dense_exact_distance(self):
        from sklearn.cluster import DBSCAN
        anchors = np.array([
            [0., 0., 0., np.pi - .02],
            [0., 50., 0., -np.pi + .02],
            [0., 100., 0., np.pi - .01],
            [100., 100000., 100000., 0.],
            [100., 100030., 100000., 0.],
            [200., 2000000., 2000000., 0.],
        ])
        distances = anchor_distance(anchors[:, None, :], anchors[None, :, :])
        dense = DBSCAN(eps=.05, min_samples=2, metric='precomputed').fit_predict(distances)
        sparse = cluster_anchors(anchors, eps=.05, min_samples=2)
        np.testing.assert_array_equal(sparse == -1, dense == -1)
        np.testing.assert_array_equal(sparse[:, None] == sparse[None, :], dense[:, None] == dense[None, :])


class PositionalLabelTests(unittest.TestCase):
    @staticmethod
    def records():
        return validate_records({'000123': {'gaussian': {
            'positions': np.array([1], dtype=np.int64),
            'timestamps': ['2024-05-22 01:00:00.917'],
        }}}, group_lengths={'000123': 3})

    def test_repeated_timestamps_do_not_expand_position_labels(self):
        times = normalize_times(['2024-05-22 01:00:00.917'] * 3)
        result = label_group('000123', times, 0, self.records(), total_length=3)
        np.testing.assert_array_equal(result['exact_mask'], [False, True, False])
        self.assertEqual(result['timestamp_only_matches'], 2)

    def test_chunk_offsets_match_whole_group(self):
        times = normalize_times(['2024-05-22 01:00:00.917'] * 3)
        whole = label_group('000123', times, 0, self.records())
        first = label_group('000123', times[:1], 0, self.records())
        second = label_group('000123', times[1:], 1, self.records())
        np.testing.assert_array_equal(np.r_[first['exact_mask'], second['exact_mask']], whole['exact_mask'])

    def test_timestamp_mismatch_is_reported_instead_of_silently_relabeling(self):
        times = normalize_times(['2024-05-22 01:00:01'] * 3)
        result = label_group('000123', times, 0, self.records())
        self.assertEqual(result['position_timestamp_mismatch'], 1)
        np.testing.assert_array_equal(result['exact_mask'], [False, True, False])

    def test_out_of_bounds_annotation_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_records({'000123': {'gaussian': {
                'positions': [3], 'timestamps': ['2024-05-22 01:00:00.917'],
            }}}, group_lengths={'000123': 3})

    def test_restricted_loader_reads_numpy_record_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.npy'
            np.save(path, {'000123': {'gaussian': {
                'positions': np.array([1]), 'timestamps': np.array(['2024-05-22 01:00:00.917']),
            }}})
            loaded = validate_records(load_records(path), group_lengths={'000123': 3})
            np.testing.assert_array_equal(loaded['000123']['gaussian']['positions'], [1])


class LegacyPhaseTests(unittest.TestCase):
    def test_level_and_climb_are_declared_approximation_codes(self):
        times = np.arange(6, dtype=float)
        self.assertEqual(phase_ranges_vertical_rate(times, np.full(6, 1000.), 1.), [(0, 6, 0)])
        self.assertEqual(phase_ranges_vertical_rate(times, 1000. + 2. * times, 1.), [(0, 6, 1)])

    def test_nonincreasing_time_is_rejected(self):
        with self.assertRaises(ValueError):
            phase_ranges_vertical_rate(np.array([0., 0., 1.]), np.array([1., 2., 3.]), 1.)


if __name__ == '__main__':
    unittest.main()
