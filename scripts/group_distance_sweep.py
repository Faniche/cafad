"""group_distance_sweep.py | 2026-11-09 | 训练正常片段距离尺度单因素筛选 | Codex."""
from __future__ import annotations
import argparse
from dataclasses import replace
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.artifacts import write_json
from src.modules.group.api import _read_segment_store, _anchors_for_segments, _candidate_pairs, _fit_candidate, _select_candidate, build_parser
from src.modules.group.distance import DistanceConfig, load_distance_config
from src.utils.logging_utils import setup_project_logging, close_project_logging


def distance_matrix(anchors: list[np.ndarray], config: DistanceConfig) -> np.ndarray:
    """按右侧锚点数量分批向量化，保持与 segment_distance 完全相同的对应规则。"""
    count = len(anchors)
    matrix = np.zeros((count, count), dtype=np.float64)
    batches = {}
    for length in sorted({len(value) for value in anchors}):
        indices = np.array([i for i, value in enumerate(anchors) if len(value) == length])
        batches[length] = (indices, np.stack([anchors[i] for i in indices]))
    for i, left in enumerate(anchors):
        for length, (indices, right) in batches.items():
            size = min(len(left), length)
            a = left[np.linspace(0, len(left) - 1, size, dtype=int)]
            b = right[:, np.linspace(0, length - 1, size, dtype=int)]
            dt = np.abs(a[:, 0] - b[:, :, 0])
            if config.time_mode == 'time_of_day':
                dt = np.abs(a[:, 0] % 86400 - b[:, :, 0] % 86400)
                dt = np.minimum(dt, 86400 - dt)
            hav = np.sin((a[:, 2] - b[:, :, 2]) / 2) ** 2 + np.cos(a[:, 2]) * np.cos(b[:, :, 2]) * np.sin((a[:, 1] - b[:, :, 1]) / 2) ** 2
            ds = 2 * config.earth_radius_m * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))
            dh = np.abs(a[:, 3] - b[:, :, 3]); dh = np.minimum(dh, 2 * np.pi - dh)
            matrix[i, indices] = np.sqrt((dt / config.sigma_time_s) ** 2 + (ds / config.sigma_space_m) ** 2
                                          + (dh / config.sigma_heading_rad) ** 2).mean(axis=1)
    np.fill_diagonal(matrix, 0)
    return matrix


def run(input_path: Path, config_path: Path, output: Path, logger: logging.Logger) -> None:
    base = load_distance_config(config_path)
    arguments = build_parser().parse_args(['--input', str(input_path), '--distance-config', str(config_path)])
    segments, values = _read_segment_store(input_path)
    normal = segments.loc[segments['split'].eq('train') & segments['segment_label'].eq(0) & ~segments['phase'].eq('NA')]
    variants = [('baseline', 8, base), ('anchors4', 4, base), ('anchors16', 16, base)]
    for field in ('sigma_time_s', 'sigma_space_m', 'sigma_heading_rad'):
        for factor in (.5, 2.):
            variants.append((f'{field}_x{factor}', 8, replace(base, **{field: getattr(base, field) * factor})))
    rows = []
    cached = {}
    for name, maximum, config in variants:
        if maximum not in cached:
            cached[maximum] = _anchors_for_segments(normal, values.loc[values.segment_id.isin(normal.segment_id)], maximum)
        anchors = cached[maximum]
        total, groups, largest = 0, 0, 0.
        details = []
        for phase, frame in normal.groupby('phase', sort=True):
            frame = frame.sort_values(['flight_id', 'ordinal', 'segment_id'], kind='stable').reset_index(drop=True)
            matrix = distance_matrix([anchors[str(key)] for key in frame.segment_id], config)
            candidates = [_fit_candidate(str(phase), frame, matrix, size, samples, arguments.min_group_members,
                                         arguments.min_reference_flights) for size, samples in _candidate_pairs(arguments) if size <= len(frame)]
            if not candidates:
                continue
            selected = _select_candidate(candidates, arguments.max_group_fraction)
            covered = sum(len(item[1]) for item in selected.effective_clusters)
            total += covered; groups += len(selected.effective_clusters)
            largest = max(largest, selected.largest_group_fraction)
            details.append({'phase': str(phase), 'coverage': selected.coverage, 'groups': len(selected.effective_clusters),
                            'min_cluster_size': selected.min_cluster_size, 'min_samples': selected.min_samples})
            logger.info('距离实验：%s phase=%s coverage=%.4f groups=%d', name, phase, selected.coverage, len(selected.effective_clusters))
        rows.append({'name': name, 'anchors': maximum, 'sigma_time_s': config.sigma_time_s,
                     'sigma_space_m': config.sigma_space_m, 'sigma_heading_rad': config.sigma_heading_rad,
                     'normal_segments': len(normal), 'covered': total, 'coverage': total / len(normal),
                     'groups': groups, 'largest_phase_group_fraction': largest, 'phases': details})
        write_json(output, {'input': str(input_path), 'protocol': 'train-normal-only one-factor screening; no test/val tuning', 'results': rows})
        logger.info('距离实验完成：%s coverage=%.4f groups=%d', name, total / len(normal), groups)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--distance-config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    logger, _ = setup_project_logging(project_root=ROOT, name='cafad.distance_sweep', console=True)
    try:
        run(args.input, args.distance_config, args.output, logger)
    finally:
        close_project_logging(logger)


if __name__ == '__main__':
    main()
