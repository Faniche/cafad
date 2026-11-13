"""inference.py | 2026-11-09 | 无标签路由及冻结专家评分，供校准和检测复用 | Codex."""
from pathlib import Path
import json
import logging
import pandas as pd
from src.app.artifacts import read_manifest
from src.app.tables import read_table
from src.modules.group.api import _read_segment_store, read_segment_store_data
from src.modules.group.features import (FEATURE_NAMES, build_segment_features, centroids_by_group,
                                        feature_scaler_from_dict, transform_features)
from src.modules.train.api import _group_manifest, _segment_manifest, _load_sequences, _load_sequences_from_data, _devices
from .scoring import score_pairs
from .fusion import fuse_candidates, fallback_score


def upstream(path: Path, module: str) -> Path:
    """从直接依赖中恢复唯一模块，校验其产物。"""
    value = read_manifest(path)
    matches = [Path(item['manifest']) for item in value['dependencies']
               if read_manifest(Path(item['manifest']))['module'] == module]
    if len(matches) != 1:
        raise ValueError(f'无法唯一溯源 {module}：{path}')
    return matches[0]


def _routing_pairs(targets: pd.DataFrame, groups: pd.DataFrame, centers: dict[str, object],
                   vectors_by_segment: pd.DataFrame, top_r: int) -> pd.DataFrame:
    """以冻结组中心为目标片段生成同阶段 top-r 候选；标签不参与路由。"""
    pairs: list[dict[str, object]] = []
    for row in targets.itertuples(index=False):
        same = groups.loc[groups['phase'].eq(row.phase), 'group_id'].tolist()
        fallback = not same
        vector = vectors_by_segment.loc[row.segment_id, list(FEATURE_NAMES)].to_numpy(dtype=float)
        distances = [(group, float(((vector - centers[group]) ** 2).sum() ** .5))
                     for group in (same or groups['group_id'].tolist())]
        selected = sorted(distances, key=lambda item: (item[1], item[0]))
        if not fallback:
            selected = selected[:top_r]
        pairs.extend({'segment_id': row.segment_id, 'group_id': group, 'distance': distance,
                      'route_mode': 'fallback' if fallback else 'same_phase'} for group, distance in selected)
    return pd.DataFrame(pairs)


def score_partition(trained: Path, split: str, top_r: int, temperature: float, gpus: str,
                    logger: logging.Logger, workers: int = 1, task_size: int = 32,
                    target_data: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """用注入后的目标序列评分；验证沿用干净边界，测试由 detect 传入动态分段结果。"""
    read_manifest(trained, expected_module='train')
    injected = upstream(trained, 'inject')
    grouped = _group_manifest(injected)
    segmented = _segment_manifest(grouped)
    source_data = target_data or injected.parent / 'data'
    metadata = read_table(source_data, 'segments', columns=['segment_id', 'flight_id', 'phase', 'split', 'segment_label'],
                          keep_default_na=False)
    targets = metadata.loc[metadata['split'].eq(split)].copy()
    if targets.empty:
        raise ValueError(f'{split} 分区没有片段')
    sequences, _ = (_load_sequences_from_data(source_data) if target_data is not None
                    else _load_sequences(segmented, injected))
    routing = read_table(grouped.parent / 'data', 'routing_segments', keep_default_na=False)
    groups = read_table(grouped.parent / 'data', 'groups', keep_default_na=False)
    clean_rows, clean_values = _read_segment_store(segmented)
    scaler = feature_scaler_from_dict(json.loads((grouped.parent / 'data/feature_scaler.json').read_text(encoding='utf-8')))
    clean_features = build_segment_features(clean_rows, clean_values, scaler.time_origin_ns)
    features = clean_features
    if target_data is not None:
        target_rows, target_values = read_segment_store_data(target_data)
        features = build_segment_features(target_rows, target_values, scaler.time_origin_ns)
    vectors = transform_features(features, scaler)
    vectors.insert(0, 'segment_id', features['segment_id'].astype(str))
    vectors_by_segment = vectors.set_index('segment_id', verify_integrity=True)
    clean_vectors = transform_features(clean_features, scaler)
    clean_vectors.insert(0, 'segment_id', clean_features['segment_id'].astype(str))
    centers = centroids_by_group(routing, clean_vectors.set_index('segment_id', verify_integrity=True))
    scores = _routing_pairs(targets, groups, centers, vectors_by_segment, top_r)
    scores = score_pairs(scores, sequences, trained, _devices(gpus), workers, task_size, logger)
    fused = []
    for key, frame in scores.groupby('segment_id', sort=False):
        mode = frame['route_mode'].iloc[0]
        value = fallback_score(frame['score'].to_numpy()) if mode == 'fallback' else fuse_candidates(
            frame['score'].to_numpy(), frame['distance'].to_numpy(), temperature)
        fused.append({'segment_id': key, 'score': value, 'route_mode': mode, 'experts': len(frame)})
    return targets.merge(pd.DataFrame(fused), on='segment_id', validate='one_to_one'), scores, segmented
