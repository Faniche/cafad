"""inference.py | 2026-11-09 | 无标签路由及冻结专家评分，供校准和检测复用 | Codex."""
from pathlib import Path
import json
import logging
import pandas as pd
from src.app.artifacts import read_manifest
from src.modules.group.distance import load_distance_config, sample_anchors, segment_distance
from src.modules.train.api import _segment_manifest, _load_sequences, _devices
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


def score_partition(trained: Path, split: str, top_r: int, temperature: float, gpus: str,
                    logger: logging.Logger, workers: int = 1, task_size: int = 32) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """前向只读观测、ID、阶段和分区，绝不读取目标标签；按专家加载以限制显存。"""
    read_manifest(trained, expected_module='train')
    grouped = upstream(trained, 'group')
    segmented = _segment_manifest(grouped)
    metadata = pd.read_csv(segmented.parent / 'data/segments.csv',
                           usecols=['segment_id', 'flight_id', 'phase', 'split'], keep_default_na=False)
    targets = metadata.loc[metadata['split'].eq(split)].copy()
    if targets.empty:
        raise ValueError(f'{split} 分区没有片段')
    sequences, _ = _load_sequences(segmented)
    config = load_distance_config(grouped.parent / 'distance_config.yaml')
    resolved = json.loads((grouped.parent / 'resolved_config.json').read_text(encoding='utf-8'))
    maximum = int(resolved['module']['anchors'])
    routing = pd.read_csv(grouped.parent / 'data/routing_segments.csv', keep_default_na=False)
    groups = pd.read_csv(grouped.parent / 'data/groups.csv', keep_default_na=False)
    ids = set(targets['segment_id']) | set(routing['segment_id'])
    anchors = {key: sample_anchors(sequences[key]['time_ns'].to_numpy(), sequences[key]['longitude_deg'].to_numpy(),
                                  sequences[key]['latitude_deg'].to_numpy(), sequences[key]['heading_deg'].to_numpy(), maximum)
               for key in ids}
    members = routing.groupby('group_id')['segment_id'].apply(list).to_dict()
    pairs = []
    for row in targets.itertuples(index=False):
        same = groups.loc[groups['phase'].eq(row.phase), 'group_id'].tolist()
        fallback = not same
        distances = [(group, min(segment_distance(anchors[row.segment_id], anchors[key], config) for key in members[group]))
                     for group in (same or groups['group_id'].tolist())]
        selected = sorted(distances, key=lambda item: (item[1], item[0]))
        if not fallback:
            selected = selected[:top_r]
        pairs.extend({'segment_id': row.segment_id, 'group_id': group, 'distance': distance,
                      'route_mode': 'fallback' if fallback else 'same_phase'} for group, distance in selected)
    scores = pd.DataFrame(pairs)
    scores = score_pairs(scores, sequences, trained, _devices(gpus), workers, task_size, logger)
    fused = []
    for key, frame in scores.groupby('segment_id', sort=False):
        mode = frame['route_mode'].iloc[0]
        value = fallback_score(frame['score'].to_numpy()) if mode == 'fallback' else fuse_candidates(
            frame['score'].to_numpy(), frame['distance'].to_numpy(), temperature)
        fused.append({'segment_id': key, 'score': value, 'route_mode': mode, 'experts': len(frame)})
    return targets.merge(pd.DataFrame(fused), on='segment_id', validate='one_to_one'), scores, segmented
