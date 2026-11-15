"""inference.py | 2026-11-15 | 硬时空单元的专家路由和评分融合 | Codex."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.app.artifacts import read_manifest
from src.app.tables import read_table
from src.modules.train.api import _devices, _group_manifest, _load_sequences, _load_sequences_from_data, _segment_manifest

from .fusion import fallback_score
from .scoring import score_pairs


def upstream(path: Path, module: str) -> Path:
    """沿 manifest 依赖恢复唯一上游，避免在后续模块重复暴露早期输入。"""
    manifest = read_manifest(path)
    matches = [Path(item['manifest']) for item in manifest.get('dependencies', [])
               if read_manifest(Path(item['manifest'])).get('module') == module]
    if len(matches) != 1:
        raise ValueError(f'无法唯一溯源 {module}：{path}')
    return matches[0]


def _routing_pairs(targets: pd.DataFrame, groups: pd.DataFrame, cell_groups: pd.DataFrame) -> pd.DataFrame:
    """优先经基础单元映射命中合并专家；未知单元才交给全部专家稳健回退。"""
    available = set(groups['group_id'].astype(str))
    all_groups = sorted(available)
    if not all_groups:
        raise ValueError('没有已训练的硬时空专家可供评分')
    mapping = (cell_groups.loc[cell_groups['group_id'].notna(), ['cell_key', 'group_id']]
               .set_index('cell_key')['group_id'].astype(str).to_dict())
    rows: list[dict[str, object]] = []
    for target in targets.itertuples(index=False):
        cell_key = str(target.cell_key)
        group_id = mapping.get(cell_key)
        candidates = [group_id] if group_id in available else all_groups
        mode = 'hard_exact' if group_id in available else 'fallback_all_groups'
        rows.extend({'segment_id': str(target.segment_id), 'group_id': group_id, 'distance': 0.0,
                     'route_mode': mode} for group_id in candidates)
    return pd.DataFrame(rows)


def score_partition(trained: Path, split: str, gpus: str,
                    logger: logging.Logger, workers: int = 1, task_size: int = 32,
                    target_data: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, Path | None]:
    """评分训练期验证集或动态测试集；专家只按硬单元精确路由或统一回退。"""
    injected = upstream(trained, 'inject')
    grouped = _group_manifest(injected)
    segmented = _segment_manifest(grouped)
    source_data = target_data or injected.parent / 'data'
    metadata = read_table(source_data, 'segments', columns=['segment_id', 'flight_id', 'cell_key', 'split', 'segment_label'],
                          keep_default_na=False, dtype={'segment_id': 'string', 'flight_id': 'string'})
    targets = metadata.loc[metadata['split'].eq(split)].copy()
    if targets.empty:
        raise ValueError(f'{split} 分区没有片段')
    sequences, _ = (_load_sequences_from_data(source_data) if target_data is not None else _load_sequences(segmented, injected))
    group_data = grouped.parent / 'data'
    groups = read_table(group_data, 'groups', columns=['group_id'], keep_default_na=False)
    try:
        cell_groups = read_table(group_data, 'cell_group_map', columns=['cell_key', 'group_id'],
                                 keep_default_na=False, dtype={'cell_key': 'string', 'group_id': 'string'})
    except FileNotFoundError:
        # 旧硬分组产物尚未发布基础单元映射时，保留其 group_id=cell_key 的原有精确路由语义。
        cell_groups = read_table(group_data, 'groups', columns=['cell_key', 'group_id'],
                                 keep_default_na=False, dtype={'cell_key': 'string', 'group_id': 'string'})
    pairs = _routing_pairs(targets, groups, cell_groups)
    scores = score_pairs(pairs, sequences, trained.parent, _devices(gpus), workers, task_size, logger)
    fused: list[dict[str, object]] = []
    for segment_id, frame in scores.groupby('segment_id', sort=False):
        mode = str(frame['route_mode'].iloc[0])
        value = float(frame['score'].iloc[0]) if mode == 'hard_exact' else fallback_score(frame['score'].to_numpy())
        fused.append({'segment_id': str(segment_id), 'score': value, 'route_mode': mode, 'experts': int(len(frame))})
    result = targets.merge(pd.DataFrame(fused), on='segment_id', validate='one_to_one')
    return result, scores, target_data
