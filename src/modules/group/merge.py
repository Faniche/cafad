"""merge.py | 2026-11-15 | 相邻稀疏硬时空单元的确定性受约束合并 | Codex."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import pandas as pd


CELL_GROUP_COLUMNS = ('cell_key', 'group_id', 'height_band', 'time_bucket', 'longitude_cell', 'latitude_cell',
                      'members', 'reference_flights', 'status')
MERGE_EVENT_COLUMNS = ('step', 'source_group_id', 'target_group_id', 'result_group_id', 'height_band',
                       'source_members', 'target_members', 'result_members', 'source_reference_flights',
                       'target_reference_flights', 'result_reference_flights')


@dataclass(frozen=True)
class CellCluster:
    """由相邻基础硬单元组成的临时专家候选，始终限定在一个高度层。"""

    group_id: str
    cells: frozenset[str]
    height_band: int
    members: int
    flights: frozenset[str]
    time_min: int
    time_max: int
    longitude_min: int
    longitude_max: int
    latitude_min: int
    latitude_max: int

    @property
    def reference_flights(self) -> int:
        """同一原始航班在合并专家内只贡献一次参考。"""
        return len(self.flights)

    def is_valid(self, minimum_members: int, minimum_flights: int) -> bool:
        """最低可训练约束与最终发布规则一致。"""
        return self.members >= minimum_members and self.reference_flights >= minimum_flights

    def reaches_target(self, target_members: int, target_flights: int) -> bool:
        """达到期望支持规模后不再作为贪心合并的发起端。"""
        return self.members >= target_members and self.reference_flights >= target_flights


def merge_hard_cells(normal: pd.DataFrame, *, minimum_members: int, minimum_flights: int,
                     target_members: int, target_flights: int, max_time_buckets: int,
                     max_grid_cells: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """合并同高度层的时空相邻稀疏单元，返回基础键路由、最终组和逐步审计。"""
    _validate_arguments(minimum_members, minimum_flights, target_members, target_flights,
                        max_time_buckets, max_grid_cells)
    clusters, coordinates, positions = _initial_clusters(normal)
    owners = {cell_key: cluster.group_id for cluster in clusters.values() for cell_key in cluster.cells}
    events: list[dict[str, object]] = []
    step = 0
    blocked: set[str] = set()
    while True:
        source = _next_sparse_cluster(clusters, minimum_members, minimum_flights, target_members, target_flights, blocked)
        if source is None:
            break
        candidates = _neighbor_candidates(source, clusters, owners, coordinates, positions, max_time_buckets, max_grid_cells)
        if not candidates:
            blocked.add(source.group_id)
            continue
        target = min(candidates, key=lambda value: _merge_priority(
            source, value, target_members, target_flights, positions,
        ))
        result = _combine(source, target)
        step += 1
        events.append({
            'step': step, 'source_group_id': source.group_id, 'target_group_id': target.group_id,
            'result_group_id': result.group_id, 'height_band': result.height_band,
            'source_members': source.members, 'target_members': target.members, 'result_members': result.members,
            'source_reference_flights': source.reference_flights,
            'target_reference_flights': target.reference_flights,
            'result_reference_flights': result.reference_flights,
        })
        clusters.pop(source.group_id); clusters.pop(target.group_id)
        clusters[result.group_id] = result
        owners.update({cell_key: result.group_id for cell_key in result.cells})
        blocked.discard(source.group_id); blocked.discard(target.group_id)
    groups = _final_groups(clusters, minimum_members, minimum_flights, positions)
    cell_groups = _cell_group_map(clusters, groups, positions)
    return cell_groups, groups, pd.DataFrame(events, columns=MERGE_EVENT_COLUMNS)


def _validate_arguments(minimum_members: int, minimum_flights: int, target_members: int, target_flights: int,
                        max_time_buckets: int, max_grid_cells: int) -> None:
    """拒绝无法满足“目标不低于有效下限”及无意义范围的配置。"""
    if min(minimum_members, minimum_flights, max_time_buckets, max_grid_cells) < 1:
        raise ValueError('硬单元合并的最小支持与最大跨度必须为正整数')
    if target_members < minimum_members or target_flights < minimum_flights:
        raise ValueError('合并目标支持不得小于有效专家的最低支持约束')


def _initial_clusters(normal: pd.DataFrame) -> tuple[dict[str, CellCluster], dict[tuple[int, int, int, int], str],
                                                             dict[str, tuple[int, int, int, int]]]:
    """每个基础 cell_key 初始化一个候选，身份只由正常训练片段决定。"""
    required = {'cell_key', 'height_band', 'time_bucket', 'longitude_cell', 'latitude_cell', 'segment_id', 'flight_id'}
    if not required.issubset(normal.columns):
        raise ValueError('正常训练片段缺少硬单元合并字段')
    clusters: dict[str, CellCluster] = {}
    coordinates: dict[tuple[int, int, int, int], str] = {}
    positions: dict[str, tuple[int, int, int, int]] = {}
    for cell_key, frame in normal.groupby('cell_key', sort=True, observed=True):
        values = frame[['height_band', 'time_bucket', 'longitude_cell', 'latitude_cell']].drop_duplicates()
        if len(values) != 1:
            raise ValueError(f'基础硬单元 {cell_key} 的时空坐标不唯一')
        height_band, time_bucket, longitude_cell, latitude_cell = (int(item) for item in values.iloc[0])
        key = (height_band, time_bucket, longitude_cell, latitude_cell)
        if key in coordinates:
            raise ValueError('不同 cell_key 映射到同一硬时空坐标')
        identity = str(cell_key)
        coordinates[key] = identity
        positions[identity] = key
        clusters[identity] = CellCluster(
            group_id=identity, cells=frozenset((identity,)), height_band=height_band, members=len(frame),
            flights=frozenset(frame['flight_id'].astype(str)), time_min=time_bucket, time_max=time_bucket,
            longitude_min=longitude_cell, longitude_max=longitude_cell,
            latitude_min=latitude_cell, latitude_max=latitude_cell,
        )
    return clusters, coordinates, positions


def _next_sparse_cluster(clusters: dict[str, CellCluster], minimum_members: int, minimum_flights: int,
                         target_members: int, target_flights: int, blocked: set[str]) -> CellCluster | None:
    """固定排序选择一个未达到目标的组；硬无效组优先，确保结果可重放。"""
    candidates = [value for value in clusters.values()
                  if value.group_id not in blocked and not value.reaches_target(target_members, target_flights)]
    if not candidates:
        return None
    return min(candidates, key=lambda value: (
        value.is_valid(minimum_members, minimum_flights), value.members, value.reference_flights, value.group_id,
    ))


def _neighbor_candidates(source: CellCluster, clusters: dict[str, CellCluster], owners: dict[str, str],
                         coordinates: dict[tuple[int, int, int, int], str],
                         positions: dict[str, tuple[int, int, int, int]], max_time_buckets: int,
                         max_grid_cells: int) -> list[CellCluster]:
    """只寻找同高度层且基础网格相邻、合并后跨度不超限的候选。"""
    candidates: dict[str, CellCluster] = {}
    for cell_key in source.cells:
        height_band, time_bucket, longitude_cell, latitude_cell = positions[cell_key]
        for offsets in product((-1, 0, 1), repeat=3):
            candidate_cell = coordinates.get((
                height_band, time_bucket + offsets[0], longitude_cell + offsets[1], latitude_cell + offsets[2],
            ))
            if candidate_cell is None:
                continue
            candidate_id = owners[candidate_cell]
            if candidate_id == source.group_id:
                continue
            candidate = clusters[candidate_id]
            if _within_extent(source, candidate, max_time_buckets, max_grid_cells):
                candidates[candidate_id] = candidate
    return list(candidates.values())


def _within_extent(left: CellCluster, right: CellCluster, max_time_buckets: int, max_grid_cells: int) -> bool:
    """控制最终专家覆盖范围，防止贪心链式扩张跨越遥远时空区域。"""
    return (max(left.time_max, right.time_max) - min(left.time_min, right.time_min) + 1 <= max_time_buckets
            and max(left.longitude_max, right.longitude_max) - min(left.longitude_min, right.longitude_min) + 1 <= max_grid_cells
            and max(left.latitude_max, right.latitude_max) - min(left.latitude_min, right.latitude_min) + 1 <= max_grid_cells)


def _merge_priority(source: CellCluster, target: CellCluster, target_members: int, target_flights: int,
                    positions: dict[str, tuple[int, int, int, int]]) -> tuple[int, int, int, int, str]:
    """优先消除目标支持缺口，再偏好近邻与支持更高的稳定候选。"""
    combined = _combine(source, target)
    shortfall = max(0, target_members - combined.members) + max(0, target_flights - combined.reference_flights)
    distance = min(
        abs(a[1] - b[1]) + abs(a[2] - b[2]) + abs(a[3] - b[3])
        for left_key in source.cells for right_key in target.cells
        for a, b in ((positions[left_key], positions[right_key]),)
    )
    return shortfall, distance, -combined.members, -combined.reference_flights, target.group_id


def _combine(left: CellCluster, right: CellCluster) -> CellCluster:
    """保留字典序最小基础键为稳定最终 group_id，合并支持与外包范围。"""
    if left.height_band != right.height_band:
        raise ValueError('不同高度层不得合并')
    return CellCluster(
        group_id=min(left.group_id, right.group_id), cells=left.cells | right.cells, height_band=left.height_band,
        members=left.members + right.members, flights=left.flights | right.flights,
        time_min=min(left.time_min, right.time_min), time_max=max(left.time_max, right.time_max),
        longitude_min=min(left.longitude_min, right.longitude_min), longitude_max=max(left.longitude_max, right.longitude_max),
        latitude_min=min(left.latitude_min, right.latitude_min), latitude_max=max(left.latitude_max, right.latitude_max),
    )


def _final_groups(clusters: dict[str, CellCluster], minimum_members: int, minimum_flights: int,
                  positions: dict[str, tuple[int, int, int, int]]) -> pd.DataFrame:
    """仅发布达到最低支持约束的最终专家，每组保留规范基础键及覆盖范围。"""
    rows = []
    for cluster in clusters.values():
        if not cluster.is_valid(minimum_members, minimum_flights):
            continue
        coordinate = positions[cluster.group_id]
        rows.append({
            'group_id': cluster.group_id, 'cell_key': cluster.group_id, 'height_band': cluster.height_band,
            'time_bucket': coordinate[1], 'longitude_cell': coordinate[2], 'latitude_cell': coordinate[3],
            'members': cluster.members, 'routing_members': cluster.members,
            'reference_flights': cluster.reference_flights, 'base_cells': len(cluster.cells),
            'min_time_bucket': cluster.time_min, 'max_time_bucket': cluster.time_max,
            'min_longitude_cell': cluster.longitude_min, 'max_longitude_cell': cluster.longitude_max,
            'min_latitude_cell': cluster.latitude_min, 'max_latitude_cell': cluster.latitude_max,
        })
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError('相邻合并后仍没有满足最少正常参考约束的硬时空专家组')
    return result.sort_values(['height_band', 'time_bucket', 'longitude_cell', 'latitude_cell', 'group_id'],
                              kind='stable').reset_index(drop=True)


def _cell_group_map(clusters: dict[str, CellCluster], groups: pd.DataFrame,
                    positions: dict[str, tuple[int, int, int, int]]) -> pd.DataFrame:
    """输出所有基础单元的最终路由或排除状态，供训练期与测试期一致复放。"""
    valid = set(groups['group_id'].astype(str))
    rows = []
    for cluster in clusters.values():
        status = 'merged_or_direct' if cluster.group_id in valid else 'excluded_insufficient_support'
        group_id = cluster.group_id if cluster.group_id in valid else None
        for cell_key in sorted(cluster.cells):
            coordinate = positions[cell_key]
            rows.append({
                'cell_key': cell_key, 'group_id': group_id, 'height_band': coordinate[0], 'time_bucket': coordinate[1],
                'longitude_cell': coordinate[2], 'latitude_cell': coordinate[3], 'members': cluster.members,
                'reference_flights': cluster.reference_flights, 'status': status,
            })
    return pd.DataFrame(rows, columns=CELL_GROUP_COLUMNS).sort_values(
        ['height_band', 'time_bucket', 'longitude_cell', 'latitude_cell', 'cell_key'], kind='stable',
    ).reset_index(drop=True)
