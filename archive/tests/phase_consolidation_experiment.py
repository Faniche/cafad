"""phase_consolidation_experiment.py | 2026-11-12 | Sun 阶段时序合并诊断 | Codex.

这是离线诊断脚本：仅读取一个已发布 segment 产物，在较短的 Sun
分类窗口上试验保守的短片段桥接规则；不会覆盖正式阶段结果。
"""
from __future__ import annotations

import argparse
from concurrent.futures import as_completed
import json
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from src.app.artifacts import read_manifest
from src.app.tables import read_table
from src.data_processing.sun_phase import identify_sun_phases
from src.utils.process_pool import InterruptibleProcessPoolExecutor


PHASE_ORDER = ('GND', 'CL', 'LVL', 'CR', 'DE')
UNUSUAL_TRANSITIONS = {
    ('GND', 'CR'), ('GND', 'LVL'), ('CL', 'GND'), ('CR', 'GND'), ('CR', 'DE'),
}


@dataclass(frozen=True)
class ConsolidationConfig:
    """保守合并规则；单位均为秒或米每秒。"""

    window_seconds: float
    bridge_seconds: float
    stable_vertical_rate_mps: float
    high_altitude_quantile: float


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='诊断 Sun 阶段切分与保守时序合并，不修改正式产物。')
    parser.add_argument('--segment-manifest', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--window-seconds', type=float, default=60.0)
    parser.add_argument('--bridge-seconds', type=float, default=180.0)
    parser.add_argument('--stable-vertical-rate-mps', type=float, default=0.5)
    parser.add_argument('--high-altitude-quantile', type=float, default=0.60)
    parser.add_argument('--limit-flights', type=int, default=0, help='0 表示处理全部可识别航班。')
    parser.add_argument('--workers', type=int, default=30, help='航班级诊断进程数上限（默认 30）。')
    return parser


def _load_observations(manifest_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = read_manifest(manifest_path, expected_module='segment')
    if manifest.get('status') != 'SUCCEEDED':
        raise ValueError('只接受已成功发布的 segment manifest')
    data = manifest_path.parent / 'data'
    segments = read_table(data, 'segments', dtype={'segment_id': 'string', 'flight_id': 'string'})
    mapping = read_table(data, 'point_segment_map', dtype={'record_id': 'string', 'flight_id': 'string'})
    observations = read_table(
        data, 'input_observations',
        columns={'record_id', 'time_ns', 'longitude_deg', 'latitude_deg', 'geometric_altitude_m', 'heading_deg', 'speed_mps'},
        dtype={'record_id': 'string'},
    )
    values = mapping[['record_id', 'flight_id', 'time_ns']].merge(observations, on=['record_id', 'time_ns'], how='left', validate='one_to_one')
    if values.isna().any().any():
        raise ValueError('阶段映射无法完整回接最终观测')
    return segments, mapping, values


def _runs_from_labels(flight: pd.DataFrame, labels: np.ndarray, *, prefix: str) -> pd.DataFrame:
    """按给定点标签生成原子阶段片段，并保留合并判据需要的稳健统计量。"""
    if len(flight) != len(labels):
        raise ValueError('标签与航班观测长度不一致')
    cuts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    rows: list[dict[str, object]] = []
    for ordinal, (start, end) in enumerate(zip(cuts[:-1], cuts[1:], strict=True)):
        frame = flight.iloc[start:end]
        times = frame['time_ns'].to_numpy(dtype=np.int64)
        altitude = frame['geometric_altitude_m'].to_numpy(dtype=np.float64)
        speed = frame['speed_mps'].to_numpy(dtype=np.float64)
        elapsed = np.diff(times).astype(np.float64) / 1_000_000_000.0
        rate = np.diff(altitude) / elapsed if len(elapsed) else np.zeros(1, dtype=np.float64)
        rows.append({
            'source': prefix,
            'ordinal': ordinal,
            'phase': str(labels[start]),
            'points': int(end - start),
            'start_index': int(start),
            'end_index_exclusive': int(end),
            'start_time_ns': int(times[0]),
            'end_time_ns': int(times[-1]),
            'duration_seconds': float((times[-1] - times[0]) / 1_000_000_000.0),
            'mean_altitude_m': float(altitude.mean()),
            'median_vertical_rate_mps': float(np.median(rate)),
            'mean_speed_mps': float(speed.mean()),
        })
    return pd.DataFrame(rows)


def _can_bridge(left: pd.Series, middle: pd.Series, right: pd.Series, config: ConsolidationConfig,
                high_altitude_m: float) -> bool:
    """仅消除有相邻同阶段支持、且运动学不支持中间标签的短孤岛。"""
    if left['phase'] != right['phase'] or middle['duration_seconds'] > config.bridge_seconds:
        return False
    target = str(left['phase'])
    rate = float(middle['median_vertical_rate_mps'])
    altitude = float(middle['mean_altitude_m'])
    if target in {'CR', 'LVL'}:
        return abs(rate) <= config.stable_vertical_rate_mps and abs(float(left['mean_altitude_m']) - altitude) <= 1000.0
    if target == 'CL':
        return rate >= config.stable_vertical_rate_mps
    if target == 'DE':
        return rate <= -config.stable_vertical_rate_mps
    if target == 'GND':
        return altitude < high_altitude_m * 0.1 and abs(rate) <= config.stable_vertical_rate_mps
    return False


def _merge_rows(rows: pd.DataFrame, indices: list[int], *, phase: str) -> dict[str, object]:
    selected = rows.iloc[indices]
    return {
        'source': 'consolidated',
        'ordinal': -1,
        'phase': phase,
        'points': int(selected['points'].sum()),
        'start_index': int(selected['start_index'].iloc[0]),
        'end_index_exclusive': int(selected['end_index_exclusive'].iloc[-1]),
        'start_time_ns': int(selected['start_time_ns'].iloc[0]),
        'end_time_ns': int(selected['end_time_ns'].iloc[-1]),
        'duration_seconds': float(selected['duration_seconds'].sum()),
        'mean_altitude_m': float(np.average(selected['mean_altitude_m'], weights=selected['points'])),
        'median_vertical_rate_mps': float(np.median(selected['median_vertical_rate_mps'])),
        'mean_speed_mps': float(np.average(selected['mean_speed_mps'], weights=selected['points'])),
    }


def consolidate_runs(runs: pd.DataFrame, config: ConsolidationConfig, high_altitude_m: float) -> pd.DataFrame:
    """反复桥接短孤岛；不跨越 NA，也不直接合并不同且均有运动学支持的阶段。"""
    work = runs.copy().reset_index(drop=True)
    changed = True
    while changed and len(work) >= 3:
        changed = False
        for middle_index in range(1, len(work) - 1):
            left, middle, right = work.iloc[middle_index - 1], work.iloc[middle_index], work.iloc[middle_index + 1]
            if middle['phase'] == 'NA' or not _can_bridge(left, middle, right, config, high_altitude_m):
                continue
            replacement = _merge_rows(work, [middle_index - 1, middle_index, middle_index + 1], phase=str(left['phase']))
            work = pd.concat((work.iloc[:middle_index - 1], pd.DataFrame([replacement]), work.iloc[middle_index + 2:]), ignore_index=True)
            changed = True
            break
    work['ordinal'] = np.arange(len(work), dtype=np.int64)
    return work


def _sequence_audit(phases: list[str]) -> tuple[bool, str, str]:
    repeated = len(phases) != len(dict.fromkeys(phases))
    unusual = [f'{left}>{right}' for left, right in zip(phases[:-1], phases[1:]) if (left, right) in UNUSUAL_TRANSITIONS]
    return repeated, '|'.join(unusual), '>'.join(phases)


def _analyse_flight(flight_id: str, flight: pd.DataFrame, existing: dict[str, object],
                    config: ConsolidationConfig) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """子进程独立重识别一条航班；返回可安全由父进程统一写入的结果。"""
    flight = flight.sort_values('time_ns', kind='stable').reset_index(drop=True)
    times = (flight['time_ns'].to_numpy(dtype=np.int64) - int(flight['time_ns'].iloc[0])) / 1_000_000_000.0
    labels = identify_sun_phases(
        times, flight['geometric_altitude_m'].to_numpy(dtype=np.float64) / .3048,
        flight['speed_mps'].to_numpy(dtype=np.float64) * 3600.0 / 1852.0,
        window_seconds=config.window_seconds,
    )
    atomic = _runs_from_labels(flight, labels, prefix='sun_atomic')
    high_altitude_m = float(flight['geometric_altitude_m'].quantile(config.high_altitude_quantile))
    consolidated = consolidate_runs(atomic, config, high_altitude_m)
    before_repeat, before_unusual, before_sequence = _sequence_audit(atomic['phase'].tolist())
    after_repeat, after_unusual, after_sequence = _sequence_audit(consolidated['phase'].tolist())
    row = {
        'flight_id': str(flight_id), 'icao': str(existing['icao']), 'call': str(existing['call']),
        'points': int(len(flight)), 'existing_segments': int(existing['segments']),
        'existing_sequence': str(existing['sequence']), 'atomic_segments': int(len(atomic)),
        'atomic_sequence': before_sequence, 'consolidated_segments': int(len(consolidated)),
        'consolidated_sequence': after_sequence, 'reduced_segments': int(len(atomic) - len(consolidated)),
        'atomic_repeated_label': before_repeat, 'consolidated_repeated_label': after_repeat,
        'atomic_unusual_transitions': before_unusual, 'consolidated_unusual_transitions': after_unusual,
    }
    for kind, table in (('atomic', atomic), ('consolidated', consolidated)):
        table.insert(0, 'flight_id', str(flight_id))
        table.insert(1, 'icao', row['icao'])
        table.insert(2, 'call', row['call'])
        table.insert(3, 'kind', kind)
    return row, atomic, consolidated


def run_experiment(arguments: argparse.Namespace) -> dict[str, object]:
    if (not 0 < arguments.high_altitude_quantile < 1
            or min(arguments.window_seconds, arguments.bridge_seconds) <= 0 or arguments.workers < 1):
        raise ValueError('窗口、桥接时长必须为正，且高度分位数必须位于 (0,1)')
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    segments, _, values = _load_observations(arguments.segment_manifest.resolve())
    existing_by_flight = {
        str(flight_id): {'icao': str(part['icao'].iloc[0]), 'call': str(part['call'].iloc[0]),
                         'segments': len(part), 'sequence': '>'.join(part['phase'].astype(str))}
        for flight_id, part in segments.sort_values(['flight_id', 'ordinal'], kind='stable').groupby('flight_id', sort=True)
    }
    config = ConsolidationConfig(arguments.window_seconds, arguments.bridge_seconds,
                                 arguments.stable_vertical_rate_mps, arguments.high_altitude_quantile)
    sequence_rows: list[dict[str, object]] = []
    run_rows: list[pd.DataFrame] = []
    failed_rows: list[dict[str, object]] = []
    tasks = list(values.sort_values(['flight_id', 'time_ns'], kind='stable').groupby('flight_id', sort=True))
    if arguments.limit_flights:
        tasks = tasks[:arguments.limit_flights]
    if not tasks:
        raise ValueError('没有可识别航班可供阶段合并诊断')
    with InterruptibleProcessPoolExecutor(max_workers=min(arguments.workers, len(tasks)), mp_context=mp.get_context('spawn')) as pool:
        futures = {
            pool.submit(_analyse_flight, str(flight_id), flight,
                        existing_by_flight.get(str(flight_id), {'icao': '', 'call': '', 'segments': 0, 'sequence': ''}), config): str(flight_id)
            for flight_id, flight in tasks
        }
        for future in as_completed(futures):
            flight_id = futures[future]
            try:
                row, atomic, consolidated = future.result()
            except ValueError as exc:
                existing = existing_by_flight.get(flight_id, {'icao': '', 'call': ''})
                failed_rows.append({'flight_id': flight_id, 'icao': existing['icao'], 'call': existing['call'],
                                    'reason': f'{type(exc).__name__}: {exc}'})
                continue
            sequence_rows.append(row)
            run_rows.extend((atomic, consolidated))
    sequence = pd.DataFrame(sequence_rows).sort_values(['reduced_segments', 'atomic_segments'], ascending=False, kind='stable')
    runs = pd.concat(run_rows, ignore_index=True) if run_rows else pd.DataFrame()
    sequence.to_csv(output / 'flight_sequences.csv', index=False)
    runs.to_csv(output / 'candidate_runs.csv', index=False)
    changed = sequence.loc[sequence['reduced_segments'].gt(0)].copy()
    changed.to_csv(output / 'changed_flights.csv', index=False)
    pd.DataFrame(failed_rows, columns=('flight_id', 'icao', 'call', 'reason')).to_csv(output / 'failed_flights.csv', index=False)
    summary = {
        'segment_manifest': str(arguments.segment_manifest.resolve()),
        'config': {key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()},
        'flights': int(len(sequence)),
        'failed_flights': int(len(failed_rows)),
        'existing_segments': int(sequence['existing_segments'].sum()),
        'atomic_segments': int(sequence['atomic_segments'].sum()),
        'consolidated_segments': int(sequence['consolidated_segments'].sum()),
        'changed_flights': int(len(changed)),
        'reduced_segments': int(sequence['reduced_segments'].sum()),
        'unusual_transition_flights_before': int(sequence['atomic_unusual_transitions'].ne('').sum()),
        'unusual_transition_flights_after': int(sequence['consolidated_unusual_transitions'].ne('').sum()),
    }
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'README.md').write_text(
        '# 阶段合并诊断\n\n'
        '本目录是离线试验，不修改正式 segment 产物。`flight_sequences.csv` 每行一条 flight instance；'
        '`candidate_runs.csv` 给出 60 秒 Sun 原子片段和保守桥接后的片段。桥接仅处理“相邻两端阶段相同、'
        '中间段较短且其运动学不支持中间标签”的情况；不能据此直接作为正式规则。\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    print(json.dumps(run_experiment(_argument_parser().parse_args()), ensure_ascii=False, indent=2))
