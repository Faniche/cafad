"""api.py | 2026-11-06 | 基于航班级事件的异常注入 | Codex."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, wait
from src.utils.process_pool import InterruptibleProcessPoolExecutor as ProcessPoolExecutor
from contextlib import nullcontext
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.app.arguments import ProjectArgumentParser
from src.app.artifacts import hash_file, read_manifest
from src.app.contracts import ModuleResult, RunContext, ValidationReport
from src.data_processing.phase_pipeline import configure_worker_threads
from src.modules.preprocess.canonical import CANONICAL_COLUMNS
from src.modules.split.api import SPLIT_NAMES, TRAJECTORY_KINDS
from src.utils.env import Env

from .corruption import (CorruptionPlan, apply_corruption_plan, flight_seed, load_protocol,
                         minimum_required_points, prepare_corruption_plan)
from .provenance import (EVENT_COLUMNS, IMPACT_COLUMNS, LABEL_COLUMNS, event_rows, final_labels,
                         impact_rows, normal_record_labels, record_labels, validate_provenance)


OBSERVATION_COLUMNS = CANONICAL_COLUMNS + ('split', 'injection_plan', 'injection_outcome', 'injection_skip_reason')
OUTCOME_COLUMNS = ('flight_id', 'split', 'injection_plan', 'injection_outcome', 'injection_skip_reason')


def build_parser() -> argparse.ArgumentParser:
    """定义单副本合成异常注入所需的持久化输入。"""
    parser = ProjectArgumentParser(
        prog='main.py -m inject',
        description='按 split 计划和显式协议在整条合格航班内生成唯一最终观测版本。',
    )
    parser.add_argument('--input', required=True, type=Path, help='preprocess 模块已发布的 manifest.json')
    parser.add_argument('--split-manifest', required=True, type=Path, help='对应 preprocess 的 split 模块 manifest.json')
    parser.add_argument('--protocol', required=True, type=Path,
                        help='定义事件类型、概率、长度、强度与平滑的 schema_version=2 JSON/YAML 注入配置')
    return parser


def _project_path(context: RunContext, value: Path) -> Path:
    return value if value.is_absolute() else context.project_root / value


def _read_inputs(context: RunContext, args: argparse.Namespace) -> tuple[Path, Path, Path, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """读取并交叉核验 preprocess、split 与显式注入协议。"""
    input_manifest = _project_path(context, args.input).resolve()
    split_manifest = _project_path(context, args.split_manifest).resolve()
    protocol_path = _project_path(context, args.protocol).resolve()
    preprocess_manifest = read_manifest(input_manifest, expected_module='preprocess')
    split_manifest_value = read_manifest(split_manifest, expected_module='split')
    if any(manifest.get('dataset_id') != context.dataset_id for manifest in (preprocess_manifest, split_manifest_value)):
        raise ValueError('preprocess 与 split 必须和本次运行的数据集身份一致')
    if not protocol_path.is_file():
        raise FileNotFoundError(f'找不到注入协议：{protocol_path}')
    expected_hash = hash_file(input_manifest)
    dependencies = split_manifest_value.get('dependencies')
    if not isinstance(dependencies, list) or not any(
            isinstance(item, dict) and item.get('sha256') == expected_hash for item in dependencies):
        raise ValueError('split 清单未声明当前 preprocess manifest 作为已校验依赖')
    observations = pd.read_csv(input_manifest.parent / 'data' / 'observations.csv',
                               dtype={'record_id': 'string', 'flight_id': 'string', 'icao': 'string', 'call': 'string'})
    splits = pd.read_csv(split_manifest.parent / 'data' / 'splits.csv', dtype={'flight_id': 'string'})
    if tuple(observations.columns) != CANONICAL_COLUMNS or observations.empty:
        raise ValueError('preprocess 规范观测字段不符合约定或为空')
    required_split_columns = {'flight_id', 'split', 'injection_plan', 'injection_eligible', 'injection_skip_reason'}
    if splits.empty or splits['flight_id'].duplicated().any() or not required_split_columns.issubset(splits.columns):
        raise ValueError('split 表为空、航班 ID 重复或缺少注入资格字段')
    source_ids = set(observations['flight_id'])
    if source_ids != set(splits['flight_id']) or observations['flight_id'].isna().any():
        raise ValueError('split 表必须恰好覆盖 preprocess 的全部来源航班')
    if set(splits['split']) != set(SPLIT_NAMES) or set(splits['injection_plan']) - set(TRAJECTORY_KINDS):
        raise ValueError('split 表包含未知分区或注入计划')
    protocol = load_protocol(protocol_path)
    configured_minimum = Env.get_int('injection_minimum_trajectory_points')
    required_minimum = minimum_required_points(protocol)
    if configured_minimum < required_minimum:
        raise ValueError('injection_minimum_trajectory_points 小于协议最长事件及上下文所需点数')
    return input_manifest, split_manifest, protocol_path, observations, splits, protocol


def run(context: RunContext, args: argparse.Namespace) -> ModuleResult:
    """按固定 split 注入单个事件，写入最终观测、标签、事件和影响范围账本。"""
    input_manifest, split_manifest, protocol_path, observations, splits, protocol = _read_inputs(context, args)
    planned_injections = int((splits['injection_plan'].eq('abnormal') & splits['injection_eligible']).sum())
    injection_workers = min(context.workers, planned_injections)
    context.logger.debug('注入输入：preprocess=%s；split=%s；协议=%s；观测=%d；航班=%d；协议事件=%s',
                         input_manifest, split_manifest, protocol_path, len(observations), len(splits),
                         list(protocol['event_type_probabilities']))
    context.logger.info('开始航班级异常注入：eligible_abnormal_flights=%d，workers=%d，队列上限=%d',
                        planned_injections, injection_workers, injection_workers * 2)
    final_observations, label_table, event_table, impact_table, outcome_table = _inject_observations(
        observations, splits, protocol, context.seed, injection_workers, logger=context.logger,
    )
    data_dir = context.staging_dir / 'data'
    data_dir.mkdir()
    final_observations.to_csv(data_dir / 'observations.csv', index=False, encoding='utf-8')
    label_table.to_csv(data_dir / 'record_labels.csv', index=False, encoding='utf-8')
    event_table.to_csv(data_dir / 'injection_events.csv', index=False, encoding='utf-8')
    impact_table.to_csv(data_dir / 'injection_impacts.csv', index=False, encoding='utf-8')
    outcome_table.to_csv(data_dir / 'injection_outcomes.csv', index=False, encoding='utf-8')
    actual_flight_labels = final_observations.groupby('flight_id', sort=False)['injected_label'].max()
    outcome_counts = outcome_table['injection_outcome'].value_counts().to_dict()
    summary: dict[str, object] = {
        'input_manifest': str(input_manifest), 'split_manifest': str(split_manifest), 'protocol': str(protocol_path),
        'protocol_sha256': hash_file(protocol_path), 'flight_gap_seconds': _flight_gap_seconds(input_manifest),
        'injection_minimum_trajectory_points': Env.get_int('injection_minimum_trajectory_points'),
        'minimum_event_support_points': minimum_required_points(protocol),
        'injection_workers': injection_workers,
        'flights': int(len(actual_flight_labels)), 'observations': int(len(final_observations)),
        'abnormal_flights': int(actual_flight_labels.sum()), 'normal_flights': int(len(actual_flight_labels) - actual_flight_labels.sum()),
        'anomalous_records': int(final_observations['injected_label'].sum()),
        'normal_records': int(len(final_observations) - final_observations['injected_label'].sum()),
        'events': int(len(event_table)), 'outcomes': {str(key): int(value) for key, value in outcome_counts.items()},
        'protocol_schema_version': protocol['schema_version'],
    }
    context.logger.info('注入完成：航班=%d，异常航班=%d，异常记录=%d，事件=%d，结果=%s',
                        summary['flights'], summary['abnormal_flights'], summary['anomalous_records'],
                        summary['events'], summary['outcomes'])
    return ModuleResult(summary=summary, dependencies=(input_manifest, split_manifest))


def _inject_observations(observations: pd.DataFrame, splits: pd.DataFrame, protocol: dict[str, object],
                         root_seed: int, workers: int, *, logger: logging.Logger | None = None
                         ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按固定计划构造单副本观测；主进程选供体，工作进程仅处理单条航班。"""
    plan_by_flight = splits.set_index('flight_id')[['split', 'injection_plan', 'injection_eligible', 'injection_skip_reason']]
    source_positions = observations.loc[:, ['record_id']].copy()
    source_positions['_inject_input_position'] = np.arange(len(observations), dtype=np.int64)
    donor_index, flight_positions = _build_flight_indexes(observations, plan_by_flight)
    if logger is not None:
        logger.debug('已建立供体轻量索引：split=%s；航班=%d；不复制完整观测表。',
                     sorted(donor_index), sum(len(rows) for rows in donor_index.values()))
    chunks: list[pd.DataFrame] = []
    labels: list[pd.DataFrame] = []
    all_event_rows: list[dict[str, object]] = []
    all_impacts: list[dict[str, object]] = []
    outcomes: list[dict[str, object]] = []
    passthrough, passthrough_labels, passthrough_outcomes, flights = _prepare_passthrough(
        observations, plan_by_flight, flight_positions,
    )
    chunks.append(passthrough)
    labels.append(passthrough_labels)
    outcomes.extend(passthrough_outcomes)
    if logger is not None:
        logger.debug('批量透传正常／跳过航班：flights=%d；observations=%d；待注入航班=%d。',
                     len(passthrough_outcomes), len(passthrough), len(flights))
    flights_iter = iter(flights)
    pending: dict[object, tuple[pd.DataFrame, str, str, str, int]] = {}
    worker_count = workers
    queue_limit = max(1, worker_count * 2)
    first_submission_logged = False
    completed_injections = 0

    def collect(original: pd.DataFrame, final: pd.DataFrame, events: list[dict[str, object]], *, flight_id: str,
                split: str, planned: str, outcome: str, reason: str, seed: int) -> None:
        _collect_flight_result(chunks, labels, all_event_rows, all_impacts, outcomes, original, final, events,
                               flight_id=flight_id, split=split, planned=planned, outcome=outcome,
                               reason=reason, seed=seed)

    executor_context = (ProcessPoolExecutor(max_workers=worker_count, mp_context=_process_context(),
                                            initializer=configure_worker_threads)
                        if worker_count else nullcontext(None))
    with executor_context as pool:
        if logger is not None:
            logger.debug('注入工作池已就绪：workers=%d；context=%s。', worker_count,
                         _process_context().get_start_method())
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < queue_limit:
                try:
                    flight_id, positions = next(flights_iter)
                except StopIteration:
                    exhausted = True
                    break
                plan = plan_by_flight.loc[flight_id]
                split = str(plan['split'])
                original_group = observations.iloc[positions]
                working = original_group.sort_values('time_ns', kind='stable').reset_index(drop=True)
                original = working.loc[:, CANONICAL_COLUMNS].copy()
                seed = flight_seed(root_seed, flight_id)
                input_frame = original.copy()
                input_frame['split'] = split
                planning_rng = np.random.default_rng(seed)
                corruption_plan = prepare_corruption_plan(
                    input_frame, protocol, planning_rng,
                    has_donor=lambda length: _has_indexed_donor(donor_index, split, flight_id, length),
                    select_donor=lambda length: _select_indexed_donor(observations, donor_index, split, flight_id, length, planning_rng),
                )
                if pool is None:
                    raise RuntimeError('存在合格异常航班，但未分配注入工作进程')
                apply_seed = flight_seed(root_seed, f'{flight_id}|apply')
                if logger is not None and not first_submission_logged:
                    logger.debug('首次提交异常航班：flight_id=%s；points=%d；event_type=%s。',
                                 flight_id, len(input_frame), corruption_plan.event_type)
                future = pool.submit(_apply_corruption_task, input_frame, corruption_plan, protocol, apply_seed)
                if logger is not None and not first_submission_logged:
                    logger.debug('首次异常航班已提交至工作池。')
                    first_submission_logged = True
                pending[future] = original, flight_id, split, 'abnormal', seed
            if not pending:
                continue
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                original, flight_id, split, planned, seed = pending.pop(future)
                try:
                    final, events = future.result()
                except Exception as exc:
                    for queued in pending:
                        queued.cancel()
                    raise RuntimeError(f'航班 {flight_id} 注入失败：{exc}') from exc
                collect(original, final, events, flight_id=flight_id, split=split, planned=planned,
                        outcome='injected', reason='', seed=seed)
                completed_injections += 1
                if logger is not None and (completed_injections % 50 == 0 or completed_injections == len(flights)):
                    logger.debug('异常注入进度：completed=%d/%d；in_flight=%d。',
                                 completed_injections, len(flights), len(pending))
    if logger is not None:
        logger.debug('异常注入任务完成，开始合并观测、标签和事件账本。')
    final_observations = (pd.concat(chunks, ignore_index=True)
                          .merge(source_positions, on='record_id', how='left', validate='one_to_one')
                          .sort_values('_inject_input_position', kind='stable')
                          .drop(columns='_inject_input_position').reset_index(drop=True))
    if logger is not None:
        logger.debug('最终观测合并完成：rows=%d；开始对齐逐点标签。', len(final_observations))
    label_table = pd.concat(labels, ignore_index=True).set_index('record_id').loc[final_observations['record_id']].reset_index()
    event_table = pd.DataFrame(all_event_rows, columns=EVENT_COLUMNS)
    impact_table = pd.DataFrame(all_impacts, columns=IMPACT_COLUMNS)
    outcome_table = pd.DataFrame(outcomes, columns=OUTCOME_COLUMNS)
    validate_provenance(observations, final_observations, label_table, event_table, impact_table)
    return final_observations, label_table, event_table, impact_table, outcome_table


def _process_context() -> mp.context.BaseContext:
    """Windows 使用 spawn；Linux CLI 无活动线程时以 fork 避免逐 worker 重导入大数据栈。"""
    return mp.get_context('spawn' if os.name == 'nt' else 'fork')


def _prepare_passthrough(observations: pd.DataFrame, plan_by_flight: pd.DataFrame,
                         flight_positions: tuple[tuple[str, np.ndarray], ...]
                         ) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]], tuple[tuple[str, np.ndarray], ...]]:
    """一次透传所有不注入航班，仅将合格异常航班留给工作池，避免按正常航班重复生成差异账本。"""
    eligible_abnormal = (plan_by_flight['injection_plan'].eq('abnormal')
                         & plan_by_flight['injection_eligible'].astype(bool))
    injection_ids = frozenset(str(item) for item in plan_by_flight.index[eligible_abnormal])
    passthrough_plan = plan_by_flight.loc[~eligible_abnormal]
    passthrough = observations.loc[~observations['flight_id'].isin(injection_ids), CANONICAL_COLUMNS].copy()
    passthrough['split'] = passthrough['flight_id'].map(passthrough_plan['split'])
    passthrough['injection_plan'] = passthrough['flight_id'].map(passthrough_plan['injection_plan'])
    passthrough['injection_outcome'] = np.where(
        passthrough['injection_plan'].eq('abnormal'), 'skipped_points_below_minimum', 'normal_planned',
    )
    passthrough['injection_skip_reason'] = (
        passthrough['flight_id'].map(passthrough_plan['injection_skip_reason']).astype('string').fillna('')
    )
    if passthrough[['split', 'injection_plan']].isna().any().any() or passthrough['injected_label'].ne(0).any():
        raise ValueError('批量透传航班的 split、计划或正常标签不完整')
    outcomes = [
        {
            'flight_id': str(flight_id), 'split': str(row['split']), 'injection_plan': str(row['injection_plan']),
            'injection_outcome': 'skipped_points_below_minimum' if row['injection_plan'] == 'abnormal' else 'normal_planned',
            'injection_skip_reason': '' if pd.isna(row['injection_skip_reason']) else str(row['injection_skip_reason']),
        }
        for flight_id, row in passthrough_plan.iterrows()
    ]
    to_inject = tuple((flight_id, positions) for flight_id, positions in flight_positions if flight_id in injection_ids)
    return passthrough, normal_record_labels(passthrough), outcomes, to_inject


def _build_flight_indexes(source: pd.DataFrame, plan_by_flight: pd.DataFrame
                          ) -> tuple[dict[str, tuple[tuple[str, np.ndarray], ...]], tuple[tuple[str, np.ndarray], ...]]:
    """一次扫描轻量时间视图，同时建立供体及目标航班位置索引，避免第二次全表 groupby。"""
    candidates: dict[str, list[tuple[str, np.ndarray]]] = {}
    flight_positions: list[tuple[str, np.ndarray]] = []
    times = source.loc[:, ['flight_id', 'time_ns']].copy()
    times['_position'] = np.arange(len(source), dtype=np.int64)
    for flight_id, group in times.groupby('flight_id', sort=False, observed=True):
        identity = str(flight_id)
        positions = group['_position'].to_numpy(dtype=np.int64, copy=True)
        order = np.argsort(group['time_ns'].to_numpy(dtype=np.int64), kind='stable')
        ordered_positions = positions[order]
        split = str(plan_by_flight.loc[identity, 'split'])
        candidates.setdefault(split, []).append((identity, ordered_positions))
        flight_positions.append((identity, positions))
    return {split: tuple(rows) for split, rows in candidates.items()}, tuple(flight_positions)


def _has_indexed_donor(donor_index: dict[str, tuple[tuple[str, np.ndarray], ...]], split: str,
                      target_flight_id: str, length: int) -> bool:
    """判断同分区中是否存在长度足够且异源的连续供体。"""
    return any(flight_id != target_flight_id and len(positions) >= length
               for flight_id, positions in donor_index.get(split, ()))


def _select_indexed_donor(source: pd.DataFrame, donor_index: dict[str, tuple[tuple[str, np.ndarray], ...]],
                          split: str, target_flight_id: str, length: int,
                          rng: np.random.Generator) -> dict[str, object]:
    """主进程按轻量索引选择供体，仅把一个短窗口传给相应工作进程。"""
    candidates = [(flight_id, positions) for flight_id, positions in donor_index.get(split, ())
                  if flight_id != target_flight_id and len(positions) >= length]
    if not candidates:
        raise ValueError('不存在同分区且足够长的异源真实运动供体')
    flight_id, positions = candidates[int(rng.integers(len(candidates)))]
    start = int(rng.integers(0, len(positions) - length + 1))
    chosen = positions[start:start + length]
    rows = source.iloc[chosen].loc[:, CANONICAL_COLUMNS].copy().reset_index(drop=True)
    return {
        'flight_id': flight_id, 'start_index': start, 'end_index_exclusive': start + length,
        'start_record_id': str(rows['record_id'].iloc[0]),
        'end_record_id_exclusive': str(source.iloc[positions[start + length]]['record_id']) if start + length < len(positions) else None,
        'rows': rows,
    }


def _apply_corruption_task(frame: pd.DataFrame, plan: CorruptionPlan, protocol: dict[str, object],
                           seed: int) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """工作进程仅修改一条目标航班，不读取共享文件或全局供体库。"""
    final, events = apply_corruption_plan(frame, plan, protocol, np.random.default_rng(seed))
    return final.loc[:, CANONICAL_COLUMNS].copy(), events


def _collect_flight_result(chunks: list[pd.DataFrame], labels: list[pd.DataFrame],
                           all_event_rows: list[dict[str, object]], all_impacts: list[dict[str, object]],
                           outcomes: list[dict[str, object]], original: pd.DataFrame, final: pd.DataFrame,
                           events: list[dict[str, object]], *, flight_id: str, split: str, planned: str,
                           outcome: str, reason: str, seed: int) -> None:
    """在主进程统一生成标签和账本，保持单写者及稳定产物顺序。"""
    final = final.loc[:, CANONICAL_COLUMNS].copy()
    labels_by_position = final_labels(original, final) if events else np.zeros(len(final), dtype=np.int8)
    final['injected_label'] = labels_by_position
    final['split'] = split
    final['injection_plan'] = planned
    final['injection_outcome'] = outcome
    final['injection_skip_reason'] = reason
    if events:
        all_event_rows.extend(event_rows(events, final, flight_id=flight_id, split=split, seed=seed))
        all_impacts.extend(impact_rows(events, flight_id=flight_id, flight_points=len(final)))
    chunks.append(final)
    labels.append(record_labels(original, final, events, labels=labels_by_position) if events else normal_record_labels(final))
    outcomes.append({'flight_id': flight_id, 'split': split, 'injection_plan': planned,
                     'injection_outcome': outcome, 'injection_skip_reason': reason})


def _flight_gap_seconds(input_manifest: Path) -> float:
    """从 preprocess 摘要复制航班断开阈值，供后续 segment 无歧义继承。"""
    summary = json.loads((input_manifest.parent / 'summary.json').read_text(encoding='utf-8'))
    if not isinstance(summary, dict):
        raise ValueError('preprocess 摘要格式无效')
    value = float(summary.get('flight_gap_seconds'))
    if not np.isfinite(value) or value <= 0:
        raise ValueError('preprocess 摘要中的 flight_gap_seconds 无效')
    return value


def validate_outputs(context: RunContext, args: argparse.Namespace, result: ModuleResult) -> ValidationReport:
    """重新读取产物，验证单副本、事件区间、标签、影响账本与跳过政策完全一致。"""
    _, _, _, original, splits, _ = _read_inputs(context, args)
    data_dir = context.staging_dir / 'data'
    observations = pd.read_csv(data_dir / 'observations.csv', dtype={'record_id': 'string', 'flight_id': 'string', 'icao': 'string', 'call': 'string'})
    labels = pd.read_csv(data_dir / 'record_labels.csv', dtype={'record_id': 'string', 'flight_id': 'string'})
    events = pd.read_csv(data_dir / 'injection_events.csv', dtype={'flight_id': 'string', 'start_record_id': 'string',
                                                                    'end_record_id_exclusive': 'string'})
    impacts = pd.read_csv(data_dir / 'injection_impacts.csv', dtype={'flight_id': 'string', 'event_id': 'string'})
    outcomes = pd.read_csv(data_dir / 'injection_outcomes.csv', dtype={'flight_id': 'string'})
    if tuple(observations.columns) != OBSERVATION_COLUMNS:
        raise ValueError('inject 最终观测字段或顺序不符合约定')
    if tuple(labels.columns) != LABEL_COLUMNS or tuple(events.columns) != EVENT_COLUMNS or tuple(impacts.columns) != IMPACT_COLUMNS:
        raise ValueError('记录标签、事件账本或影响账本字段不符合约定')
    validate_provenance(original, observations, labels, events, impacts)
    if outcomes['flight_id'].duplicated().any() or set(outcomes['flight_id']) != set(observations['flight_id']):
        raise ValueError('航班注入结果表未覆盖每条来源航班一次')
    plans = splits.set_index('flight_id')['injection_plan']
    actual = observations.groupby('flight_id', sort=False)['injected_label'].max()
    event_counter = Counter(events['flight_id'])
    for row in outcomes.itertuples(index=False):
        label = int(actual.loc[row.flight_id])
        if row.injection_plan == 'normal' and (label != 0 or row.injection_outcome != 'normal_planned'):
            raise ValueError('计划正常航班不得被修改')
        if row.injection_plan == 'abnormal' and row.injection_outcome == 'injected' and (label != 1 or event_counter[row.flight_id] != 1):
            raise ValueError('成功注入航班必须具有一个事件且为异常')
        if row.injection_plan == 'abnormal' and row.injection_outcome.startswith('skipped_') and (label != 0 or event_counter[row.flight_id] != 0):
            raise ValueError('跳过注入的航班必须原样作为正常轨迹')
        if row.injection_plan != plans.loc[row.flight_id]:
            raise ValueError('最终航班注入计划与 split 不一致')
    return ValidationReport(checks={
        'observations': int(len(observations)), 'flights': int(len(actual)),
        'anomalous_records': int(observations['injected_label'].sum()), 'events': int(len(events)),
        'single_final_variant_per_flight': True, 'labels_verified_against_physical_difference': True,
        'whole_flight_bounded_events': True, 'split_plan_verified': True,
    })
