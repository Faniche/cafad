"""api.py | 2026-11-03 | split 模块参数、运行与落盘验收 | Codex."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.app.arguments import ProjectArgumentParser
from src.app.artifacts import read_manifest
from src.app.contracts import ModuleResult, RunContext, ValidationReport
from src.utils.env import Env


SPLIT_NAMES = ('train', 'val', 'test')
TRAJECTORY_KINDS = ('normal', 'abnormal')


def build_parser() -> argparse.ArgumentParser:
    """定义航班级分区的模块参数。"""
    parser = ProjectArgumentParser(prog='main.py -m split', description='按来源航班划分训练、验证和测试分区。')
    parser.add_argument('--input', required=True, type=Path, help='preprocess 模块已发布的 manifest.json')
    parser.add_argument('--ratios', type=float, nargs=3, default=(0.70, 0.15, 0.15), metavar=('TRAIN', 'VAL', 'TEST'))
    parser.add_argument('--normal-ratios', type=float, nargs=3, default=(0.50, 0.50, 0.50),
                        metavar=('TRAIN', 'VAL', 'TEST'),
                        help='各分区中计划保持正常的来源航班比例；与 --anomaly-ratios 逐列相加为 1')
    parser.add_argument('--anomaly-ratios', type=float, nargs=3, default=(0.50, 0.50, 0.50),
                        metavar=('TRAIN', 'VAL', 'TEST'),
                        help='各分区中计划由 inject 加噪的来源航班比例；尚不代表已经注入')
    return parser


def _project_path(context: RunContext, value: Path) -> Path:
    return value if value.is_absolute() else context.project_root / value


def _counts(total: int, ratios: tuple[float, ...], labels: tuple[str, ...], *, require_each: bool) -> np.ndarray:
    """使用受最小类别数约束的最大余数法分配整数航班数。"""
    values = np.asarray(ratios, dtype=np.float64)
    if len(values) != len(labels) or not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1.0):
        raise ValueError(f'{"、".join(labels)} 的比例必须非负，且和为 1')
    active = values > 0
    if require_each and total < int(active.sum()):
        raise ValueError(f'航班数 {total} 不足以给 {"、".join(np.asarray(labels)[active])} 各分配至少一个来源航班')
    raw = values * total
    counts = np.floor(raw).astype(np.int64)
    for index in np.argsort(-(raw - counts), kind='stable')[:total - int(counts.sum())]:
        counts[index] += 1
    if require_each:
        for missing_index in np.flatnonzero(active & (counts == 0)):
            donor_candidates = np.flatnonzero(counts > 1)
            if not len(donor_candidates):
                raise ValueError('比例约束无法为每个要求的类别保留至少一个来源航班')
            donor_index = donor_candidates[np.argmax(counts[donor_candidates])]
            counts[donor_index] -= 1
            counts[missing_index] += 1
    return counts


def _plan_counts(split_counts: np.ndarray, normal_ratios: tuple[float, float, float],
                 anomaly_ratios: tuple[float, float, float]) -> np.ndarray:
    """为每个分区分配未来正常和异常航班计划，异常只在 inject 后成为真实标签。"""
    normal = np.asarray(normal_ratios, dtype=np.float64)
    anomaly = np.asarray(anomaly_ratios, dtype=np.float64)
    if (len(normal) != len(SPLIT_NAMES) or len(anomaly) != len(SPLIT_NAMES)
            or not np.isfinite(normal).all() or not np.isfinite(anomaly).all()
            or (normal < 0).any() or (anomaly < 0).any() or not np.allclose(normal + anomaly, 1.0)):
        raise ValueError('每个训练／验证／测试分区中的正常与异常比例必须非负，且相加为 1')
    return np.asarray([
        _counts(int(total), (float(normal[index]), float(anomaly[index])), TRAJECTORY_KINDS, require_each=True)
        for index, total in enumerate(split_counts)
    ], dtype=np.int64)


def _minimum_injection_points() -> int:
    """读取全局注入资格下限；该值与协议最长事件及上下文约束须一致。"""
    minimum = Env.get_int('injection_minimum_trajectory_points')
    if minimum < 1:
        raise ValueError('injection_minimum_trajectory_points 必须为正整数')
    return minimum


def run(context: RunContext, args: argparse.Namespace) -> ModuleResult:
    """从 preprocess 的航班清单产生稳定、互斥的 split 表。"""
    input_manifest = _project_path(context, args.input).resolve()
    manifest = read_manifest(input_manifest, expected_module='preprocess')
    if manifest.get('dataset_id') != context.dataset_id:
        raise ValueError('上游数据集身份与本次从 --input 自动识别的身份不一致')
    source_dir = input_manifest.parent / 'data'
    flights = pd.read_csv(source_dir / 'flights.csv', dtype={'flight_id': 'string', 'icao': 'string', 'call': 'string'})
    if flights.empty or flights['flight_id'].duplicated().any():
        raise ValueError('上游航班表为空或 flight_id 不唯一')
    if (np.asarray(args.ratios, dtype=np.float64) <= 0).any():
        raise ValueError('训练、验证和测试三个分区比例必须均大于 0')
    counts = _counts(len(flights), tuple(args.ratios), SPLIT_NAMES, require_each=True)
    requested_plans = _plan_counts(counts, tuple(args.normal_ratios), tuple(args.anomaly_ratios))
    minimum_points = _minimum_injection_points()
    context.logger.debug('split 输入：manifest=%s；航班=%d；比例=%s；计划正常=%s；计划异常=%s；最小点数=%d',
                         input_manifest, len(flights), args.ratios, args.normal_ratios, args.anomaly_ratios,
                         minimum_points)
    ordered = flights.sort_values('flight_id', kind='stable').reset_index(drop=True)
    random = np.random.default_rng(context.seed)
    shuffled = ordered.iloc[random.permutation(len(ordered))].reset_index(drop=True)
    partitions: list[pd.DataFrame] = []
    start = 0
    for index, split_name in enumerate(SPLIT_NAMES):
        stop = start + int(counts[index])
        partition = shuffled.iloc[start:stop].copy().reset_index(drop=True)
        partition['split'] = split_name
        partition['injection_eligible'] = partition['records'].ge(minimum_points)
        partition['injection_skip_reason'] = np.where(
            partition['injection_eligible'], '', f'points_below_minimum_{minimum_points}',
        )
        requested_abnormal = int(requested_plans[index, 1])
        eligible_positions = np.flatnonzero(partition['injection_eligible'].to_numpy(dtype=bool))
        selected_count = min(requested_abnormal, len(eligible_positions))
        selected_positions = (random.choice(eligible_positions, size=selected_count, replace=False)
                              if selected_count else np.empty(0, dtype=np.int64))
        partition['injection_plan'] = 'normal'
        partition.loc[selected_positions, 'injection_plan'] = 'abnormal'
        partitions.append(partition)
        start = stop
    result = pd.concat(partitions, ignore_index=True)[
        ['flight_id', 'split', 'injection_plan', 'injection_eligible', 'injection_skip_reason']
    ]
    result = result.merge(flights[['flight_id', 'icao', 'call']], on='flight_id', validate='one_to_one')
    output = context.staging_dir / 'data'
    output.mkdir()
    result.to_csv(output / 'splits.csv', index=False, encoding='utf-8')
    requested_plan_summary = {
        split_name: {kind: int(requested_plans[split_index, kind_index]) for kind_index, kind in enumerate(TRAJECTORY_KINDS)}
        for split_index, split_name in enumerate(SPLIT_NAMES)
    }
    actual_plan_summary = {
        split_name: {kind: int(((result['split'] == split_name) & (result['injection_plan'] == kind)).sum())
                     for kind in TRAJECTORY_KINDS}
        for split_name in SPLIT_NAMES
    }
    summary: dict[str, object] = {
        'input_manifest': str(input_manifest), 'flights': int(len(result)),
        'train_flights': int(counts[0]), 'val_flights': int(counts[1]), 'test_flights': int(counts[2]),
        'ratios': list(args.ratios), 'normal_ratios': list(args.normal_ratios),
        'anomaly_ratios': list(args.anomaly_ratios), 'injection_minimum_trajectory_points': minimum_points,
        'requested_injection_plan_counts': requested_plan_summary, 'injection_plan_counts': actual_plan_summary,
        'injection_ineligible_flights': int((~result['injection_eligible']).sum()), 'seed': context.seed,
    }
    context.logger.info('航班划分完成：train=%d，val=%d，test=%d；最小注入点数=%d；请求计划=%s；实际计划=%s；短轨迹=%d',
                        counts[0], counts[1], counts[2], minimum_points, requested_plan_summary,
                        actual_plan_summary, int((~result['injection_eligible']).sum()))
    return ModuleResult(summary=summary, dependencies=(input_manifest,))


def validate_outputs(context: RunContext, args: argparse.Namespace, result: ModuleResult) -> ValidationReport:
    """检查分区表对所有来源航班覆盖一次且标签合法。"""
    table = pd.read_csv(context.staging_dir / 'data' / 'splits.csv', dtype={'flight_id': 'string'})
    if table.empty or table['flight_id'].duplicated().any():
        raise ValueError('split 表为空或 flight_id 重复')
    allowed = set(SPLIT_NAMES)
    if set(table['split']) - allowed:
        raise ValueError('split 表存在未知分区标签')
    required = {'injection_plan', 'injection_eligible', 'injection_skip_reason'}
    if not required.issubset(table.columns) or set(table['injection_plan']) - set(TRAJECTORY_KINDS):
        raise ValueError('split 表缺少或包含无效的注入资格／计划字段')
    if (np.asarray(args.ratios, dtype=np.float64) <= 0).any():
        raise ValueError('训练、验证和测试三个分区比例必须均大于 0')
    minimum_points = _minimum_injection_points()
    if table.loc[~table['injection_eligible'], 'injection_plan'].ne('normal').any():
        raise ValueError('点数不足的航班不得计划为异常注入')
    if table.loc[~table['injection_eligible'], 'injection_skip_reason'].fillna('').ne(f'points_below_minimum_{minimum_points}').any():
        raise ValueError('点数不足航班的跳过原因不一致')
    if table.loc[table['injection_eligible'], 'injection_skip_reason'].astype('string').fillna('').ne('').any():
        raise ValueError('合格航班不应携带跳过原因')
    requested = _plan_counts(_counts(len(table), tuple(args.ratios), SPLIT_NAMES, require_each=True),
                             tuple(args.normal_ratios), tuple(args.anomaly_ratios))
    actual = table.groupby(['split', 'injection_plan']).size()
    for split_index, split_name in enumerate(SPLIT_NAMES):
        requested_abnormal = int(requested[split_index, 1])
        actual_abnormal = int(actual.get((split_name, 'abnormal'), 0))
        eligible = int(((table['split'] == split_name) & table['injection_eligible']).sum())
        if actual_abnormal != min(requested_abnormal, eligible):
            raise ValueError('split 表的异常计划未按合格轨迹数量截断')
    counts = table['split'].value_counts().to_dict()
    return ValidationReport(checks={'flights': int(len(table)), 'split_counts': {str(key): int(value) for key, value in counts.items()},
                                    'injection_plan_counts': result.summary['injection_plan_counts'],
                                     'injection_minimum_trajectory_points': minimum_points,
                                    'flight_level_disjoint': True})
