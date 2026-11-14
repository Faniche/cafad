"""dlr_lstm.py | 2026-11-13 | DLR 预训练 LSTM 阶段识别适配器 | Codex.

复用 DLR-MO/flight-phase-lstm（MIT）的预训练权重、三变量归一化、连续 K-means
分窗和八阶段语义；不导入其锁定在旧版 NumPy/PyTorch 的运行代码。
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, wait
from dataclasses import dataclass
from hashlib import sha256
import csv
import math
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as functional

from src.data_processing.phase_pipeline import (
    FLIGHT_COLUMNS,
    POINT_COLUMNS,
    SEGMENT_COLUMNS,
    UNRECOGNIZED_FLIGHT_COLUMNS,
    UNRECOGNIZED_POINT_COLUMNS,
    configure_worker_threads,
    write_segment_progress,
)
from src.utils.process_pool import InterruptibleProcessPoolExecutor as ProcessPoolExecutor


DLR_PHASES = ('taxi', 'take_off', 'initial_climb', 'climb', 'cruise', 'descent', 'approach', 'landing')
DLR_CLUSTER_COUNT = 160
DLR_FEATURE_COUNT = 7
DLR_KMEANS_WEIGHTS = np.asarray((0.7, 0.8, 0.1), dtype=np.float64)
DLR_KMEANS_ITERATIONS = 100
DLR_DEFAULT_MODEL = Path('models/dlr_flight_phase_lstm.pt')


class Logger(Protocol):
    """DLR 适配器所需的最小项目日志接口。"""

    def debug(self, message: str, *args: object) -> None: ...
    def info(self, message: str, *args: object) -> None: ...
    def warning(self, message: str, *args: object) -> None: ...


class DlrLstmTagger(nn.Module):
    """与 DLR 已发布 checkpoint 参数布局一致的双层 LSTM。"""

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(DLR_FEATURE_COUNT, 16, num_layers=2, batch_first=True)
        self.hidden2tag = nn.Linear(16, len(DLR_PHASES))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """返回每个连续 K-means 窗的八类对数概率。"""
        output, _ = self.lstm(inputs)
        return functional.log_softmax(self.hidden2tag(output), dim=-1)


@dataclass
class PreparedFlight:
    """一个航班已完成 CPU 特征准备，等待主进程批量 LSTM 推理。"""

    frame: pd.DataFrame
    raw_positions: np.ndarray
    features: np.ndarray
    clusters: np.ndarray
    worker_pid: int
    seconds: float


@dataclass
class RejectedFlight:
    """一条不满足预训练输入前提的航班及其完整可追溯排除记录。"""

    report: dict[str, object]
    points: pd.DataFrame


def default_model_path(project_root: Path) -> Path:
    """返回项目随代码部署的 DLR 权重位置。"""
    return project_root / DLR_DEFAULT_MODEL


def _model_digest(path: Path) -> str:
    digest = sha256()
    with path.open('rb') as reader:
        while block := reader.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_model(path: Path) -> DlrLstmTagger:
    """严格加载公开 checkpoint，拒绝不兼容的权重而非静默随机初始化。"""
    if not path.is_file():
        raise FileNotFoundError(f'DLR 预训练模型不存在：{path}；请部署 models/dlr_flight_phase_lstm.pt 或传入 --dlr-model')
    try:
        state = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:  # 兼容仍使用旧 torch.load 签名的环境。
        state = torch.load(path, map_location='cpu')
    model = DlrLstmTagger()
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _normalise(values: np.ndarray, lower: float, upper: float, *, clip: bool) -> np.ndarray:
    result = (values - lower) / (upper - lower)
    return np.clip(result, 0.0, 1.0) if clip else result


def _regularise_flight(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """按 DLR 的一秒时间轴补齐并线性插值，返回规则轨迹和原始点对应秒索引。"""
    ordered = frame.sort_values('time_ns', kind='stable').reset_index(drop=True)
    elapsed_seconds = ((ordered['time_ns'].to_numpy(dtype=np.int64) - int(ordered['time_ns'].iloc[0])) // 1_000_000_000)
    source = pd.DataFrame({
        'second': elapsed_seconds,
        'alt_ft': ordered['geometric_altitude_m'].to_numpy(dtype=np.float64) / 0.3048,
        'speed_kt': ordered['speed_mps'].to_numpy(dtype=np.float64) * 3600.0 / 1852.0,
    })
    grouped = source.groupby('second', sort=True, as_index=True).mean(numeric_only=True)
    regular = grouped.reindex(range(int(elapsed_seconds.max()) + 1)).interpolate(method='linear', limit_area='inside')
    valid = regular[['alt_ft', 'speed_kt']].notna().all(axis=1).to_numpy()
    if not valid.any():
        raise ValueError('一秒重采样后缺少同时有效的几何高度和速度')
    first, last = int(np.flatnonzero(valid)[0]), int(np.flatnonzero(valid)[-1])
    regular = regular.iloc[first:last + 1].reset_index(drop=True)
    seconds = np.arange(first, last + 1, dtype=np.int64)
    point_positions = elapsed_seconds - first
    if len(regular) < DLR_CLUSTER_COUNT or np.any(point_positions < 0) or np.any(point_positions >= len(regular)):
        raise ValueError(f'DLR 预训练模型要求至少 {DLR_CLUSTER_COUNT} 个连续有效秒；当前为 {len(regular)} 秒')
    altitude = regular['alt_ft'].to_numpy(dtype=np.float64)
    regular['roc_fpm'] = np.gradient(altitude) * 60.0
    regular['second'] = seconds
    return regular, point_positions.astype(np.int64)


def _continuous_kmeans(values: np.ndarray) -> np.ndarray:
    """复现 DLR 连续 K-means 的相邻边界移动，不依赖其过期 np.int 实现。"""
    count = len(values)
    if count < DLR_CLUSTER_COUNT:
        raise ValueError(f'连续 K-means 需要至少 {DLR_CLUSTER_COUNT} 个点')
    labels = np.empty(count, dtype=np.int64)
    sizes = np.full(DLR_CLUSTER_COUNT, count // DLR_CLUSTER_COUNT, dtype=np.int64)
    sizes[:count % DLR_CLUSTER_COUNT] += 1
    offset = 0
    for label, size in enumerate(sizes):
        labels[offset:offset + size] = label
        offset += int(size)

    def means(current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        counts = np.bincount(current, minlength=DLR_CLUSTER_COUNT).astype(np.int64)
        result = np.zeros((DLR_CLUSTER_COUNT, values.shape[1]), dtype=np.float64)
        np.add.at(result, current, values)
        result /= np.maximum(counts, 1)[:, None]
        return result, counts

    centers, counts = means(labels)
    for _ in range(DLR_KMEANS_ITERATIONS):
        updated = labels.copy()
        for index in range(1, count - 1):
            current = labels[index]
            previous, following = labels[index - 1], labels[index + 1]
            if counts[following] < 1800 and counts[previous] < 1800:
                current_distance = math.sqrt(float(np.sum(DLR_KMEANS_WEIGHTS * (values[index] - centers[current]) ** 2)))
                if following != current and counts[current] > 4:
                    next_distance = math.sqrt(float(np.sum(DLR_KMEANS_WEIGHTS * (values[index] - centers[following]) ** 2)))
                    if next_distance < current_distance:
                        updated[index] = following
                elif previous != current and counts[current] > 4:
                    previous_distance = math.sqrt(float(np.sum(DLR_KMEANS_WEIGHTS * (values[index] - centers[previous]) ** 2)))
                    if previous_distance < current_distance:
                        if updated[index - 1] == previous:
                            updated[index] = previous
                        else:
                            earlier_current = math.sqrt(float(np.sum(DLR_KMEANS_WEIGHTS * (values[index - 1] - centers[previous]) ** 2)))
                            earlier_previous = math.sqrt(float(np.sum(DLR_KMEANS_WEIGHTS * (values[index - 1] - centers[current]) ** 2)))
                            if earlier_current - earlier_previous < current_distance - previous_distance:
                                updated[index] = previous
                                updated[index - 1] = previous
        if np.array_equal(updated, labels):
            break
        labels = updated
        centers, counts = means(labels)
    if np.any(np.diff(labels) < 0) or np.any(np.diff(labels) > 1):
        raise RuntimeError('连续 K-means 未保持相邻连续窗口顺序')
    return labels


def _cluster_features(regular: pd.DataFrame, clusters: np.ndarray) -> np.ndarray:
    """按 DLR 的七维首末运动特征，将 160 个连续窗口编码为模型输入。"""
    features = np.zeros((DLR_CLUSTER_COUNT, DLR_FEATURE_COUNT), dtype=np.float32)
    cuts = np.r_[0, np.flatnonzero(clusters[1:] != clusters[:-1]) + 1, len(clusters)]
    for start, end in zip(cuts[:-1], cuts[1:], strict=True):
        cluster = int(clusters[start])
        altitude = regular['alt_ft'].iloc[start:end].to_numpy(dtype=np.float64)
        speed = regular['speed_kt'].iloc[start:end].to_numpy(dtype=np.float64)
        vertical_rate = regular['roc_fpm'].iloc[start:end].to_numpy(dtype=np.float64)
        features[cluster] = (
            _normalise(np.asarray([end - start]), 4.0, 1800.0, clip=True)[0],
            _normalise(np.asarray([altitude[0]]), 0.0, 41000.0, clip=True)[0],
            _normalise(np.asarray([altitude[-1]]), 0.0, 41000.0, clip=True)[0],
            _normalise(np.asarray([speed[0]]), 0.0, 470.0, clip=True)[0],
            _normalise(np.asarray([speed[-1]]), 0.0, 470.0, clip=True)[0],
            _normalise(np.asarray([vertical_rate[0]]), -10000.0, 10000.0, clip=True)[0],
            _normalise(np.asarray([vertical_rate[-1]]), -10000.0, 10000.0, clip=True)[0],
        )
    return features


def _decode_transitions(log_probabilities: np.ndarray) -> np.ndarray:
    """在 DLR 合法顺序 taxi→…→landing→taxi 上求最大后验路径。

    前后 taxi 共享网络的类别 0；内部使用第九个状态表示落地后的 taxi，
    因而允许不完整的起飞、巡航和降落观测，同时禁止阶段回跳。
    """
    if log_probabilities.shape != (DLR_CLUSTER_COUNT, len(DLR_PHASES)):
        raise ValueError('DLR LSTM 输出维度与 160×8 预训练配置不一致')
    state_labels = np.asarray((0, 1, 2, 3, 4, 5, 6, 7, 0), dtype=np.int64)
    scores = np.full((DLR_CLUSTER_COUNT, len(state_labels)), -np.inf, dtype=np.float64)
    parents = np.zeros_like(scores, dtype=np.int64)
    scores[0, :8] = log_probabilities[0, :]
    for index in range(1, DLR_CLUSTER_COUNT):
        for state in range(8):
            previous = scores[index - 1, :state + 1]
            parent = int(np.argmax(previous))
            scores[index, state] = previous[parent] + log_probabilities[index, state_labels[state]]
            parents[index, state] = parent
        previous = scores[index - 1, 7:]
        parent = int(np.argmax(previous)) + 7
        scores[index, 8] = previous[parent - 7] + log_probabilities[index, 0]
        parents[index, 8] = parent
    states = np.empty(DLR_CLUSTER_COUNT, dtype=np.int64)
    states[-1] = int(np.argmax(scores[-1]))
    for index in range(DLR_CLUSTER_COUNT - 1, 0, -1):
        states[index - 1] = parents[index, states[index]]
    return state_labels[states]


def _iso(value: int) -> str:
    return pd.Timestamp(int(value), tz='UTC').isoformat()


def _span_km(frame: pd.DataFrame, start: int, end: int) -> float:
    latitudes = np.deg2rad(frame['latitude_deg'].iloc[[start, end - 1]].to_numpy(dtype=np.float64))
    longitudes = np.deg2rad(frame['longitude_deg'].iloc[[start, end - 1]].to_numpy(dtype=np.float64))
    value = np.sin((latitudes[1] - latitudes[0]) / 2) ** 2
    value += np.cos(latitudes[0]) * np.cos(latitudes[1]) * np.sin((longitudes[1] - longitudes[0]) / 2) ** 2
    return float(6371.0088 * 2 * np.arcsin(np.sqrt(value)))


def _unrecognized(frame: pd.DataFrame, reason: str) -> tuple[dict[str, object], pd.DataFrame]:
    """返回与 Sun 分段器相同的不可识别航班与点映射记录。"""
    first, last = frame.iloc[0], frame.iloc[-1]
    report = {
        'flight_id': str(first.flight_id), 'icao': str(first.icao), 'call': str(first.call), 'points': int(len(frame)),
        'duration_seconds': (int(last.time_ns) - int(first.time_ns)) / 1e9,
        'start_time': _iso(int(first.time_ns)), 'end_time': _iso(int(last.time_ns)),
        'start_longitude_deg': float(first.longitude_deg), 'start_latitude_deg': float(first.latitude_deg),
        'end_longitude_deg': float(last.longitude_deg), 'end_latitude_deg': float(last.latitude_deg), 'reason': reason,
    }
    return report, frame[['source_row_index', 'flight_id']].copy()


def _flight_rows(frame: pd.DataFrame, point_labels: np.ndarray) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """将 DLR 对每个一秒窗口的标签回接到原始记录，再构造标准片段表。"""
    phases = np.asarray(DLR_PHASES, dtype=object)[point_labels]
    cuts = np.r_[0, np.flatnonzero(phases[1:] != phases[:-1]) + 1, len(frame)]
    point_segment_ids = np.empty(len(frame), dtype=object)
    segments: list[dict[str, object]] = []
    for ordinal, (start, end) in enumerate(zip(cuts[:-1], cuts[1:], strict=True)):
        segment_id = f'{frame["flight_id"].iloc[0]}:{ordinal}'
        point_segment_ids[start:end] = segment_id
        first, last = frame.iloc[start], frame.iloc[end - 1]
        segments.append({
            'segment_id': segment_id, 'flight_id': str(first.flight_id), 'icao': str(first.icao), 'call': str(first.call),
            'ordinal': ordinal, 'phase': str(phases[start]), 'points': int(end - start),
            'start_time': _iso(int(first.time_ns)), 'end_time': _iso(int(last.time_ns)),
            'duration_seconds': (int(last.time_ns) - int(first.time_ns)) / 1e9,
            'start_longitude_deg': float(first.longitude_deg), 'start_latitude_deg': float(first.latitude_deg),
            'end_longitude_deg': float(last.longitude_deg), 'end_latitude_deg': float(last.latitude_deg),
            'span_km': _span_km(frame, start, end), 'start_altitude_m': float(first.geometric_altitude_m),
            'end_altitude_m': float(last.geometric_altitude_m), 'first_source_row_index': int(first.source_row_index),
            'last_source_row_index': int(last.source_row_index),
        })
    points = frame[['source_row_index', 'flight_id', 'icao', 'call', 'time_ns']].copy()
    points['phase'] = phases
    points['segment_id'] = point_segment_ids
    return points, segments


def _prepare_flight(frame: pd.DataFrame) -> PreparedFlight | RejectedFlight:
    """在一个 CPU 工作进程中完成一条航班的一秒规则化和连续 K-means。"""
    started = time.monotonic()
    flight = frame.sort_values(['time_ns', 'source_row_index'], kind='stable').reset_index(drop=True)
    try:
        regular, raw_positions = _regularise_flight(flight)
        values = np.column_stack((
            _normalise(regular['alt_ft'].to_numpy(dtype=np.float64), 0.0, 41000.0, clip=False),
            _normalise(regular['speed_kt'].to_numpy(dtype=np.float64), 0.0, 470.0, clip=False),
            _normalise(regular['roc_fpm'].to_numpy(dtype=np.float64), -10000.0, 10000.0, clip=False),
        ))
        clusters = _continuous_kmeans(values)
    except ValueError as exc:
        report, points = _unrecognized(flight, f'{type(exc).__name__}: {exc}')
        return RejectedFlight(report=report, points=points)
    return PreparedFlight(frame=flight, raw_positions=raw_positions, features=_cluster_features(regular, clusters),
                          clusters=clusters, worker_pid=os.getpid(), seconds=time.monotonic() - started)


def run_dlr_lstm(observations: pd.DataFrame, output: Path, progress_database: Path, project_root: Path,
                 model_override: Path | None, inference_batch_size: int, workers: int, logger: Logger) -> dict[str, object]:
    """多进程准备航班特征、主进程批量推理并写出与 Sun 分段器同构的中间 CSV 表。"""
    if inference_batch_size < 1:
        raise ValueError('DLR inference_batch_size 必须为正整数')
    output.mkdir(parents=True, exist_ok=False)
    model_path = (model_override if model_override is not None else default_model_path(project_root)).resolve()
    model = _load_model(model_path)
    ordered = observations.copy()
    ordered['source_row_index'] = np.arange(len(ordered), dtype=np.int64)
    ordered = ordered.sort_values(['flight_id', 'time_ns', 'source_row_index'], kind='stable')
    flights_total = int(ordered['flight_id'].nunique())
    actual_workers = max(1, min(workers, flights_total))
    logger.info('DLR 航班特征准备：flights=%d；workers=%d；LSTM batch_max=%d',
                flights_total, actual_workers, inference_batch_size)
    processed = unrecognized = recognized_points = unrecognized_points = 0
    phase_counts: Counter[str] = Counter()
    worker_pids: set[int] = set()
    pending_batch: list[PreparedFlight] = []
    output_batch_size = min(inference_batch_size, 32)

    with (output / 'points.csv').open('w', encoding='utf-8', newline='') as points_file, \
         (output / 'segments.csv').open('w', encoding='utf-8', newline='') as segments_file, \
         (output / 'flights.csv').open('w', encoding='utf-8', newline='') as flights_file, \
         (output / 'unrecognized_flights.csv').open('w', encoding='utf-8', newline='') as unrecognized_file, \
         (output / 'unrecognized_points.csv').open('w', encoding='utf-8', newline='') as unrecognized_points_file:
        points_writer = csv.DictWriter(points_file, fieldnames=POINT_COLUMNS)
        segments_writer = csv.DictWriter(segments_file, fieldnames=SEGMENT_COLUMNS)
        flights_writer = csv.DictWriter(flights_file, fieldnames=FLIGHT_COLUMNS)
        unrecognized_writer = csv.DictWriter(unrecognized_file, fieldnames=UNRECOGNIZED_FLIGHT_COLUMNS)
        unrecognized_points_writer = csv.DictWriter(unrecognized_points_file, fieldnames=UNRECOGNIZED_POINT_COLUMNS)
        points_writer.writeheader(); segments_writer.writeheader(); flights_writer.writeheader()
        unrecognized_writer.writeheader(); unrecognized_points_writer.writeheader()

        def write_rejected(result: RejectedFlight) -> None:
            nonlocal unrecognized, unrecognized_points
            unrecognized_writer.writerow(result.report)
            unrecognized_points_writer.writerows(result.points.to_dict('records'))
            unrecognized += 1
            unrecognized_points += len(result.points)
            logger.warning('DLR 跳过不可识别轨迹 [ICAO=%s CALL=%s id=%s]：%d 点；原因=%s',
                           result.report['icao'], result.report['call'], result.report['flight_id'],
                           result.report['points'], result.report['reason'])

        def write_predicted_batch() -> None:
            nonlocal processed, recognized_points
            if not pending_batch:
                return
            features = np.stack([item.features for item in pending_batch])
            with torch.inference_mode():
                outputs = model(torch.from_numpy(features)).cpu().numpy()
            for prepared, output_values in zip(pending_batch, outputs, strict=True):
                labels = _decode_transitions(output_values)
                points, current_segments = _flight_rows(prepared.frame, labels[prepared.clusters[prepared.raw_positions]])
                points_writer.writerows(points.to_dict('records'))
                segments_writer.writerows(current_segments)
                write_segment_progress(progress_database, current_segments)
                current_counts = Counter(row['phase'] for row in current_segments)
                phase_counts.update(current_counts)
                first, last = prepared.frame.iloc[0], prepared.frame.iloc[-1]
                flights_writer.writerow({
                    'flight_id': str(first.flight_id), 'icao': str(first.icao), 'call': str(first.call),
                    'points': int(len(prepared.frame)), 'phase_types': int(len(current_counts)), 'duplicate_time_points': 0,
                    'unique_time_points': int(len(prepared.frame)), 'start_time': _iso(int(first.time_ns)),
                    'end_time': _iso(int(last.time_ns)), 'worker_pid': prepared.worker_pid, 'seconds': prepared.seconds,
                    'segments': int(len(current_segments)), 'phase_counts': str(dict(current_counts)),
                })
                processed += 1
                recognized_points += len(points)
                worker_pids.add(prepared.worker_pid)
                logger.info('DLR 已处理 %d 个航班 [ICAO=%s CALL=%s id=%s]：%d 点 -> %d 个阶段片段，%s；worker=%d；prepare=%.2fs',
                            processed, first.icao, first.call, first.flight_id, len(prepared.frame), len(current_segments),
                            dict(current_counts), prepared.worker_pid, prepared.seconds)
            pending_batch.clear()

        tasks = (frame for _, frame in ordered.groupby('flight_id', sort=False, observed=True))
        def consume(result: PreparedFlight | RejectedFlight) -> None:
            if isinstance(result, RejectedFlight):
                write_rejected(result)
            else:
                pending_batch.append(result)
                if len(pending_batch) >= output_batch_size:
                    write_predicted_batch()

        if actual_workers == 1:
            for task in tasks:
                consume(_prepare_flight(task))
        else:
            configure_worker_threads()
            with ProcessPoolExecutor(max_workers=actual_workers, mp_context=mp.get_context('spawn'),
                                     initializer=configure_worker_threads) as pool:
                pending: dict[object, str] = {}

                def enqueue() -> bool:
                    try:
                        task = next(tasks)
                    except StopIteration:
                        return False
                    future = pool.submit(_prepare_flight, task)
                    pending[future] = str(task['flight_id'].iloc[0])
                    return True

                for _ in range(actual_workers * 2):
                    if not enqueue():
                        break
                while pending:
                    completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed:
                        identity = pending.pop(future)
                        try:
                            consume(future.result())
                        except Exception as exc:
                            raise RuntimeError(f'DLR 航班特征准备失败：{identity}') from exc
                        enqueue()
        write_predicted_batch()
    if recognized_points + unrecognized_points != len(observations):
        raise RuntimeError('DLR 分段未保持输入点守恒')
    return {
        'segmenter': 'dlr_lstm', 'model_path': str(model_path), 'model_sha256': _model_digest(model_path),
        'input_points': int(len(observations)), 'processed_flights': processed,
        'unrecognized_flights': unrecognized, 'unrecognized_points': unrecognized_points,
        'submitted_flights': processed + unrecognized, 'total_segments': int(sum(phase_counts.values())),
        'configured_workers': actual_workers, 'observed_worker_processes': len(worker_pids),
        'phase_segment_counts': {phase: int(phase_counts[phase]) for phase in DLR_PHASES},
        'point_conservation_verified': True,
    }
