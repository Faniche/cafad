"""api.py | 2026-11-08 | 由 GroupIndex 训练独立专家并保存固定记忆 | Codex."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.app.arguments import ProjectArgumentParser
from src.app.artifacts import read_manifest, write_json
from src.app.contracts import ModuleResult, RunContext, ValidationReport
from src.utils.precision import precision_context, validate_precision

from .model import ExpertModel
from .progress import ProgressReporter
from .batching import collate_frames, encode_frames, length_batches, accumulation_windows, reference_memories
from .timing import StageTimer, dominant_timing, format_timing


def build_parser() -> argparse.ArgumentParser:
    """定义启动期小模型训练参数；输入只接受直接上游 group 清单。"""
    parser = ProjectArgumentParser(prog='main.py -m train', description='训练每个正常参考组对应的独立双分支记忆专家。')
    parser.add_argument('--input', required=True, type=Path, help='group 模块已发布的 manifest.json')
    parser.add_argument('--gpus', default='0,1,2,3', help='逗号分隔的 CUDA 设备号；故障时终止，不自动回退；仅显式 cpu 使用CPU')
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--branch-layers', type=int, default=2)
    parser.add_argument('--branch-heads', type=int, default=4)
    parser.add_argument('--group-layers', type=int, default=1)
    parser.add_argument('--group-heads', type=int, default=4)
    parser.add_argument('--pool-heads', type=int, default=4)
    parser.add_argument('--memory-slots', type=int, default=8)
    parser.add_argument('--ref-limit', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=16, help='每次优化器更新的有效目标片段数；不足时按实际尾批数量归一化')
    parser.add_argument('--micro-batch-size', type=int, default=4, help='同时编码的片段数上限；按长度组批并动态填充，调大增加显存需求')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--min-delta', type=float, default=0.0001)
    parser.add_argument('--learning-rate', type=float, default=0.0003)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--precision', choices=('fp32', 'fp16', 'bf16'), default='bf16',
                        help='计算精度：默认BF16 AMP；FP16 AMP使用梯度缩放；FP32关闭AMP。校准/检测沿用检查点精度')
    parser.add_argument('--ablation', choices=('none', 'no-source-exclusion'), default='none',
                        help='当前仅实现同源排除消融；其他结构消融待对应网络实现后再开放')
    return parser


def _project_path(context: RunContext, value: Path) -> Path:
    return value if value.is_absolute() else context.project_root / value


def _segment_manifest(group_manifest: Path) -> Path:
    """从直接 group 清单的已记录依赖恢复唯一 segment 清单。"""
    group = read_manifest(group_manifest, expected_module='group')
    matches: list[Path] = []
    for item in group.get('dependencies', []):
        if isinstance(item, dict) and isinstance(item.get('manifest'), str):
            path = Path(item['manifest'])
            if read_manifest(path).get('module') == 'segment':
                matches.append(path)
    if len(matches) != 1:
        raise ValueError('group 清单无法唯一溯源其 segment 输入')
    return matches[0]


def _load_sequences(segment_manifest: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """由 S0 读取变长最终片段；模型从不读取干净副本或注入事件位置。"""
    data = segment_manifest.parent / 'data'
    columns = ['record_id', 'time_ns', 'longitude_deg', 'latitude_deg', 'geometric_altitude_m', 'barometric_altitude_m', 'heading_deg', 'speed_mps']
    points = pd.read_csv(data / 'point_segment_map.csv', usecols=['segment_id', 'record_id'],
                         dtype={'segment_id': 'string', 'record_id': 'string'})
    observations = pd.read_csv(data / 'input_observations.csv', usecols=columns, dtype={'record_id': 'string'})
    values = points[['segment_id', 'record_id']].merge(observations[columns], on='record_id', how='left', validate='one_to_one')
    if values.isna().any().any() or values['segment_id'].isna().any():
        raise ValueError('训练片段无法与最终观测完整对齐')
    sequences = {str(segment_id): frame.sort_values('time_ns', kind='stable').reset_index(drop=True)
                 for segment_id, frame in values.groupby('segment_id', sort=False, observed=True)}
    metadata = pd.read_csv(data / 'segments.csv', usecols=['segment_id', 'flight_id'],
                           dtype={'segment_id': 'string', 'flight_id': 'string'})
    return sequences, metadata


def _stats(reference_ids: list[str], sequences: dict[str, pd.DataFrame]) -> dict[str, list[float]]:
    """仅由该专家正常参考的有效观测拟合输入统计量。"""
    values = pd.concat([sequences[item] for item in reference_ids], ignore_index=True)
    names = ['longitude_deg', 'latitude_deg', 'geometric_altitude_m', 'barometric_altitude_m', 'speed_mps']
    mean = values[names].mean().to_numpy(dtype=np.float32)
    std = values[names].std(ddof=0).to_numpy(dtype=np.float32)
    std[std == 0] = 1.0
    return {'names': names, 'mean': mean.tolist(), 'std': std.tolist()}


def _tensor(frame: pd.DataFrame, stats: dict[str, list[float]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """保留检测调用的单片段接口，与训练共用同一特征转换和 mask 规则。"""
    return collate_frames([frame], stats, device)


def _encode(model: ExpertModel, frame: pd.DataFrame, stats: dict[str, list[float]], device: torch.device) -> torch.Tensor:
    space, kinematic, times, mask = _tensor(frame, stats, device)
    return model.encode(space, kinematic, times, mask)


def _select_references(references: pd.DataFrame, flight_id: str, limit: int, seed: int, exclude_source: bool) -> list[str]:
    candidates = references if not exclude_source else references.loc[~references['flight_id'].eq(flight_id)]
    if candidates.empty:
        raise ValueError('同源排除后没有正常参考，group 产物或参考数量约束不一致')
    ids = candidates['segment_id'].astype(str).tolist()
    if len(ids) <= limit:
        return ids
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(np.asarray(ids, dtype=object), size=limit, replace=False).tolist())


def _devices(gpus: str) -> list[torch.device]:
    """显式选择设备；GPU初始化失败不能伪装成正常CPU实验。"""
    if gpus.strip().lower() == 'cpu':
        return [torch.device('cpu')]
    requested = [item.strip() for item in gpus.split(',') if item.strip()]
    if not requested or any(not item.isdigit() for item in requested):
        raise ValueError('--gpus 必须为非空 CUDA 设备号列表，例如 0,1,2,3')
    if torch.cuda.is_available():
        numbers = [int(item) for item in requested]
        if len(set(numbers)) != len(numbers):
            raise ValueError('--gpus 不能重复指定同一 CUDA 设备')
        if max(numbers) >= torch.cuda.device_count():
            raise RuntimeError(f'CUDA_DEVICE_UNAVAILABLE: 请求 CUDA:{max(numbers)}，但可用设备数为 {torch.cuda.device_count()}；终止运行')
        return [torch.device(f'cuda:{number}') for number in numbers]
    raise RuntimeError('CUDA_DEVICE_UNAVAILABLE: 已指定GPU，但CUDA初始化失败或不可用；禁止自动回退CPU')


def _seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256('|'.join(str(part) for part in parts).encode()).digest()[:8], 'big')


def _train_expert(group_id: str, expert_index: int, assignments: pd.DataFrame, validation: pd.DataFrame, references: pd.DataFrame,
                  sequences: dict[str, pd.DataFrame], args: argparse.Namespace, root_seed: int, output: Path,
                  logger: object, device: torch.device) -> dict[str, object]:
    """按长度组微批训练独立专家，每个目标在批内仍有自己的参考集合与记忆。"""
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    validate_precision(args.precision, device)
    progress = ProgressReporter(logger, group_id, device, args.epochs)
    progress.update(force=True, state='initializing')
    train = assignments.loc[assignments['group_id'].eq(group_id)].copy()
    validate = validation.loc[validation['group_id'].eq(group_id)].copy()
    refs = references.loc[references['group_id'].eq(group_id)].copy()
    reference_ids = refs['segment_id'].astype(str).tolist()
    stats = _stats(reference_ids, sequences)
    model = ExpertModel(args.d_model, args.branch_layers, args.branch_heads, args.group_layers, args.group_heads,
                        args.pool_heads, args.memory_slots, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=args.precision == 'fp16')
    logger.info('专家 %s 计算精度=%s；AMP=%s；梯度缩放=%s；参数与检查点保留FP32；device=%s',
                group_id, args.precision, args.precision != 'fp32', scaler.is_enabled(), device)
    loss_fn = nn.BCEWithLogitsLoss()
    best_loss, stale, history, best_state = float('inf'), 0, [], None
    checkpoint_selection = 'validation_loss' if not validate.empty else 'training_loss_no_validation'
    if validate.empty:
        logger.warning('专家 %s 没有验证目标；保留冻结组，按训练损失选择检查点，不能用于该组内早停评估', group_id)
    exclude_source = args.ablation != 'no-source-exclusion'
    for epoch in range(args.epochs):
        epoch_started = StageTimer(device)
        progress.update(force=True, state='training', epoch=epoch + 1, completed=0, total=len(train))
        model.train(); running = 0.0; completed = 0; micro_batches = 0; padded = 0; valid = 0
        with epoch_started.stage('train_batch_plan'):
            batches = length_batches(train, sequences, args.micro_batch_size, _seed(root_seed, group_id, epoch))
        for window in accumulation_windows(batches, args.batch_size):
            window_size = sum(len(batch) for batch in window)
            with epoch_started.stage('optimizer_zero_grad', gpu=True):
                optimizer.zero_grad(set_to_none=True)
            for batch in window:
                with epoch_started.stage('train_prepare'):
                    selections = [_select_references(refs, str(row.flight_id), args.ref_limit,
                                  _seed(root_seed, epoch, row.segment_id), exclude_source) for row in batch.itertuples(index=False)]
                    frames = [sequences[str(item)] for item in batch['segment_id']]
                    labels = torch.tensor(batch['segment_label'].to_numpy(dtype=np.float32), device=device)
                with epoch_started.stage('train_reference_memory', gpu=True):
                    with precision_context(args.precision, device):
                        memory = reference_memories(model, selections, sequences, stats, device, args.micro_batch_size)
                with epoch_started.stage('train_target_forward_and_loss', gpu=True):
                    with precision_context(args.precision, device):
                        target_q = model.encode(*collate_frames(frames, stats, device))
                        loss = loss_fn(model.logits(target_q, memory).float(), labels)
                with epoch_started.stage('train_backward', gpu=True):
                    scaler.scale(loss * (len(batch) / window_size)).backward()
                running += float(loss.detach()) * len(batch)
                completed += len(batch); micro_batches += 1
                valid += sum(len(frame) for frame in frames)
                padded += len(frames) * max(len(frame) for frame in frames)
                progress.update(completed=completed, train_loss=running / completed,
                                micro_batch_size=len(batch), effective_batch_size=window_size)
            with epoch_started.stage('optimizer_step', gpu=True):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
            if scaler.is_enabled() and scaler.get_scale() < previous_scale:
                logger.warning('专家 %s epoch=%d FP16梯度非有限，跳过本次更新并降低scale：%g→%g',
                               group_id, epoch + 1, previous_scale, scaler.get_scale())
        logger.debug('专家 %s epoch=%d；微批数=%d；有效目标=%d；目标有效点=%d；填充后点=%d；padding比例=%.4f',
                     group_id, epoch + 1, micro_batches, completed, valid, padded, 1 - valid / padded)
        train_loss = running / completed
        progress.update(force=True, state='validating', completed=0, total=len(validate), train_loss=train_loss)
        validation_loss, validation_timing = _validation_loss(model, validate, refs, sequences, stats, args, root_seed,
                                                              device, loss_fn, progress)
        progress.update(force=True, completed=len(validate), validation_loss=validation_loss)
        checkpoint_loss = validation_loss if checkpoint_selection == 'validation_loss' else train_loss
        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'validation_loss': validation_loss,
                        'checkpoint_loss': checkpoint_loss, 'checkpoint_selection': checkpoint_selection})
        logger.debug('专家 %s epoch=%d；train_loss=%.6f；validation_loss=%s；checkpoint_loss=%.6f；device=%s',
                     group_id, epoch + 1, train_loss,
                     f'{validation_loss:.6f}' if np.isfinite(validation_loss) else 'NA', checkpoint_loss, device)
        training_timing = epoch_started.summary_seconds()
        logger.debug('专家 %s epoch=%d 阶段耗时；训练=%s；训练慢项=%s；验证=%s；验证慢项=%s', group_id, epoch + 1,
                     format_timing(training_timing), dominant_timing(training_timing),
                     format_timing(validation_timing), dominant_timing(validation_timing))
        if checkpoint_loss < best_loss - args.min_delta:
            best_loss, stale = checkpoint_loss, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError('专家训练未产生有效检查点')
    model.load_state_dict(best_state); model.eval()
    progress.update(force=True, state='saving', best_loss=best_loss)
    saving_timing = StageTimer(device)
    with saving_timing.stage('final_reference_memory', gpu=True):
        with torch.no_grad(), precision_context(args.precision, device):
            final_ids = reference_ids[:args.ref_limit]
            reference_q = encode_frames(model, [sequences[item] for item in final_ids], stats, device, args.micro_batch_size)
            final_memory = model.memory(reference_q).float().cpu().numpy()
    output.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': best_state, 'architecture': _architecture(args), 'precision': args.precision}, output / 'best.pt')
    np.save(output / 'memory.npy', final_memory)
    write_json(output / 'stats.json', stats); write_json(output / 'reference_ids.json', {'reference_ids': final_ids})
    pd.DataFrame(history).to_csv(output / 'history.csv', index=False, encoding='utf-8')
    progress.update(force=True, state='completed', early_stopped=len(history) < args.epochs)
    logger.debug('专家 %s 保存阶段耗时；%s', group_id, format_timing(saving_timing.summary_seconds()))
    logger.info('专家 %s 完成：device=%s；targets=%d；validation=%d；checkpoint_selection=%s；best_loss=%.6f',
                group_id, device, len(train), len(validate), checkpoint_selection, best_loss)
    return {'group_id': group_id, 'expert_id': expert_index, 'targets': int(len(train)), 'validation_targets': int(len(validate)),
            'best_checkpoint_loss': best_loss, 'checkpoint_selection': checkpoint_selection,
            'device': str(device), 'precision': args.precision, 'epochs': len(history)}


def _validation_loss(model: ExpertModel, validation: pd.DataFrame, references: pd.DataFrame, sequences: dict[str, pd.DataFrame],
                     stats: dict[str, list[float]], args: argparse.Namespace, root_seed: int, device: torch.device, loss_fn: nn.Module,
                     progress: ProgressReporter | None = None) -> tuple[float, dict[str, tuple[float, float | None]]]:
    if validation.empty:
        return float('inf'), {}
    model.eval(); total_loss = 0.0; completed = 0
    timing = StageTimer(device)
    with timing.stage('validation_reference_memory', gpu=True):
        with torch.no_grad(), precision_context(args.precision, device):
            full_ids = references['segment_id'].astype(str).tolist()[:args.ref_limit]
            memory = model.memory(encode_frames(model, [sequences[item] for item in full_ids], stats, device, args.micro_batch_size))
    with torch.no_grad(), precision_context(args.precision, device):
        for batch in length_batches(validation, sequences, args.micro_batch_size):
            with timing.stage('validation_prepare'):
                frames = [sequences[str(item)] for item in batch['segment_id']]
                labels = torch.tensor(batch['segment_label'].to_numpy(dtype=np.float32), device=device)
            with timing.stage('validation_target_forward_and_loss', gpu=True):
                target_q = model.encode(*collate_frames(frames, stats, device))
                loss = loss_fn(model.logits(target_q, memory).float(), labels)
            total_loss += float(loss) * len(batch)
            completed += len(batch)
            if progress is not None:
                progress.update(completed=completed)
    model.train()
    return total_loss / completed, timing.summary_seconds()


def _architecture(args: argparse.Namespace) -> dict[str, object]:
    return {name: getattr(args, name) for name in ('d_model', 'branch_layers', 'branch_heads', 'group_layers', 'group_heads', 'pool_heads', 'memory_slots', 'dropout')}


def run(context: RunContext, args: argparse.Namespace) -> ModuleResult:
    """读取直接 group 输入，每设备一个进程从共享池动态领取专家。"""
    if min(args.micro_batch_size, args.d_model, args.branch_layers, args.branch_heads, args.group_layers, args.group_heads,
                                         args.pool_heads, args.memory_slots, args.ref_limit, args.batch_size, args.epochs, args.patience) < 1:
        raise ValueError('微批、有效批、网络层数、头数、epoch 与参考上限必须为正')
    if not 0 <= args.dropout < 1 or args.learning_rate <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        raise ValueError('训练优化器参数无效')
    group_manifest = _project_path(context, args.input).resolve()
    read_manifest(group_manifest, expected_module='group')
    segment_manifest = _segment_manifest(group_manifest)
    group_data = group_manifest.parent / 'data'
    groups = pd.read_csv(group_data / 'groups.csv', dtype={'group_id': 'string'})
    references = pd.read_csv(group_data / 'references.csv', dtype={'group_id': 'string', 'segment_id': 'string', 'flight_id': 'string'})
    assignments = pd.read_csv(group_data / 'train_assignments.csv', dtype={'group_id': 'string', 'segment_id': 'string', 'flight_id': 'string'})
    validation = pd.read_csv(group_data / 'validation_assignments.csv', dtype={'group_id': 'string', 'segment_id': 'string', 'flight_id': 'string'})
    devices = _devices(args.gpus)
    for device in devices:
        validate_precision(args.precision, device)
    # 大任务先入池，降低末尾留下单个大专家的概率；不改变训练归属。
    counts = assignments.groupby('group_id').size().to_dict()
    rows = sorted(groups.itertuples(index=False), key=lambda row: (-counts.get(row.group_id, 0), row.expert_id))
    model_dir = context.staging_dir / 'models'
    context.logger.info('开始进程池训练：groups=%d；训练目标=%d；验证目标=%d；devices=%s；并发进程上限=%d',
                        len(groups), len(assignments), len(validation), ','.join(str(item) for item in devices), len(devices))
    context.logger.info('长度组批：micro_batch_size=%d；每次更新目标数=%d；动态padding+mask；尾批按实际目标数归一化',
                        args.micro_batch_size, args.batch_size)
    from .parallel import train_pool
    summaries = train_pool([(str(row.group_id), int(row.expert_id)) for row in rows], devices, group_manifest,
                           segment_manifest, args, context.seed, model_dir, context.logger, context.workers)
    summaries.sort(key=lambda item: int(item['expert_id']))
    pd.DataFrame(summaries).to_csv(context.staging_dir / 'expert_summary.csv', index=False, encoding='utf-8')
    return ModuleResult(summary={'input_manifest': str(group_manifest), 'segment_manifest': str(segment_manifest),
                                 'experts': len(summaries), 'targets': int(len(assignments)), 'validation_targets': int(len(validation)),
                                 'ablation': args.ablation, 'precision': args.precision,
                                 'micro_batch_size': args.micro_batch_size, 'batch_size': args.batch_size,
                                 'batching': 'length-sorted-shuffled-microbatches'}, dependencies=(group_manifest,))


def validate_outputs(context: RunContext, args: argparse.Namespace, result: ModuleResult) -> ValidationReport:
    """检查每个有效组都有可重新加载的权重、统计量、参考 ID 和固定记忆。"""
    summary = pd.read_csv(context.staging_dir / 'expert_summary.csv', dtype={'group_id': 'string'})
    if summary.empty or summary['group_id'].duplicated().any():
        raise ValueError('训练没有为每个有效组产出唯一专家摘要')
    for group_id in summary['group_id'].astype(str):
        directory = context.staging_dir / 'models' / group_id
        required = [directory / name for name in ('best.pt', 'memory.npy', 'stats.json', 'reference_ids.json', 'history.csv')]
        if not all(item.is_file() for item in required):
            raise ValueError(f'专家 {group_id} 缺少权重、记忆或配置')
        memory = np.load(directory / 'memory.npy', allow_pickle=False)
        if memory.shape != (args.memory_slots, 2 * args.d_model) or not np.isfinite(memory).all():
            raise ValueError(f'专家 {group_id} 最终记忆形状或数值无效')
        checkpoint = torch.load(directory / 'best.pt', map_location='cpu', weights_only=True)
        if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
            raise ValueError(f'专家 {group_id} 检查点无法安全读取')
    return ValidationReport(checks={'experts': int(len(summary)), 'models_reloadable': True, 'fixed_memories_valid': True})
