"""api.py | 2026-11-08 | 由 GroupIndex 训练独立专家并保存固定记忆 | Codex."""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.app.arguments import ProjectArgumentParser
from src.app.artifacts import read_manifest, write_json
from src.app.contracts import ModuleResult, RunContext, ValidationReport

from .model import ExpertModel


def build_parser() -> argparse.ArgumentParser:
    """定义启动期小模型训练参数；输入只接受直接上游 group 清单。"""
    parser = ProjectArgumentParser(prog='main.py -m train', description='训练每个正常参考组对应的独立双分支记忆专家。')
    parser.add_argument('--input', required=True, type=Path, help='group 模块已发布的 manifest.json')
    parser.add_argument('--gpus', default='0', help='逗号分隔的 CUDA 设备号；不可用时明确回退 CPU')
    parser.add_argument('--d-model', type=int, default=32)
    parser.add_argument('--branch-layers', type=int, default=2)
    parser.add_argument('--branch-heads', type=int, default=4)
    parser.add_argument('--group-layers', type=int, default=1)
    parser.add_argument('--group-heads', type=int, default=4)
    parser.add_argument('--pool-heads', type=int, default=4)
    parser.add_argument('--memory-slots', type=int, default=8)
    parser.add_argument('--ref-limit', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--micro-batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--min-delta', type=float, default=0.0001)
    parser.add_argument('--learning-rate', type=float, default=0.0003)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--precision', choices=('fp32',), default='fp32')
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
    """构造一个无填充计算微批，满足启动期 micro_batch_size=1。"""
    names = stats['names']; mean = np.asarray(stats['mean'], dtype=np.float32); std = np.asarray(stats['std'], dtype=np.float32)
    raw = frame[names].to_numpy(dtype=np.float32)
    normalized = (raw - mean) / std
    space = normalized[:, :4]
    heading = np.deg2rad(frame['heading_deg'].to_numpy(dtype=np.float32))
    kinematic = np.column_stack((normalized[:, 4], np.sin(heading), np.cos(heading))).astype(np.float32)
    times = (frame['time_ns'].to_numpy(dtype=np.float64) / 1_000_000_000.0).astype(np.float32)
    return (torch.from_numpy(space).unsqueeze(0).to(device), torch.from_numpy(kinematic).unsqueeze(0).to(device),
            torch.from_numpy(times).unsqueeze(0).to(device), torch.zeros((1, len(frame)), dtype=torch.bool, device=device))


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
    """解析可用设备；每个 CUDA 设备由一个独立专家队列独占。"""
    requested = [item.strip() for item in gpus.split(',') if item.strip()]
    if not requested or any(not item.isdigit() for item in requested):
        raise ValueError('--gpus 必须为非空 CUDA 设备号列表，例如 0,1,2,3')
    if torch.cuda.is_available():
        numbers = [int(item) for item in requested]
        if len(set(numbers)) != len(numbers):
            raise ValueError('--gpus 不能重复指定同一 CUDA 设备')
        if max(numbers) >= torch.cuda.device_count():
            raise ValueError(f'请求 CUDA:{max(numbers)}，但可用设备数为 {torch.cuda.device_count()}')
        return [torch.device(f'cuda:{number}') for number in numbers]
    return [torch.device('cpu')]


def _seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256('|'.join(str(part) for part in parts).encode()).digest()[:8], 'big')


def _train_expert(group_id: str, expert_index: int, assignments: pd.DataFrame, validation: pd.DataFrame, references: pd.DataFrame,
                  sequences: dict[str, pd.DataFrame], args: argparse.Namespace, root_seed: int, output: Path,
                  logger: object, device: torch.device) -> dict[str, object]:
    """按单目标计算微批训练一个专家，并以冻结验证路由的 BCE 选择检查点。"""
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    train = assignments.loc[assignments['group_id'].eq(group_id)].copy()
    validate = validation.loc[validation['group_id'].eq(group_id)].copy()
    refs = references.loc[references['group_id'].eq(group_id)].copy()
    reference_ids = refs['segment_id'].astype(str).tolist()
    stats = _stats(reference_ids, sequences)
    model = ExpertModel(args.d_model, args.branch_layers, args.branch_heads, args.group_layers, args.group_heads,
                        args.pool_heads, args.memory_slots, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    best_loss, stale, history, best_state = float('inf'), 0, [], None
    checkpoint_selection = 'validation_loss' if not validate.empty else 'training_loss_no_validation'
    if validate.empty:
        logger.warning('专家 %s 没有验证目标；保留冻结组，按训练损失选择检查点，不能用于该组内早停评估', group_id)
    exclude_source = args.ablation != 'no-source-exclusion'
    for epoch in range(args.epochs):
        model.train(); optimizer.zero_grad(set_to_none=True); running = 0.0
        order = train.sample(frac=1.0, random_state=_seed(root_seed, group_id, epoch) % (2 ** 32)).reset_index(drop=True)
        for offset, row in enumerate(order.itertuples(index=False), start=1):
            selected = _select_references(refs, str(row.flight_id), args.ref_limit, _seed(root_seed, epoch, row.segment_id), exclude_source)
            reference_q = torch.cat([_encode(model, sequences[item], stats, device) for item in selected], dim=0)
            memory = model.memory(reference_q)
            logit = model.logits(_encode(model, sequences[str(row.segment_id)], stats, device), memory)
            loss = loss_fn(logit, torch.tensor([float(row.segment_label)], device=device)) / args.batch_size
            loss.backward(); running += float(loss.detach()) * args.batch_size
            if offset % args.batch_size == 0 or offset == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        train_loss = running / len(order)
        validation_loss = _validation_loss(model, validate, refs, sequences, stats, args, root_seed, device, loss_fn)
        checkpoint_loss = validation_loss if checkpoint_selection == 'validation_loss' else train_loss
        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'validation_loss': validation_loss,
                        'checkpoint_loss': checkpoint_loss, 'checkpoint_selection': checkpoint_selection})
        logger.debug('专家 %s epoch=%d；train_loss=%.6f；validation_loss=%s；checkpoint_loss=%.6f；device=%s',
                     group_id, epoch + 1, train_loss,
                     f'{validation_loss:.6f}' if np.isfinite(validation_loss) else 'NA', checkpoint_loss, device)
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
    with torch.no_grad():
        final_ids = reference_ids[:args.ref_limit]
        final_memory = model.memory(torch.cat([_encode(model, sequences[item], stats, device) for item in final_ids], dim=0)).cpu().numpy()
    output.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': best_state, 'architecture': _architecture(args)}, output / 'best.pt')
    np.save(output / 'memory.npy', final_memory)
    write_json(output / 'stats.json', stats); write_json(output / 'reference_ids.json', {'reference_ids': final_ids})
    pd.DataFrame(history).to_csv(output / 'history.csv', index=False, encoding='utf-8')
    logger.info('专家 %s 完成：device=%s；targets=%d；validation=%d；checkpoint_selection=%s；best_loss=%.6f',
                group_id, device, len(train), len(validate), checkpoint_selection, best_loss)
    return {'group_id': group_id, 'expert_id': expert_index, 'targets': int(len(train)), 'validation_targets': int(len(validate)),
            'best_checkpoint_loss': best_loss, 'checkpoint_selection': checkpoint_selection,
            'device': str(device), 'epochs': len(history)}


def _validation_loss(model: ExpertModel, validation: pd.DataFrame, references: pd.DataFrame, sequences: dict[str, pd.DataFrame],
                     stats: dict[str, list[float]], args: argparse.Namespace, root_seed: int, device: torch.device, loss_fn: nn.Module) -> float:
    if validation.empty:
        return float('inf')
    model.eval(); losses: list[float] = []
    with torch.no_grad():
        full_ids = references['segment_id'].astype(str).tolist()[:args.ref_limit]
        memory = model.memory(torch.cat([_encode(model, sequences[item], stats, device) for item in full_ids], dim=0))
        for row in validation.itertuples(index=False):
            logit = model.logits(_encode(model, sequences[str(row.segment_id)], stats, device), memory)
            losses.append(float(loss_fn(logit, torch.tensor([float(row.segment_label)], device=device))))
    model.train()
    return float(np.mean(losses))


def _architecture(args: argparse.Namespace) -> dict[str, object]:
    return {name: getattr(args, name) for name in ('d_model', 'branch_layers', 'branch_heads', 'group_layers', 'group_heads', 'pool_heads', 'memory_slots', 'dropout')}


def _train_device_queue(rows: list[object], device: torch.device, assignments: pd.DataFrame, validation: pd.DataFrame,
                        references: pd.DataFrame, sequences: dict[str, pd.DataFrame], args: argparse.Namespace,
                        root_seed: int, model_dir: Path, logger: object) -> list[dict[str, object]]:
    """在一个设备上顺序训练分配给它的专家，避免同一 GPU 出现并发模型。"""
    summaries: list[dict[str, object]] = []
    for row in rows:
        summaries.append(_train_expert(str(row.group_id), int(row.expert_id), assignments, validation, references,
                                       sequences, args, root_seed, model_dir / str(row.group_id), logger, device))
    return summaries


def run(context: RunContext, args: argparse.Namespace) -> ModuleResult:
    """读取直接 group 输入，溯源 segment S0，并发训练不同 GPU 队列中的独立专家。"""
    if args.micro_batch_size != 1 or min(args.d_model, args.branch_layers, args.branch_heads, args.group_layers, args.group_heads,
                                         args.pool_heads, args.memory_slots, args.ref_limit, args.batch_size, args.epochs, args.patience) < 1:
        raise ValueError('当前首版仅支持 micro-batch-size=1，且网络层数、头数、批次、epoch 与参考上限必须为正')
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
    sequences, _ = _load_sequences(segment_manifest)
    needed = set(assignments['segment_id'].astype(str)) | set(validation['segment_id'].astype(str)) | set(references['segment_id'].astype(str))
    if not needed.issubset(sequences):
        raise ValueError('group 索引包含 segment S0 中不存在的片段')
    devices = _devices(args.gpus)
    rows = list(groups.sort_values('expert_id', kind='stable').itertuples(index=False))
    queues = [rows[offset::len(devices)] for offset in range(len(devices))]
    model_dir = context.staging_dir / 'models'
    context.logger.info('开始训练独立专家：groups=%d；训练目标=%d；验证目标=%d；devices=%s；并发队列=%d',
                        len(groups), len(assignments), len(validation), ','.join(str(item) for item in devices), len(devices))
    context.logger.debug('训练输入已就绪：S0 sequences=%d；references=%d；每设备专家队列=%s',
                         len(sequences), len(references), [[str(row.group_id) for row in queue] for queue in queues])
    if len(devices) == 1:
        summaries = _train_device_queue(queues[0], devices[0], assignments, validation, references, sequences,
                                        args, context.seed, model_dir, context.logger)
    else:
        with ThreadPoolExecutor(max_workers=len(devices), thread_name_prefix='cafad-expert') as executor:
            futures = [executor.submit(_train_device_queue, queue, device, assignments, validation, references, sequences,
                                       args, context.seed, model_dir, context.logger)
                       for queue, device in zip(queues, devices, strict=True) if queue]
            summaries = [summary for future in futures for summary in future.result()]
    summaries.sort(key=lambda item: int(item['expert_id']))
    pd.DataFrame(summaries).to_csv(context.staging_dir / 'expert_summary.csv', index=False, encoding='utf-8')
    return ModuleResult(summary={'input_manifest': str(group_manifest), 'segment_manifest': str(segment_manifest),
                                 'experts': len(summaries), 'targets': int(len(assignments)), 'validation_targets': int(len(validation)),
                                 'ablation': args.ablation, 'precision': args.precision}, dependencies=(group_manifest,))


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
