"""parameter_search.py | 2026-11-09 | 随机20组后以同一研究的历史进行TPE，单组独占四卡 | Codex."""
from pathlib import Path
import sys

# 独立脚本入口只建立项目导入路径，业务执行位于 main。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from datetime import datetime
import hashlib
import json
import logging
import time
import math
import os
from src.app.artifacts import write_json
from src.utils.strings import PathUtils
from src.utils.logging_utils import setup_project_logging, close_project_logging
from scripts.search_pipeline import search_lock, pipeline, export_results, FatalDeviceError


def main() -> int:
    parser = argparse.ArgumentParser(description='固定 inject 产物，串行运行随机搜索或TPE，每组经四个标准模块。')
    parser.add_argument('--mode', choices=('random', 'tpe'), required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/parameter_search.json'))
    args = parser.parse_args()
    root = PathUtils.get_project_root()
    logger, logfile = setup_project_logging(console=True, project_root=root, name='cafad.parameter_search', level=logging.DEBUG,
        filename=f'search-{args.mode}-{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}.log')
    started = time.monotonic()
    try:
        import optuna
        optuna.logging.disable_default_handler()
        optuna_logger = logging.getLogger('optuna')
        optuna_logger.handlers = list(logger.handlers)
        optuna_logger.propagate = False
        config = json.loads((root / args.config).read_text(encoding='utf-8'))
        name = config['study_name']
        if not name or any(character not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for character in name):
            raise ValueError('study_name 只允许字母、数字、下划线和连字符')
        if config['detect']['split'] != 'val' or config['random_trials'] != 20:
            raise ValueError('随机搜索固定20组，搜索评分必须使用val')
        if config['train_fixed']['precision'] not in {'fp32', 'fp16', 'bf16'}:
            raise ValueError('搜索精度必须为fp32、fp16或bf16')
        input_value = str(config.get('inject_manifest', '')).strip()
        if not input_value:
            raise ValueError('请先在 parameter_search.json 的 inject_manifest 中填写已发布 inject 产物')
        injected = (root / input_value).resolve()
        manifest = json.loads(injected.read_text(encoding='utf-8'))
        if manifest['status'] != 'SUCCEEDED' or manifest['module'] != 'inject':
            raise ValueError('inject 输入未成功发布')
        directory = root / 'data/parameter_search' / name
        directory.mkdir(parents=True, exist_ok=True)
        # 同一研究冻结配置及方法代码，避免把代码修改前后的成绩混入TPE。
        sources = sorted(path for folder in ('train','calibrate','detect','evaluate','group')
                         for path in (root / 'src/modules' / folder).glob('*.py'))
        sources.append(root / 'src/utils/precision.py')
        fingerprint = hashlib.sha256(b''.join(path.read_bytes() for path in sources) + injected.read_bytes()).hexdigest()
        frozen = {'config': config, 'source_and_inject_sha256': fingerprint}
        with search_lock(directory.parent / 'search.lock'):
            snapshot = directory / 'search_config.json'
            if snapshot.exists() and json.loads(snapshot.read_text(encoding='utf-8')) != frozen:
                raise ValueError('研究配置或方法代码已改变，请为新实验修改study_name；旧研究不可混用')
            write_json(snapshot, frozen)
            study = optuna.create_study(study_name=name, storage='sqlite:///' + str(directory / 'study.sqlite3'),
                                        direction='maximize', load_if_exists=True)
            # 同一锁下遗留RUNNING说明上次意外退出，标失败并保留中间产物。
            for trial in study.get_trials(deepcopy=False):
                if trial.state == optuna.trial.TrialState.RUNNING:
                    study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
            previous = study.get_trials(deepcopy=False)
            random_done = [trial for trial in previous if trial.user_attrs.get('phase') == 'random']
            if args.mode == 'tpe' and (len(random_done) < 20 or not any(trial.state.name == 'COMPLETE' for trial in random_done)):
                raise ValueError('先完成随机20组且至少一组成功，才可基于这些结果执行TPE')
            target = config[args.mode + '_trials']
            done = sum(trial.user_attrs.get('phase') == args.mode for trial in previous)
            logger.info('搜索开始：mode=%s；已尝试=%d；目标=%d；四卡仅执行一组；日志=%s', args.mode, done, target, logfile)
            for _ in range(done, target):
                seed = config['search_seed'] + len(study.trials)
                study.sampler = (optuna.samplers.RandomSampler(seed=seed) if args.mode == 'random' else
                                 optuna.samplers.TPESampler(seed=seed, n_startup_trials=0))
                trial = study.ask()
                trial.set_user_attr('phase', args.mode)
                parameters = {key: trial.suggest_categorical(key, spec['choices']) if 'choices' in spec else
                              trial.suggest_float(key, spec['low'], spec['high'], log=spec.get('log', False))
                              for key, spec in config['space'].items()}
                trial_dir = directory / f'trial_{trial.number:04d}'
                trial_dir.mkdir(exist_ok=True)
                write_json(trial_dir / 'parameters.json', {'phase': args.mode, 'parameters': parameters, 'log': str(logfile)})
                begin = time.monotonic()
                logger.info('开始组合 %d：%s', trial.number, parameters)
                try:
                    score = pipeline(root, config, parameters, trial_dir, logger)
                    if not math.isfinite(score):
                        raise ValueError('验证F1不是有限数')
                    trial.set_user_attr('artifacts', json.loads((trial_dir / 'artifacts.json').read_text(encoding='utf-8')))
                    write_json(trial_dir / 'result.json', {'state': 'success', 'validation_segment_f1': score,
                                                         'seconds': time.monotonic() - begin})
                    study.tell(trial, score)
                    logger.info('组合 %d 完成：验证片段F1=%.6f', trial.number, score)
                except Exception as exc:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    write_json(trial_dir / 'result.json', {'state': 'failure', 'error': str(exc)})
                    if isinstance(exc, FatalDeviceError):
                        logger.error('组合 %d 发生设备故障，保留全部证据并终止搜索，待设备恢复后再处理', trial.number)
                        raise
                    logger.exception('组合 %d 失败，保留中间产物并继续下一组', trial.number)
                except BaseException:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    write_json(trial_dir / 'result.json', {'state': 'interrupted'})
                    raise
                finally:
                    export_results(study, directory)
            logger.info('搜索阶段完成：%s；结果目录=%s', args.mode, directory)
            export_results(study, directory)
            if not any(trial.state.name == 'COMPLETE' and trial.user_attrs.get('phase') == args.mode for trial in study.trials):
                raise RuntimeError('本阶段无成功组合，请检查失败日志')
        return 0
    except (Exception, KeyboardInterrupt):
        logger.exception('搜索终止或启动失败')
        return 1
    finally:
        logger.info('搜索结束时间=%s；运行秒数=%.1f', datetime.now().astimezone().isoformat(), time.monotonic()-started)
        close_project_logging(logger)


if __name__ == '__main__':
    raise SystemExit(main())
