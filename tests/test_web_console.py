"""test_web_console.py | 2026-11-04 | Web 控制台认证、文件浏览及非阻塞日志测试 | Codex."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.web_console.server import create_handler
from src.web_console.restart import RestartController
from src.web_console.history import delete_history_run, history_detail, list_module_history
from src.web_console.service import JobManager, build_command, module_schemas


ROOT = Path(__file__).resolve().parents[1]
TOKEN = 'fixture-token'


@contextmanager
def test_server(project_root: Path) -> Iterator[ThreadingHTTPServer]:
    """启动临时本机 Web 服务，测试结束后关闭线程与端口。"""
    manager = JobManager(project_root, lambda message, *values: None)
    server = ThreadingHTTPServer(('127.0.0.1', 0), create_handler(project_root, manager, TOKEN))
    server.cafad_manager = manager  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class WebConsoleTests(unittest.TestCase):
    """控制台只接受已声明参数，并要求先完成令牌摘要登录。"""

    def setUp(self) -> None:
        self.schemas = module_schemas()

    def test_build_command_uses_argument_array_and_omits_optional_multiple_value(self) -> None:
        """参数值始终是独立数组元素，空的可选 ratios 保留 argparse 默认值。"""
        command = build_command(ROOT, {
            'module': 'split',
            'global': {'seed': '10', 'console': True},
            'module_args': {'input': 'data/prep/manifest.json', 'ratios': None},
        }, self.schemas)
        self.assertEqual(command[:4], (sys.executable, str(ROOT / 'main.py'), '--seed', '10'))
        self.assertIn('--console', command)
        module_index = command.index('-m')
        self.assertEqual(command[module_index:module_index + 3], ('-m', 'split', '--input'))
        self.assertNotIn('--ratios', command)
        self.assertIn('data/prep/manifest.json', command)

    def test_split_plan_arguments_are_passed_as_three_value_ratio_lists(self) -> None:
        """网页自定义比例表最终仍映射到 split API 的受限参数数组。"""
        command = build_command(ROOT, {
            'module': 'split',
            'global': {},
            'module_args': {
                'input': 'data/0_preprocess/fixture/1/manifest.json',
                'ratios': ['0.7', '0.15', '0.15'],
                'normal_ratios': ['0.5', '0.6', '0.4'],
                'anomaly_ratios': ['0.5', '0.4', '0.6'],
            },
        }, self.schemas)
        self.assertEqual(command[command.index('--ratios') + 1:command.index('--ratios') + 4], ('0.7', '0.15', '0.15'))
        self.assertEqual(command[command.index('--normal-ratios') + 1:command.index('--normal-ratios') + 4],
                         ('0.5', '0.6', '0.4'))
        self.assertEqual(command[command.index('--anomaly-ratios') + 1:command.index('--anomaly-ratios') + 4],
                         ('0.5', '0.4', '0.6'))

    def test_inject_derives_preprocess_manifest_from_selected_split_manifest(self) -> None:
        """网页只选择最近的 split 产物，早期 preprocess 输入必须沿依赖链自动恢复。"""
        with TemporaryDirectory() as directory:
            project_root = Path(directory)
            preprocess = project_root / 'data' / '0_preprocess' / 'fixture' / '1#20261107_100000#42' / 'manifest.json'
            split = project_root / 'data' / '1_split' / 'fixture' / '1#20261107_100100#42' / 'manifest.json'
            preprocess.parent.mkdir(parents=True)
            split.parent.mkdir(parents=True)
            preprocess.write_text(json.dumps({
                'status': 'SUCCEEDED', 'module': 'preprocess', 'dataset_id': 'fixture', 'outputs': [], 'dependencies': [],
            }), encoding='utf-8')
            split.write_text(json.dumps({
                'status': 'SUCCEEDED', 'module': 'split', 'dataset_id': 'fixture', 'outputs': [],
                'dependencies': [{'manifest': str(preprocess.resolve()), 'sha256': 'fixture'}],
            }), encoding='utf-8')
            command = build_command(project_root, {
                'module': 'inject', 'global': {},
                'module_args': {
                    'input': 'malicious/ignored/manifest.json',
                    'split_manifest': split.relative_to(project_root).as_posix(),
                    'protocol': 'configs/injection_protocol_example.yaml',
                },
            }, self.schemas)
            self.assertEqual(command[command.index('--input') + 1], preprocess.relative_to(project_root).as_posix())
            self.assertEqual(command[command.index('--split-manifest') + 1], split.relative_to(project_root).as_posix())

    def test_preprocess_display_defaults_are_declared_in_the_static_form(self) -> None:
        """首次渲染 preprocess 时显示原始数据常用单位，仍允许下拉列表覆盖。"""
        script = (ROOT / 'src' / 'web_console' / 'static' / 'app.js').read_text(encoding='utf-8')
        self.assertIn("altitude_unit: 'm'", script)
        self.assertIn("speed_unit: 'km/h'", script)
        self.assertIn("heading_unit: 'deg'", script)
        self.assertIn('moduleOrder.forEach', script)
        self.assertIn('moduleView.hidden = !isModuleImplemented(name)', script)
        self.assertIn('split-plan-table', script)
        self.assertIn('splitModuleArguments', script)
        self.assertIn('populateUpstreamRuns', script)
        self.assertIn("default: true", script)
        self.assertIn('auto_derived', script)
        self.assertIn("default: 'DEBUG'", script)
        self.assertIn('stopButton', script)
        self.assertIn('runElapsed', script)
        self.assertIn('startElapsed', script)
        self.assertIn('运行计时', script)
        self.assertIn('loadHistory', script)
        self.assertNotIn('已由所选最近上游产物自动推导', script)

    def test_job_manager_drains_large_console_output_without_browser_polling(self) -> None:
        """后台读线程持续排空 PIPE；即使浏览器暂不读取，子进程也能结束。"""
        messages: list[str] = []

        def logger(message: str, *values: object) -> None:
            messages.append(message % values if values else message)

        manager = JobManager(ROOT, logger)
        job = manager.start((sys.executable, '-c', "for _ in range(12000): print('log')"), 'preprocess')
        deadline = time.monotonic() + 10.0
        while manager.snapshot(job.job_id)['status'] == 'running' and time.monotonic() < deadline:
            time.sleep(0.02)
        snapshot = manager.snapshot(job.job_id)
        self.assertEqual(snapshot['status'], 'succeeded')
        data = manager.logs_after(job.job_id, 0)
        self.assertTrue(data['truncated'])
        self.assertTrue(any('退出码=0' in line for line in data['lines']))
        self.assertTrue(any('网页任务结束' in message for message in messages))

    def test_stopping_job_cleans_only_its_new_unpublished_staging_directory(self) -> None:
        """停止请求要等 CLI 退出，再依据本次新增台账精确清理 `.pending`。"""
        with TemporaryDirectory() as directory:
            project_root = Path(directory)
            manager = JobManager(project_root, lambda message, *values: None)
            with patch.object(manager, '_interrupt_process', side_effect=lambda process: process.terminate()):
                job = manager.start((sys.executable, '-c', 'import time; time.sleep(30)'), 'preprocess')
                final_dir = project_root / 'data' / '0_preprocess' / 'fixture' / '1#20261107_180000#42'
                staging_dir = final_dir.parent / f'.{final_dir.name}.pending'
                staging_dir.mkdir(parents=True)
                (project_root / 'run_registry.json').write_text(json.dumps({'schema_version': 1, 'runs': [{
                    'dataset': 'fixture', 'module': 'preprocess', 'counter': 1, 'started_at': '2026-11-07T18:00:00+08:00',
                    'artifact_dir': final_dir.relative_to(project_root).as_posix(), 'state': 'running',
                }]}), encoding='utf-8')
                manager.stop(job.job_id)
                deadline = time.monotonic() + 5.0
                while manager.snapshot(job.job_id)['status'] in {'running', 'stopping'} and time.monotonic() < deadline:
                    time.sleep(0.02)
            snapshot = manager.snapshot(job.job_id)
            self.assertEqual(snapshot['status'], 'cancelled')
            self.assertEqual(snapshot['cleaned_artifacts'], 1)
            self.assertFalse(staging_dir.exists())
            self.assertFalse(final_dir.exists())

    def test_history_detail_and_delete_follow_only_manifest_dependency_chain(self) -> None:
        """历史删除只移除目标和显式引用它的下游，不按相同数据集误删独立重跑。"""
        with TemporaryDirectory() as directory:
            project_root = Path(directory)
            first_preprocess = self._write_history_run(project_root, 'preprocess', 'fixture', '1#20261107_180000#42')
            other_preprocess = self._write_history_run(project_root, 'preprocess', 'fixture', '2#20261107_180100#42')
            split = self._write_history_run(project_root, 'split', 'fixture', '1#20261107_180200#42', [first_preprocess / 'manifest.json'])
            inject = self._write_history_run(project_root, 'inject', 'fixture', '1#20261107_180300#42',
                                             [first_preprocess / 'manifest.json', split / 'manifest.json'])
            independent_split = self._write_history_run(project_root, 'split', 'fixture', '2#20261107_180400#42',
                                                        [other_preprocess / 'manifest.json'])
            (project_root / 'logs').mkdir()
            (project_root / 'logs' / '20261107_180000#fixture#preprocess#1_42.log').write_text('history log\n', encoding='utf-8')
            rows = list_module_history(project_root, 'preprocess')
            self.assertEqual(len(rows), 2)
            detail = history_detail(project_root, 'preprocess', first_preprocess.relative_to(project_root).as_posix())
            self.assertEqual(detail['summary'], {'run': '1#20261107_180000#42'})
            self.assertIn('history log', detail['logs'])
            self.assertEqual(len(detail['delete_preview']), 3)
            deleted = delete_history_run(project_root, 'preprocess', first_preprocess.relative_to(project_root).as_posix())
            self.assertEqual(len(deleted), 3)
            self.assertFalse(first_preprocess.exists())
            self.assertFalse(split.exists())
            self.assertFalse(inject.exists())
            self.assertTrue(other_preprocess.exists())
            self.assertTrue(independent_split.exists())

    def test_job_manager_returns_only_the_selected_modules_last_job(self) -> None:
        """任务列表按模块查询；未运行模块不能复用其他模块的输出。"""
        manager = JobManager(ROOT, lambda message, *values: None)
        job = manager.start((sys.executable, '-c', "print('preprocess')"), 'preprocess')
        deadline = time.monotonic() + 5.0
        while manager.snapshot(job.job_id)['status'] == 'running' and time.monotonic() < deadline:
            time.sleep(0.02)
        snapshot = manager.latest_snapshot('preprocess')
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot['job_id'], job.job_id)
        self.assertEqual(snapshot['module_name'], 'preprocess')
        self.assertIsNone(manager.latest_snapshot('segment'))

    def test_latest_job_endpoint_is_scoped_to_the_requested_module(self) -> None:
        """HTTP 路由不能把 preprocess 的任务错误展示在未运行的 segment 页面。"""
        with test_server(ROOT) as server:
            manager = server.cafad_manager  # type: ignore[attr-defined]
            job = manager.start((sys.executable, '-c', "print('preprocess')"), 'preprocess')
            base_url = f'http://127.0.0.1:{server.server_port}'
            cookie = self._login(base_url, TOKEN)
            preprocess = self._get_json(f'{base_url}/api/jobs/latest?module=preprocess', cookie)
            segment = self._get_json(f'{base_url}/api/jobs/latest?module=segment', cookie)
            inject = self._get_json(f'{base_url}/api/jobs/latest?module=inject', cookie)
            self.assertEqual(preprocess['job']['job_id'], job.job_id)
            self.assertEqual(preprocess['job']['module_name'], 'preprocess')
            self.assertIsNone(segment['job'])
            self.assertIsNone(inject['job'])

    def test_schema_requires_login_and_accepts_cookie_after_token_hash_login(self) -> None:
        """明文令牌仅用于登录；后续 schema 请求只凭 HttpOnly 会话 Cookie。"""
        with test_server(ROOT) as server:
            base_url = f'http://127.0.0.1:{server.server_port}'
            with self.assertRaises(HTTPError) as raised:
                urlopen(f'{base_url}/api/schema', timeout=5)
            self.assertEqual(raised.exception.code, HTTPStatus.UNAUTHORIZED)
            raised.exception.close()

            with self.assertRaises(HTTPError) as raised:
                self._login(base_url, 'wrong-token')
            self.assertEqual(raised.exception.code, HTTPStatus.UNAUTHORIZED)
            raised.exception.close()

            cookie = self._login(base_url, TOKEN)
            request = Request(f'{base_url}/api/schema', headers={'Cookie': cookie})
            with urlopen(request, timeout=5) as response:
                schema = json.loads(response.read().decode('utf-8'))
            self.assertIn('split', schema['modules'])
            self.assertEqual(schema['module_order'], [
                'preprocess', 'split', 'inject', 'segment', 'group', 'train', 'calibrate', 'detect', 'evaluate',
            ])
            split_input = next(option for option in schema['modules']['split'] if option['dest'] == 'input')
            self.assertEqual(split_input['upstream_module'], 'preprocess')
            inject_input = next(option for option in schema['modules']['inject'] if option['dest'] == 'input')
            inject_split = next(option for option in schema['modules']['inject'] if option['dest'] == 'split_manifest')
            segment_input = next(option for option in schema['modules']['segment'] if option['dest'] == 'input')
            group_input = next(option for option in schema['modules']['group'] if option['dest'] == 'input')
            train_input = next(option for option in schema['modules']['train'] if option['dest'] == 'input')
            self.assertEqual(inject_input['upstream_module'], 'preprocess')
            self.assertTrue(inject_input['auto_derived'])
            self.assertEqual(inject_split['upstream_module'], 'split')
            self.assertEqual(segment_input['upstream_module'], 'inject')
            self.assertEqual(group_input['upstream_module'], 'segment')
            self.assertEqual(train_input['upstream_module'], 'group')
            self.assertEqual([item['dest'] for item in schema['modules']['train'] if item['upstream_module']], ['input'])

    def test_restart_endpoint_requires_login_and_idle_console(self) -> None:
        """重启请求只在认证且控制台空闲时登记，不在处理线程直接退出进程。"""
        with test_server(ROOT) as server:
            base_url = f'http://127.0.0.1:{server.server_port}'
            with self.assertRaises(HTTPError) as raised:
                urlopen(Request(f'{base_url}/api/server/restart', data=b'', method='POST'), timeout=5)
            self.assertEqual(raised.exception.code, HTTPStatus.UNAUTHORIZED)
            raised.exception.close()
            cookie = self._login(base_url, TOKEN)
            request = Request(f'{base_url}/api/server/restart', data=b'', headers={'Cookie': cookie}, method='POST')
            with self.assertRaises(HTTPError) as raised:
                urlopen(request, timeout=5)
            self.assertEqual(raised.exception.code, HTTPStatus.NOT_IMPLEMENTED)
            raised.exception.close()

        with TemporaryDirectory() as directory:
            project_root = Path(directory)
            manager = JobManager(project_root, lambda message, *values: None)
            restart = RestartController()
            server = ThreadingHTTPServer(('127.0.0.1', 0), create_handler(project_root, manager, TOKEN, restart))
            restart.bind_shutdown(server.shutdown)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f'http://127.0.0.1:{server.server_port}'
                cookie = self._login(base_url, TOKEN)
                job = manager.start((sys.executable, '-c', 'import time; time.sleep(30)'), 'preprocess')
                request = Request(f'{base_url}/api/server/restart', data=b'', headers={'Cookie': cookie}, method='POST')
                with self.assertRaises(HTTPError) as raised:
                    urlopen(request, timeout=5)
                self.assertEqual(raised.exception.code, HTTPStatus.CONFLICT)
                raised.exception.close()
                manager.stop(job.job_id)
                deadline = time.monotonic() + 5.0
                while manager.snapshot(job.job_id)['status'] in {'running', 'stopping'} and time.monotonic() < deadline:
                    time.sleep(0.02)
                with urlopen(request, timeout=5) as response:
                    accepted = json.loads(response.read().decode('utf-8'))
                self.assertTrue(accepted['restarting'])
                self.assertTrue(restart.requested)
            finally:
                if thread.is_alive():
                    server.shutdown()
                thread.join(timeout=5)
                server.server_close()

    def test_upstream_run_endpoint_lists_only_successful_declared_manifest_runs(self) -> None:
        """split 的输入下拉只列出 0_preprocess 下已经发布成功的运行。"""
        with TemporaryDirectory() as directory:
            project_root = Path(directory)
            successful = project_root / 'data' / '0_preprocess' / 'fixture' / '1#20261105_120000#42'
            successful.mkdir(parents=True)
            (successful / 'manifest.json').write_text(json.dumps({
                'status': 'SUCCEEDED', 'module': 'preprocess', 'outputs': [], 'summary': {},
            }), encoding='utf-8')
            ignored = project_root / 'data' / '0_preprocess' / 'fixture' / '.2#20261105_120001#42.pending'
            ignored.mkdir()
            (ignored / 'manifest.json').write_text(json.dumps({
                'status': 'SUCCEEDED', 'module': 'preprocess', 'outputs': [], 'summary': {},
            }), encoding='utf-8')
            with test_server(project_root) as server:
                base_url = f'http://127.0.0.1:{server.server_port}'
                cookie = self._login(base_url, TOKEN)
                result = self._get_json(f'{base_url}/api/upstream-runs?module=split&argument=input', cookie)
            self.assertEqual(result['runs'], [{
                'dataset': 'fixture', 'run_name': '1#20261105_120000#42',
                'label': 'fixture / 1#20261105_120000#42',
                'manifest': 'data/0_preprocess/fixture/1#20261105_120000#42/manifest.json',
            }])
            split_run = project_root / 'data' / '1_split' / 'fixture' / '1#20261105_120010#42'
            split_run.mkdir(parents=True)
            (split_run / 'manifest.json').write_text(json.dumps({
                'status': 'SUCCEEDED', 'module': 'split', 'outputs': [], 'summary': {},
            }), encoding='utf-8')
            with test_server(project_root) as server:
                base_url = f'http://127.0.0.1:{server.server_port}'
                cookie = self._login(base_url, TOKEN)
                inject_preprocess = self._get_json(
                    f'{base_url}/api/upstream-runs?module=inject&argument=input', cookie)
                inject_split = self._get_json(
                    f'{base_url}/api/upstream-runs?module=inject&argument=split_manifest', cookie)
            self.assertEqual(inject_preprocess['runs'][0]['manifest'],
                             'data/0_preprocess/fixture/1#20261105_120000#42/manifest.json')
            self.assertEqual(inject_split['runs'], [{
                'dataset': 'fixture', 'run_name': '1#20261105_120010#42',
                'label': 'fixture / 1#20261105_120010#42',
                'manifest': 'data/1_split/fixture/1#20261105_120010#42/manifest.json',
            }])

    def test_file_browser_lists_project_items_and_rejects_parent_escape(self) -> None:
        """文件选择器只允许 home/project 根中的目录项，不能以 .. 越界读取。"""
        with TemporaryDirectory() as directory:
            project_root = Path(directory)
            inputs = project_root / 'inputs'
            inputs.mkdir()
            (inputs / 'flight.csv').write_text('fixture', encoding='utf-8')
            with test_server(project_root) as server:
                base_url = f'http://127.0.0.1:{server.server_port}'
                cookie = self._login(base_url, TOKEN)
                request = Request(f'{base_url}/api/files?root=project&path=inputs', headers={'Cookie': cookie})
                with urlopen(request, timeout=5) as response:
                    listing = json.loads(response.read().decode('utf-8'))
                self.assertEqual(listing['entries'][0]['name'], 'flight.csv')
                self.assertEqual(listing['entries'][0]['kind'], 'file')

                rejected = Request(f'{base_url}/api/files?root=project&path=..%2F..', headers={'Cookie': cookie})
                with self.assertRaises(HTTPError) as raised:
                    urlopen(rejected, timeout=5)
                self.assertEqual(raised.exception.code, HTTPStatus.BAD_REQUEST)
                raised.exception.close()

    def test_form_state_persists_within_one_session_but_not_another(self) -> None:
        """全局和模块配置绑定随机会话 Cookie，不能串到另一登录会话。"""
        with test_server(ROOT) as server:
            base_url = f'http://127.0.0.1:{server.server_port}'
            first_cookie = self._login(base_url, TOKEN)
            state = {
                'global': {'seed': '10', 'console': True},
                'modules': {'split': {'input': '/tmp/preprocess/manifest.json', 'ratios': ['70', '15', '15']}},
                'selected_module': 'split',
            }
            payload = json.dumps({'state': state}).encode('utf-8')
            save_request = Request(f'{base_url}/api/form-state', data=payload,
                                   headers={'Content-Type': 'application/json', 'Cookie': first_cookie}, method='PUT')
            with urlopen(save_request, timeout=5) as response:
                self.assertEqual(response.status, HTTPStatus.OK)

            loaded = self._load_state(base_url, first_cookie)
            self.assertEqual(loaded, state)
            second_cookie = self._login(base_url, TOKEN)
            self.assertEqual(self._load_state(base_url, second_cookie), {})

    @staticmethod
    def _write_history_run(project_root: Path, module_name: str, dataset: str, run_name: str,
                           dependencies: list[Path] | None = None) -> Path:
        """生成最小已发布运行，用于验证历史与删除闭包而不依赖业务大文件。"""
        counter, timestamp, seed = run_name.split('#')
        run_dir = project_root / 'data' / f'{["preprocess", "split", "inject"].index(module_name)}_{module_name}' / dataset / run_name
        run_dir.mkdir(parents=True)
        (run_dir / 'resolved_config.json').write_text(json.dumps({
            'global': {'seed': int(seed)}, 'module_name': module_name, 'dataset_id': dataset,
            'dataset_name': dataset, 'run_id': counter, 'run_timestamp': timestamp,
            'module': {'input': str(dependencies[0]) if dependencies else ''},
        }), encoding='utf-8')
        (run_dir / 'summary.json').write_text(json.dumps({'run': run_name}), encoding='utf-8')
        (run_dir / 'manifest.json').write_text(json.dumps({
            'status': 'SUCCEEDED', 'module': module_name, 'dataset_id': dataset,
            'dependencies': [{'manifest': str(path), 'sha256': 'fixture'} for path in dependencies or []], 'outputs': [],
        }), encoding='utf-8')
        return run_dir

    @staticmethod
    def _login(base_url: str, token: str) -> str:
        """发起登录并抽取响应 Cookie；测试中不把令牌放入后续 API 请求。"""
        payload = json.dumps({'token': token}).encode('utf-8')
        request = Request(f'{base_url}/api/auth/login', data=payload,
                          headers={'Content-Type': 'application/json'}, method='POST')
        with urlopen(request, timeout=5) as response:
            cookie = response.headers['Set-Cookie']
        return cookie.split(';', 1)[0]

    @staticmethod
    def _load_state(base_url: str, cookie: str) -> dict[str, object]:
        """读取某一会话的暂存表单配置。"""
        request = Request(f'{base_url}/api/form-state', headers={'Cookie': cookie})
        with urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode('utf-8'))['state']

    @staticmethod
    def _get_json(url: str, cookie: str) -> dict[str, object]:
        """以已登录会话读取一个 JSON API 响应。"""
        request = Request(url, headers={'Cookie': cookie})
        with urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode('utf-8'))


if __name__ == '__main__':
    unittest.main()
