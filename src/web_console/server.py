"""server.py | 2026-11-04 | 公开监听的 CAFAD Web 控制台 HTTP 服务 | Codex."""
from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import sqlite3
import sys
from time import sleep
from pathlib import Path
from typing import Mapping
from urllib.parse import parse_qs, urlparse

import duckdb

from src.utils.env import Env
from src.utils.logging_utils import close_project_logging, setup_project_logging
from src.utils.strings import PathUtils
from .auth import SESSION_TTL_SECONDS, SessionAuthenticator
from .file_browser import FileBrowserError, browser_roots, list_directory
from .history import delete_history_run, history_detail, list_module_history
from .restart import RestartController
from .train_monitor import training_progress
from .train_util import TrainTaskQueue
from .web_util import ModuleHistoryUtil
from .group_view import group_result
from .segment_view import segment_result
from .segment_plot import plotly_javascript, segment_flight_html, segment_flight_svg
from .service import JobManager, build_command, module_catalog, module_schemas, serialize_schemas, upstream_manifest_choices


STATIC_DIR = Path(__file__).resolve().parent / 'static'
MAX_REQUEST_BYTES = 64 * 1024
MAX_FORM_STATE_BYTES = 48 * 1024
RESTART_DELAY_SECONDS = 5


def build_parser() -> argparse.ArgumentParser:
    """声明控制台服务参数；默认监听所有网络接口。"""
    parser = argparse.ArgumentParser(description='CAFAD Web 控制台服务')
    parser.add_argument('--host', default='0.0.0.0', help='监听地址；默认允许局域网和反向代理访问')
    return parser


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, object]:
    """读取有大小上限的 JSON 请求体，拒绝不合法对象根节点。"""
    raw_length = handler.headers.get('Content-Length')
    if raw_length is None:
        raise ValueError('请求缺少 Content-Length')
    try:
        length = int(raw_length)
    except ValueError as exc:
        raise ValueError('Content-Length 无效') from exc
    if not 0 <= length <= MAX_REQUEST_BYTES:
        raise ValueError('请求体大小超出限制')
    try:
        value = json.loads(handler.rfile.read(length).decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError('请求体不是有效 JSON') from exc
    if not isinstance(value, dict):
        raise ValueError('请求根节点必须是对象')
    return value


def _json_response(
    handler: BaseHTTPRequestHandler,
    status: HTTPStatus,
    value: dict[str, object],
    headers: Mapping[str, str] | None = None,
) -> None:
    """输出不缓存的 UTF-8 JSON 响应，并按需附加 Cookie 等响应头。"""
    data = json.dumps(value, ensure_ascii=False).encode('utf-8')
    handler.send_response(status)
    handler.send_header('Content-Type', 'application/json; charset=utf-8')
    handler.send_header('Content-Length', str(len(data)))
    handler.send_header('Cache-Control', 'no-store')
    if headers is not None:
        for name, header_value in headers.items():
            handler.send_header(name, header_value)
    handler.end_headers()
    handler.wfile.write(data)


def _static_response(handler: BaseHTTPRequestHandler, relative_path: str, content_type: str) -> None:
    """仅返回预定义静态文件，避免 URL 映射为任意文件读取。"""
    path = STATIC_DIR / relative_path
    data = path.read_bytes()
    handler.send_response(HTTPStatus.OK)
    handler.send_header('Content-Type', content_type)
    handler.send_header('Content-Length', str(len(data)))
    handler.send_header('Cache-Control', 'no-store')
    handler.end_headers()
    handler.wfile.write(data)


def _svg_response(handler: BaseHTTPRequestHandler, value: bytes) -> None:
    """返回不可缓存的阶段着色 SVG，供已登录控制台的单航班预览直接嵌入。"""
    handler.send_response(HTTPStatus.OK)
    handler.send_header('Content-Type', 'image/svg+xml; charset=utf-8')
    handler.send_header('Content-Length', str(len(value)))
    handler.send_header('Cache-Control', 'no-store')
    handler.end_headers()
    handler.wfile.write(value)


def _html_response(handler: BaseHTTPRequestHandler, value: bytes) -> None:
    """返回已缓存的交互式轨迹 HTML；只来自受限的 segment 运行目录。"""
    handler.send_response(HTTPStatus.OK)
    handler.send_header('Content-Type', 'text/html; charset=utf-8')
    handler.send_header('Content-Length', str(len(value)))
    handler.send_header('Cache-Control', 'no-store')
    handler.end_headers()
    handler.wfile.write(value)


def _javascript_response(handler: BaseHTTPRequestHandler, value: bytes) -> None:
    """返回交互式预览共用的本机 Plotly 脚本，避免访问外部 CDN。"""
    handler.send_response(HTTPStatus.OK)
    handler.send_header('Content-Type', 'text/javascript; charset=utf-8')
    handler.send_header('Content-Length', str(len(value)))
    handler.send_header('Cache-Control', 'public, max-age=86400')
    handler.end_headers()
    handler.wfile.write(value)


def create_handler(
    project_root: Path,
    manager: JobManager,
    access_token: str,
    restart: RestartController | None = None,
    train_queue: TrainTaskQueue | None = None,
) -> type[BaseHTTPRequestHandler]:
    """创建绑定项目、会话认证、受限资源浏览和任务管理器的 HTTP Handler。"""
    schemas = module_schemas()
    schema_json = serialize_schemas(schemas)
    catalog = module_catalog()
    authenticator = SessionAuthenticator(access_token)
    roots = browser_roots(project_root)
    history_cache = {module: ModuleHistoryUtil.query(project_root, module) for module in catalog}

    class ConsoleHandler(BaseHTTPRequestHandler):
        """先登录，再提供表单 schema、文件选择、任务启动和日志读取。"""

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == '/':
                _static_response(self, 'index.html', 'text/html; charset=utf-8')
                return
            if parsed.path == '/static/app.js':
                _static_response(self, 'app.js', 'text/javascript; charset=utf-8')
                return
            if parsed.path == '/static/group_view.js':
                _static_response(self, 'group_view.js', 'text/javascript; charset=utf-8')
                return
            if parsed.path == '/static/segment_view.js':
                _static_response(self, 'segment_view.js', 'text/javascript; charset=utf-8')
                return
            if parsed.path == '/static/style.css':
                _static_response(self, 'style.css', 'text/css; charset=utf-8')
                return
            if parsed.path == '/api/auth/status':
                _json_response(self, HTTPStatus.OK, {'authenticated': self._authorized_response(False)})
                return
            if not self._authorized_response(True):
                return
            if parsed.path == '/api/schema':
                _json_response(self, HTTPStatus.OK, {'modules': schema_json, 'module_order': list(catalog)})
                return
            if parsed.path == '/api/form-state':
                self._form_state_get()
                return
            if parsed.path == '/api/files':
                self._files_get(parse_qs(parsed.query))
                return
            if parsed.path == '/api/upstream-runs':
                self._upstream_runs_get(parse_qs(parsed.query))
                return
            if parsed.path == '/api/history':
                self._history_get(parse_qs(parsed.query))
                return
            if parsed.path == '/api/history/state':
                module = parse_qs(parsed.query).get('module', [''])[0]
                if module not in catalog:
                    _json_response(self, HTTPStatus.BAD_REQUEST, {'error': '模块不存在'})
                    return
                latest = manager.latest_snapshot(module)
                active = latest is not None and latest['status'] in {'running', 'stopping'}
                if module == 'train' and train_queue is not None:
                    active = active or train_queue.has_pending()
                _json_response(self, HTTPStatus.OK, {'running': active or any(
                    row['state'] == 'running' for row in list_module_history(project_root, module))})
                return
            if parsed.path == '/api/train/progress':
                try:
                    artifact = parse_qs(parsed.query).get('artifact', [None])[0]
                    _json_response(self, HTTPStatus.OK, training_progress(project_root, artifact))
                except (ValueError, OSError) as exc:
                    _json_response(self, HTTPStatus.BAD_REQUEST, {'error': str(exc)})
                return
            if parsed.path == '/api/train/queue':
                if train_queue is None:
                    _json_response(self, HTTPStatus.NOT_IMPLEMENTED, {'error': '当前服务未启用训练队列'})
                else:
                    _json_response(self, HTTPStatus.OK, train_queue.snapshot())
                return
            if parsed.path == '/api/history/detail':
                self._history_detail_get(parse_qs(parsed.query))
                return
            if parsed.path == '/api/group/result':
                try:
                    artifact = parse_qs(parsed.query).get('artifact', [''])[0]
                    _json_response(self, HTTPStatus.OK, group_result(project_root, artifact))
                except (ValueError, OSError, KeyError) as exc:
                    _json_response(self, HTTPStatus.BAD_REQUEST, {'error': str(exc)})
                return
            if parsed.path == '/api/segment/result':
                try:
                    artifact = parse_qs(parsed.query).get('artifact', [''])[0]
                    _json_response(self, HTTPStatus.OK, segment_result(project_root, artifact))
                except (ValueError, OSError, RuntimeError, sqlite3.Error) as exc:
                    _json_response(self, HTTPStatus.BAD_REQUEST, {'error': str(exc)})
                return
            if parsed.path == '/api/segment/flight-plot':
                try:
                    query = parse_qs(parsed.query)
                    _svg_response(self, segment_flight_svg(
                        project_root, query.get('artifact', [''])[0], query.get('flight_id', [''])[0],
                    ))
                except (ValueError, OSError, RuntimeError, duckdb.Error) as exc:
                    _json_response(self, HTTPStatus.BAD_REQUEST, {'error': str(exc)})
                return
            if parsed.path == '/api/segment/flight-plot-interactive':
                try:
                    query = parse_qs(parsed.query)
                    _html_response(self, segment_flight_html(
                        project_root, query.get('artifact', [''])[0], query.get('flight_id', [''])[0],
                    ))
                except (ValueError, OSError, RuntimeError, duckdb.Error) as exc:
                    _json_response(self, HTTPStatus.BAD_REQUEST, {'error': str(exc)})
                return
            if parsed.path == '/api/segment/plotly.min.js':
                _javascript_response(self, plotly_javascript())
                return
            if parsed.path == '/api/jobs/latest':
                self._latest_job_get(parse_qs(parsed.query))
                return
            if parsed.path.startswith('/api/jobs/'):
                self._job_get(parsed.path, parse_qs(parsed.query))
                return
            _json_response(self, HTTPStatus.NOT_FOUND, {'error': '路由不存在'})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path in ('/api/search/random', '/api/search/tpe'):
                if not self._authorized_response(True):
                    return
                try:
                    mode = path.rsplit('/', 1)[1]
                    command = (sys.executable, str(project_root / 'scripts/parameter_search.py'), '--mode', mode)
                    job = manager.start(command, 'search_' + mode)
                    _json_response(self, HTTPStatus.ACCEPTED, manager.snapshot(job.job_id))
                except (ValueError, RuntimeError) as exc:
                    _json_response(self, HTTPStatus.BAD_REQUEST, {'error': str(exc)})
                return
            if path == '/api/auth/login':
                self._login()
                return
            if path == '/api/auth/logout':
                self._logout()
                return
            if path == '/api/server/restart':
                self._restart()
                return
            if path.startswith('/api/jobs/') and path.endswith('/stop'):
                if not self._authorized_response(True):
                    return
                self._job_stop(path)
                return
            if path == '/api/train/queue':
                if not self._authorized_response(True):
                    return
                if train_queue is None:
                    _json_response(self, HTTPStatus.NOT_IMPLEMENTED, {'error': '当前服务未启用训练队列'})
                    return
                try:
                    payload = _read_json(self)
                    if payload.get('module') != 'train':
                        raise ValueError('训练队列只接受 train 模块')
                    auto_go = payload.pop('auto_go', False)
                    if not isinstance(auto_go, bool):
                        raise ValueError('auto_go 必须为布尔值')
                    command = build_command(project_root, payload, schemas)
                    entry = ModuleHistoryUtil.from_command(command)
                    result = train_queue.enqueue_train(command, auto_go)
                    result['history_entry'] = entry
                    _json_response(self, HTTPStatus.ACCEPTED, result)
                except (ValueError, RuntimeError) as exc:
                    _json_response(self, HTTPStatus.BAD_REQUEST, {'error': str(exc)})
                return
            if path != '/api/jobs':
                _json_response(self, HTTPStatus.NOT_FOUND, {'error': '路由不存在'})
                return
            if not self._authorized_response(True):
                return
            try:
                payload = _read_json(self)
                command = build_command(project_root, payload, schemas)
                module_name = payload['module']
                assert isinstance(module_name, str)  # build_command 已验证模块名。
                entry = ModuleHistoryUtil.from_command(command)
                job = manager.start(command, module_name)
                snapshot = manager.snapshot(job.job_id)
                snapshot['history_entry'] = entry
                _json_response(self, HTTPStatus.ACCEPTED, snapshot)
            except (ValueError, RuntimeError) as exc:
                _json_response(self, HTTPStatus.BAD_REQUEST, {'error': str(exc)})

        def do_DELETE(self) -> None:
            """删除一条历史运行以及由它显式派生的下游目录。"""
            parsed = urlparse(self.path)
            if parsed.path != '/api/history':
                _json_response(self, HTTPStatus.NOT_FOUND, {'error': '路由不存在'})
                return
            if not self._authorized_response(True):
                return
            self._history_delete(parse_qs(parsed.query))

        def do_PUT(self) -> None:
            """保存当前会话的非敏感表单配置；令牌从不属于该状态。"""
            if urlparse(self.path).path != '/api/form-state':
                _json_response(self, HTTPStatus.NOT_FOUND, {'error': '路由不存在'})
                return
            if not self._authorized_response(True):
                return
            try:
                state = _form_state_from_payload(_read_json(self))
                if not authenticator.save_form_state(self.headers.get('Cookie'), state):
                    _json_response(self, HTTPStatus.UNAUTHORIZED, {'error': '尚未登录或会话已过期'})
                    return
                _json_response(self, HTTPStatus.OK, {'saved': True})
            except ValueError as exc:
                _json_response(self, HTTPStatus.BAD_REQUEST, {'error': str(exc)})

        def _login(self) -> None:
            """以令牌摘要认证；成功后设置 JavaScript 不可读的随机 Cookie。"""
            try:
                payload = _read_json(self)
                token = payload.get('token')
                if not isinstance(token, str):
                    raise ValueError('访问令牌无效')
                session = authenticator.issue_session(token)
            except ValueError:
                session = None
            if session is None:
                _json_response(self, HTTPStatus.UNAUTHORIZED, {'error': '访问令牌无效'})
                return
            _json_response(
                self,
                HTTPStatus.OK,
                {'authenticated': True, 'expires_in_seconds': SESSION_TTL_SECONDS},
                {'Set-Cookie': SessionAuthenticator.make_set_cookie(session, SESSION_TTL_SECONDS)},
            )

        def _logout(self) -> None:
            """注销当前会话并通知浏览器立即丢弃 Cookie。"""
            if not self._authorized_response(True):
                return
            authenticator.revoke_session(self.headers.get('Cookie'))
            _json_response(
                self,
                HTTPStatus.OK,
                {'authenticated': False},
                {'Set-Cookie': SessionAuthenticator.make_set_cookie(None, 0)},
            )

        def _restart(self) -> None:
            """仅空闲控制台可自行重启，避免丢失仍在运行任务的停止能力。"""
            if not self._authorized_response(True):
                return
            if restart is None:
                _json_response(self, HTTPStatus.NOT_IMPLEMENTED, {'error': '当前服务未启用网页重启'})
                return
            if manager.has_active_job() or (train_queue is not None and train_queue.has_pending()):
                _json_response(self, HTTPStatus.CONFLICT, {'error': '当前仍有网页任务运行、停止中或排队，不能重启控制台'})
                return
            if not restart.request():
                _json_response(self, HTTPStatus.CONFLICT, {'error': '控制台重启已在进行中'})
                return
            _json_response(self, HTTPStatus.ACCEPTED, {
                'restarting': True,
                'delay_seconds': RESTART_DELAY_SECONDS,
                'message': f'控制台将在 {RESTART_DELAY_SECONDS} 秒后以当前磁盘代码重启；需重新登录。',
            })

        def _authorized_response(self, send_error: bool) -> bool:
            """检查短期会话；未登录请求不能读取 schema、资源或任务信息。"""
            authenticated = authenticator.is_session_valid(self.headers.get('Cookie'))
            if not authenticated and send_error:
                _json_response(self, HTTPStatus.UNAUTHORIZED, {'error': '尚未登录或会话已过期'})
            return authenticated

        def _files_get(self, query: dict[str, list[str]]) -> None:
            """列出当前用户目录或项目目录下允许访问的文件系统项。"""
            root_name = query.get('root', ['home'])[0]
            relative_path = query.get('path', [''])[0]
            try:
                _json_response(self, HTTPStatus.OK, list_directory(roots, root_name, relative_path))
            except FileBrowserError as exc:
                _json_response(self, HTTPStatus.BAD_REQUEST, {'error': str(exc)})

        def _upstream_runs_get(self, query: dict[str, list[str]]) -> None:
            """返回当前模块一个已声明上游输入可选的成功运行目录。"""
            module_name = query.get('module', [''])[0]
            argument_name = query.get('argument', [''])[0]
            if module_name not in schemas:
                _json_response(self, HTTPStatus.BAD_REQUEST, {'error': '模块不存在或尚未实现'})
                return
            try:
                choices = upstream_manifest_choices(project_root, module_name, argument_name)
            except ValueError as exc:
                _json_response(self, HTTPStatus.BAD_REQUEST, {'error': str(exc)})
                return
            _json_response(self, HTTPStatus.OK, {'runs': choices})

        def _history_get(self, query: dict[str, list[str]]) -> None:
            """读取一个模块的磁盘历史；不同页面不会再共享内存中的旧控制台输出。"""
            module_name = query.get('module', [''])[0]
            if module_name not in catalog:
                _json_response(self, HTTPStatus.BAD_REQUEST, {'error': '模块不存在'})
                return
            latest = manager.latest_snapshot(module_name)
            active = latest is not None and latest['status'] in {'running', 'stopping'}
            if module_name == 'train' and train_queue is not None:
                active = active or train_queue.has_pending()
            if query.get('refresh') == ['1'] and active:
                _json_response(self, HTTPStatus.CONFLICT, {'error': '当前模块仍在运行，完成后才能刷新历史'})
                return
            history_cache[module_name] = ModuleHistoryUtil.query(project_root, module_name)
            response = dict(history_cache[module_name])
            response['running'] = active or response['running']
            if query.get('refresh') == ['1'] and response['running']:
                _json_response(self, HTTPStatus.CONFLICT, {'error': '当前模块有未结束产物，完成后才能刷新历史'})
                return
            _json_response(self, HTTPStatus.OK, response)

        def _history_detail_get(self, query: dict[str, list[str]]) -> None:
            """展开一条历史运行的摘要、失败信息和日志尾部。"""
            module_name = query.get('module', [''])[0]
            artifact = query.get('artifact', [''])[0]
            if module_name not in catalog:
                _json_response(self, HTTPStatus.BAD_REQUEST, {'error': '模块不存在'})
                return
            try:
                _json_response(self, HTTPStatus.OK, history_detail(project_root, module_name, artifact))
            except ValueError as exc:
                _json_response(self, HTTPStatus.BAD_REQUEST, {'error': str(exc)})

        def _history_delete(self, query: dict[str, list[str]]) -> None:
            """删除前禁止并行正式运行，防止删除器与发布器争用相同目录。"""
            module_name = query.get('module', [''])[0]
            artifact = query.get('artifact', [''])[0]
            if module_name not in catalog:
                _json_response(self, HTTPStatus.BAD_REQUEST, {'error': '模块不存在'})
                return
            if manager.has_active_job() or (train_queue is not None and train_queue.has_pending()):
                _json_response(self, HTTPStatus.CONFLICT, {'error': '当前有网页任务在运行或停止中，暂不能删除历史产物'})
                return
            try:
                _json_response(self, HTTPStatus.OK, {'deleted': delete_history_run(project_root, module_name, artifact)})
            except ValueError as exc:
                _json_response(self, HTTPStatus.BAD_REQUEST, {'error': str(exc)})

        def _form_state_get(self) -> None:
            """读取当前登录会话的表单状态，供刷新与导航后恢复。"""
            state = authenticator.load_form_state(self.headers.get('Cookie'))
            if state is None:
                _json_response(self, HTTPStatus.UNAUTHORIZED, {'error': '尚未登录或会话已过期'})
                return
            _json_response(self, HTTPStatus.OK, {'state': state})

        def _latest_job_get(self, query: dict[str, list[str]]) -> None:
            """返回当前模块最后一次网页任务；没有记录时不伪造空任务。"""
            module_name = query.get('module', [''])[0]
            if module_name not in (*catalog, 'search_random', 'search_tpe'):
                _json_response(self, HTTPStatus.BAD_REQUEST, {'error': '模块不存在'})
                return
            job = manager.latest_snapshot(module_name)
            # 搜索持有整个四步流水线的进程组；必须停止父任务，避免又启动下一轮。
            if module_name == 'train':
                if train_queue is not None:
                    active_job_id = train_queue.active_job_id()
                    if active_job_id is not None:
                        job = manager.snapshot(active_job_id)
                for search_module in ('search_random', 'search_tpe'):
                    search_job = manager.latest_snapshot(search_module)
                    if search_job is not None and search_job['status'] in {'running', 'stopping'}:
                        job = search_job
                        break
            response = {'job': job}
            if job is None and module_name.startswith('search_'):
                files = sorted((project_root / 'logs').glob(f"search-{module_name[7:]}-*.log"))
                if files:
                    with files[-1].open('rb') as reader:
                        reader.seek(max(0, files[-1].stat().st_size - 96000))
                        response['logs'] = reader.read().decode('utf-8', errors='replace')
            _json_response(self, HTTPStatus.OK, response)

        def _job_get(self, path: str, query: dict[str, list[str]]) -> None:
            """读取受限任务的快照或自指定偏移起的日志。"""
            pieces = path.strip('/').split('/')
            if len(pieces) not in {3, 4} or pieces[:2] != ['api', 'jobs']:
                _json_response(self, HTTPStatus.NOT_FOUND, {'error': '路由不存在'})
                return
            job_id = pieces[2]
            try:
                if len(pieces) == 3:
                    _json_response(self, HTTPStatus.OK, manager.snapshot(job_id))
                    return
                if pieces[3] != 'logs':
                    _json_response(self, HTTPStatus.NOT_FOUND, {'error': '路由不存在'})
                    return
                after_text = query.get('after', ['0'])[0]
                _json_response(self, HTTPStatus.OK, manager.logs_after(job_id, int(after_text)))
            except (KeyError, ValueError) as exc:
                _json_response(self, HTTPStatus.BAD_REQUEST, {'error': str(exc)})

        def _job_stop(self, path: str) -> None:
            """处理 `/api/jobs/<id>/stop`，保持其他任务路由不带副作用。"""
            pieces = path.strip('/').split('/')
            if len(pieces) != 4 or pieces[:2] != ['api', 'jobs'] or pieces[3] != 'stop':
                _json_response(self, HTTPStatus.NOT_FOUND, {'error': '路由不存在'})
                return
            try:
                _json_response(self, HTTPStatus.ACCEPTED, manager.stop(pieces[2]))
            except (KeyError, ValueError) as exc:
                _json_response(self, HTTPStatus.BAD_REQUEST, {'error': str(exc)})

        def log_message(self, format: str, *args: object) -> None:
            """HTTP 访问记录不写 stderr，避免与项目运行日志混淆。"""
            return

    return ConsoleHandler


def _form_state_from_payload(payload: dict[str, object]) -> dict[str, object]:
    """校验浏览器提交的表单状态为有限大小的标准 JSON 对象。"""
    state = payload.get('state')
    if not isinstance(state, dict):
        raise ValueError('表单状态必须是对象')
    try:
        encoded = json.dumps(state, ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise ValueError('表单状态必须只包含标准 JSON 数据') from exc
    if len(encoded) > MAX_FORM_STATE_BYTES:
        raise ValueError('表单状态超过 48 KiB 限制')
    return state


def run_server(host: str) -> None:
    """创建会话登录保护的控制台服务并在前台持续运行。"""
    project_root = PathUtils.get_project_root()
    logger, log_path = setup_project_logging(project_root=project_root, name='cafad.web_console')
    restart_requested = False
    try:
        Env.load(project_root / '.env')
        access_token = Env.get('web_console_token')
        if access_token is None or not access_token.strip():
            raise ValueError('公开 Web 控制台需要在根目录 .env 设置非空 web_console_token')
        port = Env.get_int('web_console_port')
        if not 1 <= port <= 65535:
            raise ValueError('web_console_port 必须在 1 到 65535 之间')
        manager = JobManager(project_root, logger.info)
        restart = RestartController()
        train_queue = TrainTaskQueue(manager)
        server = ThreadingHTTPServer((host, port), create_handler(
            project_root, manager, access_token, restart, train_queue,
        ))
        restart.bind_shutdown(server.shutdown)
        logger.info('Web 控制台启动：http://%s:%d；使用令牌登录和短期会话；日志=%s', host, port, log_path)
        server.serve_forever()
        restart_requested = restart.requested
    except Exception:
        logger.exception('Web 控制台启动或运行失败')
        raise
    finally:
        if 'server' in locals():
            server.server_close()
        close_project_logging(logger)
    if restart_requested:
        sleep(RESTART_DELAY_SECONDS)
        os.execv(sys.executable, (sys.executable, *sys.argv))


def main(argv: list[str] | None = None) -> int:
    """解析服务参数并启动控制台。"""
    args = build_parser().parse_args(argv)
    try:
        run_server(args.host)
    except KeyboardInterrupt:
        return 130
    except Exception:
        # run_server 已将启动异常写入统一项目日志；入口不重复打印堆栈。
        return 1
    return 0
