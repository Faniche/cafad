"""local_server.py | 2026-11-11 | 仅本机使用的分组轨迹三维图服务 | Codex."""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass, field
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import logging
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse

import pandas as pd

from src.utils.logging_utils import close_project_logging, setup_project_logging
from src.utils.strings import PathUtils
from src.utils.env import Env
from .local_plot import GroupIndex, load_group_index, render_plot, segment_choices


@dataclass
class LocalPlotState:
    """单用户本机服务的当前选择和一次已生成图像；不写入实验产物。"""

    dataset_path: Path | None = None
    metadata_path: Path | None = None
    group_id: str | None = None
    segment_id: str | None = None
    whole_flight: bool = False
    index: GroupIndex | None = None
    image_data_uri: str | None = None
    summary: dict[str, object] = field(default_factory=dict)
    message: str = ''


def build_parser() -> argparse.ArgumentParser:
    """定义本地绘图服务地址；默认只监听回环接口。"""
    parser = argparse.ArgumentParser(description='CAFAD 本地分组轨迹三维绘图工具')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8766)
    return parser


def _pick_csv() -> Path | None:
    """在本机弹出文件选择器；失败时保留页面的手动路径输入方式。"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        chosen = filedialog.askopenfilename(title='选择 CAFAD CSV 或 manifest.json',
                                             filetypes=(('CSV/JSON', '*.csv *.json'), ('All files', '*.*')))
        root.destroy()
        return Path(chosen).resolve() if chosen else None
    except Exception:
        return None


def create_handler(state: LocalPlotState, logger: logging.Logger) -> type[BaseHTTPRequestHandler]:
    """创建共享单用户状态的 HTTP handler；所有绘图均发生在后端。"""
    class LocalHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if urlparse(self.path).path != '/':
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._page()

        def do_POST(self) -> None:
            route = urlparse(self.path).path
            values = self._form_values()
            if route == '/choose':
                field = values.get('field', '')
                selected = _pick_csv()
                if selected is None:
                    state.message = '未选择文件；也可直接在输入框粘贴本地路径。'
                elif field == 'dataset':
                    state.dataset_path, state.index, state.image_data_uri = selected, None, None
                    state.message = f'已选择数据集：{selected}'
                elif field == 'metadata':
                    state.metadata_path, state.index, state.image_data_uri = selected, None, None
                    state.message = f'已选择分组元数据：{selected}'
                else:
                    state.message = '未知的文件选择字段。'
                self._redirect()
                return
            if route == '/plot':
                self._plot(values)
                self._redirect()
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def _form_values(self) -> dict[str, str]:
            length = int(self.headers.get('Content-Length', '0'))
            raw = self.rfile.read(min(length, 16 * 1024)).decode('utf-8', errors='replace')
            return {name: rows[-1] for name, rows in parse_qs(raw).items() if rows}

        def _plot(self, values: dict[str, str]) -> None:
            try:
                dataset = Path(values.get('dataset_path', '')).expanduser().resolve()
                metadata = Path(values.get('metadata_path', '')).expanduser().resolve()
                if not dataset.is_file() or not metadata.is_file():
                    raise ValueError('请先选择存在的数据集 CSV 与分组元数据文件。')
                if state.index is None or dataset != state.dataset_path or metadata != state.metadata_path:
                    state.index = load_group_index(dataset, metadata)
                state.dataset_path, state.metadata_path = dataset, metadata
                state.group_id = values.get('group_id') or None
                options = set(segment_choices(state.index, state.group_id))
                state.segment_id = values.get('segment_id') if values.get('segment_id') in options else None
                state.whole_flight = values.get('whole_flight') == '1'
                workers = Env.local_plot_workers()
                png, state.summary = render_plot(dataset, state.index, state.group_id, state.segment_id,
                                                  state.whole_flight, workers)
                state.image_data_uri = 'data:image/png;base64,' + base64.b64encode(png).decode('ascii')
                state.message = '绘图完成。图像由 Python/Matplotlib 后端生成。'
                logger.info('本地绘图完成：groups=%s；segments=%s；flights=%s；mode=%s',
                            state.summary['group_count'], state.summary['segment_count'],
                            state.summary['flight_count'], state.summary['mode'])
            except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
                state.image_data_uri = None
                state.message = f'绘图失败：{exc}'
                logger.warning('%s', state.message)

        def _redirect(self) -> None:
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header('Location', '/')
            self.end_headers()

        def _page(self) -> None:
            body = _page_html(state)
            payload = body.encode('utf-8')
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            logger.debug('HTTP %s', format % args)
    return LocalHandler


def _page_html(state: LocalPlotState) -> str:
    """纯 HTML 表单布局：左导航、三栏内容和后端渲染的 PNG。"""
    groups = state.index.group_ids if state.index is not None else ()
    segments = segment_choices(state.index, state.group_id) if state.index is not None else ()
    group_options = _options(groups, state.group_id, '全部分组')
    segment_options = _options(segments, state.segment_id, '全部片段')
    checked = ' checked' if state.whole_flight else ''
    image = f'<img src="{state.image_data_uri}" alt="三维轨迹图">' if state.image_data_uri else '<p>尚未绘图。</p>'
    summary = '<br>'.join(f'{escape(str(key))}: {escape(str(value))}' for key, value in state.summary.items())
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>CAFAD 本地绘图</title></head>
<body><table width="100%"><tr><td width="150" valign="top"><nav><strong>功能</strong><ul><li><a href="/">画图</a></li></ul></nav></td>
<td valign="top"><h1>分组轨迹三维图</h1><p>{escape(state.message)}</p>
<form method="post" action="/choose"><input name="field" type="hidden" value="dataset"><button>选择数据集文件</button></form>
<form method="post" action="/choose"><input name="field" type="hidden" value="metadata"><button>选择分组元数据文件</button></form>
<form method="post" action="/plot"><table width="100%"><tr><td width="260" valign="top"><h3>筛选</h3>
<p>数据集 CSV<br><input name="dataset_path" size="34" value="{_value(state.dataset_path)}"></p>
<p>分组元数据<br><input name="metadata_path" size="34" value="{_value(state.metadata_path)}"></p>
<p>分组<br><select name="group_id">{group_options}</select></p><p>轨迹片段<br><select name="segment_id">{segment_options}</select></p>
<p><button type="submit">画图／更新选择</button></p></td><td valign="top"><h3>三维图</h3>{image}<p>{summary}</p></td>
<td width="190" valign="top"><h3>显示方式</h3><p><label><input type="checkbox" name="whole_flight" value="1"{checked}> 按整条轨迹展示</label></p>
<p>未勾选时按阶段片段展示；勾选后展示所选分组成员所属航班的完整轨迹。</p></td></tr></table></form>
<hr><p>数据集应包含 record_id、flight_id、time_ns、longitude_deg、latitude_deg、geometric_altitude_m。分组元数据使用 routing_segments.csv 或其 group manifest.json。</p></td></tr></table></body></html>'''


def _options(values: tuple[str, ...], selected: str | None, all_label: str) -> str:
    """构造转义后的下拉选项。"""
    rows = [f'<option value=""{" selected" if selected is None else ""}>{escape(all_label)}</option>']
    rows.extend(f'<option value="{escape(value)}"{" selected" if value == selected else ""}>{escape(value)}</option>'
                for value in values)
    return ''.join(rows)


def _value(path: Path | None) -> str:
    """返回可安全嵌入 HTML 属性的当前文件路径。"""
    return escape(str(path) if path is not None else '', quote=True)


def main(argv: list[str] | None = None) -> int:
    """启动仅本机监听的服务，并写入统一项目日志。"""
    args = build_parser().parse_args(argv)
    root = PathUtils.get_project_root()
    logger, path = setup_project_logging(console=True, project_root=root, name='cafad.local_plot')
    server = ThreadingHTTPServer((args.host, args.port), create_handler(LocalPlotState(), logger))
    logger.info('本地绘图服务启动：http://%s:%d/；本机绘图进程上限=%d；日志=%s',
                args.host, args.port, Env.local_plot_workers(), path)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info('本地绘图服务收到中断。')
    finally:
        server.server_close()
        close_project_logging(logger)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
