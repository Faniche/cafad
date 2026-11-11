"""restart.py | 2026-11-11 | Web 控制台进程内安全重启协调 | Codex."""
from __future__ import annotations

from collections.abc import Callable
from threading import Lock, Thread
from time import sleep


RESPONSE_GRACE_SECONDS = 0.2


class RestartController:
    """在 HTTP 响应写出后关闭服务，并将实际重启交还给服务主循环。"""

    def __init__(self) -> None:
        self._lock = Lock()
        self._requested = False
        self._shutdown: Callable[[], None] | None = None

    @property
    def requested(self) -> bool:
        """返回本轮服务是否已接受重启请求。"""
        with self._lock:
            return self._requested

    def bind_shutdown(self, shutdown: Callable[[], None]) -> None:
        """绑定 HTTPServer.shutdown；绑定发生在监听套接字创建后。"""
        with self._lock:
            self._shutdown = shutdown

    def request(self) -> bool:
        """只接受一次重启，并异步等待当前 HTTP 响应写出。"""
        with self._lock:
            if self._requested or self._shutdown is None:
                return False
            self._requested = True
            shutdown = self._shutdown
        Thread(target=self._shutdown_after_response, args=(shutdown,), daemon=True).start()
        return True

    @staticmethod
    def _shutdown_after_response(shutdown: Callable[[], None]) -> None:
        """避免在请求处理线程尚未完成发送响应时关闭监听循环。"""
        sleep(RESPONSE_GRACE_SECONDS)
        shutdown()
