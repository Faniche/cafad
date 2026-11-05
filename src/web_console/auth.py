"""auth.py | 2026-11-04 | Web 控制台令牌摘要校验与短期会话管理 | Codex."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from hmac import compare_digest
from http.cookies import SimpleCookie
import secrets
import threading
import time


SESSION_COOKIE_NAME = 'cafad_web_session'
SESSION_TTL_SECONDS = 8 * 60 * 60
MAX_SESSIONS = 100


class SessionAuthenticator:
    """以令牌摘要换取短期随机会话，不在后续请求中重复传输访问令牌。"""

    def __init__(self, access_token: str) -> None:
        self._access_token_digest = sha256(access_token.encode('utf-8')).digest()
        self._sessions: dict[bytes, float] = {}
        self._form_states: dict[bytes, dict[str, object]] = {}
        self._lock = threading.Lock()

    def issue_session(self, supplied_token: str) -> str | None:
        """校验提交令牌的 SHA-256 摘要；成功时返回一次随机会话值。"""
        supplied_digest = sha256(supplied_token.encode('utf-8')).digest()
        if not compare_digest(supplied_digest, self._access_token_digest):
            return None
        session = secrets.token_urlsafe(32)
        session_digest = sha256(session.encode('utf-8')).digest()
        with self._lock:
            self._prune_expired_locked(time.time())
            if len(self._sessions) >= MAX_SESSIONS:
                oldest = min(self._sessions, key=self._sessions.__getitem__)
                del self._sessions[oldest]
                self._form_states.pop(oldest, None)
            self._sessions[session_digest] = time.time() + SESSION_TTL_SECONDS
        return session

    def is_session_valid(self, cookie_header: str | None) -> bool:
        """判断 Cookie 中的随机会话是否尚未失效。"""
        with self._lock:
            return self._valid_session_digest_locked(cookie_header, time.time()) is not None

    def load_form_state(self, cookie_header: str | None) -> dict[str, object] | None:
        """返回当前有效会话暂存的表单状态；副本不会暴露服务端可变对象。"""
        with self._lock:
            session_digest = self._valid_session_digest_locked(cookie_header, time.time())
            if session_digest is None:
                return None
            return deepcopy(self._form_states.get(session_digest, {}))

    def save_form_state(self, cookie_header: str | None, state: dict[str, object]) -> bool:
        """把已由 HTTP 层校验大小的 JSON 表单状态绑定到当前登录会话。"""
        with self._lock:
            session_digest = self._valid_session_digest_locked(cookie_header, time.time())
            if session_digest is None:
                return False
            self._form_states[session_digest] = deepcopy(state)
            return True

    def revoke_session(self, cookie_header: str | None) -> None:
        """撤销当前浏览器会话；令牌本身不受影响。"""
        with self._lock:
            session_digest = self._valid_session_digest_locked(cookie_header, time.time())
            if session_digest is not None:
                self._sessions.pop(session_digest, None)
                self._form_states.pop(session_digest, None)

    @staticmethod
    def make_set_cookie(session: str | None, max_age: int) -> str:
        """生成 JavaScript 无法读取的同源会话 Cookie。"""
        cookie = SimpleCookie()
        cookie[SESSION_COOKIE_NAME] = session or ''
        morsel = cookie[SESSION_COOKIE_NAME]
        morsel['path'] = '/'
        morsel['httponly'] = True
        morsel['samesite'] = 'Strict'
        morsel['max-age'] = str(max_age)
        return morsel.OutputString()

    @staticmethod
    def _session_from_cookie(cookie_header: str | None) -> str | None:
        """从请求 Cookie 安全提取本服务会话值。"""
        if not cookie_header:
            return None
        try:
            cookies = SimpleCookie()
            cookies.load(cookie_header)
        except (ValueError, KeyError):
            return None
        item = cookies.get(SESSION_COOKIE_NAME)
        return item.value if item is not None else None

    def _valid_session_digest_locked(self, cookie_header: str | None, now: float) -> bytes | None:
        """在调用方持锁时校验 Cookie 对应会话并先清理已过期状态。"""
        session = self._session_from_cookie(cookie_header)
        if session is None:
            return None
        self._prune_expired_locked(now)
        session_digest = sha256(session.encode('utf-8')).digest()
        expires_at = self._sessions.get(session_digest)
        return session_digest if expires_at is not None and expires_at > now else None

    def _prune_expired_locked(self, now: float) -> None:
        """在调用方持锁时移除过期会话，控制长期服务内存。"""
        for digest, expires_at in tuple(self._sessions.items()):
            if expires_at <= now:
                del self._sessions[digest]
                self._form_states.pop(digest, None)
