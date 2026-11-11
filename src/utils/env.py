"""Project-local environment configuration without third-party dependencies."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_FLOOR
import os
from pathlib import Path
import re


class Env:
    """Static configuration access; process environment overrides the .env file."""

    _values: dict[str, str] = {}
    _path: Path | None = None

    @staticmethod
    def load(path: str | Path | None = None) -> None:
        config = Path(path) if path is not None else Path(__file__).resolve().parents[2] / '.env'
        values: dict[str, str] = {}
        for number, raw in enumerate(config.read_text(encoding='utf-8-sig').splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[7:].strip()
            if '=' not in line:
                raise ValueError(f'{config}:{number}: expected KEY=VALUE')
            key, value = line.split('=', 1)
            key, value = key.strip(), value.strip()
            if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
                raise ValueError(f'{config}:{number}: invalid configuration key')
            if value[:1] in {'"', "'"}:
                quote = value[0]
                end = value.find(quote, 1)
                if end < 0 or (value[end + 1:].strip() and not value[end + 1:].lstrip().startswith('#')):
                    raise ValueError(f'{config}:{number}: invalid quoted value')
                value = value[1:end]
            else:
                value = re.split(r'\s+#', value, maxsplit=1)[0].strip()
            values[key.lower()] = value
        Env._values, Env._path = values, config.resolve()

    @staticmethod
    def get(key: str, default: str | None = None) -> str | None:
        if Env._path is None:
            Env.load()
        for candidate in (key, key.lower(), key.upper()):
            if candidate in os.environ:
                return os.environ[candidate]
        return Env._values.get(key.lower(), default)

    @staticmethod
    def get_int(key: str, default: int | None = None) -> int:
        value = Env.get(key, None if default is None else str(default))
        if value is None or not re.fullmatch(r'[+-]?\d+', value):
            raise ValueError(f'{key} must be an integer')
        return int(value)

    @staticmethod
    def get_fraction(key: str) -> Decimal:
        value = Env.get(key)
        if value is None:
            raise ValueError(f'Missing {key}')
        try:
            fraction = Decimal(value.rstrip('%').strip())
            if value.endswith('%') or fraction > 1:
                fraction /= 100
        except InvalidOperation as exc:
            raise ValueError(f'{key} must be a percentage such as 80%') from exc
        if not fraction.is_finite() or not 0 < fraction <= 1:
            raise ValueError(f'{key} must be greater than zero and at most 100%')
        return fraction

    @staticmethod
    def data_process_workers() -> int:
        cores = Env.get_int('nproc')
        if cores < 1:
            raise ValueError('nproc must be positive')
        workers = int((Decimal(cores) * Env.get_fraction('data_proc_usage')).to_integral_value(rounding=ROUND_FLOOR))
        if workers < 1:
            raise ValueError('floor(nproc * data_proc_usage) is zero; increase the configured budget')
        return workers
