"""test_artifacts.py | 2026-11-03 | 产物清单与正式运行锁的边界测试 | Codex."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.app.artifacts import ProjectRunLock, read_manifest


class ArtifactTests(unittest.TestCase):
    """检查下游拒绝已损坏产物，且锁不允许第二个模块进入。"""

    def test_read_manifest_rejects_tampered_output(self) -> None:
        """清单状态成功不足以放行，输出内容必须仍与发布时哈希一致。"""
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / 'artifact'
            artifact.mkdir()
            output = artifact / 'data.txt'
            output.write_text('original', encoding='utf-8')
            digest = hashlib.sha256(output.read_bytes()).hexdigest()
            (artifact / 'manifest.json').write_text(json.dumps({
                'status': 'SUCCEEDED',
                'module': 'preprocess',
                'outputs': [{'path': 'data.txt', 'sha256': digest}],
            }), encoding='utf-8')
            read_manifest(artifact / 'manifest.json', expected_module='preprocess')
            output.write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '哈希不匹配'):
                read_manifest(artifact / 'manifest.json', expected_module='preprocess')

    def test_project_run_lock_rejects_a_second_holder_and_releases_cleanly(self) -> None:
        """第二个入口不能绕开锁；原持有者离开后锁目录必须消失。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with ProjectRunLock(root):
                with self.assertRaisesRegex(RuntimeError, '拒绝并发执行'):
                    with ProjectRunLock(root):
                        pass
            self.assertFalse((root / '.cafad-run.lock').exists())


if __name__ == '__main__':
    unittest.main()
