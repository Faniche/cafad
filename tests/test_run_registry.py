"""test_run_registry.py | 2026-11-04 | 自动运行编号与持久化台账测试 | Codex."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest
import sqlite3

from src.app.module_catalog import MODULE_SEQUENCE, artifact_module_directory
from src.app.run_registry import DatasetIdentity, allocate_run, identify_dataset, update_run_state


class RunRegistryTests(unittest.TestCase):

    def test_module_catalog_has_the_fixed_pipeline_order_and_directories(self) -> None:
        """磁盘目录编号只来自唯一目录表，不能由各模块自行决定。"""
        self.assertEqual(MODULE_SEQUENCE, (
            'preprocess', 'split', 'segment', 'group', 'inject',
            'train', 'calibrate', 'detect', 'evaluate',
        ))
        self.assertEqual(
            tuple(artifact_module_directory(name) for name in MODULE_SEQUENCE),
            ('0_preprocess', '1_split', '2_segment', '3_group', '4_inject',
             '5_train', '6_calibrate', '7_detect', '8_evaluate'),
        )
    """验证每个数据集模块组合单独递增，并始终写回最终状态。"""

    def test_preprocess_dataset_identity_is_the_full_filename_stem(self) -> None:
        """文件名中的后缀不是数据集 ID 的截断规则。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'preprocessed_data_d1.csv'
            source.write_text('fixture', encoding='utf-8')
            identity = identify_dataset(root, 'preprocess', argparse.Namespace(input=source))
            self.assertEqual(identity.dataset_id, 'preprocessed_data_d1')
            self.assertEqual(identity.dataset_name, 'preprocessed_data_d1')

    def test_counter_and_json_registry_record_success_and_failure(self) -> None:
        """同一 dataset/module 的后续运行获得下一编号，状态在两个台账一致。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = DatasetIdentity(dataset_id='preprocessed_data_d1', dataset_name='preprocessed_data_d1')
            first = allocate_run(root, dataset, 'preprocess', seed=10)
            update_run_state(root, first, 'success')
            second = allocate_run(root, dataset, 'preprocess', seed=11)
            update_run_state(root, second, 'failure')

            database = sqlite3.connect(root / 'data' / 'runs.sqlite')
            try:
                rows = database.execute(
                    'SELECT dataset, module, counter, state FROM runs ORDER BY counter'
                ).fetchall()
            finally:
                database.close()
            self.assertEqual(rows, [
                ('preprocessed_data_d1', 'preprocess', 1, 'success'),
                ('preprocessed_data_d1', 'preprocess', 2, 'failure'),
            ])
            registry = json.loads((root / 'run_registry.json').read_text(encoding='utf-8'))
            self.assertEqual([entry['run_id'] for entry in registry['runs']], ['1', '2'])
            self.assertEqual([entry['state'] for entry in registry['runs']], ['success', 'failure'])
            self.assertTrue(first.log_filename.endswith('#preprocessed_data_d1#preprocess#1_10.log'))
            self.assertEqual(second.directory_name.rsplit('#', 1)[-1], '11')
            self.assertTrue(registry['runs'][0]['artifact_dir'].startswith('data/0_preprocess/'))


if __name__ == '__main__':
    unittest.main()
