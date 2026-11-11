"""Check file persistence, console isolation, source locations and tracebacks."""
from contextlib import redirect_stdout
import io
import logging
from pathlib import Path
import re
import tempfile
import unittest

from src.utils.logging_utils import close_project_logging, setup_project_logging
from src.app.cli import _format_elapsed


PREFIX = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4} test_logging\.py:\d+ (DEBUG|INFO|WARNING|ERROR) ')


class LoggingTests(unittest.TestCase):
    def test_elapsed_formatter_rounds_to_hhmmss(self):
        self.assertEqual(_format_elapsed(0.0), '00:00:00')
        self.assertEqual(_format_elapsed(65.4), '00:01:05')
        self.assertEqual(_format_elapsed(3661.5), '01:01:02')

    def test_file_only_logging_retains_chinese_and_prefixes_multiline_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            console = io.StringIO()
            with redirect_stdout(console):
                logger, path = setup_project_logging(console=False, project_root=Path(directory), level=logging.DEBUG)
                try:
                    logger.debug('调试日志')
                    logger.info('中文日志')
                    logger.warning('第一行\n第二行')
                finally:
                    close_project_logging(logger)
            self.assertEqual(console.getvalue(), '')
            self.assertEqual(path.parent, Path(directory) / 'logs')
            lines = path.read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(lines), 4)
            self.assertTrue(all(PREFIX.match(line) for line in lines))
            self.assertTrue(lines[0].endswith('调试日志'))
            self.assertTrue(lines[1].endswith('中文日志'))

    def test_console_mirroring_reconfiguration_and_root_handler_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            console, inherited = io.StringIO(), io.StringIO()
            root_handler = logging.StreamHandler(inherited)
            logging.getLogger().addHandler(root_handler)
            try:
                with redirect_stdout(console):
                    logger, first = setup_project_logging(console=True, project_root=Path(directory))
                    logger.info('first-only')
                    logger, second = setup_project_logging(console=False, project_root=Path(directory))
                    try:
                        logger.info('second-only')
                    finally:
                        close_project_logging(logger)
                self.assertEqual(console.getvalue().count('first-only'), 1)
                self.assertNotIn('second-only', console.getvalue())
                self.assertEqual(inherited.getvalue(), '')
                self.assertEqual(first.read_text(encoding='utf-8').count('first-only'), 1)
                self.assertNotIn('second-only', first.read_text(encoding='utf-8'))
                self.assertEqual(second.read_text(encoding='utf-8').count('second-only'), 1)
                self.assertNotEqual(first, second)
            finally:
                logging.getLogger().removeHandler(root_handler)
                root_handler.close()

    def test_every_traceback_line_has_the_requested_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            logger, path = setup_project_logging(console=False, project_root=Path(directory))
            try:
                try:
                    raise ValueError('fixture failure')
                except ValueError:
                    logger.exception('执行失败')
            finally:
                close_project_logging(logger)
            lines = path.read_text(encoding='utf-8').splitlines()
            self.assertGreater(len(lines), 2)
            self.assertTrue(all(PREFIX.match(line) for line in lines))
            self.assertIn('ValueError: fixture failure', lines[-1])


if __name__ == '__main__':
    unittest.main()
