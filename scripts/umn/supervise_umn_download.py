"""Keep the authorized public-data download running and publish local progress."""
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from download_umn_uav import ROOT, META, write_report

WORKSPACE = Path(__file__).resolve().parent
KERNEL = ctypes.WinDLL('kernel32', use_last_error=True)
KERNEL.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
KERNEL.OpenProcess.restype = ctypes.c_void_p
KERNEL.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
KERNEL.CloseHandle.argtypes = [ctypes.c_void_p]


def running(pid):
    handle = KERNEL.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    code = ctypes.c_ulong()
    try:
        return bool(KERNEL.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        KERNEL.CloseHandle(handle)


def snapshot(active_pid=None):
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    records = []
    for path in (META / 'download_status').glob('*.json'):
        try:
            records.append(json.loads(path.read_text(encoding='utf-8')))
        except (OSError, json.JSONDecodeError):
            continue
    by_id = {r['uuid']: r for r in records}
    specs = [f for item in manifest['items'] for f in item['files']]
    good = {f['uuid'] for f in specs if by_id.get(f['uuid'], {}).get('status') == 'verified'
            and (ROOT / f['relative_path']).exists()
            and (ROOT / f['relative_path']).stat().st_size == f['size_bytes']}
    complete = sum(bool(item['files']) and all(f['uuid'] in good for f in item['files']) for item in manifest['items'])
    size = sum(f['size_bytes'] for f in specs if f['uuid'] in good)
    total = sum(f['size_bytes'] for f in specs)
    status = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'active_download_pid': active_pid,
              'all_complete': len(good) == len(specs) and complete == manifest['expected_records'],
              'expected_records': manifest['expected_records'], 'complete_records': complete,
              'expected_files': len(specs), 'verified_files': len(good),
              'expected_bytes': total, 'verified_bytes': size,
              'failed_files': [by_id[f['uuid']] for f in specs if by_id.get(f['uuid'], {}).get('status') == 'failed']}
    text = json.dumps(status, ensure_ascii=False, indent=2)
    temporary = ROOT / 'progress.json.tmp'
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(ROOT / 'progress.json')
    state = '全部完成' if status['all_complete'] else '正在下载' if active_pid else '下载尚未完成'
    (ROOT / 'STATUS.md').write_text(
        '# UMN UAV 下载进度\n\n'
        f'状态：{state}。\n\n'
        f'- 完整记录：{complete}/{manifest["expected_records"]}\n'
        f'- 已校验文件：{len(good)}/{len(specs)}\n'
        f'- 已校验数据：{size / 1024**3:.3f}/{total / 1024**3:.3f} GiB\n'
        f'- 最近更新（UTC）：{status["updated_utc"]}\n\n'
        + ('下载进程在后台运行，关闭聊天窗口不影响它。' if active_pid else '下载进程已结束。')
        + '成功文件均已核对官方 MD5 和字节数。\n\n'
        '原始文件在 raw/；完整来源与官方校验值见 manifest.json；全部结束后详见 README.md。\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in status.items() if k != 'failed_files'}), flush=True)
    return manifest, status, by_id


def main():
    pid = int((ROOT / 'download.pid').read_text(encoding='utf-8').strip())
    for round_number in range(5):
        while running(pid):
            snapshot(pid)
            time.sleep(20)
        manifest, status, by_id = snapshot()
        if status['all_complete']:
            result = {'file_results': list(by_id.values()), 'verified_files': status['verified_files'],
                      'expected_files': status['expected_files'], 'expected_bytes': status['expected_bytes'],
                      'verified_bytes': status['verified_bytes']}
            write_report(manifest, result)
            print('ALL DOWNLOADS VERIFIED', flush=True)
            return
        if round_number == 4:
            print('STOPPED AFTER FIVE DOWNLOAD PASSES; SEE progress.json FOR REMAINING FILES', flush=True)
            return
        time.sleep(10)
        with (ROOT / 'download.log').open('a', encoding='utf-8') as log, (ROOT / 'download_errors.log').open('a', encoding='utf-8') as errors:
            child = subprocess.Popen([sys.executable, '-B', '-u', str(WORKSPACE / 'download_umn_uav.py'),
                                      '--download', '--workers', '24'], cwd=WORKSPACE,
                                     stdout=log, stderr=errors, creationflags=subprocess.CREATE_NO_WINDOW)
        pid = child.pid
        (ROOT / 'download.pid').write_text(str(pid), encoding='utf-8')


if __name__ == '__main__':
    main()
