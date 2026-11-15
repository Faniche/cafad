"""Download public UMN UAV flight data with metadata and checksum verification."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = 'https://conservancy.umn.edu/server/api'
COLLECTION = '03a03027-857c-4a30-9e21-cd1df50211c4'
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / 'data' / 'umn'
META = ROOT / 'metadata'
META.mkdir(parents=True, exist_ok=True)
CONTENT_LOCKS = {}
CONTENT_LOCKS_GUARD = threading.Lock()
VERIFIED_CONTENT = {}


def fetch_json(url, cache=None):
    if cache and cache.exists():
        return json.loads(cache.read_text(encoding='utf-8'))
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/hal+json,application/json'})
            with urllib.request.urlopen(request, timeout=45) as response:
                result = json.load(response)
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            return result
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


def discover():
    collection = fetch_json(f'{BASE}/core/collections/{COLLECTION}', META / 'collection.json')
    print('COLLECTION', collection['name'], 'archivedItemsCount', collection['archivedItemsCount'], flush=True)
    results = []
    page = 0
    while True:
        query = urllib.parse.urlencode({'scope': COLLECTION, 'size': 100, 'page': page})
        document = fetch_json(f'{BASE}/discover/search/objects?{query}', META / f'discovery_page_{page}.json')
        search_result = document.get('_embedded', {}).get('searchResult', {})
        objects = search_result.get('_embedded', {}).get('objects', [])
        if not objects:
            print('DISCOVERY SHAPE', json.dumps(document)[:3000], flush=True)
            break
        for wrapper in objects:
            item = wrapper.get('_embedded', {}).get('indexableObject', {})
            if item.get('type') == 'item':
                results.append(item)
        paging = search_result.get('page', {})
        print('DISCOVERY PAGE', page, 'items', len(objects), 'paging', paging, flush=True)
        if page + 1 >= paging.get('totalPages', 1):
            break
        page += 1
    unique = {item['uuid']: item for item in results}
    items = list(unique.values())
    (META / 'items.json').write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DISCOVERED', len(items), flush=True)
    print('SAMPLE TITLES', [item['name'] for item in items[:8]], flush=True)
    return items


def list_embedded(url, key, prefix):
    result = []
    page = 0
    while True:
        separator = '&' if '?' in url else '?'
        document = fetch_json(f'{url}{separator}size=100&page={page}', META / f'{prefix}_{page}.json')
        result.extend(document.get('_embedded', {}).get(key, []))
        if page + 1 >= document.get('page', {}).get('totalPages', 1):
            break
        page += 1
    return result


def safe_name(value):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', value).strip(' .')
    if not name or name.upper().split('.')[0] in {'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'LPT1'}:
        name = 'file_' + name
    return name


def inventory_item(item):
    uid = item['uuid']
    cache = META / 'inventory' / f'{uid}.json'
    if cache.exists():
        return json.loads(cache.read_text(encoding='utf-8'))
    bundles = list_embedded(item['_links']['bundles']['href'], 'bundles', f'items/{uid}/bundles')
    files = []
    directory = 'raw/' + safe_name(item['name']).replace(' ', '_') + '__' + uid[:8]
    used = set()
    for bundle in bundles:
        if bundle['name'] != 'ORIGINAL':
            continue
        streams = list_embedded(bundle['_links']['bitstreams']['href'], 'bitstreams', f'items/{uid}/bitstreams_{bundle["uuid"]}')
        for stream in streams:
            name = safe_name(stream['name'])
            if name.lower() in used:
                name = stream['uuid'][:8] + '_' + name
            used.add(name.lower())
            files.append({
                'uuid': stream['uuid'], 'name': stream['name'], 'size_bytes': stream['sizeBytes'],
                'checksum': stream.get('checkSum'), 'url': stream['_links']['content']['href'],
                'relative_path': directory + '/' + name, 'metadata': stream.get('metadata', {}),
            })
    result = {'uuid': uid, 'title': item['name'], 'handle': item.get('handle'),
              'item_url': 'https://conservancy.umn.edu/items/' + uid,
              'metadata': item.get('metadata', {}), 'files': files}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def inventory(items, workers):
    results, errors = [], []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(inventory_item, item): item for item in items}
        for future in as_completed(pending):
            item = pending[future]
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({'uuid': item['uuid'], 'title': item['name'], 'error': str(exc)})
                print('INVENTORY ERROR', item['name'], str(exc), flush=True)
            done = len(results) + len(errors)
            if done % 10 == 0 or done == len(items):
                print('INVENTORY', done, '/', len(items), 'files', sum(len(i['files']) for i in results),
                      'GiB', round(sum(f['size_bytes'] for i in results for f in i['files']) / 1024**3, 3),
                      'elapsed_s', round(time.monotonic() - started), flush=True)
    results.sort(key=lambda item: (item['title'], item['uuid']))
    manifest = {'collection': COLLECTION, 'source': f'https://conservancy.umn.edu/collections/{COLLECTION}',
                'expected_records': len(items), 'items': results, 'inventory_errors': errors}
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def check_file(path, spec):
    if not path.exists() or path.stat().st_size != spec['size_bytes']:
        return None
    expected = spec.get('checksum') or {}
    algorithm = expected.get('checkSumAlgorithm', 'MD5').lower().replace('-', '')
    digest = hashlib.new(algorithm)
    sha256 = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
            sha256.update(block)
    if expected.get('value') and digest.hexdigest().lower() != expected['value'].lower():
        return None
    return {'algorithm': algorithm, 'checksum': digest.hexdigest(), 'sha256': sha256.hexdigest(),
            'bytes': path.stat().st_size}


def download_file(spec):
    checksum = spec.get('checksum') or {}
    key = (spec['size_bytes'], checksum.get('checkSumAlgorithm'), checksum.get('value') or spec['uuid'])
    with CONTENT_LOCKS_GUARD:
        lock = CONTENT_LOCKS.setdefault(key, threading.Lock())
    with lock:
        original = VERIFIED_CONTENT.get(key)
        target = ROOT / spec['relative_path']
        if original and original != target and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
        result = download_file_locked(spec)
        if result['status'] == 'verified':
            VERIFIED_CONTENT[key] = target
        return result


def download_file_locked(spec):
    target = ROOT / spec['relative_path']
    assert target.resolve().is_relative_to(ROOT.resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    status_path = META / 'download_status' / (spec['uuid'] + '.json')
    status_path.parent.mkdir(parents=True, exist_ok=True)
    check = check_file(target, spec)
    if check:
        result = {'uuid': spec['uuid'], 'relative_path': spec['relative_path'], 'status': 'verified', **check}
        status_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
    partial = target.with_name(target.name + '.part')
    last_error = ''
    for attempt in range(5):
        try:
            # A connection can fail after delivering all bytes but before clean EOF.
            # Verify that complete partial before requesting an invalid end-of-file range.
            partial_check = check_file(partial, spec)
            if partial_check:
                partial.replace(target)
                result = {'uuid': spec['uuid'], 'relative_path': spec['relative_path'], 'status': 'verified', **partial_check}
                status_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
                return result
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > spec['size_bytes']:
                offset = 0
            headers = {'User-Agent': 'Mozilla/5.0', 'Accept': '*/*'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            request = urllib.request.Request(spec['url'], headers=headers)
            # Both routes use normal TLS verification; alternate only for transient
            # transport failures. Authorization failures are never retried this way.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if attempt % 2 else urllib.request.build_opener()
            with opener.open(request, timeout=50) as response:
                append = offset > 0 and response.status == 206
                if append:
                    content_range = response.headers.get('Content-Range', '')
                    if not content_range.startswith(f'bytes {offset}-'):
                        raise RuntimeError('Unexpected Content-Range: ' + content_range)
                with partial.open('ab' if append else 'wb') as stream:
                    for block in iter(lambda: response.read(256 * 1024), b''):
                        stream.write(block)
                        stream.flush()
            check = check_file(partial, spec)
            if check is None:
                if partial.exists() and partial.stat().st_size >= spec['size_bytes']:
                    partial.unlink()
                raise RuntimeError('File length or official checksum mismatch')
            partial.replace(target)
            result = {'uuid': spec['uuid'], 'relative_path': spec['relative_path'], 'status': 'verified', **check}
            status_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
            return result
        except Exception as exc:
            last_error = f'{type(exc).__name__}: {exc}'
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403, 404):
                break
            time.sleep(min(2 ** attempt, 8))
    result = {'uuid': spec['uuid'], 'relative_path': spec['relative_path'], 'status': 'failed', 'error': last_error}
    status_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def download(manifest, workers):
    specs = [f for item in manifest['items'] for f in item['files']]
    total = sum(f['size_bytes'] for f in specs)
    print('DOWNLOAD START', len(manifest['items']), 'records', len(specs), 'files', round(total / 1024**3, 3), 'GiB', flush=True)
    results = []
    started = time.monotonic()
    last_print = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(download_file, spec): spec for spec in specs}
        for future in as_completed(pending):
            spec = pending[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {'uuid': spec['uuid'], 'relative_path': spec['relative_path'], 'status': 'failed',
                          'error': f'{type(exc).__name__}: {exc}'}
                status_path = META / 'download_status' / (spec['uuid'] + '.json')
                status_path.parent.mkdir(parents=True, exist_ok=True)
                status_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
            results.append(result)
            elapsed = time.monotonic() - started
            if result['status'] == 'failed':
                print('DOWNLOAD ERROR', spec['relative_path'], result.get('error'), flush=True)
            if elapsed - last_print >= 15 or len(results) == len(specs):
                good = [r for r in results if r['status'] == 'verified']
                print('DOWNLOAD', len(results), '/', len(specs), 'verified', len(good),
                      'GiB', round(sum(r['bytes'] for r in good) / 1024**3, 3), '/', round(total / 1024**3, 3),
                      'elapsed_s', round(elapsed), flush=True)
                last_print = elapsed
    status = {'file_results': results, 'verified_files': sum(r['status'] == 'verified' for r in results),
              'expected_files': len(specs), 'expected_bytes': total,
              'verified_bytes': sum(r.get('bytes', 0) for r in results if r['status'] == 'verified')}
    (ROOT / 'download_result.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    write_report(manifest, status)
    print('DOWNLOAD FINISHED', {k: v for k, v in status.items() if k != 'file_results'}, flush=True)
    return status


def write_report(manifest, status):
    by_id = {r['uuid']: r for r in status['file_results']}
    coverage = []
    for item in manifest['items']:
        complete = bool(item['files']) and all(by_id.get(f['uuid'], {}).get('status') == 'verified' for f in item['files'])
        coverage.append({'title': item['title'], 'uuid': item['uuid'], 'item_url': item['item_url'],
                         'files': len(item['files']), 'bytes': sum(f['size_bytes'] for f in item['files']),
                         'complete': complete})
    ground = sum('Ground' in i['title'] for i in manifest['items'])
    total_complete = sum(i['complete'] for i in coverage)
    all_good = total_complete == manifest['expected_records'] and not manifest['inventory_errors']
    summary = {'all_complete': all_good, 'expected_records': manifest['expected_records'],
               'complete_records': total_complete, 'flight_titled_records': len(coverage) - ground,
               'ground_titled_records': ground,
               'verified_files': status['verified_files'], 'expected_files': status['expected_files'],
               'verified_bytes': status['verified_bytes'], 'expected_bytes': status['expected_bytes'],
               'items': coverage}
    (ROOT / 'verification_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 明尼苏达大学 UAV Flight Data 下载结果', '',
             f'来源：[官方 Flight Data 集合]({manifest["source"]})。本次公开检索返回 {manifest["expected_records"]} 条记录；其中 {len(coverage) - ground} 条标题为 Flight，{ground} 条为 Ground。记录数不等于不同无人机架数。', '',
             f'完成状态：{"全部完成" if all_good else "尚有未完成文件"}。完整记录 {total_complete}/{manifest["expected_records"]}；已验证文件 {status["verified_files"]}/{status["expected_files"]}；已验证大小 {status["verified_bytes"] / 1024**3:.3f} GiB。', '',
             '所有成功文件已核对官方文件字节数和 MD5，并额外保存 SHA-256。原始文件未转换；ZIP 尚未解压。', '',
             'raw/ 下按每条记录独立存放；manifest.json 保存文件来源、官方校验值、元数据和相对路径；download_result.json 保存逐文件核验结果；metadata/ 保存原始接口响应。', '',
             '在项目根目录运行 `.venv\\Scripts\\python.exe -B scripts\\umn\\download_umn_uav.py --download --workers 12` 可跳过已验证文件并重试缺失文件。', '',
             '| 记录 | 原始附件数 | 大小 MiB | 全部附件已验证 |', '|---|---:|---:|---|']
    for item in coverage:
        lines.append(f'| [{item["title"]}]({item["item_url"]}) | {item["files"]} | {item["bytes"] / 1024**2:.2f} | {"是" if item["complete"] else "否"} |')
    (ROOT / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('COVERAGE', total_complete, '/', manifest['expected_records'], 'ALL_COMPLETE', all_good, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', action='store_true')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.download and (ROOT / 'manifest.json').exists() and not args.inventory:
        manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    else:
        items_path = META / 'items.json'
        items = json.loads(items_path.read_text(encoding='utf-8')) if items_path.exists() else discover()
        manifest = inventory(items, args.workers) if args.inventory or args.download else None
    if args.download:
        download(manifest, args.workers)
