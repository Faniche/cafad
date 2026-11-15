import concurrent.futures
import json
from pathlib import Path
import urllib.request
from download_umn_uav import ROOT

URLS = [
    'https://conservancy.umn.edu/collections/03a03027-857c-4a30-9e21-cd1df50211c4',
    'https://conservancy.umn.edu/server/api',
    'https://conservancy.umn.edu/server/api/core/collections/03a03027-857c-4a30-9e21-cd1df50211c4',
]

def probe(url):
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json,text/html'})
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
            target = ROOT / 'metadata'
            target.mkdir(parents=True, exist_ok=True)
            file = target / ('probe_' + str(URLS.index(url)) + '.txt')
            file.write_bytes(body)
            return {'url': url, 'status': response.status, 'bytes': len(body), 'body': body[:3500].decode('utf-8', errors='replace')}
    except Exception as exc:
        return {'url': url, 'error': str(exc)}

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    for result in pool.map(probe, URLS):
        print(json.dumps(result), flush=True)
