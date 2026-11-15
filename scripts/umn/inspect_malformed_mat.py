import json
from pathlib import Path
from scipy.io import loadmat, whosmat
from download_umn_uav import ROOT

manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
for title in ['Thor Flight 123', 'Thor Ground 04']:
    item = next(i for i in manifest['items'] if i['title'] == title)
    path = next(ROOT / f['relative_path'] for f in item['files'] if f['name'].endswith('.mat'))
    errors = []
    variables = whosmat(path)
    for name, shape, dtype in variables:
        try:
            loadmat(path, simplify_cells=True, variable_names=[name])
        except Exception as exc:
            errors.append({'name': name, 'shape': shape, 'dtype': dtype, 'error': str(exc)})
    print(title, 'variables', len(variables), 'unreadable', json.dumps(errors), flush=True)
