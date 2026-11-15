import io
import json
import zipfile

import h5py
import numpy as np
from scipy.io import loadmat

from download_umn_uav import ROOT


def describe(value, path=''):
    if isinstance(value, dict):
        for key, child in value.items():
            if not key.startswith('__'):
                describe(child, path + '.' + key)
    elif isinstance(value, np.ndarray):
        print(path, value.shape, str(value.dtype))
    else:
        print(path, type(value).__name__, str(value)[:70])


manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
zips = json.loads((ROOT / 'inspection/zip_members.json').read_text(encoding='utf-8'))
print('VIREO_ZIP_MATS', [(i['item'], [j['name'] for j in i['members'] if j['name'].lower().endswith('.mat')]) for i in zips if i['item'].startswith('Vireo')])
item = next(i for i in manifest['items'] if i['title'] == 'Vireo Flight 1')
spec = next(f for f in item['files'] if f['name'].endswith('.zip'))
with zipfile.ZipFile(ROOT / spec['relative_path']) as archive:
    member = next(n for n in archive.namelist() if n.endswith('alldata.mat'))
    data = loadmat(io.BytesIO(archive.read(member)), simplify_cells=True)
    describe(data)
item = next(i for i in manifest['items'] if i['title'] == 'FASER Flight 04')
specs = [f for f in item['files'] if f['name'].endswith('.h5')]
with h5py.File(ROOT / specs[0]['relative_path'], 'r') as a, h5py.File(ROOT / specs[1]['relative_path'], 'r') as b:
    print('FASER4', [(f['name'], h['time'].shape) for f, h in zip(specs, [a, b])])
    print('FASER4_EQUAL_TIME', np.array_equal(a['time'][...], b['time'][...]))
