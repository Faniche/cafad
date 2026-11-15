"""Inspect local original files without executing downloaded MATLAB/flight code."""
from collections import Counter
import json
from pathlib import Path
import zipfile

import h5py
import numpy as np
from pypdf import PdfReader
from scipy.io import loadmat, whosmat

from download_umn_uav import ROOT


def describe(value, depth=0):
    if isinstance(value, dict):
        return {k: describe(v, depth + 1) for k, v in value.items() if not k.startswith('__')}
    if isinstance(value, np.ndarray):
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'first_values': value.reshape(-1)[:3].tolist() if value.dtype.kind in 'biuf' else None}
    return {'type': type(value).__name__, 'value': str(value)[:100]}


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    output = ROOT / 'inspection'
    output.mkdir(exist_ok=True)
    representatives = {}
    for item in manifest['items']:
        family = item['title'].split(' Flight')[0].split(' Ground')[0]
        representatives.setdefault(family, item)
    result = {}
    for family, item in representatives.items():
        entry = {'title': item['title'], 'files': [f['name'] for f in item['files']]}
        mat = next((ROOT / f['relative_path'] for f in item['files'] if f['name'].lower().endswith('.mat')), None)
        hdf = next((ROOT / f['relative_path'] for f in item['files'] if f['name'].lower().endswith('.h5')), None)
        if mat:
            try:
                entry['mat'] = describe(loadmat(mat, simplify_cells=True))
            except Exception as exc:
                entry['mat_error'] = repr(exc)
        if hdf:
            datasets = {}
            with h5py.File(hdf, 'r') as file:
                def visit(name, value):
                    if isinstance(value, h5py.Dataset):
                        datasets[name] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                                          'attributes': {str(k): str(v) for k, v in value.attrs.items()}}
                file.visititems(visit)
            entry['h5'] = datasets
        for spec in item['files']:
            if spec['name'].lower().endswith('.zip'):
                with zipfile.ZipFile(ROOT / spec['relative_path']) as archive:
                    entry.setdefault('zip', {})[spec['name']] = [{'name': i.filename, 'bytes': i.file_size} for i in archive.infolist()]
        result[family] = entry
        print(family, json.dumps(entry, ensure_ascii=False)[:18000], flush=True)
    (output / 'representative_structures.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    # Each PDF checksum is read once: tables provide authoritative field/unit meanings.
    dictionaries = {}
    for item in manifest['items']:
        for spec in item['files']:
            if 'dictionary' in spec['name'].lower() or 'flightdataformat' in spec['name'].lower():
                dictionaries.setdefault(spec['checksum']['value'], spec)
    for checksum, spec in dictionaries.items():
        path = ROOT / spec['relative_path']
        try:
            reader = PdfReader(path)
            text = '\n\n'.join(page.extract_text() or '' for page in reader.pages)
            (output / f'{checksum}_{path.stem}.txt').write_text(text, encoding='utf-8')
        except Exception as exc:
            print('PDF ERROR', path, repr(exc), flush=True)
    zip_inventory = []
    for item in manifest['items']:
        for spec in item['files']:
            if spec['name'].lower().endswith('.zip'):
                with zipfile.ZipFile(ROOT / spec['relative_path']) as archive:
                    zip_inventory.append({'item': item['title'], 'path': spec['relative_path'],
                                          'members': [{'name': i.filename, 'bytes': i.file_size} for i in archive.infolist()]})
    (output / 'zip_members.json').write_text(json.dumps(zip_inventory, ensure_ascii=False, indent=2), encoding='utf-8')
    print('INSPECTED', len(result), 'families;', len(dictionaries), 'unique data dictionaries;', len(zip_inventory), 'ZIP files')


if __name__ == '__main__':
    main()
