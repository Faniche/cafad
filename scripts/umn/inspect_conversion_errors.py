import json

import numpy as np

from convert_umn_to_csv import ROOT, candidates_for, read_source, flatten

manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
for title in ['FASER Flight 02', 'Loki Flight 45']:
    item = next(i for i in manifest['items'] if i['title'] == title)
    choices = candidates_for(item)
    print('ITEM', title, 'CHOICES', choices)
    data = read_source(choices[0])
    for key, value in flatten(data).items():
        array = np.asarray(value)
        if title.startswith('FASER'):
            print(key, array.shape, array.dtype, str(array.reshape(-1)[:3])[:150])
        elif 'time' in key.lower():
            print(key, array.shape, array.dtype, 'nonfinite', int((~np.isfinite(array)).sum()),
                  'locations', np.flatnonzero(~np.isfinite(array))[:15].tolist())
