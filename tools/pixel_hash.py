"""sha1 нормализованных пикселей (inference.normalize_pixels) для каждого файла из labels_for_embeddings.csv."""
import os, sys, hashlib
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "1"
ROOT = Path(os.environ.get('DENSITO_ROOT', Path(__file__).resolve().parents[1])); sys.path.insert(0, str(ROOT / 'src'))
import numpy as np, pandas as pd, pydicom
from inference import normalize_pixels
lab = pd.read_csv(ROOT / 'data' / 'labels_for_embeddings.csv')
hashes = []
for p in lab['file_path']:
    ds = pydicom.dcmread(p, force=True)
    img = normalize_pixels(ds)
    hashes.append(hashlib.sha1(img.tobytes() + str(img.shape).encode()).hexdigest())
lab['pixel_hash'] = hashes
lab[['study', 'file_path', 'region', 'pixel_hash']].to_csv(ROOT / 'outputs' / 'pixel_hashes.csv', index=False)
vc = lab['pixel_hash'].value_counts()
dup = vc[vc > 1]
print('files', len(lab), 'unique hashes', lab['pixel_hash'].nunique(), 'dup groups', len(dup), 'files in dup groups', int(dup.sum()))
# дубликаты между разными исследованиями
g = lab.groupby('pixel_hash')['study'].nunique()
print('hash groups spanning >1 study:', int((g > 1).sum()))
for h in dup.index:
    sub = lab[lab['pixel_hash'] == h]
    print(h[:10], sub['region'].iloc[0], 'studies:', sub['study'].nunique(), 'files:', len(sub))
