import os
import pandas as pd, numpy as np, pydicom
g = pd.read_csv('data/geometry_features.csv')
t = pd.read_csv('out/dicom_tags_499.csv')
h = pd.read_csv('outputs/nested_gate/results/pixel_hashes.csv') if __import__('os').path.exists('outputs/nested_gate/results/pixel_hashes.csv') else None
print('hash file', None if h is None else h.shape)
hip = g[g.region.isin(['right_hip','left_hip'])].copy()
hip['lab_side'] = hip.region.map({'right_hip':'right','left_hip':'left'})
hip['mismatch'] = hip.lab_side != hip.hip_side_detected
print('n hip', len(hip), 'mismatch', hip.mismatch.sum())
print(pd.crosstab(hip.lab_side, hip.hip_side_detected))
# acquisition time
acq = {}
for fp in hip.file_path:
    ds = pydicom.dcmread(fp, force=True, stop_before_pixels=True)
    acq[fp] = (str(ds.get('AcquisitionTime','')), str(ds.get('ExposedArea','')), str(ds.get('SeriesNumber','')), str(ds.get('EntranceDoseInmGy','')))
hip['acq_time'] = hip.file_path.map(lambda f: acq[f][0]); hip['exposed_area']=hip.file_path.map(lambda f: acq[f][1]); hip['series_no']=hip.file_path.map(lambda f: acq[f][2])
if h is not None:
    hip = hip.merge(h[['file_path','pixel_hash']], on='file_path', how='left')
# per study: detected side counts
per = hip.groupby('study').agg(n=('file_path','size'), n_right_det=('hip_side_detected', lambda s:(s=='right').sum()), n_right_lab=('lab_side', lambda s:(s=='right').sum()), n_uniq=('pixel_hash','nunique') if h is not None else ('file_path','size'))
print(per.describe())
print('studies where detected right==left count (balanced):', ((per.n_right_det*2==per.n)).sum(), 'of', len(per))
print('studies where label right==left count:', ((per.n_right_lab*2==per.n)).sum())
# side score distribution
print(hip.hip_side_score.describe()); print('|score|<0.05:', (hip.hip_side_score.abs()<0.05).sum())
print(hip[hip.mismatch].hip_side_score.abs().describe())
# within-study order: instance number vs detected side
hip = hip.sort_values(['study','instance_number'])
hip['rank_in_study'] = hip.groupby('study').cumcount()
print(pd.crosstab(hip.rank_in_study, hip.hip_side_detected))
print('unique hashes: same hash different detected side?')
if h is not None:
    bad = hip.groupby('pixel_hash').hip_side_detected.nunique(); print((bad>1).sum())
    badl = hip.groupby('pixel_hash').lab_side.nunique(); print('same hash, different label side:', (badl>1).sum())
hip.to_csv('out/hip_side_table.csv', index=False)
