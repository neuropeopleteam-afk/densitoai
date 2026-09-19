import os; os.environ['OMP_NUM_THREADS']='1'
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'tools'); sys.path.insert(0,'src')
os.environ['NESTED_GATE_WORK']='/home/user/workspace/work/B'
from nested_gate import fit_geom, fit_emb, connected_groups, impute, safe_auc
from sklearn.model_selection import GroupKFold
from train_stacked import CRITERION_GEOMETRY_COLS
import build_dataset; from pathlib import Path; build_dataset.LABELS_XLSX = Path("/home/user/workspace/external_datasets/own_dataset/разметка.xlsx"); load_labels = build_dataset.load_labels
xl = load_labels(); xl.index = xl.index.astype(str)
g = pd.read_csv('data/geometry_features.csv')
lab = pd.read_csv('data/labels_for_embeddings.csv')
E = np.load('data/embeddings.npy')
hipm = g.region.isin(['right_hip','left_hip']).values
hip = g[hipm].reset_index(drop=True); Eh = E[np.nonzero(lab.region.isin(['right_hip','left_hip']).values)[0]]
h = pd.read_csv('/home/user/workspace/work/B/results/pixel_hashes.csv'); hip['pixel_hash']=hip.file_path.map(dict(zip(h.file_path,h.pixel_hash)))
groups = connected_groups(hip.study.values, hip.pixel_hash.values)
det = hip.hip_side_detected.values
def lbl(sides, crit):
    return np.array([float(xl.loc[str(s)][('rh_' if sd=='right' else 'lh_')+crit]) if str(s) in xl.index else np.nan for s, sd in zip(hip.study, sides)], float)
flip = np.where(det=='right','left','right')
dens = hip.region.str.replace('_hip','').values
variants = {}
for crit in ['pos','roi']:
    variants[f'{crit}_det'] = lbl(det, crit)
    variants[f'{crit}_swapped'] = lbl(flip, crit)
    variants[f'{crit}_density'] = lbl(dens, crit)
    a = np.vstack([variants[f'{crit}_det'], variants[f'{crit}_swapped']])
    variants[f'{crit}_any'] = np.where(np.isnan(a).all(0), np.nan, np.nanmax(a,0))
    variants[f'{crit}_both'] = np.where(np.isnan(a).all(0), np.nan, np.nanmin(a,0))
m = ~np.isnan(variants['pos_det']); print('det == hip_pos_c:', (variants['pos_det'][m]==hip.hip_pos_c.values[m]).all(), m.sum())
# agreement of labels between sides
for crit in ['pos','roi']:
    a, b = variants[f'{crit}_det'], variants[f'{crit}_swapped']; ok = ~np.isnan(a)&~np.isnan(b)
    print(crit, 'label agreement between sides (file-level):', round((a[ok]==b[ok]).mean(),3), ok.sum(), '| positive-in-both / positive-in-any:', int(((a==1)&(b==1)).sum()), int(((a==1)|(b==1))[ok].sum()))
    print(crit, 'files whose label changes if side flipped:', int((a[ok]!=b[ok]).sum()), 'of', ok.sum())
def oof(X, Em, y, groups, n_rep=10):
    valid = ~np.isnan(y); y=y[valid].astype(int); X=X[valid]; Em=Em[valid]; gr=groups[valid]
    ag, ae = [], []
    for r in range(n_rep):
        og, oe = np.full(len(y),np.nan), np.full(len(y),np.nan)
        for tr, te in GroupKFold(5, shuffle=True, random_state=42+r).split(X, groups=gr):
            med = np.nanmedian(X[tr],0); Xi = impute(X, med)
            og[te] = fit_geom(Xi[tr], y[tr])(Xi[te]); oe[te] = fit_emb(Em[tr], y[tr])(Em[te])
        ag.append(safe_auc(y, og)); ae.append(safe_auc(y, oe))
    return np.mean(ag), np.std(ag), np.mean(ae), np.std(ae), int(y.sum()), len(y)
rows=[]
for crit, cols in [('pos', CRITERION_GEOMETRY_COLS['hip_pos']), ('roi', CRITERION_GEOMETRY_COLS['hip_roi'])]:
    X = hip[cols].values.astype(float)
    for v in ['det','swapped','density','any','both']:
        y = variants[f'{crit}_{v}']
        if np.nansum(y) < 5: continue
        mg, sg, me, se, npos, n = oof(X, Eh, y, groups)
        rows.append(dict(crit=crit, labels=v, n=n, n_pos=npos, auc_geom=round(mg,3), sd_geom=round(sg,3), auc_emb=round(me,3), sd_emb=round(se,3)))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv('out/label_variant_auc.csv', index=False)
pd.DataFrame({**{'file_path':hip.file_path,'study':hip.study,'det':det,'density':dens}, **variants}).to_csv('out/label_variants.csv', index=False)
