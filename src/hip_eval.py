"""
Быстрая диагностика признаков бедра (sanity-check без CV + честный GroupKFold).

Метки берутся из разметка.xlsx ПО СТОРОНЕ, ОПРЕДЕЛЁННОЙ ПО ИЗОБРАЖЕНИЮ
(hip_features.hip_side_score), а не по колонке region старой версии labels_full.csv
(data/labels_full_v1_density_side.csv; с 24.09 labels_full.csv перегенерирован детектором):
старая эвристика стороны (плотность по половинам) ошибалась на ~24% снимков.
Смена стороны меняет метку только там, где у исследования rh_* != lh_*
(~4% снимков), поэтому это не главный источник низкого AUC, но корректность
стороны критична для текста violation_type ("...правого/левого бедра").

Использование:  python3 hip_eval.py [--recompute]
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from geometry_features import read_dicom_normalized
from hip_features import hip_all_features
from build_dataset import load_labels

DATA = Path("/home/user/workspace/densito_rebuild/data")
CACHE = Path("/tmp/hip_feats_cache.csv")


def compute_hip_table(recompute=False):
    if CACHE.exists() and not recompute:
        return pd.read_csv(CACHE)
    xl = load_labels()
    df = pd.read_csv(DATA / 'labels_full_v1_density_side.csv')  # pos_old — сторона старой эвристики
    h = df[df['region'].isin(['right_hip', 'left_hip'])].copy()
    rows = []
    for _, r in h.iterrows():
        img, _ = read_dicom_normalized(r['file_path'])
        f = hip_all_features(img)
        side = f['hip_side_detected']
        lab = xl.loc[str(r['study'])]
        f['pos_c'] = lab['rh_pos'] if side == 'right' else lab['lh_pos']
        f['roi_c'] = lab['rh_roi'] if side == 'right' else lab['lh_roi']
        f['pos_old'] = r['rh_pos'] if r['region'] == 'right_hip' else r['lh_pos']
        f['roi_old'] = r['rh_roi'] if r['region'] == 'right_hip' else r['lh_roi']
        f['study'] = r['study']; f['file_path'] = r['file_path']; f['region_old'] = r['region']
        rows.append(f)
    F = pd.DataFrame(rows)
    F.to_csv(CACHE, index=False)
    return F


def oof_auc(F, label, cols, n_pca=None, C=1.0, repeats=3):
    d = F[F[label].notna()].reset_index(drop=True)
    y = d[label].astype(int).values
    X = d[cols].astype(float).values
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), med, X)
    g = d['study'].values
    oof = np.zeros((repeats, len(y)))
    for rep in range(repeats):
        rng = np.random.default_rng(rep)
        perm = rng.permutation(len(y))
        for tr, va in GroupKFold(5).split(X[perm], groups=g[perm]):
            tr, va = perm[tr], perm[va]
            sc = StandardScaler().fit(X[tr])
            Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
            if n_pca:
                p = PCA(n_components=min(n_pca, Xtr.shape[1])).fit(Xtr)
                Xtr, Xva = p.transform(Xtr), p.transform(Xva)
            clf = LogisticRegression(C=C, class_weight='balanced', max_iter=2000).fit(Xtr, y[tr])
            oof[rep, va] = clf.predict_proba(Xva)[:, 1]
    return roc_auc_score(y, oof.mean(0)), int(y.sum()), len(y)


if __name__ == '__main__':
    F = compute_hip_table(recompute='--recompute' in sys.argv)
    scalar = ['signed_shaft_angle_deg', 'abs_shaft_angle_deg', 'shaft_width_mm', 'shaft_len_below_troch_mm',
              'lesser_troch_prominence_mm', 'greater_troch_offset_mm', 'lateral_margin_mm',
              'medial_neck_extent_mm', 'merge_height_mm', 'neck_min_width_mm', 'scan_length_mm',
              'shaft_bottom_center_ratio', 'bone_area_ratio', 'femur_fill_ratio',
              'bone_top_touch_ratio', 'bone_bottom_touch_ratio']
    prof = [c for c in F.columns if c.startswith('prof_')]
    for lab in ['pos_c', 'roi_c']:
        d = F[F[lab].notna()]
        print(f"\n== {lab}: n={len(d)} n_pos={int(d[lab].sum())}  (single-feature AUC, no CV)")
        for c in scalar:
            x = d[c].astype(float).fillna(d[c].median())
            if x.nunique() > 1:
                print(f"  {c:30s} {roc_auc_score(d[lab], x):.3f}")
        for side in ['right', 'left']:
            ds = d[d.hip_side_detected == side]
            print(f"  -- side={side} n={len(ds)} n_pos={int(ds[lab].sum())}: "
                  + ", ".join(f"{c}={roc_auc_score(ds[lab], ds[c].astype(float).fillna(ds[c].median())):.2f}"
                              for c in ['signed_shaft_angle_deg', 'abs_shaft_angle_deg', 'scan_length_mm', 'medial_neck_extent_mm']))
    print("\n== OOF GroupKFold AUC (LR, balanced) ==")
    for name, cols, npca in [
        ('pos: profile PCA6', prof, 6),
        ('pos: profile PCA10', prof, 10),
        ('pos: profile+scalars PCA8', prof + scalar, 8),
        ('pos: scalars', scalar, None),
        ('pos: angle+neck+troch', ['signed_shaft_angle_deg', 'medial_neck_extent_mm', 'lesser_troch_prominence_mm', 'neck_min_width_mm'], None),
    ]:
        auc, npos, n = oof_auc(F, 'pos_c', cols, npca)
        print(f"  {name:32s} AUC={auc:.3f} (n_pos={npos}/{n})")
    for name, cols in [
        ('roi: scan_length', ['scan_length_mm']),
        ('roi: scan_length+shaft_len', ['scan_length_mm', 'shaft_len_below_troch_mm']),
        ('roi: scan_length+shaft_len+area', ['scan_length_mm', 'shaft_len_below_troch_mm', 'bone_area_ratio']),
    ]:
        auc, npos, n = oof_auc(F, 'roi_c', cols, None)
        print(f"  {name:32s} AUC={auc:.3f} (n_pos={npos}/{n})")
