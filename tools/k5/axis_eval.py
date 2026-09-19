"""OOF AUC признаков оси (geom logreg, GroupKFold 5 x 10, группы study+pixel_hash) и nested paired gain для sp_axis."""
import os; os.environ.setdefault('OMP_NUM_THREADS','1')
import sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
sys.path.insert(0,'tools'); sys.path.insert(0,'src')
os.environ.setdefault('NESTED_GATE_WORK', 'outputs/nested_gate')
from nested_gate import (fit_geom, fit_emb, connected_groups, impute, safe_auc, pct_rank, ref_rank, stack, choose_threshold, macro_f1, boot_ci, degenerate, W_BASE, N_OUTER, N_INNER, GAIN_MIN, GAIN_REPEATS)
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score
OUT='outputs/k5/out'
d = pd.read_csv(f'{OUT}/axis_features_spine.csv')
lab = pd.read_csv('data/labels_for_embeddings.csv'); E = np.load('data/embeddings.npy')
sm = lab.region.eq('spine').values; assert (lab.file_path.values[sm]==d.file_path.values).all(); Es = E[sm]
h = pd.read_csv(os.environ.get('PIXEL_HASHES_CSV', 'outputs/pixel_hashes.csv')); d['pixel_hash']=d.file_path.map(dict(zip(h.file_path,h.pixel_hash)))
groups = connected_groups(d.study.values, d.pixel_hash.values); y = d.sp_axis.values.astype(int); studies=d.study.values
NEW = ['axis_bodies_deg','axis_col_deg','tilt_local_max_deg','tilt_local_mean_deg','tilt_step_max_deg']
sets = {'old: axis_angle_deg': ['axis_angle_deg'], 'axis_bodies_deg': ['axis_bodies_deg'], 'axis_col_deg': ['axis_col_deg'],
        'tilt_local_max_deg': ['tilt_local_max_deg'], 'tilt_local_mean_deg':['tilt_local_mean_deg'], 'tilt_step_max_deg': ['tilt_step_max_deg'],
        'old + axis_bodies': ['axis_angle_deg','axis_bodies_deg'], 'old + axis_col': ['axis_angle_deg','axis_col_deg'],
        'old + bodies + tilt_local_max': ['axis_angle_deg','axis_bodies_deg','tilt_local_max_deg'],
        'old + all new': ['axis_angle_deg']+NEW, 'bodies + tilts (без old)': NEW}
rows=[]
for name, cols in sets.items():
    X = d[cols].values.astype(float); aucs=[]
    for r in range(10):
        o = np.full(len(y), np.nan)
        for tr, te in GroupKFold(5, shuffle=True, random_state=42+r).split(X, groups=groups):
            Xi = impute(X, np.nanmedian(X[tr],0)); o[te] = fit_geom(Xi[tr], y[tr])(Xi[te])
        aucs.append(roc_auc_score(y, o))
    raw = roc_auc_score(y, np.nan_to_num(X[:,0], nan=np.nanmedian(X[:,0]))) if len(cols)==1 else np.nan
    rows.append(dict(features=name, auc_oof_mean=round(np.mean(aucs),3), auc_oof_sd=round(np.std(aucs),3), auc_raw_single=round(raw,3) if raw==raw else None))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/axis_oof_auc.csv', index=False)

# nested paired gain: база geom=[axis_angle_deg] стэк 0.5 с emb; варианты — дополнительные признаки
variants = {'base': ['axis_angle_deg'], 'plus_bodies': ['axis_angle_deg','axis_bodies_deg'], 'plus_col': ['axis_angle_deg','axis_col_deg'],
            'plus_bodies_tiltmax': ['axis_angle_deg','axis_bodies_deg','tilt_local_max_deg'], 'plus_all': ['axis_angle_deg']+NEW}
n=len(y); S={v:np.full((10,n),np.nan) for v in variants}; P={v:np.full((10,n),np.nan) for v in variants}; rep=[]
for r in range(10):
    for k,(tr,te) in enumerate(GroupKFold(N_OUTER,shuffle=True,random_state=42+r).split(d, groups=groups)):
        if degenerate(y[tr]): continue
        Xs = {v: impute(d[c].values.astype(float), np.nanmedian(d[c].values.astype(float)[tr],0)) for v,c in variants.items()}
        og={v:np.full(len(tr),np.nan) for v in variants}; oe=np.full(len(tr),np.nan)
        for itr,iva in GroupKFold(N_INNER,shuffle=True,random_state=1000*r+k).split(tr, groups=groups[tr]):
            if degenerate(y[tr][itr]): continue
            for v in variants: og[v][iva]=fit_geom(Xs[v][tr][itr],y[tr][itr])(Xs[v][tr][iva])
            oe[iva]=fit_emb(Es[tr][itr],y[tr][itr])(Es[tr][iva])
        ok=~np.isnan(oe)&np.all([~np.isnan(og[v]) for v in variants],0)
        if ok.sum()<2 or len(np.unique(y[tr][ok]))<2: continue
        re_=np.full(len(tr),np.nan); re_[ok]=pct_rank(oe[ok]); s_in={}; thr={}
        for v in variants:
            rg=np.full(len(tr),np.nan); rg[ok]=pct_rank(og[v][ok]); s_in[v]=stack(rg,re_,W_BASE); thr[v]=choose_threshold(y[tr][ok],s_in[v][ok],int(y[tr].sum()))[0]
        pe=fit_emb(Es[tr],y[tr])(Es[te]); Re=ref_rank(pe,oe)
        for v in variants:
            pg=fit_geom(Xs[v][tr],y[tr])(Xs[v][te]); s=stack(ref_rank(pg,og[v]),Re,W_BASE); S[v][r,te]=s; P[v][r,te]=(s>=thr[v]).astype(int)
    m=~np.isnan(S['base'][r]); row={'repeat':r}
    for v in variants: row[f'auc_{v}']=safe_auc(y[m],S[v][r][m]); row[f'mf1_{v}']=macro_f1(y[m],P[v][r][m])
    rep.append(row)
rep=pd.DataFrame(rep); rep.to_csv(f'{OUT}/nested_axis_sp_axis_per_repeat.csv',index=False); dec=[]
for v in list(variants)[1:]:
    dd=rep[f'auc_{v}']-rep['auc_base']; ng=int((dd>=GAIN_MIN).sum()); acc=bool(ng>=GAIN_REPEATS and rep[f'mf1_{v}'].mean()>=rep['mf1_base'].mean()-1e-12)
    sb,sv=np.nanmean(S['base'],0),np.nanmean(S[v],0); ci=boot_ci(studies,lambda idx:safe_auc(y[idx],sv[idx])-safe_auc(y[idx],sb[idx]))
    dec.append(dict(variant=v, n=n, n_pos=int(y.sum()), n_repeats_gain_ge_0_03=ng, mean_delta_auc=round(float(dd.mean()),4), min_delta_auc=round(float(dd.min()),4), auc_base_mean=round(float(rep['auc_base'].mean()),4), auc_variant_mean=round(float(rep[f'auc_{v}'].mean()),4), mf1_base_mean=round(float(rep['mf1_base'].mean()),4), mf1_variant_mean=round(float(rep[f'mf1_{v}'].mean()),4), delta_auc_pooled_ci=[round(float(c),4) for c in ci], accepted=acc))
    print(dec[-1], flush=True)
pd.DataFrame(dec).to_csv(f'{OUT}/nested_axis_sp_axis_decisions.csv',index=False)
