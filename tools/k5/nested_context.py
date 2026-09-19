"""
К5 п.2: признаки контекста исследования для hip_pos (другое бедро того же исследования) — nested paired gain.
Протокол = tools/nested_gate.py (GroupKFold 5x10 внешн., 3 внутр., группы study+pixel_hash, база w=0.5).
Варианты (все — ДОПОЛНИТЕЛЬНЫЕ признаки, скор файла не заменяется):
  V1 geomctx : контур A = logreg на [own 5 признаков, mean(own, other), |own-other|] (15), стэк 0.5/0.5 с emb.
  V2 scorectx: уровень 2 = logreg на [s_own, s_other, max, |diff|], где s = стэк 0.5/0.5 (inner-OOF для train,
               ref-rank внешних моделей для test); other-side скор считается только из данных того же фолда.
  V3 both    : V1 + V2.
fallback: нет второго бедра -> other = own (diff = 0).
Утечки: other-side признаки для train-строк = inner-OOF предсказания (та же inner-разбивка: группы = study, значит
оба бедра всегда в одном inner-фолде); для test-строк = предсказания внешних моделей (оба бедра в test).
"""
import os; os.environ.setdefault('OMP_NUM_THREADS','1')
import sys, json, time, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
sys.path.insert(0,'tools'); sys.path.insert(0,'src')
os.environ['NESTED_GATE_WORK']='/home/user/workspace/work/B'
from nested_gate import (fit_geom, fit_emb, connected_groups, impute, safe_auc, pct_rank, ref_rank, stack, choose_threshold,
                         macro_f1, boot_ci, degenerate, W_BASE, N_OUTER, N_INNER, GAIN_MIN, GAIN_REPEATS)
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from train_stacked import CRITERION_GEOMETRY_COLS
N_REPEATS = int(os.environ.get('N_REPEATS', 10))
OUT = 'outputs/k5/out'
CRIT = os.environ.get('CRIT', 'hip_pos')

def load():
    g = pd.read_csv('data/geometry_features.csv')
    lab = pd.read_csv('data/labels_for_embeddings.csv')
    E = np.load('data/embeddings.npy')
    m = g.region.isin(['right_hip','left_hip']).values
    hip = g[m].reset_index(drop=True); Eh = E[np.nonzero(lab.region.isin(['right_hip','left_hip']).values)[0]]
    assert (lab.file_path.values[lab.region.isin(['right_hip','left_hip']).values] == hip.file_path.values).all()
    h = pd.read_csv('/home/user/workspace/work/B/results/pixel_hashes.csv'); hip['pixel_hash']=hip.file_path.map(dict(zip(h.file_path,h.pixel_hash)))
    hip['group'] = connected_groups(hip.study.values, hip.pixel_hash.values)
    return hip, Eh

def other_index(hip):
    """для каждой строки: индексы строк другой стороны того же исследования (по detected side); [] если нет."""
    out = []
    for i, r in hip.iterrows():
        m = (hip.study == r.study) & (hip.hip_side_detected != r.hip_side_detected)
        out.append(np.nonzero(m.values)[0])
    return out

def ctx_geom(X, other):
    """[own, mean(own,other), |own-other|]; other = среднее по строкам другой стороны, fallback own."""
    O = X.copy()
    for i, idx in enumerate(other):
        if len(idx): O[i] = X[idx].mean(0)
    return np.hstack([X, 0.5*(X+O), np.abs(X-O)])

def ctx_score(s, other):
    O = s.copy()
    for i, idx in enumerate(other):
        if len(idx): O[i] = np.nanmean(s[idx])
    O = np.where(np.isnan(O), s, O)
    return np.column_stack([s, O, np.maximum(s, O), np.abs(s-O)])

def fit_l2(F, y):
    sc = StandardScaler().fit(F); clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced').fit(sc.transform(F), y)
    return lambda Z: clf.predict_proba(sc.transform(Z))[:,1]

def main():
    t0=time.time(); hip, Eh = load()
    label_col = {'hip_pos':'hip_pos_c','hip_roi':'hip_roi_c'}[CRIT]
    valid = hip[label_col].notna().values
    hip = hip[valid].reset_index(drop=True); Eh = Eh[valid]
    y = hip[label_col].values.astype(int); groups = hip.group.values; studies = hip.study.values
    cols = CRITERION_GEOMETRY_COLS[CRIT]; Xraw = hip[cols].values.astype(float)
    other = other_index(hip)
    n = len(y); n_with_other = sum(len(o)>0 for o in other)
    print(f'{CRIT}: n={n} pos={y.sum()} rows with other side={n_with_other}', flush=True)
    variants = ['base','geomctx','scorectx','both','geomctx_mean','rule_max','rule_mean']
    S = {v: np.full((N_REPEATS,n), np.nan) for v in variants}; P = {v: np.full((N_REPEATS,n), np.nan) for v in variants}
    rep_rows=[]
    for r in range(N_REPEATS):
        for k,(tr,te) in enumerate(GroupKFold(N_OUTER, shuffle=True, random_state=42+r).split(Xraw, groups=groups)):
            if degenerate(y[tr]): continue
            med = np.nanmedian(Xraw[tr],0); X = impute(Xraw, med)
            Xc = ctx_geom(X, other)          # контекст по сырым признакам: без обучения -> нет утечки
            Xm = Xc[:, :2*X.shape[1]]        # own + mean(own, other) (10 признаков)
            # локальные индексы other внутри tr / te (оба бедра одной study всегда в одном множестве)
            pos_tr = {g:i for i,g in enumerate(tr)}; pos_te = {g:i for i,g in enumerate(te)}
            oth_tr = [np.array([pos_tr[j] for j in other[g] if j in pos_tr], int) for g in tr]
            oth_te = [np.array([pos_te[j] for j in other[g] if j in pos_te], int) for g in te]
            # inner-OOF на train: geom, geomctx, emb
            og, ogc, oe, ogm = (np.full(len(tr),np.nan) for _ in range(4))
            for itr, iva in GroupKFold(N_INNER, shuffle=True, random_state=1000*r+k).split(X[tr], groups=groups[tr]):
                if degenerate(y[tr][itr]): continue
                og[iva] = fit_geom(X[tr][itr], y[tr][itr])(X[tr][iva])
                ogc[iva] = fit_geom(Xc[tr][itr], y[tr][itr])(Xc[tr][iva])
                ogm[iva] = fit_geom(Xm[tr][itr], y[tr][itr])(Xm[tr][iva])
                oe[iva] = fit_emb(Eh[tr][itr], y[tr][itr])(Eh[tr][iva])
            ok = ~np.isnan(og)&~np.isnan(oe)&~np.isnan(ogc)
            if ok.sum()<2 or len(np.unique(y[tr][ok]))<2: continue
            rg, rgc, re_, rgm = (np.full(len(tr),np.nan) for _ in range(4))
            rg[ok], rgc[ok], re_[ok], rgm[ok] = pct_rank(og[ok]), pct_rank(ogc[ok]), pct_rank(oe[ok]), pct_rank(ogm[ok])
            s_in = {'base': stack(rg,re_,W_BASE), 'geomctx': stack(rgc,re_,W_BASE), 'geomctx_mean': stack(rgm,re_,W_BASE)}
            Fb = ctx_score(s_in['base'], oth_tr); s_in['rule_max'] = 0.5*Fb[:,0]+0.5*Fb[:,2]; s_in['rule_mean'] = 0.5*Fb[:,0]+0.5*Fb[:,1]
            # уровень 2 на inner-OOF стэке (train)
            F_in = ctx_score(s_in['base'], oth_tr); F_in_c = ctx_score(s_in['geomctx'], oth_tr)
            l2 = fit_l2(F_in[ok], y[tr][ok]); l2c = fit_l2(F_in_c[ok], y[tr][ok])
            s_in['scorectx'] = np.full(len(tr),np.nan); s_in['scorectx'][ok] = l2(F_in[ok])
            s_in['both'] = np.full(len(tr),np.nan); s_in['both'][ok] = l2c(F_in_c[ok])
            thr = {v: choose_threshold(y[tr][ok], s_in[v][ok], int(y[tr].sum()))[0] for v in variants}
            # внешние модели
            pg, pgc, pe, pgm = fit_geom(X[tr],y[tr])(X[te]), fit_geom(Xc[tr],y[tr])(Xc[te]), fit_emb(Eh[tr],y[tr])(Eh[te]), fit_geom(Xm[tr],y[tr])(Xm[te])
            s_te = {'base': stack(ref_rank(pg,og), ref_rank(pe,oe), W_BASE), 'geomctx': stack(ref_rank(pgc,ogc), ref_rank(pe,oe), W_BASE), 'geomctx_mean': stack(ref_rank(pgm,ogm), ref_rank(pe,oe), W_BASE)}
            Fbt = ctx_score(s_te['base'], oth_te); s_te['rule_max'] = 0.5*Fbt[:,0]+0.5*Fbt[:,2]; s_te['rule_mean'] = 0.5*Fbt[:,0]+0.5*Fbt[:,1]
            s_te['scorectx'] = l2(ctx_score(s_te['base'], oth_te)); s_te['both'] = l2c(ctx_score(s_te['geomctx'], oth_te))
            for v in variants:
                S[v][r,te] = s_te[v]; P[v][r,te] = (s_te[v] >= thr[v]).astype(int)
        m = ~np.isnan(S['base'][r])
        row = {'repeat': r}
        for v in variants:
            row[f'auc_{v}'] = safe_auc(y[m], S[v][r][m]); row[f'mf1_{v}'] = macro_f1(y[m], P[v][r][m]); row[f'f1pos_{v}'] = f1_score(y[m], P[v][r][m], zero_division=0)
        rep_rows.append(row)
        print(f"repeat {r}: " + ' | '.join(f"{v} AUC={row[f'auc_{v}']:.3f} mF1={row[f'mf1_{v}']:.3f}" for v in variants), flush=True)
    rep = pd.DataFrame(rep_rows); rep.to_csv(f'{OUT}/nested_context_{CRIT}_per_repeat.csv', index=False)
    dec=[]
    for v in variants[1:]:
        d = rep[f'auc_{v}'] - rep['auc_base']; n_gain = int((d>=GAIN_MIN).sum())
        acc = bool(n_gain>=GAIN_REPEATS and rep[f'mf1_{v}'].mean() >= rep['mf1_base'].mean()-1e-12)
        sb, sv = np.nanmean(S['base'],0), np.nanmean(S[v],0)
        ci = boot_ci(studies, lambda idx: safe_auc(y[idx], sv[idx]) - safe_auc(y[idx], sb[idx]))
        dec.append(dict(criterion=CRIT, variant=v, n=n, n_pos=int(y.sum()), n_rows_with_other=n_with_other, n_repeats_gain_ge_0_03=n_gain, mean_delta_auc=float(d.mean()), min_delta_auc=float(d.min()),
                        auc_base_mean=float(rep['auc_base'].mean()), auc_variant_mean=float(rep[f'auc_{v}'].mean()),
                        mf1_base_mean=float(rep['mf1_base'].mean()), mf1_variant_mean=float(rep[f'mf1_{v}'].mean()), delta_mf1=float((rep[f'mf1_{v}']-rep['mf1_base']).mean()),
                        f1pos_base_mean=float(rep['f1pos_base'].mean()), f1pos_variant_mean=float(rep[f'f1pos_{v}'].mean()),
                        delta_auc_pooled_ci=list(ci), accepted=acc))
        print(dec[-1], flush=True)
    pd.DataFrame(dec).to_csv(f'{OUT}/nested_context_{CRIT}_decisions.csv', index=False)
    print(f'done {time.time()-t0:.0f}s')
if __name__=='__main__': main()
