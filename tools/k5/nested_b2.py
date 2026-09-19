"""К5 п.3: контур B2 (эмбеддинг патча малого вертела) как дополнительный контур для hip_pos — nested paired gain (протокол nested_gate)."""
import os; os.environ.setdefault('OMP_NUM_THREADS','1')
import sys, time, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
sys.path.insert(0,'tools'); sys.path.insert(0,'src')
os.environ['NESTED_GATE_WORK']='/home/user/workspace/work/B'
from nested_gate import (fit_geom, fit_emb, impute, safe_auc, pct_rank, ref_rank, stack, choose_threshold, macro_f1, boot_ci, degenerate, W_BASE, N_OUTER, N_INNER, GAIN_MIN, GAIN_REPEATS)
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from train_stacked import CRITERION_GEOMETRY_COLS
from nested_context import load
OUT='outputs/k5/out'; CRIT=os.environ.get('CRIT','hip_pos'); N_REPEATS=10
B2 = np.load(f'{OUT}/emb_troch_patch_hip.npy')

def main():
    t0=time.time(); hip, Eh = load(); assert len(B2)==len(hip)
    label_col={'hip_pos':'hip_pos_c','hip_roi':'hip_roi_c'}[CRIT]; valid=hip[label_col].notna().values
    hip=hip[valid].reset_index(drop=True); Eh=Eh[valid]; Eb=B2[valid]
    y=hip[label_col].values.astype(int); groups=hip.group.values; studies=hip.study.values
    Xraw=hip[CRITERION_GEOMETRY_COLS[CRIT]].values.astype(float); n=len(y)
    variants=['base','b2_third','b2_quarter','b2_only_vs_emb']
    S={v:np.full((N_REPEATS,n),np.nan) for v in variants}; P={v:np.full((N_REPEATS,n),np.nan) for v in variants}; rows=[]
    for r in range(N_REPEATS):
        for k,(tr,te) in enumerate(GroupKFold(N_OUTER,shuffle=True,random_state=42+r).split(Xraw,groups=groups)):
            if degenerate(y[tr]): continue
            X=impute(Xraw,np.nanmedian(Xraw[tr],0))
            og,oe,ob=(np.full(len(tr),np.nan) for _ in range(3))
            for itr,iva in GroupKFold(N_INNER,shuffle=True,random_state=1000*r+k).split(X[tr],groups=groups[tr]):
                if degenerate(y[tr][itr]): continue
                og[iva]=fit_geom(X[tr][itr],y[tr][itr])(X[tr][iva]); oe[iva]=fit_emb(Eh[tr][itr],y[tr][itr])(Eh[tr][iva]); ob[iva]=fit_emb(Eb[tr][itr],y[tr][itr])(Eb[tr][iva])
            ok=~np.isnan(og)&~np.isnan(oe)&~np.isnan(ob)
            if ok.sum()<2 or len(np.unique(y[tr][ok]))<2: continue
            rg,re_,rb=(np.full(len(tr),np.nan) for _ in range(3)); rg[ok],re_[ok],rb[ok]=pct_rank(og[ok]),pct_rank(oe[ok]),pct_rank(ob[ok])
            s_in={'base':stack(rg,re_,W_BASE),'b2_third':(rg+re_+rb)/3,'b2_quarter':0.5*rg+0.25*re_+0.25*rb,'b2_only_vs_emb':stack(rg,rb,W_BASE)}
            thr={v:choose_threshold(y[tr][ok],s_in[v][ok],int(y[tr].sum()))[0] for v in variants}
            pg,pe,pb=fit_geom(X[tr],y[tr])(X[te]),fit_emb(Eh[tr],y[tr])(Eh[te]),fit_emb(Eb[tr],y[tr])(Eb[te])
            Rg,Re,Rb=ref_rank(pg,og),ref_rank(pe,oe),ref_rank(pb,ob)
            s_te={'base':stack(Rg,Re,W_BASE),'b2_third':(Rg+Re+Rb)/3,'b2_quarter':0.5*Rg+0.25*Re+0.25*Rb,'b2_only_vs_emb':stack(Rg,Rb,W_BASE)}
            for v in variants: S[v][r,te]=s_te[v]; P[v][r,te]=(s_te[v]>=thr[v]).astype(int)
        m=~np.isnan(S['base'][r]); row={'repeat':r}
        for v in variants: row[f'auc_{v}']=safe_auc(y[m],S[v][r][m]); row[f'mf1_{v}']=macro_f1(y[m],P[v][r][m]); row[f'f1pos_{v}']=f1_score(y[m],P[v][r][m],zero_division=0)
        rows.append(row); print(f"repeat {r}: "+' | '.join(f"{v} AUC={row[f'auc_{v}']:.3f} mF1={row[f'mf1_{v}']:.3f}" for v in variants),flush=True)
    rep=pd.DataFrame(rows); rep.to_csv(f'{OUT}/nested_b2_{CRIT}_per_repeat.csv',index=False); dec=[]
    for v in variants[1:]:
        d=rep[f'auc_{v}']-rep['auc_base']; ng=int((d>=GAIN_MIN).sum()); acc=bool(ng>=GAIN_REPEATS and rep[f'mf1_{v}'].mean()>=rep['mf1_base'].mean()-1e-12)
        sb,sv=np.nanmean(S['base'],0),np.nanmean(S[v],0); ci=boot_ci(studies,lambda idx:safe_auc(y[idx],sv[idx])-safe_auc(y[idx],sb[idx]))
        dec.append(dict(criterion=CRIT,variant=v,n=n,n_pos=int(y.sum()),n_repeats_gain_ge_0_03=ng,mean_delta_auc=round(float(d.mean()),4),min_delta_auc=round(float(d.min()),4),auc_base_mean=round(float(rep['auc_base'].mean()),4),auc_variant_mean=round(float(rep[f'auc_{v}'].mean()),4),mf1_base_mean=round(float(rep['mf1_base'].mean()),4),mf1_variant_mean=round(float(rep[f'mf1_{v}'].mean()),4),delta_auc_pooled_ci=[round(float(c),4) for c in ci],accepted=acc))
        print(dec[-1],flush=True)
    pd.DataFrame(dec).to_csv(f'{OUT}/nested_b2_{CRIT}_decisions.csv',index=False); print(f'done {time.time()-t0:.0f}s')
if __name__=='__main__': main()
