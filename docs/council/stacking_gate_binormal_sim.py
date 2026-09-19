"""Моделирование (бинормальная модель), использованное в model-council-claude_fable_5.md.
Запуск: python3 council_fable5_sim.py  (нужны numpy, scipy, scikit-learn)."""
import numpy as np
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score, f1_score
rng=np.random.default_rng(0)
def dprime(auc): return np.sqrt(2)*norm.ppf(auc)
def sim_rank_avg(a1,a2,n=200000,prev=0.1,w=0.5):
    y=rng.random(n)<prev
    s1=rng.normal(0,1,n)+dprime(a1)*y; s2=rng.normal(0,1,n)+dprime(a2)*y
    return roc_auc_score(y,w*rankdata(s1)/n+(1-w)*rankdata(s2)/n)
for a1,a2,name in [(0.839,0.491,'sp_axis'),(0.560,0.897,'sp_art'),(0.611,0.797,'sp_pos'),(0.694,0.635,'hip_pos'),(0.902,0.877,'hip_roi')]:
    print(name, "0.5/0.5:",round(sim_rank_avg(a1,a2),3),"best-of:",max(a1,a2),"0.75 к сильному:",round(sim_rank_avg(max(a1,a2),min(a1,a2),w=0.75),3))
def f1opt(y,s):
    best=(0,None)
    for t in np.unique(s):
        f=f1_score(y,s>=t)
        if f>best[0]: best=(f,t)
    return best
for npos,auc in [(17,0.74),(10,0.72),(79,0.70)]:
    n=166 if npos<50 else 329
    d=dprime(auc); res=[];test=[];prev=[]
    yt=np.zeros(200000,bool); yt[:int(200000*npos/n)]=True; st=rng.normal(0,1,200000)+d*yt
    for rep in range(200):
        y=np.zeros(n,bool); y[:npos]=True; s=rng.normal(0,1,n)+d*y
        f,t=f1opt(y,s); res.append(f); test.append(f1_score(yt,st>=t))
        prev.append(f1_score(yt,st>=np.quantile(s,1-npos/n)))
    oracle=max(f1_score(yt,st>=t) for t in np.quantile(st,np.linspace(0.3,0.99,70)))
    print(f"{npos}/{n} AUC {auc}: OOF-opt F1 {np.mean(res):.3f} -> на тесте {np.mean(test):.3f}±{np.std(test):.3f}; prevalence-порог {np.mean(prev):.3f}; оракул {oracle:.3f}")
