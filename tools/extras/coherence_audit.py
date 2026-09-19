"""Пункт 4: study coherence auditor на 499 (таблица предупреждений по исследованиям)."""
import os, sys, warnings, hashlib; os.environ.setdefault("OMP_NUM_THREADS","1"); warnings.filterwarnings("ignore")
from pathlib import Path
from collections import Counter
import pandas as pd, pydicom
HERE=Path(__file__).resolve().parent; B=Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(HERE/"patch/src"))
from extras import study_coherence, pixel_hash
lab=pd.read_csv(B/"data/labels_for_embeddings.csv")
rows=[]
for i,r in lab.iterrows():
    ds=pydicom.dcmread(r.file_path,force=True)
    pid=str(getattr(ds,"PatientID","")); 
    rows.append({"file":Path(r.file_path).name,"study_uid":r.study,"image_uid":r.sop_instance_uid,"region":r.region,
                 "pixel_hash":pixel_hash(ds),"study_date":str(getattr(ds,"StudyDate","")),
                 "patient_hash":hashlib.sha1(pid.encode()).hexdigest()[:12] if pid else ""})
R=pd.DataFrame(rows); R.to_csv(HERE/"out/coherence_rows_499.csv",index=False)
W=study_coherence(rows)
n_st=R.study_uid.nunique(); n_w=sum(1 for v in W.values() if v)
kinds=Counter()
for v in W.values():
    for w in v: kinds[w.split(" x")[0].split(":")[0]]+=1
dup_extra=sum(int(w.split(":")[1].split()[0]) for v in W.values() for w in v if w.startswith("дубликаты"))
L=["# Study coherence auditor: 499 кадров, %d исследований\n"%n_st,
   f"Исследований с предупреждениями: **{n_w}/{n_st}**. Дубликатов кадров (лишних копий по pixel hash): {dup_extra} из 499, уникальных кадров {R.pixel_hash.nunique()}.\n",
   "| Тип предупреждения | исследований |","|---|---|"]
for k,c in kinds.most_common(): L.append(f"| {k} | {c} |")
L+=["", f"Разные StudyDate внутри одного StudyInstanceUID: {int(sum(R.groupby('study_uid').study_date.nunique()>1))} исследований; разные PatientID: {int(sum(R.groupby('study_uid').patient_hash.nunique()>1))}. "
    "Замечание: «нет бедра» (21) — исследования только позвоночника, это допустимый протокол, предупреждение информационное; повторы регионов (>1 кадра на регион) почти всегда совпадают с дубликатами экспорта (pixel hash).","",
    "| Исследование (хэш) | кадров | регионы | предупреждения |","|---|---|---|---|"]
for s,g in R.groupby("study_uid"):
    if W.get(s):
        reg=", ".join(f"{k}×{v}" for k,v in Counter(g.region).items())
        L.append(f"| {hashlib.sha1(s.encode()).hexdigest()[:8]} | {len(g)} | {reg} | {'; '.join(W[s])} |")
(HERE/"out/STUDY_COHERENCE_499.md").write_text("\n".join(L),encoding="utf-8")
print("\n".join(L[:12])); print("... total lines",len(L))
print("StudyDate values per study >1:", sum(R.groupby('study_uid').study_date.nunique()>1), "PatientID >1:", sum(R.groupby('study_uid').patient_hash.nunique()>1))
