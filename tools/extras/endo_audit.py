"""Пункт 3: правило «эндопротез» на 499 кадрах (метрики, срабатывания, миниатюры)."""
import os, sys, warnings; os.environ.setdefault("OMP_NUM_THREADS","1"); warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, pydicom, cv2
HERE=Path(__file__).resolve().parent; B=Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(HERE/"patch/src")); sys.path.insert(0,str(B/"src"))
from extras import endoprosthesis, ENDO_DEFAULTS
from inference import normalize_pixels
lab=pd.read_csv(B/"data/labels_for_embeddings.csv")
rows=[]; imgs={}
for i,r in lab.iterrows():
    img=normalize_pixels(pydicom.dcmread(r.file_path,force=True)); imgs[i]=img
    m=endoprosthesis(img, r.region); m.update(i=i, region=r.region, study=r.study); rows.append(m)
D=pd.DataFrame(rows); D.to_csv(HERE/"out/endo_per_frame.csv",index=False)
H=D[D.region!="spine"]
print("hips",len(H),"flagged",H.endoprosthesis_suspected.sum(), D[D.endoprosthesis_suspected][["i","region","study","dense_area_px","max_half_thick_px","extent_rows"]].to_string())
for c in ("dense_area_px","max_half_thick_px","extent_rows"):
    print(c, "hip quantiles 50/90/99/99.5/max:", np.round(np.quantile(H[c],[.5,.9,.99,.995,1]),2).tolist())
# top-12 by half-thickness among hips -> thumbnails
top=H.sort_values(["max_half_thick_px","dense_area_px"],ascending=False).head(12)
tiles=[]
for r in top.itertuples():
    img=imgs[r.i]; dense=(img>=ENDO_DEFAULTS["dense_level"]).astype(np.uint8)
    vis=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR); vis[dense>0]=(0,0,255)
    vis=cv2.resize(vis,(200,int(200*img.shape[0]/img.shape[1])))
    canvas=np.zeros((260,200,3),np.uint8); canvas[:min(260,vis.shape[0])]=vis[:260]
    txt=f"#{r.i} {r.region[:1].upper()} a={r.dense_area_px} t={r.max_half_thick_px} e={r.extent_rows} {'FLAG' if r.endoprosthesis_suspected else ''}"
    cv2.putText(canvas,txt,(2,255),cv2.FONT_HERSHEY_SIMPLEX,0.32,(0,255,0),1)
    tiles.append(canvas)
grid=np.vstack([np.hstack(tiles[k:k+6]) for k in range(0,12,6)])
cv2.imwrite(str(HERE/"out/img/endo_top12_hips.png"),grid)
print("saved", HERE/"out/img/endo_top12_hips.png")
