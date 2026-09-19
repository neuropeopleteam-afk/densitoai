import os; os.environ['OMP_NUM_THREADS']='1'
import sys, numpy as np, pandas as pd, cv2, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'outputs/k5')
from axis_features import spine_axis_by_bodies, draw_axis, read_dicom_normalized, FEATURE_NAMES
g=pd.read_csv('data/geometry_features.csv'); sp=g[g.region=='spine']
sel = pd.concat([sp[sp.sp_axis==1], sp[sp.sp_axis==0].sort_values('axis_angle_deg', ascending=False).head(6)])
tiles=[]; rows=[]
for _, r in sel.iterrows():
    img,_=read_dicom_normalized(r.file_path); res=spine_axis_by_bodies(img); t=draw_axis(img,res,2)
    txt=f"ax={int(r.sp_axis)} old={r.axis_angle_deg:.1f} new={res['axis_bodies_deg']:.1f} nb={int(res['n_bodies'])}"
    cv2.putText(t,txt,(3,14),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,255),1); tiles.append(cv2.resize(t,(300,320)))
    rows.append(dict(study=r.study[-8:], file=os.path.basename(r.file_path), sp_axis=int(r.sp_axis), axis_angle_deg=round(r.axis_angle_deg,2), **{k:(round(res[k],2) if res[k]==res[k] else None) for k in FEATURE_NAMES}))
pd.DataFrame(rows).to_csv('out/axis_positives_table.csv', index=False)
n=len(tiles); cols=6; 
while len(tiles)%cols: tiles.append(np.zeros_like(tiles[0]))
grid=np.vstack([np.hstack(tiles[i:i+cols]) for i in range(0,len(tiles),cols)]); cv2.imwrite('out/img/sp_axis_positives_axis.png', grid)
print(pd.DataFrame(rows).to_string())
