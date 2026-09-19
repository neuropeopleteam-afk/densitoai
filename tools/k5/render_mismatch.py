import os
import sys, os, numpy as np, pandas as pd, cv2
sys.path.insert(0, 'src')
from geometry_features import read_dicom_normalized
from hip_features import segment_bone_hip, hip_side_score, track_femur, hip_features_canonical
hip = pd.read_csv('out/hip_side_table.csv')
u = hip.drop_duplicates('pixel_hash')
mm = u[u.mismatch].sort_values('hip_side_score', key=abs)
sel = pd.concat([mm.head(6), mm.tail(4), u[~u.mismatch].sample(6, random_state=0)])
tiles=[]
for i,(_,r) in enumerate(sel.iterrows()):
    img,_ = read_dicom_normalized(r.file_path)
    col = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mask = segment_bone_hip(img)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(col, cnts, -1, (0,0,255), 1)
    col = cv2.resize(col, (280, int(col.shape[0]*280/col.shape[1])))
    canvas = np.zeros((360, 280, 3), np.uint8); canvas[:min(360,col.shape[0])] = col[:360]
    tag = ('MISMATCH ' if r.mismatch else 'ok ')
    cv2.putText(canvas, f"{i} {tag}", (3,345), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
    cv2.putText(canvas, f"lab={r.lab_side} det={r.hip_side_detected} s={r.hip_side_score:+.2f}", (3,357), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,255,0), 1)
    tiles.append(canvas)
rows=[np.hstack(tiles[i:i+4]) for i in range(0,16,4)]
grid=np.vstack(rows)
cv2.imwrite('out/img/side_mismatch_grid.png', grid)
sel[['study','instance_number','lab_side','hip_side_detected','hip_side_score','rows','cols','file_path']].to_csv('out/side_mismatch_selected.csv', index=False)
print(sel[['study','instance_number','lab_side','hip_side_detected','hip_side_score','rows']].to_string())
