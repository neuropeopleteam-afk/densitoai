"""Боковое ограничение зоны sp_art (PREREG_spart_lateral.md): признаки lat<m> для кадров позвоночника.
Кандидаты — ровно как geometry_features.foreign_object_features; полоса — как _position_features (band70).
Запуск в образе densitoai:2.5.0: python /w/lat_extract.py  (DICOM в /ds, таблица /w/geometry_features.csv)."""
import sys, os, zipfile, glob
sys.path.insert(0, "/app/src")
import numpy as np, pandas as pd, cv2
import geometry_features as gf

OLD = "/home/user/workspace/dataset_extracted/Датасет/НД_для_обучения/Исследования/"
MS = (5, 10, 20, 40)
C = 0.7


def comps(img_u8, mask):
    kernel_big = np.ones((25, 25), np.uint8)
    bone_dilated = cv2.dilate(mask, kernel_big, iterations=1)
    stb = ((bone_dilated > 0) & (mask == 0)).astype(np.uint8)
    st = img_u8[stb > 0]; body_thresh = 8; st = st[st > body_thresh]
    if len(st) < 20:
        return None, [], {}
    bg_mean = float(np.mean(st)); bg_std = float(np.std(st)) + 1e-6
    cand = ((img_u8.astype(np.float32) > bg_mean + 3.0 * bg_std) & (mask == 0) & (img_u8 > body_thresh)).astype(np.uint8) * 255
    k = np.ones((3, 3), np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k, iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand)
    kept, gaps = [], {}
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 4:
            continue
        kept.append(i); gaps[i] = float((img_u8[labels == i].mean() - bg_mean) / bg_std)
    return labels, kept, gaps


def feats(path):
    img, _ = gf.read_dicom_normalized(path)
    mask = gf.segment_bone(img)
    px = gf.PIXEL_SPACING_X_MM * gf.PIXEL_SPACING_Y_MM
    out = {"chk_band70_area_mm2": 0.0}
    for m in MS:
        out.update({f"lat{m}_area_mm2": 0.0, f"lat{m}_area_log": 0.0, f"lat{m}_max_gap": 0.0, f"lat{m}_n": 0})
    labels, kept, gaps = comps(img, mask)
    has = mask.any(axis=1)
    rows = np.nonzero(has)[0]
    if labels is None or not kept or len(rows) == 0:
        return out
    y0, y1 = float(rows.min()), float(rows.max()); hi = y0 + C * (y1 - y0)
    H = mask.shape[0]
    xl = np.full(H, -1e9); xr = np.full(H, -1e9)
    for y in rows:
        xs = np.nonzero(mask[y])[0]; xl[y] = xs.min(); xr[y] = xs.max()
    b70 = 0; acc = {m: [0, 0, 0.0] for m in MS}
    for i in kept:
        ys, xs = np.nonzero(labels == i)
        inb = (ys >= y0) & (ys <= hi)
        if int(inb.sum()) >= 4:
            b70 += int(inb.sum())
        for m in MS:
            mp = m / gf.PIXEL_SPACING_X_MM
            ok = inb & has[ys] & (xs >= xl[ys] - mp) & (xs <= xr[ys] + mp)
            c = int(ok.sum())
            if c >= 4:
                acc[m][0] += c; acc[m][1] += 1; acc[m][2] = max(acc[m][2], gaps[i])
    out["chk_band70_area_mm2"] = float(b70 * px)
    for m in MS:
        a = acc[m][0] * px
        out.update({f"lat{m}_area_mm2": float(a), f"lat{m}_area_log": float(np.log1p(a)),
                    f"lat{m}_max_gap": float(acc[m][2]), f"lat{m}_n": int(acc[m][1])})
    return out


g = pd.read_csv("/w/geometry_features.csv")
g = g[g.region == "spine"].reset_index(drop=True)
res = []
for _, r in g.iterrows():
    f = feats("/ds/" + r.file_path[len(OLD):]); f["file_path"] = r.file_path; res.append(f)
d = pd.DataFrame(res)
d.to_csv("/w/out/lat_features.csv", index=False)
diff = np.abs(d["chk_band70_area_mm2"].values - g["metal_metal_band70_area_mm2"].values)
print("кадров", len(d), "паритет band70: макс |Δ| мм²", float(diff.max()), "совпало", int((diff < 1e-6).sum()))
# образец организаторов — только для отчёта после решения
os.makedirs("/tmp/s", exist_ok=True)
zipfile.ZipFile("/in/test_data.zip").extractall("/tmp/s")
for p in sorted(glob.glob("/tmp/s/**/*.dcm", recursive=True)):
    if "ПОП" in p or "\u041f\u041e\u041f" in p or p.endswith("ПОП.dcm"):
        pass
rows = []
for p in sorted(glob.glob("/tmp/s/**/*.dcm", recursive=True)):
    f = feats(p); f["file_path"] = os.path.basename(p); rows.append(f)
pd.DataFrame(rows).to_csv("/w/out/lat_sample.csv", index=False)
print("образец:", [(x["file_path"], round(x["chk_band70_area_mm2"], 1)) for x in rows])
