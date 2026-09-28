"""Диагностика без меток: где по высоте лежат плотные компоненты (центр по строкам / протяжённость маски кости)
на 166 кадрах позвоночника. Выход: outputs/spart_position/component_positions.csv (по компонентам)."""
import os, sys
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, cv2  # noqa: E402
import geometry_features as gf  # noqa: E402
import pydicom  # noqa: E402
OLD = "/home/user/workspace/dataset_extracted/Датасет/НД_для_обучения/Исследования/"
NEW = os.environ.get("DENSITO_DICOM_ROOT", "/home/user/workspace/work/dataset/Исследования/")
g = pd.read_csv(ROOT / "data" / "geometry_features.csv"); g = g[g.region == "spine"].reset_index(drop=True)
rows = []
for _, r in g.iterrows():
    img, _ = gf.read_dicom_normalized(NEW + r.file_path[len(OLD):])
    mask = gf.segment_bone(img)
    h, w = img.shape
    ys_m = np.nonzero(mask.any(axis=1))[0]; y0, y1 = ys_m.min(), ys_m.max()
    kb = np.ones((25, 25), np.uint8); band = (cv2.dilate(mask, kb) > 0) & (mask == 0)
    st = img[band]; st = st[st > 8]
    bg, sd = float(st.mean()), float(st.std()) + 1e-6
    cand = ((img.astype(np.float32) > bg + 3 * sd) & (mask == 0) & (img > 8)).astype(np.uint8) * 255
    k = np.ones((3, 3), np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k); cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k, iterations=2)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(cand)
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < 4:
            continue
        cx, cy = cent[i]
        rows.append({"file_path": r.file_path, "h": h, "w": w, "y0": y0, "y1": y1, "area_mm2": a * 0.63,
                     "cy_ratio": (cy - y0) / max(1, y1 - y0), "top_ratio": (stats[i, cv2.CC_STAT_TOP] - y0) / max(1, y1 - y0),
                     "cx_ratio": cx / w})
d = pd.DataFrame(rows); d.to_csv(ROOT / "outputs" / "spart_position" / "component_positions.csv", index=False)
print("компонент", len(d), "кадров с компонентами", d.file_path.nunique())
print("доля кадров, где маска кости занимает всю высоту кадра (y0=0, y1>=h-2):",
      float(((d.groupby('file_path').y0.first() == 0) & (d.groupby('file_path').y1.first() >= d.groupby('file_path').h.first() - 2)).mean()))
print("гистограмма центров компонент по доле высоты (0 = верх):")
hist, edges = np.histogram(d.cy_ratio, bins=np.linspace(0, 1, 11))
for c, e in zip(hist, edges): print(f"  [{e:.1f}; {e+0.1:.1f}): {c}")
print("верхний край компонент в [0.5; 0.8):", int(((d.top_ratio >= 0.5) & (d.top_ratio < 0.8)).sum()))
