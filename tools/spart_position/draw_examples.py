"""Картинки для отчёта: кадр позвоночника, контур маски кости, плотные компоненты (зелёный — в верхних 70 %
протяжённости кости, красный — ниже). Выход: work/exp_spart/figs/*.png"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, cv2  # noqa: E402
import geometry_features as gf  # noqa: E402
OLD = "/home/user/workspace/dataset_extracted/Датасет/НД_для_обучения/Исследования/"
NEW = "/home/user/workspace/work/dataset/Исследования/"
OUTD = ROOT.parent / "figs"; OUTD.mkdir(exist_ok=True)
def draw(fp, name, cut=0.7):
    img, _ = gf.read_dicom_normalized(NEW + fp[len(OLD):]); mask = gf.segment_bone(img)
    h, w = img.shape; ys = np.nonzero(mask.any(axis=1))[0]; y0, y1 = ys.min(), ys.max(); hi = y0 + cut * (y1 - y0)
    kb = np.ones((25, 25), np.uint8); band = (cv2.dilate(mask, kb) > 0) & (mask == 0)
    st = img[band]; st = st[st > 8]; bg, sd = st.mean(), st.std() + 1e-6
    cand = ((img.astype(np.float32) > bg + 3 * sd) & (mask == 0) & (img > 8)).astype(np.uint8) * 255
    k = np.ones((3, 3), np.uint8); cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k); cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k, iterations=2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand)
    col = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(col, cnts, -1, (255, 200, 0), 1)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 4: continue
        yy = np.nonzero(lab == i)[0]; inside = ((yy >= y0) & (yy <= hi)).sum() >= 4
        col[lab == i] = (0, 200, 0) if inside else (0, 0, 255)
    cv2.line(col, (0, int(hi)), (w - 1, int(hi)), (0, 255, 255), 1)
    big = cv2.resize(col, (w * 2, int(h * 2 * 1.05 / 0.6 / 1.0 * 0.6)), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(OUTD / f"{name}.png"), big)
if __name__ == "__main__":
    g = pd.read_csv(ROOT / "data" / "geometry_features.csv"); g = g[g.region == "spine"].reset_index(drop=True)
    for fp, name in zip(sys.argv[1::2], sys.argv[2::2]):
        draw(g.file_path.iloc[int(fp)] if fp.isdigit() else fp, name)
