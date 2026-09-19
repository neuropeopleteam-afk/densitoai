"""
Ориентиры/линии, которые РЕАЛЬНО выдаёт существующая геометрия проекта
(densito_rebuild/src/geometry_features.py и hip_features.py), в координатах
исходного изображения (пиксели, y вниз). Никакой новой сегментации — только
повторение внутренних шагов кода, чтобы получить положение точек, которые сам
код агрегирует в скалярные признаки.

Позвоночник (geometry_features.segment_bone + spine_axis_features):
  * centerline: центроид маски в каждой строке -> x_c(y)  (массивы centerline_x/_y);
  * axis line: линейная аппроксимация x = a*y + b -> axis_angle_deg (модуль угла к вертикали, в мм);
  * body edges: крайние ненулевые пиксели маски в строке y (min/max) — это край МАСКИ кости
    (Otsu, крупнейшая компонента), а не анатомический край тела позвонка.
  Код НЕ находит отдельные позвонки (L1/L4): «верхняя/нижняя точка оси» сопоставляются
  с линией оси по горизонтали в той же строке, а не как точки.

Бедро (hip_features.segment_bone_hip + hip_side_score + track_femur + _fit_shaft,
       далее шаги hip_features_canonical):
  * shaft axis: x = a*y + b — среднее двух краевых прямых диафиза (в канонической ориентации
    «правое бедро», для левого кадр зеркалится по x);
  * gt_point: строка максимального латерального отклонения контура от прямой диафиза выше
    диафиза (то, из чего берётся greater_troch_offset_mm) — это точка максимального
    выступа большого вертела ВБОК, а не его верхушка;
  * lt_point: вершина первого пика медиального отклонения (lesser_troch_prominence_mm) —
    соответствует «малому вертелу» на медиальном контуре;
  * head_center: в коде НЕТ (центр головки не вычисляется) -> None.
"""
import os, sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

B = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(B / "src"))
import geometry_features as gf  # noqa: E402
import hip_features as hf  # noqa: E402

SX, SY = hf.PIXEL_SPACING_X_MM, hf.PIXEL_SPACING_Y_MM  # 0.6, 1.05


def spine_code_geometry(img_u8):
    mask = gf.segment_bone(img_u8)
    feats = gf.spine_axis_features(img_u8, mask)
    out = {"ok": feats["axis_angle_deg"] is not None, "axis_angle_deg": feats["axis_angle_deg"],
           "mask": mask, "a": None, "b": None, "signed_angle_deg": None}
    if not out["ok"]:
        return out
    ys, xs = feats["centerline_y"], feats["centerline_x"]
    a, b = np.polyfit(ys, xs, 1)
    out.update(a=float(a), b=float(b), centerline_y=ys, centerline_x=xs,
               signed_angle_deg=float(np.degrees(np.arctan2(a * SX, SY))))
    return out


def spine_axis_x_at(geom, y):
    """x линии оси кода в строке y (пиксели исходного кадра)."""
    return geom["a"] * y + geom["b"]


def spine_mask_edges_at(geom, y):
    """Левый/правый крайний пиксель маски кости в строке y; None, если строка пустая."""
    h = geom["mask"].shape[0]
    yi = int(round(y))
    if not (0 <= yi < h):
        return None, None
    nz = np.nonzero(geom["mask"][yi])[0]
    if len(nz) < 4:
        return None, None
    return float(nz.min()), float(nz.max())


def hip_code_geometry(img_u8):
    """Повторяет шаги hip_features_canonical до точки, где получаются координаты."""
    mask = hf.segment_bone_hip(img_u8)
    side = "right" if hf.hip_side_score(img_u8, mask) >= 0 else "left"
    mask_c = np.ascontiguousarray(mask if side == "right" else mask[:, ::-1])
    h, w = mask_c.shape
    out = {"ok": False, "side": side, "w": w, "a": None, "b": None, "gt_point": None, "lt_point": None,
           "head_center": None, "shaft_rows": None}
    tr = hf.track_femur(mask_c)
    if tr is None or len(tr["y"]) < 15:
        return out
    y, L, R, W = tr["y"], tr["left"], tr["right"], tr["width"]
    n = len(y)
    shaft_top, (al, bl), (ar, br) = hf._fit_shaft(y, L, R, SX, SY)
    a, b = 0.5 * (al + ar), 0.5 * (bl + br)
    w0 = float(np.median(W[:shaft_top]))
    yf = y.astype(float)
    dev_lat = (np.polyval([al, bl], yf) - L) * SX
    dev_med = (R - np.polyval([ar, br], yf)) * SX
    jump = np.nonzero(np.diff(R) * SX > 15.0)[0] + 1
    wide = np.nonzero(W > 3.0 * w0)[0]
    cands = [i for i in list(jump) + list(wide) if i > shaft_top]
    end = min(cands) if cands else n
    seg = slice(shaft_top, end)
    dm, dl = dev_med[seg], dev_lat[seg]

    def to_orig(xc, yc):
        return (float(w - 1 - xc) if side == "left" else float(xc)), float(yc)

    gt = lt = None
    if len(dm) >= 3:
        k = shaft_top + int(np.argmax(dl))
        gt = to_orig(L[k], y[k])
        dm_s = np.convolve(np.pad(dm, 2, mode="edge"), np.ones(5) / 5, mode="valid")
        peaks, props = find_peaks(dm_s, prominence=1.0)
        for pk in peaks:
            if dm_s[pk] > 1.5:
                kk = shaft_top + int(pk)
                lt = to_orig(R[kk], y[kk])
                break
    out.update(ok=True, a=float(a), b=float(b), gt_point=gt, lt_point=lt,
               shaft_rows=(int(y[shaft_top - 1]), int(y[0])), shaft_top_idx=int(shaft_top),
               signed_shaft_angle_deg=float(np.degrees(np.arctan2(a * SX, SY))))
    return out


def hip_shaft_x_at(geom, y):
    """x оси диафиза кода в строке y, в координатах ИСХОДНОГО кадра."""
    xc = geom["a"] * y + geom["b"]
    return float(geom["w"] - 1 - xc) if geom["side"] == "left" else float(xc)


def hip_shaft_signed_angle_orig(geom):
    """Знаковый угол оси диафиза (мм) в исходной ориентации: >0 — низ смещён вправо на экране."""
    ang = geom["signed_shaft_angle_deg"]
    return -ang if geom["side"] == "left" else ang
