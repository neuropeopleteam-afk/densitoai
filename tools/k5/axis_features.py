"""
К5 п.4: ось позвоночника по определению постановщика — линия через центры верхнего и нижнего тел
позвонков (L1 и L4) и локальные наклоны тел; альтернатива axis_angle_deg (линейная аппроксимация
центроидов ВСЕХ строк маски, включая рёбра и гребни таза, geometry_features.spine_axis_features).

Алгоритм (без обучения, только правило):
 1. маска кости = geometry_features.segment_bone; по строкам центроид x и ширина w;
 2. «столб» позвоночника: строки с шириной в [0.5; 1.6] * медианной ширины (отсекает рёбра/таз);
 3. профиль яркости вдоль оси: среднее по полосе +-w/4 вокруг центроида, сглаживание 5 строк;
    межпозвонковые диски = локальные минимумы профиля (расстояние >= 22 мм, prominence >= 5 ед.);
 4. тела позвонков = отрезки между соседними дисками длиной 15..45 мм; полные тела — не касающиеся
    краёв столба; центр тела = (медиана центроидов, середина по y);
 5. признаки:
    axis_bodies_deg      угол (к вертикали, мм) линии через центры верхнего и нижнего полных тел;
    axis_col_deg         угол линейной аппроксимации центроидов только по строкам столба (без рёбер/таза);
    tilt_local_max_deg   максимум |угла| линейной аппроксимации центроидов внутри одного тела;
    tilt_local_mean_deg  среднее |локального угла|;
    tilt_step_max_deg    максимум |угла| между центрами соседних тел;
    n_bodies             число полных тел (контроль качества; при < 2 — axis_bodies_deg = axis_col_deg).
Пиксель: X 0.6 мм, Y 1.05 мм (как в hip_features).
"""
from __future__ import annotations
import os
import sys
import numpy as np
import cv2
from scipy.signal import find_peaks

sys.path.insert(0, 'src')
from geometry_features import read_dicom_normalized, segment_bone  # noqa: E402

SX, SY = 0.6, 1.05           # мм/пиксель
DISC_MIN_DIST_MM = 22.0
BODY_MIN_MM, BODY_MAX_MM = 15.0, 45.0
FEATURE_NAMES = ['axis_bodies_deg', 'axis_col_deg', 'tilt_local_max_deg', 'tilt_local_mean_deg',
                 'tilt_step_max_deg', 'n_bodies']


def _angle_deg(dx_px: float, dy_px: float) -> float:
    return float(np.degrees(np.arctan2(abs(dx_px * SX), abs(dy_px * SY)))) if dy_px != 0 else 0.0


def _fit_angle(ys: np.ndarray, xs: np.ndarray) -> float:
    if len(ys) < 4:
        return np.nan
    a, _ = np.polyfit(ys.astype(float), xs.astype(float), 1)
    return _angle_deg(a, 1.0)


def spine_axis_by_bodies(img_u8: np.ndarray, mask: np.ndarray | None = None) -> dict:
    if mask is None:
        mask = segment_bone(img_u8)
    h, w = mask.shape
    rows = [(y, xs.mean(), xs.min(), xs.max()) for y in range(h) for xs in [np.nonzero(mask[y])[0]] if len(xs) > 3]
    out = {k: np.nan for k in FEATURE_NAMES}
    out['bodies'] = []
    out['col_rows'] = None
    if len(rows) < 20:
        return out
    R = np.array(rows, float)
    ys, cx, x0, x1 = R[:, 0], R[:, 1], R[:, 2], R[:, 3]
    width = x1 - x0
    wmed = np.median(width)
    col = (width >= 0.5 * wmed) & (width <= 1.6 * wmed)
    # оставляем самый длинный непрерывный участок столба
    idx = np.nonzero(col)[0]
    if len(idx) < 20:
        return out
    breaks = np.nonzero(np.diff(idx) > 3)[0]
    segs = np.split(idx, breaks + 1)
    seg = max(segs, key=len)
    ys_c, cx_c, x0_c, x1_c, w_c = ys[seg], cx[seg], x0[seg], x1[seg], width[seg]
    out['col_rows'] = (int(ys_c[0]), int(ys_c[-1]))
    out['axis_col_deg'] = _fit_angle(ys_c, cx_c)
    # профиль яркости вдоль оси (полоса +-w/4 вокруг центроида)
    prof = np.array([img_u8[int(y), max(0, int(c - wd / 4)):min(w, int(c + wd / 4) + 1)].mean()
                     for y, c, wd in zip(ys_c, cx_c, w_c)])
    k = 5
    prof_s = np.convolve(np.pad(prof, k // 2, mode='edge'), np.ones(k) / k, mode='valid')
    dist = max(3, int(DISC_MIN_DIST_MM / SY))
    mins, _ = find_peaks(-prof_s, distance=dist, prominence=5.0)
    # границы тел: диски + края столба
    bounds = np.concatenate([[0], mins, [len(ys_c) - 1]]).astype(int)
    bodies = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        length_mm = (ys_c[b] - ys_c[a]) * SY
        full = (i > 0) and (i < len(bounds) - 2)
        if BODY_MIN_MM <= length_mm <= BODY_MAX_MM and b - a >= 4:
            yy, xx = ys_c[a:b + 1], cx_c[a:b + 1]
            bodies.append(dict(y_top=float(ys_c[a]), y_bot=float(ys_c[b]), yc=float((ys_c[a] + ys_c[b]) / 2),
                               xc=float(np.median(xx)), tilt=_fit_angle(yy, xx), full=full, len_mm=float(length_mm)))
    fulls = [bd for bd in bodies if bd['full']]
    out['bodies'] = bodies
    out['n_bodies'] = float(len(fulls))
    if len(fulls) >= 2:
        top, bot = fulls[0], fulls[-1]
        out['axis_bodies_deg'] = _angle_deg(bot['xc'] - top['xc'], bot['yc'] - top['yc'])
        tilts = np.array([bd['tilt'] for bd in fulls if not np.isnan(bd['tilt'])])
        if len(tilts):
            out['tilt_local_max_deg'] = float(np.max(np.abs(tilts)))
            out['tilt_local_mean_deg'] = float(np.mean(np.abs(tilts)))
        steps = [_angle_deg(fulls[i + 1]['xc'] - fulls[i]['xc'], fulls[i + 1]['yc'] - fulls[i]['yc']) for i in range(len(fulls) - 1)]
        out['tilt_step_max_deg'] = float(np.max(steps)) if steps else np.nan
    else:
        out['axis_bodies_deg'] = out['axis_col_deg']
        out['tilt_local_max_deg'] = out['tilt_local_mean_deg'] = out['tilt_step_max_deg'] = out['axis_col_deg']
    return out


def draw_axis(img_u8: np.ndarray, res: dict, scale: int = 2) -> np.ndarray:
    t = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    t = cv2.resize(t, (img_u8.shape[1] * scale, img_u8.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    for bd in res.get('bodies', []):
        color = (0, 255, 0) if bd['full'] else (0, 128, 255)
        cv2.circle(t, (int(bd['xc'] * scale), int(bd['yc'] * scale)), 4, color, -1)
        cv2.line(t, (0, int(bd['y_top'] * scale)), (t.shape[1] - 1, int(bd['y_top'] * scale)), (80, 80, 80), 1)
    fulls = [bd for bd in res.get('bodies', []) if bd['full']]
    if len(fulls) >= 2:
        cv2.line(t, (int(fulls[0]['xc'] * scale), int(fulls[0]['yc'] * scale)),
                 (int(fulls[-1]['xc'] * scale), int(fulls[-1]['yc'] * scale)), (0, 0, 255), 2)
    cr = res.get('col_rows')
    if cr:
        for y in cr:
            cv2.line(t, (0, y * scale), (t.shape[1] - 1, y * scale), (255, 0, 0), 1)
    return t


def compute_for_file(path: str) -> dict:
    img, _ = read_dicom_normalized(path)
    res = spine_axis_by_bodies(img)
    return {k: res[k] for k in FEATURE_NAMES}


if __name__ == '__main__':
    import pandas as pd
    g = pd.read_csv('data/geometry_features.csv')
    sp = g[g.region == 'spine'].reset_index(drop=True)
    rows = []
    for _, r in sp.iterrows():
        f = compute_for_file(r.file_path)
        f.update(file_path=r.file_path, study=r.study, sp_axis=r.sp_axis, axis_angle_deg=r.axis_angle_deg, curvature=r.curvature)
        rows.append(f)
    df = pd.DataFrame(rows)
    df.to_csv('outputs/k5/out/axis_features_spine.csv', index=False)
    print(df[FEATURE_NAMES].describe().round(2).T)
    print(df.groupby('sp_axis')[FEATURE_NAMES].mean().round(2))
