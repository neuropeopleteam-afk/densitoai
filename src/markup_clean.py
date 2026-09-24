# -*- coding: utf-8 -*-
"""Очистка впечатанной разметки денситометра (контуры ROI, линии межпозвонковых промежутков) до инференса.

Основание: разъяснение организаторов 24.09.2026 — «на части снимков разметка денситометра есть прямо на изображении,
но сохраняется она не всегда, поэтому оценивать её не требуется». Стресс-тест `tools/markup_stress.py`: на 499
кадрах заказчика с синтетической разметкой без очистки класс меняется у 29 % строк.

Детектор: тонкая (1–2 px) прямая горизонтальная или вертикальная линия почти постоянной яркости, заметно ярче
соседей с обеих сторон, длиной не меньше MIN_RUN px. Очистка включается, только если таких отрезков не меньше
MIN_SEGMENTS: на 499 кадрах заказчика максимум — 3 отрезка (края маски поля и импланты), поэтому исходные кадры
не меняются ни одним пикселем. Найденные пиксели закрашиваются средним ближайших незатронутых соседей поперёк линии.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

MIN_RUN = 25          # минимальная длина отрезка, px
CONTRAST = 40         # линия ярче соседей по обе стороны не меньше чем на столько уровней (8 бит)
FLAT = 2              # допуск постоянства яркости вдоль линии
MIN_SEGMENTS = 4      # порог включения очистки (на 499 кадрах заказчика максимум 3)


def _runs_mask(arr: np.ndarray) -> Tuple[np.ndarray, int]:
    """Маска горизонтальных тонких линий и число отрезков длиной >= MIN_RUN (1 и 2 px толщины)."""
    a = arr.astype(np.int16)
    h, w = a.shape
    mask = np.zeros((h, w), dtype=bool)
    n_seg = 0
    for thick in (1, 2):
        if h < thick + 2:
            continue
        mid = a[1:h - thick] if thick == 1 else np.minimum(a[1:h - 2], a[2:h - 1])
        up = a[0:h - thick - 1]
        dn = a[thick + 1:h]
        cand = (mid - up > CONTRAST) & (mid - dn > CONTRAST) & (mid > 0)
        flat = np.zeros_like(cand)
        flat[:, 1:] = np.abs(np.diff(mid, axis=1)) <= FLAT
        m = cand & flat
        for i in range(m.shape[0]):
            row = m[i]
            if not row.any():
                continue
            j = 0
            while j < w:
                if row[j]:
                    k = j
                    while k < w and row[k]:
                        k += 1
                    if k - j >= MIN_RUN:
                        n_seg += 1
                        mask[i + 1:i + 1 + thick, max(j - 1, 0):k] = True
                    j = k
                else:
                    j += 1
    return mask, n_seg


def detect(img: np.ndarray) -> Tuple[np.ndarray, int]:
    """Маска разметки и число отрезков (горизонтальных + вертикальных)."""
    m, n, _, _ = _detect_hv(img)
    return m, n


def _detect_hv(img):
    if img is None or img.ndim != 2 or min(img.shape) < 8:
        z = np.zeros(img.shape[:2] if img is not None else (0, 0), dtype=bool)
        return z, 0, z, z
    mh, nh = _runs_mask(img)
    mv, nv = _runs_mask(img.T)
    return mh | mv.T, nh + nv, mh, mv.T


def _fill_lines(img: np.ndarray, mh: np.ndarray, mv: np.ndarray) -> np.ndarray:
    """Закрасить пиксели линий средним ближайших незатронутых соседей поперёк линии
    (для горизонтальной — сверху и снизу, для вертикальной — слева и справа). Для линии в 1–2 px это почти
    точное восстановление, без размытия окрестности."""
    out = img.astype(np.float32).copy()
    h, w = img.shape
    anym = mh | mv
    for mask, axis in ((mh, 0), (mv, 1)):
        ys, xs = np.nonzero(mask)
        for y, x in zip(ys, xs):
            vals = []
            for d in (-1, 1):
                yy, xx = y, x
                for _ in range(4):
                    if axis == 0:
                        yy += d
                    else:
                        xx += d
                    if not (0 <= yy < h and 0 <= xx < w):
                        break
                    if not anym[yy, xx]:
                        vals.append(float(img[yy, xx]))
                        break
            if vals:
                out[y, x] = sum(vals) / len(vals)
    return out.round().clip(np.iinfo(img.dtype).min if img.dtype.kind in "ui" else out.min(),
                            np.iinfo(img.dtype).max if img.dtype.kind in "ui" else out.max()).astype(img.dtype)


def clean(img: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Вернуть (кадр, отчёт). Кадр меняется, только если найдено >= MIN_SEGMENTS отрезков разметки."""
    mask, n, mh, mv = _detect_hv(img)
    info = {"markup_segments": int(n), "markup_cleaned": False, "markup_pixels": 0}
    if n < MIN_SEGMENTS:
        return img, info
    out = _fill_lines(img, mh, mv)
    # наклонные отрезки (рамка шейки бедра повёрнута): ищутся только когда разметка уже подтверждена
    # прямыми линиями; тонкие яркие прямые компоненты по белому top-hat, закраска cv2.inpaint без расширения маски
    diag = _diag_mask(out)
    if diag.any():
        import cv2
        u8 = out if out.dtype == np.uint8 else np.clip(out, 0, 255).astype(np.uint8)
        out = cv2.inpaint(u8, diag.astype(np.uint8), 2, cv2.INPAINT_TELEA).astype(img.dtype)
    info.update(markup_cleaned=True, markup_pixels=int(mask.sum() + diag.sum()), markup_diag_pixels=int(diag.sum()))
    return out, info


def _diag_mask(img: np.ndarray) -> np.ndarray:
    """Тонкие яркие прямые отрезки любой ориентации: белый top-hat (элемент 5×5) → HoughLinesP → маска
    только по пикселям top-hat вдоль найденных отрезков (кость и её края толще 5 px в top-hat не попадают)."""
    import cv2
    u8 = img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
    th = cv2.morphologyEx(u8, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    cand = (th > CONTRAST).astype(np.uint8)
    lines = cv2.HoughLinesP(cand * 255, 1, np.pi / 180, threshold=12, minLineLength=12, maxLineGap=1)
    keep = np.zeros(img.shape, dtype=np.uint8)
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            cv2.line(keep, (int(x1), int(y1)), (int(x2), int(y2)), 1, 2)
    return (keep > 0) & (cand > 0)
