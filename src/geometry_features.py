"""
Контур A — явные геометрические признаки для критериев ТЗ, вместо
end-to-end CNN на крошечном датасете (см. рецензию Fable 5).

Реализует:
  - сегментацию кости (порог + морфология + крупнейшая компонента)
  - центральную линию позвоночника, угол оси к вертикали кадра,
    кривизну (чтобы анатомическое искривление не штрафовалось по оси)
  - детектор высокоплотных объектов (металл: посторонние предметы / эндопротез)
  - признаки укладки (центрирование, положение верх/низ кадра)
  - признаки для бедра: угол диафиза, положение головки/вертела

Пиксельные размеры аппарата (Lunar Prodigy Advance, подтверждено организаторами
письменно): Y = 1.05 мм, X = 0.6 мм.
"""
import warnings
warnings.filterwarnings("ignore")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
import numpy as np
import cv2
import pydicom

import preprocess

PIXEL_SPACING_Y_MM = 1.05
PIXEL_SPACING_X_MM = 0.6


def read_dicom_normalized(path):
    """Читает DICOM, нормализует в [0,255] uint8, учитывает MONOCHROME1."""
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array.astype(np.float32)

    photometric = getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')
    if photometric == 'MONOCHROME1':
        arr = arr.max() - arr

    slope = float(getattr(ds, 'RescaleSlope', 1.0) or 1.0)
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0) or 0.0)
    arr = arr * slope + intercept

    lo, hi = np.percentile(arr, [1, 99])
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255)
    else:
        arr = np.zeros_like(arr)
    img_u8 = arr.astype(np.uint8)
    # Канонизация экспозиции (один параметр γ к эталону обучения): окно 1–99 % снимает
    # только линейные сдвиги яркости, степенные — нет (src/preprocess.py).
    img_u8, _ = preprocess.canonicalize_exposure(img_u8)
    return img_u8, ds


def read_dicom_raw_normalized(path):
    """То же, но без канонизации экспозиции — для диагностики и аудитов сырой экспозиции."""
    saved = preprocess.flags()["exposure_canonicalization"]
    preprocess.flags()["exposure_canonicalization"] = False
    try:
        return read_dicom_normalized(path)
    finally:
        preprocess.flags()["exposure_canonicalization"] = saved


def segment_bone(img_u8):
    """Сегментация кости: адаптивный порог (Otsu) + морфология + крупнейшая компонента."""
    blurred = cv2.GaussianBlur(img_u8, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    n_labels, labels = cv2.connectedComponents(mask)
    if n_labels <= 1:
        return mask
    sizes = [(labels == i).sum() for i in range(1, n_labels)]
    largest_label = 1 + int(np.argmax(sizes))
    largest_mask = (labels == largest_label).astype(np.uint8) * 255
    return largest_mask


def _row_runs(row, min_px=4):
    """Связные отрезки [l, r] ненулевых пикселей длиной >= min_px."""
    nz = np.nonzero(row)[0]
    if len(nz) == 0:
        return []
    breaks = np.nonzero(np.diff(nz) > 1)[0]
    starts = np.concatenate([[nz[0]], nz[breaks + 1]])
    ends = np.concatenate([nz[breaks], [nz[-1]]])
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a + 1 >= min_px]


def _tracked_centerline(mask):
    """Центральная линия ТОЛЬКО по столбику позвонков: отслеживаем связный отрезок вверх и вниз
    от самой надёжной строки (самый широкий отрезок в средней трети кадра). Возвращает (ys, xs)."""
    h, w = mask.shape
    runs_by_row = {}
    for y in range(h):
        rs = _row_runs(mask[y, :])
        if rs:
            runs_by_row[y] = rs
    if not runs_by_row:
        return None, None
    mid_lo, mid_hi = h // 3, 2 * h // 3
    cand = [(y, r) for y, rs in runs_by_row.items() if mid_lo <= y <= mid_hi for r in rs]
    if not cand:
        cand = [(y, r) for y, rs in runs_by_row.items() for r in rs]
    y_start, run_start = max(cand, key=lambda t: t[1][1] - t[1][0])

    def overlap(a, b):
        return min(a[1], b[1]) - max(a[0], b[0]) + 1

    picked = {y_start: run_start}
    for direction in (-1, 1):
        cur = run_start
        y = y_start + direction
        while 0 <= y < h:
            rs = runs_by_row.get(y)
            if not rs:
                break
            best = max(rs, key=lambda r: overlap(r, cur))
            if overlap(best, cur) <= 0:
                break
            picked[y] = best
            cur = best
            y += direction
    ys = np.array(sorted(picked), dtype=np.float64)
    xs = np.array([0.5 * (picked[int(y)][0] + picked[int(y)][1]) for y in ys], dtype=np.float64)
    return ys, xs


def _fit_angle_trimmed(ys_arr, xs_arr, trim=0.10):
    """Линейная аппроксимация x = a*y + b с отбрасыванием trim худших остатков. -> (a, b)."""
    a, b = np.polyfit(ys_arr, xs_arr, deg=1)
    if trim > 0 and len(ys_arr) >= 20:
        res = np.abs(xs_arr - (a * ys_arr + b))
        keep = res <= np.quantile(res, 1.0 - trim)
        if keep.sum() >= 10:
            a, b = np.polyfit(ys_arr[keep], xs_arr[keep], deg=1)
    return float(a), float(b)


def spine_axis_features(img_u8, mask, mode=None):
    """
    Центральная линия и угол к вертикали кадра. Два режима:
      baseline (по умолчанию, продакшен 2.3.1) — центр масс ВСЕХ пикселей маски в строке;
      tracked  — только связный отрезок позвоночного столба (см. _tracked_centerline) и
                 аппроксимация с отбрасыванием 10 % худших остатков. Включается параметром
                 mode="tracked" или переменной окружения DENSITO_AXIS_MODE=tracked.
    Квадратичная аппроксимация -> кривизна (анатомическое искривление, НЕ штрафуется по оси).
    """
    if mode is None:
        mode = os.environ.get("DENSITO_AXIS_MODE", "baseline")
    h, w = mask.shape
    if mode == "tracked":
        ys_t, xs_t = _tracked_centerline(mask)
        ys = [] if ys_t is None else list(ys_t)
        xs_centroid = [] if xs_t is None else list(xs_t)
    else:
        ys, xs_centroid = [], []
        for y in range(h):
            row = mask[y, :]
            xs_nonzero = np.nonzero(row)[0]
            if len(xs_nonzero) > 3:  # минимум пикселей, чтобы считать строку валидной
                xs_centroid.append(xs_nonzero.mean())
                ys.append(y)

    if len(ys) < 10:
        return {
            'axis_angle_deg': None, 'curvature': None, 'valid_rows': len(ys),
            'centerline_x': None, 'centerline_y': None,
        }

    ys_arr = np.array(ys, dtype=np.float64)
    xs_arr = np.array(xs_centroid, dtype=np.float64)

    # линейная аппроксимация x = a*y + b -> угол к вертикали
    if mode == "tracked":
        a, b = _fit_angle_trimmed(ys_arr, xs_arr)
    else:
        a, b = np.polyfit(ys_arr, xs_arr, deg=1)
    # угол между линией и вертикалью (ось Y). dx/dy = a (в пикселях), с учётом
    # анизотропного пикселя переводим в физические мм перед вычислением угла.
    dx_mm_per_row = a * PIXEL_SPACING_X_MM
    dy_mm_per_row = 1.0 * PIXEL_SPACING_Y_MM
    angle_rad = np.arctan2(abs(dx_mm_per_row), dy_mm_per_row)
    angle_deg = np.degrees(angle_rad)

    # квадратичная аппроксимация для кривизны
    coeffs2 = np.polyfit(ys_arr, xs_arr, deg=2)
    residual_linear = xs_arr - np.polyval([a, b], ys_arr)
    residual_quad = xs_arr - np.polyval(coeffs2, ys_arr)
    curvature_metric = float(np.std(residual_linear) - np.std(residual_quad))
    # положительное значение => квадратичная модель существенно лучше => анатомическое искривление

    return {
        'axis_angle_deg': float(angle_deg),
        'curvature': curvature_metric,
        'valid_rows': len(ys),
        'centerline_x': xs_arr,
        'centerline_y': ys_arr,
    }


def foreign_object_features(img_u8, mask):
    """
    Детектор посторонних объектов ВНЕ кости (застёжки, пуговицы, молнии,
    складки одежды) — критерий 'посторонние предметы' (позвоночник) и
    бонус 'эндопротез' (внутри маски бедра, отдельная логика).

    Ключевая идея (уточнена по визуальному анализу реальных позитивов):
    объект не обязательно ярче кости абсолютно, но заметно ярче ЛОКАЛЬНОГО
    фона мягких тканей вокруг него. Ищем яркие компактные компоненты вне
    маски кости, сравнивая с фоном в увеличенной окрестности маски (soft
    tissue band), а не с порогом по интенсивности самой кости.
    """
    h, w = img_u8.shape

    # область "мягких тканей" вокруг кости (не фон-черный, не кость) —
    # расширяем маску кости и берём кольцо вокруг неё как референс фона
    kernel_big = np.ones((25, 25), np.uint8)
    bone_dilated = cv2.dilate(mask, kernel_big, iterations=1)
    soft_tissue_band = ((bone_dilated > 0) & (mask == 0)).astype(np.uint8)

    soft_tissue_pixels = img_u8[soft_tissue_band > 0]
    # исключаем область фона (совсем чёрная, вне тела пациента)
    body_thresh = 8
    soft_tissue_pixels = soft_tissue_pixels[soft_tissue_pixels > body_thresh]

    if len(soft_tissue_pixels) < 20:
        return {'metal_area_px': 0, 'metal_area_mm2': 0.0, 'metal_outside_bone_px': 0,
                'metal_outside_bone_mm2': 0.0, 'has_metal': False, 'metal_max_intensity_gap': 0.0}

    bg_mean = float(np.mean(soft_tissue_pixels))
    bg_std = float(np.std(soft_tissue_pixels)) + 1e-6

    # candidate mask: заметно ярче фона мягких тканей, но не в самой кости
    z_thresh = 3.0
    bright_thresh = bg_mean + z_thresh * bg_std
    candidate = ((img_u8.astype(np.float32) > bright_thresh) &
                 (mask == 0) &
                 (img_u8 > body_thresh)).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate)
    px_area_mm2 = PIXEL_SPACING_X_MM * PIXEL_SPACING_Y_MM

    total_area = 0
    max_gap = 0.0
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 4:  # шум
            continue
        total_area += area
        component_intensity = img_u8[labels == i].mean()
        gap = (component_intensity - bg_mean) / bg_std
        max_gap = max(max_gap, gap)

    return {
        'metal_area_px': int(total_area),
        'metal_area_mm2': float(total_area * px_area_mm2),
        'metal_outside_bone_px': int(total_area),
        'metal_outside_bone_mm2': float(total_area * px_area_mm2),
        'has_metal': total_area * px_area_mm2 > 3.0,
        'metal_max_intensity_gap': float(max_gap),
    }


def spine_positioning_features(img_u8, mask):
    """Признаки укладки: центрирование по X, доля кадра, занятая костью сверху/снизу."""
    h, w = mask.shape
    ys_nonzero, xs_nonzero = np.nonzero(mask)
    if len(ys_nonzero) < 10:
        return {'center_offset_ratio': None, 'top_margin_ratio': None,
                'bottom_margin_ratio': None, 'bone_width_ratio': None}

    bone_center_x = xs_nonzero.mean()
    frame_center_x = w / 2.0
    center_offset_ratio = (bone_center_x - frame_center_x) / w

    top_margin_ratio = ys_nonzero.min() / h
    bottom_margin_ratio = (h - ys_nonzero.max()) / h
    bone_width_ratio = (xs_nonzero.max() - xs_nonzero.min()) / w

    return {
        'center_offset_ratio': float(center_offset_ratio),
        'top_margin_ratio': float(top_margin_ratio),
        'bottom_margin_ratio': float(bottom_margin_ratio),
        'bone_width_ratio': float(bone_width_ratio),
    }


def hip_positioning_features(img_u8, mask):
    """
    Признаки для бедра: угол диафиза бедренной кости к вертикали
    (абдукция/аддукция + ротация), положение относительно краёв кадра.
    """
    h, w = mask.shape
    ys_nonzero, xs_nonzero = np.nonzero(mask)
    if len(ys_nonzero) < 10:
        return {'shaft_angle_deg': None, 'edge_distance_ratio': None,
                'bone_area_ratio': None}

    # диафиз — нижняя треть маски (femoral shaft ближе к низу кадра типично)
    y_thresh = np.percentile(ys_nonzero, 60)
    shaft_pixels = mask.copy()
    shaft_pixels[:int(y_thresh), :] = 0

    ys_shaft, xs_shaft = np.nonzero(shaft_pixels)
    if len(ys_shaft) > 10:
        a, b = np.polyfit(ys_shaft, xs_shaft, deg=1)
        dx_mm = a * PIXEL_SPACING_X_MM
        dy_mm = 1.0 * PIXEL_SPACING_Y_MM
        shaft_angle_deg = float(np.degrees(np.arctan2(abs(dx_mm), dy_mm)))
    else:
        shaft_angle_deg = None

    min_edge_dist = min(xs_nonzero.min(), w - xs_nonzero.max())
    edge_distance_ratio = min_edge_dist / w
    bone_area_ratio = mask.sum() / 255.0 / (h * w)

    return {
        'shaft_angle_deg': shaft_angle_deg,
        'edge_distance_ratio': float(edge_distance_ratio),
        'bone_area_ratio': float(bone_area_ratio),
    }


def extract_all_features(file_path, region):
    """Единая точка входа: читает DICOM, извлекает все геометрические признаки."""
    try:
        img_u8, ds = read_dicom_normalized(file_path)
    except Exception as e:
        return {'error': f'dicom_read_failed: {e}'}

    mask = segment_bone(img_u8)

    features = {'region': region}
    foreign = foreign_object_features(img_u8, mask)
    features.update({f'metal_{k}': v for k, v in foreign.items()})

    if region == 'spine':
        axis = spine_axis_features(img_u8, mask)
        features['axis_angle_deg'] = axis['axis_angle_deg']
        features['curvature'] = axis['curvature']
        features['valid_rows'] = axis['valid_rows']
        pos = spine_positioning_features(img_u8, mask)
        features.update(pos)
    elif region in ('right_hip', 'left_hip', 'hip'):
        # Признаки бедра вычисляются в канонической ориентации (левое бедро
        # зеркалится), сторона определяется по изображению — см. hip_features.py.
        # Старый hip_positioning_features() оставлен только для сравнения:
        # его "диафиз" (нижние 40% маски, включая таз) давал угол со std 12°
        # и AUC≈0.5 на левом бедре.
        from hip_features import hip_all_features
        hip = hip_all_features(img_u8)
        features.update(hip)
        # legacy-ключи для обратной совместимости с более старыми колонками
        features['shaft_angle_deg'] = hip['abs_shaft_angle_deg']
        features['edge_distance_ratio'] = (
            None if hip['lateral_margin_mm'] is None
            else hip['lateral_margin_mm'] / max(hip['scan_width_mm'], 1e-6))

    return features


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv("/home/user/workspace/densito_rebuild/data/labels_full.csv")
    spine = df[df['region'] == 'spine']
    pos_row = spine[spine['sp_axis'] == 1].iloc[0]
    neg_row = spine[spine['sp_axis'] == 0].iloc[0]
    for name, row in [('axis_violation (label=1)', pos_row), ('axis_ok (label=0)', neg_row)]:
        feats = extract_all_features(row['file_path'], 'spine')
        print(name, '-> angle:', feats.get('axis_angle_deg'), 'curvature:', feats.get('curvature'),
              'metal_outside_mm2:', feats.get('metal_outside_bone_mm2'))
