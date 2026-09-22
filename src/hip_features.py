"""
Контур A для ПРОКСИМАЛЬНОГО ОТДЕЛА БЕДРА: анатомически осмысленные признаки
укладки/ротации и корректности области интереса (ROI).

Почему отдельный модуль. Исходная реализация hip_positioning_features() брала
"нижние 40% строк общей маски кости" за диафиз. На снимках бедра Lunar
Prodigy маска Otsu почти всегда содержит и таз (седалищная/лонная кость,
вертлужная впадина), который на коротких сканах заходит в нижнюю часть
кадра. Из-за этого "угол диафиза" имел std ~12° и знак, зависящий от
стороны (правое бедро: диафиз слева, головка справа-сверху; левое — зеркально).
Итог: AUC lh_pos = 0.33 (< 0.5), т.е. признак был сломан, а не инвертирован.

Что делаем здесь:
  1. Канонизация стороны: левое бедро зеркалим по горизонтали, чтобы ВСЕ
     снимки выглядели как правое бедро (диафиз слева-снизу, головка
     справа-сверху). Сторона определяется по самому изображению (центроид
     кости в нижней трети кадра относительно верхней), поэтому на тесте
     не нужна метка стороны. Это же реализует рекомендацию Fable 5 о единой
     модели "бедро" на 2 критерия вместо 2x2.
  2. Трекинг диафиза снизу вверх по связным отрезкам (run) в каждой строке:
     начинаем с самой нижней строки с костью, поднимаемся, выбирая отрезок с
     максимальным перекрытием с предыдущим. Так мы получаем именно бедренную
     кость (латеральный и медиальный контуры по строкам), а таз отсекается
     автоматически там, где шейка сливается с вертлужной впадиной (резкий
     скачок ширины).
  3. По латеральному/медиальному контурам считаем:
     - signed_shaft_angle_deg: знаковый угол оси диафиза к вертикали кадра
       (в мм, с учётом анизотропного пикселя 1.05x0.6). Положительный =
       низ диафиза смещён медиально (норма — небольшая аддукция).
     - lesser_troch_prominence_mm: выступ малого вертела на медиальном
       контуре (мм). При правильной внутренней ротации 15-25° малый вертел
       скрыт за диафизом; при недостаточной ротации он выступает бугром —
       классический признак нарушения ротации в протоколе DXA бедра.
     - shaft_width_mm, shaft_len_below_troch_mm (сколько диафиза ниже
       вертелов попало в кадр), scan_length_mm (физическая длина скана),
       lateral_margin_mm (расстояние от латерального края кости до края
       кадра) — признаки полноты ROI: область Total Hip требует наличия
       диафиза ниже малого вертела; слишком короткий скан = ROI некорректна.

ИТОГ ПОСЛЕ ДОРАБОТКИ (см. docs/hip_features_report.md): фактически работающие
признаки позиционирования — femur_solidity (0.71), shaft_width_mm (0.68),
abs_shaft_angle_deg (0.63), medial_neck_extent_mm / merge_height_mm (~0.65
инверт.); lesser_troch_prominence_mm вопреки ожиданию сигнала НЕ дал (0.53-0.56)
— малый вертел виден в обоих классах, разница в ротации тоньше разрешения
маски. Знаковый угол также ≈0.5: норма в этих данных — низ диафиза чуть
ЛАТЕРАЛЬНЕЕ (среднее -5°, std 5-6°), отклонение флагуется в обе стороны.
Объединённая модель hip_pos: OOF AUC 0.69 (geom) / 0.70 (stacked);
hip_roi: 0.90 / 0.92 (16 позитивов; per-side lh_roi/rh_roi ненадёжны).

Наблюдение по данным (визуальный осмотр всех 7 rh_roi- и 4 lh_roi-позитивов,
см. docs/hip/*_roi.png): все "ROI некорректна" на правом бедре — это сканы
высотой 180-207 строк (190-217 мм) против 233-346 у нормы, т.е. оператор
остановил скан слишком рано и диафиза ниже малого вертела не хватает.
Это физическая причина, а не артефакт разметки, поэтому scan_length_mm и
shaft_len_below_troch_mm — легитимные признаки. При этом для lh_roi всего
4 позитива (один из них — вообще нестандартный снимок), надёжной оценки
для этого критерия по 4 точкам получить нельзя; см. комментарии в
train_stacked.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import os as _os
import numpy as np
import cv2
from scipy.signal import find_peaks

import preprocess

PIXEL_SPACING_Y_MM = 1.05
PIXEL_SPACING_X_MM = 0.6


# ---------------------------------------------------------------------------
# Сегментация кости для бедра
# ---------------------------------------------------------------------------
def segment_bone_hip(img_u8, frac=0.75):
    """
    Сегментация кости для бедра.
    Отличия от segment_bone() (позвоночник):
      * порог Otsu считается только по пикселям тела (preprocess.body_mask:
        порог 8 по сглаженному кадру — иначе шум σ=3 % загоняет воздух в «тело»), а не по всему кадру
        (воздух/вырезанные углы скана занимают до половины кадра и смещают порог);
      * порог строчно-адаптивный: отдельный Otsu для верхних 60% (таз+головка,
        ярко) и нижних 40% (диафиз+мягкие ткани), с линейным переходом между
        50% и 75% высоты. Иначе на снимках с ярким тазом медуллярный канал
        диафиза выпадает из маски и трекинг "цепляется" за одну кортикальную
        стенку шириной 5-10 мм (docs/hip/track_fail.png -> docs/hip/seg_v3.png);
      * берём frac*Otsu (0.75): межвертельная область и большой вертел имеют
        низкую плотность и при "чистом" Otsu выпадают дырами
        (docs/hip/seg_compare.png);
      * заливка внутренних дыр (flood fill с нулевой рамкой) + закрытие 9x9;
      * НЕ оставляем одну крупнейшую компоненту: на коротких сканах бедро и таз
        могут быть разными компонентами, а диафиз нужен всегда. Компоненты
        < 0.5% кадра отбрасываем как шум/оверлеи.
    """
    h, w = img_u8.shape
    body = preprocess.body_mask(img_u8)
    if body.sum() < 100:
        return np.zeros_like(img_u8)
    blurred = cv2.GaussianBlur(img_u8, (5, 5), 0)

    def _otsu(region):
        px = blurred[region]
        if px.size < 100:
            return 255.0
        t, _ = cv2.threshold(px, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return float(t)

    top = np.zeros_like(body); top[:int(0.6 * h)] = True
    t_top = _otsu(body & top)
    t_bot = min(_otsu(body & ~top), t_top)
    thr = np.empty(h, dtype=float)
    y0, y1 = int(0.5 * h), int(0.75 * h)
    thr[:y0] = t_top; thr[y1:] = t_bot; thr[y0:y1] = np.linspace(t_top, t_bot, y1 - y0)
    mask = ((blurred > frac * thr[:, None]) & body).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
    # заливка внутренних дыр (с нулевой рамкой: угол кадра может быть внутри
    # кости, тогда заливка от (0,0) без рамки ошибочно "заливает" весь фон)
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(padded, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(padded)[1:-1, 1:-1]
    mask = cv2.bitwise_or(mask, holes)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    min_area = 0.005 * h * w
    out = np.zeros_like(mask)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def hip_side_score(img_u8, mask=None):
    """
    Скор стороны бедра по самому изображению. >0 => правое бедро в стандартной
    рентгенологической проекции (правая сторона пациента слева на экране:
    диафиз слева-снизу, таз/вертлужная впадина справа-сверху), <0 => левое.

    Две независимые подсказки:
      1) центроид кости в ВЕРХНЕЙ трети кадра относительно центра кадра —
         там доминирует таз, который всегда медиально (вес 2);
      2) доля "тела" (не воздух) в крайних 12% колонок справа минус слева:
         латерально от бедра — воздух, медиально — мягкие ткани (вес 1).

    Проверка без меток стороны: в каждом исследовании с чётным числом
    снимков бедра (2/4/8) детектор даёт ровно половину правых и половину левых
    в 63/64 исследованиях; единственное исключение — исследование с 5+3
    дубликатами одного и того же снимка. Исходная эвристика build_dataset
    (плотность ярких пикселей по половинам) ошибалась в ~24% снимков.
    """
    if mask is None:
        mask = segment_bone_hip(img_u8)
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(ys) < 10:
        return 0.0
    top = xs[ys <= h // 3]
    top_cue = (top.mean() / w - 0.5) if len(top) >= 5 else (xs.mean() / w - 0.5)
    k = max(1, int(0.12 * w))
    body = preprocess.body_mask(img_u8)
    edge_cue = float(body[:, w - k:].mean() - body[:, :k].mean())
    if _os.environ.get("DENSITO_SIDE_MODE", "crop_aware") == "crop_aware":
        # Подсказка по краям кадра (воздух латерально, мягкие ткани медиально) неверна,
        # когда поле сканирования обрезано: силуэт тела упирается в край кадра и «воздуха»
        # с латеральной стороны просто нет. В этом случае оставляем только анатомическую
        # подсказку по положению таза в верхней трети кадра.
        touch_left = float(body[:, 0].mean())
        touch_right = float(body[:, w - 1].mean())
        if max(touch_left, touch_right) > 0.5:
            return float(2.0 * top_cue)
    return float(2.0 * top_cue + edge_cue)


def detect_hip_side(img_u8, mask=None):
    """'right' (снимок уже в канонической ориентации) или 'left'."""
    return 'right' if hip_side_score(img_u8, mask) >= 0 else 'left'


def _runs(row):
    """Связные отрезки [l, r] (включительно) ненулевых пикселей в строке."""
    nz = np.nonzero(row)[0]
    if len(nz) == 0:
        return []
    breaks = np.nonzero(np.diff(nz) > 1)[0]
    starts = np.concatenate([[nz[0]], nz[breaks + 1]])
    ends = np.concatenate([nz[breaks], [nz[-1]]])
    return list(zip(starts, ends))


def track_femur(mask_canon, min_run_px=4):
    """
    Трекинг бедренной кости снизу вверх по отрезкам.
    Старт: среди 6 самых нижних строк с костью берём строку с самым широким
    отрезком (нижняя кромка кадра часто содержит обрезки/узкие фрагменты);
    стартовый отрезок — самый широкий в этой строке (диафиз всегда шире
    случайных фрагментов седалищной кости у нижнего края).
    Далее для каждой строки выше выбираем отрезок с максимальным перекрытием
    с предыдущим; останавливаемся, когда перекрытия нет.
    Возвращает dict с массивами (снизу вверх): y, left (латеральный край),
    right (медиальный край), width; либо None, если кость не найдена.
    """
    h, w = mask_canon.shape
    bottom_rows = []
    for y in range(h - 1, -1, -1):
        runs = [r for r in _runs(mask_canon[y]) if r[1] - r[0] + 1 >= min_run_px]
        if runs:
            best = max(runs, key=lambda r: r[1] - r[0])
            bottom_rows.append((y, best))
            if len(bottom_rows) >= 6:
                break
    if not bottom_rows:
        return None
    y0, cur = max(bottom_rows, key=lambda t: t[1][1] - t[1][0])

    ys, ls, rs = [y0], [cur[0]], [cur[1]]
    for y in range(y0 - 1, -1, -1):
        runs = _runs(mask_canon[y])
        best, best_ov = None, 0
        for r in runs:
            ov = min(r[1], cur[1]) - max(r[0], cur[0]) + 1
            if ov > best_ov:
                best, best_ov = r, ov
        if best is None or best_ov <= 0:
            break
        cur = best
        ys.append(y); ls.append(cur[0]); rs.append(cur[1])

    ys = np.array(ys); ls = np.array(ls, dtype=float); rs = np.array(rs, dtype=float)
    return {'y': ys, 'left': ls, 'right': rs, 'width': rs - ls + 1}


PROFILE_STEP_MM = 4.0       # шаг сэмплирования контура над диафизом, мм
PROFILE_N = 20              # 20 * 4 мм = 80 мм над диафизом (вертелы + шейка)


def _fit_shaft(y, L, R, SX, SY, tol_mm=2.5):
    """
    Итеративно определяет диафиз: стартуем с нижних 30% трека, аппроксимируем
    латеральный и медиальный края прямыми, расширяем вверх пока оба края
    отклоняются от прямых менее tol_mm. Возвращает индекс верхней строки
    диафиза (exclusive) и коэффициенты прямых.
    """
    n = len(y)
    n0 = max(8, int(0.3 * n))
    top = n0
    for _ in range(4):
        idx = np.arange(0, top)
        yy = y[idx].astype(float)
        al, bl = np.polyfit(yy, L[idx], 1)
        ar, br = np.polyfit(yy, R[idx], 1)
        dl = np.abs(np.polyval([al, bl], y.astype(float)) - L) * SX
        dr = np.abs(np.polyval([ar, br], y.astype(float)) - R) * SX
        ok = (dl < tol_mm) & (dr < tol_mm)
        new_top = n0
        for i in range(n0, n):
            if ok[i:i + 3].any() if i + 3 <= n else ok[i]:
                new_top = i + 1
            else:
                break
        new_top = max(new_top, n0)
        if new_top == top:
            break
        top = new_top
    idx = np.arange(0, top)
    yy = y[idx].astype(float)
    al, bl = np.polyfit(yy, L[idx], 1)
    ar, br = np.polyfit(yy, R[idx], 1)
    return top, (al, bl), (ar, br)


def _shape_features(mask_canon, y_seed, x_seed, SX, SY):
    """
    Форма силуэта компоненты, содержащей диафиз (в изотропных мм):
      femur_solidity    = area / area(convex hull). Самый сильный одиночный
                          признак ротации (AUC 0.71 на 79 позитивах): при
                          наружной ротации шейка укорачивается в проекции,
                          вырезы между большим вертелом и головкой и под
                          головкой "заполняются" -> силуэт становится выпуклее;
      femur_compactness = 4*pi*A/P^2 (та же физика, слабее, AUC 0.65);
      femur_eccentricity= эксцентриситет вписанного эллипса (AUC ~0.5, оставлен
                          для диагностики, в модель не входит).
    Идея взята из предложения использовать MTDDH (детские рентгенограммы таза)
    как референс формы: Mahalanobis-расстояние до распределения формы
    подвздошной кости MTDDH дало AUC 0.29 (~0.71 инверт.), т.е. ничего сверх
    собственной solidity, а референс анатомически не относится к бедру ->
    внешний датасет не используется.
    """
    res = {'femur_solidity': None, 'femur_compactness': None, 'femur_eccentricity': None}
    n_labels, labels, _, _ = cv2.connectedComponentsWithStats(mask_canon)
    li = labels[y_seed, x_seed]
    if li == 0:
        return res
    comp = (labels == li).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return res
    c = max(cnts, key=cv2.contourArea)
    pts = c[:, 0, :].astype(np.float32) * np.array([SX, SY], np.float32)
    if len(pts) < 5:
        return res
    A = cv2.contourArea(pts); P = cv2.arcLength(pts, True)
    hull_a = cv2.contourArea(cv2.convexHull(pts))
    (_, _), (ma, MA), _ = cv2.fitEllipse(pts)
    res['femur_solidity'] = float(A / max(hull_a, 1e-6))
    res['femur_compactness'] = float(4 * np.pi * A / max(P ** 2, 1e-6))
    res['femur_eccentricity'] = float(np.sqrt(max(0.0, 1 - (min(ma, MA) / max(ma, MA, 1e-6)) ** 2)))
    return res


def hip_features_canonical(mask_canon):
    """Все признаки бедра по канонически ориентированной маске (правое бедро)."""
    h, w = mask_canon.shape
    SX, SY = PIXEL_SPACING_X_MM, PIXEL_SPACING_Y_MM
    out = {
        'signed_shaft_angle_deg': None, 'abs_shaft_angle_deg': None,
        'shaft_width_mm': None, 'shaft_len_below_troch_mm': None,
        'lesser_troch_prominence_mm': None, 'greater_troch_offset_mm': None,
        'lateral_margin_mm': None, 'medial_neck_extent_mm': None,
        'merge_height_mm': None, 'neck_min_width_mm': None,
        'scan_length_mm': float(h * SY), 'scan_width_mm': float(w * SX),
        'femur_track_rows': 0, 'shaft_bottom_center_ratio': None,
        'bone_area_ratio': float((mask_canon > 0).mean()),
        'femur_fill_ratio': None, 'bone_top_touch_ratio': float((mask_canon[0] > 0).mean()),
        'bone_bottom_touch_ratio': float((mask_canon[-1] > 0).mean()),
        'femur_solidity': None, 'femur_compactness': None, 'femur_eccentricity': None,
    }
    for i in range(PROFILE_N):
        out[f'prof_lat_{i}'] = None
        out[f'prof_med_{i}'] = None

    tr = track_femur(mask_canon)
    if tr is None or len(tr['y']) < 15:
        return out
    y, L, R, W = tr['y'], tr['left'], tr['right'], tr['width']
    n = len(y)
    out['femur_track_rows'] = int(n)
    out.update(_shape_features(mask_canon, y[0], int((L[0] + R[0]) // 2), SX, SY))
    out['femur_fill_ratio'] = float(n / h)

    shaft_top, (al, bl), (ar, br) = _fit_shaft(y, L, R, SX, SY)
    idx_shaft = np.arange(0, shaft_top)
    w0 = float(np.median(W[idx_shaft]))
    out['shaft_width_mm'] = w0 * SX
    out['shaft_len_below_troch_mm'] = float((y[0] - y[shaft_top - 1] + 1) * SY)

    # ось диафиза: среднее двух краевых прямых. x = a*y + b, y растёт вниз.
    a = 0.5 * (al + ar)
    # a > 0: при движении вниз x растёт => низ диафиза медиальнее (аддукция, +)
    angle = np.degrees(np.arctan2(a * SX, SY))
    out['signed_shaft_angle_deg'] = float(angle)
    out['abs_shaft_angle_deg'] = float(abs(angle))
    c = (L + R) / 2.0
    out['shaft_bottom_center_ratio'] = float(c[0] / w)
    out['lateral_margin_mm'] = float(L[idx_shaft].min() * SX)

    # отклонения контуров от прямых диафиза (мм); >0 = выступ наружу
    yf = y.astype(float)
    dev_lat = (np.polyval([al, bl], yf) - L) * SX
    dev_med = (R - np.polyval([ar, br], yf)) * SX

    # точка слияния с тазом: ширина > 3*w0 либо медиальный край делает
    # скачок > 15 мм за одну строку
    jump = np.nonzero(np.diff(R) * SX > 15.0)[0] + 1
    wide = np.nonzero(W > 3.0 * w0)[0]
    cands = [i for i in list(jump) + list(wide) if i > shaft_top]
    end = min(cands) if cands else n
    out['merge_height_mm'] = float((y[shaft_top - 1] - y[end - 1]) * SY) if end > shaft_top else 0.0

    seg = slice(shaft_top, end)
    dm, dl, wseg = dev_med[seg], dev_lat[seg], W[seg]
    if len(dm) >= 3:
        out['medial_neck_extent_mm'] = float(np.max(dm))
        out['greater_troch_offset_mm'] = float(np.max(dl))
        out['neck_min_width_mm'] = float(np.min(wseg[len(wseg) // 4:]) * SX) if len(wseg) >= 8 else float(np.min(wseg) * SX)
        dm_s = np.convolve(np.pad(dm, 2, mode='edge'), np.ones(5) / 5, mode='valid')
        peaks, props = find_peaks(dm_s, prominence=1.0)
        prom = 0.0
        for pk, pr in zip(peaks, props['prominences']):
            if dm_s[pk] > 1.5:
                prom = float(pr)
                break
        out['lesser_troch_prominence_mm'] = prom
    else:
        out['medial_neck_extent_mm'] = 0.0
        out['greater_troch_offset_mm'] = 0.0
        out['neck_min_width_mm'] = w0 * SX
        out['lesser_troch_prominence_mm'] = 0.0

    # профиль контура над диафизом: сэмплы через PROFILE_STEP_MM, мм от линий
    # диафиза; за точкой слияния с тазом — последнее валидное значение
    # (медиальный) / NaN->последнее (латеральный), чтобы вектор был фиксированной длины.
    y_top = y[shaft_top - 1]
    step_rows = PROFILE_STEP_MM / SY
    for i in range(PROFILE_N):
        yi = y_top - (i + 1) * step_rows
        k = int(round((y[0] - yi)))   # индекс в треке (трек идёт снизу вверх по 1 строке)
        k = min(max(k, 0), n - 1)
        if k >= end:
            k = end - 1
        out[f'prof_lat_{i}'] = float(np.clip(dev_lat[k], -30, 60))
        out[f'prof_med_{i}'] = float(np.clip(dev_med[k], -30, 60))
    return out


def hip_all_features(img_u8, mask=None):
    """
    Точка входа: img_u8 -> маска (если не передана) -> канонизация стороны ->
    признаки. Возвращает dict; ключ 'hip_side_detected' — определённая по
    изображению сторона ('right'/'left').
    """
    if mask is None:
        mask = segment_bone_hip(img_u8)
    side_score = hip_side_score(img_u8, mask)
    side = 'right' if side_score >= 0 else 'left'
    mask_c = mask if side == 'right' else mask[:, ::-1]
    feats = hip_features_canonical(np.ascontiguousarray(mask_c))
    feats['hip_side_detected'] = side
    feats['hip_side_score'] = float(side_score)
    return feats


def draw_hip_debug(img_u8, mask=None):
    """Отладочная визуализация: контур маски + трек диафиза в исходной ориентации."""
    if mask is None:
        mask = segment_bone_hip(img_u8)
    side = detect_hip_side(img_u8, mask)
    mask_c = mask if side == 'right' else mask[:, ::-1]
    mask_c = np.ascontiguousarray(mask_c)
    img_c = img_u8 if side == 'right' else np.ascontiguousarray(img_u8[:, ::-1])
    col = cv2.cvtColor(img_c, cv2.COLOR_GRAY2BGR)
    cnts, _ = cv2.findContours(mask_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(col, cnts, -1, (0, 0, 255), 1)
    tr = track_femur(mask_c)
    f = hip_features_canonical(mask_c)
    if tr is not None:
        for yy, l, r in zip(tr['y'], tr['left'], tr['right']):
            col[yy, int(l)] = (0, 255, 0)
            col[yy, int(r)] = (255, 255, 0)
        if f['shaft_len_below_troch_mm'] is not None:
            y_top = int(tr['y'][0] - f['shaft_len_below_troch_mm'] / PIXEL_SPACING_Y_MM)
            cv2.line(col, (0, y_top), (col.shape[1] - 1, y_top), (255, 0, 255), 1)

    def _r(v):
        return None if v is None else round(v, 1)
    cv2.putText(col, f"a={_r(f['signed_shaft_angle_deg'])} lt={_r(f['lesser_troch_prominence_mm'])}",
                (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
    cv2.putText(col, f"{side} L={f['scan_length_mm']:.0f} sh={_r(f['shaft_len_below_troch_mm'])}",
                (3, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
    return col
