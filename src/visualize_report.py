#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуализация результата (бонус ТЗ п.2.6 / п.4: "кейсы, демонстрирующие
работу решения").

Архитектурное решение. Классификатор densito_rebuild — геометрические признаки
плюс логистическая регрессия на замороженных эмбеддингах (Контур A + Контур B);
у него нет обучаемых свёрточных активаций, поэтому тепловые карты по градиентам
сети здесь не имели бы смысла (это были бы случайные активации предобученного
ImageNet-бэкбона, а не приближение обученной модели).

Вместо тепловой карты эксперту показываются явные измеренные геометрические
примитивы Контура A, которые непосредственно определяют quality_prob:
  - контур сегментированной кости;
  - линия оси позвоночника (центральная линия) vs вертикаль кадра,
    с численным углом отклонения;
  - обнаруженные высокоплотные объекты (металл/посторонние предметы),
    выделенные прямоугольником;
  - для бедра: трек диафиза, положение малого/большого вертела,
    измеренные отступы области интереса от краёв кадра.
Эксперт видит измеренную величину, а не размытое пятно; организаторы
спрашивали именно про локализацию находок на изображении/доп. серии
(см. QA_ANALYSIS.md, п.A.2).

Функции:
  render_overlay(img_u8, region, feats, crit_results) -> BGR uint8 image
  save_overlay_png(path, img_u8, region, feats, crit_results)
  overlay_to_dicom_sc(overlay_bgr, ref_ds) -> pydicom Dataset (Secondary Capture)
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import datetime
from typing import Any, Dict, Optional

import numpy as np
import cv2

from geometry_features import (
    PIXEL_SPACING_X_MM, PIXEL_SPACING_Y_MM, segment_bone, spine_axis_features,
)

# Цвета в BGR (OpenCV)
COL_BONE_CONTOUR = (60, 200, 60)      # зелёный — контур сегментированной кости
COL_AXIS = (0, 140, 255)              # оранжевый — измеренная ось/диафиз
COL_VERTICAL_REF = (180, 180, 180)    # серый — вертикаль кадра (референс)
COL_METAL = (0, 0, 255)               # красный — обнаруженный металл/посторонний предмет
COL_ROI_OK = (0, 220, 0)              # зелёный — измеренный ROI-отступ в норме
CRIT_SHORT = {"sp_pos": "укладка", "sp_axis": "ось", "sp_art": "предметы",
              "rh_pos": "укладка", "rh_roi": "ROI", "lh_pos": "укладка", "lh_roi": "ROI",
              "hip_pos": "укладка", "hip_roi": "ROI"}
COL_ROI_BAD = (0, 0, 255)             # красный — измеренный ROI-отступ ниже порога
COL_TEXT_BG = (0, 0, 0)
COL_TEXT = (255, 255, 255)
COL_VIOLATION_TEXT = (0, 0, 255)
COL_OK_TEXT = (0, 220, 0)
COL_UNCERTAIN_TEXT = (0, 200, 255)    # жёлтый — решение в зоне «не уверен» (нужен просмотр)


def _put_label(img, text, org, color=COL_TEXT, scale=0.42, thickness=1):
    """Текст с чёрной подложкой для читаемости на светлом/тёмном фоне."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + base + 2), COL_TEXT_BG, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _put_label_wrapped(img, text, x, y, max_width, color=COL_TEXT, scale=0.42,
                        thickness=1, line_gap=3, align_right=False):
    """Как _put_label, но переносит текст на несколько строк, чтобы не выйти
    за правый/левый край изображения (критично для маленьких DICOM, где
    сноски легко выезжают за границу кадра)."""
    words = text.split(" ")
    lines, cur = [], ""
    for word in words:
        trial = (cur + " " + word).strip()
        (tw, _), _ = cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if tw > max_width and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    (_, th), base = cv2.getTextSize("A", cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    line_h = th + base + line_gap
    for i, line in enumerate(lines):
        ly = y + i * line_h
        if align_right:
            (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            lx = x + max_width - tw
        else:
            lx = x
        _put_label(img, line, (lx, ly), color, scale, thickness)
    return y + len(lines) * line_h


def _metal_boxes(img_u8: np.ndarray, mask: np.ndarray):
    """Пересчитывает bounding-box'ы металла/посторонних объектов для отрисовки
    (та же логика порога, что и foreign_object_features, но с координатами)."""
    kernel_big = np.ones((25, 25), np.uint8)
    bone_dilated = cv2.dilate(mask, kernel_big, iterations=1)
    soft_tissue_band = ((bone_dilated > 0) & (mask == 0)).astype(np.uint8)
    soft_tissue_pixels = img_u8[soft_tissue_band > 0]
    body_thresh = 8
    soft_tissue_pixels = soft_tissue_pixels[soft_tissue_pixels > body_thresh]
    if len(soft_tissue_pixels) < 20:
        return []
    bg_mean = float(np.mean(soft_tissue_pixels))
    bg_std = float(np.std(soft_tissue_pixels)) + 1e-6
    bright_thresh = bg_mean + 3.0 * bg_std
    candidate = ((img_u8.astype(np.float32) > bright_thresh) & (mask == 0) &
                 (img_u8 > body_thresh)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate)
    boxes = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 4:
            continue
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        boxes.append((x, y, w, h))
    return boxes


def _render_spine_overlay(col: np.ndarray, img_u8: np.ndarray, feats: Dict[str, Any],
                           crit_results: Optional[Dict[str, Any]]) -> tuple:
    """Рисует ТОЛЬКО геометрию (контуры/линии/боксы) на изображении. Текстовые
    подписи возвращаются отдельно как список (текст, цвет) — они размещаются
    на отдельной панели в render_overlay, чтобы никогда не обрезаться и не
    перекрывать сам снимок (критично для маленьких DICOM, где текст на
    изображении легко выходит за границы кадра)."""
    h, w = img_u8.shape
    mask = segment_bone(img_u8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(col, cnts, -1, COL_BONE_CONTOUR, 1)

    axis = spine_axis_features(img_u8, mask)
    if axis["centerline_x"] is not None:
        xs, ys = axis["centerline_x"], axis["centerline_y"]
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.polylines(col, [pts], False, COL_AXIS, 2, cv2.LINE_AA)
        # вертикаль кадра, проходящая через верхнюю точку центральной линии, для сравнения
        x0 = int(xs[0])
        cv2.line(col, (x0, int(ys[0])), (x0, int(ys[-1])), COL_VERTICAL_REF, 1, cv2.LINE_AA)

    boxes = _metal_boxes(img_u8, mask)
    for (x, y, bw, bh) in boxes:
        cv2.rectangle(col, (x, y), (x + bw, y + bh), COL_METAL, 1)

    labels = []
    angle = feats.get("axis_angle_deg")
    if angle is not None:
        ok = angle <= 5.0
        labels.append((f"Ось к вертикали кадра: {angle:.1f}°", COL_OK_TEXT if ok else COL_VIOLATION_TEXT))
    curvature = feats.get("curvature")
    if curvature is not None and curvature > 0.6:
        labels.append(("Кривизна оси повышена — анатомическая особенность, по оси не штрафуем", (0, 200, 200)))
    if boxes:
        labels.append((f"Посторонние объекты: {len(boxes)}", COL_VIOLATION_TEXT))
    return col, labels


def _render_hip_overlay(col: np.ndarray, img_u8: np.ndarray, feats: Dict[str, Any],
                         crit_results: Optional[Dict[str, Any]]) -> tuple:
    from hip_features import segment_bone_hip, detect_hip_side, track_femur, hip_features_canonical
    h, w = img_u8.shape
    mask = segment_bone_hip(img_u8)
    side = detect_hip_side(img_u8, mask)
    mirrored = side != "right"
    mask_c = mask if not mirrored else np.ascontiguousarray(mask[:, ::-1])

    # Рисуем геометрию (контур, треки, вертикальные линии — симметричны при
    # отражении) в канонической (зеркальной) системе координат, затем
    # отражаем ТОЛЬКО геометрию обратно. Текст добавляем ПОСЛЕ отражения,
    # в исходной ориентации кадра — иначе буквы получаются зеркальными.
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    cnts, _ = cv2.findContours(mask_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, cnts, -1, COL_BONE_CONTOUR, 1)

    tr = track_femur(mask_c)
    f = hip_features_canonical(mask_c)
    if tr is not None:
        pts = np.stack([tr["left"], tr["y"]], axis=1).astype(np.int32)
        cv2.polylines(canvas, [pts], False, COL_AXIS, 1, cv2.LINE_AA)
        pts = np.stack([tr["right"], tr["y"]], axis=1).astype(np.int32)
        cv2.polylines(canvas, [pts], False, COL_AXIS, 1, cv2.LINE_AA)

    # ROI-отступ от латерального края кадра (физический критерий ТЗ >= 2 см)
    edge_mm = feats.get("edge_distance_mm")
    lateral_margin_mm = feats.get("lateral_margin_mm")
    margin_mm = lateral_margin_mm if lateral_margin_mm is not None else edge_mm
    roi_color = None
    if margin_mm is not None:
        margin_px = int(round(margin_mm / PIXEL_SPACING_X_MM))
        roi_ok = margin_mm >= 20.0
        roi_color = COL_ROI_OK if roi_ok else COL_ROI_BAD
        cv2.line(canvas, (margin_px, 0), (margin_px, h - 1), roi_color, 1, cv2.LINE_AA)

    # geometry-only mask of drawn pixels, so compositing doesn't touch untouched areas
    drawn = (canvas.sum(axis=2) > 0)
    geometry_layer = canvas if not mirrored else np.ascontiguousarray(canvas[:, ::-1])
    drawn = drawn if not mirrored else np.ascontiguousarray(drawn[:, ::-1])
    col[drawn] = geometry_layer[drawn]

    # --- текст рисуется отдельно, на панели снизу (см. render_overlay) ---
    merge_h = feats.get("merge_height_mm")
    shaft_len = feats.get("shaft_len_below_troch_mm")

    labels = []
    angle = feats.get("abs_shaft_angle_deg") or feats.get("shaft_angle_deg")
    if angle is not None:
        labels.append((f"Угол диафиза: {angle:.1f}°", (200, 200, 0)))
    labels.append((f"Сторона (детект.): {'правое' if side == 'right' else 'левое'}", (200, 200, 0)))
    if margin_mm is not None:
        labels.append((f"Край ROI до кости: {margin_mm:.0f} мм", roi_color))
    if merge_h is not None and shaft_len is not None:
        shaft_ok = shaft_len >= 30.0
        color = COL_ROI_OK if shaft_ok else COL_ROI_BAD
        labels.append((f"Диафиз ниже вертела: {shaft_len:.0f} мм", color))
    return col, labels


DISCLAIMER_TEXT = ("DensitoAI - вспомогательная оценка качества укладки. "
                   "Не медицинское изделие, не диагноз.")
COL_DISCLAIMER = (170, 170, 170)  # BGR, серый


def render_overlay(img_u8: np.ndarray, region: str, feats: Dict[str, Any],
                    crit_results: Optional[Dict[str, Any]] = None,
                    quality_class: Optional[int] = None,
                    violation_type: str = "") -> np.ndarray:
    """Строит цветную (BGR) визуализацию измеренных геометрических примитивов
    поверх исходного снимка (вместо тепловой карты сети — см. докстринг модуля)
    для архитектуры Контур A + Контур B.

    Все текстовые подписи (заголовок НАРУШЕНИЕ/тип нарушения, измеренные
    величины) рисуются НЕ поверх самого рентгеновского снимка, а на отдельной
    чёрной панели снизу, шириной во всю картинку, с переносом строк по фактической
    ширине текста. Это гарантирует, что текст никогда не обрезается за край
    и не сливается с цветными линиями геометрии (важно для маленьких DICOM,
    где исходное изображение может быть всего ~200-300px шириной)."""
    col = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    if region == "spine":
        col, geo_labels = _render_spine_overlay(col, img_u8, feats, crit_results)
    else:
        col, geo_labels = _render_hip_overlay(col, img_u8, feats, crit_results)

    h, w = img_u8.shape

    # --- собираем все подписи для нижней панели ---
    # К3: если хотя бы один критерий попал в зону «не уверен», на картинке это должно быть
    # видно так же, как в карточке кабинета и в колонке needs_review debug-CSV: иначе оверлей
    # уверенно пишет «БЕЗ НАРУШЕНИЙ» там, где сервис просит просмотр человека.
    uncertain_any = bool(crit_results) and any(int(r.get("uncertain", 0) or 0) for r in crit_results.values())
    header = "НАРУШЕНИЕ" if quality_class else "БЕЗ НАРУШЕНИЙ"
    header_color = COL_VIOLATION_TEXT if quality_class else COL_OK_TEXT
    if uncertain_any:
        header = f"{header} · НЕ УВЕРЕН, НУЖЕН ПРОСМОТР"
        header_color = COL_UNCERTAIN_TEXT
    panel_lines = [(header, header_color, 0.44)]
    if violation_type:
        # в CSV-выводе разделитель ";" без пробела (спецификация ТЗ) — для
        # отображения на картинке добавляем пробел после ";", иначе два нарушения
        # сливаются в одно длинное "слово" и не переносятся корректно.
        display_violation = violation_type.replace(";", "; ")
        panel_lines.append((display_violation, COL_VIOLATION_TEXT, 0.36))
    for text, color in geo_labels:
        panel_lines.append((text, color, 0.36))
    # оценки модели по каждому критерию — чтобы решение было прозрачным и не
    # противоречило измеренным величинам (решение принимает стек геометрия+эмбеддинги,
    # а не эвристический порог по одной величине)
    if crit_results:
        for crit, r in crit_results.items():
            if r.get("score") is None:
                continue
            name = CRIT_SHORT.get(crit, crit)
            thr = r.get("threshold", 0.5)
            flagged = bool(r.get("flag"))
            unc = bool(int(r.get("uncertain", 0) or 0))
            mark = "НАРУШЕНИЕ" if flagged else "норма"
            if unc:
                mark = f"{mark} (не уверен)"
            color = COL_UNCERTAIN_TEXT if unc else (COL_VIOLATION_TEXT if flagged else COL_OK_TEXT)
            panel_lines.append((f"Модель · {name}: {r['score']:.2f} / порог {thr:.2f} → {mark}", color, 0.34))

    # предупреждающая надпись в пикселях: результат ИИ вспомогательный, не диагноз.
    # Требование к отображению результатов ИИ; дублируется в DICOM SR и в веб-кабинете.
    panel_lines.append((DISCLAIMER_TEXT, COL_DISCLAIMER, 0.30))

    # --- вычисляем высоту панели с учётом переноса строк ---
    pad_x = 6
    max_text_w = w - 2 * pad_x
    line_gap = 4
    total_lines = 0
    wrapped_lines = []  # (line_text, color, scale)
    for text, color, scale in panel_lines:
        words = text.split(" ")
        cur = ""
        sub_lines = []
        for word in words:
            trial = (cur + " " + word).strip()
            (tw, _), _ = cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            if tw > max_text_w and cur:
                sub_lines.append(cur)
                cur = word
            else:
                cur = trial
        if cur:
            sub_lines.append(cur)
        for sl in sub_lines:
            wrapped_lines.append((sl, color, scale))
        total_lines += len(sub_lines)

    line_h = 15  # px на строку (с запасом под шрифт scale~0.36-0.44)
    panel_h = max(1, total_lines) * line_h + 2 * pad_x

    panel = np.zeros((panel_h, w, 3), dtype=np.uint8)
    y = pad_x + 10
    for text, color, scale in wrapped_lines:
        cv2.putText(panel, text, (pad_x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += line_h

    out = np.vstack([col, panel])
    return out


def save_overlay_png(path: str, img_u8: np.ndarray, region: str, feats: Dict[str, Any],
                      crit_results: Optional[Dict[str, Any]] = None,
                      quality_class: Optional[int] = None, violation_type: str = "") -> str:
    col = render_overlay(img_u8, region, feats, crit_results, quality_class, violation_type)
    cv2.imwrite(path, col)
    return path


def overlay_to_dicom_sc(overlay_bgr: np.ndarray, ref_ds, series_description: str = "DensitoAI QC Overlay",
                        model_version: str = "", config_hash: str = ""):
    """Оборачивает цветную визуализацию в DICOM Secondary Capture (SC) — бонус ТЗ п.2.6
    («серия с визуализацией»): эксперт открывает результат в обычном DICOM-вьюере рядом
    с исходной серией.

    Свойства файла, важные для приёмки:
      * идентификаторы пациента/исследования наследуются из исходного DICOM;
      * SOP Instance UID детерминирован от исходного SOP UID и sha256 пикселей оверлея,
        Series UID — от исследования, версии модели и config_hash: повторный прогон даёт
        тот же файл, а не новую серию (иначе PACS засоряется дублями);
      * ImageType = DERIVED/SECONDARY/OTHER, SourceImageSequence ссылается на исходный снимок,
        DerivationDescription описывает происхождение;
      * BurnedInAnnotation = YES: в пикселях есть подписи и предупреждение о том, что это
        вспомогательная оценка, а не диагноз.
    """
    import hashlib
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.uid import SecondaryCaptureImageStorage, generate_uid
    from pydicom.sequence import Sequence

    h, w = overlay_bgr.shape[:2]
    rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    pix = rgb.tobytes()

    def _det_uid(*parts):
        return generate_uid(entropy_srcs=[str(x) for x in parts])

    src_sop = str(getattr(ref_ds, "SOPInstanceUID", "") or "")
    src_class = str(getattr(ref_ds, "SOPClassUID", "") or "")
    study_uid = str(getattr(ref_ds, "StudyInstanceUID", "") or "")
    pix_sha = hashlib.sha256(pix).hexdigest()

    sop_uid = _det_uid("densito-sc-instance", src_sop or pix_sha, pix_sha)
    series_uid = _det_uid("densito-sc-series", study_uid or src_sop or pix_sha, model_version, config_hash)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = sop_uid
    ds.SeriesInstanceUID = series_uid
    ds.Modality = "OT"
    ds.ConversionType = "WSD"
    ds.SeriesDescription = series_description
    ds.Manufacturer = "DensitoAI"
    ds.ManufacturerModelName = "DensitoAI DXA QC"
    if model_version:
        ds.SoftwareVersions = str(model_version)
    ds.ImageType = ["DERIVED", "SECONDARY", "OTHER"]
    ds.DerivationDescription = ("DensitoAI QC overlay: geometricheskie primitivy i veroyatnosti "
                               "kriteriev kachestva; ne meditsinskoe izdelie, ne diagnoz")
    ds.BurnedInAnnotation = "YES"
    ds.SeriesNumber = 9001
    ds.InstanceNumber = int(getattr(ref_ds, "InstanceNumber", 1) or 1)

    # даты — из исходного исследования: файл должен быть одинаковым при повторном прогоне
    for attr, src in (("StudyDate", "StudyDate"), ("StudyTime", "StudyTime"),
                      ("SeriesDate", "StudyDate"), ("SeriesTime", "StudyTime"),
                      ("ContentDate", "StudyDate"), ("ContentTime", "StudyTime")):
        v = getattr(ref_ds, src, None)
        if v:
            setattr(ds, attr, v)
    if not getattr(ds, "StudyDate", None):
        ds.StudyDate = "19000101"
        ds.StudyTime = "000000"
        ds.ContentDate = ds.StudyDate
        ds.ContentTime = ds.StudyTime

    for attr in ("PatientName", "PatientID", "PatientBirthDate", "PatientSex",
                 "StudyInstanceUID", "StudyID", "AccessionNumber", "BodyPartExamined"):
        if hasattr(ref_ds, attr):
            setattr(ds, attr, getattr(ref_ds, attr))
    if not getattr(ds, "StudyInstanceUID", None):
        ds.StudyInstanceUID = _det_uid("densito-sc-study", pix_sha)

    if src_sop and src_class:
        src_item = Dataset()
        src_item.ReferencedSOPClassUID = src_class
        src_item.ReferencedSOPInstanceUID = src_sop
        ds.SourceImageSequence = Sequence([src_item])
        ds.ReferencedImageSequence = Sequence([src_item])

    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.PlanarConfiguration = 0
    ds.Rows, ds.Columns = h, w
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = pix
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    return ds


def save_overlay_sc(path: str, overlay_bgr: np.ndarray, ref_ds, model_version: str = "",
                    config_hash: str = "") -> str:
    """Пишет DICOM SC с визуализацией на диск (см. overlay_to_dicom_sc)."""
    ds = overlay_to_dicom_sc(overlay_bgr, ref_ds, model_version=model_version, config_hash=config_hash)
    ds.save_as(path, enforce_file_format=True)
    return path


if __name__ == "__main__":
    import sys
    from geometry_features import read_dicom_normalized, extract_all_features
    fp = sys.argv[1] if len(sys.argv) > 1 else None
    region = sys.argv[2] if len(sys.argv) > 2 else "spine"
    if fp:
        img_u8, ds = read_dicom_normalized(fp)
        feats = extract_all_features(fp, region if region == "spine" else "hip")
        out = save_overlay_png("/tmp/overlay_test.png", img_u8, region, feats)
        print("saved:", out)
