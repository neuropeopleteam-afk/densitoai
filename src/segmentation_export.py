#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Экспорт сегментации структур (бонус ТЗ «сегментация», дополнительный функционал).

Решение уже считает внутри маски, по которым строятся признаки качества укладки:
  * кость (geometry_features.segment_bone — позвоночник, hip_features.segment_bone_hip — бедро);
  * посторонние предметы / металл вне кости (та же логика, что в
    geometry_features.foreign_object_features, но здесь возвращается сама маска, а не статистика);
  * поле сканирования — тело пациента в кадре (preprocess.body_mask);
  * область интереса бедра — прямоугольник текущего контура кости (как bone_box_px в auto_roi).

Модуль отдаёт эти маски наружу тремя файлами на снимок:
  (а) DICOM Segmentation (SEG, SOP Class 1.2.840.10008.5.1.4.1.1.66.4), BINARY, по одному сегменту
      и кадру на структуру, SegmentSequence с описаниями по-русски, ссылка на исходный снимок
      (ReferencedSeriesSequence + SourceImageSequence), детерминированные UID (dicom_sr.deterministic_uid):
      повторный прогон даёт байт-в-байт тот же файл;
  (б) PNG-маска RGBA с цветовой кодировкой (прозрачный фон — удобно накладывать поверх снимка),
      легенда цветов — в JSON;
  (в) JSON с полигонами контуров в пикселях и миллиметрах
      (pixel spacing как в geometry_features: Y 1.05 мм, X 0.6 мм), площадями и легендой.

Зависимости — только numpy, cv2, pydicom из образа (highdicom не используется: образ офлайн).
Эталонной разметки структур в датасете нет, поэтому качество масок здесь не измеряется;
их устойчивость к преобразованиям снимка проверяет tools/seg_stability.py.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from geometry_features import PIXEL_SPACING_X_MM, PIXEL_SPACING_Y_MM, segment_bone
from hip_features import segment_bone_hip
import preprocess
from dicom_sr import deterministic_uid, is_valid_uid, SERVICE_NAME

SEG_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.66.4"   # Segmentation Storage
SEG_MODULE_VERSION = "1"                              # версия схемы сегментов (входит в UID серии)
CODING_SCHEME = "99DENSITO"                           # локальная схема кодов, как в dicom_sr
SEG_SERIES_NUMBER = 9002                              # 9001 занят серией визуализации (SC)

# Структуры по области: (ключ, подпись по-русски, описание, категория, RGB, как рисовать в PNG)
# Категория: "anatomy" — анатомическая структура, "object" — физический объект, "region" — область кадра.
STRUCTURES: Dict[str, List[Dict[str, Any]]] = {
    "spine": [
        {"key": "bone", "label": "Кость: поясничный отдел позвоночника",
         "description": "Маска кости: Otsu, морфология, наибольшая компонента; по ней считаются ось и центрирование",
         "short": "Маска кости позвоночника (Otsu, морфология)",
         "category": "anatomy", "rgb": (255, 196, 0), "draw": "fill"},
        {"key": "foreign", "label": "Посторонние предметы (металл)",
         "description": "Плотные компактные объекты вне кости, заметно ярче окружающих мягких тканей",
         "short": "Плотные объекты вне кости ярче мягких тканей",
         "category": "object", "rgb": (255, 40, 40), "draw": "fill"},
    ],
    "hip": [
        {"key": "bone", "label": "Кость: проксимальный отдел бедра",
         "description": "Маска кости бедра (порог Otsu по телу, морфология, наибольшая компонента)",
         "short": "Маска кости бедра (Otsu по телу, морфология)",
         "category": "anatomy", "rgb": (255, 196, 0), "draw": "fill"},
        {"key": "scan_field", "label": "Поле сканирования (тело в кадре)",
         "description": "Пиксели тела пациента: область, реально покрытая сканом",
         "short": "Тело пациента в кадре (область, покрытая сканом)",
         "category": "region", "rgb": (60, 150, 255), "draw": "fill"},
        {"key": "roi", "label": "Область интереса (контур кости)",
         "description": "Прямоугольник текущего контура кости, относительно которого оцениваются отступы области интереса",
         "short": "Прямоугольник текущего контура кости",
         "category": "region", "rgb": (80, 220, 120), "draw": "outline"},
    ],
}
PNG_FILL_ALPHA = 150       # прозрачность заливки в PNG (0..255)
PNG_OUTLINE_PX = 2         # толщина контура для структур типа outline


def region_family(region: str) -> str:
    """Внутренний регион инференса (spine / right_hip / left_hip) -> семейство структур."""
    return "spine" if str(region) == "spine" else "hip"


# --------------------------------------------------------------------------- #
# Маски
# --------------------------------------------------------------------------- #
def foreign_object_mask(img_u8: np.ndarray, bone_mask: np.ndarray) -> np.ndarray:
    """Маска посторонних предметов вне кости — та же последовательность операций, что в
    geometry_features.foreign_object_features (там наружу отдаётся только статистика):
    фон — кольцо мягких тканей вокруг кости, кандидаты — ярче фона на 3 сигмы, вне кости,
    внутри тела, open 3x3 + close 3x3 x2, компоненты меньше 4 px отбрасываются."""
    mask = (np.asarray(bone_mask) > 0).astype(np.uint8) * 255
    out = np.zeros(img_u8.shape, dtype=bool)
    kernel_big = np.ones((25, 25), np.uint8)
    bone_dilated = cv2.dilate(mask, kernel_big, iterations=1)
    soft_band = (bone_dilated > 0) & (mask == 0)
    body_thresh = 8
    soft_px = img_u8[soft_band]
    soft_px = soft_px[soft_px > body_thresh]
    if len(soft_px) < 20:
        return out
    bg_mean = float(np.mean(soft_px))
    bg_std = float(np.std(soft_px)) + 1e-6
    bright = bg_mean + 3.0 * bg_std
    candidate = ((img_u8.astype(np.float32) > bright) & (mask == 0) & (img_u8 > body_thresh)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] < 4:
            continue
        out |= labels == i
    return out


def scan_field_mask(img_u8: np.ndarray) -> np.ndarray:
    """Поле сканирования: маска тела (preprocess.body_mask), наибольшая компонента, дыры заполнены."""
    body = np.asarray(preprocess.body_mask(img_u8)).astype(np.uint8)
    if int(body.sum()) == 0:
        return np.zeros(img_u8.shape, dtype=bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(body)
    if n <= 1:
        return np.zeros(img_u8.shape, dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (labels == largest).astype(np.uint8) * 255
    # заполнение дыр: всё, что не достижимо от рамки кадра, считается телом.
    # Заливка идёт из кадра, расширенного нулевой рамкой, поэтому результат не зависит от того,
    # закрыт ли телом угол (0, 0), и симметричен относительно переворота.
    h, w = comp.shape
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1:-1, 1:-1] = comp
    flood_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(padded, flood_mask, (0, 0), 255)
    holes = padded[1:-1, 1:-1] == 0
    return (comp > 0) | holes


def roi_box_mask(bone_mask: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """Прямоугольник текущего контура кости (x0, y0, x1, y1) и его заливка как маска."""
    ys, xs = np.nonzero(bone_mask)
    out = np.zeros(bone_mask.shape, dtype=bool)
    if len(ys) < 5:
        return out, None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    out[y0:y1 + 1, x0:x1 + 1] = True
    return out, (x0, y0, x1, y1)


def build_masks(img_u8: np.ndarray, region: str) -> "OrderedDict[str, np.ndarray]":
    """Маски структур для снимка (uint8 0..255 baseline-кадр инференса) и региона
    (spine / right_hip / left_hip или spine / hip). Порядок ключей = порядок сегментов в SEG."""
    img_u8 = np.ascontiguousarray(np.asarray(img_u8, dtype=np.uint8))
    fam = region_family(region)
    masks: "OrderedDict[str, np.ndarray]" = OrderedDict()
    if fam == "spine":
        bone = segment_bone(img_u8) > 0
        masks["bone"] = bone
        masks["foreign"] = foreign_object_mask(img_u8, bone)
    else:
        bone = segment_bone_hip(img_u8) > 0
        masks["bone"] = bone
        masks["scan_field"] = scan_field_mask(img_u8)
        masks["roi"], _ = roi_box_mask(bone)
    for k in list(masks):
        masks[k] = np.ascontiguousarray(masks[k].astype(bool))
    return masks


def structures_for(region: str) -> List[Dict[str, Any]]:
    return [dict(s) for s in STRUCTURES[region_family(region)]]


# --------------------------------------------------------------------------- #
# Контуры и площади
# --------------------------------------------------------------------------- #
def mask_polygons(mask: np.ndarray, min_area_px: int = 4) -> List[Dict[str, Any]]:
    """Внешние контуры маски: полигоны в пикселях и мм (x*0.6, y*1.05)."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: List[Dict[str, Any]] = []
    for c in contours:
        area = float(cv2.contourArea(c))
        pts = c.reshape(-1, 2)
        if len(pts) < 3 or area < min_area_px:
            continue
        px = [[int(x), int(y)] for x, y in pts]
        mm = [[round(float(x) * PIXEL_SPACING_X_MM, 2), round(float(y) * PIXEL_SPACING_Y_MM, 2)] for x, y in pts]
        x, y, w, h = cv2.boundingRect(c)
        polys.append({"area_px": area, "area_mm2": round(area * PIXEL_SPACING_X_MM * PIXEL_SPACING_Y_MM, 2),
                      "bbox_px": [int(x), int(y), int(x + w - 1), int(y + h - 1)],
                      "points_px": px, "points_mm": mm})
    polys.sort(key=lambda p: (-p["area_px"], p["bbox_px"]))
    return polys


def mask_stats(mask: np.ndarray) -> Dict[str, Any]:
    n = int(np.count_nonzero(mask))
    ys, xs = np.nonzero(mask)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if n else None
    return {"area_px": n, "area_mm2": round(n * PIXEL_SPACING_X_MM * PIXEL_SPACING_Y_MM, 2),
            "bbox_px": bbox, "empty": n == 0}


# --------------------------------------------------------------------------- #
# PNG
# --------------------------------------------------------------------------- #
def render_png(masks: "OrderedDict[str, np.ndarray]", region: str) -> np.ndarray:
    """RGBA-маска: заливка полупрозрачным цветом структуры, для outline — контур. Прозрачный фон."""
    first = next(iter(masks.values()))
    h, w = first.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # порядок отрисовки: сначала крупные области кадра, затем анатомия и объекты, контуры поверх всего
    order = {"region": 0, "anatomy": 1, "object": 2}
    structs = sorted(structures_for(region), key=lambda s: (s["draw"] == "outline", order.get(s["category"], 3)))
    for s in structs:
        m = masks.get(s["key"])
        if m is None or not m.any():
            continue
        r, g, b = s["rgb"]
        if s["draw"] == "outline":
            edge = np.zeros((h, w), np.uint8)
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(edge, contours, -1, 255, PNG_OUTLINE_PX)
            sel = (edge > 0) & m      # линия контура целиком внутри структуры
            alpha = 255
        else:
            sel = m
            alpha = PNG_FILL_ALPHA
        rgba[sel] = (r, g, b, alpha)
    return rgba


def write_png(path: str, rgba: np.ndarray) -> str:
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    ok = cv2.imwrite(str(path), bgra)
    if not ok:
        raise IOError(f"cv2.imwrite не смог записать {path}")
    return str(path)


def legend(region: str) -> List[Dict[str, Any]]:
    out = []
    for i, s in enumerate(structures_for(region), start=1):
        out.append({"segment_number": i, "key": s["key"], "label": s["label"], "description": s["description"],
                    "category": s["category"], "rgb": list(s["rgb"]),
                    "png_draw": s["draw"], "png_alpha": 255 if s["draw"] == "outline" else PNG_FILL_ALPHA})
    return out


# --------------------------------------------------------------------------- #
# DICOM SEG
# --------------------------------------------------------------------------- #
def _code(value: str, meaning: str, scheme: str = CODING_SCHEME):
    from pydicom.dataset import Dataset
    d = Dataset()
    d.CodeValue = str(value)[:16]
    d.CodingSchemeDesignator = scheme
    d.CodeMeaning = str(meaning)[:64]
    return d


def _cielab_u16(rgb: Tuple[int, int, int]) -> List[int]:
    """RGB -> CIELab в кодировке DICOM (RecommendedDisplayCIELabValue, 3 x US)."""
    px = np.array([[list(rgb)]], dtype=np.uint8)
    lab = cv2.cvtColor(px, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float64)   # L 0..255, a,b 0..255 (сдвиг 128)
    L = lab[0] / 255.0 * 100.0
    a = lab[1] - 128.0
    b = lab[2] - 128.0
    return [int(round(L / 100.0 * 65535)), int(round((a + 128.0) / 255.0 * 65535)), int(round((b + 128.0) / 255.0 * 65535))]


_CATEGORY_CODES = {
    "anatomy": ("SEGCAT-ANAT", "Анатомическая структура"),
    "object": ("SEGCAT-OBJ", "Физический объект"),
    "region": ("SEGCAT-REG", "Область кадра"),
}


def pack_frames(masks: List[np.ndarray]) -> bytes:
    """Упаковка кадров в 1 бит/пиксель (LSB первым, кадры подряд без выравнивания, как в PS3.5 8.1.1)."""
    flat = np.concatenate([np.asarray(m, dtype=bool).ravel() for m in masks]).astype(np.uint8)
    packed = np.packbits(flat, bitorder="little").tobytes()
    if len(packed) % 2:
        packed += b"\x00"
    return packed


def unpack_frames(pixel_data: bytes, n_frames: int, rows: int, cols: int) -> np.ndarray:
    """Обратная операция для pack_frames -> bool (n_frames, rows, cols)."""
    bits = np.unpackbits(np.frombuffer(pixel_data, dtype=np.uint8), bitorder="little")
    need = n_frames * rows * cols
    if bits.size < need:
        raise ValueError(f"PixelData короче ожидаемого: {bits.size} бит < {need}")
    return bits[:need].reshape(n_frames, rows, cols).astype(bool)


def _copy_tags(dst, src, names: Tuple[str, ...]) -> None:
    for n in names:
        try:
            v = getattr(src, n, None)
        except Exception:  # noqa: BLE001
            v = None
        if v not in (None, ""):
            try:
                setattr(dst, n, v)
            except Exception:  # noqa: BLE001
                pass


def build_seg_dataset(masks: "OrderedDict[str, np.ndarray]", region: str, ref_ds,
                      image_uid: str = "", study_uid: str = "", model_version: str = "",
                      config_hash: str = ""):
    """Собирает pydicom FileDataset DICOM SEG (BINARY, один кадр на структуру).

    Детерминизм: даты берутся из исходного снимка (StudyDate/StudyTime; иначе 19000101), UID серии —
    от UID снимка, версии модели, config_hash и версии схемы сегментов; UID экземпляра — дополнительно
    от sha256 упакованных масок и подписей сегментов. Повторный прогон даёт тот же файл."""
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.sequence import Sequence

    structs = structures_for(region)
    frames = [np.asarray(masks[s["key"]], dtype=bool) for s in structs]
    rows, cols = frames[0].shape
    for f in frames:
        if f.shape != (rows, cols):
            raise ValueError("маски структур разного размера")

    src_sop = str(getattr(ref_ds, "SOPInstanceUID", "") or "") if ref_ds is not None else ""
    src_class = str(getattr(ref_ds, "SOPClassUID", "") or "") if ref_ds is not None else ""
    src_series = str(getattr(ref_ds, "SeriesInstanceUID", "") or "") if ref_ds is not None else ""
    src_study = str(getattr(ref_ds, "StudyInstanceUID", "") or "") if ref_ds is not None else ""
    image_uid = str(image_uid or src_sop or "")
    study_uid = str(study_uid or src_study or "")

    pixel_bytes = pack_frames(frames)
    digest = hashlib.sha256(pixel_bytes + "|".join(s["label"] for s in structs).encode("utf-8")).hexdigest()
    series_uid = deterministic_uid("densito-seg-series", image_uid or digest, model_version, config_hash, SEG_MODULE_VERSION)
    sop_uid = deterministic_uid("densito-seg-instance", image_uid or digest, digest, model_version, SEG_MODULE_VERSION)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SEG_SOP_CLASS_UID
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.SpecificCharacterSet = "ISO_IR 192"
    ds.SOPClassUID = SEG_SOP_CLASS_UID
    ds.SOPInstanceUID = sop_uid
    ds.Modality = "SEG"
    ds.ImageType = ["DERIVED", "PRIMARY"]
    ds.Manufacturer = SERVICE_NAME
    ds.ManufacturerModelName = f"{SERVICE_NAME} DXA QC"
    if model_version:
        ds.SoftwareVersions = str(model_version)
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = SEG_SERIES_NUMBER
    ds.InstanceNumber = 1
    ds.SeriesDescription = "DensitoAI: сегментация структур"
    ds.ContentLabel = "DENSITO_SEG"
    ds.ContentDescription = "Сегментация структур DXA для контроля укладки"
    ds.ContentCreatorName = SERVICE_NAME
    ds.SegmentationType = "BINARY"
    ds.LossyImageCompression = "00"

    # идентификаторы пациента/исследования — из исходного снимка
    if ref_ds is not None:
        _copy_tags(ds, ref_ds, ("PatientName", "PatientID", "PatientBirthDate", "PatientSex",
                                "StudyInstanceUID", "StudyDate", "StudyTime", "StudyID",
                                "AccessionNumber", "ReferringPhysicianName", "StudyDescription",
                                "FrameOfReferenceUID", "PositionReferenceIndicator"))
    if not getattr(ds, "StudyInstanceUID", None):
        ds.StudyInstanceUID = study_uid if is_valid_uid(study_uid) else deterministic_uid("densito-seg-study", study_uid or image_uid or digest)
    if not getattr(ds, "StudyDate", None):
        ds.StudyDate = "19000101"
        ds.StudyTime = "000000"
    ds.SeriesDate = ds.StudyDate
    ds.SeriesTime = getattr(ds, "StudyTime", "000000") or "000000"
    ds.ContentDate = ds.StudyDate
    ds.ContentTime = ds.SeriesTime
    for n in ("PatientName", "PatientID", "PatientBirthDate", "PatientSex", "ReferringPhysicianName",
              "AccessionNumber", "StudyID", "PositionReferenceIndicator"):
        if not hasattr(ds, n):
            setattr(ds, n, "")
    if not getattr(ds, "FrameOfReferenceUID", None):
        ds.FrameOfReferenceUID = deterministic_uid("densito-seg-for", image_uid or digest)

    # ссылка на исходный снимок
    ref_inst = Dataset()
    ref_inst.ReferencedSOPClassUID = src_class or "1.2.840.10008.5.1.4.1.1.7"
    ref_inst.ReferencedSOPInstanceUID = image_uid if is_valid_uid(image_uid) else deterministic_uid("densito-seg-src", image_uid or digest)
    ref_series = Dataset()
    ref_series.SeriesInstanceUID = src_series if is_valid_uid(src_series) else deterministic_uid("densito-seg-src-series", study_uid or image_uid or digest)
    ref_series.ReferencedInstanceSequence = Sequence([ref_inst])
    ds.ReferencedSeriesSequence = Sequence([ref_series])

    # пиксельный модуль
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = int(rows)
    ds.Columns = int(cols)
    ds.BitsAllocated = 1
    ds.BitsStored = 1
    ds.HighBit = 0
    ds.PixelRepresentation = 0
    ds.NumberOfFrames = len(frames)
    ds.PixelSpacing = [str(PIXEL_SPACING_Y_MM), str(PIXEL_SPACING_X_MM)]
    ds.ImageOrientationPatient = ["1", "0", "0", "0", "1", "0"]

    # сегменты
    seg_seq = []
    for i, s in enumerate(structs, start=1):
        seg = Dataset()
        seg.SegmentNumber = i
        seg.SegmentLabel = s["label"][:64]
        seg.SegmentDescription = s["short"]   # LO, не длиннее 64 символов; полный текст — в JSON-легенде
        seg.SegmentAlgorithmType = "AUTOMATIC"
        seg.SegmentAlgorithmName = f"{SERVICE_NAME} {model_version}".strip()[:64]
        cat_val, cat_mean = _CATEGORY_CODES[s["category"]]
        seg.SegmentedPropertyCategoryCodeSequence = Sequence([_code(cat_val, cat_mean)])
        seg.SegmentedPropertyTypeCodeSequence = Sequence([_code("SEG-" + s["key"].upper()[:12], s["label"])])
        seg.RecommendedDisplayCIELabValue = _cielab_u16(s["rgb"])
        seg_seq.append(seg)
    ds.SegmentSequence = Sequence(seg_seq)

    # организация измерений: один индекс — номер сегмента
    dim_org_uid = deterministic_uid("densito-seg-dimorg", sop_uid)
    dim_org = Dataset()
    dim_org.DimensionOrganizationUID = dim_org_uid
    ds.DimensionOrganizationSequence = Sequence([dim_org])
    dim_idx = Dataset()
    dim_idx.DimensionOrganizationUID = dim_org_uid
    dim_idx.DimensionIndexPointer = 0x0062000B          # ReferencedSegmentNumber
    dim_idx.FunctionalGroupPointer = 0x0062000A         # SegmentIdentificationSequence
    dim_idx.DimensionDescriptionLabel = "Segment number"
    ds.DimensionIndexSequence = Sequence([dim_idx])

    # общие функциональные группы
    shared = Dataset()
    pm = Dataset()
    pm.PixelSpacing = ds.PixelSpacing
    shared.PixelMeasuresSequence = Sequence([pm])
    po = Dataset()
    po.ImageOrientationPatient = ds.ImageOrientationPatient
    shared.PlaneOrientationSequence = Sequence([po])
    ds.SharedFunctionalGroupsSequence = Sequence([shared])

    # покадровые функциональные группы
    per_frame = []
    for i, s in enumerate(structs, start=1):
        fg = Dataset()
        fc = Dataset()
        fc.DimensionIndexValues = [i]
        fc.StackID = "1"
        fc.InStackPositionNumber = i
        fg.FrameContentSequence = Sequence([fc])
        si = Dataset()
        si.ReferencedSegmentNumber = i
        fg.SegmentIdentificationSequence = Sequence([si])
        der = Dataset()
        der.DerivationCodeSequence = Sequence([_code("113076", "Segmentation", "DCM")])
        src = Dataset()
        src.ReferencedSOPClassUID = ref_inst.ReferencedSOPClassUID
        src.ReferencedSOPInstanceUID = ref_inst.ReferencedSOPInstanceUID
        src.PurposeOfReferenceCodeSequence = Sequence([_code("121322", "Source image for image processing operation", "DCM")])
        der.SourceImageSequence = Sequence([src])
        fg.DerivationImageSequence = Sequence([der])
        per_frame.append(fg)
    ds.PerFrameFunctionalGroupsSequence = Sequence(per_frame)

    ds.PixelData = pixel_bytes
    return ds


def write_seg(path: str, ds) -> str:
    ds.save_as(str(path), enforce_file_format=True)
    return str(path)


# --------------------------------------------------------------------------- #
# Общий сценарий: снимок + регион -> три файла
# --------------------------------------------------------------------------- #
def export_segmentation(img_u8: np.ndarray, region: str, ref_ds, out_dir: str, stem: str,
                        image_uid: str = "", study_uid: str = "", model_version: str = "",
                        config_hash: str = "", source_path: str = "",
                        masks: Optional["OrderedDict[str, np.ndarray]"] = None) -> Dict[str, Any]:
    """Пишет <stem>_seg.dcm, <stem>_seg.png, <stem>_seg.json в out_dir. Возвращает пути и сводку."""
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    if masks is None:
        masks = build_masks(img_u8, region)
    structs = structures_for(region)
    rows, cols = next(iter(masks.values())).shape

    ds = build_seg_dataset(masks, region, ref_ds, image_uid=image_uid, study_uid=study_uid,
                           model_version=model_version, config_hash=config_hash)
    seg_path = write_seg(out_dir_p / f"{stem}_seg.dcm", ds)
    png_path = write_png(out_dir_p / f"{stem}_seg.png", render_png(masks, region))

    segments = []
    for i, s in enumerate(structs, start=1):
        m = masks[s["key"]]
        st = mask_stats(m)
        segments.append({"segment_number": i, "key": s["key"], "label": s["label"], "category": s["category"],
                         "rgb": list(s["rgb"]), **st, "polygons": mask_polygons(m)})
    payload = {
        "service": SERVICE_NAME, "model_version": model_version, "seg_schema_version": SEG_MODULE_VERSION,
        "source_path": str(source_path), "study_uid": str(study_uid), "image_uid": str(image_uid),
        "region": str(region), "structure_family": region_family(region),
        "rows": int(rows), "cols": int(cols),
        "pixel_spacing_mm": {"y": PIXEL_SPACING_Y_MM, "x": PIXEL_SPACING_X_MM},
        "coordinates": "points_px — [x, y] в пикселях исходного кадра (x вправо, y вниз); "
                       "points_mm — те же точки, умноженные на pixel spacing (x*0.6, y*1.05)",
        "seg_sop_instance_uid": str(ds.SOPInstanceUID), "seg_series_instance_uid": str(ds.SeriesInstanceUID),
        "files": {"seg_dcm": os.path.basename(seg_path), "png": os.path.basename(png_path)},
        "legend": legend(region),
        "segments": segments,
        "note": "Эталонной разметки структур в датасете нет: маски получены автоматически и служат "
                "для контроля укладки, а не для диагностики.",
    }
    json_path = out_dir_p / f"{stem}_seg.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=False), encoding="utf-8")
    return {"seg_dcm": seg_path, "png": png_path, "json": str(json_path),
            "n_segments": len(structs), "sop_instance_uid": str(ds.SOPInstanceUID),
            "areas_px": {s["key"]: int(np.count_nonzero(masks[s["key"]])) for s in structs}}


def read_seg_masks(path: str) -> Tuple[Any, np.ndarray]:
    """Читает SEG и распаковывает кадры -> (dataset, bool (n, rows, cols))."""
    import pydicom
    ds = pydicom.dcmread(str(path))
    n = int(getattr(ds, "NumberOfFrames", 1) or 1)
    return ds, unpack_frames(bytes(ds.PixelData), n, int(ds.Rows), int(ds.Columns))


def dice(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Коэффициент Dice двух бинарных масок; None, если обе пустые."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    s = int(a.sum()) + int(b.sum())
    if s == 0:
        return None
    return float(2.0 * np.count_nonzero(a & b) / s)


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Экспорт сегментации структур одного DICOM (SEG + PNG + JSON).")
    ap.add_argument("dicom", help="путь к DICOM-снимку")
    ap.add_argument("--region", default=None, help="spine / right_hip / left_hip (по умолчанию по ширине кадра)")
    ap.add_argument("--out", default="seg_out", help="каталог для файлов")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pydicom
    from inference import normalize_pixels, DEFAULT_CONFIG
    ds0 = pydicom.dcmread(a.dicom)
    img = normalize_pixels(ds0)
    reg = a.region or ("spine" if img.shape[1] >= DEFAULT_CONFIG["regions"]["spine_min_cols"] else "right_hip")
    res = export_segmentation(img, reg, ds0, a.out, Path(a.dicom).stem,
                              image_uid=str(getattr(ds0, "SOPInstanceUID", "")),
                              study_uid=str(getattr(ds0, "StudyInstanceUID", "")), source_path=a.dicom)
    print(json.dumps(res, ensure_ascii=False, indent=1))
