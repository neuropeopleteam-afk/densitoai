#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DICOM Structured Report (SR) для результата контроля качества (бонус ТЗ п.2.6,
портирование функциональности v1 dicom_sr.py в архитектуру densito_rebuild).

Формирует TID 1500-подобный (упрощённый, Basic Text SR — template
"Basic Diagnostic Imaging Report", SOP Class Comprehensive SR) документ на
исследование с:
  - текстовым заключением (аналог поля "Заключение" рентгенолога);
  - измеренными числовыми величинами Контура A как NUM content items
    (угол оси/диафиза, физические отступы ROI в мм) — используя единицы UCUM
    ("deg", "mm"), чтобы значения были машиночитаемы в любом PACS/просмотрщике;
  - закодированным выводом (CODE content item) quality_class и violation_type.

Это НЕ полноценный TID 1500 (Measurement Report) с формальными concept-name
кодами из DCID — для хакатона это избыточно и организаторы не давали свой
codebook кодов SNOMED/DCM. Вместо кодов из внешних словарей используются
текстовые CONTAINER/TEXT/NUM content items с понятными человеку названиями
(concept name как локальный private code + читаемое CodeMeaning) — документ
валиден как DICOM SR (SOP Class 1.2.840.10008.5.1.4.1.1.88.33, Comprehensive
SR Storage) и открывается стандартными вьюерами (Weasis, OHIF, RadiAnt),
показывая дерево находок.

Функция:
  build_sr(ref_ds, region, quality_class, violation_type, quality_prob, feats) -> pydicom Dataset
  save_sr(path, ref_ds, ...) -> path
"""
from __future__ import annotations

import datetime
from typing import Any, Dict, Optional

import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ComprehensiveSRStorage, generate_uid

DENSITO_CODING_SCHEME = "99DENSITO"  # локальная (private) coding scheme — не официальный реестр


def _code_item(code_value: str, code_meaning: str, scheme: str = DENSITO_CODING_SCHEME) -> Dataset:
    d = Dataset()
    d.CodeValue = code_value
    d.CodingSchemeDesignator = scheme
    d.CodeMeaning = code_meaning
    return d


def _content_item(relationship: str, value_type: str, concept_code: str, concept_meaning: str,
                   **kwargs) -> Dataset:
    item = Dataset()
    item.RelationshipType = relationship
    item.ValueType = value_type
    item.ConceptNameCodeSequence = Sequence([_code_item(concept_code, concept_meaning)])
    for k, v in kwargs.items():
        setattr(item, k, v)
    return item


def _text_item(relationship: str, concept_code: str, concept_meaning: str, text: str) -> Dataset:
    return _content_item(relationship, "TEXT", concept_code, concept_meaning, TextValue=str(text))


def _code_content_item(relationship: str, concept_code: str, concept_meaning: str,
                        value_code: str, value_meaning: str) -> Dataset:
    item = _content_item(relationship, "CODE", concept_code, concept_meaning)
    item.ConceptCodeSequence = Sequence([_code_item(value_code, value_meaning)])
    return item


def _num_item(relationship: str, concept_code: str, concept_meaning: str,
              value: float, unit: str) -> Dataset:
    item = _content_item(relationship, "NUM", concept_code, concept_meaning)
    meas = Dataset()
    meas.NumericValue = round(float(value), 3)
    meas.MeasurementUnitsCodeSequence = Sequence([_code_item(unit, unit, scheme="UCUM")])
    item.MeasuredValueSequence = Sequence([meas])
    return item


def _container_item(relationship: str, concept_code: str, concept_meaning: str,
                     children: list) -> Dataset:
    item = _content_item(relationship, "CONTAINER", concept_code, concept_meaning)
    item.ContinuityOfContent = "SEPARATE"
    item.ContentSequence = Sequence(children)
    return item


# Признаки Контура A -> (код, читаемое имя, единица UCUM), которые имеет
# смысл вынести в SR как измеренные величины (по региону).
_SPINE_MEASURES = [
    ("axis_angle_deg", "AXIS-ANGLE", "Угол оси позвоночника к вертикали кадра", "deg"),
    ("curvature", "CURVATURE", "Показатель кривизны центральной линии (сколиоз)", "1"),
    ("metal_outside_bone_mm2", "METAL-AREA", "Площадь посторонних объектов вне кости", "mm2"),
]
_HIP_MEASURES = [
    ("abs_shaft_angle_deg", "SHAFT-ANGLE", "Угол диафиза бедренной кости к вертикали", "deg"),
    ("lateral_margin_mm", "ROI-LATERAL-MARGIN", "Отступ ROI от латерального края кадра", "mm"),
    ("shaft_len_below_troch_mm", "SHAFT-BELOW-TROCH", "Длина диафиза ниже малого вертела в кадре", "mm"),
    ("merge_height_mm", "MERGE-HEIGHT", "Высота слияния диафиза с тазом", "mm"),
]


def build_sr(ref_ds, region: str, quality_class: int, violation_type: str,
             quality_prob: float, feats: Dict[str, Any],
             manufacturer: str = "DensitoAI") -> Dataset:
    """Строит DICOM Comprehensive SR (заключение контроля качества) для одного
    обработанного изображения, ссылающийся на исходный DICOM как источник."""
    now = datetime.datetime.now()

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = ComprehensiveSRStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.SOPClassUID = ComprehensiveSRStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = generate_uid()
    ds.SpecificCharacterSet = "ISO_IR 192"  # UTF-8 — обязательно для кириллицы в LO/SH/UT
    ds.Modality = "SR"
    ds.Manufacturer = manufacturer
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")
    ds.SeriesDescription = "DensitoAI Quality Control Report"
    ds.CompletionFlag = "COMPLETE"
    ds.VerificationFlag = "UNVERIFIED"  # автоматический анализ, не подписан врачом

    for attr in ("PatientName", "PatientID", "PatientBirthDate", "PatientSex",
                 "StudyInstanceUID", "StudyID", "AccessionNumber", "StudyDate", "StudyTime"):
        if hasattr(ref_ds, attr):
            setattr(ds, attr, getattr(ref_ds, attr))
    if not hasattr(ds, "StudyInstanceUID"):
        ds.StudyInstanceUID = generate_uid()

    # ссылка на исходное изображение, к которому относится заключение
    ref_sop = Dataset()
    ref_sop.ReferencedSOPClassUID = getattr(ref_ds, "SOPClassUID", "1.2.840.10008.5.1.4.1.1.7")
    ref_sop.ReferencedSOPInstanceUID = getattr(ref_ds, "SOPInstanceUID", generate_uid())
    evidence = Dataset()
    evidence.ReferencedSOPSequence = Sequence([ref_sop])
    ds.CurrentRequestedProcedureEvidenceSequence = Sequence([evidence])

    # --- корневой контейнер отчёта ---
    verdict_code, verdict_text = (
        ("VIOLATION", "Обнаружено нарушение качества укладки/снимка")
        if quality_class else ("OK", "Нарушений качества не обнаружено")
    )
    region_name = {"spine": "Поясничный отдел позвоночника"}.get(
        region, "Проксимальный отдел бедра")

    root_children = [
        _text_item("HAS CONCEPT MOD", "REGION", "Анатомическая область", region_name),
        _code_content_item("CONTAINS", "VERDICT", "Итоговое заключение", verdict_code, verdict_text),
        _num_item("CONTAINS", "QPROB", "Вероятность нарушения (quality_prob)", quality_prob, "1"),
        _text_item("CONTAINS", "VIOL-LIST", "Типы нарушений", violation_type or "нет"),
    ]

    measures = _SPINE_MEASURES if region == "spine" else _HIP_MEASURES
    measure_children = []
    for key, code, meaning, unit in measures:
        val = feats.get(key)
        if val is not None:
            try:
                measure_children.append(_num_item("CONTAINS", code, meaning, float(val), unit))
            except (TypeError, ValueError):
                continue
    if measure_children:
        root_children.append(
            _container_item("CONTAINS", "MEASURES", "Измеренные геометрические величины (Контур A)",
                             measure_children))

    root = _content_item("", "CONTAINER", "REPORT", "Отчёт контроля качества DXA (DensitoAI)")
    root.ContinuityOfContent = "SEPARATE"
    root.ContentTemplateSequence = Sequence()  # без формального DCID-шаблона
    root.ContentSequence = Sequence(root_children)
    ds.ContentSequence = Sequence([])
    # верхний уровень SR — единственный корневой content item хранится прямо в датасете
    for attr in ("ValueType", "ConceptNameCodeSequence", "ContinuityOfContent", "ContentSequence"):
        setattr(ds, attr, getattr(root, attr))

    ds.is_little_endian = True
    ds.is_implicit_VR = False
    return ds


def save_sr(path: str, ref_ds, region: str, quality_class: int, violation_type: str,
            quality_prob: float, feats: Dict[str, Any]) -> str:
    ds = build_sr(ref_ds, region, quality_class, violation_type, quality_prob, feats)
    ds.save_as(path, write_like_original=False)
    return path


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from geometry_features import extract_all_features
    import pydicom as pyd

    fp = sys.argv[1] if len(sys.argv) > 1 else None
    if fp:
        ref = pyd.dcmread(fp, force=True)
        feats = extract_all_features(fp, "spine")
        out = save_sr("/tmp/test_sr.dcm", ref, "spine", 1, "Не выравнена ось позвоночника", 0.83, feats)
        print("saved:", out)
        check = pyd.dcmread(out)
        print(check)
