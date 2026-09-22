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

Функции (режим «SR на снимок», исторический, флаг --sr-dir):
  build_sr(ref_ds, region, quality_class, violation_type, quality_prob, feats) -> pydicom Dataset
  save_sr(path, ref_ds, ...) -> path

Режим «ОДИН SR на исследование» (флаг --sr-study, К10 плана; методология НПКЦ ДиТ:
отсутствие SR или два и более SR на исследование — технологический дефект, SR нужен и при норме):
  build_study_sr(study_uid, items, model_version, config_hash, study_header) -> pydicom Dataset
  save_study_sr(path, ...) -> path
  study_sr_filename(study_uid) -> "<study_uid>_SR.dcm"
Один документ на study_uid, в нём по каждому снимку: ссылка на изображение (IMAGE content item с
ReferencedSOPSequence), область, класс, список нарушений, quality_prob, sha256 файла и пикселей
оригинала (хэш неизменности), статус обработки; в контексте документа — имя сервиса, версия модели,
config_hash и предупреждение об использовании ИИ. Series Instance UID детерминирован от
(study_uid, версия модели); SOP Instance UID детерминирован от тех же величин плюс содержимое
(повторный прогон с тем же результатом даёт тот же UID, другой результат — другой UID).
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
    ("curvature", "CURVATURE", "Показатель кривизны центральной линии", "1"),
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
        _num_item("CONTAINS", "QPROB", "Оценка риска нарушения (quality_prob, шкала 0-1)", quality_prob, "1"),
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


# =========================================================================== #
# Режим «один SR на исследование» (--sr-study)
# =========================================================================== #
import hashlib
import json
import re

SERVICE_NAME = "DensitoAI"
AI_WARNING = ("Результат получен автоматически программным обеспечением с применением технологий "
              "искусственного интеллекта и не является медицинским заключением. Требует проверки "
              "врачом.")
_UID_RE = re.compile(r"^[0-9]+(\.[0-9]+)*$")


def is_valid_uid(uid: Optional[str]) -> bool:
    """Синтаксическая проверка DICOM UID (цифры и точки, не более 64 символов)."""
    return bool(uid) and len(uid) <= 64 and bool(_UID_RE.match(uid))


def deterministic_uid(*parts: str) -> str:
    """UID, воспроизводимый от набора строк (pydicom generate_uid с entropy_srcs: sha512 →
    префикс pydicom 1.2.826.0.1.3680043.8.498. + 39 цифр)."""
    return generate_uid(entropy_srcs=[str(p) for p in parts])


def study_sr_filename(study_uid: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._-]", "_", str(study_uid))[:180]
    return f"{safe}_SR.dcm"


def _image_item(image_uid: str, sop_class_uid: Optional[str]) -> Optional[Dataset]:
    """IMAGE content item со ссылкой на исходное изображение (ReferencedSOPSequence)."""
    if not is_valid_uid(image_uid):
        return None
    item = _content_item("CONTAINS", "IMAGE", "SRC-IMAGE", "Исходное изображение")
    ref = Dataset()
    ref.ReferencedSOPClassUID = sop_class_uid if is_valid_uid(sop_class_uid) else "1.2.840.10008.5.1.4.1.1.7"
    ref.ReferencedSOPInstanceUID = image_uid
    item.ReferencedSOPSequence = Sequence([ref])
    return item


def _items_digest(items: list) -> str:
    keys = ("image_uid", "anatomical_region", "quality_class", "violations", "quality_prob",
            "processing_status", "sha256_file", "sha256_pixels")
    payload = [{k: it.get(k) for k in keys} for it in sorted(items, key=lambda x: str(x.get("image_uid")))]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def build_study_sr(study_uid: str, items: list, model_version: str, config_hash: str,
                   study_header: Optional[Dict[str, Any]] = None,
                   manufacturer: str = SERVICE_NAME, now: Optional[datetime.datetime] = None) -> Dataset:
    """Один DICOM Comprehensive SR на исследование.

    items — список словарей по каждому снимку исследования (включая норму и Failure):
      image_uid, sop_class_uid, series_uid, anatomical_region (официальная строка), quality_class (0/1),
      violations (list[str]), quality_prob (float), processing_status, sha256_file, sha256_pixels,
      path_to_study (строка для человека).
    study_header — атрибуты пациента/исследования, снятые с любого прочитанного DICOM исследования
      (PatientName, PatientID, PatientBirthDate, PatientSex, StudyDate, StudyTime, StudyID,
      AccessionNumber, ReferringPhysicianName). Может быть пустым.
    """
    now = now or datetime.datetime.now()
    study_header = study_header or {}
    items = list(items)

    sr_study_uid = study_uid if is_valid_uid(study_uid) else deterministic_uid("densito-study", study_uid)
    series_uid = deterministic_uid("densito-sr-series", study_uid, model_version)
    sop_uid = deterministic_uid("densito-sr-instance", study_uid, model_version, config_hash, _items_digest(items))

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = ComprehensiveSRStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.SpecificCharacterSet = "ISO_IR 192"
    # --- SOP Common
    ds.SOPClassUID = ComprehensiveSRStorage
    ds.SOPInstanceUID = sop_uid
    ds.InstanceCreationDate = now.strftime("%Y%m%d")
    ds.InstanceCreationTime = now.strftime("%H%M%S")
    # --- Patient / General Study (Type 2 — присутствуют, могут быть пустыми)
    for attr in ("PatientName", "PatientID", "PatientBirthDate", "PatientSex",
                 "StudyDate", "StudyTime", "StudyID", "AccessionNumber", "ReferringPhysicianName"):
        setattr(ds, attr, study_header.get(attr, ""))
    ds.StudyInstanceUID = sr_study_uid
    # --- SR Document Series
    ds.Modality = "SR"
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = 9001
    ds.SeriesDescription = "DensitoAI QC report (one SR per study)"
    ds.ReferencedPerformedProcedureStepSequence = Sequence()
    # --- General Equipment
    ds.Manufacturer = manufacturer
    ds.ManufacturerModelName = "DensitoAI DXA QC"
    ds.SoftwareVersions = str(model_version)
    # --- SR Document General
    ds.InstanceNumber = 1
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")
    ds.CompletionFlag = "COMPLETE"
    ds.VerificationFlag = "UNVERIFIED"  # автоматический анализ, врачом не подписан
    ds.PerformedProcedureCodeSequence = Sequence()

    # Evidence: все реально существующие изображения исследования, сгруппированные по серии
    by_series: Dict[str, list] = {}
    for it in items:
        if not is_valid_uid(it.get("image_uid")):
            continue
        s_uid = it.get("series_uid") if is_valid_uid(it.get("series_uid")) else deterministic_uid("densito-unknown-series", study_uid)
        by_series.setdefault(s_uid, []).append(it)
    if by_series:
        ev = Dataset()
        ev.StudyInstanceUID = sr_study_uid
        ref_series = []
        for s_uid, its in by_series.items():
            rs = Dataset()
            rs.SeriesInstanceUID = s_uid
            refs = []
            for it in its:
                r = Dataset()
                r.ReferencedSOPClassUID = it.get("sop_class_uid") if is_valid_uid(it.get("sop_class_uid")) else "1.2.840.10008.5.1.4.1.1.7"
                r.ReferencedSOPInstanceUID = it["image_uid"]
                refs.append(r)
            rs.ReferencedSOPSequence = Sequence(refs)
            ref_series.append(rs)
        ev.ReferencedSeriesSequence = Sequence(ref_series)
        ds.CurrentRequestedProcedureEvidenceSequence = Sequence([ev])

    # --- Дерево содержимого
    n_total = len(items)
    n_fail = sum(1 for it in items if str(it.get("processing_status", "")).lower() == "failure")
    n_viol = sum(1 for it in items if int(it.get("quality_class") or 0) == 1)
    if n_viol:
        verdict = ("VIOLATION", f"Выявлены нарушения качества на {n_viol} из {n_total} снимков")
    elif n_fail == n_total and n_total:
        verdict = ("NOT-EVALUATED", "Ни один снимок исследования не удалось обработать")
    else:
        verdict = ("OK", "Нарушений качества укладки и снимков не выявлено")

    root_children = [
        _text_item("HAS OBS CONTEXT", "SERVICE-NAME", "Наименование ИИ-сервиса", SERVICE_NAME),
        _text_item("HAS OBS CONTEXT", "MODEL-VERSION", "Версия модели", str(model_version)),
        _text_item("HAS OBS CONTEXT", "CONFIG-HASH", "Хэш конфигурации (config_hash)", str(config_hash)),
        _text_item("HAS OBS CONTEXT", "STUDY-UID-SRC", "StudyInstanceUID исходного исследования", str(study_uid)),
        _text_item("HAS OBS CONTEXT", "AI-WARNING", "Предупреждение об использовании ИИ", AI_WARNING),
        _code_content_item("CONTAINS", "STUDY-VERDICT", "Итог по исследованию", *verdict),
        _num_item("CONTAINS", "N-IMAGES", "Число снимков в исследовании", n_total, "1"),
        _num_item("CONTAINS", "N-VIOLATION", "Число снимков с нарушениями", n_viol, "1"),
        _num_item("CONTAINS", "N-FAILURE", "Число снимков, не обработанных (Failure)", n_fail, "1"),
    ]

    for idx, it in enumerate(sorted(items, key=lambda x: (str(x.get("anatomical_region", "")), str(x.get("image_uid", "")))), 1):
        status = str(it.get("processing_status", ""))
        is_fail = status.lower() == "failure"
        viols = [v for v in (it.get("violations") or []) if str(v).strip()]
        if is_fail:
            v_code, v_text = "NOT-EVALUATED", "Снимок не обработан (Failure)"
        elif viols:
            v_code, v_text = "VIOLATION", "Выявлены нарушения качества"
        else:
            v_code, v_text = "OK", "Нарушений не выявлено"
        children = []
        img_item = _image_item(str(it.get("image_uid", "")), it.get("sop_class_uid"))
        if img_item is not None:
            children.append(img_item)
        children += [
            _text_item("CONTAINS", "IMAGE-UID", "image_uid (SOPInstanceUID снимка)", str(it.get("image_uid", ""))),
            _text_item("CONTAINS", "FILE", "Файл (path_to_study)", str(it.get("path_to_study", ""))),
            _text_item("CONTAINS", "REGION", "Анатомическая область", str(it.get("anatomical_region", ""))),
            _code_content_item("CONTAINS", "IMAGE-VERDICT", "Заключение по снимку", v_code, v_text),
            _num_item("CONTAINS", "QCLASS", "quality_class", int(it.get("quality_class") or 0), "1"),
            _text_item("CONTAINS", "VIOL-LIST", "Типы нарушений (violation_type)", "; ".join(viols) if viols else "нет"),
            _text_item("CONTAINS", "STATUS", "Статус обработки (processing_status)", status),
        ]
        qp = it.get("quality_prob")
        if qp is not None:
            try:
                children.append(_num_item("CONTAINS", "QPROB", "Оценка риска нарушения (quality_prob, шкала 0-1)", float(qp), "1"))
            except (TypeError, ValueError):
                pass
        for key, code, meaning in (("sha256_file", "SHA256-FILE", "SHA-256 исходного файла DICOM (неизменность оригинала)"),
                                   ("sha256_pixels", "SHA256-PIXELS", "SHA-256 массива пикселей оригинала")):
            if it.get(key):
                children.append(_text_item("CONTAINS", code, meaning, str(it[key])))
        root_children.append(_container_item("CONTAINS", "IMAGE-REPORT", f"Снимок {idx}", children))

    root = _content_item("", "CONTAINER", "STUDY-REPORT", "Отчёт контроля качества DXA по исследованию (DensitoAI)")
    root.ContinuityOfContent = "SEPARATE"
    root.ContentSequence = Sequence(root_children)
    for attr in ("ValueType", "ConceptNameCodeSequence", "ContinuityOfContent", "ContentSequence"):
        setattr(ds, attr, getattr(root, attr))
    # у корневого элемента RelationshipType отсутствует — копируем только нужные атрибуты

    ds.is_little_endian = True
    ds.is_implicit_VR = False
    return ds


def save_study_sr(path: str, study_uid: str, items: list, model_version: str, config_hash: str,
                  study_header: Optional[Dict[str, Any]] = None) -> str:
    ds = build_study_sr(study_uid, items, model_version, config_hash, study_header)
    ds.save_as(path, write_like_original=False)
    return path


def study_header_from_ds(ds) -> Dict[str, Any]:
    """Атрибуты пациента/исследования для SR из любого DICOM исследования."""
    out: Dict[str, Any] = {}
    # значения, не соответствующие VR (например, «Anonymized» в DA/TM/CS обезличенных данных),
    # не копируем — иначе SR формально невалиден; Type 2 атрибут остаётся пустым
    checks = {
        "PatientBirthDate": r"^\d{8}$", "StudyDate": r"^\d{8}$",
        "StudyTime": r"^\d{2}(\d{2}(\d{2}(\.\d{1,6})?)?)?$", "PatientSex": r"^[MFO]$",
        "AccessionNumber": r"^.{1,16}$", "StudyID": r"^.{1,16}$", "PatientID": r"^.{1,64}$",
    }
    for attr in ("PatientName", "PatientID", "PatientBirthDate", "PatientSex",
                 "StudyDate", "StudyTime", "StudyID", "AccessionNumber", "ReferringPhysicianName"):
        v = getattr(ds, attr, None)
        sv = str(v).strip() if v is not None else ""
        if not sv:
            continue
        pat = checks.get(attr)
        if pat and not re.match(pat, sv):
            continue
        out[attr] = sv
    return out


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
