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

Полнота исследования (идея 4 бэклога): если в исследовании представлена только одна из двух областей
(поясничный отдел позвоночника или проксимальный отдел бедра), в корень SR добавляется TEXT-элемент
STUDY-COMPLETENESS с нейтральным примечанием. Это не критерий качества снимка: колонки CSV, per-image SR
и вердикты не меняются. Функция study_completeness(items) возвращает тот же результат в виде словаря
{"spine": bool, "hip": bool, "note": str | None} — его отдаёт API (/api/analyze -> study_completeness).
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


def _items_digest(items: list, study_notes: Optional[list] = None) -> str:
    """Дайджест содержимого SR для детерминированного SOP Instance UID.

    study_notes — примечания уровня исследования (например, примечание о полноте). Они добавляются в
    дайджест только когда непусты, поэтому для исследований без примечаний дайджест (и SOP Instance UID)
    совпадает с версиями до появления примечания."""
    keys = ("image_uid", "anatomical_region", "quality_class", "violations", "quality_prob",
            "processing_status", "sha256_file", "sha256_pixels")
    payload = [{k: it.get(k) for k in keys} for it in sorted(items, key=lambda x: str(x.get("image_uid")))]
    notes = [str(n) for n in (study_notes or []) if n]
    if notes:
        payload.append({"study_notes": notes})
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


# --- Полнота исследования: обе области (позвоночник и бедро) или только одна ------------------------ #
REGION_SPINE_NAME = "Поясничный отдел позвоночника"   # официальная строка (config.yaml -> regions.spine)
REGION_HIP_NAME = "Проксимальный отдел бедра"        # официальная строка (config.yaml -> regions.hip)
COMPLETENESS_CODE = "STUDY-COMPLETENESS"
COMPLETENESS_MEANING = "Примечание о полноте исследования (представленные области)"
COMPLETENESS_NOTE = ("Примечание: в исследовании представлена только одна область из двух ({region}); "
                     "полнота исследования не является критерием оценки качества снимка.")


def _region_kind(name: Any) -> Optional[str]:
    """'spine' / 'hip' / None по официальной строке области или внутреннему имени
    (spine, right_hip, left_hip). Сравнение без учёта регистра; лишние пробелы игнорируются."""
    s = str(name or "").strip().lower()
    if not s:
        return None
    if s == REGION_SPINE_NAME.lower() or s == "spine" or "позвоночник" in s:
        return "spine"
    if s == REGION_HIP_NAME.lower() or s in ("right_hip", "left_hip", "hip") or "бедр" in s:
        return "hip"
    return None


def study_completeness(items: list) -> Dict[str, Any]:
    """Какие области представлены в исследовании и нужно ли примечание.

    Возвращает {"spine": bool, "hip": bool, "note": str | None}. Примечание формируется только
    когда представлена ровно одна из двух областей. Учитываются все строки исследования, включая
    Failure: область строки-заглушки определяется по заголовку DICOM, как и в CSV. Если область не
    распознана ни у одной строки (пустой список) или ни один снимок исследования не обработан
    (все строки Failure — область там лишь предположение по заголовку), примечания нет — утверждать
    нечего."""
    kinds = {_region_kind(it.get("anatomical_region")) for it in items}
    has_spine = "spine" in kinds
    has_hip = "hip" in kinds
    any_processed = any(str(it.get("processing_status", "")).lower() != "failure" for it in items)
    note = None
    if any_processed and has_spine != has_hip:
        present = REGION_SPINE_NAME if has_spine else REGION_HIP_NAME
        note = COMPLETENESS_NOTE.format(region=present)
    return {"spine": has_spine, "hip": has_hip, "note": note}


# --- Главное действие на визит (C1): одно приоритетное действие по исследованию ------------------- #
# Правило: среди критериев с флагом по всем снимкам исследования выбирается критерий с максимальным
# относительным запасом над порогом  (score - threshold) / (1 - threshold)  (0 — ровно на пороге,
# 1 — максимум шкалы). Остальные замечания сохраняются в others (свёрнуты в кабинете, не скрыты).
# Повторы одной и той же пары «область + критерий» (копии кадра) сворачиваются в одно замечание с числом
# снимков. Снимок с классом 1 без флага по критерию (решение общей модели) идёт после критериев.
# На quality_class, violation_type и 9 колонок CSV не влияет.
PRIORITY_CODE = "PRIORITY-ACTION"
PRIORITY_MEANING = "Приоритетное действие"
PRIORITY_TEXT_CODE = "PRIORITY-TEXT"  # CodeValue (SH) — не длиннее 16 символов
PRIORITY_TEXT_MEANING = "Приоритетное действие: подробно"
PRIORITY_RULE = "максимальный относительный запас над порогом (score - threshold) / (1 - threshold)"

_SIDE_TITLES = {"spine": "Позвоночник", "right_hip": "Правое бедро", "left_hip": "Левое бедро"}
_CRIT_TITLES = {
    "sp_pos": "укладка позвоночника", "sp_axis": "ось позвоночника", "sp_art": "посторонние предметы в поле",
    "rh_pos": "укладка бедра", "lh_pos": "укладка бедра",
    "rh_roi": "область интереса бедра", "lh_roi": "область интереса бедра",
}
# код значения -> (краткий смысл для CodeMeaning, <= 64 символов; хвост текста действия).
# В SR и в API значение кода — "PA-" + короткий код (_PRIORITY_VALUE), CodeValue (SH) не длиннее 16 символов.
_PRIORITY_ACTIONS = {
    "RETAKE-CHECK": ("Проверить снимок, при подтверждении переснять", "проверить снимок, при подтверждении переснять"),
    "RESCAN-DISCUSS": ("Поле неполное: обсудить повторное сканирование",
                       "поле сканирования неполное, обсудить повторное сканирование"),
    "ANALYSIS-CHECK": ("Поле полное: проверить анализ на аппарате",
                       "поле полное, вопрос к области анализа: проверить анализ на аппарате, повторно не облучать"),
    "DOCTOR": ("Решение врача по снимку", "измерений поля недостаточно, решение врача по снимку"),
    "REVIEW": ("Проверить снимок (общая оценка)", "нарушение по общей оценке снимка, проверить снимок"),
    "FILES": ("Проверить необработанные файлы", "проверить необработанные файлы"),
    "NONE": ("Действий по качеству не требуется", "нарушений не выявлено, действий по качеству не требуется"),
}
_PRIORITY_VALUE = {"RETAKE-CHECK": "PA-RETAKE", "RESCAN-DISCUSS": "PA-RESCAN", "ANALYSIS-CHECK": "PA-ANALYSIS",
                   "DOCTOR": "PA-DOCTOR", "REVIEW": "PA-REVIEW", "FILES": "PA-FILES", "NONE": "PA-NONE"}
_ROI_ROUTE_ACTION = {"field_incomplete": "RESCAN-DISCUSS", "field_complete": "ANALYSIS-CHECK",
                     "insufficient_data": "DOCTOR"}


def _fnum(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _item_region(it: Dict[str, Any]) -> str:
    r = str(it.get("internal_region") or "").strip()
    if r in _SIDE_TITLES:
        return r
    return "spine" if _region_kind(it.get("anatomical_region")) == "spine" else (r or "")


def relative_margin(score: Any, threshold: Any) -> Optional[float]:
    """(score - threshold) / (1 - threshold); None, если чисел нет или порог >= 1."""
    s, t = _fnum(score), _fnum(threshold)
    if s is None or t is None or t >= 1.0:
        return None
    return round((s - t) / (1.0 - t), 6)


def study_priority_action(items: list) -> Dict[str, Any]:
    """Одно главное действие по исследованию + остальные замечания.

    items — как в build_study_sr, плюс необязательные поля: internal_region, criteria
    ([{code, score, threshold, flag, uncertain}]) и roi_route (код hip_roi: field_incomplete /
    field_complete / insufficient_data). Без criteria снимок с классом 1 даёт замечание «общая оценка».
    Возвращает {status, action_code, action_meaning, text, image_uid, region, criterion,
    relative_margin, uncertain, n_images, others: [...], n_others, rule}. Детерминирован: при равных
    запасах порядок — по коду критерия и image_uid."""
    cands: Dict[tuple, Dict[str, Any]] = {}
    general: Dict[str, Dict[str, Any]] = {}
    n_fail = 0
    for it in items or []:
        if str(it.get("processing_status", "")).lower() == "failure":
            n_fail += 1
            continue
        reg = _item_region(it)
        uid = str(it.get("image_uid") or "")
        any_flag = False
        for c in it.get("criteria") or []:
            try:
                flagged = int(c.get("flag") or 0) == 1
            except (TypeError, ValueError):
                flagged = False
            if not flagged:
                continue
            any_flag = True
            code = str(c.get("code") or "")
            rm = relative_margin(c.get("score"), c.get("threshold"))
            if code.endswith("_roi"):
                act = _ROI_ROUTE_ACTION.get(str(it.get("roi_route") or ""), "DOCTOR")
            else:
                act = "RETAKE-CHECK"
            key = (reg, code)
            cur = cands.get(key)
            entry = {"image_uid": uid, "region": reg, "criterion": code, "relative_margin": rm,
                     "uncertain": bool(c.get("uncertain")), "action_code": act, "n_images": 1}
            if cur is None:
                cands[key] = entry
            else:
                n = cur["n_images"] + 1
                a = rm if rm is not None else -9.0
                b = cur["relative_margin"] if cur["relative_margin"] is not None else -9.0
                better = a > b or (a == b and uid < cur["image_uid"])
                cands[key] = {**(entry if better else cur), "n_images": n}
        if not any_flag and int(it.get("quality_class") or 0) == 1:
            g = general.get(reg)
            general[reg] = {"image_uid": min(uid, g["image_uid"]) if g else uid, "region": reg, "criterion": "",
                            "relative_margin": None, "uncertain": False, "action_code": "REVIEW",
                            "n_images": (g["n_images"] + 1) if g else 1}

    def _text(e: Dict[str, Any]) -> str:
        side = _SIDE_TITLES.get(e["region"], e["region"] or "Снимок")
        tail = _PRIORITY_ACTIONS[e["action_code"]][1]
        crit = _CRIT_TITLES.get(e["criterion"], "")
        s = f"{side}: {crit}, {tail}" if crit else f"{side}: {tail}"
        if e.get("n_images", 1) > 1:
            s += f" (снимков: {e['n_images']})"
        if e.get("uncertain"):
            s += "; оценка у порога"
        return s

    ranked = sorted(cands.values(), key=lambda e: (-(e["relative_margin"] if e["relative_margin"] is not None else -9.0),
                                                   e["criterion"], e["region"], e["image_uid"]))
    ranked += sorted(general.values(), key=lambda e: (e["region"], e["image_uid"]))
    for e in ranked:
        e["text"] = _text(e)
    if n_fail:
        ranked.append({"image_uid": "", "region": "", "criterion": "", "relative_margin": None, "uncertain": False,
                       "action_code": "FILES", "n_images": n_fail,
                       "text": f"Проверить необработанные файлы (Failure): {n_fail}"})
    if ranked:
        main, others = ranked[0], ranked[1:]
        status = "files" if main["action_code"] == "FILES" else "action"
    else:
        main = {"image_uid": "", "region": "", "criterion": "", "relative_margin": None, "uncertain": False,
                "action_code": "NONE", "n_images": 0, "text": "Нарушений не выявлено, действий по качеству не требуется"}
        others, status = [], "ok"
    return {"status": status, "action_code": _PRIORITY_VALUE[main["action_code"]],
            "action_meaning": _PRIORITY_ACTIONS[main["action_code"]][0], "text": main["text"],
            "image_uid": main["image_uid"], "region": main["region"], "criterion": main["criterion"],
            "relative_margin": main["relative_margin"], "uncertain": main["uncertain"],
            "n_images": main["n_images"],
            "others": [dict({k: e[k] for k in ("text", "image_uid", "region", "criterion", "relative_margin",
                                               "uncertain", "n_images")}, action_code=_PRIORITY_VALUE[e["action_code"]])
                       for e in others],
            "n_others": len(others), "rule": PRIORITY_RULE}


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

    completeness = study_completeness(items)
    study_notes = [completeness["note"]] if completeness["note"] else []

    sr_study_uid = study_uid if is_valid_uid(study_uid) else deterministic_uid("densito-study", study_uid)
    series_uid = deterministic_uid("densito-sr-series", study_uid, model_version)
    sop_uid = deterministic_uid("densito-sr-instance", study_uid, model_version, config_hash,
                                _items_digest(items, study_notes))

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
        # Порядок серий и ссылок — по UID, а не по порядку обхода файлов: байты SR не должны
        # зависеть от того, в каком порядке пришли снимки (прогон-двойник, идея 1).
        for s_uid in sorted(by_series):
            its = sorted(by_series[s_uid], key=lambda x: str(x["image_uid"]))
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
    priority = study_priority_action(items)  # C1: главное действие на визит, рядом с итогом
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
        _code_content_item("CONTAINS", PRIORITY_CODE, PRIORITY_MEANING, priority["action_code"],
                           priority["action_meaning"]),
        _text_item("CONTAINS", PRIORITY_TEXT_CODE, PRIORITY_TEXT_MEANING, priority["text"]),
        _num_item("CONTAINS", "N-IMAGES", "Число снимков в исследовании", n_total, "1"),
        _num_item("CONTAINS", "N-VIOLATION", "Число снимков с нарушениями", n_viol, "1"),
        _num_item("CONTAINS", "N-FAILURE", "Число снимков, не обработанных (Failure)", n_fail, "1"),
    ]
    if completeness["note"]:
        # мягкое примечание: представлена только одна область из двух; на вердикты не влияет
        root_children.append(_text_item("CONTAINS", COMPLETENESS_CODE, COMPLETENESS_MEANING, completeness["note"]))

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


# =========================================================================== #
# Идея 23: отдельный SR «Решение специалиста по области интереса» (бедро).
#
# Основной SR исследования (build_study_sr) не меняется — его UID детерминированы и проверяются
# тестами. Решения специалиста по предложенной области интереса пишутся в ОТДЕЛЬНЫЙ документ
# <study_uid>_SR_decisions.dcm (своя серия SeriesNumber 9002), который формируется по запросу
# GET /api/results/{job}/decisions_sr/{study_uid}. Содержимое: по каждому решению — ссылка на исходный
# снимок (IMAGE), предложенная область (пиксели и мм), решение (CODE), итоговая область при решении
# «своя», специалист (строка без персональных данных: должность/инициалы), комментарий, время.
# UID: Series — от (study_uid, версия модели); SOP Instance — от тех же величин плюс дайджест решений:
# повторный запрос без новых решений даёт тот же файл, новое решение — новый UID.
# =========================================================================== #
DECISION_CONFIRMED = "подтверждено"
DECISION_REJECTED = "отклонено"
DECISION_CUSTOM = "своя"
DECISION_VALUES = (DECISION_CONFIRMED, DECISION_REJECTED, DECISION_CUSTOM)
_DECISION_CODES = {
    DECISION_CONFIRMED: ("ROI-CONFIRMED", "Предложенная область интереса подтверждена специалистом"),
    DECISION_REJECTED: ("ROI-REJECTED", "Предложенная область интереса отклонена специалистом"),
    DECISION_CUSTOM: ("ROI-CUSTOM", "Специалист задал свою область интереса"),
}
ROI_REASON_TEXT = {
    "scan_too_short": "поле сканирования короче требуемого",
    "shaft_below_trochanter_too_short": "в кадр вошло мало диафиза ниже вертелов",
    "lateral_margin_below_threshold": "кость ближе к краю кадра, чем требует ТЗ (не менее 20 мм)",
}
DECISION_SR_SUFFIX = "_SR_decisions.dcm"


def decision_sr_filename(study_uid: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._-]", "_", str(study_uid))[:170]
    return f"{safe}{DECISION_SR_SUFFIX}"


def _box_text(box) -> str:
    """[x0, y0, x1, y1] -> "x0, y0, x1, y1" (для TEXT-элементов SR); пустая строка, если рамки нет."""
    if not box or len(box) != 4:
        return ""
    try:
        vals = [float(v) for v in box]
    except (TypeError, ValueError):
        return ""
    return ", ".join(str(int(round(v))) if abs(v - round(v)) < 1e-6 else f"{v:.1f}" for v in vals)


def _decisions_digest(decisions: list) -> str:
    keys = ("decision_id", "image_uid", "decision", "roi_box_px", "suggested_box_px", "specialist",
            "comment", "created_at")
    payload = [{k: d.get(k) for k in keys} for d in sorted(decisions, key=lambda x: str(x.get("created_at", "")) + str(x.get("decision_id", "")))]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def _dicom_datetime(iso: str) -> str:
    """ISO 8601 ("2026-09-23T11:52:00") -> DICOM DT ("20260923115200"); пустая строка, если формат иной."""
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2}))?", str(iso or ""))
    if not m:
        return ""
    return "".join(g or "00" for g in m.groups())


def build_decision_sr(study_uid: str, decisions: list, model_version: str, config_hash: str,
                      study_header: Optional[Dict[str, Any]] = None,
                      manufacturer: str = SERVICE_NAME, now: Optional[datetime.datetime] = None) -> Dataset:
    """DICOM Comprehensive SR «Решение специалиста по области интереса» для одного исследования.

    decisions — список словарей (записи decisions.json задачи, отфильтрованные по study_uid):
      decision_id, image_uid, sop_class_uid, path_to_study, decision (подтверждено/отклонено/своя),
      suggested_box_px [x0,y0,x1,y1], suggested_box_mm, deficit_mm, reason, roi_box_px (при «своя»),
      roi_box_mm, specialist, comment, created_at (ISO 8601).
    Пустой список решений допустим: документ содержит контекст и число решений 0.
    """
    now = now or datetime.datetime.now()
    study_header = study_header or {}
    decisions = [d for d in decisions if isinstance(d, dict)]

    sr_study_uid = study_uid if is_valid_uid(study_uid) else deterministic_uid("densito-study", study_uid)
    series_uid = deterministic_uid("densito-sr-decisions-series", study_uid, model_version)
    sop_uid = deterministic_uid("densito-sr-decisions-instance", study_uid, model_version, config_hash,
                                _decisions_digest(decisions))

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = ComprehensiveSRStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.SpecificCharacterSet = "ISO_IR 192"
    ds.SOPClassUID = ComprehensiveSRStorage
    ds.SOPInstanceUID = sop_uid
    ds.InstanceCreationDate = now.strftime("%Y%m%d")
    ds.InstanceCreationTime = now.strftime("%H%M%S")
    for attr in ("PatientName", "PatientID", "PatientBirthDate", "PatientSex",
                 "StudyDate", "StudyTime", "StudyID", "AccessionNumber", "ReferringPhysicianName"):
        setattr(ds, attr, study_header.get(attr, ""))
    ds.StudyInstanceUID = sr_study_uid
    ds.Modality = "SR"
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = 9002
    ds.SeriesDescription = "DensitoAI ROI decision by specialist"
    ds.ReferencedPerformedProcedureStepSequence = Sequence()
    ds.Manufacturer = manufacturer
    ds.ManufacturerModelName = "DensitoAI DXA QC"
    ds.SoftwareVersions = str(model_version)
    ds.InstanceNumber = 1
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")
    ds.CompletionFlag = "COMPLETE"
    # Решение вводит специалист через кабинет по коду доступа задачи; электронной подписи и
    # аутентификации наблюдателя нет, поэтому документ формально не верифицирован.
    ds.VerificationFlag = "UNVERIFIED"
    ds.PerformedProcedureCodeSequence = Sequence()

    # Evidence: снимки, по которым есть решения (одна «неизвестная» серия — series_uid снимка в
    # decisions.json не хранится)
    ref_imgs = sorted({str(d.get("image_uid")): d.get("sop_class_uid") for d in decisions
                       if is_valid_uid(d.get("image_uid"))}.items())
    if ref_imgs:
        ev = Dataset()
        ev.StudyInstanceUID = sr_study_uid
        rs = Dataset()
        rs.SeriesInstanceUID = deterministic_uid("densito-unknown-series", study_uid)
        refs = []
        for img_uid, sop_cls in ref_imgs:
            r = Dataset()
            r.ReferencedSOPClassUID = sop_cls if is_valid_uid(sop_cls) else "1.2.840.10008.5.1.4.1.1.7"
            r.ReferencedSOPInstanceUID = img_uid
            refs.append(r)
        rs.ReferencedSOPSequence = Sequence(refs)
        ev.ReferencedSeriesSequence = Sequence([rs])
        ds.CurrentRequestedProcedureEvidenceSequence = Sequence([ev])

    counts = {v: sum(1 for d in decisions if str(d.get("decision")) == v) for v in DECISION_VALUES}
    root_children = [
        _text_item("HAS OBS CONTEXT", "SERVICE-NAME", "Наименование ИИ-сервиса", SERVICE_NAME),
        _text_item("HAS OBS CONTEXT", "MODEL-VERSION", "Версия модели", str(model_version)),
        _text_item("HAS OBS CONTEXT", "CONFIG-HASH", "Хэш конфигурации (config_hash)", str(config_hash)),
        _text_item("HAS OBS CONTEXT", "STUDY-UID-SRC", "StudyInstanceUID исходного исследования", str(study_uid)),
        _text_item("HAS OBS CONTEXT", "AI-WARNING", "Предупреждение об использовании ИИ", AI_WARNING),
        _text_item("HAS OBS CONTEXT", "DOC-PURPOSE", "Назначение документа",
                   "Предложение коррекции области интереса бедра сформировано системой и требует подтверждения "
                   "специалистом; в документе зафиксированы решения специалиста. Исходный снимок не изменяется."),
        _num_item("CONTAINS", "N-DECISIONS", "Число решений", len(decisions), "1"),
        _num_item("CONTAINS", "N-CONFIRMED", "Число решений «подтверждено»", counts[DECISION_CONFIRMED], "1"),
        _num_item("CONTAINS", "N-REJECTED", "Число решений «отклонено»", counts[DECISION_REJECTED], "1"),
        _num_item("CONTAINS", "N-CUSTOM", "Число решений «своя область»", counts[DECISION_CUSTOM], "1"),
    ]

    ordered = sorted(decisions, key=lambda x: (str(x.get("created_at", "")), str(x.get("decision_id", ""))))
    for idx, d in enumerate(ordered, 1):
        children = []
        img_item = _image_item(str(d.get("image_uid", "")), d.get("sop_class_uid"))
        if img_item is not None:
            children.append(img_item)
        children += [
            _text_item("CONTAINS", "IMAGE-UID", "image_uid (SOPInstanceUID снимка)", str(d.get("image_uid", "")) or "—"),
            _text_item("CONTAINS", "FILE", "Файл (path_to_study)", str(d.get("path_to_study", "")) or "—"),
            _text_item("CONTAINS", "SUGGEST-SOURCE", "Источник предложенной области", "предложение системы"),
        ]
        sug_px = _box_text(d.get("suggested_box_px"))
        if sug_px:
            children.append(_text_item("CONTAINS", "SUGGESTED-BOX-PX", "Предложенная область, пиксели (x0, y0, x1, y1)", sug_px))
        sug_mm = _box_text(d.get("suggested_box_mm"))
        if sug_mm:
            children.append(_text_item("CONTAINS", "SUGGESTED-BOX-MM", "Предложенная область, мм от угла кадра (x0, y0, x1, y1)", sug_mm))
        if d.get("deficit_mm") is not None:
            try:
                children.append(_num_item("CONTAINS", "ROI-DEFICIT", "Недостающая длина поля сканирования", float(d["deficit_mm"]), "mm"))
            except (TypeError, ValueError):
                pass
        if d.get("reason"):
            children.append(_text_item("CONTAINS", "ROI-REASON", "Причина предложения",
                                       ROI_REASON_TEXT.get(str(d["reason"]), str(d["reason"]))))
        code, meaning = _DECISION_CODES.get(str(d.get("decision")), ("ROI-UNKNOWN", "Решение не распознано"))
        children.append(_code_content_item("CONTAINS", "ROI-DECISION", "Решение специалиста", code, meaning))
        fin_px = _box_text(d.get("roi_box_px"))
        if fin_px:
            children.append(_text_item("CONTAINS", "FINAL-BOX-PX", "Область интереса специалиста, пиксели (x0, y0, x1, y1)", fin_px))
        fin_mm = _box_text(d.get("roi_box_mm"))
        if fin_mm:
            children.append(_text_item("CONTAINS", "FINAL-BOX-MM", "Область интереса специалиста, мм (x0, y0, x1, y1)", fin_mm))
        children.append(_text_item("CONTAINS", "SPECIALIST", "Специалист (должность/инициалы)",
                                   str(d.get("specialist") or "").strip() or "не указан"))
        if str(d.get("comment") or "").strip():
            children.append(_text_item("CONTAINS", "COMMENT", "Комментарий специалиста", str(d["comment"]).strip()))
        dt = _dicom_datetime(str(d.get("created_at", "")))
        if dt:
            children.append(_content_item("CONTAINS", "DATETIME", "DECISION-TIME", "Время решения", DateTime=dt))
        else:
            children.append(_text_item("CONTAINS", "DEC-TIME-TEXT", "Время решения", str(d.get("created_at", "")) or "—"))
        root_children.append(_container_item("CONTAINS", "ROI-DEC-ITEM", f"Решение {idx}", children))

    root = _content_item("", "CONTAINER", "ROI-DEC-REPORT", "Решение специалиста по области интереса (DensitoAI)")
    root.ContinuityOfContent = "SEPARATE"
    root.ContentSequence = Sequence(root_children)
    for attr in ("ValueType", "ConceptNameCodeSequence", "ContinuityOfContent", "ContentSequence"):
        setattr(ds, attr, getattr(root, attr))
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    return ds


def save_decision_sr(path: str, study_uid: str, decisions: list, model_version: str, config_hash: str,
                     study_header: Optional[Dict[str, Any]] = None) -> str:
    ds = build_decision_sr(study_uid, decisions, model_version, config_hash, study_header)
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
