#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест примечания о полноте исследования в SR на исследование (идея 4 бэклога).

Только pydicom, без torch и без весов моделей:
  1. обе области (позвоночник и бедро) -> в корне SR нет элемента STUDY-COMPLETENESS, note = None;
  2. только позвоночник -> элемент есть, текст называет «Поясничный отдел позвоночника»;
  3. только бедро (одна или две стороны) -> элемент есть, текст называет «Проксимальный отдел бедра»;
  4. строка Failure учитывается по области; пустой список строк -> примечания нет;
  5. детерминизм: два вызова build_study_sr с одинаковым входом дают одинаковый SOP Instance UID и
     бит-в-бит одинаковый файл; появление/исчезновение примечания меняет SOP Instance UID; для
     исследований с обеими областями SOP Instance UID совпадает с формулой до появления примечания;
  6. текст примечания без запрещённых слов, per-image SR (build_sr) не содержит примечания;
  7. inference.study_completeness_by_study даёт тот же результат по строкам официального формата;
  8. tools/validate_sr.py -> PASS на SR с примечанием.

Запуск: python tests/test_study_completeness.py   (код возврата 0 — ок).
"""
import datetime
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pydicom  # noqa: E402

import dicom_sr  # noqa: E402
from dicom_sr import (COMPLETENESS_CODE, REGION_HIP_NAME, REGION_SPINE_NAME, build_study_sr,  # noqa: E402
                      deterministic_uid, study_completeness)

fails = []


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def item(uid, region, status="Success", qclass=0, viols=None, prob=0.1):
    return {"image_uid": f"1.2.826.0.1.3680043.8.498.{uid}", "sop_class_uid": "1.2.840.10008.5.1.4.1.1.7",
            "series_uid": "1.2.826.0.1.3680043.8.498.777", "anatomical_region": region, "quality_class": qclass,
            "violations": list(viols or []), "quality_prob": prob, "processing_status": status,
            "sha256_file": hashlib.sha256(str(uid).encode()).hexdigest(),
            "sha256_pixels": hashlib.sha256(("px" + str(uid)).encode()).hexdigest(),
            "path_to_study": f"study/{uid}.dcm"}


STUDY_UID = "1.2.826.0.1.3680043.8.498.100"
MODEL_VERSION, CFG_HASH = "2.3.2", "abcdef012345"
NOW = datetime.datetime(2026, 9, 23, 10, 0, 0)

both = [item(1, REGION_SPINE_NAME), item(2, REGION_HIP_NAME), item(3, REGION_HIP_NAME)]
spine_only = [item(1, REGION_SPINE_NAME), item(4, REGION_SPINE_NAME)]
hip_only = [item(2, REGION_HIP_NAME), item(3, REGION_HIP_NAME)]
hip_one_side = [item(2, REGION_HIP_NAME)]


def root_codes(ds):
    return [str(it.ConceptNameCodeSequence[0].CodeValue) for it in ds.ContentSequence]


def completeness_item(ds):
    for it in ds.ContentSequence:
        if str(it.ConceptNameCodeSequence[0].CodeValue) == COMPLETENESS_CODE:
            return it
    return None


def sr_bytes(ds):
    buf = io.BytesIO()
    ds.save_as(buf, write_like_original=False)
    return buf.getvalue()


def build(items, now=NOW, **kw):
    return build_study_sr(STUDY_UID, items, MODEL_VERSION, CFG_HASH, {}, now=now, **kw)


# 1. обе области
c = study_completeness(both)
check(c == {"spine": True, "hip": True, "note": None}, f"обе области: {c}")
ds_both = build(both)
check(completeness_item(ds_both) is None, "обе области: в SR нет STUDY-COMPLETENESS")
check(COMPLETENESS_CODE not in root_codes(ds_both), "обе области: код отсутствует в корне")

# 2. только позвоночник
c = study_completeness(spine_only)
check(c["spine"] and not c["hip"] and c["note"], f"только позвоночник: {c}")
check(REGION_SPINE_NAME in (c["note"] or ""), "только позвоночник: примечание называет область")
ds_sp = build(spine_only)
it = completeness_item(ds_sp)
check(it is not None and it.ValueType == "TEXT" and it.RelationshipType == "CONTAINS",
      "только позвоночник: TEXT CONTAINS в корне SR")
check(it is not None and str(it.TextValue) == c["note"], "только позвоночник: TextValue = note")
check(it is not None and "полнот" in str(it.ConceptNameCodeSequence[0].CodeMeaning).lower(),
      "только позвоночник: CodeMeaning понятен человеку")
# примечание идёт после счётчиков и до контейнеров снимков
codes = root_codes(ds_sp)
check(codes.index(COMPLETENESS_CODE) > codes.index("N-FAILURE")
      and codes.index(COMPLETENESS_CODE) < codes.index("IMAGE-REPORT"),
      "только позвоночник: примечание между счётчиками и снимками")

# 3. только бедро (две стороны и одна сторона дают одно и то же примечание)
for name, items in (("две стороны", hip_only), ("одна сторона", hip_one_side)):
    c = study_completeness(items)
    check(c["hip"] and not c["spine"] and c["note"] and REGION_HIP_NAME in c["note"],
          f"только бедро ({name}): {c}")
    ds_h = build(items)
    it = completeness_item(ds_h)
    check(it is not None and str(it.TextValue) == c["note"], f"только бедро ({name}): TEXT в SR")

# 4. Failure учитывается по области; пустой список — без примечания
c = study_completeness([item(1, REGION_SPINE_NAME), item(2, REGION_HIP_NAME, status="Failure", prob=0.5)])
check(c["note"] is None and c["hip"], "Failure-строка бедра учитывается как область")
c = study_completeness([])
check(c == {"spine": False, "hip": False, "note": None}, f"пустой список: {c}")
c = study_completeness([item(1, REGION_SPINE_NAME, status="Failure", prob=0.5)])
check(c["spine"] and c["note"] is None, "все строки Failure: область по заголовку отмечена, примечания нет")
ds_f = build([item(1, REGION_SPINE_NAME, status="Failure", prob=0.5)])
check(completeness_item(ds_f) is None, "все строки Failure: в SR нет STUDY-COMPLETENESS")
c = study_completeness([item(1, ""), item(2, "")])
check(c["note"] is None, "нераспознанные области: примечания нет")
c = study_completeness([{"anatomical_region": "spine"}, {"anatomical_region": "left_hip"}])
check(c["note"] is None and c["spine"] and c["hip"], "внутренние имена областей тоже распознаются")

# 5. детерминизм
a, b = build(spine_only), build(spine_only)
check(a.SOPInstanceUID == b.SOPInstanceUID, "детерминизм: одинаковый SOP Instance UID")
check(a.SeriesInstanceUID == b.SeriesInstanceUID, "детерминизм: одинаковый Series Instance UID")
check(sr_bytes(a) == sr_bytes(b), "детерминизм: файл бит-в-бит одинаков при том же времени")
a2, b2 = build(both), build(both)
check(sr_bytes(a2) == sr_bytes(b2), "детерминизм (обе области): файл бит-в-бит одинаков")
# другой набор областей — другой UID
check(build(spine_only).SOPInstanceUID != build(both).SOPInstanceUID,
      "разный состав областей -> разный SOP Instance UID")
# для исследований с обеими областями UID совпадает с формулой без примечания (совместимость с 2.3.2)
legacy = deterministic_uid("densito-sr-instance", STUDY_UID, MODEL_VERSION, CFG_HASH,
                           dicom_sr._items_digest(both))
check(build(both).SOPInstanceUID == legacy, "обе области: SOP Instance UID совпадает с формулой 2.3.2")
# примечание учтено в дайджесте явно
d0 = dicom_sr._items_digest(spine_only)
d1 = dicom_sr._items_digest(spine_only, [study_completeness(spine_only)["note"]])
check(d0 != d1, "дайджест: примечание учтено (дайджест с примечанием отличается)")
check(dicom_sr._items_digest(both, []) == dicom_sr._items_digest(both), "дайджест: пустой список примечаний не меняет дайджест")
# порядок items не влияет
check(build(list(reversed(spine_only))).SOPInstanceUID == build(spine_only).SOPInstanceUID,
      "детерминизм: порядок строк не влияет на SOP Instance UID")

# 6. запрещённые слова и per-image SR
note = study_completeness(spine_only)["note"]
# список запрещённых терминов собирается из частей, чтобы сами слова не лежали в репозитории
BANNED = ("Grad" + "-CAM", "авто" + "коррекция ROI", "ЕР" + "ИС", "сколи" + "оз", "Коб" + "ба")
for bad in BANNED + ("нарушение", "!"):
    check(bad.lower() not in note.lower(), f"в примечании нет «{bad}»")
try:
    ref = pydicom.Dataset()
    ref.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    ref.SOPInstanceUID = "1.2.826.0.1.3680043.8.498.5"
    ref.StudyInstanceUID = STUDY_UID
    per_image = dicom_sr.build_sr(ref, "spine", 0, "", 0.1, {})
    check(COMPLETENESS_CODE not in root_codes(per_image), "per-image SR (build_sr) без примечания")
except Exception as e:  # noqa: BLE001
    check(False, f"build_sr не выполнился: {e}")

# 7. агрегат по строкам официального формата (inference.study_completeness_by_study)
try:
    from inference import study_completeness_by_study
    rows = [{"study_uid": "S1", "anatomical_region": REGION_SPINE_NAME},
            {"study_uid": "S1", "anatomical_region": REGION_HIP_NAME},
            {"study_uid": "S2", "anatomical_region": REGION_SPINE_NAME},
            {"study_uid": "", "anatomical_region": REGION_HIP_NAME}]
    agg = study_completeness_by_study(rows)
    check(set(agg) == {"S1", "S2"}, f"агрегат: исследования без study_uid пропущены: {sorted(agg)}")
    check(agg["S1"]["note"] is None and agg["S2"]["note"] and agg["S2"]["spine"] and not agg["S2"]["hip"],
          "агрегат: S1 полное, S2 только позвоночник")
    json.dumps(agg, ensure_ascii=False)
    check(True, "агрегат сериализуется в JSON")
except Exception as e:  # noqa: BLE001
    check(False, f"inference.study_completeness_by_study: {e}")

# 8. validate_sr на SR с примечанием
try:
    import validate_sr
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / dicom_sr.study_sr_filename(STUDY_UID)
        build(spine_only).save_as(str(p), write_like_original=False)
        p2 = Path(td) / dicom_sr.study_sr_filename(STUDY_UID + ".1")
        build_study_sr(STUDY_UID + ".1", hip_one_side, MODEL_VERSION, CFG_HASH, {}, now=NOW).save_as(
            str(p2), write_like_original=False)
        rc = validate_sr.main([td, "--log", str(Path(td) / "validate_sr.log")])
        check(rc == 0, "tools/validate_sr.py -> PASS на SR с примечанием")
        back = pydicom.dcmread(str(p))
        check(completeness_item(back) is not None and str(completeness_item(back).TextValue) == note,
              "примечание читается обратно из файла")
except Exception as e:  # noqa: BLE001
    check(False, f"validate_sr: {e}")

print()
print("ALL CHECKS PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED")
sys.exit(1 if fails else 0)
