#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pii_scan.py — проверка DICOM-файлов на персональные данные (ПДн) перед публикацией.

    python tools/pii_scan.py [paths...] [--json report.json] [--md report.md] [--strict]

По умолчанию сканируются tests/ и data/ (все *.dcm и файлы с преамбулой DICM, включая внутри zip).
Для каждого файла проверяются идентифицирующие теги (PS3.15 приложение E, базовый профиль):
PatientName, PatientID, PatientBirthDate, OtherPatientIDs, PatientAddress, PatientTelephoneNumbers,
InstitutionName/Address, ReferringPhysicianName, PerformingPhysicianName, OperatorsName,
AccessionNumber, StudyID, DeviceSerialNumber, StationName, а также приватные теги (нечётные группы),
даты (StudyDate, AcquisitionDate, ContentDate) и PatientAge/Sex как «ограниченные».

Значение считается ЧИСТЫМ, если оно пустое или входит в набор заглушек:
Anonymized, ANONYMIZED, Anonymous, PHANTOM*, SYNTHETIC*, «^», «-», «unknown», «xxx».
Любое другое значение в идентифицирующем теге -> ВЫЯВЛЕНО (код возврата 1 при --strict).
Скрипт ничего не изменяет и не пишет пиксели — только читает заголовки.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import pydicom

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))

IDENTIFYING = ["PatientName", "PatientID", "PatientBirthDate", "OtherPatientIDs", "OtherPatientNames",
               "PatientAddress", "PatientTelephoneNumbers", "PatientMotherBirthName", "PatientBirthName",
               "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName",
               "ReferringPhysicianName", "PerformingPhysicianName", "OperatorsName", "PhysiciansOfRecord",
               "AccessionNumber", "StudyID", "DeviceSerialNumber", "StationName", "RequestingPhysician",
               "PatientComments", "ImageComments", "StudyComments"]
RESTRICTED = ["StudyDate", "AcquisitionDate", "ContentDate", "SeriesDate", "PatientAge", "PatientSex",
              "EthnicGroup", "PatientWeight", "PatientSize", "PerformedProcedureStepStartDate"]
PLACEHOLDER = re.compile(r"^(anonym(ized|ous)?|phantom.*|synthetic.*|de-?identified|unknown|none|n/?a|x+|\^*|-+|0+|test.*)$", re.I)
UID_LIKE = re.compile(r"^[0-9.]+$")


def is_clean(value) -> bool:
    s = str(value).strip() if value is not None else ""
    if not s:
        return True
    if PLACEHOLDER.match(s):
        return True
    return False


def scan_dataset(ds: pydicom.Dataset, name: str) -> dict:
    found, restricted, private = [], [], []
    for kw in IDENTIFYING:
        if kw in ds:
            val = ds.data_element(kw).value
            if not is_clean(val):
                found.append({"tag": kw, "value": str(val)[:80]})
    for kw in RESTRICTED:
        if kw in ds:
            val = ds.data_element(kw).value
            if not is_clean(val):
                restricted.append({"tag": kw, "value": str(val)[:40]})
    for el in ds:
        if el.tag.is_private and el.VR not in ("SQ",) and el.value not in (None, b"", ""):
            private.append(str(el.tag))
    # Burned-in annotation: текстовый флаг стандарта; сами пиксели не анализируются
    bia = str(getattr(ds, "BurnedInAnnotation", "") or "")
    return {"file": name, "pii_found": found, "restricted": restricted, "private_tags": private[:20],
            "n_private": len(private), "burned_in_annotation_tag": bia,
            "patient_identity_removed": str(getattr(ds, "PatientIdentityRemoved", "") or ""),
            "deidentification_method": str(getattr(ds, "DeidentificationMethod", "") or "")[:120],
            "clean": not found}


def is_dicom_bytes(b: bytes) -> bool:
    return len(b) > 132 and b[128:132] == b"DICM"


def iter_files(paths: list[Path]):
    for p in paths:
        if p.is_file():
            yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    yield f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--strict", action="store_true", help="код 1, если найдены ПДн")
    a = ap.parse_args()
    paths = [Path(p) for p in a.paths] if a.paths else [ROOT / "tests", ROOT / "data"]
    results, skipped = [], []
    for f in iter_files(paths):
        suffix = f.suffix.lower()
        try:
            if suffix == ".zip":
                with zipfile.ZipFile(f) as z:
                    for zi in z.infolist():
                        if zi.is_dir():
                            continue
                        b = z.read(zi)
                        if zi.filename.lower().endswith(".dcm") or is_dicom_bytes(b):
                            ds = pydicom.dcmread(io.BytesIO(b), force=True, stop_before_pixels=True)
                            results.append(scan_dataset(ds, f"{f}::{zi.filename}"))
                continue
            if suffix in (".png", ".jpg", ".jpeg", ".csv", ".json", ".md", ".py", ".txt", ".xlsx", ".yaml", ".npy", ".pkl", ".pth"):
                continue
            with open(f, "rb") as fh:
                head = fh.read(132)
            if suffix not in (".dcm", ".dicom", ".ima", ".dic") and not is_dicom_bytes(head):
                continue
            ds = pydicom.dcmread(str(f), force=True, stop_before_pixels=True)
            if len(ds) == 0:
                skipped.append({"file": str(f), "reason": "не разобран как DICOM"})
                continue
            results.append(scan_dataset(ds, str(f)))
        except Exception as e:  # noqa: BLE001
            skipped.append({"file": str(f), "reason": f"{type(e).__name__}: {str(e)[:80]}"})

    n_bad = sum(1 for r in results if not r["clean"])
    report = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "paths": [str(p) for p in paths],
              "n_files": len(results), "n_with_pii": n_bad, "files": results, "skipped": skipped}
    if a.json:
        Path(a.json).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"# Отчёт проверки ПДн в DICOM ({report['generated_at']})", "",
             f"Проверено файлов: {len(results)}; с выявленными ПДн: {n_bad}; пропущено (не DICOM/ошибка): {len(skipped)}.",
             f"Каталоги: {', '.join(report['paths'])}.", "",
             "| Файл | ПДн | Идентифицирующие теги (не заглушки) | Ограниченные теги | Приватных тегов | PatientIdentityRemoved |",
             "|---|---|---|---|---|---|"]
    for r in results:
        fn = r["file"].replace(str(ROOT) + "/", "")
        pii = "; ".join(f"{x['tag']}={x['value']}" for x in r["pii_found"]) or "—"
        rs = "; ".join(f"{x['tag']}={x['value']}" for x in r["restricted"]) or "—"
        lines.append(f"| `{fn}` | {'ВЫЯВЛЕНО' if not r['clean'] else 'чисто'} | {pii} | {rs} | {r['n_private']} | {r['patient_identity_removed'] or '—'} |")
    for s in skipped:
        lines.append(f"| `{s['file'].replace(str(ROOT) + '/', '')}` | пропущен | {s['reason']} | | | |")
    lines += ["", "Правило: тег считается чистым, если пустой или содержит заглушку (Anonymized, PHANTOM, SYNTHETIC и т.п.). "
              "Пиксели на «вшитый» текст не анализируются (для DXA GE Lunar вшитых аннотаций в экспорте не наблюдалось; тег BurnedInAnnotation отсутствует)."]
    md = "\n".join(lines) + "\n"
    if a.md:
        Path(a.md).write_text(md, encoding="utf-8")
    print(md)
    return 1 if (a.strict and n_bad) else 0


if __name__ == "__main__":
    sys.exit(main())
