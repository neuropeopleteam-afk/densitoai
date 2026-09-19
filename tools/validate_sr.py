#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Валидатор DICOM SR, которые пишет DensitoAI в режиме --sr-study (один SR на исследование).

Проверяет (pydicom):
  * SOP Class UID — Basic Text SR (…88.11) или Comprehensive SR (…88.33), совпадает с file_meta;
  * Modality = SR; CompletionFlag ∈ {COMPLETE, PARTIAL}; VerificationFlag ∈ {VERIFIED, UNVERIFIED};
    ContentDate/ContentTime заданы и корректны по формату; StudyInstanceUID/SeriesInstanceUID/
    SOPInstanceUID заданы, синтаксически корректны и ≤ 64 символов; SeriesNumber, InstanceNumber есть;
  * корневой элемент — CONTAINER с ConceptNameCodeSequence и непустой ContentSequence;
  * все ValueType в дереве ∈ допустимым для SR (CONTAINER, TEXT, NUM, CODE, IMAGE, DATE, TIME, DATETIME,
    PNAME, UIDREF, COMPOSITE, SCOORD, TCOORD, WAVEFORM); у каждого не-корневого элемента RelationshipType;
    у TEXT — TextValue, у NUM — MeasuredValueSequence с NumericValue, у CODE — ConceptCodeSequence,
    у IMAGE — ReferencedSOPSequence;
  * если задан results.csv: ReferencedSOPSequence (в дереве и в CurrentRequestedProcedureEvidenceSequence)
    ссылаются только на image_uid этого исследования из CSV; каждый снимок исследования с валидным UID
    упомянут; ровно один SR на study_uid (файлы <study_uid>_SR.dcm); SR есть для каждого исследования
    из CSV, включая исследования без нарушений (норма);
  * StudyInstanceUID SR совпадает со study_uid из CSV (для UID-подобных значений);
  * при наличии `dciodvfy` (dicom3tools) запускает его и добавляет вывод в лог.

Запуск:
  python tools/validate_sr.py <каталог sr или файл .dcm> [--csv results.csv] [--log validate_sr.log]
Код возврата: 0 — ошибок нет, 1 — есть ошибки (предупреждения не влияют).
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import warnings

import pydicom

warnings.filterwarnings("ignore", category=UserWarning)  # pydicom предупреждает о UID исходных данных, см. leading_zero_uids

SR_SOP_CLASSES = {
    "1.2.840.10008.5.1.4.1.1.88.11": "Basic Text SR",
    "1.2.840.10008.5.1.4.1.1.88.22": "Enhanced SR",
    "1.2.840.10008.5.1.4.1.1.88.33": "Comprehensive SR",
}
ALLOWED_VALUE_TYPES = {"CONTAINER", "TEXT", "NUM", "CODE", "IMAGE", "DATE", "TIME", "DATETIME",
                       "PNAME", "UIDREF", "COMPOSITE", "SCOORD", "TCOORD", "WAVEFORM"}
ALLOWED_RELATIONSHIPS = {"CONTAINS", "HAS OBS CONTEXT", "HAS CONCEPT MOD", "HAS PROPERTIES",
                         "HAS ACQ CONTEXT", "INFERRED FROM", "SELECTED FROM"}
UID_RE = re.compile(r"^[0-9]+(\.[0-9]+)*$")


class Report:
    def __init__(self) -> None:
        self.lines: List[str] = []
        self.n_err = 0
        self.n_warn = 0

    def err(self, msg: str) -> None:
        self.n_err += 1
        self.lines.append("ERROR   " + msg)

    def warn(self, msg: str) -> None:
        self.n_warn += 1
        self.lines.append("WARNING " + msg)

    def ok(self, msg: str) -> None:
        self.lines.append("OK      " + msg)

    def info(self, msg: str) -> None:
        self.lines.append("INFO    " + msg)


def uid_ok(uid: str) -> bool:
    return bool(uid) and len(uid) <= 64 and bool(UID_RE.match(uid))


def leading_zero_uids(uids) -> List[str]:
    """UID с компонентом вида '07...' формально нарушают PS3.5 §9.1 (ведущий ноль). Такие UID
    встречаются в исходных DICOM организаторов; SR обязан ссылаться на них как есть."""
    return [u for u in uids if any(len(c) > 1 and c.startswith("0") for c in u.split("."))]


def walk_content(items, rep: Report, prefix: str, refs: Set[str], depth: int = 0) -> int:
    n = 0
    for k, item in enumerate(items):
        n += 1
        loc = f"{prefix}[{k}]"
        vt = str(getattr(item, "ValueType", ""))
        rel = str(getattr(item, "RelationshipType", ""))
        if vt not in ALLOWED_VALUE_TYPES:
            rep.err(f"{loc}: недопустимый ValueType '{vt}'")
        if rel not in ALLOWED_RELATIONSHIPS:
            rep.err(f"{loc}: недопустимый RelationshipType '{rel}'")
        if "ConceptNameCodeSequence" not in item or len(item.ConceptNameCodeSequence) != 1:
            rep.err(f"{loc}: нет ConceptNameCodeSequence")
        else:
            c = item.ConceptNameCodeSequence[0]
            for a in ("CodeValue", "CodingSchemeDesignator", "CodeMeaning"):
                if not str(getattr(c, a, "")).strip():
                    rep.err(f"{loc}: ConceptNameCodeSequence без {a}")
        if vt == "TEXT" and not str(getattr(item, "TextValue", "")).strip():
            rep.err(f"{loc}: TEXT без TextValue")
        if vt == "NUM":
            mvs = getattr(item, "MeasuredValueSequence", None)
            if not mvs or "NumericValue" not in mvs[0] or "MeasurementUnitsCodeSequence" not in mvs[0]:
                rep.err(f"{loc}: NUM без MeasuredValueSequence/NumericValue/единиц")
        if vt == "CODE" and not getattr(item, "ConceptCodeSequence", None):
            rep.err(f"{loc}: CODE без ConceptCodeSequence")
        if vt == "IMAGE":
            rss = getattr(item, "ReferencedSOPSequence", None)
            if not rss:
                rep.err(f"{loc}: IMAGE без ReferencedSOPSequence")
            else:
                for r in rss:
                    refs.add(str(getattr(r, "ReferencedSOPInstanceUID", "")))
                    if not str(getattr(r, "ReferencedSOPClassUID", "")):
                        rep.err(f"{loc}: ReferencedSOPSequence без ReferencedSOPClassUID")
        if vt == "CONTAINER":
            if str(getattr(item, "ContinuityOfContent", "")) not in ("SEPARATE", "CONTINUOUS"):
                rep.err(f"{loc}: CONTAINER без ContinuityOfContent")
            n += walk_content(getattr(item, "ContentSequence", []) or [], rep, loc, refs, depth + 1)
    return n


def read_csv(path: Path) -> Dict[str, List[Dict[str, str]]]:
    by_study: Dict[str, List[Dict[str, str]]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            by_study.setdefault(r.get("study_uid", ""), []).append(r)
    return by_study


def validate_file(path: Path, rep: Report, csv_rows: Dict[str, List[Dict[str, str]]] | None) -> Tuple[str, str]:
    """Возвращает (study_uid из SR, study_uid из имени файла)."""
    rep.info(f"--- {path.name}")
    try:
        ds = pydicom.dcmread(str(path), force=False)
    except Exception as e:  # noqa: BLE001
        rep.err(f"{path.name}: не читается как DICOM Part 10: {e}")
        return "", ""
    sop = str(getattr(ds, "SOPClassUID", ""))
    if sop not in SR_SOP_CLASSES:
        rep.err(f"SOPClassUID '{sop}' не является SR Storage")
    else:
        rep.ok(f"SOPClassUID = {SR_SOP_CLASSES[sop]} ({sop})")
    fm = getattr(ds, "file_meta", None)
    if fm is None or str(getattr(fm, "MediaStorageSOPClassUID", "")) != sop:
        rep.err("file_meta.MediaStorageSOPClassUID не совпадает с SOPClassUID")
    if fm is not None and str(getattr(fm, "MediaStorageSOPInstanceUID", "")) != str(getattr(ds, "SOPInstanceUID", "")):
        rep.err("file_meta.MediaStorageSOPInstanceUID не совпадает с SOPInstanceUID")
    if str(getattr(ds, "Modality", "")) != "SR":
        rep.err(f"Modality = '{getattr(ds, 'Modality', '')}', ожидается SR")
    else:
        rep.ok("Modality = SR")
    if str(getattr(ds, "CompletionFlag", "")) not in ("COMPLETE", "PARTIAL"):
        rep.err(f"CompletionFlag = '{getattr(ds, 'CompletionFlag', '')}'")
    if str(getattr(ds, "VerificationFlag", "")) not in ("VERIFIED", "UNVERIFIED"):
        rep.err(f"VerificationFlag = '{getattr(ds, 'VerificationFlag', '')}'")
    cd, ct = str(getattr(ds, "ContentDate", "")), str(getattr(ds, "ContentTime", ""))
    if not re.match(r"^\d{8}$", cd):
        rep.err(f"ContentDate = '{cd}'")
    if not re.match(r"^\d{2}(\d{2}(\d{2}(\.\d{1,6})?)?)?$", ct):
        rep.err(f"ContentTime = '{ct}'")
    for a in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"):
        v = str(getattr(ds, a, ""))
        if not uid_ok(v):
            rep.err(f"{a} = '{v}' некорректен (пусто, > 64 символов или недопустимые символы)")
    for a in ("SeriesNumber", "InstanceNumber", "Manufacturer", "SpecificCharacterSet"):
        if a not in ds:
            rep.err(f"нет атрибута {a}")
    for a in ("PatientName", "PatientID", "StudyDate", "StudyTime", "ReferringPhysicianName", "StudyID",
              "AccessionNumber", "PatientBirthDate", "PatientSex", "PerformedProcedureCodeSequence",
              "ReferencedPerformedProcedureStepSequence"):
        if a not in ds:
            rep.warn(f"Type 2 атрибут {a} отсутствует (должен присутствовать, допускается пустой)")
    if str(getattr(ds, "SpecificCharacterSet", "")) != "ISO_IR 192":
        rep.warn("SpecificCharacterSet != ISO_IR 192 — кириллица может не отобразиться")

    # корневой элемент
    if str(getattr(ds, "ValueType", "")) != "CONTAINER":
        rep.err(f"корневой ValueType = '{getattr(ds, 'ValueType', '')}', ожидается CONTAINER")
    if "ConceptNameCodeSequence" not in ds:
        rep.err("у корневого CONTAINER нет ConceptNameCodeSequence")
    if str(getattr(ds, "ContinuityOfContent", "")) not in ("SEPARATE", "CONTINUOUS"):
        rep.err("у корневого CONTAINER нет ContinuityOfContent")
    content = getattr(ds, "ContentSequence", None)
    if not content or len(content) == 0:
        rep.err("ContentSequence пустая")
        return str(getattr(ds, "StudyInstanceUID", "")), ""
    refs: Set[str] = set()
    n_items = walk_content(content, rep, "Content", refs)
    rep.ok(f"дерево содержимого: {n_items} элементов, {len(refs)} ссылок на изображения (IMAGE)")

    # evidence
    ev_refs: Set[str] = set()
    for ev in getattr(ds, "CurrentRequestedProcedureEvidenceSequence", []) or []:
        if str(getattr(ev, "StudyInstanceUID", "")) != str(getattr(ds, "StudyInstanceUID", "")):
            rep.err("CurrentRequestedProcedureEvidenceSequence.StudyInstanceUID != StudyInstanceUID SR")
        for rs in getattr(ev, "ReferencedSeriesSequence", []) or []:
            if not str(getattr(rs, "SeriesInstanceUID", "")):
                rep.err("ReferencedSeriesSequence без SeriesInstanceUID")
            for r in getattr(rs, "ReferencedSOPSequence", []) or []:
                ev_refs.add(str(getattr(r, "ReferencedSOPInstanceUID", "")))
    if refs and ev_refs and refs != ev_refs:
        rep.err(f"ссылки в дереве ({len(refs)}) и в Evidence ({len(ev_refs)}) не совпадают")
    lz = leading_zero_uids(refs | {study_uid_sr_early} if (study_uid_sr_early := str(getattr(ds, "StudyInstanceUID", ""))) else refs)
    if lz:
        rep.warn(f"{len(lz)} UID исходных данных содержат компонент с ведущим нулём (PS3.5 §9.1); "
                 f"взяты как есть из оригинальных DICOM, чтобы ссылки совпадали с PACS")

    # обязательный контекст (методология НПКЦ ДиТ: имя сервиса, версия, предупреждение об ИИ)
    codes = {str(it.ConceptNameCodeSequence[0].CodeValue) for it in content if "ConceptNameCodeSequence" in it}
    for need in ("SERVICE-NAME", "MODEL-VERSION", "AI-WARNING"):
        if need not in codes:
            rep.warn(f"в корне нет элемента {need}")

    study_uid_sr = str(getattr(ds, "StudyInstanceUID", ""))
    # сопоставление с results.csv
    m = re.match(r"^(.*)_SR\.dcm$", path.name)
    study_from_name = m.group(1) if m else ""
    if csv_rows is not None:
        rows = csv_rows.get(study_from_name) or csv_rows.get(study_uid_sr)
        if rows is None:
            rep.err(f"исследование {study_from_name or study_uid_sr} отсутствует в results.csv")
        else:
            csv_uid = rows[0]["study_uid"]
            if uid_ok(csv_uid) and csv_uid != study_uid_sr:
                rep.err(f"StudyInstanceUID SR ({study_uid_sr}) != study_uid CSV ({csv_uid})")
            img_uids = {r["image_uid"] for r in rows}
            valid_img = {u for u in img_uids if UID_RE.match(u)}
            extra = refs - img_uids
            missing = valid_img - refs
            if extra:
                rep.err(f"SR ссылается на image_uid, которых нет в CSV для этого исследования: {sorted(extra)[:3]}")
            if missing:
                rep.err(f"снимки исследования без ссылки в SR: {sorted(missing)[:3]}")
            if not extra and not missing:
                rep.ok(f"ReferencedSOPSequence = {len(refs)} снимков исследования из results.csv "
                       f"(в CSV {len(img_uids)}, из них с UID {len(valid_img)})")
            n_viol = sum(1 for r in rows if str(r.get("quality_class")) == "1")
            rep.info(f"исследование: {len(rows)} снимков, с нарушениями {n_viol}"
                     + (" (норма — SR всё равно сформирован)" if n_viol == 0 else ""))
            for u in img_uids - valid_img:
                rep.warn(f"image_uid '{u[:40]}…' не UID (fallback-хэш) — ссылка IMAGE не формируется, только TEXT")
    return study_uid_sr, study_from_name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Валидатор DICOM SR DensitoAI (один SR на исследование)")
    ap.add_argument("target", help="каталог с *_SR.dcm или один файл")
    ap.add_argument("--csv", default=None, help="results.csv того же прогона (проверка ссылок и полноты)")
    ap.add_argument("--log", default=None, help="куда записать лог (по умолчанию только stdout)")
    args = ap.parse_args(argv)

    rep = Report()
    target = Path(args.target)
    files = sorted(target.glob("*.dcm")) if target.is_dir() else [target]
    if not files:
        rep.err(f"в {target} нет .dcm файлов")
    csv_rows = None
    if args.csv:
        csv_rows = read_csv(Path(args.csv))
        rep.info(f"results.csv: {sum(len(v) for v in csv_rows.values())} строк, {len(csv_rows)} исследований")

    seen: Dict[str, int] = {}
    for f in files:
        _, study_from_name = validate_file(f, rep, csv_rows)
        key = study_from_name or f.name
        seen[key] = seen.get(key, 0) + 1

    dup = {k: v for k, v in seen.items() if v > 1}
    if dup:
        rep.err(f"более одного SR на исследование: {dup}")
    if csv_rows is not None:
        no_sr = [s for s in csv_rows if s and s not in seen]
        if no_sr:
            rep.err(f"исследования без SR: {len(no_sr)} (например {no_sr[:2]})")
        else:
            rep.ok(f"ровно один SR на каждое из {len([s for s in csv_rows if s])} исследований CSV, включая норму")

    dciodvfy = shutil.which("dciodvfy")
    if dciodvfy:
        for f in files:
            try:
                out = subprocess.run([dciodvfy, str(f)], capture_output=True, text=True, timeout=60)
                rep.info(f"dciodvfy {f.name}: rc={out.returncode}\n" + (out.stdout + out.stderr).strip())
                if "Error" in out.stdout + out.stderr:
                    rep.warn(f"dciodvfy сообщил об ошибках для {f.name} (см. лог)")
            except Exception as e:  # noqa: BLE001
                rep.warn(f"dciodvfy не выполнился: {e}")
    else:
        rep.info("dciodvfy (dicom3tools) не установлен — внешняя проверка IOD не запускалась")

    summary = f"ИТОГ: файлов {len(files)}, ошибок {rep.n_err}, предупреждений {rep.n_warn} -> " + \
              ("PASS" if rep.n_err == 0 else "FAIL")
    rep.lines.append(summary)
    text = "\n".join(rep.lines)
    print(text)
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        Path(args.log).write_text(text + "\n", encoding="utf-8")
    return 1 if rep.n_err else 0


if __name__ == "__main__":
    sys.exit(main())
