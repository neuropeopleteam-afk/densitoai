#!/usr/bin/env python3
"""Архив дополнительных DICOM-серий (ТЗ п. 2.7: «zip-архив с дополнительными сериями (при наличии функционала)»).

В архив попадают все дополнительные DICOM-серии, созданные прогоном:
  * SR на исследование (один на исследование, включая норму и Failure) — `study_SR.dcm`;
  * наложение результата как Secondary Capture — `image_NNN_<область>_overlay_SC.dcm`;
  * сегментация структур как DICOM SEG — `image_NNN_<область>_SEG.dcm`;
  * SR на снимок, если он включён явно — `image_NNN_<область>_SR.dcm`.

Имена внутри архива не берутся из путей и имён входных файлов (в них бывают ФИО): каталог исследования —
`study_NNN_<StudyInstanceUID>`, номер исследования — по порядку первого появления в results.csv, номер
снимка — по порядку внутри исследования. Соответствие «файл архива -> строка results.csv» — в
`series_index.csv` в корне архива (колонки zip_path, kind, study_uid, image_uid, results_csv_row, region;
results_csv_row — номер строки данных results.csv, с 1). Путь к исходному файлу в индекс не пишется.

Модуль не влияет на results.csv: вызывается после записи выгрузки, любая ошибка упаковки перехватывается
вызывающим.
"""
import csv
import io
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SERIES_ZIP_NAME = "additional_series.zip"
INDEX_NAME = "series_index.csv"
INDEX_COLUMNS = ["zip_path", "kind", "study_uid", "image_uid", "results_csv_row", "region"]
# вид серии -> суффикс имени файла снимка
IMAGE_KINDS = (("sc", "overlay_SC"), ("seg", "SEG"), ("sr_image", "SR"))
# ключи debug-строки движка (DensitoInference) для файлов снимка
DEBUG_KEYS = {"sc": "bonus_overlay_dcm", "seg": "bonus_seg_dcm", "sr_image": "bonus_sr_dcm"}


def _uid_safe(uid: str) -> str:
    s = re.sub(r"[^0-9A-Za-z.\-]+", "_", str(uid or ""))[:64]
    return s or "no_uid"


def _region_code(internal_region: str, official: str = "") -> str:
    r = (internal_region or "").lower()
    if r in ("spine", "right_hip", "left_hip"):
        return r
    if "hip" in r or "бедр" in (official or "").lower():
        return "hip"
    if "spine" in r or "позвон" in (official or "").lower():
        return "spine"
    return "image"


def plan_entries(rows: Sequence[Dict[str, Any]], image_files: Sequence[Dict[str, str]],
                 study_sr: Optional[Dict[str, str]] = None,
                 internal_regions: Optional[Sequence[str]] = None,
                 include_rows: Optional[Sequence[bool]] = None) -> List[Dict[str, Any]]:
    """Список элементов архива.

    rows — строки официального формата (порядок results.csv); image_files[i] — {вид: путь к .dcm} для строки i
    (виды: sc, seg, sr_image); study_sr — {study_uid: путь к SR исследования}; internal_regions[i] — внутренняя
    область строки (spine / right_hip / left_hip); include_rows[i] = False — файлы снимка строки i не включать
    (например, снимок вне поддерживаемой области в API)."""
    study_sr = dict(study_sr or {})
    study_no: Dict[str, int] = {}
    img_no: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        uid = str(r.get("study_uid") or "")
        if uid not in study_no:
            study_no[uid] = len(study_no) + 1
        img_no[uid] = img_no.get(uid, 0) + 1
        folder = f"study_{study_no[uid]:03d}_{_uid_safe(uid)}"
        if include_rows is not None and i < len(include_rows) and not include_rows[i]:
            continue
        files = image_files[i] if i < len(image_files) and isinstance(image_files[i], dict) else {}
        region = _region_code(internal_regions[i] if internal_regions and i < len(internal_regions) else "",
                              str(r.get("anatomical_region") or ""))
        for kind, suffix in IMAGE_KINDS:
            p = str(files.get(kind) or "").strip()
            if p and Path(p).is_file():
                out.append({"src": p, "arc": f"{folder}/image_{img_no[uid]:03d}_{region}_{suffix}.dcm", "kind": kind,
                            "study_uid": uid, "image_uid": str(r.get("image_uid") or ""), "results_csv_row": i + 1,
                            "region": region})
    for uid, n in study_no.items():
        p = str(study_sr.get(uid) or "").strip()
        if p and Path(p).is_file():
            out.append({"src": p, "arc": f"study_{n:03d}_{_uid_safe(uid)}/study_SR.dcm", "kind": "sr_study",
                        "study_uid": uid, "image_uid": "", "results_csv_row": "", "region": ""})
    out.sort(key=lambda e: e["arc"])
    return out


def write_series_zip(entries: Sequence[Dict[str, Any]], zip_path: Path) -> Dict[str, Any]:
    """Записать архив атомарно (временный файл рядом + os.replace). Возвращает {path, n_dicom, by_kind}."""
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".series_", suffix=".zip.tmp", dir=str(zip_path.parent))
    os.close(fd)
    by_kind: Dict[str, int] = {}
    try:
        with zipfile.ZipFile(tmp_name, "w", zipfile.ZIP_DEFLATED) as zf:
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=INDEX_COLUMNS, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            for e in entries:
                zf.write(e["src"], e["arc"])
                w.writerow({**e, "zip_path": e["arc"]})
                by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1
            zf.writestr(INDEX_NAME, buf.getvalue().encode("utf-8"))
        # mkstemp создаёт файл с правами 0600; архив — такой же результат, как results.csv (0644), иначе при
        # монтировании каталога результатов из контейнера его не прочитать другим пользователем
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, zip_path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return {"path": str(zip_path), "n_dicom": sum(by_kind.values()), "by_kind": by_kind}


def image_files_from_debug(debug_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    """{вид: путь} по debug-строкам движка (пакетный режим)."""
    out = []
    for dbg in debug_rows or []:
        d = dbg if isinstance(dbg, dict) else {}
        out.append({k: str(d.get(key) or "") for k, key in DEBUG_KEYS.items()})
    return out
