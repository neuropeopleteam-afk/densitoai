#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сводка по отделению: агрегированная картина качества за период без персональных данных.

Для кого. Заведующий отделением и старший лаборант: сколько исследований и снимков обработано,
какая доля с нарушением по области и по типу нарушения, как распределены нарушения по аппаратам
и по датам исследования, сколько файлов не обработано и почему, сколько снимков попало в зону
«не уверен», какие исследования стоит пересмотреть первыми. Решение по исследованию принимает врач;
сводка ничего не меняет в 9 официальных колонках и не содержит персональных данных.

Вход.
  * один или несколько `results.csv` (официальные 9 колонок);
  * технические CSV того же прогона, если есть: `<stem>_debug.csv` (needs_review, uncertain_criteria,
    sha256_pixels, error) и `<stem>_extras.csv` (study_warnings) — подбираются автоматически или
    передаются явно (`--debug`, `--extras`);
  * сведения об аппарате и дате исследования. В технических CSV пайплайна их нет, поэтому источники:
      - `--device-tags` CSV (пишет API в каталоге задачи: device_tags.csv), колонки
        path_to_study, image_uid, manufacturer, model_name, station_hash, serial_hash, study_date;
      - `--dicom-root` — прочитать из заголовков DICOM по путям path_to_study только теги аппарата и даты
        (Manufacturer, ManufacturerModelName, StationName, DeviceSerialNumber, StudyDate/AcquisitionDate/
        SeriesDate/ContentDate). Оператор, учреждение, пациент не читаются и не выводятся.
    StationName и DeviceSerialNumber в сводку попадают только как короткий sha256-хэш.

Выход (`--out-dir`): summary.json (схема schema/department_summary.schema.json), summary.md, summary.csv.
Рядом с каждой долей — доверительный интервал Уилсона 95 % и пометка «мало данных» при n < 20
(порог `--min-n`). Доли нарушений считаются по обработанным файлам (processing_status = Success):
у файлов со статусом Failure класс 0 по конвенции формата, и они не должны занижать долю.

Зона «не уверен» берётся из технического CSV (needs_review — то же поле, что показывает кабинет
в API details); по quality_prob и config.yaml она не выводится, поэтому без технического CSV поле null.

Запуск:
    python tools/department_summary.py --results out/results.csv --out-dir out/summary
    python tools/department_summary.py --results a.csv b.csv --dicom-root /data/Исследования --out-dir out/summary
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, OrderedDict, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = 1
Z95 = 1.959963984540054
DEFAULT_MIN_N = 20
DEFAULT_TOP_N = 10
COLUMNS_9 = ["path_to_study", "study_uid", "image_uid", "anatomical_region", "quality_class",
             "violation_type", "quality_prob", "processing_status", "time_of_processing"]
STATUS_SUCCESS = "Success"
STATUS_FAILURE = "Failure"
SEP = ";"
REGIONS_DEFAULT = ["Поясничный отдел позвоночника", "Проксимальный отдел бедра"]
VIOLATIONS_DEFAULT = {
    "Поясничный отдел позвоночника": ["Некорректная укладка", "Не выравнена ось позвоночника",
                                      "Присутствуют посторонние предметы"],
    "Проксимальный отдел бедра": ["Некорректная укладка", "Некорректная область интереса"],
}
NO_VALUE = "не указан"
NO_DATE = "дата не указана"
# Значения-заглушки обезличивания: не аппарат и не дата.
PLACEHOLDERS = {"", "anonymized", "anonymous", "anon", "none", "null", "n/a", "na", "-", "unknown"}
DEVICE_TAGS_COLUMNS = ["path_to_study", "image_uid", "manufacturer", "model_name", "station_hash",
                       "serial_hash", "study_date"]
DATE_TAGS = ("StudyDate", "AcquisitionDate", "SeriesDate", "ContentDate")
LOW_DATA_TEXT = "мало данных"
# Подписи критериев — те же, что в src/api_server.CRITERION_TITLES
CRITERION_TITLES = {
    "sp_pos": "Укладка (позвоночник)", "sp_axis": "Ось позвоночника", "sp_art": "Посторонние предметы",
    "rh_pos": "Укладка (правое бедро)", "rh_roi": "Область интереса (правое бедро)",
    "lh_pos": "Укладка (левое бедро)", "lh_roi": "Область интереса (левое бедро)",
}


# --------------------------------------------------------------------------- #
# Статистика
# --------------------------------------------------------------------------- #
def wilson_ci(k: int, n: int, z: float = Z95) -> Tuple[Optional[float], Optional[float]]:
    """Интервал Уилсона для доли k/n. При n = 0 — (None, None)."""
    if n <= 0:
        return None, None
    k = max(0, min(int(k), int(n)))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    lo, hi = center - half, center + half
    return max(0.0, lo), min(1.0, hi)


def rate(k: int, n: int, min_n: int = DEFAULT_MIN_N, digits: int = 4) -> Dict[str, Any]:
    """Доля с ДИ Уилсона 95 % и пометкой «мало данных» (n < min_n)."""
    k, n = int(k), int(n)
    lo, hi = wilson_ci(k, n)
    return OrderedDict([
        ("k", k), ("n", n),
        ("rate", None if n == 0 else round(k / n, digits)),
        ("ci_low", None if lo is None else round(lo, digits)),
        ("ci_high", None if hi is None else round(hi, digits)),
        ("low_data", bool(n < min_n)),
    ])


def short_hash(value: Any, salt: str = "", length: int = 12) -> str:
    return hashlib.sha256((salt + str(value)).encode("utf-8")).hexdigest()[:length]


def _clean(value: Any) -> str:
    s = "" if value is None else str(value).strip()
    return "" if s.lower() in PLACEHOLDERS else s


def _num(v: Any) -> Optional[float]:
    try:
        if v is None or str(v).strip() == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _int01(v: Any) -> int:
    try:
        return 1 if int(float(str(v).strip())) == 1 else 0
    except (TypeError, ValueError):
        return 1 if str(v).strip().lower() in ("true", "да") else 0


def parse_dicom_date(value: Any) -> Optional[str]:
    """DA-тег (YYYYMMDD, допускаются разделители) -> ISO-дата или None."""
    s = _clean(value)
    if not s:
        return None
    m = re.search(r"(\d{4})[-./]?(\d{2})[-./]?(\d{2})", s)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
    except ValueError:
        return None


def iso_week(day_iso: str) -> str:
    y, w, _ = date.fromisoformat(day_iso).isocalendar()
    return f"{y}-W{w:02d}"


# --------------------------------------------------------------------------- #
# Чтение входов
# --------------------------------------------------------------------------- #
def read_csv_rows(path: Path, encoding: str = "utf-8-sig") -> List[Dict[str, str]]:
    with open(path, "r", encoding=encoding, newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def load_results(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    """Строки нескольких results.csv; проверяется состав 9 колонок; добавляется _source (имя файла)."""
    rows: List[Dict[str, Any]] = []
    for p in paths:
        with open(p, "r", encoding="utf-8-sig", newline="") as f:
            rd = csv.DictReader(f)
            missing = [c for c in COLUMNS_9 if c not in (rd.fieldnames or [])]
            if missing:
                raise ValueError(f"{p}: нет колонок {missing}")
            for r in rd:
                r = {k: (v if v is not None else "") for k, v in r.items()}
                r["_source"] = p.name
                rows.append(r)
    return rows


def sibling_csv(results_path: Path, suffix: str) -> Optional[Path]:
    cand = results_path.with_name(results_path.stem + suffix + ".csv")
    return cand if cand.is_file() else None


def load_optional(paths: Sequence[Optional[Path]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for p in paths:
        if p is not None and Path(p).is_file():
            out.extend(read_csv_rows(Path(p)))
    return out


class FileIndex:
    """Поиск строки технического CSV по пути файла. В results.csv путь относительный (к входной папке),
    в debug-CSV — тот, что был передан пайплайну (часто абсолютный), поэтому сравниваем по хвосту пути:
    берём самый длинный общий хвост, при котором соответствие однозначно."""

    def __init__(self, rows: Iterable[Dict[str, Any]], keys: Sequence[str] = ("file", "path_to_study")) -> None:
        self.by_suffix: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
        self.n = 0
        for r in rows:
            self.n += 1
            for k in keys:
                v = str(r.get(k) or "").strip()
                if v:
                    parts = tuple(Path(v.replace("\\", "/")).as_posix().split("/"))
                    parts = tuple(x for x in parts if x not in ("", "."))
                    for i in range(len(parts)):
                        self.by_suffix[parts[i:]].append(r)
                    break

    def get(self, path: Any) -> Optional[Dict[str, Any]]:
        v = str(path or "").strip()
        if not v or not self.n:
            return None
        parts = tuple(x for x in Path(v.replace("\\", "/")).as_posix().split("/") if x not in ("", "."))
        for i in range(len(parts)):
            cand = self.by_suffix.get(parts[i:])
            if cand:
                return cand[0] if len(cand) == 1 else None
        return None


def index_by_file(rows: Iterable[Dict[str, Any]], keys: Sequence[str] = ("file", "path_to_study")) -> FileIndex:
    return FileIndex(rows, keys)


def index_by_image(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r.get("image_uid") or ""): r for r in rows if str(r.get("image_uid") or "")}


# --------------------------------------------------------------------------- #
# Аппарат и дата
# --------------------------------------------------------------------------- #
def device_tags_from_dataset(ds, salt: str = "") -> Dict[str, Any]:
    """Только аппарат и дата. Ничего про оператора, учреждение или пациента."""
    def g(name: str) -> str:
        try:
            v = getattr(ds, name, None)
        except Exception:  # noqa: BLE001
            v = None
        return _clean(v)

    study_date = None
    for t in DATE_TAGS:
        study_date = parse_dicom_date(g(t))
        if study_date:
            break
    station, serial = g("StationName"), g("DeviceSerialNumber")
    return OrderedDict([
        ("manufacturer", g("Manufacturer") or NO_VALUE),
        ("model_name", g("ManufacturerModelName") or NO_VALUE),
        ("station_hash", short_hash(station, salt) if station else ""),
        ("serial_hash", short_hash(serial, salt) if serial else ""),
        ("study_date", study_date or ""),
    ])


def read_device_tags_dicom(rows: Sequence[Dict[str, Any]], dicom_root: Path, salt: str = "") -> List[Dict[str, Any]]:
    """Прочитать теги аппарата/даты по путям path_to_study (stop_before_pixels)."""
    import pydicom  # локальный импорт: инструмент работает и без pydicom, если теги не нужны
    out: List[Dict[str, Any]] = []
    cache: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        rel = str(r.get("path_to_study") or "")
        if not rel:
            continue
        cands = [dicom_root / rel, Path(rel)]
        src = next((c for c in cands if c.is_file()), None)
        if src is None:
            # относительный путь может включать имя корневой папки (Исследования/...): пробуем без первого сегмента
            parts = Path(rel).parts
            if len(parts) > 1 and (dicom_root / Path(*parts[1:])).is_file():
                src = dicom_root / Path(*parts[1:])
        if src is None:
            continue
        key = str(src)
        if key not in cache:
            try:
                ds = pydicom.dcmread(str(src), stop_before_pixels=True, force=True)
                cache[key] = device_tags_from_dataset(ds, salt)
            except Exception:  # noqa: BLE001 — не DICOM или битый файл: сводка не должна падать
                cache[key] = {}
        tags = cache[key]
        if tags:
            out.append(OrderedDict([("path_to_study", rel), ("image_uid", str(r.get("image_uid") or ""))] + list(tags.items())))
    return out


def write_device_tags_csv(tags_rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DEVICE_TAGS_COLUMNS, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in tags_rows:
            w.writerow(r)


def device_tags_from_debug(dbg: Dict[str, Any], salt: str = "") -> Dict[str, Any]:
    """Если будущий технический CSV будет содержать теги аппарата — подхватываем их (сейчас их там нет)."""
    def pick(*names: str) -> str:
        for n in names:
            if n in dbg:
                return _clean(dbg.get(n))
        return ""
    man = pick("manufacturer", "Manufacturer", "tag_Manufacturer")
    model = pick("model_name", "ManufacturerModelName", "tag_ManufacturerModelName")
    station = pick("station_hash")
    if not station:
        raw = pick("station_name", "StationName", "tag_StationName")
        station = short_hash(raw, salt) if raw else ""
    serial = pick("serial_hash")
    if not serial:
        raw = pick("device_serial", "DeviceSerialNumber", "tag_DeviceSerialNumber")
        serial = short_hash(raw, salt) if raw else ""
    d = ""
    for n in ("study_date", "StudyDate", "tag_StudyDate", "AcquisitionDate", "SeriesDate", "ContentDate"):
        if n in dbg:
            d = parse_dicom_date(dbg.get(n)) or ""
            if d:
                break
    if not (man or model or station or serial or d):
        return {}
    return {"manufacturer": man or NO_VALUE, "model_name": model or NO_VALUE, "station_hash": station,
            "serial_hash": serial, "study_date": d}


def device_key(tags: Dict[str, Any]) -> Tuple[str, str, str]:
    """Ключ аппарата: хэш StationName, иначе хэш серийного номера, иначе производитель+модель."""
    man = _clean(tags.get("manufacturer")) or NO_VALUE
    model = _clean(tags.get("model_name")) or NO_VALUE
    ident = _clean(tags.get("station_hash")) or _clean(tags.get("serial_hash")) or ""
    return (man, model, ident)


def device_label(key: Tuple[str, str, str]) -> str:
    man, model, ident = key
    base = " ".join(x for x in (man, model) if x and x != NO_VALUE) or NO_VALUE
    return f"{base} · аппарат {ident}" if ident else f"{base} · аппарат {NO_VALUE}"


# --------------------------------------------------------------------------- #
# Сводка
# --------------------------------------------------------------------------- #
def _violations(row: Dict[str, Any]) -> List[str]:
    return [v.strip() for v in str(row.get("violation_type") or "").split(SEP) if v.strip()]


def _failure_reason(dbg: Optional[Dict[str, Any]]) -> str:
    if not dbg:
        return "причина не записана (нет технического CSV)"
    err = str(dbg.get("error") or "").strip()
    if not err:
        rs = str(dbg.get("region_support_reason") or "").strip()
        if rs:
            return rs
        return "причина не записана"
    # без путей и без UID: только тип/текст ошибки
    err = re.sub(r"(/[^\s:]+)+", "<путь>", err)
    err = re.sub(r"\b\d{1,3}(\.\d{1,3}){3}\b", "<адрес>", err)
    err = re.sub(r"\b[0-9]+(\.[0-9]+){4,}\b", "<uid>", err)
    return err[:120]


def build_summary(rows: List[Dict[str, Any]], debug_rows: Optional[List[Dict[str, Any]]] = None,
                  extras_rows: Optional[List[Dict[str, Any]]] = None,
                  device_rows: Optional[List[Dict[str, Any]]] = None,
                  cfg: Optional[Dict[str, Any]] = None, min_n: int = DEFAULT_MIN_N,
                  top_n: int = DEFAULT_TOP_N, salt: str = "", inputs: Optional[Dict[str, Any]] = None,
                  pipeline_version: Optional[str] = None, config_hash: Optional[str] = None) -> Dict[str, Any]:
    cfg = cfg or {}
    out_cfg = cfg.get("output", {}) if isinstance(cfg, dict) else {}
    st_ok = str(out_cfg.get("status_success", STATUS_SUCCESS))
    st_fail = str(out_cfg.get("status_failure", STATUS_FAILURE))
    regions_cfg = cfg.get("regions", {}) if isinstance(cfg, dict) else {}
    regions = [regions_cfg.get("spine", REGIONS_DEFAULT[0]), regions_cfg.get("hip", REGIONS_DEFAULT[1])]
    viol_cfg = cfg.get("violations", {}) if isinstance(cfg, dict) else {}
    viol_by_region = {
        regions[0]: [viol_cfg.get(c, VIOLATIONS_DEFAULT[REGIONS_DEFAULT[0]][i]) for i, c in enumerate(("sp_pos", "sp_axis", "sp_art"))],
        regions[1]: [viol_cfg.get(c, VIOLATIONS_DEFAULT[REGIONS_DEFAULT[1]][i]) for i, c in enumerate(("hip_pos", "hip_roi"))],
    }
    # порядок уникальных по своему появлению
    def uniq(seq: Iterable[str]) -> List[str]:
        seen: List[str] = []
        for x in seq:
            if x not in seen:
                seen.append(x)
        return seen
    for reg in regions:
        viol_by_region[reg] = uniq(viol_by_region[reg])

    dbg_by_file = index_by_file(debug_rows or [])
    dbg_by_img = index_by_image(debug_rows or [])
    ext_by_file = index_by_file(extras_rows or [])
    dev_by_file = index_by_file(device_rows or [], keys=("path_to_study", "file"))
    dev_by_img = index_by_image(device_rows or [])

    def dbg_for(r: Dict[str, Any], i: int) -> Optional[Dict[str, Any]]:
        p = str(r.get("path_to_study") or "")
        d = dbg_by_file.get(p) or dbg_by_img.get(str(r.get("image_uid") or ""))
        if d is None and debug_rows and len(debug_rows) == len(rows):
            d = debug_rows[i]  # тот же прогон: порядок строк совпадает
        return d

    def dev_for(r: Dict[str, Any], dbg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        p = str(r.get("path_to_study") or "")
        d = dev_by_file.get(p) or dev_by_img.get(str(r.get("image_uid") or ""))
        if d:
            return d
        return device_tags_from_debug(dbg, salt) if dbg else {}

    notes: List[str] = []
    has_debug = bool(debug_rows)
    has_device = False
    has_date = False

    # ---- по файлам
    n_files = len(rows)
    per_region_files: Dict[str, Dict[str, int]] = {reg: Counter() for reg in regions}
    per_region_viol: Dict[str, Counter] = {reg: Counter() for reg in regions}
    per_region_study_viol: Dict[str, Dict[str, set]] = {reg: defaultdict(set) for reg in regions}
    per_region_studies: Dict[str, set] = {reg: set() for reg in regions}
    unknown_regions: Counter = Counter()
    uncertain_files = 0
    uncertain_success = 0
    uncertain_crits: Counter = Counter()
    fail_reasons: Counter = Counter()
    pixel_hashes: set = set()
    devices: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    dates: Dict[str, Dict[str, Any]] = {}
    n_no_date = 0
    studies: Dict[str, Dict[str, Any]] = {}
    total_time = 0.0
    n_success = n_failure = n_violation = 0
    study_warn: Counter = Counter()

    for i_row, r in enumerate(rows):
        status = str(r.get("processing_status") or "")
        is_fail = status == st_fail
        is_ok = status == st_ok
        qc = _int01(r.get("quality_class"))
        is_viol = bool(is_ok and qc == 1)
        reg = str(r.get("anatomical_region") or "")
        suid = str(r.get("study_uid") or "")
        iuid = str(r.get("image_uid") or "")
        qp = _num(r.get("quality_prob"))
        total_time += _num(r.get("time_of_processing")) or 0.0
        dbg = dbg_for(r, i_row)
        n_success += int(is_ok)
        n_failure += int(is_fail)
        n_violation += int(is_viol)
        if reg not in per_region_files:
            unknown_regions[reg] += 1
        else:
            c = per_region_files[reg]
            c["n_files"] += 1
            c["n_success"] += int(is_ok)
            c["n_failure"] += int(is_fail)
            c["n_violation"] += int(is_viol)
            if suid:
                per_region_studies[reg].add(suid)
            if is_viol:
                for v in _violations(r):
                    per_region_viol[reg][v] += 1
                    per_region_study_viol[reg][v].add(suid)
        if is_fail:
            fail_reasons[_failure_reason(dbg)] += 1
        if dbg:
            if _int01(dbg.get("needs_review")):
                uncertain_files += 1
                uncertain_success += int(is_ok)
                for cr in str(dbg.get("uncertain_criteria") or "").split(SEP):
                    if cr.strip():
                        uncertain_crits[cr.strip()] += 1
            ph = str(dbg.get("sha256_pixels") or "").strip()
            if ph:
                pixel_hashes.add(ph)
        ext = ext_by_file.get(r.get("path_to_study"))
        if ext:
            for wtxt in str(ext.get("study_warnings") or "").split("|"):
                if wtxt.strip():
                    study_warn[wtxt.strip()] += 1
        # аппарат / дата
        dev = dev_for(r, dbg)
        if dev:
            has_device = True
        key = device_key(dev) if dev else (NO_VALUE, NO_VALUE, "")
        dv = devices.setdefault(key, {"n_files": 0, "n_success": 0, "n_failure": 0, "n_violation": 0, "studies": set(), "n_uncertain": 0})
        dv["n_files"] += 1
        dv["n_success"] += int(is_ok)
        dv["n_failure"] += int(is_fail)
        dv["n_violation"] += int(is_viol)
        if dbg and _int01(dbg.get("needs_review")):
            dv["n_uncertain"] += 1
        if suid:
            dv["studies"].add(suid)
        d_iso = parse_dicom_date((dev or {}).get("study_date")) if dev else None
        if d_iso:
            has_date = True
            dd = dates.setdefault(d_iso, {"n_files": 0, "n_success": 0, "n_failure": 0, "n_violation": 0, "studies": set()})
            dd["n_files"] += 1
            dd["n_success"] += int(is_ok)
            dd["n_failure"] += int(is_fail)
            dd["n_violation"] += int(is_viol)
            if suid:
                dd["studies"].add(suid)
        else:
            n_no_date += 1
        # исследования
        if suid:
            s = studies.setdefault(suid, {"study_uid": suid, "n_files": 0, "n_success": 0, "n_failure": 0, "n_violation": 0,
                                          "regions": set(), "violation_types": set(), "max_quality_prob": None,
                                          "image_uid_max": "", "n_uncertain": 0, "study_date": None})
            s["n_files"] += 1
            s["n_success"] += int(is_ok)
            s["n_failure"] += int(is_fail)
            s["n_violation"] += int(is_viol)
            if reg:
                s["regions"].add(reg)
            if is_viol:
                s["violation_types"].update(_violations(r))
            if dbg and _int01(dbg.get("needs_review")):
                s["n_uncertain"] += 1
            if is_ok and qp is not None and (s["max_quality_prob"] is None or qp > s["max_quality_prob"]
                                             or (qp == s["max_quality_prob"] and iuid < s["image_uid_max"])):
                s["max_quality_prob"] = qp
                s["image_uid_max"] = iuid
            if d_iso and not s["study_date"]:
                s["study_date"] = d_iso

    n_studies = len(studies)
    n_norm = n_success - n_violation
    studies_with_viol = sum(1 for s in studies.values() if s["n_violation"] > 0)
    studies_with_fail = sum(1 for s in studies.values() if s["n_failure"] > 0)
    studies_processed = sum(1 for s in studies.values() if s["n_success"] > 0)
    studies_uncertain = sum(1 for s in studies.values() if s["n_uncertain"] > 0)

    # ---- разрезы
    by_region = []
    for reg in regions:
        c = per_region_files[reg]
        by_region.append(OrderedDict([
            ("region", reg), ("n_files", c["n_files"]), ("n_studies", len(per_region_studies[reg])),
            ("n_success", c["n_success"]), ("n_failure", c["n_failure"]), ("n_violation", c["n_violation"]),
            ("n_norm", c["n_success"] - c["n_violation"]),
            ("violation_rate", rate(c["n_violation"], c["n_success"], min_n)),
            ("failure_rate", rate(c["n_failure"], c["n_files"], min_n)),
            ("studies_with_violation", rate(len({s for vs in per_region_study_viol[reg].values() for s in vs}),
                                            len(per_region_studies[reg]), min_n)),
        ]))
    by_violation_type = []
    for reg in regions:
        c = per_region_files[reg]
        for v in viol_by_region[reg]:
            by_violation_type.append(OrderedDict([
                ("region", reg), ("violation_type", v), ("n_files", per_region_viol[reg].get(v, 0)),
                ("n_studies", len(per_region_study_viol[reg].get(v, set()))),
                ("rate_files", rate(per_region_viol[reg].get(v, 0), c["n_success"], min_n)),
                ("rate_studies", rate(len(per_region_study_viol[reg].get(v, set())), len(per_region_studies[reg]), min_n)),
            ]))
        for v, k in sorted(per_region_viol[reg].items()):
            if v not in viol_by_region[reg]:
                notes.append(f"{reg}: тип нарушения вне официального списка: «{v}» ({k})")
    by_device = []
    for key in sorted(devices, key=lambda k: (-devices[k]["n_files"], k)):
        dv = devices[key]
        by_device.append(OrderedDict([
            ("device", device_label(key)), ("manufacturer", key[0]), ("model_name", key[1]), ("device_hash", key[2]),
            ("n_files", dv["n_files"]), ("n_studies", len(dv["studies"])), ("n_success", dv["n_success"]),
            ("n_failure", dv["n_failure"]), ("n_violation", dv["n_violation"]),
            ("violation_rate", rate(dv["n_violation"], dv["n_success"], min_n)),
            ("failure_rate", rate(dv["n_failure"], dv["n_files"], min_n)),
            ("uncertain_rate", rate(dv["n_uncertain"], dv["n_success"], min_n) if has_debug else None),
        ]))
    by_day = []
    weeks: Dict[str, Dict[str, Any]] = {}
    for d_iso in sorted(dates):
        dd = dates[d_iso]
        by_day.append(OrderedDict([
            ("period", d_iso), ("n_files", dd["n_files"]), ("n_studies", len(dd["studies"])), ("n_success", dd["n_success"]),
            ("n_failure", dd["n_failure"]), ("n_violation", dd["n_violation"]),
            ("violation_rate", rate(dd["n_violation"], dd["n_success"], min_n)),
        ]))
        wk = weeks.setdefault(iso_week(d_iso), {"n_files": 0, "n_success": 0, "n_failure": 0, "n_violation": 0, "studies": set()})
        for k in ("n_files", "n_success", "n_failure", "n_violation"):
            wk[k] += dd[k]
        wk["studies"] |= dd["studies"]
    by_week = [OrderedDict([
        ("period", w), ("n_files", wk["n_files"]), ("n_studies", len(wk["studies"])), ("n_success", wk["n_success"]),
        ("n_failure", wk["n_failure"]), ("n_violation", wk["n_violation"]),
        ("violation_rate", rate(wk["n_violation"], wk["n_success"], min_n)),
    ]) for w, wk in sorted(weeks.items())]

    # ---- топ для пересмотра: только UID, по quality_prob обработанных файлов
    ranked = sorted((s for s in studies.values() if s["max_quality_prob"] is not None),
                    key=lambda s: (-s["max_quality_prob"], -s["n_violation"], s["study_uid"]))
    review_top = [OrderedDict([
        ("rank", i + 1), ("study_uid", s["study_uid"]), ("image_uid", s["image_uid_max"]),
        ("max_quality_prob", round(s["max_quality_prob"], 6)), ("n_files", s["n_files"]), ("n_violation", s["n_violation"]),
        ("n_uncertain", s["n_uncertain"] if has_debug else None),
        ("regions", sorted(s["regions"])), ("violation_types", sorted(s["violation_types"])),
        ("study_date", s["study_date"]),
    ]) for i, s in enumerate(ranked[:max(0, int(top_n))])]

    if not has_debug:
        notes.append("Технический CSV не передан: зона «не уверен», причины отказов и число уникальных кадров не определены.")
    if not has_device:
        notes.append("Сведения об аппарате отсутствуют во входных данных: разрез по аппарату содержит одну группу «не указан».")
    elif not has_date:
        notes.append("Дата исследования в заголовках не указана (обезличена): разрез по датам пуст.")
    if unknown_regions:
        notes.append("Строки с областью вне официального списка: " + ", ".join(f"«{k}» ({v})" for k, v in sorted(unknown_regions.items())))
    if n_files < min_n:
        notes.append(f"Мало данных: файлов меньше {min_n}, доли и интервалы ориентировочные.")

    date_values = sorted(dates)
    summary = OrderedDict([
        ("schema_version", SCHEMA_VERSION),
        ("kind", "department_summary"),
        ("pipeline_version", pipeline_version),
        ("config_hash", config_hash),
        ("inputs", inputs or {}),
        ("method", OrderedDict([
            ("ci", "Уилсон, 95 %"), ("min_n", int(min_n)), ("low_data_text", LOW_DATA_TEXT),
            ("violation_denominator", "файлы со статусом Success"),
            ("uncertain_source", "needs_review из технического CSV" if has_debug else None),
            ("device_id", "sha256 StationName (иначе DeviceSerialNumber), первые 12 символов"),
        ])),
        ("period", OrderedDict([("date_min", date_values[0] if date_values else None),
                                ("date_max", date_values[-1] if date_values else None),
                                ("n_files_without_date", n_no_date)])),
        ("totals", OrderedDict([
            ("n_files", n_files), ("n_studies", n_studies), ("n_unique_images", len({str(r.get("image_uid") or "") for r in rows if r.get("image_uid")})),
            ("n_unique_pixel_hashes", len(pixel_hashes) if has_debug and pixel_hashes else None),
            ("n_success", n_success), ("n_failure", n_failure), ("n_violation", n_violation), ("n_norm", n_norm),
            ("n_uncertain", uncertain_files if has_debug else None),
            ("n_studies_processed", studies_processed), ("n_studies_with_violation", studies_with_viol),
            ("n_studies_with_failure", studies_with_fail), ("n_studies_uncertain", studies_uncertain if has_debug else None),
            ("processing_time_total_s", round(total_time, 3)),
            ("processing_time_mean_s", round(total_time / n_files, 4) if n_files else None),
        ])),
        ("violation_rate", rate(n_violation, n_success, min_n)),
        ("violation_rate_studies", rate(studies_with_viol, studies_processed, min_n)),
        ("failure_rate", rate(n_failure, n_files, min_n)),
        ("uncertain_rate", rate(uncertain_success, n_success, min_n) if has_debug else None),
        ("uncertain_by_criterion", [OrderedDict([("criterion", c), ("title", CRITERION_TITLES.get(c, c)), ("n_files", k)])
                                    for c, k in sorted(uncertain_crits.items())] if has_debug else None),
        ("failure_reasons", [OrderedDict([("reason", rsn), ("n_files", k), ("share", rate(k, n_failure, min_n))])
                             for rsn, k in sorted(fail_reasons.items(), key=lambda kv: (-kv[1], kv[0]))]),
        ("by_region", by_region),
        ("by_violation_type", by_violation_type),
        ("by_device", by_device),
        ("by_date", OrderedDict([("day", by_day), ("week", by_week), ("n_files_without_date", n_no_date)])),
        ("study_warnings", [OrderedDict([("warning", w), ("n_files", k)]) for w, k in sorted(study_warn.items(), key=lambda kv: (-kv[1], kv[0]))]),
        ("review_top", review_top),
        ("notes", notes),
    ])
    return summary


# --------------------------------------------------------------------------- #
# Представления: Markdown и CSV
# --------------------------------------------------------------------------- #
def fmt_rate(r: Optional[Dict[str, Any]]) -> str:
    if not r or r.get("rate") is None:
        return "—"
    s = f"{100 * r['rate']:.1f} % [{100 * r['ci_low']:.1f}; {100 * r['ci_high']:.1f}]"
    if r.get("low_data"):
        s += f" ({LOW_DATA_TEXT}, n = {r['n']})"
    return s


def to_markdown(s: Dict[str, Any]) -> str:
    t = s["totals"]
    L: List[str] = []
    L.append("# Сводка по отделению: качество DXA-снимков")
    L.append("")
    per = s.get("period") or {}
    if per.get("date_min"):
        L.append(f"Период по дате исследования: {per['date_min']} — {per['date_max']}"
                 + (f" (без даты: {per.get('n_files_without_date', 0)} файлов)" if per.get("n_files_without_date") else "") + ".")
    else:
        L.append("Период по дате исследования: дата в заголовках не указана.")
    if s.get("pipeline_version"):
        L.append(f"Версия пайплайна: {s['pipeline_version']}" + (f", конфигурация {s['config_hash']}" if s.get("config_hash") else "") + ".")
    L.append(f"Доли даны по обработанным файлам (Success); в квадратных скобках — доверительный интервал Уилсона 95 %; "
             f"пометка «{LOW_DATA_TEXT}» при n < {s['method']['min_n']}.")
    L.append("")
    L.append("## Итого")
    L.append("")
    L.append("| Показатель | Значение |")
    L.append("|---|---|")
    L.append(f"| Исследований | {t['n_studies']} |")
    L.append(f"| Файлов (снимков) | {t['n_files']} |")
    if t.get("n_unique_pixel_hashes") is not None:
        L.append(f"| Уникальных кадров по хэшу пикселей | {t['n_unique_pixel_hashes']} |")
    L.append(f"| Обработано / не обработано | {t['n_success']} / {t['n_failure']} |")
    L.append(f"| С нарушением (файлы) | {t['n_violation']} — {fmt_rate(s['violation_rate'])} |")
    L.append(f"| Норма (файлы) | {t['n_norm']} |")
    L.append(f"| Исследований с нарушением | {t['n_studies_with_violation']} из {t['n_studies_processed']} — {fmt_rate(s['violation_rate_studies'])} |")
    L.append(f"| Не обработано (Failure) | {t['n_failure']} — {fmt_rate(s['failure_rate'])} |")
    if s.get("uncertain_rate") is not None:
        L.append(f"| Зона «не уверен» (файлы) | {t['n_uncertain']} — {fmt_rate(s['uncertain_rate'])} |")
    else:
        L.append("| Зона «не уверен» | не определена (нет технического CSV) |")
    L.append(f"| Время обработки | {t['processing_time_total_s']} с всего, {t['processing_time_mean_s']} с на файл |")
    L.append("")
    L.append("## По области")
    L.append("")
    L.append("| Область | Исследований | Файлов | Обработано | С нарушением | Доля нарушений | Исследований с нарушением | Не обработано |")
    L.append("|---|---:|---:|---:|---:|---|---|---|")
    for r in s["by_region"]:
        L.append(f"| {r['region']} | {r['n_studies']} | {r['n_files']} | {r['n_success']} | {r['n_violation']} | {fmt_rate(r['violation_rate'])} | {fmt_rate(r['studies_with_violation'])} | {fmt_rate(r['failure_rate'])} |")
    L.append("")
    L.append("## По типу нарушения")
    L.append("")
    L.append("Один файл может иметь несколько типов нарушения, поэтому доли по типам не складываются в долю нарушений.")
    L.append("")
    L.append("| Область | Тип нарушения | Файлов | Доля файлов | Исследований | Доля исследований |")
    L.append("|---|---|---:|---|---:|---|")
    for r in s["by_violation_type"]:
        L.append(f"| {r['region']} | {r['violation_type']} | {r['n_files']} | {fmt_rate(r['rate_files'])} | {r['n_studies']} | {fmt_rate(r['rate_studies'])} |")
    L.append("")
    L.append("## По аппарату")
    L.append("")
    L.append("Аппарат обозначен производителем, моделью и хэшем StationName (или серийного номера); имена операторов не читаются.")
    L.append("")
    L.append("| Аппарат | Исследований | Файлов | С нарушением | Доля нарушений | Не обработано | Зона «не уверен» |")
    L.append("|---|---:|---:|---:|---|---|---|")
    for r in s["by_device"]:
        L.append(f"| {r['device']} | {r['n_studies']} | {r['n_files']} | {r['n_violation']} | {fmt_rate(r['violation_rate'])} | {fmt_rate(r['failure_rate'])} | {fmt_rate(r.get('uncertain_rate'))} |")
    L.append("")
    L.append("## По дате исследования")
    L.append("")
    bd = s["by_date"]
    if not bd["day"]:
        L.append(f"Дата исследования не указана ни в одном файле (без даты: {bd['n_files_without_date']}).")
    else:
        L.append("| Неделя | Исследований | Файлов | С нарушением | Доля нарушений |")
        L.append("|---|---:|---:|---:|---|")
        for r in bd["week"]:
            L.append(f"| {r['period']} | {r['n_studies']} | {r['n_files']} | {r['n_violation']} | {fmt_rate(r['violation_rate'])} |")
        L.append("")
        L.append("| День | Исследований | Файлов | С нарушением | Доля нарушений |")
        L.append("|---|---:|---:|---:|---|")
        for r in bd["day"]:
            L.append(f"| {r['period']} | {r['n_studies']} | {r['n_files']} | {r['n_violation']} | {fmt_rate(r['violation_rate'])} |")
        if bd["n_files_without_date"]:
            L.append("")
            L.append(f"Без даты: {bd['n_files_without_date']} файлов.")
    L.append("")
    L.append("## Не обработано (Failure)")
    L.append("")
    if not s["failure_reasons"]:
        L.append("Отказов обработки нет.")
    else:
        L.append("| Причина | Файлов | Доля среди отказов |")
        L.append("|---|---:|---|")
        for r in s["failure_reasons"]:
            L.append(f"| {r['reason']} | {r['n_files']} | {fmt_rate(r['share'])} |")
    L.append("")
    if s.get("uncertain_by_criterion"):
        L.append("## Зона «не уверен» по критериям")
        L.append("")
        L.append("| Критерий | Код | Файлов |")
        L.append("|---|---|---:|")
        for r in s["uncertain_by_criterion"]:
            L.append(f"| {r['title']} | {r['criterion']} | {r['n_files']} |")
        L.append("")
    if s.get("study_warnings"):
        L.append("## Предупреждения по исследованиям")
        L.append("")
        L.append("| Предупреждение | Файлов |")
        L.append("|---|---:|")
        for r in s["study_warnings"]:
            L.append(f"| {r['warning']} | {r['n_files']} |")
        L.append("")
    L.append("## Исследования для пересмотра")
    L.append("")
    L.append("Первые по quality_prob (максимум по файлам исследования); только UID, без персональных данных. Решение принимает врач.")
    L.append("")
    L.append("| № | study_uid | image_uid | quality_prob | Файлов | С нарушением | Типы нарушений |")
    L.append("|---:|---|---|---:|---:|---:|---|")
    for r in s["review_top"]:
        L.append(f"| {r['rank']} | {r['study_uid']} | {r['image_uid']} | {r['max_quality_prob']:.3f} | {r['n_files']} | {r['n_violation']} | {'; '.join(r['violation_types']) or '—'} |")
    if s.get("notes"):
        L.append("")
        L.append("## Примечания")
        L.append("")
        for n in s["notes"]:
            L.append(f"- {n}")
    L.append("")
    return "\n".join(L)


CSV_COLUMNS = ["section", "group", "subgroup", "metric", "k", "n", "rate", "ci_low", "ci_high", "low_data"]


def to_csv_rows(s: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Плоская таблица долей: одна строка — одна доля с ДИ (для Excel/BI)."""
    out: List[Dict[str, Any]] = []

    def add(section: str, group: str, subgroup: str, metric: str, r: Optional[Dict[str, Any]]) -> None:
        if not r:
            return
        out.append({"section": section, "group": group, "subgroup": subgroup, "metric": metric,
                    "k": r["k"], "n": r["n"], "rate": r["rate"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                    "low_data": int(bool(r["low_data"]))})

    add("итого", "", "", "доля нарушений (файлы)", s["violation_rate"])
    add("итого", "", "", "доля исследований с нарушением", s["violation_rate_studies"])
    add("итого", "", "", "доля Failure", s["failure_rate"])
    add("итого", "", "", "доля зоны «не уверен»", s.get("uncertain_rate"))
    for r in s["by_region"]:
        add("по области", r["region"], "", "доля нарушений (файлы)", r["violation_rate"])
        add("по области", r["region"], "", "доля исследований с нарушением", r["studies_with_violation"])
        add("по области", r["region"], "", "доля Failure", r["failure_rate"])
    for r in s["by_violation_type"]:
        add("по типу нарушения", r["region"], r["violation_type"], "доля файлов", r["rate_files"])
        add("по типу нарушения", r["region"], r["violation_type"], "доля исследований", r["rate_studies"])
    for r in s["by_device"]:
        add("по аппарату", r["device"], "", "доля нарушений (файлы)", r["violation_rate"])
        add("по аппарату", r["device"], "", "доля Failure", r["failure_rate"])
        add("по аппарату", r["device"], "", "доля зоны «не уверен»", r.get("uncertain_rate"))
    for r in s["by_date"]["week"]:
        add("по неделе", r["period"], "", "доля нарушений (файлы)", r["violation_rate"])
    for r in s["by_date"]["day"]:
        add("по дню", r["period"], "", "доля нарушений (файлы)", r["violation_rate"])
    for r in s["failure_reasons"]:
        add("причины Failure", r["reason"], "", "доля среди отказов", r["share"])
    return out


def to_csv_text(s: Dict[str, Any]) -> str:
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, delimiter=";", lineterminator="\n")
    w.writeheader()
    for r in to_csv_rows(s):
        w.writerow(r)
    return buf.getvalue()


def write_outputs(summary: Dict[str, Any], out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pj, pm, pc = out_dir / "summary.json", out_dir / "summary.md", out_dir / "summary.csv"
    pj.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    pm.write_text(to_markdown(summary), encoding="utf-8")
    pc.write_text(to_csv_text(summary), encoding="utf-8")
    return {"json": pj, "md": pm, "csv": pc}


# --------------------------------------------------------------------------- #
# Конфигурация (без torch)
# --------------------------------------------------------------------------- #
def load_config_light(path: Optional[Path] = None) -> Tuple[Dict[str, Any], Optional[str]]:
    p = Path(path) if path else ROOT / "config.yaml"
    if not p.is_file():
        return {}, None
    try:
        import yaml
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return {}, None
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:12]
    return cfg, h


def pipeline_version_light() -> Optional[str]:
    p = ROOT / "src" / "inference.py"
    try:
        m = re.search(r'^__version__\s*=\s*"([^"]+)"', p.read_text(encoding="utf-8"), re.M)
        return m.group(1) if m else None
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def summarize_files(results: Sequence[Path], debug: Optional[Sequence[Path]] = None,
                    extras: Optional[Sequence[Path]] = None, device_tags: Optional[Sequence[Path]] = None,
                    dicom_root: Optional[Path] = None, cfg_path: Optional[Path] = None,
                    min_n: int = DEFAULT_MIN_N, top_n: int = DEFAULT_TOP_N, salt: str = "",
                    cfg: Optional[Dict[str, Any]] = None, config_hash: Optional[str] = None,
                    pipeline_version: Optional[str] = None) -> Dict[str, Any]:
    results = [Path(p) for p in results]
    rows = load_results(results)
    debug_paths = [Path(p) for p in debug] if debug else [sibling_csv(p, "_debug") for p in results]
    extras_paths = [Path(p) for p in extras] if extras else [sibling_csv(p, "_extras") for p in results]
    debug_rows = load_optional(debug_paths)
    extras_rows = load_optional(extras_paths)
    device_rows: List[Dict[str, Any]] = load_optional([Path(p) for p in device_tags]) if device_tags else []
    device_source = "device_tags.csv" if device_rows else None
    if not device_rows and dicom_root:
        device_rows = read_device_tags_dicom(rows, Path(dicom_root), salt)
        device_source = "dicom" if device_rows else None
    if cfg is None:
        cfg, config_hash = load_config_light(cfg_path)
    if pipeline_version is None:
        pipeline_version = pipeline_version_light()
    inputs = OrderedDict([
        ("results_csv", [p.name for p in results]),
        ("debug_csv", [p.name for p in debug_paths if p is not None and Path(p).is_file()]),
        ("extras_csv", [p.name for p in extras_paths if p is not None and Path(p).is_file()]),
        ("device_source", device_source),
    ])
    return build_summary(rows, debug_rows, extras_rows, device_rows, cfg=cfg, min_n=min_n, top_n=top_n,
                         salt=salt, inputs=inputs, pipeline_version=pipeline_version, config_hash=config_hash)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Сводка по отделению по results.csv (без персональных данных)")
    ap.add_argument("--results", nargs="+", required=True, help="один или несколько results.csv (9 колонок)")
    ap.add_argument("--debug", nargs="*", default=None, help="технические CSV того же прогона (по умолчанию <stem>_debug.csv рядом)")
    ap.add_argument("--extras", nargs="*", default=None, help="extras-CSV того же прогона (по умолчанию <stem>_extras.csv рядом)")
    ap.add_argument("--device-tags", nargs="*", default=None, help="CSV с тегами аппарата/даты (device_tags.csv из каталога задачи API)")
    ap.add_argument("--dicom-root", default=None, help="корень DICOM для чтения тегов аппарата и даты по path_to_study")
    ap.add_argument("--config", default=None, help="config.yaml (по умолчанию корень репозитория)")
    ap.add_argument("--out-dir", required=True, help="каталог для summary.json, summary.md, summary.csv")
    ap.add_argument("--top", type=int, default=DEFAULT_TOP_N, help="сколько исследований в списке для пересмотра")
    ap.add_argument("--min-n", type=int, default=DEFAULT_MIN_N, help="порог пометки «мало данных»")
    ap.add_argument("--salt", default="", help="соль для хэша StationName/DeviceSerialNumber (одна на отделение)")
    ap.add_argument("--write-device-tags", default=None, help="сохранить прочитанные из DICOM теги аппарата/даты в CSV")
    args = ap.parse_args(argv)

    results = [Path(p) for p in args.results]
    if args.write_device_tags and args.dicom_root:
        rows = load_results(results)
        write_device_tags_csv(read_device_tags_dicom(rows, Path(args.dicom_root), args.salt), Path(args.write_device_tags))
    summary = summarize_files(results, debug=args.debug, extras=args.extras, device_tags=args.device_tags,
                              dicom_root=Path(args.dicom_root) if args.dicom_root else None,
                              cfg_path=Path(args.config) if args.config else None,
                              min_n=args.min_n, top_n=args.top, salt=args.salt)
    paths = write_outputs(summary, Path(args.out_dir))
    t = summary["totals"]
    print(f"исследований {t['n_studies']}, файлов {t['n_files']}, с нарушением {t['n_violation']} "
          f"({fmt_rate(summary['violation_rate'])}), Failure {t['n_failure']}, "
          f"не уверен {t['n_uncertain'] if t['n_uncertain'] is not None else '—'}")
    for k, p in paths.items():
        print(f"{k}: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
