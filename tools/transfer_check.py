#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transfer_check.py — прогон-двойник: проверка инвариантности предсказаний к форме подачи данных.

Закрытый набор организаторов приходит без суффиксов `_ПОП/_ППОБ`, в другом порядке, в других
каталогах и, возможно, одним или несколькими архивами. Скрипт делает копию входа («двойник»)
с изменённой формой подачи, прогоняет инференс и сверяет результат с эталонным прогоном по ключу
(study_uid, image_uid): совпадать должны anatomical_region, quality_class, violation_type,
quality_prob, processing_status.

Режимы двойника (--modes, через запятую):
  rename   каталоги s000.., файлы f0000.dcm, порядок перемешан (структура каталогов сохранена);
  zip      тот же переименованный набор одним zip-архивом;
  shuffle  случайный порядок файлов и каталогов, случайная вложенность подкаталогов (0–3 уровня),
           случайный регистр расширений (.dcm/.DCM/.Dcm/без расширения), имена — смесь кириллицы
           и латиницы с пробелами и скобками; файлы одного исходного каталога остаются вместе;
  mixed    shuffle + несколько zip-архивов в одном входном каталоге (часть каталогов — в архивах
           с кириллическими именами, часть — россыпью, один архив вложен в подкаталог);
  scatter  ДИАГНОСТИЧЕСКИЙ режим: каждый файл в собственном каталоге. Ломает группировку по
           папкам и потому по замыслу меняет резервный study_uid «hash-…» у файлов без
           StudyInstanceUID (это ожидаемое расхождение, режим не входит в набор по умолчанию).

Дополнительно:
  --bitwise   побитовая сверка нормализованного CSV: удаляется колонка time_of_processing,
              строки сортируются по (study_uid, image_uid), CSV сериализуется заново и сравнивается
              sha256. Считаются два хэша: с колонкой path_to_study (она меняется у двойника по
              построению — переименование и есть цель проверки) и без неё; вердикт «бит в бит»
              выносится по хэшу без path_to_study, оба хэша попадают в JSON.
  --sr        прогон с `--sr-study` и сверка DICOM SR по исследованиям: содержимое сравнивается
              после удаления тегов даты/времени создания (InstanceCreationDate/Time,
              ContentDate/Time — единственные теги, зависящие от момента запуска: SOP/Series UID
              SR детерминированы от study_uid, версии, config_hash и дайджеста строк) и текстовых
              элементов FILE (path_to_study — меняется по построению). Всё остальное, включая
              UID документа, число снимков, вердикты, quality_prob и sha256 исходников, должно
              совпасть побайтно.

  python tools/transfer_check.py --input tests/phantoms --out out/transfer_check.json \
      --modes rename,zip,shuffle,mixed --bitwise --sr [--baseline out/run1/results.csv] \
      [--workdir /tmp/tc] [--keep] [--seed 20260922]

Код возврата 0 — инвариантность подтверждена, 1 — есть расхождения.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
CMP_COLS = ["anatomical_region", "quality_class", "violation_type", "processing_status"]
PROB_COL = "quality_prob"
TIME_COL = "time_of_processing"
PATH_COL = "path_to_study"
SEED = 20260922
DEFAULT_MODES = "rename,zip,shuffle,mixed"

# Теги DICOM SR, зависящие от момента запуска (см. src/dicom_sr.py: build_study_sr(now=...)).
SR_TIME_TAGS = ("InstanceCreationDate", "InstanceCreationTime", "ContentDate", "ContentTime")
# Кодовое значение текстового элемента с path_to_study внутри SR (меняется у двойника по построению).
SR_PATH_CODE = "FILE"

CYR = ["Исследование", "снимок", "пациент", "ПОП", "ППОБ", "ЛПОБ", "серия", "архив", "выгрузка", "DXA"]
LAT = ["study", "img", "CR", "series", "DXA", "export", "batch", "IM", "scan", "case"]
EXT_CHOICES = [".dcm", ".DCM", ".Dcm", "", ".dcm", ""]


# --------------------------------------------------------------------------- #
# CSV: чтение и нормализация
# --------------------------------------------------------------------------- #
def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=","))


def normalized_csv_bytes(rows: list[dict], drop: tuple[str, ...]) -> bytes:
    """CSV без колонок `drop`, строки отсортированы по (study_uid, image_uid, path_to_study);
    сериализация фиксирована (QUOTE_MINIMAL, '\\n'), чтобы sha256 сравнивались побитово."""
    if not rows:
        return b""
    cols = [c for c in rows[0].keys() if c not in drop]
    ordered = sorted(rows, key=lambda r: (r.get("study_uid", ""), r.get("image_uid", ""),
                                          r.get(PATH_COL, "") if PATH_COL not in drop else ""))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL,
                       lineterminator="\n")
    w.writeheader()
    for r in ordered:
        w.writerow({c: r.get(c, "") for c in cols})
    return buf.getvalue().encode("utf-8")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# --------------------------------------------------------------------------- #
# Построение двойников
# --------------------------------------------------------------------------- #
def _rand_name(rng: random.Random, kind: str) -> str:
    """Имя-смесь кириллицы и латиницы: «Исследование 07 (img)», «CR_снимок-3» и т. п."""
    a = rng.choice(CYR if rng.random() < 0.6 else LAT)
    b = rng.choice(LAT if a in CYR else CYR)
    n = rng.randint(0, 99)
    pattern = rng.choice(["{a} {n:02d} ({b})", "{a}_{b}-{n}", "{b}{n:03d} {a}", "{a}-{n}", "{b}_{n:02d}_{a}"])
    return pattern.format(a=a, b=b, n=n)


def _unique(rng: random.Random, taken: set[str], make) -> str:
    for _ in range(1000):
        cand = make()
        if cand not in taken:
            taken.add(cand)
            return cand
    cand = f"x{rng.randrange(10**9)}"
    taken.add(cand)
    return cand


def _has_dicm_magic(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:  # noqa: BLE001
        return False


def make_renamed_copy(src: Path, dst: Path, seed: int = SEED) -> dict:
    """Копия входа: каталоги s000.., файлы f0000 с исходным расширением, порядок перемешан."""
    rng = random.Random(seed)
    files = sorted(p for p in src.rglob("*") if p.is_file())
    order = list(range(len(files)))
    rng.shuffle(order)
    dst.mkdir(parents=True, exist_ok=True)
    dirs = sorted({p.parent.relative_to(src).as_posix() for p in files})
    dir_map = {d: f"s{i:03d}" for i, d in enumerate(sorted(dirs, key=lambda x: rng.random()))}
    mapping = {}
    for new_i, old_i in enumerate(order):
        p = files[old_i]
        rel_dir = p.parent.relative_to(src).as_posix()
        sub = dir_map[rel_dir]
        out_dir = dst / sub if sub != "." else dst
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = p.suffix if p.suffix.lower() not in (".dcm",) else ".dcm"
        new_name = f"f{new_i:04d}{suffix}"
        shutil.copy2(p, out_dir / new_name)
        mapping[(out_dir / new_name).relative_to(dst).as_posix()] = p.relative_to(src).as_posix()
    return mapping


def make_shuffled_copy(src: Path, dst: Path, seed: int = SEED, scatter: bool = False) -> dict:
    """Копия входа со случайным порядком, случайной вложенностью каталогов (0–3 уровня),
    случайным регистром расширений и именами-смесью кириллицы и латиницы.
    Файлы одного исходного каталога остаются в одном целевом каталоге (группировка по папке
    исследования сохраняется — от неё зависит резервный study_uid «hash-…» для файлов без
    StudyInstanceUID). При scatter=True каждый файл кладётся в собственный каталог."""
    rng = random.Random(seed + 1)
    files = sorted(p for p in src.rglob("*") if p.is_file())
    order = list(range(len(files)))
    rng.shuffle(order)
    dst.mkdir(parents=True, exist_ok=True)
    taken_dirs: set[str] = set()
    dir_map: dict[str, Path] = {}

    def new_dir() -> Path:
        depth = rng.choice([0, 1, 1, 2, 2, 3])
        parts = [_unique(rng, taken_dirs, lambda: _rand_name(rng, "dir")) for _ in range(depth)]
        return dst.joinpath(*parts) if parts else dst

    mapping = {}
    taken_files: set[str] = set()
    for new_i, old_i in enumerate(order):
        p = files[old_i]
        rel_dir = p.parent.relative_to(src).as_posix()
        if scatter:
            out_dir = new_dir() / _unique(rng, taken_dirs, lambda: _rand_name(rng, "dir"))
        else:
            if rel_dir not in dir_map:
                dir_map[rel_dir] = new_dir()
            out_dir = dir_map[rel_dir]
        out_dir.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() == ".dcm":
            ext = rng.choice(EXT_CHOICES)
            if ext == "" and not _has_dicm_magic(p):
                # без расширения инференс опознаёт DICOM только по магическому числу DICM;
                # файл без преамбулы (или не-DICOM с расширением .dcm) без расширения исчез бы
                # из выгрузки по замыслу — такое сравнение было бы не про инвариантность
                ext = rng.choice([".dcm", ".DCM", ".Dcm"])
        else:
            ext = p.suffix                     # не-DICOM расширения оставляем как есть
        name = _unique(rng, taken_files, lambda: str(out_dir.relative_to(dst)) + "/" + _rand_name(rng, "file") + ext)
        target = dst / name
        shutil.copy2(p, target)
        mapping[target.relative_to(dst).as_posix()] = p.relative_to(src).as_posix()
    return mapping


def zip_dir(src: Path, zpath: Path, rng: random.Random | None = None) -> None:
    """Архив каталога; порядок элементов внутри архива случайный, если передан rng."""
    members = [p for p in src.rglob("*") if p.is_file()]
    members.sort()
    if rng is not None:
        rng.shuffle(members)
    zpath.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in members:
            zf.write(p, p.relative_to(src).as_posix())


def make_mixed_copy(shuffled: Path, dst: Path, seed: int = SEED) -> dict:
    """Из shuffle-копии собирается смешанный вход: каталоги верхнего уровня делятся на группы,
    часть групп упаковывается в отдельные zip с кириллическими именами (один — во вложенном
    подкаталоге), остальные копируются россыпью. Возвращает описание раскладки."""
    rng = random.Random(seed + 2)
    dst.mkdir(parents=True, exist_ok=True)
    top = sorted(shuffled.iterdir(), key=lambda p: p.name)
    rng.shuffle(top)
    n = len(top)
    # не менее двух архивов, если есть из чего собрать
    n_zip_groups = 2 if n >= 3 else (1 if n >= 1 else 0)
    if n >= 6:
        n_zip_groups = 3
    layout: dict = {"zips": {}, "loose": []}
    zip_names = ["Выгрузка ДЗМ (часть 1).zip", "архив_DXA_2.zip", "Исследования part3.zip"]
    idx = 0
    for g in range(n_zip_groups):
        size = max(1, (n - 1) // (n_zip_groups + 1))
        group = top[idx: idx + size]
        idx += size
        if not group:
            continue
        stage = dst.parent / f"_stage_{g}"
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        for item in group:
            if item.is_dir():
                shutil.copytree(item, stage / item.name)
            else:
                shutil.copy2(item, stage / item.name)
        zrel = Path(zip_names[g]) if g != 1 else Path("вложенный каталог") / "inner" / zip_names[g]
        zip_dir(stage, dst / zrel, rng)
        shutil.rmtree(stage)
        layout["zips"][zrel.as_posix()] = [it.name for it in group]
    for item in top[idx:]:
        if item.is_dir():
            shutil.copytree(item, dst / item.name)
        else:
            shutil.copy2(item, dst / item.name)
        layout["loose"].append(item.name)
    return layout


# --------------------------------------------------------------------------- #
# Инференс
# --------------------------------------------------------------------------- #
def run_inference(inp: Path, out_csv: Path, python: str, sr: bool) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "2")
    env["PYTHONHASHSEED"] = "0"
    env["DENSITO_ROOT"] = str(ROOT)
    cmd = [python, str(ROOT / "src" / "inference.py"), "--input", str(inp), "--output", str(out_csv)]
    if sr:
        cmd += ["--sr-study-dir", str(out_csv.parent / "sr")]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write((p.stdout or "")[-2000:] + (p.stderr or "")[-2000:])
    return p.returncode


# --------------------------------------------------------------------------- #
# Сверка CSV
# --------------------------------------------------------------------------- #
def compare(base: list[dict], var: list[dict], tol: float) -> dict:
    """Сверка по (study_uid, image_uid); строки со сбоем (пустые UID) сверяются по количеству."""
    def key(r):
        return (r.get("study_uid", ""), r.get("image_uid", ""))

    b_ok = {key(r): r for r in base if key(r) != ("", "")}
    v_ok = {key(r): r for r in var if key(r) != ("", "")}
    b_fail = [r for r in base if r.get("processing_status") != "Success"]
    v_fail = [r for r in var if r.get("processing_status") != "Success"]

    problems: list[str] = []
    if len(base) != len(var):
        problems.append(f"строк {len(base)} против {len(var)}")
    if len(b_ok) != len(base):
        problems.append(f"в эталоне {len(base) - len(b_ok)} строк с повторяющимся или пустым ключом")
    if len(v_ok) != len(var):
        problems.append(f"у двойника {len(var) - len(v_ok)} строк с повторяющимся или пустым ключом")
    missing = sorted(set(b_ok) - set(v_ok))
    extra = sorted(set(v_ok) - set(b_ok))
    for k in missing[:5]:
        problems.append(f"нет строки с UID {k[1][:24]}… (study {k[0][:24]}…)")
    for k in extra[:5]:
        problems.append(f"лишняя строка с UID {k[1][:24]}… (study {k[0][:24]}…)")
    if len(b_fail) != len(v_fail):
        problems.append(f"строк со сбоем {len(b_fail)} против {len(v_fail)}")

    diff_cols: dict[str, int] = {}
    max_dprob = 0.0
    n_cmp = 0
    n_match = 0
    for k in sorted(set(b_ok) & set(v_ok)):
        rb, rv = b_ok[k], v_ok[k]
        n_cmp += 1
        row_ok = True
        for c in CMP_COLS:
            if (rb.get(c) or "") != (rv.get(c) or ""):
                row_ok = False
                diff_cols[c] = diff_cols.get(c, 0) + 1
                if len(problems) < 12:
                    problems.append(f"{c}: «{rb.get(c)}» против «{rv.get(c)}» (UID {k[1][-12:]})")
        try:
            d = abs(float(rb.get(PROB_COL, "0") or 0) - float(rv.get(PROB_COL, "0") or 0))
            max_dprob = max(max_dprob, d)
            if d > tol or (tol == 0 and (rb.get(PROB_COL) or "") != (rv.get(PROB_COL) or "")):
                row_ok = False
                diff_cols[PROB_COL] = diff_cols.get(PROB_COL, 0) + 1
                if len(problems) < 12:
                    problems.append(f"quality_prob «{rb.get(PROB_COL)}» против «{rv.get(PROB_COL)}» (UID {k[1][-12:]})")
        except ValueError:
            row_ok = False
            problems.append(f"quality_prob не число (UID {k[1][-12:]})")
        n_match += int(row_ok)
    # порядок строк — информационно: у двойника он по построению другой (порядок обхода файлов)
    same_order = [key(r) for r in base] == [key(r) for r in var]
    return {"n_rows_baseline": len(base), "n_rows_twin": len(var), "n_compared": n_cmp,
            "n_matched": n_match, "n_missing": len(missing), "n_extra": len(extra),
            "max_abs_dprob": round(max_dprob, 6), "diff_by_column": diff_cols,
            "row_order_identical": same_order,
            "ok": not problems, "problems": problems}


def bitwise_block(base: list[dict], var: list[dict]) -> dict:
    """sha256 нормализованных CSV (без time_of_processing; и дополнительно без path_to_study)."""
    b1, v1 = normalized_csv_bytes(base, (TIME_COL,)), normalized_csv_bytes(var, (TIME_COL,))
    b2, v2 = normalized_csv_bytes(base, (TIME_COL, PATH_COL)), normalized_csv_bytes(var, (TIME_COL, PATH_COL))
    return {
        "sha256_baseline_no_time": sha256_bytes(b1),
        "sha256_twin_no_time": sha256_bytes(v1),
        "identical_with_path": b1 == v1,
        "sha256_baseline_no_time_no_path": sha256_bytes(b2),
        "sha256_twin_no_time_no_path": sha256_bytes(v2),
        "identical_without_path": b2 == v2,
        "ok": b2 == v2,
    }


# --------------------------------------------------------------------------- #
# Сверка DICOM SR
# --------------------------------------------------------------------------- #
def _strip_sr(ds) -> tuple[list[str], list[str]]:
    """Удаляет из SR теги времени и элементы FILE. Возвращает (удалённые теги, удалённые пути)."""
    removed_tags, removed_paths = [], []
    for name in SR_TIME_TAGS:
        if name in ds:
            removed_tags.append(f"{name}={ds.get(name)}")
            delattr(ds, name)

    def walk(seq):
        keep = []
        for item in seq:
            code = ""
            try:
                code = str(item.ConceptNameCodeSequence[0].CodeValue)
            except Exception:  # noqa: BLE001
                pass
            if code == SR_PATH_CODE and getattr(item, "ValueType", "") == "TEXT":
                removed_paths.append(str(getattr(item, "TextValue", "")))
                continue
            if "ContentSequence" in item:
                item.ContentSequence = type(item.ContentSequence)(walk(item.ContentSequence))
            keep.append(item)
        return keep

    if "ContentSequence" in ds:
        ds.ContentSequence = type(ds.ContentSequence)(walk(ds.ContentSequence))
    return removed_tags, removed_paths


def _canon_evidence(ds) -> None:
    """Сортирует ссылки в CurrentRequestedProcedureEvidenceSequence по UID: в инференсе их порядок
    повторяет порядок обхода файлов (см. dicom_sr.build_study_sr, by_series), поэтому у двойника
    он другой при том же множестве ссылок."""
    if "CurrentRequestedProcedureEvidenceSequence" not in ds:
        return
    for ev in ds.CurrentRequestedProcedureEvidenceSequence:
        if "ReferencedSeriesSequence" not in ev:
            continue
        series = list(ev.ReferencedSeriesSequence)
        for rs in series:
            if "ReferencedSOPSequence" in rs:
                refs = sorted(rs.ReferencedSOPSequence, key=lambda r: str(getattr(r, "ReferencedSOPInstanceUID", "")))
                rs.ReferencedSOPSequence = type(rs.ReferencedSOPSequence)(refs)
        series.sort(key=lambda rs: str(getattr(rs, "SeriesInstanceUID", "")))
        ev.ReferencedSeriesSequence = type(ev.ReferencedSeriesSequence)(series)


def sr_digest(path: Path) -> dict:
    import pydicom
    ds = pydicom.dcmread(str(path), force=True)
    removed_tags, removed_paths = _strip_sr(ds)
    buf = io.BytesIO()
    ds.save_as(buf, write_like_original=False)
    js = json.dumps(ds.to_json_dict(), sort_keys=True, ensure_ascii=False)
    _canon_evidence(ds)
    buf2 = io.BytesIO()
    ds.save_as(buf2, write_like_original=False)
    return {"sha256_stripped_bytes": sha256_bytes(buf.getvalue()),
            "sha256_stripped_json": sha256_bytes(js.encode("utf-8")),
            "sha256_stripped_evidence_sorted": sha256_bytes(buf2.getvalue()),
            "sop_instance_uid": str(getattr(ds, "SOPInstanceUID", "")),
            "series_uid": str(getattr(ds, "SeriesInstanceUID", "")),
            "removed_time_tags": removed_tags, "removed_file_items": len(removed_paths),
            "json": js}


def compare_sr(base_dir: Path, var_dir: Path, lenient: bool = False) -> dict:
    """Сверка SR по исследованиям: имена файлов (<study_uid>_SR.dcm), SOP UID, содержимое без
    дат/времени и без элементов FILE. При lenient=True дополнительно допускается другой порядок
    ссылок в CurrentRequestedProcedureEvidenceSequence (он повторяет порядок обхода файлов)."""
    if not base_dir.exists() and not var_dir.exists():
        return {"ok": True, "skipped": "SR не записаны ни в эталоне, ни у двойника", "n_sr_baseline": 0, "n_sr_twin": 0}
    try:
        import pydicom  # noqa: F401
    except Exception:  # noqa: BLE001
        return {"ok": True, "skipped": "pydicom недоступен", "n_sr_baseline": 0, "n_sr_twin": 0}
    b = {p.name: p for p in base_dir.glob("*.dcm")} if base_dir.exists() else {}
    v = {p.name: p for p in var_dir.glob("*.dcm")} if var_dir.exists() else {}
    problems: list[str] = []
    for name in sorted(set(b) - set(v)):
        problems.append(f"SR отсутствует у двойника: {name}")
    for name in sorted(set(v) - set(b)):
        problems.append(f"лишний SR у двойника: {name}")
    n_match = 0
    n_order_only = 0
    per_file = {}
    removed_tags_all: set[str] = set()
    for name in sorted(set(b) & set(v)):
        try:
            db, dv = sr_digest(b[name]), sr_digest(v[name])
        except Exception as e:  # noqa: BLE001
            problems.append(f"{name}: не удалось прочитать SR ({e})")
            continue
        removed_tags_all.update(t.split("=")[0] for t in db["removed_time_tags"])
        same_bytes = db["sha256_stripped_bytes"] == dv["sha256_stripped_bytes"]
        same_json = db["sha256_stripped_json"] == dv["sha256_stripped_json"]
        same_uid = db["sop_instance_uid"] == dv["sop_instance_uid"]
        same_sorted = db["sha256_stripped_evidence_sorted"] == dv["sha256_stripped_evidence_sorted"]
        per_file[name] = {"identical_stripped_bytes": same_bytes, "identical_stripped_json": same_json,
                          "identical_after_evidence_sort": same_sorted,
                          "same_sop_instance_uid": same_uid,
                          "sha256_baseline": db["sha256_stripped_bytes"], "sha256_twin": dv["sha256_stripped_bytes"],
                          "file_items_removed": [db["removed_file_items"], dv["removed_file_items"]]}
        if same_uid and (same_bytes or (lenient and same_sorted)):
            n_match += 1
            if not same_bytes:
                n_order_only += 1
        else:
            if same_sorted and not same_bytes:
                problems.append(f"{name}: SR совпадает только после сортировки ссылок Evidence (порядок обхода файлов)")
                continue
            if not same_uid:
                problems.append(f"{name}: SOPInstanceUID SR различается")
            if not same_json:
                # первое отличие в JSON-представлении — для отчёта
                jb, jv = db["json"], dv["json"]
                i = next((k for k in range(min(len(jb), len(jv))) if jb[k] != jv[k]), min(len(jb), len(jv)))
                problems.append(f"{name}: содержимое SR различается около «{jb[max(0, i-60):i+60]}»")
            elif not same_bytes:
                problems.append(f"{name}: JSON совпал, байты после удаления времени — нет (порядок/кодировка)")
    return {"ok": not problems, "n_sr_baseline": len(b), "n_sr_twin": len(v), "n_compared": len(set(b) & set(v)),
            "n_matched": n_match, "n_matched_only_after_evidence_sort": n_order_only, "lenient": lenient,
            "time_tags_removed": sorted(removed_tags_all),
            "path_items_removed_code": SR_PATH_CODE, "per_file": per_file, "problems": problems}


# --------------------------------------------------------------------------- #
def summarize_mode(cmp: dict, bit: dict | None, sr: dict | None) -> dict:
    out = dict(cmp)
    out["discrepancies"] = list(cmp.get("problems", []))
    if bit is not None:
        out["bitwise"] = bit
        if not bit["ok"]:
            out["discrepancies"].append("нормализованный CSV (без time_of_processing и path_to_study) не совпал побитово")
    if sr is not None:
        out["sr"] = sr
        out["discrepancies"] += [f"SR: {p}" for p in sr.get("problems", [])]
    out["ok"] = bool(cmp.get("ok")) and (bit is None or bit["ok"]) and (sr is None or sr.get("ok", True))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="каталог с исходными файлами")
    ap.add_argument("--baseline", default="", help="CSV эталонного прогона (если нет — прогоним сами)")
    ap.add_argument("--out", required=True, help="куда писать JSON результата")
    ap.add_argument("--modes", default=DEFAULT_MODES,
                    help=f"режимы через запятую: rename, zip, shuffle, mixed, scatter (по умолчанию {DEFAULT_MODES})")
    ap.add_argument("--workdir", default="", help="рабочий каталог (по умолчанию временный)")
    ap.add_argument("--tol", type=float, default=0.0, help="допуск по quality_prob (по умолчанию 0 — бит в бит)")
    ap.add_argument("--bitwise", action="store_true", help="сверять нормализованный CSV побитово (sha256)")
    ap.add_argument("--sr", action="store_true", help="писать SR на исследование (--sr-study-dir) и сверять их")
    ap.add_argument("--sr-lenient", action="store_true",
                    help="в сверке SR допускать другой порядок ссылок Evidence (порядок обхода файлов)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--keep", action="store_true", help="не удалять рабочий каталог")
    a = ap.parse_args()

    src = Path(a.input).resolve()
    work = Path(a.workdir).resolve() if a.workdir else Path(tempfile.mkdtemp(prefix="densito_tc_"))
    work.mkdir(parents=True, exist_ok=True)
    result: dict = {"input": str(src), "seed": a.seed, "tol": a.tol, "bitwise": a.bitwise, "sr": a.sr,
                    "n_input_files": sum(1 for p in src.rglob("*") if p.is_file()), "modes": {}}

    base_csv = Path(a.baseline).resolve() if a.baseline else work / "baseline" / "results.csv"
    if not a.baseline:
        if run_inference(src, base_csv, a.python, a.sr) != 0 and not base_csv.exists():
            result["error"] = "эталонный прогон не создал CSV"
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return 1
    base = read_rows(base_csv)
    result["baseline_csv"] = str(base_csv)
    result["baseline_sha256_no_time"] = sha256_bytes(normalized_csv_bytes(base, (TIME_COL,)))
    result["baseline_sha256_no_time_no_path"] = sha256_bytes(normalized_csv_bytes(base, (TIME_COL, PATH_COL)))

    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    copies: dict[str, Path] = {}
    mappings: dict[str, dict] = {}

    def prepared(kind: str) -> Path:
        if kind in copies:
            return copies[kind]
        d = work / kind
        if d.exists():
            shutil.rmtree(d)
        if kind == "renamed":
            mappings[kind] = make_renamed_copy(src, d, a.seed)
        elif kind == "shuffled":
            mappings[kind] = make_shuffled_copy(src, d, a.seed)
        elif kind == "scattered":
            mappings[kind] = make_shuffled_copy(src, d, a.seed, scatter=True)
        elif kind == "mixed":
            mappings[kind] = make_mixed_copy(prepared("shuffled"), d, a.seed)
        copies[kind] = d
        return d

    for mode in modes:
        if mode == "rename":
            inp = prepared("renamed")
        elif mode == "zip":
            inp = work / "renamed_bundle.zip"
            zip_dir(prepared("renamed"), inp)
        elif mode == "shuffle":
            inp = prepared("shuffled")
        elif mode == "mixed":
            inp = prepared("mixed")
        elif mode == "scatter":
            inp = prepared("scattered")
        else:
            result["modes"][mode] = {"ok": False, "problems": [f"неизвестный режим {mode}"], "discrepancies": [f"неизвестный режим {mode}"]}
            continue
        out_csv = work / f"var_{mode}" / "results.csv"
        rc = run_inference(inp, out_csv, a.python, a.sr)
        if not out_csv.exists():
            result["modes"][mode] = {"ok": False, "problems": [f"прогон не создал CSV (код {rc})"],
                                     "discrepancies": [f"прогон не создал CSV (код {rc})"]}
            continue
        var = read_rows(out_csv)
        cmp = compare(base, var, a.tol)
        bit = bitwise_block(base, var) if a.bitwise else None
        sr = compare_sr(base_csv.parent / "sr", out_csv.parent / "sr", a.sr_lenient) if a.sr else None
        block = summarize_mode(cmp, bit, sr)
        block["input"] = str(inp)
        block["twin_csv"] = str(out_csv)
        result["modes"][mode] = block

    (work / "mapping.json").write_text(json.dumps(mappings, ensure_ascii=False, indent=1), encoding="utf-8")
    result["mapping_json"] = str(work / "mapping.json")
    result["ok"] = all(v.get("ok") for v in result["modes"].values()) and bool(result["modes"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    short = {m: {"ok": d.get("ok"), "rows": [d.get("n_rows_baseline"), d.get("n_rows_twin")],
                 "matched": d.get("n_matched"), "diff": d.get("diff_by_column"),
                 "bitwise": (d.get("bitwise") or {}).get("ok"), "sr": (d.get("sr") or {}).get("ok"),
                 "discrepancies": d.get("discrepancies", [])[:6]}
             for m, d in result["modes"].items()}
    print(json.dumps({"ok": result["ok"], "input": result["input"], "n_input_files": result["n_input_files"],
                      "modes": short}, ensure_ascii=False, indent=2))
    if not a.keep and not a.workdir:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
