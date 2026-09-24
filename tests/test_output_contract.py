#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Контракт выдачи (A1, 24.09.2026). Модель и пороги не трогаются, проверяется только обвязка.

  1. Инвариант строки для ВСЕХ строк, включая Failure: quality_class 1 <=> quality_prob >= 0.5;
     у Failure quality_prob = output.fallback_quality_prob строго < 0.5 (CSV, XLSX, failure_quality_prob).
  2. Вход: повреждённый верхнеуровневый zip -> одна строка Failure, код 0; пустая папка и пустой zip ->
     CSV только с заголовком, сообщение и код 2; одиночный файл ведёт себя как в папке
     (не-DICOM .txt -> строк нет, код 2; мусор с .dcm -> строка Failure).
  3. Конфигурация fail-closed: нет файла / битый YAML / пустой файл -> ConfigError и код 2;
     встроенные значения — только при DENSITO_ALLOW_DEFAULT_CONFIG=1.
  4. Фильтр области в пакетном пути: чужой аппарат, модальность CT и боковая проекция -> Failure
     с причиной в debug CSV; свой аппарат (фантом) -> Success.

Запуск: python tests/test_output_contract.py   (код возврата 0 — ок).
"""
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pydicom

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import inference as inf  # noqa: E402

PHANTOMS = ROOT / "tests" / "phantoms"
fails = []


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def run(inp: Path, out: Path, *extra, env=None):
    cmd = [sys.executable, str(SRC / "inference.py"), "--input", str(inp), "--output", str(out), *extra]
    e = dict(os.environ)
    e.pop(inf.ALLOW_DEFAULT_CONFIG_ENV, None)
    e.update(env or {})
    return subprocess.run(cmd, capture_output=True, text=True, env=e, cwd=str(ROOT))


def rows_of(p: Path):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def header_of(p: Path):
    with open(p, encoding="utf-8", newline="") as f:
        return next(csv.reader(f), [])


def invariant_ok(rows):
    bad = []
    for r in rows:
        c, p = int(r["quality_class"]), float(r["quality_prob"])
        if (c == 1) != (p >= 0.5):
            bad.append((r["path_to_study"], c, p, r["processing_status"]))
    return bad


def phantom_files():
    return sorted(p for p in PHANTOMS.rglob("*.dcm") if "broken" not in p.parts)


cfg = inf.load_config()
cols = cfg["output"]["columns"]
tmp = Path(tempfile.mkdtemp(prefix="densito_contract_"))
try:
    # ---------------- 1. инвариант и Failure < 0.5
    fb = inf.failure_quality_prob(cfg["output"])
    check(0.0 <= fb < 0.5, f"failure_quality_prob = {fb} < 0.5")
    check(inf.failure_quality_prob({"fallback_quality_prob": 0.5}) < 0.5, "fallback 0.5 в конфиге -> всё равно < 0.5")
    check(inf.failure_quality_prob({"fallback_quality_prob": "abc"}) < 0.5, "нечисловой fallback -> < 0.5")

    mixed = tmp / "mixed"
    mixed.mkdir()
    goods = phantom_files()[:4]
    for i, p in enumerate(goods):
        shutil.copy(p, mixed / f"good_{i}.dcm")
    (mixed / "garbage.dcm").write_bytes(b"not a dicom at all")
    (mixed / "empty.dcm").write_bytes(b"")
    out = tmp / "o_mixed" / "results.csv"
    r = run(mixed, out, "--xlsx")
    check(r.returncode == 0, f"смешанная папка: код {r.returncode}")
    rows = rows_of(out)
    check(len(rows) == len(goods) + 2, f"строк {len(rows)} = {len(goods) + 2}")
    check(invariant_ok(rows) == [], f"инвариант class 1 <=> prob >= 0.5 для всех строк: {invariant_ok(rows)[:2]}")
    failures = [x for x in rows if x["processing_status"] == cfg["output"]["status_failure"]]
    check(len(failures) == 2 and all(float(x["quality_prob"]) < 0.5 for x in failures),
          f"Failure-строки ({len(failures)}): quality_prob < 0.5 -> {[x['quality_prob'] for x in failures]}")
    check(all(x["quality_prob"] == repr(fb) for x in failures), f"CSV печатает {repr(fb)} без округления до 0.5")
    check(inf.validate_output_csv(out, cfg) == [], "validate_output_csv: OK")
    try:
        import openpyxl
        wb = openpyxl.load_workbook(out.with_suffix(".xlsx"), read_only=True)
        vals = []
        for ws in wb.worksheets:
            hdr = None
            for row in ws.iter_rows(values_only=True):
                if hdr is None:
                    hdr = [str(h or "") for h in row]
                    continue
                for h, v in zip(hdr, row):
                    if "quality_prob" in h or "prob" in h.lower() or "вероятн" in h.lower():
                        if isinstance(v, (int, float)):
                            vals.append(float(v))
        check(all(v != 0.5 for v in vals) and any(abs(v - fb) < 1e-9 for v in vals),
              f"XLSX: значения Failure не округлены до 0.5 ({len(vals)} чисел)")
    except ImportError:
        print("SKIP openpyxl не установлен")

    # ---------------- 2. вход
    bad_zip = tmp / "Выгрузка битая.zip"
    bad_zip.write_bytes(b"PK\x03\x04" + os.urandom(200))
    out = tmp / "o_badzip" / "results.csv"
    r = run(bad_zip, out)
    rows = rows_of(out) if out.exists() else []
    check(r.returncode == 0 and len(rows) == 1 and rows[0]["processing_status"] == "Failure",
          f"битый верхнеуровневый zip: код {r.returncode}, строк {len(rows)}, "
          f"статус {rows[0]['processing_status'] if rows else '-'}")
    check(bool(rows) and float(rows[0]["quality_prob"]) < 0.5 and rows[0]["path_to_study"].endswith(".zip"),
          "строка битого zip: prob < 0.5, path_to_study — имя архива")

    empty_dir = tmp / "empty_dir"
    empty_dir.mkdir()
    out = tmp / "o_empty" / "results.csv"
    r = run(empty_dir, out)
    check(r.returncode == 2, f"пустая папка: код {r.returncode} (ожидается 2)")
    check(out.exists() and header_of(out) == cols and rows_of(out) == [], "пустая папка: CSV только с заголовком")
    check("DICOM" in (r.stderr + r.stdout), "пустая папка: понятное сообщение")

    empty_zip = tmp / "empty.zip"
    with zipfile.ZipFile(empty_zip, "w"):
        pass
    out = tmp / "o_emptyzip" / "results.csv"
    r = run(empty_zip, out)
    check(r.returncode == 2 and out.exists() and header_of(out) == cols and rows_of(out) == [],
          f"пустой zip: код {r.returncode}, CSV только с заголовком")

    txt = tmp / "readme.txt"
    txt.write_text("ignore me", encoding="utf-8")
    out = tmp / "o_txt" / "results.csv"
    r = run(txt, out)
    check(r.returncode == 2 and (not out.exists() or rows_of(out) == []),
          f"одиночный .txt: как в папке (строк нет), код {r.returncode}")
    garbage = tmp / "single_garbage.dcm"
    garbage.write_bytes(b"not a dicom at all")
    out = tmp / "o_single" / "results.csv"
    r = run(garbage, out)
    rows = rows_of(out) if out.exists() else []
    check(r.returncode == 0 and len(rows) == 1 and rows[0]["processing_status"] == "Failure",
          f"одиночный мусор .dcm: строка Failure, код {r.returncode}")

    # ---------------- 3. конфигурация fail-closed
    missing = tmp / "nope" / "config.yaml"
    try:
        inf.load_config(missing)
        check(False, "нет config -> ConfigError")
    except inf.ConfigError as e:
        check("не найден" in str(e), f"нет config -> ConfigError ({str(e)[:60]})")
    os.environ[inf.ALLOW_DEFAULT_CONFIG_ENV] = "1"
    try:
        c = inf.load_config(missing)
        check(c["output"]["columns"] == cols, "DENSITO_ALLOW_DEFAULT_CONFIG=1 -> встроенные значения")
    finally:
        os.environ.pop(inf.ALLOW_DEFAULT_CONFIG_ENV, None)
    broken_yaml = tmp / "broken.yaml"
    broken_yaml.write_text("output: [unclosed\n  : :", encoding="utf-8")
    for name, pth in (("битый YAML", broken_yaml),):
        try:
            inf.load_config(pth)
            check(False, f"{name} -> ConfigError")
        except inf.ConfigError:
            check(True, f"{name} -> ConfigError")
    empty_yaml = tmp / "empty.yaml"
    empty_yaml.write_text("", encoding="utf-8")
    try:
        inf.load_config(empty_yaml)
        check(False, "пустой config -> ConfigError")
    except inf.ConfigError:
        check(True, "пустой config -> ConfigError")
    list_yaml = tmp / "list.yaml"
    list_yaml.write_text("- a\n- b\n", encoding="utf-8")
    try:
        inf.load_config(list_yaml)
        check(False, "config-список -> ConfigError")
    except inf.ConfigError:
        check(True, "config-список -> ConfigError")
    out = tmp / "o_cfg" / "results.csv"
    r = run(mixed, out, "--config", str(broken_yaml))
    check(r.returncode == 2 and "конфигурац" in r.stderr, f"CLI с битым config: код {r.returncode}, сообщение в stderr")
    r = run(mixed, out, env={"DENSITO_CONFIG": str(missing)})
    check(r.returncode == 2, f"CLI с DENSITO_CONFIG на несуществующий файл: код {r.returncode}")

    # ---------------- 4. фильтр области в пакетном пути
    src = goods[0]
    filt = tmp / "filter"
    filt.mkdir()
    cases = {}
    for name, mut in (("foreign_vendor.dcm", {"Manufacturer": "Hologic", "ManufacturerModelName": "Horizon A"}),
                      ("modality_ct.dcm", {"Modality": "CT"}),
                      ("lateral.dcm", {"SeriesDescription": "LATERAL SPINE"}),
                      ("bilateral.dcm", {"SeriesDescription": "DualFemur BILATERAL"}),
                      ("own.dcm", {})):
        ds = pydicom.dcmread(str(src))
        for k, v in mut.items():
            setattr(ds, k, v)
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.save_as(str(filt / name))
        cases[name] = mut
    out = tmp / "o_filter" / "results.csv"
    r = run(filt, out, "--debug-csv")
    rows = {Path(x["path_to_study"]).name: x for x in rows_of(out)} if out.exists() else {}
    dbg_p = next(iter(sorted(out.parent.glob("*debug*.csv"))), None)
    dbg = {Path(x.get("path_to_study") or x.get("file") or "").name: x for x in rows_of(dbg_p)} if dbg_p else {}
    for name in ("foreign_vendor.dcm", "modality_ct.dcm", "lateral.dcm"):
        row = rows.get(name, {})
        d = dbg.get(name, {})
        check(row.get("processing_status") == "Failure" and float(row.get("quality_prob", 1)) < 0.5,
              f"пакетный путь: {name} -> Failure ({row.get('processing_status')})")
        check(str(d.get("region_supported")) == "0" and bool(d.get("region_support_reason")),
              f"debug CSV: {name} region_supported=0, причина: {str(d.get('region_support_reason'))[:50]}")
    for name in ("own.dcm", "bilateral.dcm"):
        check(rows.get(name, {}).get("processing_status") == "Success",
              f"пакетный путь: {name} -> Success ({rows.get(name, {}).get('processing_status')})")
        check(str(dbg.get(name, {}).get("region_supported")) == "1", f"debug CSV: {name} region_supported=1")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\n{'ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ' if not fails else f'ПРОВАЛЕНО: {len(fails)}'}")
sys.exit(1 if fails else 0)
