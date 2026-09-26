#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест JSON Schema результата и SR на исследование.

  1. schema/results_row.schema.json: корректная строка проходит; заведомо неверные строки
     (регион не из списка, quality_prob > 1, class=1 без нарушения, неизвестное нарушение,
     лишняя колонка, статус не Success/Failure) — отклоняются.
  2. Строки регионов/нарушений/статусов в схеме совпадают с config.yaml (единый источник строк).
  3. Прогон inference.py --sr-study на tests/sample_test_zip: results.csv валиден по схеме через
     inference.validate_output_csv, в <out>/sr ровно один SR на исследование, tools/validate_sr.py -> PASS.
  4. Если установлен FastAPI: ответ POST /api/analyze валиден по schema/api_analyze_response.schema.json,
     в ответе есть study_sr со ссылкой, файл скачивается (200).

Запуск: python tests/test_schema.py   (код возврата 0 — ок).
"""
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

from inference import load_config, validate_output_csv  # noqa: E402
from schema_check import BACKEND, coerce_csv_row, load_schema, validate_instance  # noqa: E402

fails = []


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


cfg = load_config()
schema = load_schema("results_row")
print(f"schema backend: {BACKEND}")

# 1. корректная строка и набор неверных
good = {"path_to_study": "a/b.dcm", "study_uid": "1.2.3", "image_uid": "1.2.3.4",
        "anatomical_region": cfg["regions"]["spine"], "quality_class": "1",
        "violation_type": cfg["violations"]["sp_axis"] + ";" + cfg["violations"]["sp_art"],
        "quality_prob": "0.77", "processing_status": "Success", "time_of_processing": "0.05"}
check(validate_instance(coerce_csv_row(good), schema) == [], "корректная строка проходит схему")
normal = dict(good, quality_class="0", violation_type="", quality_prob="0.12")
check(validate_instance(coerce_csv_row(normal), schema) == [], "строка нормы проходит схему")
failure = dict(good, quality_class="0", violation_type="", quality_prob=str(cfg["output"]["fallback_quality_prob"]),
               processing_status="Failure")
check(validate_instance(coerce_csv_row(failure), schema) == [], "строка Failure проходит схему")

bad_cases = {
    "регион не из списка": dict(good, anatomical_region="Позвоночник"),
    "quality_prob > 1": dict(good, quality_prob="1.2"),
    "quality_prob не число": dict(good, quality_prob="abc"),
    "class=1 без нарушения": dict(good, violation_type=""),
    "class=0 с нарушением": dict(good, quality_class="0"),
    "неизвестное нарушение": dict(good, violation_type="Плохой снимок"),
    "класс 2": dict(good, quality_class="2"),
    "статус не из списка": dict(good, processing_status="OK"),
    "Failure с нарушением": dict(good, processing_status="Failure"),
    "лишняя колонка": dict(good, extra="1"),
    "пустой image_uid": dict(good, image_uid=""),
    # инвариант quality_class 1 <=> quality_prob >= 0.5 (в т. ч. Failure)
    "Failure с quality_prob 0.5": dict(failure, quality_prob="0.5"),
    "class=0 при quality_prob 0.6": dict(normal, quality_prob="0.6"),
    "class=1 при quality_prob 0.3": dict(good, quality_prob="0.3"),
    # нарушения по области (schema/official_dictionary.json)
    "бедро + нарушение оси позвоночника": dict(good, anatomical_region=cfg["regions"]["hip"],
                                                violation_type=cfg["violations"]["sp_axis"]),
    "бедро + посторонние предметы": dict(good, anatomical_region=cfg["regions"]["hip"],
                                         violation_type=cfg["violations"]["sp_art"]),
    "позвоночник + некорректная область интереса": dict(good, violation_type=cfg["violations"]["rh_roi"]),
}
for name, row in bad_cases.items():
    errs = validate_instance(coerce_csv_row(row), schema)
    check(bool(errs), f"отклонено: {name}" + (f" -> {errs[0][:80]}" if errs else ""))

# 1b. «Некорректная укладка» законна для обеих областей; область интереса — для бедра
for reg in ("spine", "hip"):
    row = dict(good, anatomical_region=cfg["regions"][reg], violation_type=cfg["violations"]["sp_pos"])
    check(validate_instance(coerce_csv_row(row), schema) == [], f"«Некорректная укладка» проходит для {reg}")
hip_ok = dict(good, anatomical_region=cfg["regions"]["hip"],
              violation_type=cfg["violations"]["rh_pos"] + ";" + cfg["violations"]["rh_roi"])
check(validate_instance(coerce_csv_row(hip_ok), schema) == [], "бедро: укладка;область интереса проходит схему")
check(validate_instance(coerce_csv_row(dict(good, quality_prob="0.5")), schema) == [], "class=1 при prob ровно 0.5 проходит")
check(0.0 <= float(cfg["output"]["fallback_quality_prob"]) < 0.5, "config.yaml output.fallback_quality_prob < 0.5")

# 1c. validate_output_csv (правила без схемы) ловит те же нарушения по области и инварианту
tmpd = Path(tempfile.mkdtemp(prefix="densito_schema_neg_"))
cols = cfg["output"]["columns"]
csv_cases = {
    "бедро + ось позвоночника": (dict(good, anatomical_region=cfg["regions"]["hip"],
                                      violation_type=cfg["violations"]["sp_axis"]), "not allowed for region"),
    "позвоночник + область интереса": (dict(good, violation_type=cfg["violations"]["rh_roi"]), "not allowed for region"),
    "Failure 0.5": (dict(failure, quality_prob="0.5"), "quality_prob 0.5 >= 0.5"),
    "class=1 при 0.3": (dict(good, quality_prob="0.3"), "< 0.5"),
}
for name, (row, needle) in csv_cases.items():
    pth = tmpd / (str(abs(hash(name))) + ".csv")
    with open(pth, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow({c: row[c] for c in cols})
    probs = validate_output_csv(pth, cfg)
    check(any(needle in p for p in probs), f"validate_output_csv отклоняет: {name}" + (f" -> {probs[:1]}" if probs else ""))
pth = tmpd / "ok.csv"
with open(pth, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for row in (good, normal, failure, hip_ok,
                dict(good, anatomical_region=cfg["regions"]["hip"], violation_type=cfg["violations"]["sp_pos"])):
        w.writerow({c: row[c] for c in cols})
probs = validate_output_csv(pth, cfg)
check(probs == [], "validate_output_csv: корректные строки обеих областей и Failure проходят" + (f" -> {probs[:2]}" if probs else ""))

# 2. согласованность со config.yaml
check(set(schema["properties"]["anatomical_region"]["enum"]) == {cfg["regions"]["spine"], cfg["regions"]["hip"]},
      "enum регионов = config.yaml regions")
check(set(schema["properties"]["violation_type"]["x-allowed-values"]) == set(cfg["violations"].values()),
      "список нарушений = config.yaml violations")
check(schema["x-column-order"] == cfg["output"]["columns"], "порядок 9 колонок = config.yaml output.columns")
check(set(schema["properties"]["processing_status"]["enum"]) == {cfg["output"]["status_success"], cfg["output"]["status_failure"]},
      "статусы = config.yaml")

# 3. прогон с --sr-study: образец организаторов, если лежит рядом (в репозитории его нет), иначе синтетические фантомы
sample = ROOT / "tests" / "sample_test_zip"
if not sample.exists():
    sample = ROOT / "tests" / "phantoms" / "study_01"
if sample.exists():
    out = Path(tempfile.mkdtemp(prefix="densito_schema_test_")) / "results.csv"
    cmd = [sys.executable, str(ROOT / "src" / "inference.py"), "--input", str(sample), "--output", str(out), "--sr-study"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    check(r.returncode == 0, f"inference.py --sr-study завершился с кодом {r.returncode}")
    problems = validate_output_csv(out, cfg)
    check(problems == [], "validate_output_csv (правила + JSON Schema): " + ("OK" if not problems else str(problems[:3])))
    with open(out, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    studies = {r["study_uid"] for r in rows}
    sr_files = sorted((out.parent / "sr").glob("*_SR.dcm"))
    check(len(sr_files) == len(studies) > 0, f"один SR на исследование: {len(sr_files)} SR / {len(studies)} исследований")
    try:
        import validate_sr  # tools/validate_sr.py
        rc = validate_sr.main([str(out.parent / "sr"), "--csv", str(out), "--log", str(out.parent / "validate_sr.log")])
        check(rc == 0, f"tools/validate_sr.py -> {'PASS' if rc == 0 else 'FAIL'}")
    except Exception as e:  # noqa: BLE001
        check(False, f"validate_sr не выполнился: {e}")
    # детерминизм UID: повторный прогон даёт те же Series/SOP Instance UID
    import pydicom
    uids1 = {p.name: (pydicom.dcmread(p).SeriesInstanceUID, pydicom.dcmread(p).SOPInstanceUID) for p in sr_files}
    out2 = out.parent / "second" / "results.csv"
    subprocess.run(cmd[:-1] + ["--output", str(out2), "--sr-study"], capture_output=True, text=True)
    uids2 = {p.name: (pydicom.dcmread(p).SeriesInstanceUID, pydicom.dcmread(p).SOPInstanceUID)
             for p in sorted((out2.parent / "sr").glob("*_SR.dcm"))}
    check(uids1 == uids2, "UID SR детерминированы (повторный прогон -> те же Series/SOP UID)")
else:
    print("SKIP нет ни образца организаторов, ни фантомов")

# 4. API-ответ по схеме
try:
    from fastapi.testclient import TestClient  # noqa: E402
    os.environ["DENSITO_OUTPUT_DIR"] = tempfile.mkdtemp(prefix="densito_schema_api_")
    import api_server  # noqa: E402
    client = TestClient(api_server.app)
    files = sorted(sample.rglob("*.dcm"))[:2]
    resp = client.post("/api/analyze", files=[("files", (p.name, p.read_bytes(), "application/dicom")) for p in files])
    check(resp.status_code == 200, f"/api/analyze -> {resp.status_code}")
    if resp.status_code == 200:
        j = resp.json()
        errs = validate_instance(j, load_schema("api_analyze_response"))
        check(errs == [], "ответ /api/analyze валиден по api_analyze_response.schema.json" + (f" ({errs[:2]})" if errs else ""))
        check(isinstance(j.get("study_sr"), dict) and len(j["study_sr"]) >= 1, f"study_sr в ответе: {len(j.get('study_sr') or {})} исследований")
        for uid, url in (j.get("study_sr") or {}).items():
            check(client.get(url).status_code == 200, f"SR исследования скачивается: {url[-50:]}")
            break
        check(all("study_sr_download" in r for r in j["rows"] if r["processing_status"] == "Success"),
              "у каждой Success-строки есть study_sr_download")
except (ImportError, RuntimeError) as e:  # RuntimeError: starlette.testclient без httpx
    print(f"SKIP FastAPI/TestClient недоступен ({str(e)[:60]}) — проверка ответа API пропущена")

print(f"\n{'ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ' if not fails else f'ПРОВАЛЕНО: {len(fails)}'}")
sys.exit(1 if fails else 0)
