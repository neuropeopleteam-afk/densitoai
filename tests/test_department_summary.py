#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Проверки сводки по отделению (tools/department_summary.py, маршруты /api/results/{job}/summary*).

Без torch и без инференса: синтетический results.csv + технический CSV собираются во временном каталоге.
  * интервал Уилсона на известных примерах (k=5,n=10 -> [0.2366; 0.7634]; k=0,n=20 -> [0; 0.1611]; n=0 -> None);
  * пометка «мало данных» при n < min_n; доли и ДИ в [0; 1];
  * суммы: файлы по областям / аппаратам / датам складываются в итог, нарушения по областям — в общее число,
    норма + нарушения + Failure = файлы, k по типам нарушения = число упоминаний типа в violation_type;
  * знаменатель долей нарушений — Success (Failure не занижает долю);
  * зона «не уверен» и причины Failure берутся из технического CSV; без него — null и заметка;
  * аппарат и дата: device_tags.csv, хэш StationName без исходного значения; чтение тегов из DICOM
    (pydicom, синтетический файл) — только аппарат и дата, оператор в сводку не попадает;
  * детерминизм: два прогона дают побайтово одинаковые JSON/MD/CSV; порядок входных файлов не влияет;
  * summary.json валиден по schema/department_summary.schema.json (src/schema_check.py);
  * запрещённые слова и персональные поля отсутствуют в выходах;
  * HTTP через fastapi.testclient (если есть httpx): 403/404, JSON по схеме, Markdown, CSV, ?top=, файлы в каталоге
    задачи; иначе SKIP, как в других тестах.

Запуск: python tests/test_department_summary.py  (код возврата 0 — все проверки пройдены)
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

TMP = Path(tempfile.mkdtemp(prefix="densito_dept_summary_"))
os.environ.setdefault("DENSITO_OUTPUT_DIR", str(TMP / "out"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import department_summary as DS  # noqa: E402
import schema_check  # noqa: E402

N_OK = N_FAIL = N_SKIP = 0


def check(cond, msg):
    global N_OK, N_FAIL
    if cond:
        N_OK += 1
        print("OK  ", msg)
    else:
        N_FAIL += 1
        print("FAIL", msg)


def skip(msg):
    global N_SKIP
    N_SKIP += 1
    print("SKIP", msg)


SPINE, HIP = "Поясничный отдел позвоночника", "Проксимальный отдел бедра"
V_POS, V_AXIS, V_ART, V_ROI = ("Некорректная укладка", "Не выравнена ось позвоночника",
                               "Присутствуют посторонние предметы", "Некорректная область интереса")
COLS = DS.COLUMNS_9


def uid(i: int) -> str:
    return f"1.2.826.0.1.3680043.8.498.{1000 + i}"


def make_rows():
    """3 исследования, 8 файлов: 5 позвоночник (2 нарушения, 1 Failure), 3 бедро (2 нарушения)."""
    rows = []

    def add(study, i, region, qc, viol, prob, status="Success", t="1.5"):
        rows.append({"path_to_study": f"Исследования/{study}/CR{i:06d}.dcm", "study_uid": uid(study), "image_uid": uid(100 + i),
                     "anatomical_region": region, "quality_class": qc, "violation_type": viol, "quality_prob": prob,
                     "processing_status": status, "time_of_processing": t})
    add(1, 0, SPINE, "0", "", "0.20")
    add(1, 1, SPINE, "1", f"{V_AXIS};{V_ART}", "0.91")
    add(1, 2, HIP, "1", V_POS, "0.77")
    add(2, 3, SPINE, "0", "", "0.5", status="Failure", t="0.1")
    add(2, 4, SPINE, "1", V_POS, "0.66")
    add(2, 5, HIP, "0", "", "0.30")
    add(3, 6, HIP, "1", f"{V_POS};{V_ROI}", "0.95")
    add(3, 7, SPINE, "0", "", "0.10")
    return rows


def make_debug(rows):
    dbg = []
    for i, r in enumerate(rows):
        d = {"file": "/data/" + r["path_to_study"], "internal_region": "spine" if r["anatomical_region"] == SPINE else "left_hip",
             "sha256_pixels": f"h{i % 6:02d}", "needs_review": "1" if i in (0, 5) else "0",
             "uncertain_criteria": "sp_pos" if i == 0 else ("lh_roi" if i == 5 else ""),
             "error": "ValueError: unable to read pixel data from /data/x/CR000003.dcm" if r["processing_status"] == "Failure" else "",
             "time_of_processing": r["time_of_processing"]}
        dbg.append(d)
    return dbg


def write_csv(path: Path, rows, cols=None):
    cols = cols or list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


ROWS = make_rows()
DBG = make_debug(ROWS)
DEV = []
for i, r in enumerate(ROWS):
    station = "DXA-01" if i < 5 else "DXA-02"
    DEV.append({"path_to_study": r["path_to_study"], "image_uid": r["image_uid"], "manufacturer": "GE Healthcare",
                "model_name": "Lunar Prodigy Advance", "station_hash": DS.short_hash(station), "serial_hash": "",
                "study_date": "2026-03-02" if i < 3 else ("2026-03-09" if i < 6 else "")})
RES = TMP / "results.csv"
write_csv(RES, ROWS, COLS)
write_csv(TMP / "results_debug.csv", DBG)
DS.write_device_tags_csv(DEV, TMP / "device_tags.csv")

# ------------------------------------------------------------------ 1. Уилсон
print("--- интервал Уилсона")
lo, hi = DS.wilson_ci(5, 10)
check(abs(lo - 0.2366) < 5e-4 and abs(hi - 0.7634) < 5e-4, f"k=5, n=10 -> [{lo:.4f}; {hi:.4f}] (ожидается [0.2366; 0.7634])")
lo, hi = DS.wilson_ci(0, 20)
check(lo == 0.0 and abs(hi - 0.1611) < 5e-4, f"k=0, n=20 -> [0; {hi:.4f}] (ожидается [0; 0.1611])")
lo, hi = DS.wilson_ci(20, 20)
check(abs(lo - 0.8389) < 5e-4 and hi == 1.0, f"k=20, n=20 -> [{lo:.4f}; 1] (симметрично k=0)")
check(DS.wilson_ci(0, 0) == (None, None), "n=0 -> (None, None)")
r = DS.rate(3, 15, min_n=20)
check(r["low_data"] is True and r["rate"] == 0.2 and r["ci_low"] <= 0.2 <= r["ci_high"], "rate(): пометка «мало данных» при n=15 < 20, точка внутри ДИ")
check(DS.rate(3, 20, min_n=20)["low_data"] is False, "rate(): n=20 не «мало данных»")
check(DS.rate(0, 0)["rate"] is None, "rate(): n=0 -> rate null")

# ------------------------------------------------------------------ 2. сводка на синтетике
print("--- сводка: суммы и знаменатели")
S = DS.summarize_files([RES], device_tags=[TMP / "device_tags.csv"], cfg={}, min_n=20, top_n=3)
t = S["totals"]
check(t["n_files"] == 8 and t["n_studies"] == 3, "итого: 8 файлов, 3 исследования")
check(t["n_success"] == 7 and t["n_failure"] == 1 and t["n_violation"] == 4 and t["n_norm"] == 3, "итого: Success 7, Failure 1, нарушений 4, норма 3")
check(t["n_norm"] + t["n_violation"] + t["n_failure"] == t["n_files"], "норма + нарушения + Failure = файлы")
check(S["violation_rate"]["n"] == 7 and S["violation_rate"]["k"] == 4, "знаменатель доли нарушений — Success (7), не все файлы")
check(S["failure_rate"]["n"] == 8 and S["failure_rate"]["k"] == 1, "доля Failure — от всех файлов")
check(sum(x["n_files"] for x in S["by_region"]) == 8 and sum(x["n_violation"] for x in S["by_region"]) == 4, "по областям: файлы и нарушения складываются в итог")
reg = {x["region"]: x for x in S["by_region"]}
check(reg[SPINE]["n_files"] == 5 and reg[SPINE]["n_success"] == 4 and reg[SPINE]["n_violation"] == 2 and reg[SPINE]["violation_rate"]["rate"] == 0.5, "позвоночник: 5 файлов, 4 Success, 2 нарушения -> 50 %")
check(reg[HIP]["n_studies"] == 3 and reg[SPINE]["n_studies"] == 3, "исследований по областям: 3 и 3")
vt = {(x["region"], x["violation_type"]): x for x in S["by_violation_type"]}
check(len(vt) == 5 and [x["violation_type"] for x in S["by_violation_type"]] == [V_POS, V_AXIS, V_ART, V_POS, V_ROI], "по типу: 5 официальных строк в официальном порядке")
check(vt[(SPINE, V_POS)]["n_files"] == 1 and vt[(SPINE, V_AXIS)]["n_files"] == 1 and vt[(SPINE, V_ART)]["n_files"] == 1
      and vt[(HIP, V_POS)]["n_files"] == 2 and vt[(HIP, V_ROI)]["n_files"] == 1, "по типу: k = число упоминаний типа в violation_type")
check(all(x["rate_files"]["n"] == reg[x["region"]]["n_success"] for x in S["by_violation_type"]), "по типу: знаменатель — Success области")
check(sum(x["n_files"] for x in S["by_device"]) == 8 and len(S["by_device"]) == 2, "по аппарату: 2 аппарата, файлы складываются в итог")
dev_hashes = {x["device_hash"] for x in S["by_device"]}
check(dev_hashes == {DS.short_hash("DXA-01"), DS.short_hash("DXA-02")} and not any("DXA-0" in x["device"] for x in S["by_device"]), "аппарат обозначен хэшем, исходный StationName в сводке отсутствует")
check(S["by_device"][0]["n_files"] == 5 and S["by_device"][0]["n_violation"] == 3, "аппараты отсортированы по числу файлов; у первого 5 файлов, 3 нарушения")
bd = S["by_date"]
check(sum(x["n_files"] for x in bd["day"]) + bd["n_files_without_date"] == 8 and bd["n_files_without_date"] == 2, "по дате: дни + без даты = файлы (без даты 2)")
check([x["period"] for x in bd["day"]] == ["2026-03-02", "2026-03-09"] and [x["period"] for x in bd["week"]] == ["2026-W10", "2026-W11"], "по дате: дни и ISO-недели")
check(S["period"]["date_min"] == "2026-03-02" and S["period"]["date_max"] == "2026-03-09", "период: min/max даты")
check(t["n_uncertain"] == 2 and S["uncertain_rate"]["k"] == 2 and S["uncertain_rate"]["n"] == 7, "зона «не уверен» из технического CSV: 2 файла из 7 Success")
check([(x["criterion"], x["n_files"]) for x in S["uncertain_by_criterion"]] == [("lh_roi", 1), ("sp_pos", 1)], "«не уверен» по критериям: lh_roi 1, sp_pos 1")
check(t["n_unique_pixel_hashes"] == 6, "уникальных кадров по хэшу пикселей: 6 из 8")
check(len(S["failure_reasons"]) == 1 and S["failure_reasons"][0]["n_files"] == 1 and "/data" not in S["failure_reasons"][0]["reason"]
      and S["failure_reasons"][0]["reason"].startswith("ValueError"), "причина Failure из технического CSV, путь удалён: " + S["failure_reasons"][0]["reason"])
check(S["failure_reasons"][0]["share"]["n"] == 1 and S["failure_reasons"][0]["share"]["low_data"] is True, "доля причины среди отказов с пометкой «мало данных»")
top = S["review_top"]
check(len(top) == 3 and top[0]["study_uid"] == uid(3) and top[0]["max_quality_prob"] == 0.95 and top[1]["study_uid"] == uid(1), "топ для пересмотра: по max quality_prob, ограничен top_n=3")
check(top[0]["image_uid"] == uid(106) and top[0]["violation_types"] == sorted([V_POS, V_ROI]), "топ: image_uid максимума и типы нарушений исследования")
check(all(set(x.keys()) <= {"rank", "study_uid", "image_uid", "max_quality_prob", "n_files", "n_violation", "n_uncertain", "regions", "violation_types", "study_date"} for x in top), "топ: только UID и агрегаты, без путей и имён")
check(t["n_studies_with_violation"] == 3 and S["violation_rate_studies"]["n"] == 3, "исследований с нарушением 3 из 3 обработанных")
check(all(x["low_data"] is True for x in (S["violation_rate"], S["failure_rate"])), "при 8 файлах доли помечены «мало данных»")
all_rates = [S["violation_rate"], S["failure_rate"], S["uncertain_rate"]] + [x["violation_rate"] for x in S["by_region"]] + [x["rate_files"] for x in S["by_violation_type"]]
check(all(0.0 <= x["ci_low"] <= (x["rate"] if x["rate"] is not None else 0) <= x["ci_high"] <= 1.0 for x in all_rates), "все ДИ в [0; 1] и содержат точку")

# без технического CSV
S0 = DS.build_summary(ROWS, None, None, None, cfg={}, min_n=20, top_n=5)
check(S0["uncertain_rate"] is None and S0["totals"]["n_uncertain"] is None and S0["uncertain_by_criterion"] is None, "без технического CSV зона «не уверен» = null")
check(any("Технический CSV" in n for n in S0["notes"]) and any("аппарат" in n for n in S0["notes"]), "без технического CSV и тегов — заметки в notes")
check(len(S0["by_device"]) == 1 and S0["by_device"][0]["device_hash"] == "" and S0["by_device"][0]["n_files"] == 8, "без тегов один аппарат «не указан» со всеми файлами")
check(S0["failure_reasons"][0]["reason"] == "причина не записана (нет технического CSV)", "без технического CSV причина отказа помечена")
check(S0["by_date"]["day"] == [] and S0["by_date"]["n_files_without_date"] == 8, "без тегов разрез по датам пуст")

# конфигурация репозитория: строки областей/нарушений из config.yaml совпадают с встроенными
cfg, h = DS.load_config_light()
if cfg:
    Sc = DS.build_summary(ROWS, DBG, None, DEV, cfg=cfg, min_n=20, top_n=3)
    check([x["region"] for x in Sc["by_region"]] == [SPINE, HIP] and [x["violation_type"] for x in Sc["by_violation_type"]] == [V_POS, V_AXIS, V_ART, V_POS, V_ROI], "config.yaml: строки областей и нарушений совпадают с официальными")
    check(json.dumps(Sc["by_region"], sort_keys=True) == json.dumps(S["by_region"], sort_keys=True), "config.yaml и cfg={} дают одинаковый разрез по областям")
else:
    skip("config.yaml не прочитан (нет PyYAML) — сравнение с конфигурацией пропущено")

# ------------------------------------------------------------------ 3. детерминизм и выходы
print("--- детерминизм, файлы, схема")
out1, out2 = TMP / "o1", TMP / "o2"
DS.write_outputs(DS.summarize_files([RES], device_tags=[TMP / "device_tags.csv"], cfg={}, min_n=20, top_n=3), out1)
DS.write_outputs(DS.summarize_files([RES], device_tags=[TMP / "device_tags.csv"], cfg={}, min_n=20, top_n=3), out2)
check(all((out1 / n).read_bytes() == (out2 / n).read_bytes() for n in ("summary.json", "summary.md", "summary.csv")), "два прогона -> побайтово одинаковые summary.json/.md/.csv")
# порядок входных файлов: две половины в разном порядке
RES_A, RES_B = TMP / "a.csv", TMP / "b.csv"
write_csv(RES_A, ROWS[:4], COLS)
write_csv(RES_B, ROWS[4:], COLS)
write_csv(TMP / "a_debug.csv", DBG[:4])
write_csv(TMP / "b_debug.csv", DBG[4:])
Sab = DS.summarize_files([RES_A, RES_B], device_tags=[TMP / "device_tags.csv"], cfg={}, min_n=20, top_n=3)
Sba = DS.summarize_files([RES_B, RES_A], device_tags=[TMP / "device_tags.csv"], cfg={}, min_n=20, top_n=3)
strip = lambda s: {k: v for k, v in s.items() if k != "inputs"}  # noqa: E731
check(json.dumps(strip(Sab), sort_keys=True) == json.dumps(strip(Sba), sort_keys=True) == json.dumps(strip(S), sort_keys=True), "два results.csv в любом порядке = один общий results.csv (кроме списка входов)")
check(Sab["inputs"]["results_csv"] == ["a.csv", "b.csv"] and Sab["inputs"]["debug_csv"] == ["a_debug.csv", "b_debug.csv"], "технические CSV подобраны автоматически по имени <stem>_debug.csv")
schema = schema_check.load_schema("department_summary.schema.json")
errs = schema_check.validate_instance(json.loads((out1 / "summary.json").read_text(encoding="utf-8")), schema)
check(errs == [], "summary.json валиден по schema/department_summary.schema.json" + (f": {errs[:3]}" if errs else ""))
errs0 = schema_check.validate_instance(json.loads(json.dumps(S0)), schema)
check(errs0 == [], "сводка без технического CSV тоже валидна по схеме" + (f": {errs0[:3]}" if errs0 else ""))
bad = json.loads(json.dumps(S))
bad["violation_rate"]["rate"] = 1.5
check(schema_check.validate_instance(bad, schema) != [], "схема отклоняет доля > 1")
md = (out1 / "summary.md").read_text(encoding="utf-8")
check("## По области" in md and "## По аппарату" in md and "Уилсона 95 %" in md and "мало данных" in md, "Markdown: разделы, ДИ Уилсона, «мало данных»")
check("!" not in md.replace("![", ""), "Markdown без восклицательных знаков")
csv_rows = list(csv.DictReader((out1 / "summary.csv").read_text(encoding="utf-8").splitlines(), delimiter=";"))
check(list(csv_rows[0].keys()) == DS.CSV_COLUMNS and len(csv_rows) >= 20, f"summary.csv: колонки {DS.CSV_COLUMNS[:4]}…, строк {len(csv_rows)}")
check(all(r["k"] and r["n"] for r in csv_rows) and any(r["section"] == "по аппарату" for r in csv_rows), "summary.csv: k и n заполнены, есть раздел по аппарату")
banned = ["Grad" + "-CAM", "автокор" + "рекция ROI", "ЕР" + "ИС", "скол" + "иоз", "угол Ко" + "бба", "экономи"]
blob = json.dumps(S, ensure_ascii=False) + md + (out1 / "summary.csv").read_text(encoding="utf-8")
check(not any(b.lower() in blob.lower() for b in banned), "запрещённых слов в выходах нет")
pii = ["PatientName", "PatientID", "Operator", "DXA-01", "InstitutionName", "/data/"]
check(not any(p in blob for p in pii), "персональных полей, имён операторов и путей в выходах нет")
src_text = (ROOT / "tools" / "department_summary.py").read_text(encoding="utf-8")
check(not re.search(r"Operator|PatientName|PatientID|InstitutionName", src_text), "инструмент не читает теги оператора/пациента/учреждения")

# ------------------------------------------------------------------ 4. теги из DICOM (синтетический файл)
print("--- теги аппарата и даты из DICOM")
try:
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    dcm_root = TMP / "dicom"
    for i, r in enumerate(ROWS):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
        ds.file_meta.MediaStorageSOPInstanceUID = r["image_uid"]
        ds.SOPInstanceUID = r["image_uid"]
        ds.StudyInstanceUID = r["study_uid"]
        ds.Manufacturer = "GE Healthcare"
        ds.ManufacturerModelName = "Lunar Prodigy Advance"
        ds.StationName = "DXA-77" if i % 2 else "Anonymized"
        ds.DeviceSerialNumber = "SN-1" if i % 2 else ""
        ds.StudyDate = "20260315" if i < 6 else "Anonymized"
        ds.OperatorsName = "Оператор Тестовый"
        ds.PatientName = "Пациент Тестовый"
        p = dcm_root / r["path_to_study"]
        p.parent.mkdir(parents=True, exist_ok=True)
        ds.save_as(str(p), write_like_original=False)
    tags = DS.read_device_tags_dicom(ROWS, dcm_root)
    check(len(tags) == 8 and set(tags[0].keys()) == set(DS.DEVICE_TAGS_COLUMNS), "прочитаны теги для 8 файлов, только колонки device_tags")
    check(tags[1]["station_hash"] == DS.short_hash("DXA-77") and tags[0]["station_hash"] == "" and tags[0]["serial_hash"] == "", "StationName -> хэш; «Anonymized» и пусто -> не указан")
    check(tags[0]["study_date"] == "2026-03-15" and tags[7]["study_date"] == "", "StudyDate -> ISO; «Anonymized» -> пусто")
    Sd = DS.summarize_files([RES], dicom_root=dcm_root, cfg={}, min_n=20, top_n=3)
    check(Sd["inputs"]["device_source"] == "dicom" and len(Sd["by_device"]) == 2 and Sd["by_date"]["n_files_without_date"] == 2, "сводка с --dicom-root: источник dicom, 2 аппарата, 2 файла без даты")
    blob_d = json.dumps(Sd, ensure_ascii=False)
    check("Оператор" not in blob_d and "Пациент" not in blob_d and "DXA-77" not in blob_d and "SN-1" not in blob_d, "имена оператора/пациента и исходные идентификаторы аппарата в сводку не попали")
except ImportError:
    skip("pydicom недоступен — чтение тегов из DICOM пропущено")

# ------------------------------------------------------------------ 5. HTTP через TestClient (если есть httpx)
print("--- HTTP (fastapi.testclient)")
client = None
try:
    import api_server  # noqa: E402
    from fastapi.testclient import TestClient
    client = TestClient(api_server.app)
    client.get("/api/health")
except Exception as e:  # noqa: BLE001
    skip(f"TestClient недоступен ({type(e).__name__}: {str(e)[:80]}) — HTTP-часть пропущена")
    client = None
if client is not None:
    check(api_server.department_summary is not None, "api_server загрузил tools/department_summary.py")
    JOB = "20260923_120000_abc123"
    JOB_DIR = api_server.JOBS_DIR / JOB
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN = api_server._issue_job_token(JOB_DIR)
    write_csv(JOB_DIR / "results.csv", ROWS, COLS)
    write_csv(JOB_DIR / "results_debug.csv", DBG)
    DS.write_device_tags_csv(DEV, JOB_DIR / api_server.DEVICE_TAGS_FILE)
    (JOB_DIR / "summary.json").write_text(json.dumps({"job_id": JOB, "rows": []}, ensure_ascii=False), encoding="utf-8")
    base = f"/api/results/{JOB}"
    check(client.get(f"{base}/summary").status_code == 403, "GET summary без кода задачи -> 403")
    check(client.get(f"{base}/summary.md?t=wrong").status_code == 403, "GET summary.md с чужим кодом -> 403")
    check(client.get(f"/api/results/20260101_000000_ffffff/summary?t={TOKEN}").status_code == 404, "несуществующая задача -> 404")
    check(client.get(f"/api/results/..%2F{JOB}/summary?t={TOKEN}").status_code in (400, 404, 422), "traversal в job -> отказ")
    r = client.get(f"{base}/summary?t={TOKEN}")
    check(r.status_code == 200 and r.json()["kind"] == "department_summary" and r.json()["job_id"] == JOB, "GET summary -> 200, kind=department_summary, job_id")
    j = r.json()
    check(j["totals"]["n_files"] == 8 and j["totals"]["n_uncertain"] == 2 and len(j["by_device"]) == 2 and j["pipeline_version"] == api_server.PIPELINE_VERSION, "JSON из API: те же итоги, что у CLI; версия пайплайна")
    e = schema_check.validate_instance(j, schema)
    check(e == [], "ответ API валиден по schema/department_summary.schema.json" + (f": {e[:3]}" if e else ""))
    check(json.dumps(strip(j), sort_keys=True) == json.dumps(strip(dict(S, job_id=JOB, pipeline_version=j["pipeline_version"], config_hash=j["config_hash"])), sort_keys=True), "API и CLI дают одну и ту же сводку")
    r = client.get(f"{base}/summary?t={TOKEN}&top=1")
    check(r.status_code == 200 and len(r.json()["review_top"]) == 1, "?top=1 ограничивает список для пересмотра")
    r = client.get(f"{base}/summary?t={TOKEN}&min_n=5")
    check(r.status_code == 200 and r.json()["method"]["min_n"] == 5 and r.json()["violation_rate"]["low_data"] is False, "?min_n=5 меняет порог «мало данных»")
    r = client.get(f"{base}/summary.md", headers={"X-Job-Token": TOKEN})
    check(r.status_code == 200 and r.headers["content-type"].startswith("text/markdown") and "## По области" in r.text, "GET summary.md через заголовок X-Job-Token -> Markdown")
    r = client.get(f"{base}/summary.csv?t={TOKEN}")
    check(r.status_code == 200 and r.headers["content-type"].startswith("text/csv") and "section;group" in r.text, "GET summary.csv -> CSV с разделителем «;»")
    check((JOB_DIR / "department_summary.json").is_file() and (JOB_DIR / "department_summary.md").is_file() and (JOB_DIR / "department_summary.csv").is_file(), "файлы department_summary.{json,md,csv} записаны в каталог задачи")
    check(client.get(f"{base}/department_summary.md?t={TOKEN}").status_code == 200, "department_summary.md скачивается общим маршрутом файлов")
    check(client.get(f"{base}/summary.json?t={TOKEN}").status_code == 200 and client.get(f"{base}/summary.json?t={TOKEN}").json().get("job_id") == JOB, "GET summary.json (карточка задачи) не изменился")
    check(client.get(f"{base}/decisions?t={TOKEN}").status_code == 200, "маршрут decisions по-прежнему работает")
    # задача без results.csv
    JOB2 = "20260923_120001_abc124"
    (api_server.JOBS_DIR / JOB2).mkdir(parents=True, exist_ok=True)
    T2 = api_server._issue_job_token(api_server.JOBS_DIR / JOB2)
    check(client.get(f"/api/results/{JOB2}/summary?t={T2}").status_code == 404, "задача без results.csv -> 404")
    # задача без технического CSV и без тегов
    JOB3 = "20260923_120002_abc125"
    (api_server.JOBS_DIR / JOB3).mkdir(parents=True, exist_ok=True)
    T3 = api_server._issue_job_token(api_server.JOBS_DIR / JOB3)
    write_csv(api_server.JOBS_DIR / JOB3 / "results.csv", ROWS, COLS)
    r = client.get(f"/api/results/{JOB3}/summary?t={T3}")
    check(r.status_code == 200 and r.json()["uncertain_rate"] is None and len(r.json()["by_device"]) == 1, "задача без технического CSV: 200, «не уверен» null, один аппарат «не указан»")

print(f"\nИТОГ: OK {N_OK}, FAIL {N_FAIL}, SKIP {N_SKIP}")
sys.exit(1 if N_FAIL else 0)
