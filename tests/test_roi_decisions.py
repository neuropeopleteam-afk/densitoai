#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Проверки идеи 23: предложение коррекции области интереса бедра с подтверждением специалистом.

Без torch и без инференса: задача (job) собирается вручную в временном каталоге (summary.json,
.job_token), после чего проверяются
  * сборка roi_suggestion из debug-полей (api_server._roi_suggestion) — бедро / позвоночник / отказ;
  * валидация тела решения (validate_decision): значения decision, «своя» без рамки, рамка вне кадра,
    пустой специалист, неизвестный image_uid, снимок позвоночника;
  * атомарное хранение decisions.json (save_decisions_atomic / load_decisions) и CSV-выгрузка;
  * DICOM SR решений (dicom_sr.build_decision_sr): детерминированные UID, имя файла, tools/validate_sr.py;
  * схема schema/roi_decision.schema.json (src/schema_check.py);
  * HTTP-часть через fastapi.testclient — правила доступа (код задачи, 403/404, traversal) и все четыре
    маршрута. Если в окружении нет httpx (в образе его нет), HTTP-часть пропускается с пометкой SKIP;
    тогда её закрывает work/idea23_roi_confirm/verify/api_probe.py (uvicorn + urllib).

Запуск: python tests/test_roi_decisions.py  (код возврата 0 — все проверки пройдены)
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TMP = Path(tempfile.mkdtemp(prefix="densito_roi_dec_"))
os.environ.setdefault("DENSITO_OUTPUT_DIR", str(TMP / "out"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import api_server  # noqa: E402
import dicom_sr  # noqa: E402
import schema_check  # noqa: E402

N_OK = 0
N_FAIL = 0
N_SKIP = 0


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


STUDY = "2.25.410297758753619049490870449023056571"
IMG_HIP = "2.25.277731830796038092221189362316512869"
IMG_HIP2 = "2.25.277731830796038092221189362316512870"
IMG_SPINE = "2.25.1000000000000000000000000000000001"
SOP_DX = "1.2.840.10008.5.1.4.1.1.1"


def dbg_hip(needs=True):
    return {
        "internal_region": "right_hip", "rows": 291, "cols": 280, "sop_class_uid": SOP_DX,
        "bonus_roi_needs_correction": needs, "bonus_roi_reason": "lateral_margin_below_threshold" if needs else None,
        "bonus_roi_deficit_mm": 20.0 if needs else None, "bonus_roi_box_px": "0,6,250,290" if needs else "",
        "bonus_roi_ext_box_px": "", "bonus_roi_bone_box_px": "30,10,200,280", "bonus_roi_side": "right",
        "bonus_roi_pixel_spacing_mm": "1.0500,0.6000",
    }


# ------------------------------------------------------------------ 1. roi_suggestion
print("--- roi_suggestion из debug-полей")
s = api_server._roi_suggestion({}, dbg_hip(True))
check(isinstance(s, dict) and s["needs_correction"] is True, "бедро с коррекцией -> объект, needs_correction=True")
check(s and s["box_px"] == [0, 6, 250, 290], "box_px разобран из строки debug")
check(s and s["box_mm"] == [0.0, 6.3, 150.0, 304.5], f"box_mm по шагу пикселя (y 1.05, x 0.6): {s and s['box_mm']}")
check(s and s["deficit_mm"] == 20.0 and s["reason"] == "lateral_margin_below_threshold", "deficit_mm и reason переданы")
check(s and s["source"] == "предложение системы", "source = «предложение системы»")
check(s and s["image_size"] == [291, 280] and s["sop_class_uid"] == SOP_DX, "image_size и sop_class_uid для SR")
s0 = api_server._roi_suggestion({}, dbg_hip(False))
check(isinstance(s0, dict) and s0["needs_correction"] is False and s0["box_px"] is None, "бедро без коррекции -> объект с box_px=null")
check(api_server._roi_suggestion({}, {"internal_region": "spine", "bonus_roi_needs_correction": None}) is None, "позвоночник -> null")
check(api_server._roi_suggestion({}, dbg_hip(True), region_ok=False) is None, "отказ по области -> null")
check(api_server._roi_suggestion({}, {"internal_region": "left_hip"}) is None, "бедро без debug-полей auto_roi -> null")
check(api_server._roi_suggestion({}, dict(dbg_hip(True), bonus_roi_error="boom")) is None, "ошибка auto_roi -> null")
sch = schema_check.load_schema("api_analyze_response.schema.json")
row_schema = sch["$defs"]["row_with_extras"]["properties"]["roi_suggestion"]
check(schema_check.validate_instance(s, row_schema) == [], "roi_suggestion соответствует схеме ответа API")
check(schema_check.validate_instance(s0, row_schema) == [], "roi_suggestion (без коррекции) соответствует схеме")
check(schema_check.validate_instance(None, row_schema) == [], "roi_suggestion = null допускается схемой")

# ------------------------------------------------------------------ 2. задача вручную
JOB = "20260923_120000_abcdef"
JOB_DIR = api_server.JOBS_DIR / JOB
JOB_DIR.mkdir(parents=True, exist_ok=True)
TOKEN = api_server._issue_job_token(JOB_DIR)
rows = [
    {"path_to_study": "study_02/CR000000.dcm", "study_uid": STUDY, "image_uid": IMG_SPINE,
     "anatomical_region": "Поясничный отдел позвоночника", "roi_suggestion": None, "details": {"image_size": [317, 300]}},
    {"path_to_study": "study_02/CR000001.dcm", "study_uid": STUDY, "image_uid": IMG_HIP,
     "anatomical_region": "Проксимальный отдел бедра", "roi_suggestion": s, "details": {"image_size": [291, 280]}},
    {"path_to_study": "study_02/CR000002.dcm", "study_uid": STUDY, "image_uid": IMG_HIP2,
     "anatomical_region": "Проксимальный отдел бедра", "roi_suggestion": s0, "details": {"image_size": [291, 280]}},
]
(JOB_DIR / "summary.json").write_text(json.dumps({"job_id": JOB, "rows": rows}, ensure_ascii=False), encoding="utf-8")

print("--- validate_decision")
V = api_server.validate_decision


def bad(body, frag):
    try:
        V(body, rows)
        check(False, f"ожидалась ошибка «{frag}»")
    except ValueError as e:
        check(frag in str(e), f"ошибка «{frag}»: {e}")


rec1 = V({"image_uid": IMG_HIP, "decision": "подтверждено", "specialist": "врач-рентгенолог, И.И.", "comment": " ок \n"}, rows)
check(rec1["decision"] == "подтверждено" and rec1["roi_box_px"] == [0, 6, 250, 290], "«подтверждено» -> итоговая рамка = предложенная")
check(rec1["roi_box_mm"] == [0.0, 6.3, 150.0, 304.5] and rec1["comment"] == "ок", "мм пересчитаны, комментарий очищен")
check(rec1["study_uid"] == STUDY and rec1["sop_class_uid"] == SOP_DX and rec1["source"] == "предложение системы", "study_uid/sop_class_uid/source из строки задачи")
check(re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$", rec1["created_at"]) is not None, "created_at в ISO 8601")
rec2 = V({"image_uid": IMG_HIP, "decision": "отклонено", "specialist": "лаборант"}, rows)
check(rec2["roi_box_px"] is None and rec2["suggested_box_px"] == [0, 6, 250, 290], "«отклонено» -> итоговой рамки нет, предложение сохранено")
rec3 = V({"image_uid": IMG_HIP, "decision": "своя", "specialist": "врач", "roi_box_px": [10, 20, 240, 280]}, rows)
check(rec3["roi_box_px"] == [10, 20, 240, 280] and rec3["roi_box_mm"] == [6.0, 21.0, 144.0, 294.0], "«своя» -> своя рамка и мм")
rec4 = V({"image_uid": IMG_HIP2, "decision": "своя", "specialist": "врач", "roi_box_px": ["5", "5", "100", "100.4"]}, rows)
check(rec4["roi_box_px"] == [5, 5, 100, 100] and rec4["suggested_box_px"] is None, "«своя» на бедре без предложения допустима; строки в рамке приводятся к int")
bad({"image_uid": IMG_HIP, "decision": "своя", "specialist": "врач"}, "нужен roi_box_px")
bad({"image_uid": IMG_HIP, "decision": "своя", "specialist": "врач", "roi_box_px": [0, 0, 300, 100]}, "выходит за кадр")
bad({"image_uid": IMG_HIP, "decision": "своя", "specialist": "врач", "roi_box_px": [50, 50, 40, 100]}, "x0 < x1")
bad({"image_uid": IMG_HIP, "decision": "своя", "specialist": "врач", "roi_box_px": [0, 0, 2, 100]}, "меньше 4 px")
bad({"image_uid": IMG_HIP, "decision": "своя", "specialist": "врач", "roi_box_px": [1, 2, 3]}, "нужен roi_box_px")
bad({"image_uid": IMG_HIP2, "decision": "подтверждено", "specialist": "врач"}, "подтверждать нечего")
bad({"image_uid": IMG_HIP, "decision": "maybe", "specialist": "врач"}, "decision должно быть одним из")
bad({"image_uid": IMG_HIP, "decision": "подтверждено", "specialist": ""}, "укажите специалиста")
bad({"image_uid": IMG_HIP, "decision": "подтверждено", "specialist": "x" * 81}, "длиннее 80")
bad({"image_uid": IMG_HIP, "decision": "подтверждено", "specialist": "врач", "comment": "y" * 501}, "длиннее 500")
bad({"image_uid": IMG_HIP, "decision": "подтверждено", "specialist": 5}, "должно быть строкой")
bad({"image_uid": IMG_SPINE, "decision": "отклонено", "specialist": "врач"}, "не формировалось")
bad({"image_uid": "2.25.999", "decision": "отклонено", "specialist": "врач"}, "не найден")
bad({"image_uid": "../../etc", "decision": "отклонено", "specialist": "врач"}, "image_uid обязателен")
bad({"decision": "отклонено", "specialist": "врач"}, "image_uid обязателен")
try:
    V(["not", "a", "dict"], rows)
    check(False, "список вместо объекта отклонён")
except ValueError:
    check(True, "список вместо объекта отклонён")

dec_schema = schema_check.load_schema("roi_decision.schema.json")
for i, rec in enumerate((rec1, rec2, rec3, rec4), 1):
    check(schema_check.validate_instance(rec, dec_schema) == [], f"запись {i} соответствует schema/roi_decision.schema.json")

# ------------------------------------------------------------------ 3. хранение и CSV
print("--- decisions.json и CSV")
check(api_server.load_decisions(JOB_DIR) == [], "пустой каталог -> []")
p = api_server.save_decisions_atomic(JOB_DIR, [rec1, rec2])
check(p.name == "decisions.json" and p.is_file(), "decisions.json записан")
check(not list(JOB_DIR.glob(".decisions_*.tmp")), "временных файлов после записи не осталось")
loaded = api_server.load_decisions(JOB_DIR)
check(len(loaded) == 2 and loaded[0]["decision_id"] == rec1["decision_id"], "load_decisions читает записанное")
api_server.save_decisions_atomic(JOB_DIR, loaded + [rec3])
check(len(api_server.load_decisions(JOB_DIR)) == 3, "дозапись третьего решения")
(JOB_DIR / "decisions.json").write_text("{broken", encoding="utf-8")
check(api_server.load_decisions(JOB_DIR) == [], "битый decisions.json -> [] (без исключения)")
api_server.save_decisions_atomic(JOB_DIR, [rec1, rec2, rec3])
csv_text = api_server.decisions_to_csv([rec1, rec2, rec3])
lines = csv_text.lstrip("\ufeff").splitlines()
check(lines[0].startswith("decision_id;created_at;study_uid;image_uid;path_to_study;decision;specialist"), "заголовок CSV, разделитель «;»")
check(len(lines) == 4 and "подтверждено" in lines[1] and "отклонено" in lines[2] and "своя" in lines[3], "по строке на решение")
check("0,6,250,290" in lines[1] and "10,20,240,280" in lines[3], "рамки в CSV через запятую внутри ячейки")

# ------------------------------------------------------------------ 4. DICOM SR решений
print("--- build_decision_sr")
sr_dir = TMP / "sr"
sr_dir.mkdir()
fn = dicom_sr.decision_sr_filename(STUDY)
check(fn == f"{STUDY}_SR_decisions.dcm", f"имя файла {fn}")
check(dicom_sr.decision_sr_filename("a/b c") == "a_b_c_SR_decisions.dcm", "недопустимые символы в имени заменены")
ds = dicom_sr.build_decision_sr(STUDY, [rec1, rec2, rec3], "2.3.2", "cfg0")
ds_same = dicom_sr.build_decision_sr(STUDY, [rec3, rec1, rec2], "2.3.2", "cfg0")
ds_more = dicom_sr.build_decision_sr(STUDY, [rec1, rec2, rec3, rec4], "2.3.2", "cfg0")
check(ds.SOPInstanceUID == ds_same.SOPInstanceUID, "SOP UID детерминирован (порядок решений не важен)")
check(ds.SOPInstanceUID != ds_more.SOPInstanceUID and ds.SeriesInstanceUID == ds_more.SeriesInstanceUID, "новое решение -> новый SOP UID, серия та же")
check(ds.StudyInstanceUID == STUDY and int(ds.SeriesNumber) == 9002 and ds.Modality == "SR", "StudyInstanceUID исходного исследования, SeriesNumber 9002")
main_sr = dicom_sr.build_study_sr(STUDY, [], "2.3.2", "cfg0")
check(main_sr.SeriesInstanceUID != ds.SeriesInstanceUID and main_sr.SOPInstanceUID != ds.SOPInstanceUID, "серия и экземпляр отличаются от основного SR исследования")
codes = [c.ConceptNameCodeSequence[0].CodeValue for c in ds.ContentSequence]
check(codes.count("ROI-DEC-ITEM") == 3 and "AI-WARNING" in codes and "STUDY-UID-SRC" in codes, f"корень: 3 контейнера решений, контекст ИИ; {codes[:6]}")
items = [{c.ConceptNameCodeSequence[0].CodeValue: c for c in it.ContentSequence}
         for it in ds.ContentSequence if it.ConceptNameCodeSequence[0].CodeValue == "ROI-DEC-ITEM"]
dec_codes = sorted(it["ROI-DECISION"].ConceptCodeSequence[0].CodeValue for it in items)
check(dec_codes == ["ROI-CONFIRMED", "ROI-CUSTOM", "ROI-REJECTED"], f"решения как CODE: {dec_codes}")
sub = [it for it in items if it["ROI-DECISION"].ConceptCodeSequence[0].CodeValue == "ROI-CONFIRMED"][0]
check(sub["SRC-IMAGE"].ReferencedSOPSequence[0].ReferencedSOPInstanceUID == IMG_HIP, "ссылка на исходный снимок (IMAGE)")
check("FINAL-BOX-PX" not in [it for it in items if it["ROI-DECISION"].ConceptCodeSequence[0].CodeValue == "ROI-REJECTED"][0], "у «отклонено» итоговой рамки нет")
check("SUGGESTED-BOX-PX" in sub and "SUGGESTED-BOX-MM" in sub and "SPECIALIST" in sub and "DECISION-TIME" in sub, "рамка (px, мм), специалист, время присутствуют")
check(sub["DECISION-TIME"].ValueType == "DATETIME" and re.match(r"^\d{14}$", str(sub["DECISION-TIME"].DateTime)) is not None, f"время в формате DICOM DT: {sub['DECISION-TIME'].DateTime}")
ev = ds.CurrentRequestedProcedureEvidenceSequence[0]
check(ev.StudyInstanceUID == ds.StudyInstanceUID and ev.ReferencedSeriesSequence[0].ReferencedSOPSequence[0].ReferencedSOPInstanceUID == IMG_HIP, "Evidence ссылается на тот же снимок")
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("error")
    warnings.simplefilter("ignore", DeprecationWarning)  # write_like_original в pydicom 3 — как в основном коде репозитория
    try:
        out_path = sr_dir / fn
        ds.save_as(str(out_path), write_like_original=False)
        check(True, "запись без предупреждений pydicom о длине VR")
    except Warning as w:  # noqa: BLE001
        check(False, f"предупреждение pydicom при записи: {w}")
        ds.save_as(str(sr_dir / fn), write_like_original=False)
empty = dicom_sr.build_decision_sr(STUDY, [], "2.3.2", "cfg0")
empty.save_as(str(sr_dir / "empty_SR_decisions.dcm"), write_like_original=False)
check("CurrentRequestedProcedureEvidenceSequence" not in empty, "пустой список решений -> SR без Evidence (валидатор это допускает)")
import subprocess
rv = subprocess.run([sys.executable, str(ROOT / "tools" / "validate_sr.py"), str(sr_dir)], capture_output=True, text=True)
check(rv.returncode == 0 and "PASS" in rv.stdout, "tools/validate_sr.py на 2 файлах: " + (rv.stdout.strip().splitlines()[-1] if rv.stdout.strip() else rv.stderr[-200:]))

# ------------------------------------------------------------------ 5. HTTP через TestClient (если есть httpx)
print("--- HTTP (fastapi.testclient)")
client = None
try:
    from fastapi.testclient import TestClient
    client = TestClient(api_server.app)
    client.get("/api/health")
except Exception as e:  # noqa: BLE001
    skip(f"TestClient недоступен ({type(e).__name__}: {str(e)[:80]}) — HTTP проверяет verify/api_probe.py")
    client = None
if client is not None:
    base = f"/api/results/{JOB}"
    api_server.save_decisions_atomic(JOB_DIR, [])
    body = {"image_uid": IMG_HIP, "decision": "подтверждено", "specialist": "врач"}
    check(client.post(f"{base}/decisions", json=body).status_code == 403, "POST без кода задачи -> 403")
    check(client.post(f"{base}/decisions?t=wrong", json=body).status_code == 403, "POST с чужим кодом -> 403")
    check(client.get(f"{base}/decisions").status_code == 403, "GET списка без кода -> 403")
    check(client.get(f"{base}/decisions.csv?t=wrong").status_code == 403, "GET CSV с чужим кодом -> 403")
    check(client.get(f"{base}/decisions_sr/{STUDY}").status_code == 403, "GET SR без кода -> 403")
    check(client.get(f"/api/results/20260101_000000_ffffff/decisions?t={TOKEN}").status_code == 404, "несуществующая задача -> 404")
    check(client.get(f"/api/results/..%2F{JOB}/decisions?t={TOKEN}").status_code in (400, 404, 422), "traversal в job -> отказ")
    check(client.get(f"{base}/decisions_sr/..%2F..%2Fetc%2Fpasswd?t={TOKEN}").status_code in (400, 404, 422), "traversal в study_uid -> отказ")
    check(client.get(f"{base}/decisions_sr/abc?t={TOKEN}").status_code == 400, "study_uid не UID -> 400")
    check(client.get(f"{base}/decisions_sr/1.2.3?t={TOKEN}").status_code == 404, "study_uid не из задачи -> 404")
    check(client.get(f"{base}/.job_token?t={TOKEN}").status_code == 404, "файл кода доступа не отдаётся")
    r = client.post(f"{base}/decisions", json=body, headers={"X-Job-Token": TOKEN})
    check(r.status_code == 200 and r.json()["decision"]["decision"] == "подтверждено", "POST «подтверждено» через заголовок X-Job-Token")
    r = client.post(f"{base}/decisions?t={TOKEN}", json={"image_uid": IMG_HIP, "decision": "своя", "specialist": "лаборант", "roi_box_px": [10, 20, 240, 280]})
    check(r.status_code == 200 and r.json()["n_decisions"] == 2 and "decisions_sr" in (r.json().get("sr_url") or ""), "POST «своя» через ?t=; в ответе ссылка на SR решений")
    r = client.post(f"{base}/decisions?t={TOKEN}", json={"image_uid": IMG_HIP, "decision": "своя", "specialist": "лаборант"})
    check(r.status_code == 400 and "roi_box_px" in r.json()["detail"], "POST «своя» без рамки -> 400 с пояснением")
    r = client.post(f"{base}/decisions?t={TOKEN}", json={"image_uid": IMG_SPINE, "decision": "отклонено", "specialist": "врач"})
    check(r.status_code == 400, "POST по позвоночнику -> 400")
    r = client.post(f"{base}/decisions?t={TOKEN}", content=b"not json", headers={"Content-Type": "application/json"})
    check(r.status_code == 400, "POST не-JSON -> 400")
    r = client.get(f"{base}/decisions?t={TOKEN}")
    check(r.status_code == 200 and r.json()["n_decisions"] == 2 and [d["decision"] for d in r.json()["decisions"]] == ["подтверждено", "своя"], "GET список: 2 решения в порядке записи")
    check(all(schema_check.validate_instance(d, dec_schema) == [] for d in r.json()["decisions"]), "записи из GET соответствуют schema/roi_decision.schema.json")
    r = client.get(f"{base}/decisions.csv?t={TOKEN}")
    check(r.status_code == 200 and r.headers["content-type"].startswith("text/csv") and r.text.count("\n") == 3, "GET decisions.csv: заголовок + 2 строки")
    r = client.get(f"{base}/decisions_sr/{STUDY}?t={TOKEN}")
    check(r.status_code == 200 and r.content[128:132] == b"DICM" and r.headers.get("content-type", "").startswith("application/dicom"), "GET decisions_sr: DICOM-файл")
    sr_file = JOB_DIR / api_server.DECISIONS_SR_SUBDIR / fn
    check(sr_file.is_file(), f"SR сохранён в каталоге задачи: {api_server.DECISIONS_SR_SUBDIR}/{fn}")
    check(client.get(f"{base}/decisions_sr/{STUDY}?t={TOKEN}").status_code == 200, "повторный запрос SR отвечает 200")
    rv = subprocess.run([sys.executable, str(ROOT / "tools" / "validate_sr.py"), str(sr_file)], capture_output=True, text=True)
    check(rv.returncode == 0, "tools/validate_sr.py на SR из API: " + (rv.stdout.strip().splitlines()[-1] if rv.stdout.strip() else rv.stderr[-200:]))
    # общий маршрут файлов не пострадал
    check(client.get(f"{base}/summary.json?t={TOKEN}").status_code == 200, "GET summary.json по-прежнему работает")
    check(client.get(f"{base}/decisions.json?t={TOKEN}").status_code == 200, "GET decisions.json как файл задачи (тот же код доступа)")

print(f"\nИТОГ: OK {N_OK}, FAIL {N_FAIL}, SKIP {N_SKIP}")
sys.exit(1 if N_FAIL else 0)
