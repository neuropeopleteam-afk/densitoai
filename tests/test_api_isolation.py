#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тесты API (шаг 2 плана): изоляция результатов по запросу, защита от обхода путей,
лимиты загрузки, история запросов. Запуск: python tests/test_api_isolation.py
(модели загружаются один раз; ~1–2 мин на CPU).
"""
import json, os
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(tempfile.mkdtemp(prefix="densito_api_test_"))
os.environ["DENSITO_OUTPUT_DIR"] = str(OUT)
os.environ["DENSITO_MAX_FILES"] = "3"
os.environ["DENSITO_MAX_UPLOAD_MB"] = "1"
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402
import api_server  # noqa: E402

SAMPLES = sorted((ROOT / "tests" / "sample_test_zip" / "Для теста").glob("*.dcm"))
assert SAMPLES, "нет тестовых DICOM"

client = TestClient(api_server.app)
fails = []


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def upload(paths, **params):
    files = [("files", (p.name, p.read_bytes(), "application/dicom")) for p in paths]
    return client.post("/api/analyze", files=files, params=params)


# 1. два запроса с ОДНИМ И ТЕМ ЖЕ файлом → разные job_id, свои папки, свои бонус-файлы
r1 = upload([SAMPLES[0]])
r2 = upload([SAMPLES[0]])
check(r1.status_code == 200 and r2.status_code == 200, f"analyze 200/200 ({r1.status_code}/{r2.status_code})")
j1, j2 = r1.json(), r2.json()
check(j1["job_id"] != j2["job_id"], "разные job_id")
check((OUT / "jobs" / j1["job_id"] / "results.csv").exists(), "results.csv в папке первого запроса")
check((OUT / "jobs" / j2["job_id"] / "results.csv").exists(), "results.csv в папке второго запроса")
check(j1["result_csv_url"].startswith(f"/api/results/{j1['job_id']}/"), "result_csv_url ведёт в свою папку")
row = j1["rows"][0]
if "bonus_sr_dcm_download" in row:
    check(row["bonus_sr_dcm_download"].startswith(f"/api/results/{j1['job_id']}/"), "ссылка на SR внутри job")
    sr2 = j2["rows"][0].get("bonus_sr_dcm_download", "")
    check(sr2.startswith(f"/api/results/{j2['job_id']}/"), "SR второго запроса — в его папке")
    check(client.get(row["bonus_sr_dcm_download"]).status_code == 200, "SR скачивается по своей ссылке")
    # чужой job не видит файл
    other = row["bonus_sr_dcm_download"].replace(j1["job_id"], "20000101_000000_ffffff")
    check(client.get(other).status_code == 404, "несуществующий job → 404")
b1 = OUT / "jobs" / j1["job_id"] / "bonus"
check(b1.exists() and any(b1.iterdir()), "бонус-файлы перенесены в папку запроса")
check(not any((OUT / "bonus" / "viz").iterdir()), "общая папка viz пуста после запроса (нет гонок)")

# 2. скачивание файлов запроса
for name in ("results.csv", "results_debug.csv", "summary.json"):
    check(client.get(f"/api/results/{j1['job_id']}/{name}").status_code == 200, f"скачивание {name}")
check(client.get(f"/api/results/{j1['job_id']}/nope.csv").status_code == 404, "неизвестное имя → 404")

# 3. обход путей
for bad in ("../../etc/passwd", "..%2F..%2Fetc%2Fpasswd", "%2e%2e/api_server.log", ".hidden"):
    rr = client.get(f"/api/results/{j1['job_id']}/{bad}")
    check(rr.status_code in (404, 400, 422), f"traversal '{bad}' отклонён ({rr.status_code})")
check(client.get("/api/results/api_server.log").status_code == 404, "лог сервера не отдаётся через legacy-роут")
check(client.get("/api/results/..%2Fconfig.yaml").status_code in (404, 400, 422), "legacy traversal отклонён")
check(client.get("/api/results/bad-job/results.csv").status_code == 404, "job с неверным форматом → 404")

# 4. лимиты
r = upload(SAMPLES[:3] + SAMPLES[:1]) if len(SAMPLES) >= 3 else upload(SAMPLES * 4)
check(r.status_code == 413, f"больше MAX_FILES файлов → 413 ({r.status_code})")
big = [("files", ("big.dcm", b"\0" * (2 * 1024 * 1024), "application/dicom"))]
r = client.post("/api/analyze", files=big)
check(r.status_code == 413, f"больше MAX_UPLOAD_MB → 413 ({r.status_code})")
r = client.post("/api/analyze", files=[("files", ("empty.dcm", b"", "application/dicom"))])
check(r.status_code == 400, f"пустой файл → 400 ({r.status_code})")
check(len([d for d in (OUT / "jobs").iterdir() if d.is_dir()]) == 2, "отклонённые запросы не оставляют папок")

# 5. мусор вместо DICOM → 200 с processing_status = failure, не 500
r = client.post("/api/analyze", files=[("files", ("junk.dcm", b"not a dicom" * 100, "application/dicom"))])
check(r.status_code == 200, f"мусорный файл обрабатывается без 500 ({r.status_code})")
if r.status_code == 200:
    check(r.json()["summary"]["n_failures"] == 1, "мусорный файл помечен как ошибка обработки")

# 5b. битый zip → 400 с понятным русским сообщением, не 500
r = client.post("/api/analyze", files=[("files", ("broken.zip", b"PK\x03\x04" + b"\x00" * 500, "application/zip"))])
check(r.status_code == 200, f"битый zip → 200 со строкой Failure, не 500 ({r.status_code})")
if r.status_code == 200:
    rows_bz = r.json()["rows"]
    check(len(rows_bz) == 1 and rows_bz[0]["processing_status"] != "Success", "битый zip помечен как Failure")
    check("повреждён" in json.dumps(r.json(), ensure_ascii=False), "битый zip: понятное сообщение об ошибке")

# 6. история
h = client.get("/api/jobs").json()
check(len(h["jobs"]) >= 3 and h["jobs"][0]["job_id"] >= h["jobs"][-1]["job_id"], "история: список, новые первыми")
c = client.get(f"/api/jobs/{j1['job_id']}").json()
check(c["job_id"] == j1["job_id"] and c["rows"] and "bonus_overlay_png_base64" not in c["rows"][0],
      "карточка запроса без base64")

# 7. параллельные запросы с одинаковыми именами файлов — ответы не перемешиваются
res = {}
def worker(i, p):
    res[i] = upload([p]).json()
ths = [threading.Thread(target=worker, args=(i, SAMPLES[i % len(SAMPLES)])) for i in range(4)]
[t.start() for t in ths]; [t.join() for t in ths]
ids = {v["job_id"] for v in res.values()}
check(len(ids) == 4, "4 параллельных запроса → 4 разных job_id")
ok_rows = all(Path(v["rows"][0]["path_to_study"]).name == SAMPLES[i % len(SAMPLES)].name for i, v in res.items())
check(ok_rows, "каждый ответ содержит свой файл")

# 8. /api/batch — output_csv вне OUTPUT_DIR отклоняется
r = client.post("/api/batch", json={"input_dir": str(SAMPLES[0].parent), "output_csv": "/tmp/evil.csv"})
check(r.status_code == 400, f"batch output_csv вне OUTPUT_DIR → 400 ({r.status_code})")

print("\nИТОГ:", "все проверки пройдены" if not fails else f"{len(fails)} провалов: {fails}")
sys.exit(1 if fails else 0)
