#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тесты API (шаг 2 плана): изоляция результатов по запросу, защита от обхода путей,
лимиты загрузки, история запросов. Запуск: python tests/test_api_isolation.py
(модели загружаются один раз; ~1–2 мин на CPU).
"""
import json, os
import re
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(tempfile.mkdtemp(prefix="densito_api_test_"))
os.environ["DENSITO_OUTPUT_DIR"] = str(OUT)
os.environ["DENSITO_MAX_FILES"] = "3"
os.environ["DENSITO_MAX_UPLOAD_MB"] = "1"
os.environ["DENSITO_ADMIN_KEY"] = ADMIN_KEY = "test-admin-key"
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402
import api_server  # noqa: E402

# образец организаторов, если лежит рядом (в репозитории его нет), иначе синтетические фантомы tests/phantoms
SAMPLES = sorted((ROOT / "tests" / "sample_test_zip" / "Для теста").glob("*.dcm")) or \
    sorted((ROOT / "tests" / "phantoms" / "study_01").glob("*.dcm"))
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
if "study_sr_download" in row:
    check(client.get(row["study_sr_download"]).status_code == 200, "SR исследования скачивается по ссылке")
if "bonus_overlay_dcm_download" in row:
    check(client.get(row["bonus_overlay_dcm_download"]).status_code == 200,
          "серия с визуализацией (DICOM SC) скачивается по ссылке")
b1 = OUT / "jobs" / j1["job_id"] / "bonus"
check(b1.exists() and any(b1.iterdir()), "бонус-файлы перенесены в папку запроса")
check(not any((OUT / "bonus" / "viz").iterdir()), "общая папка viz пуста после запроса (нет гонок)")

# 2. скачивание файлов запроса — только по коду доступа (job_token)
TOK1 = j1["job_token"]
for name in ("results.csv", "results_debug.csv", "summary.json"):
    check(client.get(f"/api/results/{j1['job_id']}/{name}", params={"t": TOK1}).status_code == 200,
          f"скачивание {name} по коду доступа")
    check(client.get(f"/api/results/{j1['job_id']}/{name}",
                     headers={"X-Job-Token": TOK1}).status_code == 200, f"скачивание {name} по заголовку")
    check(client.get(f"/api/results/{j1['job_id']}/{name}").status_code == 403,
          f"{name} без кода доступа → 403")
check(client.get(f"/api/results/{j1['job_id']}/results.csv", params={"t": "wrong"}).status_code == 403,
      "неверный код доступа → 403")
check(client.get(f"/api/results/{j1['job_id']}/results.csv",
                 params={"t": j2["job_token"]}).status_code == 403, "код другого запроса не подходит")
check(client.get(f"/api/results/{j1['job_id']}/nope.csv", params={"t": TOK1}).status_code == 404,
      "неизвестное имя → 404")

# 3. обход путей
for bad in ("../../etc/passwd", "..%2F..%2Fetc%2Fpasswd", "%2e%2e/api_server.log", ".hidden",
            ".job_token"):
    rr = client.get(f"/api/results/{j1['job_id']}/{bad}", params={"t": TOK1})
    check(rr.status_code in (404, 400, 422), f"traversal '{bad}' отклонён с кодом доступа ({rr.status_code})")
    rr = client.get(f"/api/results/{j1['job_id']}/{bad}")
    check(rr.status_code in (403, 404, 400, 422), f"traversal '{bad}' отклонён и без кода ({rr.status_code})")
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

# 6. история: общий список закрыт админским ключом, карточка — кодом доступа запроса
check(client.get("/api/jobs").status_code == 403, "список запросов без ключа → 403")
rl = client.get("/api/jobs", headers={"X-Admin-Key": ADMIN_KEY})
check(rl.status_code == 200, f"список запросов по админскому ключу ({rl.status_code})")
if rl.status_code == 200:
    h = rl.json()
    check(len(h["jobs"]) >= 3 and h["jobs"][0]["job_id"] >= h["jobs"][-1]["job_id"],
          "история: список, новые первыми")
check(client.get(f"/api/jobs/{j1['job_id']}").status_code == 403, "карточка без кода доступа → 403")
rc = client.get(f"/api/jobs/{j1['job_id']}", params={"t": TOK1})
check(rc.status_code == 200, f"карточка запроса по коду доступа ({rc.status_code})")
if rc.status_code == 200:
    c = rc.json()
    check(c["job_id"] == j1["job_id"] and c["rows"] and "bonus_overlay_png_base64" not in c["rows"][0],
          "карточка запроса без base64")
    links = json.dumps(c, ensure_ascii=False)
    check("?t=" in links, "ссылки в карточке уже с кодом доступа")

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

# 9. веб-интерфейс раздаётся самим сервисом (требование локальной работы: кабинет едет в образе)
r = client.get("/")
check(r.status_code == 200 and "DensitoAI" in r.text, f"корень отдаёт кабинет ({r.status_code})")
check("text/html" in r.headers.get("content-type", ""), "корень отдаёт HTML")
for asset in ("actions.json", "demo_result.json", "fonts/NotoSans-Regular.woff"):
    r = client.get(f"/assets/{asset}")
    check(r.status_code == 200 and len(r.content) > 0, f"статика /assets/{asset} отдаётся ({r.status_code})")
# страница не должна тянуть ничего из интернета — иначе она не откроется у заказчика без сети
html = client.get("/").text
TEAM_LINKS = ("https://densito.ru/lct/", "https://densito.ru/review/", "https://github.com/neuropeopleteam-afk/densitoai")
external = re.findall(r'src="(https?://[^"]+)"', html) + [
    u for u in re.findall(r'href="(https?://[^"]+)"', html) if not u.startswith(TEAM_LINKS)]
check(not external, f"в кабинете нет внешних src; внешние ссылки — только на стенд и GitHub команды (найдено: {external[:3]})")
# в демо-данных не должно быть настоящих идентификаторов исследований заказчика
demo = json.loads((Path(os.environ.get("DENSITO_WEB_DIR", ROOT / "web")) / "assets" / "demo_result.json").read_text(encoding="utf-8"))
blob = json.dumps(demo, ensure_ascii=False)
# публичные UID стандарта DICOM (корень 1.2.840.10008, например SOP Class CR Image Storage) — не идентификаторы заказчика
real_uids = [u for u in re.findall(r"1\.2\.(?:840|643)[0-9.]{10,}", blob) if not u.startswith("1.2.840.10008.")]
check(not real_uids, f"в демо-данных нет настоящих DICOM UID (найдено {len(real_uids)})")
check(all(r.get("demo_role") for r in demo["rows"]), "каждый кадр демо-партии подписан ролью")
check(sum(1 for r in demo["rows"] if r.get("quality_class") == 1) >= 3,
      "демо-партия показывает найденные нарушения, а не только норму")

# обход каталога закрыт
for bad in ("../config.yaml", "../../config.yaml", "%2e%2e%2fconfig.yaml", "/etc/passwd"):
    r = client.get(f"/assets/{bad}")
    check(r.status_code == 404, f"/assets/{bad} → 404 ({r.status_code})")

# --- Бонус-файлы: только свои, без совпадений по имени входного файла -------------------------
# Регрессия на дефект 2.3.1: файлы собирались по шаблону «<имя входного файла>_overlay.png», и в
# папку запроса попадали одноимённые файлы ПРОШЛЫХ прогонов, а файлы текущего прогона оставались
# в общей папке движка. Теперь пути берутся из debug-строк, а имена — по номеру строки.
def test_bonus_isolation():
    decoy_dir = api_server.SR_DIR
    decoy_dir.mkdir(parents=True, exist_ok=True)
    stem = SAMPLES[0].stem
    decoy = decoy_dir / f"{stem}_sr.dcm"
    decoy.write_bytes(b"DECOY-NOT-FOR-THIS-REQUEST")
    r = upload(SAMPLES[:1])
    check(r.status_code == 200, f"бонус-изоляция: запрос принят (код {r.status_code})")
    if r.status_code != 200:
        return
    job = r.json()["job_id"]
    bdir = api_server.JOBS_DIR / job / "bonus"
    names = sorted(x.name for x in bdir.glob("*")) if bdir.exists() else []
    check(all(n.startswith("row") for n in names),
          f"бонус-изоляция: в папке запроса только файлы этого запроса ({names})")
    check(not (bdir / decoy.name).exists(), "бонус-изоляция: чужой одноимённый файл не перенесён")
    check(decoy.exists(), "бонус-изоляция: приманка осталась в общей папке движка")
    row = r.json()["rows"][0]
    check(bool(row.get("bonus_overlay_png_base64")), "бонус-изоляция: оверлей строки приложен к ответу")
    decoy.unlink(missing_ok=True)


test_bonus_isolation()


# ---------------------------------------------------------------------------
# Отказ по неподдерживаемой области (слой API; пакетный путь не затронут)
# ---------------------------------------------------------------------------
import pydicom  # noqa: E402

def _with_tag(src_path, **tags):
    ds = pydicom.dcmread(str(src_path), force=True)
    for k, v in tags.items():
        setattr(ds, k, v)
    out = Path(tempfile.mkdtemp(prefix="densito_region_")) / src_path.name
    try:
        ds.save_as(str(out))
    except Exception:  # разные версии pydicom требуют разных флагов записи
        try:
            ds.save_as(str(out), enforce_file_format=False)
        except TypeError:
            ds.save_as(str(out), write_like_original=True)
    return out

try:
    fake = _with_tag(SAMPLES[0], BodyPartExamined="FOREARM")
    rr = upload([fake])
    check(rr.status_code == 200, f"снимок предплечья принят без ошибки ({rr.status_code})")
    jr = rr.json()
    rowf = jr["rows"][0]
    check(rowf.get("region_supported") is False, "region_supported=false для предплечья")
    check("сервис оценивает только" in str(rowf.get("region_support_reason", "")),
          "причина отказа названа человеческим текстом")
    check(int(rowf.get("quality_class", -1)) == 0, "класс качества обнулён при отказе")
    check(str(rowf.get("violation_type", "x")) == "", "тип нарушения пуст при отказе")
    check(not [k for k in rowf if str(k).startswith("bonus_")],
          "бонус-файлы не прикладываются при отказе")
    csv_text = jr["csv"]
    check("Failure" in csv_text, "в отчёте этого запроса стоит статус отказа")
    dl = client.get(jr["result_csv_url"])
    check(dl.status_code == 200 and "Failure" in dl.text,
          "скачиваемый отчёт согласован с карточкой")
    check(len(dl.text.strip().splitlines()[0].split(",")) == 9,
          "в отчёте по-прежнему 9 колонок")
except Exception as e:  # noqa: BLE001
    check(False, f"проверка отказа по области не выполнена: {e}")

# нормальный снимок из выборки — поддерживается
rn = upload([SAMPLES[0]])
check(rn.status_code == 200, "обычный снимок принят")
check(rn.json()["rows"][0].get("region_supported") is True,
      "region_supported=true для снимка из выборки")

# ---------------------------------------------------------------------------
# Приём результатов слепой ревизии
# ---------------------------------------------------------------------------
payload = {"reviewer": "тест", "kit_sha256": "0" * 64,
           "answers": [{"idx": 1, "file": "a.png", "verdict": "ok", "ms": 1200}],
           "phrases": [{"id": "p1", "text": "фраза", "acceptable": True}],
           "free_text": {"unclear": "нет"}}
rv = client.post("/api/review", json=payload)
check(rv.status_code == 200 and rv.json().get("ok") is True, f"ревизия принята ({rv.status_code})")
saved = OUT / "review" / rv.json().get("saved", "нет")
check(saved.exists(), "файл ревизии сохранён на диск")
check(json.loads(saved.read_text(encoding="utf-8"))["reviewer"] == "тест",
      "содержимое ревизии сохранено без потерь")
rv2 = client.post("/api/review", content="не json".encode("utf-8"))
check(rv2.status_code == 400, f"мусор вместо JSON → 400 ({rv2.status_code})")
rv3 = client.post("/api/review", content=b'{"a":"' + b"x" * (2 * 1024 * 1024 + 10) + b'"}')
check(rv3.status_code == 413, f"слишком большой ответ → 413 ({rv3.status_code})")
rv4 = client.post("/api/review", json=[1, 2, 3])
check(rv4.status_code == 400, f"массив вместо объекта → 400 ({rv4.status_code})")

print("\nИТОГ:", "все проверки пройдены" if not fails else f"{len(fails)} провалов: {fails}")
sys.exit(1 if fails else 0)
