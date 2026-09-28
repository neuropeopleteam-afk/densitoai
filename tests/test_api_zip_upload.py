#!/usr/bin/env python3
"""Загрузка zip через API (2.4.1): Б6, P2 и архив дополнительных серий (ТЗ п. 2.7).

Б6. Файлы из zip распаковываются движком во временные каталоги densito_in_*. До 2.4.1 движок удалял их в конце
    DensitoInference.run, и после run API уже не находил файлы: журнал оставался без маски ФИО и даты
    исследования, device_tags.csv не писался, кадры слепой проверки (bonus/rowNNNN_frame.png) не сохранялись.
    Проверяется через FastAPI TestClient:
      * /api/analyze с zip -> в журнале маска ФИО и дата исследования из DICOM, device_tags.csv записан,
        кадры rowNNNN_frame.png есть для каждой строки, каталогов densito_in_* после запроса не осталось;
      * /api/expert/upload с zip -> набор создан, число снимков = числу строк Success поддерживаемой области.
P2. Решения своей слепой проверки не появляются в журнале ни на миг: искусственная ошибка create_set_from_job ->
    решений этой загрузки нет ни в поиске журнала, ни в карточке исследования; параллельное чтение журнала во
    время обработки (фоновый поток + чтение сразу после записи строк) решений не видит.
2.7. additional_series_zip_url в ответе /api/analyze; без кода доступа — 403, чужой/несуществующий запрос — 404,
    с кодом — zip, каждый DICOM внутри читается pydicom, имён пациента в именах файлов нет.

Запуск:  python tests/test_api_zip_upload.py     (код 0 — пройдено)
"""
import csv
import glob
import io
import os
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_TMP_OUT = Path(tempfile.mkdtemp(prefix="densito_zipapi_out_"))
os.environ["DENSITO_OUTPUT_DIR"] = str(_TMP_OUT)
os.environ["DENSITO_REGISTRY_OPEN"] = "1"
os.environ.pop("DENSITO_ADMIN_KEY", None)
os.environ.pop("DENSITO_SERIES_ZIP", None)

import pydicom  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
import api_server as A  # noqa: E402

PH = ROOT / "tests" / "phantoms"
FAILED = []
PATIENT = "Иванова^Мария^Петровна"
MASK = "Иванова М. П."


def check(name, cond, detail=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def zip_of(studies, patient=PATIENT, folder="Исследования пациента") -> bytes:
    """zip с фантомами указанных исследований; ФИО пациента подменяется (проверка маски в журнале)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for st in studies:
            for f in sorted((PH / st).glob("*.dcm")):
                ds = pydicom.dcmread(str(f))
                ds.PatientName = patient
                b = io.BytesIO()
                ds.save_as(b, enforce_file_format=True)
                zf.writestr(f"{folder}/{st}/{f.name}", b.getvalue())
    return buf.getvalue()


def densito_in_dirs() -> set:
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "densito_in_*")))


def main() -> int:
    client = TestClient(A.app)
    doc = "open|doctor|%D0%9F%D0%B5%D1%82%D1%80%D0%BE%D0%B2%D0%B0%20%D0%90%D0%BD%D0%BD%D0%B0%20%D0%A1%D0%B5%D1%80%D0%B3%D0%B5%D0%B5%D0%B2%D0%BD%D0%B0"
    hdr = {"X-Registry-Session": doc}

    print("1. /api/analyze с zip: журнал, device_tags.csv, кадры (Б6)")
    before = densito_in_dirs()
    r = client.post("/api/analyze", files=[("files", ("партия.zip", zip_of(["study_01", "study_03"]), "application/zip"))])
    check("analyze: 200", r.status_code == 200, r.text[:300])
    res = r.json()
    job, tok = res["job_id"], res["job_token"]
    rows = res["rows"]
    job_dir = A.JOBS_DIR / job
    check("analyze: 6 строк", len(rows) == 6, str(len(rows)))
    check("analyze: path_to_study — путь внутри архива",
          all(x["path_to_study"].startswith("партия.zip/Исследования пациента/study_0") for x in rows),
          str([x["path_to_study"] for x in rows][:2]))
    check("ответ: маска ФИО в каждой строке", all(x.get("patient") == MASK for x in rows), str([x.get("patient") for x in rows]))
    check("ответ: дата исследования из DICOM", all(x.get("study_date") in ("01.01.2026", "03.01.2026") for x in rows),
          str([x.get("study_date") for x in rows]))
    found = client.get("/api/registry/studies", params={"q": "Иванова"}, headers=hdr).json()
    uids = {x["study_uid"] for x in rows}
    st = [s for s in found["studies"] if s["study_uid"] in uids]
    check("журнал: поиск по фамилии находит оба исследования", len(st) == 2, str(found)[:300])
    check("журнал: маска ФИО", st and all(s["patient"] == MASK for s in st), str([s.get("patient") for s in st]))
    check("журнал: дата исследования из DICOM", st and all(s["study_date_source"] == "DICOM" for s in st)
          and sorted(s["study_date"] for s in st) == ["01.01.2026", "03.01.2026"], str([(s.get("study_date"), s.get("study_date_source")) for s in st]))
    dt = job_dir / "device_tags.csv"
    check("device_tags.csv записан", dt.is_file())
    if dt.is_file():
        with open(dt, encoding="utf-8") as f:
            dtr = list(csv.DictReader(f))
        check("device_tags.csv: строка на каждый снимок, аппарат и дата", len(dtr) == 6 and all(
            x["manufacturer"] and x["study_date"] for x in dtr), str(dtr[:1]))
    frames = [job_dir / "bonus" / f"row{i:04d}_frame.png" for i in range(len(rows))]
    check("кадры rowNNNN_frame.png сохранены для всех строк", all(p.is_file() and p.stat().st_size > 100 for p in frames),
          str([p.name for p in frames if not p.is_file()]))
    check("каталоги распаковки densito_in_* удалены после запроса", not (densito_in_dirs() - before),
          str(densito_in_dirs() - before))
    check("области поддерживаются (запасная проверка нашла файлы)", all(x.get("region_supported") for x in rows))

    print("2. архив дополнительных серий через API (ТЗ п. 2.7)")
    url = res.get("additional_series_zip_url") or ""
    check("additional_series_zip_url в ответе, подписан кодом", url.startswith(f"/api/results/{job}/additional_series.zip?t="), url)
    check("без кода доступа — 403", client.get(f"/api/results/{job}/additional_series.zip").status_code == 403)
    check("с неверным кодом — 403", client.get(f"/api/results/{job}/additional_series.zip?t=wrong").status_code == 403)
    check("несуществующий запрос — 404", client.get("/api/results/20990101_000000_abcdef/additional_series.zip?t=x").status_code == 404)
    check("битый код запроса — 404", client.get("/api/results/..%2F..%2Fetc/additional_series.zip?t=x").status_code == 404)
    z = client.get(url) if url else None
    if z is None:
        check("архив серий скачивается", False, "ссылки нет")
    else:
        check("с кодом — 200, application/zip", z.status_code == 200 and z.headers.get("content-type", "").startswith("application/zip"),
              f"{z.status_code} {z.headers.get('content-type')}")
    if z is not None and z.status_code == 200 and z.headers.get("content-type", "").startswith("application/zip"):
        zf = zipfile.ZipFile(io.BytesIO(z.content))
        names = [n for n in zf.namelist() if n.endswith(".dcm")]
        n_ok = 0
        mods = {}
        for n in names:
            try:
                ds = pydicom.dcmread(io.BytesIO(zf.read(n)))
                n_ok += 1
                mods[str(ds.Modality)] = mods.get(str(ds.Modality), 0) + 1
            except Exception:  # noqa: BLE001
                pass
        # 6 снимков Success: наложение SC + SEG на каждый, SR на каждое из 2 исследований
        check("в архиве 14 DICOM (6 SC + 6 SEG + 2 SR)", len(names) == 14, f"{len(names)} {mods}")
        check("каждый DICOM читается pydicom", n_ok == len(names), f"{n_ok}/{len(names)}")
        check("виды серий: SR, SEG, наложение", mods.get("SR") == 2 and mods.get("SEG") == 6, str(mods))
        check("в именах нет ФИО и исходных путей", all("Иванова" not in n and "study_0" not in n.split("/")[-1]
                                                      and "Исследования" not in n for n in zf.namelist()), str(zf.namelist()[:3]))
        check("индекс series_index.csv в архиве", "series_index.csv" in zf.namelist())

    print("3. /api/expert/upload с zip (Б6): набор из всех снимков Success поддерживаемой области")
    r = client.post("/api/expert/upload", files=[("files", ("мои.zip", zip_of(["study_01", "study_03"], folder="свои"), "application/zip"))],
                    data={"title": "Проверка zip"}, headers=hdr)
    check("expert/upload: 200", r.status_code == 200, r.text[:300])
    if r.status_code == 200:
        e = r.json()
        with A.REGISTRY._conn() as c:
            jid = c.execute("SELECT params FROM expert_sets WHERE id=?", (e["set_id"],)).fetchone()[0]
            import json as _json
            jid = _json.loads(jid)["job_id"]
            n_ok = c.execute("SELECT COUNT(*) FROM images WHERE job_id=? AND processing_status='Success' AND region_supported=1",
                             (jid,)).fetchone()[0]
        check("число снимков в наборе = строк Success поддерживаемой области", e["n"] == n_ok == 6, f"n={e['n']} success={n_ok}")
        check("в ответе нет решений сервиса", "service" not in str(e) and "Некорректная" not in str(e))

    print("4. P2: ошибка создания набора — решения загрузки не появляются в журнале")
    p2_zip = zip_of(["study_02", "study_04"], patient="Сидорова^Ольга^Ивановна", folder="p2")
    with zipfile.ZipFile(io.BytesIO(p2_zip)) as zf:
        p2_uids = {str(pydicom.dcmread(io.BytesIO(zf.read(n)), stop_before_pixels=True).StudyInstanceUID) for n in zf.namelist()}
    leaks = []
    stop = threading.Event()

    def visible(s) -> bool:
        return (s.get("verdict") != "hidden") or s.get("n_violation") or s.get("violations") or s.get("max_prob") is not None

    def read_journal(tag: str) -> None:
        try:
            for s in A.REGISTRY.search("", limit=200)["studies"]:
                if s["study_uid"] in p2_uids and visible(s):
                    leaks.append((tag, "search", s["study_uid"], s.get("verdict")))
            for uid in p2_uids:
                try:
                    card = A.REGISTRY.study(uid, {"login": "t", "role": "doctor", "name": "t"})
                except KeyError:
                    continue
                if visible(card) or any(im.get("quality_class") or im.get("violation_type") or im.get("card_url")
                                        for im in card["images"]):
                    leaks.append((tag, "card", uid, card.get("verdict")))
        except Exception as ex:  # noqa: BLE001
            leaks.append((tag, "error", repr(ex), ""))

    def poller():
        while not stop.is_set():
            read_journal("poll")
            time.sleep(0.02)

    orig_index = A.REGISTRY.index_rows

    def index_and_read(*a, **kw):  # читатель журнала сразу после записи строк, до создания набора
        n = orig_index(*a, **kw)
        read_journal("after_index")
        return n

    orig_create = A.EXPERT.create_set_from_job

    def broken_create(*a, **kw):
        read_journal("before_create")
        raise ValueError("искусственная ошибка создания набора")

    A.REGISTRY.index_rows = index_and_read
    A.EXPERT.create_set_from_job = broken_create
    jobs_before = {p.name for p in A.JOBS_DIR.iterdir()}
    th = threading.Thread(target=poller, daemon=True)
    th.start()
    try:
        r = client.post("/api/expert/upload", files=[("files", ("p2.zip", p2_zip, "application/zip"))],
                        data={"title": "P2"}, headers=hdr)
    finally:
        stop.set(); th.join(5)
        A.REGISTRY.index_rows = orig_index
        A.EXPERT.create_set_from_job = orig_create
    check("ошибка создания набора -> 400", r.status_code == 400, f"{r.status_code} {r.text[:200]}")
    new_jobs = {p.name for p in A.JOBS_DIR.iterdir()} - jobs_before
    with A.REGISTRY._conn() as c:
        n_idx = sum(c.execute("SELECT COUNT(*) FROM images WHERE job_id=?", (j,)).fetchone()[0] for j in new_jobs)
    check("строки загрузки есть в журнале (для поиска по ФИО), но скрыты", n_idx == 6, f"{n_idx} {new_jobs}")
    read_journal("after")
    check("решений загрузки нет ни в поиске, ни в карточке (в том числе при параллельном чтении)", not leaks, str(leaks[:5]))
    api_search = client.get("/api/registry/studies", params={"q": "Сидорова"}, headers=hdr).json()
    st2 = [s for s in api_search["studies"] if s["study_uid"] in p2_uids]
    check("API журнала: исследования найдены по ФИО и скрыты", len(st2) == 2 and all(s["verdict"] == "hidden" and not s["violations"]
                                                                                    and s["n_violation"] == 0 for s in st2),
          str([(s.get("verdict"), s.get("violations")) for s in st2]))
    viol_search = client.get("/api/registry/studies", params={"violation": "Некорректная укладка"}, headers=hdr).json()
    check("фильтр по нарушению не раскрывает скрытую загрузку", not [s for s in viol_search["studies"] if s["study_uid"] in p2_uids])
    for uid in p2_uids:
        card = client.get(f"/api/registry/studies/{uid}", headers=hdr).json()
        check(f"карточка {uid[-6:]}: без вердикта и ссылки на результат", card.get("verdict") == "hidden" and all(
            not im["quality_class"] and not im["violation_type"] and not im["card_url"] for im in card["images"]),
            str(card)[:200])
    check("скрытая загрузка не попадает в случайную выборку проверки", not (set(new_jobs) & {
        it_job for it_job in _items_jobs()}))

    print("\nИТОГ:", "OK" if not FAILED else f"FAIL ({len(FAILED)}): " + "; ".join(FAILED))
    return 0 if not FAILED else 1


def _items_jobs() -> set:
    """Задачи, попавшие в выборку из журнала (create_set берёт только нескрытые загрузки)."""
    user = A.REGISTRY.check_session("open|doctor|%D0%A5")
    try:
        s = A.EXPERT.create_set(user, 40, seed=1)
    except ValueError:
        return set()
    with A.REGISTRY._conn() as c:
        return {r[0] for r in c.execute("SELECT job_id FROM expert_items WHERE set_id=?", (s["id"],))}


if __name__ == "__main__":
    sys.exit(main())
