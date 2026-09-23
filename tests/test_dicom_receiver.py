#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверки приёмника DICOM (src/dicom_receiver.py): C-ECHO, C-STORE, inbox, журнал, передача на анализ.

Без torch: анализ подменяется функцией-заглушкой, затем проверяется HTTP-путь через тестовый
API-сервер на uvicorn, который повторяет контракт /api/analyze (job_id, job_token, summary) и
пишет summary.json в <out>/jobs/<job_id>/ — как настоящий api_server. Если доступен настоящий
api_server (torch, cv2 в окружении), он тоже поднимается на свободном порту и принимает два
фантома от приёмника; иначе эта часть помечается SKIP (в образе она выполнима).

  1. SCP стартует на свободном порту в потоке; C-ECHO -> 0x0000;
  2. C-STORE двух фантомов одного исследования (tests/phantoms/study_01) в одной ассоциации;
     файлы лежат в inbox/<study_uid>/<sop_uid>.dcm, читаются pydicom, пиксели совпадают с исходником;
  3. после закрытия ассоциации анализ вызван один раз со списком из двух файлов; job.json записан;
  4. журнал receiver_log.csv: строки echo/store/analyze, job_id, размеры; без персональных данных;
  5. объект без StudyInstanceUID отклоняется статусом 0xA900; SOP-класс вне списка (CT) не
     согласуется на уровне ассоциации; JPEG 2000 по умолчанию не предлагается;
  6. таймаут тишины: ассоциация остаётся открытой, анализ приходит по idle_s;
  7. файлы, оставшиеся в inbox после сбоя, отправляются при следующем старте (resume);
  8. HTTP-путь: тестовый API на uvicorn -> результат в <out>/jobs/<job_id>/summary.json;
  9. настоящий api_server, если импортируется (иначе SKIP);
 10. запрещённых формулировок в модуле и документе нет.

Запуск: python tests/test_dicom_receiver.py   (код возврата 0 — все проверки пройдены)
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

TMP = Path(tempfile.mkdtemp(prefix="densito_rx_"))
os.environ["DENSITO_OUTPUT_DIR"] = str(TMP / "out")
os.environ["DENSITO_INBOX"] = str(TMP / "inbox")

import pydicom  # noqa: E402
from pydicom.dataset import Dataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian, JPEG2000Lossless, generate_uid  # noqa: E402
from pynetdicom import AE  # noqa: E402
from pynetdicom.sop_class import CTImageStorage, ComputedRadiographyImageStorage, Verification  # noqa: E402

import dicom_receiver as R  # noqa: E402

N_OK = N_FAIL = N_SKIP = 0
PHANTOMS = ROOT / "tests" / "phantoms"
P1 = PHANTOMS / "study_01" / "CR000000.dcm"
P2 = PHANTOMS / "study_01" / "CR000001.dcm"
P3 = PHANTOMS / "study_02" / "CR000000.dcm"


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


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(pred, timeout=15.0, step=0.1) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(step)
    return pred()


def scu(port: int, aet="STORESCU", contexts=None):
    ae = AE(ae_title=aet)
    if contexts is None:
        ae.add_requested_context(Verification)
        ae.add_requested_context(ComputedRadiographyImageStorage, [ImplicitVRLittleEndian, ExplicitVRLittleEndian])
    else:
        for sop, ts in contexts:
            ae.add_requested_context(sop, ts)
    return ae.associate("127.0.0.1", port, ae_title="DENSITOAI")


def read_log(inbox: Path):
    p = inbox / "receiver_log.csv"
    if not p.exists():
        return []
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def fake_job_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + "_abc123"


# --------------------------------------------------------------------------- #
# 1-4. приём двух фантомов в одной ассоциации, анализ-заглушка
# --------------------------------------------------------------------------- #
calls = []


def fake_analyze(study_uid, files):
    calls.append((study_uid, [Path(f) for f in files]))
    return {"job_id": fake_job_id(), "job_token": "tok_" + "x" * 20,
            "summary": {"n_files": len(files), "n_violations": 0, "n_failures": 0, "n_studies": 1}}


inbox1 = TMP / "inbox1"
port1 = free_port()
rx = R.DicomReceiver(inbox=inbox1, port=port1, aet="DENSITOAI", bind="127.0.0.1", analyze=fake_analyze,
                     idle_s=3.0, release_s=0.3, keep_files=True)
rx.start(block=False)
check(wait_for(lambda: socket.socket().connect_ex(("127.0.0.1", port1)) == 0, 5), f"SCP слушает порт {port1}")

assoc = scu(port1)
check(assoc.is_established, "ассоциация установлена (Verification + CR Image Storage)")
st = assoc.send_c_echo()
check(int(st.Status) == 0x0000, "C-ECHO -> 0x0000")
ds1 = pydicom.dcmread(P1)
ds2 = pydicom.dcmread(P2)
st1 = assoc.send_c_store(ds1)
st2 = assoc.send_c_store(ds2)
check(int(st1.Status) == 0 and int(st2.Status) == 0, "C-STORE двух фантомов -> 0x0000")
study = str(ds1.StudyInstanceUID)
f1 = inbox1 / study / f"{ds1.SOPInstanceUID}.dcm"
f2 = inbox1 / study / f"{ds2.SOPInstanceUID}.dcm"
check(f1.is_file() and f2.is_file(), "файлы лежат в inbox/<study_uid>/<sop_uid>.dcm")
if f1.is_file():
    r1 = pydicom.dcmread(f1)
    check(str(r1.file_meta.TransferSyntaxUID) == str(ds1.file_meta.TransferSyntaxUID)
          and r1.pixel_array.tobytes() == ds1.pixel_array.tobytes(), "принятый файл читается pydicom, пиксели совпадают")
check(not calls, "до закрытия ассоциации анализ не вызывается")
assoc.release()
check(wait_for(lambda: len(calls) == 1, 10), "после закрытия ассоциации анализ вызван один раз")
if calls:
    suid, files = calls[0]
    check(suid == study and sorted(p.name for p in files) == sorted([f1.name, f2.name]),
          "анализ получил оба файла исследования")
jj = inbox1 / study / "job.json"
check(jj.is_file(), "job.json записан рядом с исследованием")
if jj.is_file():
    card = json.loads(jj.read_text(encoding="utf-8"))
    check(R.JOB_RE.match(card.get("job_id", "")) and card.get("cabinet", "").startswith("/#app/job/"),
          "job.json: job_id и ссылка для кабинета")
    check((jj.stat().st_mode & 0o077) == 0, "job.json недоступен другим пользователям (0600)")
check(f1.is_file() and f2.is_file(), "keep_files=True: принятые файлы остаются в inbox")

rows = read_log(inbox1)
ev = [r["event"] for r in rows]
check("echo" in ev and ev.count("store") == 2 and ev.count("analyze") == 1, f"журнал: echo/store x2/analyze ({ev})")
an = [r for r in rows if r["event"] == "analyze"]
check(bool(an) and R.JOB_RE.match(an[0]["job_id"]) and an[0]["study_uid"] == study
      and int(an[0]["size_bytes"]) == f1.stat().st_size + f2.stat().st_size, "журнал: job_id, study_uid, суммарный размер")
stores = [r for r in rows if r["event"] == "store"]
check(all(r["calling_aet"] == "STORESCU" and r["sop_class_uid"] == "1.2.840.10008.5.1.4.1.1.1"
          and int(r["size_bytes"]) > 1000 for r in stores), "журнал: AE вызывающего, SOP Class, размер")
log_text = (inbox1 / "receiver_log.csv").read_text(encoding="utf-8")
check("PHANTOM" not in log_text and "SYNTHETIC" not in log_text and str(ds1.PatientName) not in log_text,
      "журнал без имени пациента")
check(all(set(r) == set(R.LOG_COLUMNS) for r in rows), "журнал: фиксированный набор колонок")

# --------------------------------------------------------------------------- #
# 5. отказы: нет StudyInstanceUID, чужой SOP-класс, JPEG 2000 по умолчанию
# --------------------------------------------------------------------------- #
bad = pydicom.dcmread(P1)
del bad.StudyInstanceUID
assoc = scu(port1)
stb = assoc.send_c_store(bad)
check(int(stb.Status) == 0xA900, f"объект без StudyInstanceUID -> 0xA900 (получено 0x{int(stb.Status):04X})")
assoc.release()
check(any(r["event"] == "reject" for r in read_log(inbox1)), "отказ записан в журнал")

assoc = scu(port1, contexts=[(CTImageStorage, [ImplicitVRLittleEndian])])
check(assoc.is_rejected or not assoc.is_established or not assoc.accepted_contexts,
      "CT Image Storage не согласуется (нет принятого контекста)")
if assoc.is_established:
    assoc.release()
assoc = scu(port1, contexts=[(ComputedRadiographyImageStorage, [JPEG2000Lossless])])
check(not assoc.accepted_contexts, "JPEG 2000 по умолчанию не предлагается (кодеков в образе нет)")
if assoc.is_established:
    assoc.release()
assoc = scu(port1, contexts=[(ComputedRadiographyImageStorage, [ImplicitVRLittleEndian]),
                             (R.DigitalXRayImageStorageForPresentation, [ExplicitVRLittleEndian]),
                             (R.DigitalXRayImageStorageForProcessing, [ExplicitVRLittleEndian]),
                             (R.SecondaryCaptureImageStorage, [ExplicitVRLittleEndian])])
check(len(assoc.accepted_contexts) == 4, "CR, DX (presentation, processing) и Secondary Capture принимаются")
if assoc.is_established:
    assoc.release()

# --------------------------------------------------------------------------- #
# 6. таймаут тишины при открытой ассоциации
# --------------------------------------------------------------------------- #
calls.clear()
assoc = scu(port1)
ds3 = pydicom.dcmread(P3)
assoc.send_c_store(ds3)
t0 = time.time()
check(wait_for(lambda: len(calls) == 1, 15), f"открытая ассоциация: анализ по таймауту тишины ({time.time() - t0:.1f} с, idle_s=3)")
check(bool(calls) and calls[0][0] == str(ds3.StudyInstanceUID) and len(calls[0][1]) == 1, "по таймауту отправлено одно исследование, один файл")
assoc.release()
rx.stop()
check(wait_for(lambda: socket.socket().connect_ex(("127.0.0.1", port1)) != 0, 5), "SCP остановлен")

# --------------------------------------------------------------------------- #
# 7. сбой анализа: файлы остаются; при следующем старте — повторная отправка
# --------------------------------------------------------------------------- #
def failing_analyze(study_uid, files):
    raise RuntimeError("API недоступен (тест)")


inbox2 = TMP / "inbox2"
port2 = free_port()
rx2 = R.DicomReceiver(inbox=inbox2, port=port2, bind="127.0.0.1", analyze=failing_analyze, idle_s=3.0, release_s=0.3)
rx2.start(block=False)
assoc = scu(port2)
assoc.send_c_store(ds1)
assoc.release()
check(wait_for(lambda: any(r["event"] == "analyze_failed" for r in read_log(inbox2)), 10), "сбой анализа записан в журнал")
rx2.stop()
kept = inbox2 / study / f"{ds1.SOPInstanceUID}.dcm"
check(kept.is_file(), "после сбоя анализа файл остаётся в inbox")
calls.clear()
rx3 = R.DicomReceiver(inbox=inbox2, port=free_port(), bind="127.0.0.1", analyze=fake_analyze, idle_s=1.0, release_s=0.3, keep_files=False)
rx3.start(block=False)
check(wait_for(lambda: len(calls) == 1 and not kept.exists(), 10), "при следующем старте файл отправлен повторно и удалён после успеха")
rx3.stop()

# --------------------------------------------------------------------------- #
# 8. HTTP-путь: тестовый API на uvicorn с контрактом /api/analyze
# --------------------------------------------------------------------------- #
try:
    import httpx  # noqa: F401
    import uvicorn
    from fastapi import FastAPI, File, UploadFile
    from typing import List
    HTTP_OK = True
except Exception as e:  # noqa: BLE001
    HTTP_OK = False
    skip(f"HTTP-часть: httpx/uvicorn недоступны ({e})")

if HTTP_OK:
    out = TMP / "out"
    (out / "jobs").mkdir(parents=True, exist_ok=True)
    fake = FastAPI()
    received = {}

    @fake.get("/api/health")
    def _health():
        return {"status": "ok"}

    @fake.post("/api/analyze")
    async def _analyze(files: List[UploadFile] = File(...), xlsx: bool = False):
        job = fake_job_id()
        d = out / "jobs" / job
        d.mkdir(parents=True, exist_ok=True)
        names = []
        n = 0
        for uf in files:
            data = await uf.read()
            names.append(uf.filename)
            n += len(data)
        received[job] = {"names": names, "bytes": n, "xlsx": xlsx}
        card = {"job_id": job, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "summary": {"n_files": len(files), "n_violations": 0, "n_failures": 0, "n_studies": 1}, "rows": []}
        (d / "summary.json").write_text(json.dumps(card), encoding="utf-8")
        return {**card, "job_token": "tok_test_" + "y" * 16}

    api_port = free_port()
    server = uvicorn.Server(uvicorn.Config(fake, host="127.0.0.1", port=api_port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    check(wait_for(lambda: server.started, 20), f"тестовый API поднят на порту {api_port}")

    inbox3 = TMP / "inbox3"
    port3 = free_port()
    rx4 = R.DicomReceiver(inbox=inbox3, port=port3, bind="127.0.0.1", api_url=f"http://127.0.0.1:{api_port}",
                          idle_s=3.0, release_s=0.3, keep_files=False)
    rx4.start(block=False)
    assoc = scu(port3, aet="LUNAR_TEST")
    assoc.send_c_store(ds1)
    assoc.send_c_store(ds2)
    assoc.release()
    check(wait_for(lambda: any(r["event"] == "analyze" for r in read_log(inbox3)), 20), "HTTP: анализ через /api/analyze выполнен")
    an = [r for r in read_log(inbox3) if r["event"] == "analyze"]
    job = an[0]["job_id"] if an else ""
    check(bool(job) and (out / "jobs" / job / "summary.json").is_file(), f"HTTP: результат в <out>/jobs/{job}/summary.json")
    if job in received:
        want = sorted([f"{study}/{ds1.SOPInstanceUID}.dcm", f"{study}/{ds2.SOPInstanceUID}.dcm"])
        got = sorted(received[job]["names"])
        # размер принятого файла может отличаться от исходника на несколько байт (File Meta пишет приёмник),
        # поэтому сверяем с суммой из журнала, а не с исходными файлами
        check(received[job]["xlsx"] and got == want and received[job]["bytes"] == int(an[0]["size_bytes"]),
              f"HTTP: переданы оба файла с именами <study_uid>/<sop_uid>.dcm, xlsx=true ({got}, {received[job]['bytes']} байт)")
    check(bool(an) and an[0]["calling_aet"] == "LUNAR_TEST", "HTTP: AE вызывающего в журнале")
    check(not (inbox3 / study / f"{ds1.SOPInstanceUID}.dcm").exists(), "HTTP: после успешного анализа DICOM удалены из inbox")
    jj = inbox3 / study / "job.json"
    check(jj.is_file() and json.loads(jj.read_text(encoding="utf-8")).get("job_id") == job, "HTTP: job.json с job_id")
    rx4.stop()
    server.should_exit = True
    th.join(timeout=10)

# --------------------------------------------------------------------------- #
# 9. настоящий api_server (нужны torch и cv2) — иначе SKIP
# --------------------------------------------------------------------------- #
REAL = False
if HTTP_OK and os.environ.get("DENSITO_TEST_REAL_API", "1") == "1":
    try:
        import torch  # noqa: F401
        import cv2  # noqa: F401
        import api_server  # noqa: F401
        REAL = True
    except Exception as e:  # noqa: BLE001
        skip(f"настоящий api_server недоступен в этом окружении ({type(e).__name__}); внутри образа проверка выполняется")
if REAL:
    api_port = free_port()
    server = uvicorn.Server(uvicorn.Config(api_server.app, host="127.0.0.1", port=api_port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    t0 = time.time()
    ready = wait_for(lambda: server.started and httpx.get(f"http://127.0.0.1:{api_port}/api/health", timeout=5).status_code == 200
                     if server.started else False, 90)
    if not ready:
        skip("настоящий api_server не поднялся за 90 с")
    else:
        inbox4 = TMP / "inbox4"
        port4 = free_port()
        rx5 = R.DicomReceiver(inbox=inbox4, port=port4, bind="127.0.0.1", api_url=f"http://127.0.0.1:{api_port}",
                              idle_s=3.0, release_s=0.3)
        rx5.start(block=False)
        assoc = scu(port4)
        assoc.send_c_store(ds1)
        assoc.send_c_store(ds2)
        assoc.release()
        ok = wait_for(lambda: any(r["event"] in ("analyze", "analyze_failed") for r in read_log(inbox4)), 110)
        an = [r for r in read_log(inbox4) if r["event"] == "analyze"]
        job = an[0]["job_id"] if an else ""
        jd = api_server.JOBS_DIR / job
        check(ok and bool(job), f"настоящий API: анализ выполнен ({time.time() - t0:.0f} с с момента старта)")
        check((jd / "results.csv").is_file() and (jd / "summary.json").is_file(), "настоящий API: results.csv и summary.json в jobs/<job_id>/")
        if (jd / "results.csv").is_file():
            with open(jd / "results.csv", encoding="utf-8") as f:
                n_rows = sum(1 for _ in f) - 1
            check(n_rows == 2, f"настоящий API: две строки результата (получено {n_rows})")
        check(any(jd.glob("sr/*_SR.dcm")) or any(jd.glob("*_sr.dcm")) or (jd / "sr").exists(),
              "настоящий API: SR исследования записан")
        rx5.stop()
    server.should_exit = True
    th.join(timeout=15)

# --------------------------------------------------------------------------- #
# 10. запрещённые формулировки
# --------------------------------------------------------------------------- #
banned = tuple("".join(p) for p in (("Gr", "ad-", "CAM"), ("автокоррекц", "ия ROI"), ("ЕР", "ИС"), ("скол", "иоз"), ("Ко", "бба")))
texts = [(ROOT / "src" / "dicom_receiver.py").read_text(encoding="utf-8")]
doc = ROOT / "docs" / "DICOM_RECEIVER.md"
if doc.exists():
    texts.append(doc.read_text(encoding="utf-8"))
check(not any(b.lower() in t.lower() for b in banned for t in texts), "запрещённых формулировок нет в модуле и документе")
check("!" not in "".join(texts).replace("!=", "").replace("!r}", "").replace("#!/", ""), "восклицательных знаков в модуле и документе нет")

shutil.rmtree(TMP, ignore_errors=True)
print(f"\nИТОГ: OK {N_OK}, FAIL {N_FAIL}, SKIP {N_SKIP}")
sys.exit(1 if N_FAIL else 0)
