#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DensitoAI — приёмник DICOM (Storage SCP, C-STORE / C-ECHO) на pynetdicom.

Снимки приходят в сервис прямо с денситометра или из PACS по протоколу DICOM, без ручной
загрузки через кабинет. Приёмник:

  1. слушает порт (по умолчанию 11112, AE Title DENSITOAI), отвечает на C-ECHO и принимает
     C-STORE для SOP-классов DXA-экспорта (CR, DX for Presentation / for Processing,
     Secondary Capture) в несжатых transfer syntax и RLE — ровно то, что читает пайплайн
     без дополнительных кодеков (docs/TRANSFER_SYNTAX_MATRIX.md);
  2. пишет каждый принятый объект в inbox: <DENSITO_INBOX>/<study_uid>/<sop_uid>.dcm
     (по умолчанию /data/output/inbox); запись атомарная (tmp -> rename);
  3. по завершении ассоциации (через DENSITO_RECEIVER_RELEASE_S, по умолчанию 1 с) или по
     таймауту тишины (DENSITO_RECEIVER_IDLE_S, по умолчанию 5 с) передаёт все снимки
     исследования на анализ тем же путём, что и загрузка через кабинет:
     POST /api/analyze?xlsx=true на DENSITO_API_URL (по умолчанию http://127.0.0.1:8000).
     Результат появляется в /data/output/jobs/<job_id>/ (results.csv, results.xlsx,
     results_debug.csv, summary.json, sr/<study_uid>_SR.dcm, PNG) — как для веб-загрузки;
     карточка доступна через GET /api/jobs/<job_id>?t=<job_token>, список — /api/jobs (админский ключ);
  4. ведёт журнал приёма <inbox>/receiver_log.csv без персональных данных: время, событие,
     AE вызывающего, study_uid, sop_uid, SOP Class, transfer syntax, размер, job_id, статус.

Код запроса и код доступа записываются в <inbox>/<study_uid>/job.json (права 0600) вместе со
ссылкой для кабинета (/#app/job/<job_id>/<job_token>). Принятые DICOM после анализа по умолчанию
остаются в inbox (DENSITO_INBOX_KEEP=1, автоудаление загрузок не выполняется); удаление после успешного
анализа включается только явно: DENSITO_INBOX_KEEP=0 или --no-keep.
При ошибке анализа файлы в любом режиме остаются, и при следующем запуске приёмник отправляет их повторно.

Режим без HTTP: DENSITO_API_URL=inproc — приёмник импортирует api_server и вызывает тот же
маршрут /api/analyze внутри процесса (fastapi.testclient); модели грузятся в этот же процесс.

Запуск:
  python src/dicom_receiver.py                 # переменные окружения ниже
  python src/dicom_receiver.py --port 11112 --aet DENSITOAI --inbox /data/output/inbox
  # в контейнере: docker-entrypoint.sh receiver | api+receiver

Переменные окружения: DENSITO_RECEIVER_PORT, DENSITO_RECEIVER_AET, DENSITO_INBOX, DENSITO_API_URL,
DENSITO_RECEIVER_IDLE_S, DENSITO_RECEIVER_RELEASE_S, DENSITO_RECEIVER_ALLOWED_AET (список через
запятую; пусто — любой), DENSITO_RECEIVER_ALL_TS=1 (принимать все transfer syntax, в том числе
JPEG — потребуются кодеки pylibjpeg), DENSITO_INBOX_KEEP (1 — хранить принятые DICOM, по умолчанию;
0 — удалять после успешного анализа), DENSITO_RECEIVER_BIND (0.0.0.0).

Ограничения: без TLS — только внутренняя сеть отделения; не медицинское изделие; принимаются
объекты только перечисленных SOP-классов; C-FIND/C-MOVE не реализованы.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

import pydicom  # noqa: E402
from pydicom.dataset import Dataset  # noqa: E402
from pydicom.uid import (  # noqa: E402
    ExplicitVRBigEndian, ExplicitVRLittleEndian, ImplicitVRLittleEndian, RLELossless,
)

from pynetdicom import AE, ALL_TRANSFER_SYNTAXES, evt  # noqa: E402
from pynetdicom.sop_class import (  # noqa: E402
    ComputedRadiographyImageStorage,
    DigitalXRayImageStorageForPresentation,
    DigitalXRayImageStorageForProcessing,
    SecondaryCaptureImageStorage,
    Verification,
)

__version__ = "1.0.0"
LOG = logging.getLogger("densito.receiver")

# --------------------------------------------------------------------------- #
# Настройки
# --------------------------------------------------------------------------- #
DEFAULT_PORT = int(os.environ.get("DENSITO_RECEIVER_PORT", "11112"))
DEFAULT_AET = os.environ.get("DENSITO_RECEIVER_AET", "DENSITOAI")
DEFAULT_BIND = os.environ.get("DENSITO_RECEIVER_BIND", "0.0.0.0")
_OUT_ROOT = Path(os.environ.get("DENSITO_OUTPUT_DIR", "/data/output"))
DEFAULT_INBOX = Path(os.environ.get("DENSITO_INBOX") or (_OUT_ROOT / "inbox"))
DEFAULT_API_URL = os.environ.get("DENSITO_API_URL", "http://127.0.0.1:8000")
DEFAULT_IDLE_S = float(os.environ.get("DENSITO_RECEIVER_IDLE_S", "5"))
DEFAULT_RELEASE_S = float(os.environ.get("DENSITO_RECEIVER_RELEASE_S", "1"))
DEFAULT_KEEP = os.environ.get("DENSITO_INBOX_KEEP", "1") != "0"  # по умолчанию файлы хранятся; 0 — удалять после анализа
DEFAULT_ALL_TS = os.environ.get("DENSITO_RECEIVER_ALL_TS", "0") == "1"
DEFAULT_ALLOWED_AET = [a.strip() for a in os.environ.get("DENSITO_RECEIVER_ALLOWED_AET", "").split(",") if a.strip()]
# ожидание API после старта контейнера: модели грузятся до минуты, приёмник ждёт и повторяет
API_RETRY_S = float(os.environ.get("DENSITO_RECEIVER_API_RETRY_S", "5"))
API_RETRY_MAX = int(os.environ.get("DENSITO_RECEIVER_API_RETRIES", "36"))   # 36 x 5 с = 3 мин
API_TIMEOUT_S = float(os.environ.get("DENSITO_RECEIVER_API_TIMEOUT_S", "1800"))

# SOP-классы DXA-экспорта: GE Lunar отдаёт CR (все 499 файлов выборки — CR Image Storage
# 1.2.840.10008.5.1.4.1.1.1), часть рабочих станций перекодирует в DX или Secondary Capture.
SUPPORTED_SOP_CLASSES = (
    ComputedRadiographyImageStorage,               # 1.2.840.10008.5.1.4.1.1.1
    DigitalXRayImageStorageForPresentation,        # 1.2.840.10008.5.1.4.1.1.1.1
    DigitalXRayImageStorageForProcessing,          # 1.2.840.10008.5.1.4.1.1.1.1.1
    SecondaryCaptureImageStorage,                  # 1.2.840.10008.5.1.4.1.1.7
)
# Transfer syntax, которые пайплайн читает без плагинов (pydicom): несжатые + RLE.
NATIVE_TRANSFER_SYNTAXES = (
    ImplicitVRLittleEndian, ExplicitVRLittleEndian, ExplicitVRBigEndian, RLELossless,
)

JOB_RE = re.compile(r"^[0-9]{8}_[0-9]{6}_[0-9a-f]{6}$")
LOG_COLUMNS = ("time", "event", "calling_aet", "study_uid", "sop_uid", "sop_class_uid",
               "transfer_syntax", "size_bytes", "job_id", "status")

# статусы C-STORE (PS3.4 B.2.3)
STATUS_SUCCESS = 0x0000
STATUS_OUT_OF_RESOURCES = 0xA700
STATUS_DATASET_MISMATCH = 0xA900
STATUS_CANNOT_UNDERSTAND = 0xC000


def _safe_uid(uid: Any) -> str:
    """UID -> безопасное имя каталога/файла: только цифры и точки, не длиннее 64 символов."""
    s = str(uid or "").strip()
    if s and re.fullmatch(r"[0-9.]{1,64}", s) and ".." not in s and not s.startswith(".") and not s.endswith("."):
        return s
    return ""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# --------------------------------------------------------------------------- #
# Журнал приёма
# --------------------------------------------------------------------------- #
class ReceiverLog:
    """CSV-журнал без персональных данных: только UID, AE Title, размеры, коды запросов."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def write(self, event: str, calling_aet: str = "", study_uid: str = "", sop_uid: str = "",
              sop_class_uid: str = "", transfer_syntax: str = "", size_bytes: Any = "",
              job_id: str = "", status: str = "") -> None:
        row = {"time": _now(), "event": event, "calling_aet": _clean(calling_aet, 16),
               "study_uid": study_uid, "sop_uid": sop_uid, "sop_class_uid": sop_class_uid,
               "transfer_syntax": transfer_syntax, "size_bytes": size_bytes, "job_id": job_id,
               "status": _clean(status, 120)}
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.path.exists() or self.path.stat().st_size == 0
            with open(self.path, "a", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=LOG_COLUMNS, delimiter=";", lineterminator="\n")
                if new:
                    w.writeheader()
                w.writerow(row)


def _clean(s: Any, max_len: int) -> str:
    return re.sub(r"[\x00-\x1f\x7f;]", " ", str(s or "")).strip()[:max_len]


# --------------------------------------------------------------------------- #
# Анализ: HTTP API того же контейнера или тот же маршрут внутри процесса
# --------------------------------------------------------------------------- #
class AnalyzeError(RuntimeError):
    pass


def _multipart_files(study_uid: str, files: List[Path]) -> List[Tuple[str, Tuple[str, bytes, str]]]:
    out = []
    for p in sorted(files):
        # имя с папкой исследования: path_to_study в CSV будет <study_uid>/<sop_uid>.dcm
        out.append(("files", (f"{study_uid}/{p.name}", p.read_bytes(), "application/dicom")))
    return out


class HttpApiBackend:
    """POST /api/analyze на работающий api_server (тот же контейнер или сервис densito-api)."""

    def __init__(self, api_url: str, retry_s: float = API_RETRY_S, retries: int = API_RETRY_MAX,
                 timeout_s: float = API_TIMEOUT_S):
        self.api_url = api_url.rstrip("/")
        self.retry_s = retry_s
        self.retries = retries
        self.timeout_s = timeout_s

    @property
    def name(self) -> str:
        return self.api_url

    def wait_ready(self) -> bool:
        import httpx
        for _ in range(max(1, self.retries)):
            try:
                r = httpx.get(f"{self.api_url}/api/health", timeout=10.0)
                if r.status_code == 200:
                    return True
                LOG.warning("API %s: /api/health -> %s, ждём", self.api_url, r.status_code)
            except Exception as e:  # noqa: BLE001
                LOG.warning("API %s недоступен (%s), ждём %.0f с", self.api_url, type(e).__name__, self.retry_s)
            time.sleep(self.retry_s)
        return False

    def analyze(self, study_uid: str, files: List[Path]) -> Dict[str, Any]:
        import httpx
        payload = _multipart_files(study_uid, files)
        last_err: Optional[str] = None
        for attempt in range(1, max(1, self.retries) + 1):
            try:
                r = httpx.post(f"{self.api_url}/api/analyze", params={"xlsx": "true"}, files=payload,
                               timeout=httpx.Timeout(self.timeout_s, connect=10.0))
            except Exception as e:  # noqa: BLE001 — API ещё грузит модели или недоступен
                last_err = f"{type(e).__name__}: {e}"
                LOG.warning("analyze %s: попытка %d/%d не удалась (%s)", study_uid, attempt, self.retries, last_err)
                time.sleep(self.retry_s)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code in (502, 503, 504):
                last_err = f"HTTP {r.status_code}"
                time.sleep(self.retry_s)
                continue
            raise AnalyzeError(f"HTTP {r.status_code}: {r.text[:300]}")
        raise AnalyzeError(f"API недоступен после {self.retries} попыток: {last_err}")


class InprocBackend:
    """Тот же маршрут /api/analyze без сети: api_server импортируется в процесс приёмника."""

    name = "inproc"

    def __init__(self):
        from fastapi.testclient import TestClient
        import api_server  # noqa: F401 — модели грузятся при первом запросе
        self._client = TestClient(api_server.app)

    def wait_ready(self) -> bool:
        return self._client.get("/api/health").status_code == 200

    def analyze(self, study_uid: str, files: List[Path]) -> Dict[str, Any]:
        r = self._client.post("/api/analyze", params={"xlsx": "true"}, files=_multipart_files(study_uid, files))
        if r.status_code != 200:
            raise AnalyzeError(f"HTTP {r.status_code}: {r.text[:300]}")
        return r.json()


def make_backend(api_url: str):
    if api_url.strip().lower() in ("inproc", "local", "direct"):
        return InprocBackend()
    return HttpApiBackend(api_url)


# --------------------------------------------------------------------------- #
# Приёмник
# --------------------------------------------------------------------------- #
class StudyState:
    __slots__ = ("files", "last", "assocs", "calling_aet", "busy")

    def __init__(self):
        self.files: Dict[str, Path] = {}
        self.last = time.monotonic()
        self.assocs: set = set()
        self.calling_aet = ""
        self.busy = False


class DicomReceiver:
    """Storage SCP: приём в inbox, группировка по исследованию, передача на анализ."""

    def __init__(self, inbox: Path = DEFAULT_INBOX, port: int = DEFAULT_PORT, aet: str = DEFAULT_AET,
                 bind: str = DEFAULT_BIND, analyze: Optional[Callable[[str, List[Path]], Dict[str, Any]]] = None,
                 api_url: str = DEFAULT_API_URL, idle_s: float = DEFAULT_IDLE_S,
                 release_s: float = DEFAULT_RELEASE_S, keep_files: bool = DEFAULT_KEEP,
                 all_transfer_syntaxes: bool = DEFAULT_ALL_TS, allowed_aet: Optional[List[str]] = None):
        self.inbox = Path(inbox)
        self.port = int(port)
        self.aet = aet
        self.bind = bind
        self.idle_s = float(idle_s)
        self.release_s = float(release_s)
        self.keep_files = keep_files
        self.transfer_syntaxes = list(ALL_TRANSFER_SYNTAXES) if all_transfer_syntaxes else list(NATIVE_TRANSFER_SYNTAXES)
        self.allowed_aet = list(allowed_aet if allowed_aet is not None else DEFAULT_ALLOWED_AET)
        self.log = ReceiverLog(self.inbox / "receiver_log.csv")
        self._backend = None
        self._analyze = analyze          # подмена в тестах; None -> backend по api_url
        self._api_url = api_url
        self._studies: Dict[str, StudyState] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._server = None
        self._watcher: Optional[threading.Thread] = None
        self.stats = {"echo": 0, "stored": 0, "rejected": 0, "analyzed": 0, "failed": 0}
        self.inbox.mkdir(parents=True, exist_ok=True)

    # ---- backend -------------------------------------------------------- #
    def analyze_fn(self) -> Callable[[str, List[Path]], Dict[str, Any]]:
        if self._analyze is not None:
            return self._analyze
        if self._backend is None:
            self._backend = make_backend(self._api_url)
        return self._backend.analyze

    # ---- AE -------------------------------------------------------------- #
    def build_ae(self) -> AE:
        ae = AE(ae_title=self.aet)
        ae.add_supported_context(Verification)
        for sop in SUPPORTED_SOP_CLASSES:
            ae.add_supported_context(sop, self.transfer_syntaxes)
        ae.maximum_pdu_size = 0                 # без ограничения PDU: кадры GE Lunar до ~1 МБ
        ae.network_timeout = 60
        ae.acse_timeout = 60
        ae.dimse_timeout = 120
        if self.allowed_aet:
            ae.require_calling_aet = self.allowed_aet
        return ae

    def handlers(self) -> list:
        return [
            (evt.EVT_C_ECHO, self.on_echo),
            (evt.EVT_C_STORE, self.on_store),
            (evt.EVT_RELEASED, self.on_assoc_end),
            (evt.EVT_ABORTED, self.on_assoc_end),
            (evt.EVT_CONN_CLOSE, self.on_assoc_end),
        ]

    def start(self, block: bool = False):
        self._resume_pending()
        ae = self.build_ae()
        self._watcher = threading.Thread(target=self._watch_loop, name="densito-receiver-watch", daemon=True)
        self._watcher.start()
        LOG.info("DICOM receiver %s: AE Title %s, %s:%d, inbox %s, анализ -> %s, SOP-классов %d, TS %d",
                 __version__, self.aet, self.bind, self.port, self.inbox,
                 self._api_url if self._analyze is None else "callable", len(SUPPORTED_SOP_CLASSES),
                 len(self.transfer_syntaxes))
        self.log.write("start", status=f"aet={self.aet} port={self.port} api={self._api_url}")
        self._server = ae.start_server((self.bind, self.port), block=False, evt_handlers=self.handlers())
        if block:
            try:
                while not self._stop.is_set():
                    time.sleep(0.5)
            finally:
                self.stop()
        return self._server

    def stop(self, flush: bool = True):
        if self._stop.is_set():
            return
        self._stop.set()
        if self._server is not None:
            try:
                self._server.shutdown()
            except Exception:  # noqa: BLE001
                pass
        if flush:
            for uid in list(self._studies):
                self._dispatch(uid)
        self.log.write("stop", status=json.dumps(self.stats, ensure_ascii=False))

    # ---- события -------------------------------------------------------- #
    @staticmethod
    def _calling(event) -> str:
        try:
            return str(event.assoc.requestor.ae_title or "").strip()
        except Exception:  # noqa: BLE001
            return ""

    def on_echo(self, event) -> int:
        self.stats["echo"] += 1
        self.log.write("echo", calling_aet=self._calling(event), status="0x0000")
        return STATUS_SUCCESS

    def on_store(self, event) -> int:
        calling = self._calling(event)
        try:
            ds: Dataset = event.dataset
            ds.file_meta = event.file_meta
        except Exception as e:  # noqa: BLE001
            self.stats["rejected"] += 1
            self.log.write("reject", calling_aet=calling, status=f"cannot decode: {type(e).__name__}")
            return STATUS_CANNOT_UNDERSTAND
        study_uid = _safe_uid(getattr(ds, "StudyInstanceUID", None))
        sop_uid = _safe_uid(getattr(ds, "SOPInstanceUID", None) or getattr(ds.file_meta, "MediaStorageSOPInstanceUID", None))
        sop_class = str(getattr(ds, "SOPClassUID", "") or getattr(ds.file_meta, "MediaStorageSOPClassUID", ""))
        ts = str(getattr(ds.file_meta, "TransferSyntaxUID", ""))
        if not study_uid or not sop_uid:
            self.stats["rejected"] += 1
            self.log.write("reject", calling_aet=calling, study_uid=study_uid, sop_uid=sop_uid,
                           sop_class_uid=sop_class, transfer_syntax=ts, status="0xA900 no StudyInstanceUID/SOPInstanceUID")
            return STATUS_DATASET_MISMATCH
        target = self.inbox / study_uid / f"{sop_uid}.dcm"
        tmp = target.with_name(f".{sop_uid}.{os.getpid()}.part")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            pydicom.dcmwrite(tmp, ds, enforce_file_format=True)
            os.replace(tmp, target)
            size = target.stat().st_size
        except Exception as e:  # noqa: BLE001
            try:
                tmp.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
            self.stats["rejected"] += 1
            LOG.error("store %s/%s: запись не удалась: %s", study_uid, sop_uid, e)
            self.log.write("reject", calling_aet=calling, study_uid=study_uid, sop_uid=sop_uid,
                           sop_class_uid=sop_class, transfer_syntax=ts, status=f"0xA700 write failed: {type(e).__name__}")
            return STATUS_OUT_OF_RESOURCES
        with self._lock:
            st = self._studies.setdefault(study_uid, StudyState())
            st.files[sop_uid] = target
            st.last = time.monotonic()
            st.assocs.add(id(event.assoc))
            st.calling_aet = calling
        self.stats["stored"] += 1
        self.log.write("store", calling_aet=calling, study_uid=study_uid, sop_uid=sop_uid,
                       sop_class_uid=sop_class, transfer_syntax=ts, size_bytes=size, status="0x0000")
        return STATUS_SUCCESS

    def on_assoc_end(self, event) -> None:
        aid = id(event.assoc)
        with self._lock:
            for st in self._studies.values():
                st.assocs.discard(aid)

    # ---- диспетчер ------------------------------------------------------- #
    def _watch_loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(0.25)
            now = time.monotonic()
            ready: List[str] = []
            with self._lock:
                for uid, st in self._studies.items():
                    if st.busy or not st.files:
                        continue
                    idle = now - st.last
                    if idle >= self.idle_s or (not st.assocs and idle >= self.release_s):
                        ready.append(uid)
            for uid in ready:
                self._dispatch(uid)

    def _dispatch(self, study_uid: str) -> Optional[str]:
        with self._lock:
            st = self._studies.get(study_uid)
            if st is None or st.busy or not st.files:
                return None
            st.busy = True
            files = [p for p in st.files.values() if p.is_file()]
            calling = st.calling_aet
            del self._studies[study_uid]
        if not files:
            return None
        total = sum(p.stat().st_size for p in files)
        t0 = time.perf_counter()
        try:
            resp = self.analyze_fn()(study_uid, files)
            job_id = str(resp.get("job_id") or "")
            if not JOB_RE.match(job_id):
                raise AnalyzeError(f"ответ без корректного job_id: {job_id!r}")
        except Exception as e:  # noqa: BLE001
            self.stats["failed"] += 1
            LOG.error("analyze %s (%d файлов): %s", study_uid, len(files), e)
            self.log.write("analyze_failed", calling_aet=calling, study_uid=study_uid, size_bytes=total,
                           status=f"{type(e).__name__}: {e}")
            return None
        self.stats["analyzed"] += 1
        summary = resp.get("summary") or {}
        self.log.write("analyze", calling_aet=calling, study_uid=study_uid, size_bytes=total, job_id=job_id,
                       status=f"files={len(files)} violations={summary.get('n_violations', '')} "
                              f"failures={summary.get('n_failures', '')} t={time.perf_counter() - t0:.1f}s")
        self._write_job_json(study_uid, job_id, str(resp.get("job_token") or ""), files, summary)
        if not self.keep_files:
            for p in files:
                try:
                    p.unlink()
                except OSError:
                    pass
        LOG.info("исследование %s: %d файлов -> запрос %s (%.1f с)", study_uid, len(files), job_id,
                 time.perf_counter() - t0)
        return job_id

    def _write_job_json(self, study_uid: str, job_id: str, token: str, files: List[Path], summary: Dict[str, Any]) -> None:
        d = self.inbox / study_uid
        d.mkdir(parents=True, exist_ok=True)
        p = d / "job.json"
        card = {
            "study_uid": study_uid, "job_id": job_id, "job_token": token, "created_at": _now(),
            "n_files": len(files), "sop_uids": sorted(x.stem for x in files),
            "summary": {k: summary.get(k) for k in ("n_files", "n_violations", "n_failures", "n_studies")},
            "results_csv": f"/api/results/{job_id}/results.csv?t={token}" if token else f"/api/results/{job_id}/results.csv",
            "card": f"/api/jobs/{job_id}?t={token}" if token else f"/api/jobs/{job_id}",
            "cabinet": f"/#app/job/{job_id}/{token}" if token else f"/#app/job/{job_id}",
        }
        try:
            p.write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
            os.chmod(p, 0o600)
        except OSError as e:
            LOG.warning("job.json %s: %s", study_uid, e)

    def _resume_pending(self) -> None:
        """Снимки, оставшиеся в inbox после сбоя анализа или остановки, отправляются повторно."""
        if not self.inbox.is_dir():
            return
        n = 0
        for d in sorted(self.inbox.iterdir()):
            if not d.is_dir() or not _safe_uid(d.name):
                continue
            files = sorted(x for x in d.glob("*.dcm") if x.is_file())
            if not files:
                continue
            with self._lock:
                st = self._studies.setdefault(d.name, StudyState())
                for x in files:
                    st.files[x.stem] = x
                st.last = time.monotonic()
            n += len(files)
        if n:
            LOG.info("в inbox найдено %d необработанных файлов — будут отправлены на анализ", n)
            self.log.write("resume", size_bytes=n, status="pending files re-queued")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DensitoAI DICOM receiver (Storage SCP)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--aet", default=DEFAULT_AET)
    ap.add_argument("--bind", default=DEFAULT_BIND)
    ap.add_argument("--inbox", default=str(DEFAULT_INBOX))
    ap.add_argument("--api-url", default=DEFAULT_API_URL, help="URL API или inproc")
    ap.add_argument("--idle", type=float, default=DEFAULT_IDLE_S, help="таймаут тишины, с")
    ap.add_argument("--release", type=float, default=DEFAULT_RELEASE_S, help="пауза после закрытия ассоциации, с")
    ap.add_argument("--keep", dest="keep", action="store_true", default=DEFAULT_KEEP,
                    help="хранить принятые DICOM после анализа (по умолчанию; DENSITO_INBOX_KEEP=1)")
    ap.add_argument("--no-keep", dest="keep", action="store_false",
                    help="удалять DICOM после успешного анализа (DENSITO_INBOX_KEEP=0)")
    ap.add_argument("--all-ts", action="store_true", default=DEFAULT_ALL_TS, help="принимать все transfer syntax")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    inbox = Path(args.inbox)
    inbox.mkdir(parents=True, exist_ok=True)
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler(inbox / "receiver.log", encoding="utf-8"))
    except OSError:
        pass
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=handlers)
    logging.getLogger("pynetdicom").setLevel(logging.INFO if args.verbose else logging.WARNING)

    rx = DicomReceiver(inbox=inbox, port=args.port, aet=args.aet, bind=args.bind, api_url=args.api_url,
                       idle_s=args.idle, release_s=args.release, keep_files=args.keep,
                       all_transfer_syntaxes=args.all_ts)

    def _term(signum, frame):  # noqa: ARG001
        LOG.info("сигнал %s — остановка", signum)
        rx.stop()

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    rx.start(block=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
