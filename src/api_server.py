#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DensitoAI — HTTP API для пакетной обработки (требование ТЗ п.3.2).

Полностью локальный сервис (никаких внешних вызовов). Использует тот же
пайплайн, что и CLI (src/inference.py): одинаковые модели, пороги и формат.

Эндпоинты
---------
GET  /api/health                     — состояние сервиса, загруженные модели, версия.
POST /api/analyze  (multipart/form-data, поле `files`, один или несколько .dcm/.zip)
                                     — обработать загруженные файлы, вернуть JSON
                                       со строками официального формата
                                       (+ CSV-текст в поле `csv`).
POST /api/batch    (JSON {"input_dir": ..., "output_csv": ..., "xlsx": bool})
                                     — обработать папку/архив, уже доступный на
                                       файловой системе контейнера (смонтированный
                                       том), записать CSV (и опционально XLSX).
GET  /api/results/{name}             — скачать ранее сформированный файл из папки
                                       результатов (по имени, без путей).
GET  /docs                           — Swagger UI (генерируется FastAPI).

Запуск
------
  python src/api_server.py --host 0.0.0.0 --port 8000
  # или в контейнере: ./build_and_run.sh api
"""
import argparse
import io
import logging
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from inference import (  # noqa: E402
    DensitoInference, MODELS_DIR, PROJECT_ROOT, load_config, setup_logging,
    write_results, validate_output_csv, __version__ as PIPELINE_VERSION,
)

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel
except ImportError as e:  # pragma: no cover
    print("FastAPI/uvicorn не установлены: pip install fastapi uvicorn python-multipart", file=sys.stderr)
    raise e

LOG = logging.getLogger("densito.api")

OUTPUT_DIR = Path(os.environ.get("DENSITO_OUTPUT_DIR", PROJECT_ROOT / "outputs")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_MB = float(os.environ.get("DENSITO_MAX_UPLOAD_MB", "2048"))

app = FastAPI(
    title="DensitoAI — контроль качества DXA",
    description="Пакетная оценка качества DXA-снимков (поясничный отдел, проксимальный отдел бедра). "
                "Полностью локальный сервис, формат ответа — по ТЗ ЛЦТ 2026.",
    version=PIPELINE_VERSION,
)
_ENGINE: Optional[DensitoInference] = None
_STARTED = time.time()


BONUS_DIR = OUTPUT_DIR / "bonus"
VIZ_DIR = BONUS_DIR / "viz"
SR_DIR = BONUS_DIR / "sr"
ROI_DIR = BONUS_DIR / "roi"
for _d in (VIZ_DIR, SR_DIR, ROI_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def engine() -> DensitoInference:
    global _ENGINE
    if _ENGINE is None:
        cfg = load_config(Path(os.environ["DENSITO_CONFIG"]) if os.environ.get("DENSITO_CONFIG") else None)
        # Бонус-выходы (визуализация/SR/ROI) включены по умолчанию в API-режиме —
        # экспертному/врачебному тестированию нужен visual overlay, не только CSV-строка.
        # На основной CSV/XLSX это не влияет (см. README §14, ENGINEERING_REPORT §5).
        enable_bonus = os.environ.get("DENSITO_DISABLE_BONUS", "0") != "1"
        _ENGINE = DensitoInference(
            cfg=cfg, models_dir=MODELS_DIR,
            use_embeddings=os.environ.get("DENSITO_NO_EMBEDDINGS", "0") != "1",
            visualize_dir=VIZ_DIR if enable_bonus else None,
            sr_dir=SR_DIR if enable_bonus else None,
            roi_autocorrect_dir=ROI_DIR if enable_bonus else None,
        )
        LOG.info("Inference engine initialised (models: %d, bonus_outputs=%s)",
                 _ENGINE.registry.n_loaded, enable_bonus)
    return _ENGINE


class BatchRequest(BaseModel):
    input_dir: str
    output_csv: Optional[str] = None
    xlsx: bool = False
    limit: Optional[int] = None


def _rows_to_csv_text(rows: List[Dict[str, Any]], cfg: Dict[str, Any]) -> str:
    import csv
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cfg["output"]["columns"], extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def _summary(rows: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    fail = cfg["output"]["status_failure"]
    return {
        "n_files": len(rows),
        "n_failures": sum(1 for r in rows if r["processing_status"] == fail),
        "n_violations": sum(1 for r in rows if str(r["quality_class"]) == "1"),
        "total_processing_time_s": round(sum(float(r["time_of_processing"]) for r in rows), 3),
    }


@app.on_event("startup")
def _startup():
    setup_logging(OUTPUT_DIR / "api_server.log", verbose=False)
    try:
        engine()
    except Exception as e:  # noqa: BLE001 — сервис должен подняться даже без моделей
        LOG.error("Engine init failed: %s", e)


@app.get("/api/health")
def health():
    try:
        eng = engine()
        reg = eng.registry
        return {
            "status": "ok",
            "version": PIPELINE_VERSION,
            "uptime_s": round(time.time() - _STARTED, 1),
            "models_dir": str(reg.models_dir),
            "models_loaded": reg.n_loaded,
            "geom_models": [f"{r}/{c}" for (r, c) in reg.geom],
            "emb_models": [f"{r}/{c}" for (r, c) in reg.emb],
            "fallback_rules_active": not reg.has_any_model(),
            "thresholds": reg.thresholds,
            "output_dir": str(OUTPUT_DIR),
        }
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


def _find_bonus_file(directory: Path, stem: str, suffix_options: List[str]) -> Optional[Path]:
    if not directory.exists():
        return None
    for suf in suffix_options:
        cand = directory / f"{stem}{suf}"
        if cand.exists():
            return cand
    return None


def _attach_bonus(row: Dict[str, Any]) -> Dict[str, Any]:
    """Добавляет в строку бонус-выходы для UI/экспертного просмотра: overlay PNG (base64,
    чтобы сразу отрисовать в браузере), имя SR .dcm и ROI-коррекции PNG для скачивания
    через /api/results/{name}. Не меняет официальные поля строки ФОРМАТА ОТВЕТА."""
    stem = Path(str(row.get("path_to_study", ""))).stem
    if not stem:
        return row
    out = dict(row)
    viz = _find_bonus_file(VIZ_DIR, stem, ["_overlay.png"])
    if viz is not None:
        try:
            import base64
            out["bonus_overlay_png_base64"] = base64.b64encode(viz.read_bytes()).decode("ascii")
        except Exception:  # noqa: BLE001
            pass
    sr = _find_bonus_file(SR_DIR, stem, ["_sr.dcm"])
    if sr is not None:
        try:
            shutil.copy2(sr, OUTPUT_DIR / sr.name)
            out["bonus_sr_dcm_download"] = f"/api/results/{sr.name}"
        except Exception:  # noqa: BLE001
            pass
    roi = _find_bonus_file(ROI_DIR, stem, ["_roi_correction.png"])
    if roi is not None:
        try:
            import base64
            out["bonus_roi_png_base64"] = base64.b64encode(roi.read_bytes()).decode("ascii")
        except Exception:  # noqa: BLE001
            pass
    return out


@app.post("/api/analyze")
async def analyze(files: List[UploadFile] = File(...), xlsx: bool = False):
    """Загрузка одного или нескольких DICOM / zip. Ответ — JSON со строками официального
    формата, CSV-текстом и (если включены в движке) бонус-визуализациями (overlay PNG,
    ссылка на DICOM SR, ROI-диагностика) для каждой строки; файлы результата также сохраняются
    в OUTPUT_DIR."""
    if not files:
        raise HTTPException(400, "no files uploaded")
    job = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    tmp = Path(tempfile.mkdtemp(prefix=f"densito_api_{job}_"))
    try:
        total = 0
        for uf in files:
            name = Path(uf.filename or f"upload_{uuid.uuid4().hex[:6]}.dcm").name  # без путей
            data = await uf.read()
            total += len(data)
            if total > MAX_UPLOAD_MB * 1024 * 1024:
                raise HTTPException(413, f"upload exceeds {MAX_UPLOAD_MB} MB")
            (tmp / name).write_bytes(data)
        eng = engine()
        out_csv = OUTPUT_DIR / f"results_{job}.csv"
        rows = eng.run(tmp, out_csv, debug_csv=OUTPUT_DIR / f"results_{job}_debug.csv", xlsx=xlsx)
        problems = validate_output_csv(out_csv, eng.cfg)
        rows_with_bonus = [_attach_bonus(r) for r in rows]
        return {
            "job_id": job,
            "summary": _summary(rows, eng.cfg),
            "format_check": "OK" if not problems else problems,
            "result_csv": out_csv.name,
            "result_xlsx": out_csv.with_suffix(".xlsx").name if xlsx else None,
            "rows": rows_with_bonus,
            "csv": _rows_to_csv_text(rows, eng.cfg),
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        LOG.exception("analyze failed")
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/api/batch")
def batch(req: BatchRequest):
    """Пакетная обработка папки/архива, доступного внутри контейнера (например,
    смонтированного тома /data/input). Пишет CSV по указанному пути (по умолчанию
    в OUTPUT_DIR)."""
    src = Path(req.input_dir)
    if not src.exists():
        raise HTTPException(404, f"input_dir not found: {src}")
    out_csv = Path(req.output_csv) if req.output_csv else \
        OUTPUT_DIR / f"results_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    if out_csv.suffix.lower() == ".xlsx":
        req.xlsx, out_csv = True, out_csv.with_suffix(".csv")
    try:
        eng = engine()
        t0 = time.perf_counter()
        rows = eng.run(src, out_csv, debug_csv=out_csv.with_name(out_csv.stem + "_debug.csv"),
                       xlsx=req.xlsx, limit=req.limit)
        problems = validate_output_csv(out_csv, eng.cfg)
        return {
            "summary": {**_summary(rows, eng.cfg), "wall_time_s": round(time.perf_counter() - t0, 3)},
            "format_check": "OK" if not problems else problems,
            "output_csv": str(out_csv),
            "output_xlsx": str(out_csv.with_suffix(".xlsx")) if req.xlsx else None,
        }
    except Exception as e:  # noqa: BLE001
        LOG.exception("batch failed")
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.get("/api/results/{name}")
def download(name: str):
    safe = Path(name).name
    p = OUTPUT_DIR / safe
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(str(p), filename=safe)


def main(argv=None):
    ap = argparse.ArgumentParser(description="DensitoAI API server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("DENSITO_PORT", "8000")))
    ap.add_argument("--workers", type=int, default=1, help="1 — модели грузятся один раз (рекомендуется)")
    args = ap.parse_args(argv)
    import uvicorn
    uvicorn.run("api_server:app", host=args.host, port=args.port, workers=args.workers, log_level="info")


if __name__ == "__main__":
    main()
