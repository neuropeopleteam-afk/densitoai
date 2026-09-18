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
GET  /api/results/{job}/{name}       — скачать файл результата конкретного запроса
                                       (CSV/XLSX/технический CSV/summary.json/бонус-файлы);
                                       доступ только к файлам своего запроса.
GET  /api/jobs, /api/jobs/{job}      — история запросов и карточка запроса.
GET  /api/results/{name}             — совместимость: results_*.csv/.xlsx из /api/batch.
GET  /docs                           — Swagger UI (генерируется FastAPI).

Запуск
------
  python src/api_server.py --host 0.0.0.0 --port 8000
  # или в контейнере: ./build_and_run.sh api
"""
import argparse
import base64
import io
import json
import re
import threading
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
    from starlette.concurrency import run_in_threadpool
    from pydantic import BaseModel
except ImportError as e:  # pragma: no cover
    print("FastAPI/uvicorn не установлены: pip install fastapi uvicorn python-multipart", file=sys.stderr)
    raise e

LOG = logging.getLogger("densito.api")

OUTPUT_DIR = Path(os.environ.get("DENSITO_OUTPUT_DIR", PROJECT_ROOT / "outputs")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_MB = float(os.environ.get("DENSITO_MAX_UPLOAD_MB", "512"))
MAX_FILES_PER_REQUEST = int(os.environ.get("DENSITO_MAX_FILES", "500"))
JOBS_DIR = OUTPUT_DIR / "jobs"           # результаты каждого запроса — в отдельной папке
JOBS_DIR.mkdir(parents=True, exist_ok=True)
JOB_RE = re.compile(r"^[0-9]{8}_[0-9]{6}_[0-9a-f]{6}$")
_RUN_LOCK = threading.Lock()             # инференс сериализуем: один запрос — один прогон,
                                         # без гонок за файлы бонус-визуализаций

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


def _collect_bonus(job_dir: Path, stems: List[str]) -> None:
    """Переносит бонус-файлы (overlay/SR/ROI) текущего прогона из общих папок движка в
    папку запроса. Вызывается под _RUN_LOCK, поэтому файлы принадлежат именно этому прогону."""
    dst = job_dir / "bonus"
    dst.mkdir(parents=True, exist_ok=True)
    for stem in set(stems):
        for d, suf in ((VIZ_DIR, "_overlay.png"), (SR_DIR, "_sr.dcm"), (ROI_DIR, "_roi_correction.png")):
            src = d / f"{stem}{suf}"
            if src.exists():
                try:
                    shutil.move(str(src), str(dst / src.name))
                except Exception:  # noqa: BLE001
                    pass


def _attach_bonus(row: Dict[str, Any], job: str) -> Dict[str, Any]:
    """Добавляет в строку бонус-выходы для UI/экспертного просмотра: overlay PNG (base64,
    чтобы сразу отрисовать в браузере), ссылку на SR .dcm и ROI-коррекцию PNG. Ссылки ведут
    только в папку этого запроса (/api/results/{job}/{name}). Официальные поля не меняются."""
    stem = Path(str(row.get("path_to_study", ""))).stem
    if not stem:
        return row
    out = dict(row)
    bdir = JOBS_DIR / job / "bonus"
    viz = bdir / f"{stem}_overlay.png"
    if viz.exists():
        try:
            out["bonus_overlay_png_base64"] = base64.b64encode(viz.read_bytes()).decode("ascii")
        except Exception:  # noqa: BLE001
            pass
    sr = bdir / f"{stem}_sr.dcm"
    if sr.exists():
        out["bonus_sr_dcm_download"] = f"/api/results/{job}/{sr.name}"
    roi = bdir / f"{stem}_roi_correction.png"
    if roi.exists():
        try:
            out["bonus_roi_png_base64"] = base64.b64encode(roi.read_bytes()).decode("ascii")
        except Exception:  # noqa: BLE001
            pass
    return out


# --- Детали для карточки решения в UI (не входят в официальный CSV) ---------------------------
CRITERION_TITLES = {
    "sp_pos": "Укладка (позвоночник)",
    "sp_axis": "Ось позвоночника",
    "sp_art": "Посторонние предметы",
    "rh_pos": "Укладка (правое бедро)",
    "rh_roi": "Область интереса (правое бедро)",
    "lh_pos": "Укладка (левое бедро)",
    "lh_roi": "Область интереса (левое бедро)",
}
# Измерения, которые понятны врачу: (ключ в debug, подпись, единица, множитель, знаков)
MEASUREMENTS = {
    "spine": [
        ("feat_axis_angle_deg", "Наклон оси позвоночника", "°", 1.0, 1),
        ("feat_curvature", "Изгиб оси", "", 1.0, 3),
        ("feat_center_offset_ratio", "Смещение от центра кадра", "% ширины", 100.0, 1),
        ("feat_bone_width_ratio", "Ширина костной области", "% ширины", 100.0, 1),
        ("feat_top_margin_ratio", "Отступ сверху", "% высоты", 100.0, 1),
        ("feat_bottom_margin_ratio", "Отступ снизу", "% высоты", 100.0, 1),
        ("feat_metal_metal_area_mm2", "Площадь плотных включений", "мм²", 1.0, 0),
        ("feat_metal_metal_max_intensity_gap", "Контраст включений к кости", "сигм", 1.0, 2),
    ],
    "hip": [
        ("feat_shaft_angle_deg", "Наклон диафиза бедра", "°", 1.0, 1),
        ("feat_edge_distance_mm", "Расстояние от кости до бокового края", "мм", 1.0, 1),
        ("feat_bone_area_ratio", "Доля кости в кадре", "%", 100.0, 1),
        ("feat_metal_metal_area_mm2", "Площадь плотных включений", "мм²", 1.0, 0),
        ("bonus_roi_deficit_mm", "Недостаток поля сканирования", "мм", 1.0, 0),
    ],
}


def _num(v) -> Optional[float]:
    try:
        if v is None or v == "" or (isinstance(v, float) and v != v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _details(row: Dict[str, Any], dbg: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Карточка решения: по каждому критерию — скор, порог, флаг, источник; плюс
    измерения в понятных единицах и рекомендуемое действие. Только для UI/экспертного просмотра."""
    region = str(dbg.get("internal_region") or "")
    fail = cfg["output"]["status_failure"]
    is_fail = row.get("processing_status") == fail
    crits = []
    for c in cfg["criteria_by_region"].get(region, []):
        score, thr = _num(dbg.get(f"{c}_score")), _num(dbg.get(f"{c}_threshold"))
        flag = dbg.get(f"{c}_flag")
        crits.append({
            "code": c,
            "title": CRITERION_TITLES.get(c, c),
            "violation": cfg["violations"].get(c, c),
            "score": None if score is None else round(score, 3),
            "threshold": None if thr is None else round(thr, 3),
            "flag": bool(flag) if flag not in (None, "") else False,
            "method": dbg.get(f"{c}_method"),
            "p_geom": (lambda v: None if v is None else round(v, 3))(_num(dbg.get(f"{c}_p_geom"))),
            "p_emb": (lambda v: None if v is None else round(v, 3))(_num(dbg.get(f"{c}_p_emb"))),
        })
    meas = []
    for key, title, unit, mult, nd in MEASUREMENTS.get("spine" if region == "spine" else "hip", []):
        v = _num(dbg.get(key))
        if v is not None:
            meas.append({"title": title, "value": round(v * mult, nd), "unit": unit})
    if is_fail:
        action, action_code = "Проверить вручную: файл не обработан", "manual"
    elif str(row.get("quality_class")) == "1":
        action, action_code = "Проверить снимок; при подтверждении — переснять", "review"
    else:
        action, action_code = "Принять", "accept"
    return {
        "internal_region": region,
        "region_source": dbg.get("region_source"),
        "image_size": [dbg.get("rows"), dbg.get("cols")],
        "warnings": dbg.get("warnings") or "",
        "error": dbg.get("error"),
        "quality_prob_raw": _num(dbg.get("quality_prob_raw")),
        "criteria": crits,
        "measurements": meas,
        "roi": {
            "needs_correction": dbg.get("bonus_roi_needs_correction"),
            "reason": dbg.get("bonus_roi_reason"),
            "deficit_mm": _num(dbg.get("bonus_roi_deficit_mm")),
        },
        "action": action,
        "action_code": action_code,
    }


def _config_hash(cfg: Dict[str, Any]) -> str:
    import hashlib
    import json
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:12]


def _run_job(job: str, tmp: Path, job_dir: Path, xlsx: bool):
    """Синхронная часть: инференс под глобальной блокировкой + сбор бонус-файлов в папку запроса."""
    eng = engine()
    out_csv = job_dir / "results.csv"
    with _RUN_LOCK:
        rows = eng.run(tmp, out_csv, debug_csv=job_dir / "results_debug.csv", xlsx=xlsx)
        debug_rows = list(getattr(eng, "last_debug_rows", []) or [])
        _collect_bonus(job_dir, [Path(str(r.get("path_to_study", ""))).stem for r in rows])
    return eng, out_csv, rows, debug_rows


@app.post("/api/analyze")
async def analyze(files: List[UploadFile] = File(...), xlsx: bool = False):
    """Загрузка одного или нескольких DICOM / zip. Ответ — JSON со строками официального
    формата, CSV-текстом и (если включены в движке) бонус-визуализациями (overlay PNG,
    ссылка на DICOM SR, ROI-диагностика) для каждой строки. Все файлы запроса сохраняются
    в отдельной папке OUTPUT_DIR/jobs/{job_id}/ и доступны по /api/results/{job_id}/{name}."""
    if not files:
        raise HTTPException(400, "Файлы не переданы. Загрузите один или несколько .dcm или zip-архив исследования.")
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(413, f"Слишком много файлов в одном запросе ({len(files)} > {MAX_FILES_PER_REQUEST}). "
                                 f"Упакуйте исследование в zip или разбейте партию.")
    job = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    tmp = Path(tempfile.mkdtemp(prefix=f"densito_api_{job}_"))
    job_dir = JOBS_DIR / job
    job_dir.mkdir(parents=True, exist_ok=True)
    t_wall = time.perf_counter()
    try:
        total = 0
        limit = int(MAX_UPLOAD_MB * 1024 * 1024)
        for uf in files:
            name = Path(uf.filename or f"upload_{uuid.uuid4().hex[:6]}.dcm").name  # без путей
            data = await uf.read()
            total += len(data)
            if total > limit:
                raise HTTPException(413, f"Объём загрузки превышает {MAX_UPLOAD_MB:.0f} МБ. Разбейте партию на части.")
            if not data:
                raise HTTPException(400, f"Файл «{name}» пустой.")
            (tmp / name).write_bytes(data)
        eng, out_csv, rows, debug_rows = await run_in_threadpool(_run_job, job, tmp, job_dir, xlsx)
        problems = validate_output_csv(out_csv, eng.cfg)
        rows_out = []
        for i, r in enumerate(rows):
            rb = _attach_bonus(r, job)
            dbg = debug_rows[i] if i < len(debug_rows) else {}
            try:
                rb["details"] = _details(r, dbg, eng.cfg)
            except Exception as e:  # noqa: BLE001 — детали не должны ломать ответ
                LOG.warning("details failed for row %d: %s", i, e)
            rows_out.append(rb)
        xlsx_path = out_csv.with_suffix(".xlsx")
        has_xlsx = bool(xlsx and xlsx_path.exists())
        resp = {
            "job_id": job,
            "request_id": job,
            "model_version": PIPELINE_VERSION,
            "config_hash": _config_hash(eng.cfg),
            "summary": {**_summary(rows, eng.cfg), "wall_time_s": round(time.perf_counter() - t_wall, 3),
                        "n_studies": len({r.get("study_uid") for r in rows if r.get("study_uid")}),
                        "n_files_uploaded": len(files), "bytes_uploaded": total},
            "format_check": "OK" if not problems else problems,
            # имена файлов (совместимость) + готовые ссылки
            "result_csv": out_csv.name,
            "result_debug_csv": "results_debug.csv",
            "result_xlsx": xlsx_path.name if has_xlsx else None,
            "result_csv_url": f"/api/results/{job}/{out_csv.name}",
            "result_debug_csv_url": f"/api/results/{job}/results_debug.csv",
            "result_xlsx_url": f"/api/results/{job}/{xlsx_path.name}" if has_xlsx else None,
            "rows": rows_out,
            "csv": _rows_to_csv_text(rows, eng.cfg),
        }
        # краткая карточка запроса для истории (без base64-картинок)
        try:
            card = {k: v for k, v in resp.items() if k not in ("rows", "csv")}
            card["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            card["rows"] = [{k: v for k, v in r.items() if not str(k).endswith("_base64")} for r in rows_out]
            (job_dir / "summary.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            LOG.warning("summary.json failed: %s", e)
        return resp
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except ValueError as e:  # понятная ошибка входа (битый архив и т.п.) -> 400, а не 500
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        LOG.exception("analyze failed")
        (job_dir / "error.txt").write_text(f"{type(e).__name__}: {e}", encoding="utf-8")
        raise HTTPException(500, f"Внутренняя ошибка обработки ({type(e).__name__}). Повторите попытку; если ошибка повторяется — сообщите администратору, код запроса {job}.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/api/batch")
def batch(req: BatchRequest):
    """Пакетная обработка папки/архива, доступного внутри контейнера (например,
    смонтированного тома /data/input). Пишет CSV по указанному пути внутри OUTPUT_DIR
    (по умолчанию OUTPUT_DIR/results_<время>.csv)."""
    src = Path(req.input_dir)
    if not src.exists():
        raise HTTPException(404, f"input_dir not found: {src}")
    if req.output_csv:
        out_csv = Path(req.output_csv)
        if not out_csv.is_absolute():
            out_csv = OUTPUT_DIR / out_csv
        out_csv = out_csv.resolve()
        if OUTPUT_DIR not in out_csv.parents:
            raise HTTPException(400, f"output_csv должен лежать внутри {OUTPUT_DIR}")
    else:
        out_csv = OUTPUT_DIR / f"results_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    if out_csv.suffix.lower() == ".xlsx":
        req.xlsx, out_csv = True, out_csv.with_suffix(".csv")
    try:
        eng = engine()
        t0 = time.perf_counter()
        with _RUN_LOCK:
            rows = eng.run(src, out_csv, debug_csv=out_csv.with_name(out_csv.stem + "_debug.csv"),
                           xlsx=req.xlsx, limit=req.limit)
        problems = validate_output_csv(out_csv, eng.cfg)
        return {
            "summary": {**_summary(rows, eng.cfg), "wall_time_s": round(time.perf_counter() - t0, 3)},
            "format_check": "OK" if not problems else problems,
            "output_csv": str(out_csv),
            "output_xlsx": str(out_csv.with_suffix(".xlsx")) if req.xlsx else None,
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        LOG.exception("batch failed")
        raise HTTPException(500, f"{type(e).__name__}: {e}")


def _safe_job_file(job: str, name: str) -> Path:
    if not JOB_RE.match(job or ""):
        raise HTTPException(404, "job not found")
    safe = Path(name).name
    if not safe or safe.startswith("."):
        raise HTTPException(404, "file not found")
    job_dir = (JOBS_DIR / job).resolve()
    for cand in (job_dir / safe, job_dir / "bonus" / safe):
        cand = cand.resolve()
        if job_dir in cand.parents and cand.is_file():
            return cand
    raise HTTPException(404, "file not found")


@app.get("/api/results/{job}/{name}")
def download_job_file(job: str, name: str):
    """Файл результата конкретного запроса: results.csv, results_debug.csv, results.xlsx,
    summary.json или бонус-файлы (overlay PNG, DICOM SR, ROI PNG)."""
    p = _safe_job_file(job, name)
    return FileResponse(str(p), filename=p.name)


@app.get("/api/jobs/{job}")
def job_card(job: str):
    """Карточка запроса (summary.json): сводка, строки без картинок, ссылки на файлы."""
    p = _safe_job_file(job, "summary.json")
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


@app.get("/api/jobs")
def jobs_list(limit: int = 50):
    """История запросов (последние N): job_id, время, сводка. Для кабинета врача/лаборанта."""
    items = []
    for d in sorted(JOBS_DIR.iterdir(), key=lambda x: x.name, reverse=True):
        if not d.is_dir() or not JOB_RE.match(d.name):
            continue
        sj = d / "summary.json"
        if sj.exists():
            try:
                c = json.loads(sj.read_text(encoding="utf-8"))
                items.append({"job_id": d.name, "created_at": c.get("created_at"), "summary": c.get("summary"),
                              "format_check": c.get("format_check"), "n_rows": len(c.get("rows", []))})
            except Exception:  # noqa: BLE001
                items.append({"job_id": d.name, "error": "summary unreadable"})
        elif (d / "error.txt").exists():
            items.append({"job_id": d.name, "error": (d / "error.txt").read_text(encoding="utf-8")[:200]})
        if len(items) >= max(1, min(limit, 500)):
            break
    return {"jobs": items}


@app.get("/api/results/{name}")
def download(name: str):
    """Совместимость: файлы результатов /api/batch в корне OUTPUT_DIR (только results_*.csv/.xlsx)."""
    safe = Path(name).name
    if not re.match(r"^results_[0-9A-Za-z_\-]+\.(csv|xlsx)$", safe):
        raise HTTPException(404, "file not found")
    p = (OUTPUT_DIR / safe).resolve()
    if p.parent != OUTPUT_DIR or not p.is_file():
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
