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
import secrets
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
    write_results, validate_output_csv, config_hash as _cfg_hash, __version__ as PIPELINE_VERSION,
    unique_path,
)

try:
    from fastapi import FastAPI, File, Header, HTTPException, UploadFile
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
# Результаты запроса принадлежат тому, кто его загрузил: при загрузке выдаётся одноразовый
# код доступа (job_token). Без него карточка и файлы запроса не отдаются, даже если код
# запроса известен. Список всех запросов закрыт и включается только админским ключом.
JOB_TOKEN_FILE = ".job_token"
ADMIN_KEY = os.environ.get("DENSITO_ADMIN_KEY", "").strip()
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
            # один DICOM SR на исследование (включая норму) -> <папка запроса>/sr/<study_uid>_SR.dcm
            sr_study=enable_bonus and os.environ.get("DENSITO_SR_STUDY", "1") != "0",
            # extras (предупреждения): белые линии, OOD-gate, эндопротез, когерентность исследования
            # -> <job>/results_extras.csv и details.extras; 9 колонок не затрагивает
            extras=os.environ.get("DENSITO_EXTRAS", "1") != "0",
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
            "intended_use": eng.cfg.get("intended_use", {}),
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
        ("feat_abs_shaft_angle_deg", "Наклон диафиза бедра", "°", 1.0, 1),
        ("feat_edge_distance_mm", "Расстояние от кости до бокового края", "мм", 1.0, 1),
        ("feat_bone_area_ratio", "Доля кости в кадре", "%", 100.0, 1),
        ("feat_scan_length_mm", "Длина скана", "мм", 1.0, 0),
        ("feat_shaft_len_below_troch_mm", "Диафиз ниже вертелов", "мм", 1.0, 0),
        ("feat_lesser_troch_prominence_mm", "Выступ малого вертела", "мм", 1.0, 1),
        ("feat_metal_metal_area_mm2", "Площадь плотных включений", "мм²", 1.0, 0),
        ("bonus_roi_deficit_mm", "Недостаток поля сканирования", "мм", 1.0, 0),
    ],
}


def _meas_key(debug_key: str) -> str:
    """feat_axis_angle_deg -> axis_angle_deg; bonus_roi_deficit_mm -> roi_deficit_mm (ключ для UI-норм)."""
    for pref in ("feat_", "bonus_"):
        if debug_key.startswith(pref):
            return debug_key[len(pref):]
    return debug_key


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
            # К3: запас до порога, зона «не уверен», Platt-вероятность нарушения по критерию (только UI)
            "margin": (lambda v: None if v is None else round(v, 4))(_num(dbg.get(f"{c}_margin"))),
            "uncertain": bool(_num(dbg.get(f"{c}_uncertain")) or 0),
            "p_cal": (lambda v: None if v is None else round(v, 3))(_num(dbg.get(f"{c}_p_cal"))),
        })
    # К3: уровень риска и «нужна проверка» — правило calibration_utils.risk_level:
    # средний — не уверен хотя бы один критерий (или отказ); высокий — class=1 и уверен; низкий — class=0 и уверен.
    needs_review = bool(_num(dbg.get("needs_review")) or 0) or is_fail
    risk = dbg.get("risk_level") or ("средний" if needs_review else ("высокий" if str(row.get("quality_class")) == "1" else "низкий"))
    meas = []
    for key, title, unit, mult, nd in MEASUREMENTS.get("spine" if region == "spine" else "hip", []):
        v = _num(dbg.get(key))
        if v is not None:
            meas.append({"key": _meas_key(key), "title": title, "value": round(v * mult, nd), "unit": unit})
    if is_fail:
        action, action_code = "Проверить вручную: файл не обработан", "manual"
    elif needs_review:
        action, action_code = "Пограничный случай: проверить вручную", "review_uncertain"
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
        # паспортный шаг пикселя (мм): UI пересчитывает относительные смещения в мм, если тег PixelSpacing отсутствовал
        "pixel_spacing_mm_default": [cfg.get("pixel_spacing_mm", {}).get("y"), cfg.get("pixel_spacing_mm", {}).get("x")],
        "roi": {
            "needs_correction": dbg.get("bonus_roi_needs_correction"),
            "reason": dbg.get("bonus_roi_reason"),
            "deficit_mm": _num(dbg.get("bonus_roi_deficit_mm")),
        },
        "action": action,
        "action_code": action_code,
        "risk_level": risk,
        "needs_review": needs_review,
        "uncertain_criteria": [c for c in str(dbg.get("uncertain_criteria") or "").split(";") if c],
    }


def _json_safe(d: Dict[str, Any]) -> Dict[str, Any]:
    """numpy/NaN -> обычные типы JSON."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if hasattr(v, "item"):
            v = v.item()
        if isinstance(v, float) and v != v:
            v = None
        out[str(k)] = v
    return out


def _config_hash(cfg: Dict[str, Any]) -> str:
    return _cfg_hash(cfg)  # та же формула, что записывается в DICOM SR (inference.config_hash)


def _study_sr_urls(eng: DensitoInference, job: str) -> Dict[str, str]:
    """{study_uid: ссылка на SR исследования} — только файлы, реально записанные этим прогоном."""
    out: Dict[str, str] = {}
    for study_uid, p in (getattr(eng, "last_study_sr", {}) or {}).items():
        p = Path(p)
        if p.exists():
            out[study_uid] = f"/api/results/{job}/{p.name}"
    return out


def _safe_upload_rel(filename: Optional[str]) -> str:
    """Относительный путь для файла загрузки.

    Сохраняет структуру папок, которую прислал браузер при загрузке каталога, и режет
    опасное: абсолютные пути, `..`, букву диска Windows, служебные элементы и
    управляющие символы. Благодаря этому два файла с одинаковым базовым именем из
    разных папок остаются двумя файлами и дают две строки выгрузки (ТЗ п. 2.5),
    а path_to_study показывает тот же путь, который видит пользователь.
    """
    raw = (filename or "").replace("\\", "/")
    parts: List[str] = []
    for p in raw.split("/"):
        p = re.sub(r"[\x00-\x1f\x7f]", "", p).strip()
        if p in ("", ".", "..", "__MACOSX", ".DS_Store"):
            continue
        parts.append(p)
    if parts and len(parts[0]) == 2 and parts[0][1] == ":":
        parts = parts[1:]          # "C:" в начале пути Windows
    parts = [p[:120] for p in parts if p]
    if not parts:
        return f"upload_{uuid.uuid4().hex[:6]}.dcm"
    return "/".join(parts[-8:])    # разумный предел глубины


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
            rel = _safe_upload_rel(uf.filename)
            target = tmp / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target = unique_path(target)
                LOG.warning("Upload %r duplicates an earlier name, stored as %r", rel, target.name)
            # читаем кусками и останавливаемся на лимите, а не после полной загрузки в память
            n_file = 0
            with open(target, "wb") as dst:
                while True:
                    chunk = await uf.read(1 << 20)
                    if not chunk:
                        break
                    total += len(chunk)
                    n_file += len(chunk)
                    if total > limit:
                        raise HTTPException(413, f"Объём загрузки превышает {MAX_UPLOAD_MB:.0f} МБ. "
                                                 f"Разбейте партию на части.")
                    dst.write(chunk)
            if n_file == 0:
                raise HTTPException(400, f"Файл «{rel}» пустой.")
        eng, out_csv, rows, debug_rows = await run_in_threadpool(_run_job, job, tmp, job_dir, xlsx)
        problems = validate_output_csv(out_csv, eng.cfg)
        study_sr = _study_sr_urls(eng, job)
        extras_rows = list(getattr(eng, "last_extras_rows", []) or [])
        rows_out = []
        for i, r in enumerate(rows):
            rb = _attach_bonus(r, job)
            if str(r.get("study_uid") or "") in study_sr:
                rb["study_sr_download"] = study_sr[str(r.get("study_uid"))]
            dbg = debug_rows[i] if i < len(debug_rows) else {}
            try:
                rb["details"] = _details(r, dbg, eng.cfg)
                if i < len(extras_rows) and isinstance(extras_rows[i], dict):
                    rb["details"]["extras"] = _json_safe(extras_rows[i])
            except Exception as e:  # noqa: BLE001 — детали не должны ломать ответ
                LOG.warning("details failed for row %d: %s", i, e)
            rows_out.append(rb)
        xlsx_path = out_csv.with_suffix(".xlsx")
        has_xlsx = bool(xlsx and xlsx_path.exists())
        job_token = _issue_job_token(job_dir)
        resp = {
            "job_id": job,
            "job_token": job_token,
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
            # один DICOM SR на исследование (включая норму): {study_uid: url}
            "study_sr": study_sr,
            "result_extras_csv_url": (f"/api/results/{job}/results_extras.csv"
                                      if (out_csv.parent / "results_extras.csv").exists() else None),
            "rows": rows_out,
            "csv": _rows_to_csv_text(rows, eng.cfg),
        }
        # краткая карточка запроса для истории (без base64-картинок и без кода доступа)
        try:
            card = {k: v for k, v in resp.items() if k not in ("rows", "csv", "job_token")}
            card["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            card["rows"] = [{k: v for k, v in r.items() if not str(k).endswith("_base64")} for r in rows_out]
            (job_dir / "summary.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            LOG.warning("summary.json failed: %s", e)
        return _with_token(resp, job, job_token)
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
            "study_sr": dict(getattr(eng, "last_study_sr", {}) or {}),
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        LOG.exception("batch failed")
        raise HTTPException(500, f"{type(e).__name__}: {e}")


def _issue_job_token(job_dir: Path) -> str:
    """Создать код доступа к запросу. Файл начинается с точки — _safe_job_file его не отдаёт."""
    tok = secrets.token_urlsafe(24)
    p = job_dir / JOB_TOKEN_FILE
    p.write_text(tok, encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except OSError:  # pragma: no cover — файловая система без прав
        pass
    return tok


def _check_job_access(job: str, given: str) -> None:
    """Пустить к результатам запроса только по его коду доступа (или админскому ключу)."""
    if not JOB_RE.match(job or ""):
        raise HTTPException(404, "job not found")
    given = (given or "").strip()
    if ADMIN_KEY and given and secrets.compare_digest(given, ADMIN_KEY):
        return
    tp = JOBS_DIR / job / JOB_TOKEN_FILE
    if not tp.is_file():
        raise HTTPException(404, "job not found")
    real = tp.read_text(encoding="utf-8").strip()
    if not given or not secrets.compare_digest(given, real):
        raise HTTPException(403, "Нужен код доступа к запросу (job_token): он выдаётся в ответе на загрузку "
                                 "и передаётся как ?t=<код> или заголовком X-Job-Token.")


def _with_token(obj: Any, job: str, tok: str) -> Any:
    """Добавить код доступа во все ссылки на файлы этого запроса (включая вложенные)."""
    pref = f"/api/results/{job}/"
    if isinstance(obj, str):
        return f"{obj}?t={tok}" if obj.startswith(pref) and "?" not in obj else obj
    if isinstance(obj, dict):
        return {k: _with_token(v, job, tok) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_with_token(v, job, tok) for v in obj]
    return obj


def _safe_job_file(job: str, name: str) -> Path:
    if not JOB_RE.match(job or ""):
        raise HTTPException(404, "job not found")
    safe = Path(name).name
    if not safe or safe.startswith("."):
        raise HTTPException(404, "file not found")
    job_dir = (JOBS_DIR / job).resolve()
    for cand in (job_dir / safe, job_dir / "bonus" / safe, job_dir / "sr" / safe):
        cand = cand.resolve()
        if job_dir in cand.parents and cand.is_file():
            return cand
    raise HTTPException(404, "file not found")


@app.get("/api/results/{job}/{name}")
def download_job_file(job: str, name: str, t: Optional[str] = None,
                      x_job_token: Optional[str] = Header(None)):
    """Файл результата конкретного запроса: results.csv, results_debug.csv, results.xlsx,
    summary.json, SR исследования (sr/<study_uid>_SR.dcm) или бонус-файлы (overlay PNG, SR снимка, ROI PNG).
    Нужен код доступа к запросу: ?t=<job_token> или заголовок X-Job-Token."""
    _check_job_access(job, x_job_token or t)
    p = _safe_job_file(job, name)
    return FileResponse(str(p), filename=p.name)


@app.get("/api/jobs/{job}")
def job_card(job: str, t: Optional[str] = None, x_job_token: Optional[str] = Header(None)):
    """Карточка запроса (summary.json): сводка, строки без картинок, ссылки на файлы.
    Нужен код доступа к запросу: ?t=<job_token> или заголовок X-Job-Token."""
    given = (x_job_token or t or "").strip()
    _check_job_access(job, given)
    card = json.loads(_safe_job_file(job, "summary.json").read_text(encoding="utf-8"))
    return JSONResponse(_with_token(card, job, given))


@app.get("/api/jobs")
def jobs_list(limit: int = 50, t: Optional[str] = None, x_admin_key: Optional[str] = Header(None)):
    """Список всех запросов сервиса. Закрыт: отдаётся только по админскому ключу
    (DENSITO_ADMIN_KEY, заголовок X-Admin-Key). Кабинет ведёт свою историю в браузере —
    по кодам запросов и кодам доступа, полученным при загрузке, — поэтому один пользователь
    не видит запросы другого."""
    given = (x_admin_key or t or "").strip()
    if not ADMIN_KEY or not given or not secrets.compare_digest(given, ADMIN_KEY):
        raise HTTPException(403, "Список запросов закрыт: запросы видны только тому, кто их загрузил "
                                 "(история ведётся в браузере). Администратору — ключ X-Admin-Key.")
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


# --------------------------------------------------------------------------- #
# Веб-интерфейс внутри образа
#
# Организаторы требуют полностью локальной работы без обращений к внешним API, и решение
# разворачивает техгруппа заказчика сама. Значит кабинет должен ехать В ОБРАЗЕ, а не жить
# только на нашем демо-стенде. web/index.html ходит в API ОТНОСИТЕЛЬНЫМИ путями (/api/...),
# поэтому при раздаче с корня того же сервера он работает без единой правки в самом HTML.
# Шрифты и картинки локальные (web/assets), CDN нет — страница открывается без интернета.
# Каталог переопределяется DENSITO_WEB_DIR; если его нет — API работает как раньше, без UI.
# --------------------------------------------------------------------------- #
WEB_DIR = Path(os.environ.get("DENSITO_WEB_DIR", PROJECT_ROOT / "web")).resolve()


@app.get("/", include_in_schema=False)
def web_index():
    """Лендинг и кабинет врача (один файл, без CDN)."""
    p = WEB_DIR / "index.html"
    if not p.is_file():
        raise HTTPException(404, "web UI not bundled in this image")
    return FileResponse(str(p), media_type="text/html; charset=utf-8")


@app.get("/assets/{path:path}", include_in_schema=False)
def web_assets(path: str):
    """Статика кабинета: шрифты, PNG кейсов и демо, actions.json, demo_result.json."""
    if ".." in path or path.startswith("/"):
        raise HTTPException(404, "file not found")
    p = (WEB_DIR / "assets" / path).resolve()
    try:
        p.relative_to(WEB_DIR / "assets")
    except ValueError:
        raise HTTPException(404, "file not found")
    if not p.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(str(p))


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
