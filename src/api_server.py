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
GET  /api/results/{job}/summary      — сводка по партии для заведующего/старшего лаборанта (JSON,
     .../summary.md, .../summary.csv     schema/department_summary.schema.json): доли с ДИ Уилсона по
                                       области, типу нарушения, аппарату (хэш), дате; без персональных данных.
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
    unique_path, failure_quality_prob,
)
from region_support import check_file as _region_check  # noqa: E402

# Сводка по партии (идея «в»): библиотека лежит в tools/department_summary.py (тот же код, что и CLI),
# в образ tools/ копируется целиком. Без неё сервис работает, маршруты сводки отвечают 503.
try:
    import importlib.util as _ilu
    _DS_SPEC = _ilu.spec_from_file_location("department_summary", PROJECT_ROOT / "tools" / "department_summary.py")
    department_summary = _ilu.module_from_spec(_DS_SPEC)  # type: ignore[arg-type]
    _DS_SPEC.loader.exec_module(department_summary)  # type: ignore[union-attr]
except Exception as _e:  # noqa: BLE001
    department_summary = None  # type: ignore[assignment]
    logging.getLogger("densito.api").warning("department_summary недоступен: %s", _e)

try:
    from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
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
REVIEW_DIR = OUTPUT_DIR / "review"       # ответы слепой ревизии рентгенолога
MAX_REVIEW_BYTES = 2 * 1024 * 1024
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
SC_DIR = BONUS_DIR / "sc"
SEG_DIR = BONUS_DIR / "seg"   # бонус «сегментация»: DICOM SEG + PNG-маска + JSON контуров (src/segmentation_export.py)
for _d in (VIZ_DIR, SR_DIR, ROI_DIR, SC_DIR, SEG_DIR):
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
            # серия с визуализацией как DICOM SC (бонус ТЗ 2.6) -> <папка запроса>/sc/*.dcm
            sc_dir=SC_DIR if enable_bonus else None,
            # SR на каждый снимок выключен: на исследование пишется один SR (см. inference.sr_per_image)
            sr_per_image=os.environ.get("DENSITO_SR_PER_IMAGE", "0") == "1",
            # один DICOM SR на исследование (включая норму) -> <папка запроса>/sr/<study_uid>_SR.dcm
            sr_study=enable_bonus and os.environ.get("DENSITO_SR_STUDY", "1") != "0",
            # extras (предупреждения): белые линии, OOD-gate, эндопротез, когерентность исследования
            # -> <job>/results_extras.csv и details.extras; 9 колонок не затрагивает
            extras=os.environ.get("DENSITO_EXTRAS", "1") != "0",
            # экспорт сегментации структур (SEG/PNG/JSON) -> <папка запроса>/bonus/row####_seg.*;
            # выключается DENSITO_SEG=0, на 9 колонок не влияет
            seg_dir=SEG_DIR if (enable_bonus and os.environ.get("DENSITO_SEG", "1") != "0") else None,
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


# Бонус-файлы запроса: (ключ в debug-строке движка, суффикс имени в папке запроса)
BONUS_KINDS = (("bonus_overlay_png", "_overlay.png"),
               ("bonus_sr_dcm", "_sr.dcm"),
               ("bonus_roi_png", "_roi_correction.png"),
               ("bonus_overlay_dcm", "_overlay.dcm"),
               ("bonus_seg_dcm", "_seg.dcm"),
               ("bonus_seg_png", "_seg.png"),
               ("bonus_seg_json", "_seg.json"))
# Ключи строки ответа с сегментацией (не начинаются с bonus_, поэтому снимаются отдельно при отказе по области)
SEG_ROW_KEYS = ("seg_download", "seg_png", "seg_json_download")


def bonus_row_prefix(i: int) -> str:
    """Имя бонус-файла в папке запроса — по номеру строки результата, а не по имени входного файла."""
    return f"row{i:04d}"


def _collect_bonus(job_dir: Path, debug_rows: List[Dict[str, Any]]) -> None:
    """Переносит бонус-файлы (overlay PNG, SC, SR, ROI) текущего прогона в папку запроса.

    Пути берутся из debug-строк движка (он же их и создал), а НЕ угадываются по имени
    загруженного файла: иначе в папку запроса попадали одноимённые файлы прошлых прогонов
    (утечка между запросами), а файлы текущего прогона оставались в общей папке.
    Имя в папке запроса — по номеру строки результата, поэтому совпадения имён невозможны.
    """
    dst = job_dir / "bonus"
    dst.mkdir(parents=True, exist_ok=True)
    for i, dbg in enumerate(debug_rows or []):
        if not isinstance(dbg, dict):
            continue
        for key, suf in BONUS_KINDS:
            src_path = str(dbg.get(key) or "").strip()
            if not src_path:
                continue
            src = Path(src_path)
            if not src.is_file():
                continue
            try:
                shutil.move(str(src), str(dst / f"{bonus_row_prefix(i)}{suf}"))
            except Exception:  # noqa: BLE001
                pass


def _region_support(row: Dict[str, Any], dbg: Dict[str, Any], tmp: Path) -> tuple:
    """Поддерживается ли область исследования. Пакетный путь (DensitoInference.process_file) уже
    проверил файл до классификации и записал итог в debug (region_supported / region_support_reason):
    отказ оттуда берётся как есть. Здесь — повторная проверка по заголовку для строк без этого итога
    (например, Failure до чтения пикселей). Любая ошибка проверки трактуется как «поддерживается» —
    отказ выдумывать нельзя."""
    if isinstance(dbg, dict) and str(dbg.get("region_supported", "")) in ("0", "False"):
        return False, str(dbg.get("region_support_reason") or "")
    try:
        rel = str(row.get("path_to_study") or "")
        cand = [Path(tmp) / rel, Path(rel)]
        src = next((c for c in cand if c.exists()), None)
        rr = int((dbg or {}).get("rows") or 0)
        cc = int((dbg or {}).get("cols") or 0)
        if src is None:
            from region_support import check_tags as _rt
            return _rt({}, cols=cc, rows=rr)
        return _region_check(src, rows=rr, cols=cc)
    except Exception as e:  # noqa: BLE001
        LOG.warning("проверка области не выполнена: %s", e)
        return True, ""


def _mark_unsupported(row: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """Официальные поля переводятся в ту же конвенцию, что и при сбое обработки: класс 0,
    тип нарушения пустой, статус — отказ. Порядок и состав колонок не меняются."""
    out = cfg["output"]
    row["quality_class"] = 0
    row["violation_type"] = ""
    row["quality_prob"] = failure_quality_prob(out)  # строго < 0.5 (класс 0)
    row["processing_status"] = out["status_failure"]


def _attach_bonus(row: Dict[str, Any], job: str, idx: int) -> Dict[str, Any]:
    """Добавляет в строку бонус-выходы для UI/экспертного просмотра: overlay PNG (base64,
    чтобы сразу отрисовать в браузере), ссылки на DICOM SC и SR, ROI-диагностику. Файлы ищутся
    по номеру строки (см. _collect_bonus), поэтому строка не может получить файл другого
    запроса или другого снимка. Ссылки ведут только в папку этого запроса. Официальные поля
    не меняются."""
    out = dict(row)
    bdir = JOBS_DIR / job / "bonus"
    pref = bonus_row_prefix(idx)
    viz = bdir / f"{pref}_overlay.png"
    if viz.exists():
        try:
            out["bonus_overlay_png_base64"] = base64.b64encode(viz.read_bytes()).decode("ascii")
        except Exception:  # noqa: BLE001
            pass
    sr = bdir / f"{pref}_sr.dcm"
    if sr.exists():
        out["bonus_sr_dcm_download"] = f"/api/results/{job}/{sr.name}"
    sc = bdir / f"{pref}_overlay.dcm"
    if sc.exists():
        out["bonus_overlay_dcm_download"] = f"/api/results/{job}/{sc.name}"
    roi = bdir / f"{pref}_roi_correction.png"
    if roi.exists():
        try:
            out["bonus_roi_png_base64"] = base64.b64encode(roi.read_bytes()).decode("ascii")
        except Exception:  # noqa: BLE001
            pass
    # сегментация структур: SEG (.dcm), PNG-маска для слоя в кабинете, JSON с контурами — только ссылки
    # внутрь папки этого запроса (код доступа добавляет _with_token)
    seg_dcm = bdir / f"{pref}_seg.dcm"
    if seg_dcm.exists():
        out["seg_download"] = f"/api/results/{job}/{seg_dcm.name}"
    seg_png = bdir / f"{pref}_seg.png"
    if seg_png.exists():
        out["seg_png"] = f"/api/results/{job}/{seg_png.name}"
    seg_json = bdir / f"{pref}_seg.json"
    if seg_json.exists():
        out["seg_json_download"] = f"/api/results/{job}/{seg_json.name}"
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


# C1: подписи признаков, которые реально подаются в модели контура A (feature_cols файлов моделей;
# значения — из debug-поля <crit>_model_features, вариант предобработки критерия).
# (подпись, единица, множитель, знаков)
MODEL_FEATURE_TITLES = {
    "scan_length_mm": ("Длина скана", "мм", 1.0, 0),
    "shaft_len_below_troch_mm": ("Длина диафиза ниже вертела", "мм", 1.0, 0),
    "femur_solidity": ("Компактность контура бедра", "", 1.0, 3),
    "shaft_width_mm": ("Ширина диафиза", "мм", 1.0, 1),
    "abs_shaft_angle_deg": ("Наклон диафиза бедра", "°", 1.0, 1),
    "merge_height_mm": ("Высота слияния диафиза с тазом", "мм", 1.0, 1),
    "medial_neck_extent_mm": ("Медиальный выступ шейки бедра", "мм", 1.0, 1),
    "axis_angle_deg": ("Угол оси позвоночника к вертикали кадра", "°", 1.0, 1),
    "center_offset_ratio": ("Смещение позвоночника от центра кадра", "% ширины", 100.0, 1),
    "bone_width_ratio": ("Ширина костной области", "% ширины", 100.0, 1),
    "synth_pos_logit": ("Признак укладки по изображению (перенос из контура B)", "", 1.0, 2),
    "metal_metal_area_mm2": ("Площадь плотных включений", "мм²", 1.0, 0),
    "metal_metal_max_intensity_gap": ("Контраст включений к кости", "сигм", 1.0, 2),
}
DECISION_SOURCE_TEXT = {
    "geom_and_image": "решение по измерениям и по изображению",
    "geom": "решение по измерениям (контур A)",
    "image": "решение по изображению, измерения в норме — проверьте снимок визуально",
    "rule": "решение по резервному правилу (моделей нет)",
}


def _model_features(dbg: Dict[str, Any], crit: str) -> Dict[str, Any]:
    """{variant, items: [{key, title, unit, value, raw}]} из debug <crit>_model_features; пусто, если поля нет."""
    raw = dbg.get(f"{crit}_model_features")
    try:
        obj = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw if isinstance(raw, dict) else None)
    except (TypeError, ValueError):
        obj = None
    if not obj:
        return {"variant": None, "items": []}
    items = []
    for k, v in (obj.get("values") or {}).items():
        title, unit, mult, nd = MODEL_FEATURE_TITLES.get(k, (k, "", 1.0, 3))
        vv = _num(v)
        items.append({"key": k, "title": title, "unit": unit,
                      "value": None if vv is None else round(vv * mult, nd), "raw": vv})
    return {"variant": obj.get("variant"), "items": items}


def _decision_source(dbg: Dict[str, Any], crit: str, flag: bool) -> Optional[str]:
    """Чем поставлен флаг критерия: geom_and_image / geom / image / rule; None — флага нет.
    Скор = взвешенное среднее рангов контуров A (измерения) и B (изображение); контур «за нарушение»,
    если его ранг сам по себе не ниже порога (при равных весах хотя бы один контур не ниже порога)."""
    if not flag:
        return None
    method = str(dbg.get(f"{crit}_method") or "")
    if method == "fallback_rule":
        return "rule"
    if method == "geom_only":
        return "geom"
    if method == "emb_only":
        return "image"
    thr = _num(dbg.get(f"{crit}_threshold"))
    rg = _num(dbg.get(f"{crit}_rank_geom"))
    rg = rg if rg is not None else _num(dbg.get(f"{crit}_p_geom"))
    re_ = _num(dbg.get(f"{crit}_rank_emb"))
    re_ = re_ if re_ is not None else _num(dbg.get(f"{crit}_p_emb"))
    if thr is None or rg is None or re_ is None:
        return None
    a, b = rg >= thr, re_ >= thr
    if a and b:
        return "geom_and_image"
    if b and not a:
        return "image"
    return "geom"


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
            # C1 (добавочные поля): ранги контуров, признаки модели и источник решения
            "rank_geom": (lambda v: None if v is None else round(v, 3))(_num(dbg.get(f"{c}_rank_geom"))),
            "rank_emb": (lambda v: None if v is None else round(v, 3))(_num(dbg.get(f"{c}_rank_emb"))),
            "model_features": _model_features(dbg, c),
            "decision_source": _decision_source(dbg, c, bool(flag) if flag not in (None, "") else False),
        })
        crits[-1]["decision_source_text"] = DECISION_SOURCE_TEXT.get(crits[-1]["decision_source"] or "")
        try:
            rel = (score - thr) / (1.0 - thr) if (score is not None and thr is not None and thr < 1.0) else None
        except ZeroDivisionError:
            rel = None
        crits[-1]["relative_margin"] = None if rel is None else round(rel, 4)
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
        # C1: развилка по области интереса бедра (код причины; класс не меняется) — только при флаге *_roi
        "hip_roi_reason": _hip_roi_reason(region, dbg),
        "lateral_margin_mm": (lambda v: None if v is None else round(v, 1))(_num(dbg.get("feat_lateral_margin_mm"))),
        "action": action,
        "action_code": action_code,
        "risk_level": risk,
        "needs_review": needs_review,
        "uncertain_criteria": [c for c in str(dbg.get("uncertain_criteria") or "").split(";") if c],
    }


def _hip_roi_reason(region: str, dbg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if region not in ("right_hip", "left_hip"):
        return None
    crit = "rh_roi" if region == "right_hip" else "lh_roi"
    if not (_num(dbg.get(f"{crit}_flag")) or 0):
        return None
    try:
        import extras as _ex
        return _ex.hip_roi_reason(dbg, True)
    except Exception as e:  # noqa: BLE001
        LOG.warning("hip_roi_reason failed: %s", e)
        return None


def _study_priority(rows: List[Dict[str, Any]], debug_rows: List[Dict[str, Any]], cfg: Dict[str, Any],
                    supported: List[bool]) -> Dict[str, Dict[str, Any]]:
    """C1: {study_uid: главное действие на визит} — dicom_sr.study_priority_action (то же правило, что в SR).
    Снимок вне поддерживаемых областей даёт только общее замечание (его критерии не оцениваются)."""
    try:
        from dicom_sr import study_priority_action
        from inference import _criteria_for_priority, _roi_route_for_priority
    except Exception as e:  # noqa: BLE001
        LOG.warning("study priority unavailable: %s", e)
        return {}
    groups: Dict[str, List[Dict[str, Any]]] = {}
    sep = cfg["output"]["violation_separator"]
    for i, r in enumerate(rows):
        dbg = debug_rows[i] if i < len(debug_rows) else {}
        ok = supported[i] if i < len(supported) else True
        groups.setdefault(str(r.get("study_uid") or ""), []).append({
            "image_uid": str(r.get("image_uid") or ""),
            "anatomical_region": str(r.get("anatomical_region") or ""),
            "internal_region": (dbg or {}).get("internal_region") or "",
            "quality_class": int(r.get("quality_class") or 0),
            "violations": [v.strip() for v in str(r.get("violation_type") or "").split(sep) if v.strip()],
            "processing_status": str(r.get("processing_status") or ""),
            "criteria": _criteria_for_priority(cfg, dbg or {}) if ok else [],
            "roi_route": _roi_route_for_priority(cfg, dbg or {}) if ok else "",
        })
    out: Dict[str, Dict[str, Any]] = {}
    for suid, items in groups.items():
        if not suid:
            continue
        try:
            out[suid] = study_priority_action(items)
        except Exception as e:  # noqa: BLE001
            LOG.warning("study priority failed for %s: %s", suid, e)
    return out


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


def _study_completeness(eng: DensitoInference, rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """{study_uid: {"spine": bool, "hip": bool, "note": str|None}} — какие области пришли в исследовании.
    Считается по строкам этого запроса той же функцией, что и примечание в SR исследования; ошибка
    даёт пустой словарь, а не сбой ответа."""
    try:
        from inference import study_completeness_by_study
        return study_completeness_by_study(rows)
    except Exception as e:  # noqa: BLE001
        LOG.warning("study completeness failed: %s", e)
        return dict(getattr(eng, "last_study_completeness", {}) or {})


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
        _collect_bonus(job_dir, debug_rows)
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
        _write_device_tags(rows, tmp, job_dir)  # сводка по партии: только аппарат и дата, пока входные файлы ещё есть
        problems = validate_output_csv(out_csv, eng.cfg)
        study_sr = _study_sr_urls(eng, job)
        study_completeness = _study_completeness(eng, rows)
        extras_rows = list(getattr(eng, "last_extras_rows", []) or [])
        rows_out = []
        n_unsupported = 0
        # теги для журнала и для поиска в кабинете (ФИО — только маской «Иванова М. П.», дата исследования)
        try:
            reg_tags = registry_read_tags(rows, tmp)
        except Exception as e:  # noqa: BLE001
            LOG.warning("registry tags failed: %s", e)
            reg_tags = {}
        try:  # чистые кадры для слепой экспертной проверки (src/expert_review.py)
            expert_save_frames(rows, tmp, job_dir)
        except Exception as e:  # noqa: BLE001
            LOG.warning("expert frames failed: %s", e)
        supported: List[bool] = []
        for i, r in enumerate(rows):
            dbg = debug_rows[i] if i < len(debug_rows) else {}
            reg_ok, reg_reason = _region_support(r, dbg, tmp)
            supported.append(bool(reg_ok))
            if not reg_ok:
                n_unsupported += 1
                _mark_unsupported(r, eng.cfg)
                if isinstance(dbg, dict):
                    dbg["region_supported"] = 0
                    dbg["region_support_reason"] = reg_reason
                LOG.info("область не поддерживается: %s -> %s", r.get("path_to_study"), reg_reason)
            rb = _attach_bonus(r, job, i)
            _t = reg_tags.get(i, {})
            rb["patient"] = registry_mask_name(_t.get("PatientName", "")) if _t else None
            rb["study_date"] = registry_fmt_date(_t.get("StudyDate", "")) or None if _t else None
            rb["region_supported"] = bool(reg_ok)
            rb["region_support_reason"] = reg_reason
            if not reg_ok:
                # ни оверлея, ни отчётов: они описывали бы геометрию там, где оценка не выполняется
                for k in [k for k in list(rb) if str(k).startswith("bonus_")]:
                    rb.pop(k, None)
                for k in SEG_ROW_KEYS:   # сегментация тоже описывала бы структуры вне зоны оценки
                    rb.pop(k, None)
            if str(r.get("study_uid") or "") in study_sr:
                rb["study_sr_download"] = study_sr[str(r.get("study_uid"))]
            if str(r.get("study_uid") or "") in study_completeness:
                rb["study_completeness"] = study_completeness[str(r.get("study_uid"))]
            # идея 23: предложение коррекции области интереса (бедро) — требует подтверждения специалистом
            try:
                rb["roi_suggestion"] = _roi_suggestion(r, dbg, reg_ok)
            except Exception as e:  # noqa: BLE001
                LOG.warning("roi_suggestion failed for row %d: %s", i, e)
                rb["roi_suggestion"] = None
            try:
                rb["details"] = _details(r, dbg, eng.cfg)
                if i < len(extras_rows) and isinstance(extras_rows[i], dict):
                    rb["details"]["extras"] = _json_safe(extras_rows[i])
                    # C1 (З2): уникальные предупреждения исследования с числом (study_warnings не меняется)
                    try:
                        import extras as _ex
                        rb["details"]["study_warnings_unique"] = _ex.study_warning_items(
                            extras_rows[i].get("study_warnings") or "")
                    except Exception as e:  # noqa: BLE001
                        LOG.warning("study_warning_items failed for row %d: %s", i, e)
            except Exception as e:  # noqa: BLE001 — детали не должны ломать ответ
                LOG.warning("details failed for row %d: %s", i, e)
            rows_out.append(rb)
        # C1: главное действие на визит — по исследованию и в каждой строке исследования
        study_priority = _study_priority(rows, debug_rows, eng.cfg, supported)
        for rb in rows_out:
            sp = study_priority.get(str(rb.get("study_uid") or ""))
            if sp is not None:
                rb["study_priority"] = sp
        xlsx_path = out_csv.with_suffix(".xlsx")
        has_xlsx = bool(xlsx and xlsx_path.exists())
        if n_unsupported:
            # отчёт этого запроса приводим в соответствие с карточкой; пакетный путь не затронут
            write_results(rows, out_csv, eng.cfg, xlsx=has_xlsx)
            problems = validate_output_csv(out_csv, eng.cfg)
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
            # полнота исследования по областям: {study_uid: {"spine": bool, "hip": bool, "note": str|null}};
            # note заполнено, когда представлена только одна область из двух (тот же текст, что в SR)
            "study_completeness": study_completeness,
            # C1: главное действие на визит {study_uid: {status, action_code, text, others, ...}};
            # правило — максимальный относительный запас над порогом (то же, что «Приоритетное действие» в SR)
            "study_priority": study_priority,
            "result_extras_csv_url": (f"/api/results/{job}/results_extras.csv"
                                      if (out_csv.parent / "results_extras.csv").exists() else None),
            # идея 23: решения специалиста по предложенной области интереса (POST/GET, CSV)
            "decisions_url": f"/api/results/{job}/decisions",
            "decisions_csv_url": f"/api/results/{job}/decisions.csv",
            # сводка по партии (для заведующего/старшего лаборанта): JSON и Markdown, без персональных данных
            "department_summary_url": f"/api/results/{job}/summary",
            "department_summary_md_url": f"/api/results/{job}/summary.md",
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
        # журнал исследований отделения (src/registry.py): теги DICOM для поиска читаем, пока загрузка ещё на диске.
        # Ошибка журнала не влияет на ответ и на CSV по ТЗ.
        try:
            REGISTRY.index_rows(job, rows_out, reg_tags, card.get("created_at"))
        except Exception as e:  # noqa: BLE001
            LOG.warning("registry index failed: %s", e)
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
            "study_completeness": dict(getattr(eng, "last_study_completeness", {}) or {}),
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


# =========================================================================== #
# Идея 23: предложение коррекции области интереса бедра с подтверждением специалистом.
#
# 1) В строке ответа /api/analyze для бедра — roi_suggestion (по данным auto_roi.suggest_hip_roi через
#    debug-поля bonus_roi_*), для позвоночника — null. Источник всегда «предложение системы».
# 2) Решения специалиста хранятся в каталоге задачи в decisions.json (атомарная запись: временный файл
#    + os.replace под блокировкой). Доступ — по коду задачи, как у остальных /api/results/{job}/...
# 3) Эндпоинты: POST/GET /api/results/{job}/decisions, GET .../decisions.csv,
#    GET .../decisions_sr/{study_uid} (отдельный DICOM SR «Решение специалиста по области интереса»).
# Маршруты объявлены ДО общего /api/results/{job}/{name}, иначе «decisions» ушло бы в выдачу файла.
# =========================================================================== #
DECISIONS_FILE = "decisions.json"
DECISIONS_SR_SUBDIR = "decisions_sr"
DECISION_VALUES = ("подтверждено", "отклонено", "своя")
ROI_SUGGESTION_SOURCE = "предложение системы"
SPECIALIST_MAX_LEN = 80
COMMENT_MAX_LEN = 500
MAX_DECISIONS_PER_JOB = 1000
_DECISIONS_LOCK = threading.Lock()
ROI_REASON_TEXT_API = {
    "scan_too_short": "поле сканирования короче требуемого",
    "shaft_below_trochanter_too_short": "в кадр вошло мало диафиза ниже вертелов",
    "lateral_margin_below_threshold": "кость ближе к краю кадра, чем требует ТЗ (не менее 20 мм)",
}


def _parse_box(val: Any) -> Optional[List[int]]:
    """"x0,y0,x1,y1" или список из 4 чисел -> [x0, y0, x1, y1] (int) либо None."""
    if val is None:
        return None
    if isinstance(val, str):
        parts = [p.strip() for p in val.split(",") if p.strip() != ""]
    elif isinstance(val, (list, tuple)):
        parts = list(val)
    else:
        return None
    if len(parts) != 4:
        return None
    try:
        box = [int(round(float(p))) for p in parts]
    except (TypeError, ValueError):
        return None
    return box


def _box_mm(box: Optional[List[int]], spacing_yx: Optional[List[float]]) -> Optional[List[float]]:
    """Рамка в пикселях -> мм от левого верхнего угла кадра [x0, y0, x1, y1] (spacing = (мм/px по y, по x))."""
    if not box or not spacing_yx or len(spacing_yx) != 2:
        return None
    try:
        sy, sx = float(spacing_yx[0]), float(spacing_yx[1])
    except (TypeError, ValueError):
        return None
    if sy <= 0 or sx <= 0:
        return None
    return [round(box[0] * sx, 1), round(box[1] * sy, 1), round(box[2] * sx, 1), round(box[3] * sy, 1)]


def _roi_suggestion(row: Dict[str, Any], dbg: Dict[str, Any], region_ok: bool = True) -> Optional[Dict[str, Any]]:
    """Предложение коррекции области интереса для строки бедра; None для позвоночника, отказа
    по области, Failure или когда предложение не вычислялось (нет debug-полей bonus_roi_*)."""
    if not isinstance(dbg, dict) or not region_ok:
        return None
    region = str(dbg.get("internal_region") or "")
    if region not in ("right_hip", "left_hip", "hip"):
        return None
    if "bonus_roi_needs_correction" not in dbg or dbg.get("bonus_roi_error"):
        return None
    needs = dbg.get("bonus_roi_needs_correction")
    needs = bool(needs) if needs is not None and needs == needs else False  # NaN из CSV -> False
    spacing = None
    sp = dbg.get("bonus_roi_pixel_spacing_mm")
    if isinstance(sp, str) and "," in sp:
        try:
            spacing = [float(v) for v in sp.split(",")[:2]]
        except ValueError:
            spacing = None
    box = _parse_box(dbg.get("bonus_roi_box_px")) if needs else None
    ext = _parse_box(dbg.get("bonus_roi_ext_box_px")) if needs else None
    reason = dbg.get("bonus_roi_reason") if needs else None
    reason = str(reason) if reason and reason == reason else None
    deficit = _num(dbg.get("bonus_roi_deficit_mm")) if needs else None
    rows_n, cols_n = _num(dbg.get("rows")), _num(dbg.get("cols"))
    return {
        "needs_correction": needs,
        "box_px": box,                       # [x0, y0, x1, y1] в пикселях исходного кадра (обрезано кадром)
        "box_mm": _box_mm(box, spacing),     # то же в мм от левого верхнего угла кадра
        "extended_box_px": ext,              # рамка с учётом недостающей длины (может выходить за кадр)
        "deficit_mm": deficit,
        "reason": reason,
        "reason_text": ROI_REASON_TEXT_API.get(reason or "", None),
        "side": (str(dbg.get("bonus_roi_side")) if dbg.get("bonus_roi_side") else None) or None,
        "pixel_spacing_mm": spacing,         # [по y, по x]
        "image_size": [int(rows_n), int(cols_n)] if rows_n and cols_n else None,  # [rows, cols]
        "sop_class_uid": str(dbg.get("sop_class_uid") or "") or None,
        "source": ROI_SUGGESTION_SOURCE,
        "status": "требует подтверждения специалистом" if needs else "коррекция не требуется",
    }


def _job_dir_checked(job: str) -> Path:
    if not JOB_RE.match(job or ""):
        raise HTTPException(404, "job not found")
    d = (JOBS_DIR / job).resolve()
    if JOBS_DIR.resolve() not in d.parents or not d.is_dir():
        raise HTTPException(404, "job not found")
    return d


def load_decisions(job_dir: Path) -> List[Dict[str, Any]]:
    p = Path(job_dir) / DECISIONS_FILE
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    items = data.get("decisions") if isinstance(data, dict) else data
    return [d for d in (items or []) if isinstance(d, dict)]


def save_decisions_atomic(job_dir: Path, decisions: List[Dict[str, Any]]) -> Path:
    """Атомарная запись decisions.json: временный файл в том же каталоге + os.replace."""
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    target = job_dir / DECISIONS_FILE
    payload = {"version": 1, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "decisions": decisions}
    fd, tmp_name = tempfile.mkstemp(prefix=".decisions_", suffix=".tmp", dir=str(job_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target


def _summary_rows(job_dir: Path) -> List[Dict[str, Any]]:
    p = Path(job_dir) / "summary.json"
    if not p.is_file():
        return []
    try:
        card = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    rows = card.get("rows") if isinstance(card, dict) else None
    return [r for r in (rows or []) if isinstance(r, dict)]


def _clean_text(val: Any, max_len: int, field: str) -> str:
    if val is None:
        return ""
    if not isinstance(val, str):
        raise ValueError(f"поле {field} должно быть строкой")
    s = " ".join(val.replace("\r", " ").replace("\n", " ").split())
    if len(s) > max_len:
        raise ValueError(f"поле {field} длиннее {max_len} символов")
    return s


def validate_decision(body: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Проверить тело POST /api/results/{job}/decisions и собрать запись для decisions.json.
    rows — строки summary.json задачи (snapshot roi_suggestion берётся оттуда). ValueError -> 400."""
    if not isinstance(body, dict):
        raise ValueError("тело запроса должно быть JSON-объектом")
    image_uid = str(body.get("image_uid") or "").strip()
    if not image_uid or len(image_uid) > 128 or not re.match(r"^[0-9A-Za-z._:-]+$", image_uid):
        raise ValueError("image_uid обязателен (SOPInstanceUID снимка из строки ответа)")
    row = next((r for r in rows if str(r.get("image_uid") or "") == image_uid), None)
    if row is None:
        raise ValueError("снимок с таким image_uid в этой задаче не найден")
    sug = row.get("roi_suggestion")
    if not isinstance(sug, dict):
        raise ValueError("для этого снимка предложение области интереса не формировалось (не бедро или отказ)")
    decision = str(body.get("decision") or "").strip().lower()
    if decision not in DECISION_VALUES:
        raise ValueError("decision должно быть одним из: " + ", ".join(DECISION_VALUES))
    specialist = _clean_text(body.get("specialist"), SPECIALIST_MAX_LEN, "specialist")
    if not specialist:
        raise ValueError("укажите специалиста (должность или инициалы, без персональных данных)")
    comment = _clean_text(body.get("comment"), COMMENT_MAX_LEN, "comment")
    img_size = sug.get("image_size") or (row.get("details") or {}).get("image_size")
    spacing = sug.get("pixel_spacing_mm")
    suggested_box = _parse_box(sug.get("box_px"))
    final_box: Optional[List[int]] = None
    if decision == "своя":
        final_box = _parse_box(body.get("roi_box_px"))
        if final_box is None:
            raise ValueError("для решения «своя» нужен roi_box_px: [x0, y0, x1, y1] в пикселях исходного кадра")
        x0, y0, x1, y1 = final_box
        if x1 <= x0 or y1 <= y0:
            raise ValueError("roi_box_px: требуется x0 < x1 и y0 < y1")
        if x0 < 0 or y0 < 0:
            raise ValueError("roi_box_px: координаты не могут быть отрицательными")
        if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
            rows_n, cols_n = int(img_size[0]), int(img_size[1])
            if x1 > cols_n or y1 > rows_n:
                raise ValueError(f"roi_box_px выходит за кадр {cols_n}x{rows_n} px")
        if (x1 - x0) < 4 or (y1 - y0) < 4:
            raise ValueError("roi_box_px: рамка меньше 4 px по стороне")
    elif decision == "подтверждено":
        if suggested_box is None:
            raise ValueError("подтверждать нечего: система не предлагала коррекцию для этого снимка")
        final_box = suggested_box
    else:  # отклонено — итоговой рамки нет
        final_box = None
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    return {
        "decision_id": uuid.uuid4().hex[:12],
        "image_uid": image_uid,
        "study_uid": str(row.get("study_uid") or ""),
        "path_to_study": str(row.get("path_to_study") or ""),
        "sop_class_uid": sug.get("sop_class_uid"),
        "decision": decision,
        "roi_box_px": final_box,
        "roi_box_mm": _box_mm(final_box, spacing),
        "suggested_box_px": suggested_box,
        "suggested_box_mm": sug.get("box_mm"),
        "deficit_mm": sug.get("deficit_mm"),
        "reason": sug.get("reason"),
        "specialist": specialist,
        "comment": comment,
        "created_at": now,
        "source": ROI_SUGGESTION_SOURCE,
    }


DECISIONS_CSV_COLUMNS = ["decision_id", "created_at", "study_uid", "image_uid", "path_to_study", "decision",
                         "specialist", "comment", "suggested_box_px", "suggested_box_mm", "roi_box_px",
                         "roi_box_mm", "deficit_mm", "reason", "source"]


def decisions_to_csv(decisions: List[Dict[str, Any]]) -> str:
    """CSV решений (разделитель «;», как в остальных выгрузках; UTF-8 с BOM для Excel)."""
    import csv
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    w.writerow(DECISIONS_CSV_COLUMNS)
    for d in decisions:
        vals = []
        for c in DECISIONS_CSV_COLUMNS:
            v = d.get(c)
            if isinstance(v, (list, tuple)):
                v = ",".join(str(x) for x in v)
            vals.append("" if v is None else str(v))
        w.writerow(vals)
    return "\ufeff" + buf.getvalue()


def _decision_token(t: Optional[str], x_job_token: Optional[str]) -> str:
    return (x_job_token or t or "").strip()


@app.post("/api/results/{job}/decisions")
async def post_roi_decision(job: str, request: Request, t: Optional[str] = None,
                            x_job_token: Optional[str] = Header(None)):
    """Зафиксировать решение специалиста по предложенной области интереса (бедро).
    Тело JSON: image_uid, decision ∈ {подтверждено, отклонено, своя}, roi_box_px (при «своя»),
    specialist (должность/инициалы, без персональных данных), comment (необязательно)."""
    _check_job_access(job, _decision_token(t, x_job_token))
    job_dir = _job_dir_checked(job)
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "тело запроса должно быть JSON")
    with _DECISIONS_LOCK:
        rows = _summary_rows(job_dir)
        try:
            rec = validate_decision(body, rows)
        except ValueError as e:
            raise HTTPException(400, str(e))
        items = load_decisions(job_dir)
        if len(items) >= MAX_DECISIONS_PER_JOB:
            raise HTTPException(400, "превышен предел числа решений для задачи")
        items.append(rec)
        save_decisions_atomic(job_dir, items)
    return {"ok": True, "decision": rec, "n_decisions": len(items),
            "sr_url": _with_token(f"/api/results/{job}/decisions_sr/{rec['study_uid']}", job, _decision_token(t, x_job_token))
            if rec.get("study_uid") else None}


@app.get("/api/results/{job}/decisions")
def get_roi_decisions(job: str, t: Optional[str] = None, x_job_token: Optional[str] = Header(None)):
    """Список решений специалиста по задаче (в порядке записи)."""
    _check_job_access(job, _decision_token(t, x_job_token))
    job_dir = _job_dir_checked(job)
    items = load_decisions(job_dir)
    return {"job_id": job, "n_decisions": len(items), "decisions": items}


@app.get("/api/results/{job}/decisions.csv")
def get_roi_decisions_csv(job: str, t: Optional[str] = None, x_job_token: Optional[str] = Header(None)):
    """Выгрузка решений специалиста в CSV (разделитель «;»)."""
    from fastapi.responses import Response
    _check_job_access(job, _decision_token(t, x_job_token))
    job_dir = _job_dir_checked(job)
    text = decisions_to_csv(load_decisions(job_dir))
    return Response(content=text.encode("utf-8"), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{job}_decisions.csv"'})


@app.get("/api/results/{job}/decisions_sr/{study_uid}")
def get_roi_decisions_sr(job: str, study_uid: str, t: Optional[str] = None,
                         x_job_token: Optional[str] = Header(None)):
    """Отдельный DICOM SR «Решение специалиста по области интереса» по исследованию задачи:
    файл <study_uid>_SR_decisions.dcm (основной SR исследования не меняется)."""
    from dicom_sr import build_decision_sr, decision_sr_filename, is_valid_uid
    _check_job_access(job, _decision_token(t, x_job_token))
    job_dir = _job_dir_checked(job)
    study_uid = (study_uid or "").strip()
    if not is_valid_uid(study_uid):
        raise HTTPException(400, "некорректный study_uid")
    rows = _summary_rows(job_dir)
    if not any(str(r.get("study_uid") or "") == study_uid for r in rows):
        raise HTTPException(404, "исследование с таким study_uid в этой задаче не найдено")
    items = [d for d in load_decisions(job_dir) if str(d.get("study_uid") or "") == study_uid]
    try:
        cfg_hash = _config_hash(_ENGINE.cfg) if _ENGINE is not None else _config_hash(load_config(
            Path(os.environ["DENSITO_CONFIG"]) if os.environ.get("DENSITO_CONFIG") else None))
    except Exception:  # noqa: BLE001 — SR можно собрать и без конфигурации
        cfg_hash = ""
    out_dir = job_dir / DECISIONS_SR_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / decision_sr_filename(study_uid)
    ds = build_decision_sr(study_uid, items, PIPELINE_VERSION, cfg_hash)
    fd, tmp_name = tempfile.mkstemp(prefix=".sr_", suffix=".tmp", dir=str(out_dir))
    os.close(fd)
    try:
        ds.save_as(tmp_name, write_like_original=False)
        os.replace(tmp_name, out_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return FileResponse(str(out_path), media_type="application/dicom", filename=out_path.name)


# =========================================================================== #
# Идея «в»: сводка по партии для заведующего отделением и старшего лаборанта.
# Источник — файлы каталога задачи: results.csv (9 колонок), results_debug.csv (needs_review, хэши кадров,
# причины отказов), results_extras.csv (если есть) и device_tags.csv (аппарат и дата исследования, записан
# при загрузке, когда входные DICOM ещё доступны; StationName/DeviceSerialNumber — только хэш, оператор не
# читается). Расчёт — tools/department_summary.py, схема — schema/department_summary.schema.json.
# Маршруты объявлены ДО общего /api/results/{job}/{name}: иначе «summary» ушло бы в выдачу файла.
# =========================================================================== #
DEVICE_TAGS_FILE = "device_tags.csv"
DEPT_SUMMARY_STEM = "department_summary"
_SUMMARY_LOCK = threading.Lock()


def _write_device_tags(rows: List[Dict[str, Any]], tmp: Path, job_dir: Path) -> None:
    """device_tags.csv в каталоге задачи. Ошибки не влияют на ответ /api/analyze."""
    if department_summary is None or not rows:
        return
    try:
        tags = department_summary.read_device_tags_dicom(rows, Path(tmp), salt=os.environ.get("DENSITO_HASH_SALT", ""))
        if tags:
            department_summary.write_device_tags_csv(tags, Path(job_dir) / DEVICE_TAGS_FILE)
    except Exception as e:  # noqa: BLE001
        LOG.warning("device_tags.csv не записан: %s", e)


def build_department_summary(job_dir: Path, top_n: int = 10, min_n: int = 20) -> Dict[str, Any]:
    """Собрать сводку по каталогу задачи и сохранить department_summary.{json,md,csv} рядом с results.csv."""
    if department_summary is None:
        raise HTTPException(503, "модуль сводки (tools/department_summary.py) недоступен")
    job_dir = Path(job_dir)
    res = job_dir / "results.csv"
    if not res.is_file():
        raise HTTPException(404, "results.csv задачи не найден")
    try:
        cfg = _ENGINE.cfg if _ENGINE is not None else load_config(
            Path(os.environ["DENSITO_CONFIG"]) if os.environ.get("DENSITO_CONFIG") else None)
        cfg_hash = _config_hash(cfg)
    except Exception:  # noqa: BLE001 — сводку можно собрать и без конфигурации (строки по умолчанию)
        cfg, cfg_hash = {}, None
    dbg = job_dir / "results_debug.csv"
    ext = job_dir / "results_extras.csv"
    dev = job_dir / DEVICE_TAGS_FILE
    try:
        summary = department_summary.summarize_files(
            [res], debug=[dbg] if dbg.is_file() else [], extras=[ext] if ext.is_file() else [],
            device_tags=[dev] if dev.is_file() else None, cfg=cfg, config_hash=cfg_hash,
            pipeline_version=PIPELINE_VERSION, min_n=int(min_n), top_n=int(top_n),
            salt=os.environ.get("DENSITO_HASH_SALT", ""))
    except ValueError as e:
        raise HTTPException(400, f"results.csv задачи не в официальном формате: {e}")
    summary["job_id"] = job_dir.name
    with _SUMMARY_LOCK:
        try:
            (job_dir / f"{DEPT_SUMMARY_STEM}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
            (job_dir / f"{DEPT_SUMMARY_STEM}.md").write_text(department_summary.to_markdown(summary), encoding="utf-8")
            (job_dir / f"{DEPT_SUMMARY_STEM}.csv").write_text(department_summary.to_csv_text(summary), encoding="utf-8")
        except OSError as e:
            LOG.warning("department_summary файлы не записаны: %s", e)
    return summary


def _summary_params(top: Optional[int], min_n: Optional[int]) -> tuple:
    top_n = 10 if top is None else max(0, min(int(top), 200))
    mn = 20 if min_n is None else max(1, min(int(min_n), 10000))
    return top_n, mn


@app.get("/api/results/{job}/summary")
def get_department_summary(job: str, t: Optional[str] = None, x_job_token: Optional[str] = Header(None),
                           top: Optional[int] = None, min_n: Optional[int] = None):
    """Сводка по партии (JSON, schema/department_summary.schema.json): исследования и файлы, доли нарушений
    по области/типу/аппарату/дате с ДИ Уилсона 95 % и пометкой «мало данных» (n < min_n), доля Failure
    с причинами, зона «не уверен», исследования для пересмотра (только UID). Персональных данных нет."""
    _check_job_access(job, _decision_token(t, x_job_token))
    job_dir = _job_dir_checked(job)
    top_n, mn = _summary_params(top, min_n)
    return JSONResponse(content=build_department_summary(job_dir, top_n=top_n, min_n=mn))


@app.get("/api/results/{job}/summary.md")
def get_department_summary_md(job: str, t: Optional[str] = None, x_job_token: Optional[str] = Header(None),
                              top: Optional[int] = None, min_n: Optional[int] = None):
    """Та же сводка в Markdown (для печати или вставки в отчёт отделения)."""
    from fastapi.responses import Response
    _check_job_access(job, _decision_token(t, x_job_token))
    job_dir = _job_dir_checked(job)
    top_n, mn = _summary_params(top, min_n)
    text = department_summary.to_markdown(build_department_summary(job_dir, top_n=top_n, min_n=mn))
    return Response(content=text.encode("utf-8"), media_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": f'inline; filename="{job}_summary.md"'})


@app.get("/api/results/{job}/summary.csv")
def get_department_summary_csv(job: str, t: Optional[str] = None, x_job_token: Optional[str] = Header(None),
                               top: Optional[int] = None, min_n: Optional[int] = None):
    """Плоская таблица долей сводки (разделитель «;»): раздел, группа, метрика, k, n, доля, ДИ, «мало данных»."""
    from fastapi.responses import Response
    _check_job_access(job, _decision_token(t, x_job_token))
    job_dir = _job_dir_checked(job)
    top_n, mn = _summary_params(top, min_n)
    text = department_summary.to_csv_text(build_department_summary(job_dir, top_n=top_n, min_n=mn))
    return Response(content=("\ufeff" + text).encode("utf-8"), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{job}_summary.csv"'})


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


# --------------------------------------------------------------------------- #
# Журнал исследований отделения (поиск по ФИО/дате, фильтры, статусы, комментарии врача и лаборанта).
# Выключен, пока нет учётных записей (OUTPUT_DIR/registry_users.json); подробности — src/registry.py.
# --------------------------------------------------------------------------- #
from registry import Registry, mount as registry_mount, read_tags as registry_read_tags  # noqa: E402
from registry import mask_name as registry_mask_name, fmt_date as registry_fmt_date  # noqa: E402

REGISTRY = Registry(OUTPUT_DIR, users_file=Path(os.environ["DENSITO_USERS_FILE"]) if os.environ.get("DENSITO_USERS_FILE") else None)


def _registry_job_token(job: str) -> Optional[str]:
    """Код доступа к задаче для ссылки на карточку из журнала (только вошедшему пользователю журнала)."""
    if not JOB_RE.match(job or ""):
        return None
    tp = JOBS_DIR / job / JOB_TOKEN_FILE
    try:
        return tp.read_text(encoding="utf-8").strip() if tp.is_file() else None
    except OSError:
        return None


registry_mount(app, REGISTRY, _registry_job_token, LOG)

# Экспертная проверка сервиса в отделении (слепая оценка врачом выборки из журнала, отчёт о совпадении)
from expert_review import ExpertReview, mount as expert_mount, save_frames as expert_save_frames  # noqa: E402

EXPERT = ExpertReview(REGISTRY, JOBS_DIR)
REGISTRY.hidden_jobs = EXPERT.hidden_jobs  # журнал скрывает решения сервиса по незавершённым своим слепым проверкам
expert_mount(app, REGISTRY, EXPERT)


@app.post("/api/expert/upload")
async def expert_upload(files: List[UploadFile] = File(...), title: str = Form(""),
                        x_registry_session: Optional[str] = Header(None)):
    """Слепая проверка на своих снимках: те же правила загрузки и тот же конвейер, что у /api/analyze; снимки
    попадают в журнал. В ответе нет решений сервиса и кода доступа к задаче — они откроются в отчёте и журнале после
    того, как загрузивший оценит все снимки и нажмёт «Завершить»."""
    if not REGISTRY.enabled():
        raise HTTPException(503, "Журнал исследований не настроен — экспертная проверка работает поверх него.")
    try:
        user = REGISTRY.check_session(x_registry_session)
    except PermissionError as e:
        raise HTTPException(401, str(e))
    res = await analyze(files=files, xlsx=False)
    try:
        s = EXPERT.create_set_from_job(user, res["job_id"], title, n_files=len(files),
                                       versions={"model_version": res.get("model_version"), "config_hash": res.get("config_hash")})
    except ValueError as e:
        raise HTTPException(400, str(e))
    p = s["params"]
    return {"set_id": s["id"], "title": s["title"], "n": s["n"], "n_files": len(files), "n_rows": p.get("n_rows"),
            "skipped": p.get("skipped"), "set_hash": p.get("set_hash"), "model_version": p.get("model_version"),
            "config_hash": p.get("config_hash")}


@app.get("/expert/", include_in_schema=False)
def web_expert():
    """Страница экспертной проверки (на демо-стенде — отдельный поддомен)."""
    p = WEB_DIR / "expert" / "index.html"
    if not p.is_file():
        raise HTTPException(404, "expert UI not bundled in this image")
    return FileResponse(str(p), media_type="text/html; charset=utf-8")


@app.post("/api/review")
async def submit_review(request: Request):
    """Приём результатов слепой ревизии рентгенолога одним JSON. Страница ревизии лежит под
    тем же basic auth, отдельного кода доступа нет: эндпоинт только принимает и складывает файл,
    ничего не отдаёт и на пайплайн не влияет."""
    raw = await request.body()
    if len(raw) > MAX_REVIEW_BYTES:
        raise HTTPException(413, f"Объём ответа превышает {MAX_REVIEW_BYTES // (1024 * 1024)} МБ")
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "Ожидается JSON")
    if not isinstance(data, dict):
        raise HTTPException(400, "Ожидается объект JSON")
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}.json"
    (REVIEW_DIR / name).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    LOG.info("ревизия сохранена: %s (%d байт, ответов %d)", name, len(raw),
             len(data.get("answers") or []) if isinstance(data.get("answers"), list) else 0)
    return {"ok": True, "saved": name}


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
