#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DensitoAI — пакетный инференс контроля качества DXA-снимков (ЛЦТ 2026).

Главные инженерные требования (ТЗ п. 2.7, рецензия Fable 5):
  * инференс НИКОГДА не падает целиком из-за одного плохого файла —
    каждый файл обрабатывается в собственном try/except, при сбое пишется
    строка-заглушка с processing_status="Failure";
  * выходной CSV имеет ТОЧНЫЕ имена колонок и ТОЧНЫЕ строки регионов /
    типов нарушений (см. config.yaml, секции regions/violations);
  * воспроизводимость: детерминированные модели, фиксированные сиды,
    никаких обращений во внешнюю сеть (веса бэкбона лежат в models/).

Архитектура (двухконтурная):
  Контур A — геометрия (geometry_features.py): угол оси, металл, укладка, ROI.
  Контур B — замороженный EfficientNet-B0 -> эмбеддинг -> PCA -> логрегрессия.
  Стэкинг — ранговое усреднение A и B относительно OOF-распределений обучения.
  Если .pkl-моделей нет — физические fallback-правила контура A (config.yaml).

Использование:
  python src/inference.py --input <папка_или_zip> --output <results.csv>
  python src/inference.py --input data/test --output out/results.csv --xlsx --debug-csv

Формат выходной строки (одна строка = один DICOM-файл):
  path_to_study, study_uid, image_uid, anatomical_region, quality_class,
  violation_type, quality_prob, processing_status, time_of_processing
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import csv
import hashlib
import json
import logging
import os
import pickle
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Пути проекта. Всё относительно корня репозитория, переопределяется env.
# --------------------------------------------------------------------------- #
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("DENSITO_ROOT", SRC_DIR.parent)).resolve()
MODELS_DIR = Path(os.environ.get("DENSITO_MODELS_DIR", PROJECT_ROOT / "models")).resolve()
CONFIG_PATH = Path(os.environ.get("DENSITO_CONFIG", PROJECT_ROOT / "config.yaml")).resolve()
sys.path.insert(0, str(SRC_DIR))

# Веса EfficientNet-B0 лежат в репозитории -> torchvision берёт их локально,
# без выхода в интернет (требование ТЗ п. 3.2). Должно быть ДО import torch.
_TORCH_HOME = MODELS_DIR / "torch_home"
if _TORCH_HOME.exists():
    os.environ.setdefault("TORCH_HOME", str(_TORCH_HOME))

import pydicom  # noqa: E402

from geometry_features import (  # noqa: E402
    PIXEL_SPACING_X_MM, PIXEL_SPACING_Y_MM,
    segment_bone, spine_axis_features, foreign_object_features,
    spine_positioning_features, hip_positioning_features,
)
from hip_features import hip_all_features  # noqa: E402
from calibration_utils import risk_level  # noqa: E402  (К3: правило уровня риска)

__version__ = "2.1.0"
LOG = logging.getLogger("densito.inference")
# pydicom шумит предупреждениями о нестандартных UID в анонимизированных файлах — не ошибка
logging.getLogger("pydicom").setLevel(logging.ERROR)

# --------------------------------------------------------------------------- #
# Конфиг (с жёстко зашитыми значениями по умолчанию на случай отсутствия yaml)
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG: Dict[str, Any] = {
    "version": "2.1.0",
    "output": {
        "columns": ["path_to_study", "study_uid", "image_uid", "anatomical_region",
                    "quality_class", "violation_type", "quality_prob",
                    "processing_status", "time_of_processing"],
        "status_success": "Success", "status_failure": "Failure",
        "violation_separator": ";", "fallback_quality_prob": 0.5,
        "path_mode": "relative", "csv_encoding": "utf-8",
    },
    "regions": {"spine": "Поясничный отдел позвоночника",
                "hip": "Проксимальный отдел бедра",
                "default_when_unknown": "hip", "spine_min_cols": 290},
    "violations": {"sp_pos": "Некорректная укладка",
                   "sp_axis": "Не выравнена ось позвоночника",
                   "sp_art": "Присутствуют посторонние предметы",
                   "rh_pos": "Некорректная укладка", "rh_roi": "Некорректная область интереса",
                   "lh_pos": "Некорректная укладка", "lh_roi": "Некорректная область интереса"},
    "criteria_by_region": {"spine": ["sp_pos", "sp_axis", "sp_art"],
                           "right_hip": ["rh_pos", "rh_roi"],
                           "left_hip": ["lh_pos", "lh_roi"]},
    "geometry_cols": {"sp_pos": ["center_offset_ratio", "bone_width_ratio"],
                      "sp_axis": ["axis_angle_deg"],
                      "sp_art": ["metal_metal_area_mm2", "metal_metal_max_intensity_gap"],
                      "rh_pos": ["shaft_angle_deg"], "rh_roi": ["edge_distance_ratio", "bone_area_ratio"],
                      "lh_pos": ["shaft_angle_deg"], "lh_roi": ["edge_distance_ratio", "bone_area_ratio"]},
    "stacking": {"weight_geom": 0.5, "weight_emb": 0.5, "weights_by_criterion": {},
                 "any_violation_aggregation": "max",
                 "any_blend_weight_model": 0.5, "consistent_quality_prob": True},
    "thresholds": {}, "fallback_threshold": 0.5,
    "fallback_rules": {  # значения синхронизированы с config.yaml (калибровка на трейне)
        "sp_axis": {"feature": "axis_angle_deg", "center": 3.6, "scale": 1.0, "direction": 1},
        "sp_art": {"feature": "metal_metal_area_mm2", "center": 670.0, "scale": 200.0, "direction": 1},
        "sp_pos": {"feature": "center_offset_ratio_abs", "center": 0.074, "scale": 0.02, "direction": 1},
        "rh_pos": {"feature": "shaft_angle_deg", "center": 16.3, "scale": 3.0, "direction": 1},
        "lh_pos": {"feature": "shaft_angle_deg", "center": 14.7, "scale": 3.0, "direction": 1},
        "rh_roi": {"feature": "edge_distance_mm", "center": 20.0, "scale": 3.0, "direction": -1},
        "lh_roi": {"feature": "edge_distance_mm", "center": 20.0, "scale": 3.0, "direction": -1},
        "decision_threshold": 0.5,
    },
    "validation": {"min_rows": 64, "max_rows": 4096, "min_cols": 64, "max_cols": 4096,
                   "min_bone_fraction": 0.01, "max_bone_fraction": 0.90},
    "pixel_spacing_mm": {"y": PIXEL_SPACING_Y_MM, "x": PIXEL_SPACING_X_MM},
}

DICOM_EXTENSIONS = {".dcm", ".dicom", ".dic", ".ima"}
# Подсказки региона из имени файла (образец "Для теста.zip": CR000000_ПОП.dcm,
# CR000000_ППОБ.dcm, CR000001_ЛПОБ.dcm). На закрытом тесте суффиксов может не быть.
FILENAME_HINTS = (
    ("ППОБ", "right_hip"), ("ЛПОБ", "left_hip"), ("ПОП", "spine"),
    ("RIGHT", "right_hip"), ("LEFT", "left_hip"), ("SPINE", "spine"),
    ("_R_", "right_hip"), ("_L_", "left_hip"), ("LUMBAR", "spine"),
)


# Критерии ТЗ по сторонам бедра -> критерий единой модели бедра (ключ в stacking.weights_by_criterion,
# metrics_summary.json["hip"], models/model_hip_*.pkl). Веса стэкинга по стороне НЕ различаются.
HIP_CRIT_TO_MODEL = {"rh_pos": "hip_pos", "lh_pos": "hip_pos", "rh_roi": "hip_roi", "lh_roi": "hip_roi"}


def stacking_weights(cfg: Dict[str, Any], crit: str) -> Tuple[float, float]:
    """(w_geom, w_emb) для критерия. Приоритет: stacking.weights_by_criterion[<crit>] ->
    weights_by_criterion[<критерий единой модели бедра>] -> weight_geom / weight_emb (дефолт 0.5/0.5).
    Вес по критерию задаёт w_geom, w_emb = 1 - w_geom (вентиль К2, выбран nested CV,
    см. models/nested_gate_decisions.json и work/B/REPORT.md)."""
    st = cfg.get("stacking", {}) or {}
    by_crit = st.get("weights_by_criterion") or {}
    for key in (crit, HIP_CRIT_TO_MODEL.get(crit)):
        if key is not None and by_crit.get(key) is not None:
            wg = float(np.clip(float(by_crit[key]), 0.0, 1.0))
            return wg, 1.0 - wg
    return float(st.get("weight_geom", 0.5)), float(st.get("weight_emb", 0.5))


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    path = Path(path) if path else CONFIG_PATH
    if path.exists():
        try:
            import yaml  # PyYAML
            with open(path, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f) or {}
            cfg = _deep_update(cfg, user_cfg)
            LOG.info("Config loaded: %s", path)
        except Exception as e:  # noqa: BLE001
            LOG.warning("Config %s not loaded (%s). Using built-in defaults.", path, e)
    else:
        LOG.warning("Config %s not found. Using built-in defaults.", path)
    return cfg


# --------------------------------------------------------------------------- #
# Утилиты
# --------------------------------------------------------------------------- #
def sigmoid(x: float) -> float:
    x = float(np.clip(x, -50, 50))
    return float(1.0 / (1.0 + np.exp(-x)))


def clip01(p: float) -> float:
    if p is None or not np.isfinite(p):
        return 0.5
    return float(min(1.0, max(0.0, float(p))))


def path_hash(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8", errors="ignore")).hexdigest()[:32]


def config_hash(cfg: Dict[str, Any]) -> str:
    """Короткий sha256 конфигурации (та же формула, что в api_server._config_hash)."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:12]


def is_dicom_candidate(path: Path) -> bool:
    """DICOM-кандидат: по расширению или по магическому числу 'DICM' (offset 128)."""
    if path.suffix.lower() in DICOM_EXTENSIONS:
        return True
    if path.suffix == "" or path.suffix.lower() not in {".csv", ".xlsx", ".txt", ".json",
                                                          ".png", ".jpg", ".jpeg", ".pdf",
                                                          ".zip", ".md", ".py", ".yaml"}:
        try:
            with open(path, "rb") as f:
                f.seek(128)
                return f.read(4) == b"DICM"
        except Exception:  # noqa: BLE001
            return False
    return False


def _fix_zip_name(zi: "zipfile.ZipInfo") -> str:
    """Имена в zip без флага UTF-8 Python декодирует как cp437; архивы с Windows с русскими
    именами обычно в cp866 (иногда cp1251). Восстанавливаем читаемое имя."""
    if zi.flag_bits & 0x800:
        return zi.filename
    raw = zi.filename.encode("cp437", errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass

    def _score(txt: str) -> int:  # больше — правдоподобнее: кириллица/латиница хорошо, псевдографика плохо
        good = sum(1 for ch in txt if ch.isalnum() or ch in " _-./()")
        bad = sum(1 for ch in txt if 0x2500 <= ord(ch) <= 0x25FF or ord(ch) < 32)
        return good - 3 * bad

    best = zi.filename
    best_score = _score(best) - 1
    for enc in ("cp866", "cp1251"):
        try:
            cand = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        sc = _score(cand)
        if sc > best_score:
            best, best_score = cand, sc
    return best


def safe_extract_zip(zip_path: Path, dest: Path) -> List[Tuple[Path, str]]:
    """Безопасная распаковка: нормализуем кодировку имён, отбрасываем абсолютные пути и
    `..` (zip-slip), служебные каталоги __MACOSX. Возвращает [(файл на диске, путь внутри архива)]."""
    out: List[Tuple[Path, str]] = []
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for zi in zf.infolist():
            if zi.is_dir():
                continue
            name = _fix_zip_name(zi).replace("\\", "/")
            parts = [p for p in name.split("/") if p not in ("", ".", "..")]
            if not parts or parts[0] == "__MACOSX" or parts[-1] == ".DS_Store":
                continue
            rel = "/".join(parts)
            target = (dest / Path(*parts)).resolve()
            if dest not in target.parents:
                LOG.warning("Zip entry skipped (path escapes destination): %r", zi.filename)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(zi) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            out.append((target, rel))
    return out


def discover_files(input_path: Path, tmp_holder: List[Path],
                   display: Optional[Dict[Path, str]] = None) -> Tuple[Path, List[Path]]:
    """Возвращает (корень, отсортированный список DICOM-кандидатов).
    Поддерживает папку, одиночный файл и zip-архив (распаковка во временную папку).
    В `display` (если передан) складывает отображаемый путь для файлов из архивов:
    `<путь архива относительно корня>/<путь внутри архива>` (для входного zip — только
    путь внутри архива), чтобы path_to_study не содержал временных каталогов."""
    input_path = Path(input_path)
    display = display if display is not None else {}
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        tmp_dir = Path(tempfile.mkdtemp(prefix="densito_in_"))
        tmp_holder.append(tmp_dir)
        try:
            extracted = safe_extract_zip(input_path, tmp_dir)
        except zipfile.BadZipFile as e:
            raise ValueError(f"Архив «{input_path.name}» повреждён или не является zip-файлом "
                             f"({e}). Пересоздайте архив и загрузите снова.") from e
        for target, rel in extracted:
            display[target.resolve()] = rel
        LOG.info("Archive %s extracted to %s", input_path, tmp_dir)
        root = tmp_dir
    elif input_path.is_file():
        return input_path.parent, [input_path]
    else:
        root = input_path
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and is_dicom_candidate(p)]
    # вложенные zip внутри папки — тоже распаковываем (пакетная обработка "из архива")
    for z in sorted(root.rglob("*.zip")):
        try:
            sub = Path(tempfile.mkdtemp(prefix="densito_in_"))
            tmp_holder.append(sub)
            try:
                z_rel = str(z.resolve().relative_to(root.resolve()))
            except ValueError:
                z_rel = z.name
            extracted = safe_extract_zip(z, sub)
            for target, rel in extracted:
                display[target.resolve()] = f"{z_rel}/{rel}"
            files += [p for p in sorted(sub.rglob("*")) if p.is_file() and is_dicom_candidate(p)]
        except Exception as e:  # noqa: BLE001
            # битый архив не пропускаем молча: он попадёт в результаты строкой Failure
            LOG.warning("Nested archive %s is broken: %s", z, e)
            files.append(z)
    return root, files


# --------------------------------------------------------------------------- #
# Чтение и валидация DICOM
# --------------------------------------------------------------------------- #
@dataclass
class DicomInfo:
    ds: Any
    img_u8: np.ndarray
    rows: int
    cols: int
    study_uid: str
    image_uid: str
    pixel_spacing: Tuple[float, float]  # (y_mm, x_mm)
    warnings: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)


def _tag(ds, name: str, default: str = "") -> str:
    try:
        v = getattr(ds, name, None)
        if v is None:
            return default
        s = str(v).strip()
        return s if s else default
    except Exception:  # noqa: BLE001
        return default


def normalize_pixels(ds) -> np.ndarray:
    """Нормализация в uint8 [0,255]: MONOCHROME1 -> инверсия, Rescale, RGB -> gray,
    многокадровые -> первый кадр, перцентильное окно 1–99 %."""
    arr = ds.pixel_array
    if arr is None or arr.size == 0:
        raise ValueError("empty pixel_array")
    arr = np.asarray(arr)
    if arr.ndim == 4:      # frames x H x W x C
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4) and arr.shape[0] > 4:   # RGB(A)
            arr = arr[..., :3].astype(np.float32).mean(axis=-1)
        else:                                               # frames x H x W
            arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"unsupported pixel array shape {arr.shape}")
    arr = arr.astype(np.float32)
    if _tag(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr
    try:
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        arr = arr * slope + intercept
    except Exception:  # noqa: BLE001
        pass
    lo, hi = np.percentile(arr, [1, 99])
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255)
    else:
        arr = np.zeros_like(arr)
    return arr.astype(np.uint8)


def read_and_validate(path: Path, cfg: Dict[str, Any]) -> DicomInfo:
    """Валидатор входа. Любая проблема -> исключение (обрабатывается выше как Failure)."""
    v = cfg["validation"]
    if path.suffix.lower() == ".zip":
        raise ValueError("Архив повреждён или не является zip-файлом; пересоздайте архив и загрузите снова")
    ds = pydicom.dcmread(str(path), force=True)
    if not hasattr(ds, "PixelData") and "PixelData" not in ds:
        raise ValueError("DICOM has no PixelData")
    # Сырой поток без преамбулы/file meta (некоторые архивы и PACS-экспорты): pydicom читает теги,
    # но для декодирования пикселей нужен TransferSyntaxUID -> восстанавливаем из фактической кодировки.
    if "TransferSyntaxUID" not in getattr(ds, "file_meta", {}):
        enc = getattr(ds, "original_encoding", (None, None))
        implicit_vr = enc[0] if enc[0] is not None else True
        little_endian = enc[1] if enc[1] is not None else True
        if not hasattr(ds, "file_meta") or ds.file_meta is None:
            ds.file_meta = pydicom.dataset.FileMetaDataset()
        if implicit_vr:
            ts = pydicom.uid.ImplicitVRLittleEndian
        elif little_endian:
            ts = pydicom.uid.ExplicitVRLittleEndian
        else:
            ts = pydicom.uid.ExplicitVRBigEndian
        ds.file_meta.TransferSyntaxUID = ts
    img = normalize_pixels(ds)
    rows, cols = img.shape
    if not (v["min_rows"] <= rows <= v["max_rows"] and v["min_cols"] <= cols <= v["max_cols"]):
        raise ValueError(f"image size out of range: {rows}x{cols}")
    if int(img.max()) == int(img.min()):
        raise ValueError("constant (blank) image")

    warns: List[str] = []
    study_uid = _tag(ds, "StudyInstanceUID")
    image_uid = _tag(ds, "SOPInstanceUID")
    if not study_uid:
        # fallback: хэш родительской папки (обычно = папка исследования)
        study_uid = "hash-" + path_hash(path.parent)
        warns.append("no StudyInstanceUID -> hash of parent folder")
    if not image_uid:
        image_uid = "hash-" + path_hash(path)
        warns.append("no SOPInstanceUID -> hash of file path")

    # PixelSpacing: читаем тег, иначе константа аппарата (с предупреждением)
    ps_y, ps_x = float(cfg["pixel_spacing_mm"]["y"]), float(cfg["pixel_spacing_mm"]["x"])
    for tag_name in ("PixelSpacing", "ImagerPixelSpacing"):
        val = getattr(ds, tag_name, None)
        if val is not None:
            try:
                ps_y, ps_x = float(val[0]), float(val[1])
                break
            except Exception:  # noqa: BLE001
                continue
    else:
        warns.append("no PixelSpacing tag -> device default 1.05x0.6 mm")

    tags = {k: _tag(ds, k) for k in ("Modality", "Manufacturer", "BodyPartExamined",
                                     "SeriesDescription", "ProtocolName", "Laterality",
                                     "ImageLaterality", "StudyDescription", "PhotometricInterpretation")}
    if tags["Modality"] and tags["Modality"] not in ("OT", "CR", "DX", "RG", "SC", ""):
        warns.append(f"unexpected Modality={tags['Modality']}")
    return DicomInfo(ds=ds, img_u8=img, rows=rows, cols=cols, study_uid=study_uid,
                     image_uid=image_uid, pixel_spacing=(ps_y, ps_x), warnings=warns, tags=tags)


# --------------------------------------------------------------------------- #
# Определение региона
# --------------------------------------------------------------------------- #
def hip_side_by_density(img_u8: np.ndarray) -> str:
    """УСТАРЕВШИЙ резерв: сторона по распределению плотной кости (ошибается на ~24% снимков).
    Используется только если анатомический детектор недоступен."""
    h, w = img_u8.shape
    thr = np.percentile(img_u8, 60)
    left = (img_u8[:, : w // 2] > thr).mean()
    right = (img_u8[:, w // 2:] > thr).mean()
    return "right_hip" if right > left else "left_hip"


def detect_hip_side_region(img_u8: np.ndarray) -> Tuple[str, str]:
    """Сторона бедра -> internal region. Основной метод — анатомическое правило
    hip_features.detect_hip_side (таз всегда медиальнее диафиза); при любой ошибке —
    старая плотностная эвристика. Возвращает (region, source)."""
    try:
        from hip_features import detect_hip_side
        side = detect_hip_side(img_u8)
        return ("right_hip" if side == "right" else "left_hip"), "anatomy"
    except Exception as e:  # noqa: BLE001
        LOG.warning("detect_hip_side failed (%s) -> density heuristic", e)
        return hip_side_by_density(img_u8), "density"


def classify_region(info: DicomInfo, path: Path, cfg: Dict[str, Any]) -> Tuple[str, str]:
    """Возвращает (internal_region in {spine,right_hip,left_hip}, source).
    Приоритет: DICOM-теги -> ширина кадра (правило организаторов, стандартные 300/280/248 px)
    -> подсказка в имени файла (только для нестандартной ширины; организаторы подтвердили,
    что суффиксов в именах тестовых файлов не будет) -> ширина + анатомия стороны.
    Для нестандартной ширины без подсказки решение уточняет контентная модель (process_file)."""
    name_up = path.stem.upper()
    hint_region = next((region for hint, region in FILENAME_HINTS if hint.upper() in name_up), None)
    hint_src = next((f"filename:{hint}" for hint, region in FILENAME_HINTS if hint.upper() in name_up), "")
    std_cols = set(int(c) for c in cfg["regions"].get("standard_cols", [300, 280, 248]))

    text = " ".join([info.tags.get("BodyPartExamined", ""), info.tags.get("SeriesDescription", ""),
                     info.tags.get("ProtocolName", ""), info.tags.get("StudyDescription", "")]).upper()
    lat = (info.tags.get("Laterality") or info.tags.get("ImageLaterality") or "").upper()
    if any(k in text for k in ("SPINE", "LUMBAR", "LSPINE", "L-SPINE", "ПОЗВОНОЧ", "ПОП")):
        return "spine", "dicom_tags"
    if any(k in text for k in ("HIP", "FEMUR", "БЕДР", "ПОБ")):
        if lat.startswith("R"):
            return "right_hip", "dicom_tags+laterality"
        if lat.startswith("L"):
            return "left_hip", "dicom_tags+laterality"
        region, src = detect_hip_side_region(info.img_u8)
        return region, f"dicom_tags+{src}"

    # Имя файла — только если ширина нестандартная (иначе решает содержимое)
    if hint_region is not None and info.cols not in std_cols:
        return hint_region, hint_src

    # Правило по ширине (подтверждено организаторами): 300 px спина, 280/248 бедро
    if info.cols >= int(cfg["regions"]["spine_min_cols"]):
        return "spine", "dims"
    if lat.startswith("R"):
        return "right_hip", "dims+laterality"
    if lat.startswith("L"):
        return "left_hip", "dims+laterality"
    region, src = detect_hip_side_region(info.img_u8)
    return region, f"dims+{src}"


def guess_region_without_pixels(path: Path, cfg: Dict[str, Any]) -> str:
    """Для строки-заглушки при сбое: имя файла -> заголовок DICOM (Rows/Columns) -> дефолт."""
    name_up = path.stem.upper()
    for hint, region in FILENAME_HINTS:
        if hint.upper() in name_up:
            return region
    try:
        ds = pydicom.dcmread(str(path), force=True, stop_before_pixels=True)
        cols = int(getattr(ds, "Columns", 0) or 0)
        if cols >= int(cfg["regions"]["spine_min_cols"]):
            return "spine"
        if cols > 0:
            return "left_hip"
    except Exception:  # noqa: BLE001
        pass
    return "spine" if cfg["regions"]["default_when_unknown"] == "spine" else "left_hip"


def official_region_name(internal_region: str, cfg: Dict[str, Any]) -> str:
    return cfg["regions"]["spine"] if internal_region == "spine" else cfg["regions"]["hip"]


# --------------------------------------------------------------------------- #
# Признаки контура A
# --------------------------------------------------------------------------- #
def extract_geometry(info: DicomInfo, region: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    img = info.img_u8
    mask = segment_bone(img)
    feats: Dict[str, Any] = {"region": region}
    bone_fraction = float((mask > 0).mean())
    feats["bone_fraction"] = bone_fraction
    v = cfg["validation"]
    if not (v["min_bone_fraction"] <= bone_fraction <= v["max_bone_fraction"]):
        info.warnings.append(f"non-standard image: bone fraction {bone_fraction:.3f}")

    foreign = foreign_object_features(img, mask)
    feats.update({f"metal_{k}": v_ for k, v_ in foreign.items()})

    ps_y, ps_x = info.pixel_spacing
    if region == "spine":
        axis = spine_axis_features(img, mask)
        feats["axis_angle_deg"] = axis["axis_angle_deg"]
        feats["curvature"] = axis["curvature"]
        feats["valid_rows"] = axis["valid_rows"]
        feats.update(spine_positioning_features(img, mask))
        co = feats.get("center_offset_ratio")
        feats["center_offset_ratio_abs"] = abs(co) if co is not None else None
    else:
        feats.update(hip_positioning_features(img, mask))
        edr = feats.get("edge_distance_ratio")
        feats["edge_distance_mm"] = (edr * info.cols * ps_x) if edr is not None else None
        # Новые физически обоснованные признаки бедра (каноническая ориентация,
        # femur_solidity / shaft_width_mm / scan_length_mm / ...) — используются
        # моделями model_hip_pos_*.pkl / model_hip_roi_*.pkl. Считает свою маску
        # сегментации бедра внутри (segment_bone_hip), не переиспользует `mask`
        # от segment_bone (позвоночный сегментатор).
        try:
            feats.update(hip_all_features(img))
        except Exception as e:  # noqa: BLE001
            info.warnings.append(f"hip_all_features failed: {e}")
    return feats


# --------------------------------------------------------------------------- #
# Модели (pickle) и стэкинг
# --------------------------------------------------------------------------- #
class ModelBundle:
    """Обёртка над одним .pkl. Терпима к двум форматам:
       (а) объект с predict_proba (sklearn Pipeline);
       (б) dict {'scaler','pca'?,'clf','feature_cols'?,'medians'?,'oof_scores'?,'mirror_right'?}."""

    def __init__(self, obj: Any, name: str):
        self.name = name
        self.obj = obj
        self.meta: Dict[str, Any] = obj if isinstance(obj, dict) else {}

    def predict(self, X: np.ndarray) -> float:
        X = np.asarray(X, dtype=np.float64).reshape(1, -1)
        if isinstance(self.obj, dict):
            sc = self.obj.get("scaler")
            if sc is not None:
                X = sc.transform(X)
            pca = self.obj.get("pca")
            if pca is not None:
                X = pca.transform(X)
            clf = self.obj.get("clf") or self.obj.get("model")
            if clf is None:
                raise ValueError(f"{self.name}: dict without 'clf'")
            return float(clf.predict_proba(X)[0, 1])
        if hasattr(self.obj, "predict_proba"):
            return float(self.obj.predict_proba(X)[0, 1])
        if hasattr(self.obj, "decision_function"):
            return sigmoid(float(self.obj.decision_function(X)[0]))
        raise ValueError(f"{self.name}: unsupported model object {type(self.obj)}")


class ModelRegistry:
    """Загружает все доступные модели один раз. Отсутствие файла — warning, не ошибка."""

    def __init__(self, models_dir: Path, cfg: Dict[str, Any]):
        self.models_dir = Path(models_dir)
        self.cfg = cfg
        self.geom: Dict[Tuple[str, str], ModelBundle] = {}
        self.emb: Dict[Tuple[str, str], ModelBundle] = {}
        self.any_geom: Dict[str, ModelBundle] = {}
        self.any_emb: Dict[str, ModelBundle] = {}
        self.ref_geom: Dict[Tuple[str, str], np.ndarray] = {}
        self.ref_emb: Dict[Tuple[str, str], np.ndarray] = {}
        self.thresholds: Dict[str, float] = {}
        self.geometry_medians: Dict[str, float] = {}
        self.n_loaded = 0
        self._load()

    # ---- helpers
    def _load_pickle(self, path: Path) -> Optional[ModelBundle]:
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            self.n_loaded += 1
            LOG.info("Model loaded: %s", path.name)
            return ModelBundle(obj, path.name)
        except Exception as e:  # noqa: BLE001
            LOG.warning("Model %s could not be loaded (%s) -> ignored", path.name, e)
            return None

    def _candidates(self, region: str, crit: str, kind: str) -> List[Path]:
        names = [f"model_{region}_{crit}_{kind}.pkl"]
        if region in ("right_hip", "left_hip"):
            base = crit.split("_", 1)[-1]  # pos / roi
            names += [f"model_hip_hip_{base}_{kind}.pkl", f"model_hip_{base}_{kind}.pkl"]
        return [self.models_dir / n for n in names]

    def _load(self):
        if not self.models_dir.exists():
            LOG.warning("Models dir %s does not exist -> geometric fallback rules only", self.models_dir)
        for region, crits in self.cfg["criteria_by_region"].items():
            for crit in crits:
                for kind, store in (("geom", self.geom), ("emb_pca", self.emb)):
                    for cand in self._candidates(region, crit, kind):
                        mb = self._load_pickle(cand)
                        if mb is not None:
                            store[(region, crit)] = mb
                            break
                    else:
                        LOG.warning("No %s model for %s/%s (looked for %s) -> fallback",
                                    kind, region, crit, self._candidates(region, crit, kind)[0].name)
                # референсные OOF-распределения для рангового усреднения
                self._load_reference(region, crit)
            for kind, store in (("geom", self.any_geom), ("emb_pca", self.any_emb)):
                mb = self._load_pickle(self.models_dir / f"model_{region}_any_{kind}.pkl")
                if mb is not None:
                    store[region] = mb
        self._load_thresholds()
        self._load_medians()
        self._load_calibration()

    def _load_calibration(self):
        """К3: models/calibration.pkl (MODEL_CONTRACT.md, раздел «calibration.pkl»): Platt по критерию и запас
        зоны «не уверен». Ключи хранятся как sp_*/hip_*; rh_*/lh_* берут hip_*. Отсутствие файла — не ошибка:
        p_cal = None, запаса нет (тогда «не уверен» только при fallback-правиле/отказе).
        config.yaml: uncertainty.margin_by_criterion[<crit>] (не null) перекрывает запас из pkl."""
        self.platt: Dict[str, Tuple[float, float]] = {}
        self.margins: Dict[str, Optional[float]] = {}
        calib: Dict[str, Any] = {}
        p = self.models_dir / "calibration.pkl"
        if p.exists():
            try:
                with open(p, "rb") as f:
                    calib = pickle.load(f)
                if not isinstance(calib, dict) or calib.get("kind") != "densito_calibration":
                    LOG.warning("calibration.pkl: неожиданный формат -> игнорирую")
                    calib = {}
                else:
                    LOG.info("Calibration loaded: calibration.pkl (%d criteria)", len(calib.get("criteria", {}) or {}))
            except Exception as e:  # noqa: BLE001
                LOG.warning("calibration.pkl unreadable: %s", e)
                calib = {}
        ucfg = self.cfg.get("uncertainty", {}) or {}
        if ucfg.get("enabled", True) is False:
            cfg_margins: Dict[str, Any] = {}
            calib_margins: Dict[str, Any] = {}
        else:
            cfg_margins = ucfg.get("margin_by_criterion", {}) or {}
            calib_margins = calib.get("margin_by_criterion", {}) or {}
        for region, crits in self.cfg["criteria_by_region"].items():
            for crit in crits:
                base = ("hip_" + crit.split("_", 1)[-1]) if region in ("right_hip", "left_hip") else crit
                rec = (calib.get("criteria", {}) or {}).get(base) or {}
                pl = rec.get("platt")
                if isinstance(pl, dict) and pl.get("a") is not None:
                    self.platt[crit] = (float(pl["a"]), float(pl["b"]))
                m = cfg_margins.get(crit, cfg_margins.get(base))
                if m is None:
                    m = calib_margins.get(base)
                self.margins[crit] = None if m is None else float(m)

    def _load_reference(self, region: str, crit: str):
        key = (region, crit)
        # приоритет: oof_scores внутри pickle
        for store, ref in ((self.geom, self.ref_geom), (self.emb, self.ref_emb)):
            mb = store.get(key)
            if mb is not None and isinstance(mb.meta.get("oof_scores"), (list, np.ndarray)):
                ref[key] = np.sort(np.asarray(mb.meta["oof_scores"], dtype=np.float64))
        cands = [self.models_dir / f"oof_stacked_{region}_{crit}.csv"]
        if region in ("right_hip", "left_hip"):
            base = crit.split("_", 1)[-1]
            cands += [self.models_dir / f"oof_stacked_hip_hip_{base}.csv", self.models_dir / f"oof_stacked_hip_{base}.csv"]
        path = next((c for c in cands if c.exists()), None)
        if path is None:
            return
        try:
            import pandas as pd
            df = pd.read_csv(path)
            if key not in self.ref_geom and "oof_geom" in df:
                self.ref_geom[key] = np.sort(df["oof_geom"].dropna().values.astype(np.float64))
            if key not in self.ref_emb and "oof_emb" in df:
                self.ref_emb[key] = np.sort(df["oof_emb"].dropna().values.astype(np.float64))
        except Exception as e:  # noqa: BLE001
            LOG.warning("OOF reference %s not loaded: %s", path.name, e)

    def _load_thresholds(self):
        summary: Dict[str, Any] = {}
        p = self.models_dir / "metrics_summary.json"
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            except Exception as e:  # noqa: BLE001
                LOG.warning("metrics_summary.json unreadable: %s", e)
        for region, crits in self.cfg["criteria_by_region"].items():
            for crit in crits:
                t = self.cfg.get("thresholds", {}).get(crit)
                if t is None:  # порог, сохранённый внутри pickle (dict-формат)
                    for store in (self.geom, self.emb):
                        mb = store.get((region, crit))
                        if mb is not None and mb.meta.get("threshold") is not None:
                            t = mb.meta["threshold"]
                            break
                if t is None:
                    t = (summary.get(region, {}).get(crit, {}) or {}).get("threshold")
                if t is None and region in ("right_hip", "left_hip"):   # единая модель бедра
                    base = crit.split("_", 1)[-1]
                    hip = summary.get("hip", {}) or {}
                    for k in (f"hip_{base}", base, crit):
                        if isinstance(hip.get(k), dict) and hip[k].get("threshold") is not None:
                            t = hip[k]["threshold"]
                            break
                if t is None:
                    t = self.cfg.get("fallback_threshold", 0.5)
                    LOG.warning("No threshold for %s -> %.2f", crit, t)
                self.thresholds[crit] = float(t)

    def _load_medians(self):
        """Медианы геометрических признаков для импутации NaN (как при обучении)."""
        p = PROJECT_ROOT / "data" / "geometry_features.csv"
        cols = sorted({c for cs in self.cfg["geometry_cols"].values() for c in cs})
        if p.exists():
            try:
                import pandas as pd
                df = pd.read_csv(p, usecols=lambda c: c in cols or c == "region")
                for c in cols:
                    if c in df:
                        self.geometry_medians[c] = float(np.nanmedian(df[c].values.astype(np.float64)))
            except Exception as e:  # noqa: BLE001
                LOG.warning("geometry medians not loaded: %s", e)
        for c in cols:
            self.geometry_medians.setdefault(c, 0.0)

    # ---- inference helpers
    @staticmethod
    def percentile_rank(score: float, ref: Optional[np.ndarray]) -> Optional[float]:
        if ref is None or len(ref) == 0:
            return None
        # эквивалент pandas rank(pct=True) в пределе: доля референса <= score
        return float(np.searchsorted(ref, score, side="right") / len(ref))

    def geometry_vector(self, crit: str, feats: Dict[str, Any], mb: Optional[ModelBundle]) -> np.ndarray:
        cols = (mb.meta.get("feature_cols") if mb is not None else None) or self.cfg["geometry_cols"][crit]
        medians = (mb.meta.get("medians") if mb is not None else None) or {}
        if isinstance(medians, (list, np.ndarray)):
            medians = dict(zip(cols, medians))
        vec = []
        for c in cols:
            val = feats.get(c)
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                val = medians.get(c, self.geometry_medians.get(c, 0.0))
            vec.append(float(val))
        return np.asarray(vec, dtype=np.float64)

    def has_any_model(self) -> bool:
        return bool(self.geom or self.emb or self.any_geom or self.any_emb)


# --------------------------------------------------------------------------- #
# Контур B — эмбеддинги (ленивая инициализация, мягкий отказ)
# --------------------------------------------------------------------------- #
class EmbeddingExtractor:
    """Экстрактор эмбеддингов контура B. Поддерживает несколько источников весов одного
    EfficientNet-B0: ``imagenet`` (torchvision) и ``densito`` (models/backbone_densito.pth —
    наше GPU-предобучение на снимках кости; используется только моделями укладки, см.
    meta['emb_source'] в pkl и src/embeddings.py). Бэкбоны загружаются лениво, один раз."""

    DEFAULT_SOURCE = "imagenet"

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._backbones: Dict[str, Any] = {}
        self._failed: set = set()

    def _init(self, source: str = DEFAULT_SOURCE):
        if source in self._backbones or source in self._failed or not self.enabled:
            return
        try:
            import torch
            torch.manual_seed(0)
            torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
            from embeddings import FrozenBackbone
            self._backbones[source] = FrozenBackbone(source)
            LOG.info("EfficientNet-B0 backbone '%s' ready (TORCH_HOME=%s)", source, os.environ.get("TORCH_HOME", "-"))
        except Exception as e:  # noqa: BLE001
            self._failed.add(source)
            if source == self.DEFAULT_SOURCE:
                LOG.warning("Embedding backbone unavailable (%s) -> contour B disabled", e)
            else:
                LOG.warning("Embedding backbone '%s' unavailable (%s) -> models with this source get no contour B", source, e)

    @property
    def _backbone(self):  # совместимость со старым кодом/тестами
        return self._backbones.get(self.DEFAULT_SOURCE)

    def extract(self, img_u8: np.ndarray, mirror: bool = False, source: str = DEFAULT_SOURCE) -> Optional[np.ndarray]:
        self._init(source)
        bb = self._backbones.get(source)
        if bb is None:
            return None
        try:
            img = img_u8[:, ::-1].copy() if mirror else img_u8
            return np.asarray(bb.extract(img), dtype=np.float32)
        except Exception as e:  # noqa: BLE001
            LOG.warning("Embedding extraction failed (%s): %s", source, e)
            return None

    def extract_many(self, img_u8: np.ndarray, sources, mirror: bool = False) -> Dict[str, np.ndarray]:
        """{source: embedding} для всех запрошенных источников (недоступные пропускаются)."""
        out: Dict[str, np.ndarray] = {}
        for src in sorted(set(sources)):
            e = self.extract(img_u8, mirror=mirror, source=src)
            if e is not None:
                out[src] = e
        return out


def _emb_for(mb, embs: Optional[Dict[str, np.ndarray]]) -> Optional[np.ndarray]:
    """Эмбеддинг нужного источника для модели (meta['emb_source'], по умолчанию imagenet)."""
    if not embs or mb is None:
        return None
    return embs.get(str(mb.meta.get("emb_source", EmbeddingExtractor.DEFAULT_SOURCE)))


# --------------------------------------------------------------------------- #
# Основной класс инференса
# --------------------------------------------------------------------------- #
class DensitoInference:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None, models_dir: Optional[Path] = None,
                 use_embeddings: bool = True, visualize_dir: Optional[Path] = None,
                 sr_dir: Optional[Path] = None, roi_autocorrect_dir: Optional[Path] = None,
                 sr_study: bool = False, sr_study_dir: Optional[Path] = None, extras: bool = False):
        self.cfg = cfg or load_config()
        # [EXTRAS] экспериментальные флаги (work/D): белые линии, OOD-gate, эндопротез, когерентность исследования.
        # Не влияют на 9 колонок; пишутся в <output>_extras.csv и в self.last_extras_rows (API: details.extras).
        self.extras = bool(extras)
        self._extras_input: Optional[Dict[str, Any]] = None
        self.last_extras_rows: List[Dict[str, Any]] = []
        self.registry = ModelRegistry(Path(models_dir) if models_dir else MODELS_DIR, self.cfg)
        self.embedder = EmbeddingExtractor(enabled=use_embeddings)
        # Резервный классификатор области (позвоночник/бедро) по эмбеддингам.
        # Используется ТОЛЬКО когда ширина снимка нестандартна (не 300/280/248 px),
        # т.е. правило организаторов по ширине неприменимо. OOF-точность 100 % (499 файлов).
        self.region_fallback = None
        rf = (Path(models_dir) if models_dir else MODELS_DIR) / "model_region_emb.pkl"
        if use_embeddings and rf.exists():
            try:
                with open(rf, "rb") as fh:
                    self.region_fallback = pickle.load(fh)
            except Exception as e:  # noqa: BLE001
                LOG.warning("Region fallback model not loaded: %s", e)
        if not self.registry.has_any_model():
            LOG.warning("No trained .pkl models found in %s -> using physical fallback rules "
                        "(config.yaml: fallback_rules). Retrain/save models to enable stacking.",
                        self.registry.models_dir)
        # --- Бонус-функции (ТЗ п.2.6): визуализация, DICOM SR, ROI-автокоррекция.
        # Строго опциональны, по умолчанию выключены (None), не влияют на обязательный
        # CSV-выход даже при внутренней ошибке (каждая обёрнута в try/except в process_file).
        self.visualize_dir = Path(visualize_dir) if visualize_dir else None
        self.sr_dir = Path(sr_dir) if sr_dir else None
        self.roi_autocorrect_dir = Path(roi_autocorrect_dir) if roi_autocorrect_dir else None
        # --- Режим «один SR на исследование» (К10; методология НПКЦ ДиТ: нет SR / два SR — дефект).
        # Файлы пишутся после обработки всей партии в <каталог CSV>/sr/<study_uid>_SR.dcm
        # (или в sr_study_dir). Хэши оригинала (sha256 файла и пикселей) считаются всегда — они
        # попадают в debug-CSV и в SR; на официальный CSV режим не влияет.
        self.sr_study = bool(sr_study or sr_study_dir)
        self.sr_study_dir = Path(sr_study_dir) if sr_study_dir else None
        self._study_headers: Dict[str, Dict[str, Any]] = {}
        self.last_study_sr: Dict[str, str] = {}
        if self.visualize_dir or self.sr_dir or self.roi_autocorrect_dir:
            for d in (self.visualize_dir, self.sr_dir, self.roi_autocorrect_dir):
                if d:
                    d.mkdir(parents=True, exist_ok=True)
            LOG.info("Bonus outputs enabled: visualize=%s sr=%s roi_autocorrect=%s",
                      self.visualize_dir, self.sr_dir, self.roi_autocorrect_dir)

    # ---- скоринг одного критерия
    def score_criterion(self, region: str, crit: str, feats: Dict[str, Any],
                        embs: Optional[Dict[str, np.ndarray]]) -> Dict[str, Any]:
        reg = self.registry
        key = (region, crit)
        mb_g, mb_e = reg.geom.get(key), reg.emb.get(key)
        out: Dict[str, Any] = {"p_geom": None, "p_emb": None, "rank_geom": None,
                               "rank_emb": None, "score": None, "method": None}

        if mb_g is not None:
            try:
                out["p_geom"] = clip01(mb_g.predict(reg.geometry_vector(crit, feats, mb_g)))
                out["rank_geom"] = reg.percentile_rank(out["p_geom"], reg.ref_geom.get(key))
            except Exception as e:  # noqa: BLE001
                LOG.warning("%s/%s geom model failed: %s", region, crit, e)
        emb = _emb_for(mb_e, embs)
        if mb_e is not None and emb is not None:
            try:
                out["p_emb"] = clip01(mb_e.predict(emb))
                out["rank_emb"] = reg.percentile_rank(out["p_emb"], reg.ref_emb.get(key))
            except Exception as e:  # noqa: BLE001
                LOG.warning("%s/%s emb model failed: %s", region, crit, e)

        wg, we = stacking_weights(self.cfg, crit)
        out["w_geom"] = wg
        rg = out["rank_geom"] if out["rank_geom"] is not None else out["p_geom"]
        re_ = out["rank_emb"] if out["rank_emb"] is not None else out["p_emb"]
        if rg is not None and re_ is not None:
            # вес по критерию (вентиль): при wg=1 скор = rank_geom, при wg=0 — rank_emb
            out["score"], out["method"] = wg * rg + we * re_, "stacked_rank_avg"
        elif rg is not None:
            out["score"], out["method"] = rg, "geom_only"
        elif re_ is not None:
            out["score"], out["method"] = re_, "emb_only"
        else:
            out["score"], out["method"] = self.fallback_rule(crit, feats), "fallback_rule"

        thr = (reg.thresholds.get(crit, 0.5) if out["method"] != "fallback_rule"
               else float(self.cfg["fallback_rules"].get("decision_threshold", 0.5)))
        out["threshold"] = thr
        out["score"] = clip01(out["score"])
        out["flag"] = int(out["score"] >= thr)
        # К3 (только debug/API, на 9 колонок не влияет): запас до порога, зона «не уверен», Platt-вероятность.
        # «не уверен» <=> |score - threshold| <= margin (включительно); при fallback-правиле — всегда «не уверен»,
        # при отсутствии запаса (нет calibration.pkl) — только при fallback.
        margin = getattr(reg, "margins", {}).get(crit)
        out["margin"] = round(abs(out["score"] - thr), 6)
        if out["method"] == "fallback_rule":
            out["uncertain"] = 1
        else:
            out["uncertain"] = int(margin is not None and out["margin"] <= float(margin) + 1e-12)
        pl = getattr(reg, "platt", {}).get(crit)
        out["p_cal"] = (round(float(sigmoid(pl[0] * out["score"] + pl[1])), 6)
                        if (pl is not None and out["method"] != "fallback_rule") else None)
        return out

    def fallback_rule(self, crit: str, feats: Dict[str, Any]) -> float:
        rule = self.cfg["fallback_rules"].get(crit)
        if not rule:
            return 0.0
        x = feats.get(rule["feature"])
        if x is None or not np.isfinite(x):
            return 0.0  # признак не измерен -> консервативно "нет нарушения"
        return sigmoid(float(rule.get("direction", 1)) * (float(x) - float(rule["center"])) / float(rule["scale"]))

    def any_violation_prob(self, region: str, crit_results: Dict[str, Dict[str, Any]],
                           feats: Dict[str, Any], embs: Optional[Dict[str, np.ndarray]]) -> float:
        reg = self.registry
        probs: List[float] = []
        emb = _emb_for(reg.any_emb.get(region), embs)
        # 1) отдельная модель "есть нарушение", если сохранена
        mb_g, mb_e = reg.any_geom.get(region), reg.any_emb.get(region)
        if mb_g is not None:
            try:
                cols = mb_g.meta.get("feature_cols") or sorted({c for cr in self.cfg["criteria_by_region"][region]
                                                                 for c in self.cfg["geometry_cols"][cr]})
                vec = np.asarray([float(feats.get(c) if feats.get(c) is not None and np.isfinite(feats.get(c))
                                        else reg.geometry_medians.get(c, 0.0)) for c in cols])
                probs.append(clip01(mb_g.predict(vec)))
            except Exception as e:  # noqa: BLE001
                LOG.warning("any-geom model failed: %s", e)
        if mb_e is not None and emb is not None:
            try:
                probs.append(clip01(mb_e.predict(emb)))
            except Exception as e:  # noqa: BLE001
                LOG.warning("any-emb model failed: %s", e)
        # 2) агрегация по критериям
        scores = [r["score"] for r in crit_results.values() if r.get("score") is not None]
        if self.cfg["stacking"].get("any_violation_aggregation", "max") == "noisy_or":
            crit_agg = clip01(1.0 - float(np.prod([1.0 - s for s in scores]))) if scores else None
        else:
            crit_agg = clip01(max(scores)) if scores else None
        any_model = float(np.mean(probs)) if probs else None
        if any_model is None and crit_agg is None:
            return float(self.cfg["output"]["fallback_quality_prob"])
        if any_model is None:
            return crit_agg
        if crit_agg is None:
            return any_model
        # 3) смесь двух оценок. Валидация OOF (StratifiedGroupKFold по исследованиям, 3 сида):
        #    ROC-AUC any-модель 0.766/0.702 (spine/hip), max по критериям 0.681/0.761,
        #    смесь 0.5/0.5 -> 0.781/0.747, после согласования с классом 0.783/0.773 при F1-опт. порогах, 0.764/0.732 при nested-правилах К3 (19.09);
        #    см. docs/METRICS_REPORT.md (src/eval_oof_metrics.py).
        w = float(self.cfg["stacking"].get("any_blend_weight_model", 0.5))
        return clip01(w * any_model + (1.0 - w) * crit_agg)

    @staticmethod
    def consistent_quality_prob(prob: float, quality_class: int) -> float:
        """Согласование quality_prob с quality_class: класс определяется флагами
        критериев (порог подобран по OOF на каждый критерий), а вероятность —
        смесью моделей. Чтобы строка была непротиворечивой (class=1 <=> prob>=0.5)
        и ROC-AUC учитывал решение по критериям, вероятность монотонно сжимается
        в [0.5, 1] при нарушении и в [0, 0.5) при норме (порядок внутри класса
        сохраняется). OOF ROC-AUC при этом растёт: spine 0.735 -> 0.783 (при F1-опт. порогах),
        hip 0.704 -> 0.773 (см. docs/METRICS_REPORT.md)."""
        p = clip01(prob)
        return 0.5 + 0.5 * p if quality_class else min(0.5 * p, 0.499999)

    # ---- обработка одного файла (никогда не бросает исключение наружу)
    def process_file(self, path: Path, root: Optional[Path] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        t0 = time.perf_counter()
        cfg_out = self.cfg["output"]
        path = Path(path)
        debug: Dict[str, Any] = {"file": str(path)}
        try:
            info = read_and_validate(path, self.cfg)
            region, region_src = classify_region(info, path, self.cfg)
            std_cols = set(int(c) for c in self.cfg["regions"].get("standard_cols", [300, 280, 248]))
            if (region_src.startswith("dims") and info.cols not in std_cols
                    and self.region_fallback is not None):
                # нестандартная ширина -> правило по ширине ненадёжно, решаем по содержимому
                emb0 = self.embedder.extract(info.img_u8)
                if emb0 is not None:
                    is_spine = int(self.region_fallback["model"].predict(emb0[None, :])[0]) == 1
                    if is_spine:
                        region, region_src = "spine", "content_fallback"
                    else:
                        region, side_src = detect_hip_side_region(info.img_u8)
                        region_src = f"content_fallback+{side_src}"
                    LOG.info("%s: non-standard width %d px -> region by content: %s", path.name, info.cols, region)
            debug["region_src"] = region_src
            debug.update(self._origin_hashes(path, info.ds))
            debug["sop_class_uid"] = _tag(info.ds, "SOPClassUID")
            debug["series_uid"] = _tag(info.ds, "SeriesInstanceUID")
            if self.sr_study and info.study_uid not in self._study_headers:
                try:
                    from dicom_sr import study_header_from_ds
                    self._study_headers[info.study_uid] = study_header_from_ds(info.ds)
                except Exception:  # noqa: BLE001
                    self._study_headers[info.study_uid] = {}
            feats = extract_geometry(info, region, self.cfg)

            need_emb = any((region, c) in self.registry.emb for c in self.cfg["criteria_by_region"][region]) \
                or region in self.registry.any_emb
            mirror = False
            if need_emb and region == "right_hip":
                # унифицированная модель бедра может требовать зеркалирования правого
                mb = next((self.registry.emb.get((region, c)) for c in self.cfg["criteria_by_region"][region]
                           if self.registry.emb.get((region, c)) is not None), None)
                mirror = bool(mb.meta.get("mirror_right", False)) if mb is not None else False
            embs: Dict[str, np.ndarray] = {}
            if need_emb:
                sources = {str(mb.meta.get("emb_source", EmbeddingExtractor.DEFAULT_SOURCE))
                           for mb in [self.registry.emb.get((region, c)) for c in self.cfg["criteria_by_region"][region]]
                           + [self.registry.any_emb.get(region)] if mb is not None}
                embs = self.embedder.extract_many(info.img_u8, sources, mirror=mirror)
            emb = embs.get(EmbeddingExtractor.DEFAULT_SOURCE) if embs else None
            if self.extras:  # [EXTRAS] входы для extras (незеркалированный imagenet-эмбеддинг, как в data/embeddings.npy)
                try:
                    import extras as _extras
                    emb_x = emb if (emb is not None and not mirror) else self.embedder.extract(info.img_u8)
                    self._extras_input = {"img_u8": info.img_u8, "emb": emb_x,
                                          "tags": _extras.tags_from_dataset(info.ds),
                                          "pixel_hash": _extras.pixel_hash(info.ds)}
                except Exception as ex:  # noqa: BLE001  — extras не должны ломать основной путь
                    LOG.warning("extras input failed for %s: %s", path, ex)
                    self._extras_input = None

            crit_results = {c: self.score_criterion(region, c, feats, embs)
                            for c in self.cfg["criteria_by_region"][region]}
            violations: List[str] = []
            for c, r in crit_results.items():
                if r["flag"]:
                    name = self.cfg["violations"][c]
                    if name not in violations:
                        violations.append(name)
            quality_class = 1 if violations else 0
            quality_prob = self.any_violation_prob(region, crit_results, feats, embs)
            debug["quality_prob_raw"] = round(float(quality_prob), 6)
            if self.cfg["stacking"].get("consistent_quality_prob", True):
                quality_prob = self.consistent_quality_prob(quality_prob, quality_class)
            # инвариант: quality_class == 1  <=>  quality_prob >= 0.5; класс и список нарушений согласованы всегда.

            row = {
                "path_to_study": self._path_str(path, root),
                "study_uid": info.study_uid,
                "image_uid": info.image_uid,
                "anatomical_region": official_region_name(region, self.cfg),
                "quality_class": int(quality_class),
                "violation_type": cfg_out["violation_separator"].join(violations),
                "quality_prob": round(float(quality_prob), 6),
                "processing_status": cfg_out["status_success"],
                "time_of_processing": 0.0,
            }
            debug.update({"internal_region": region, "region_source": region_src,
                          "rows": info.rows, "cols": info.cols, "warnings": "; ".join(info.warnings),
                          "embedding_used": bool(embs), "embedding_sources": "+".join(sorted(embs)) if embs else "",
                          **{f"feat_{k}": v for k, v in feats.items()
                                                                  if not isinstance(v, np.ndarray)}})
            for c, r in crit_results.items():
                for k in ("p_geom", "p_emb", "w_geom", "score", "threshold", "flag", "method", "margin", "uncertain", "p_cal"):
                    debug[f"{c}_{k}"] = r.get(k)
            # К3: строка «не уверен», если не уверен хотя бы один критерий региона; risk_level — правило
            # calibration_utils.risk_level (высокий: class=1 и уверен; средний: не уверен; низкий: class=0 и уверен)
            needs_review = int(any(int(r.get("uncertain", 0)) for r in crit_results.values()))
            debug["needs_review"] = needs_review
            debug["risk_level"] = risk_level(quality_class, needs_review)
            debug["uncertain_criteria"] = ";".join(c for c, r in crit_results.items() if int(r.get("uncertain", 0)))
            if info.warnings:
                LOG.info("%s: %s", path.name, "; ".join(info.warnings))

            self._emit_bonus_outputs(path, info, region, feats, crit_results, quality_class,
                                      violations, quality_prob, debug)
        except Exception as e:  # noqa: BLE001  — ЖЕЛЕЗНОЕ правило: строка всегда есть
            err = f"{type(e).__name__}: {e}"
            LOG.error("FAILURE %s -> %s", path, err)
            LOG.debug(traceback.format_exc())
            region = guess_region_without_pixels(path, self.cfg)
            rel = self._path_str(path, root)
            study_uid, image_uid = self._uids_without_pixels(path, rel)
            row = {
                "path_to_study": rel,
                "study_uid": study_uid,
                "image_uid": image_uid,
                "anatomical_region": official_region_name(region, self.cfg),
                "quality_class": 0,
                "violation_type": "",
                "quality_prob": float(cfg_out["fallback_quality_prob"]),
                "processing_status": cfg_out["status_failure"],
                "time_of_processing": 0.0,
            }
            debug.update({"internal_region": region, "region_source": "fallback", "error": err,
                          "needs_review": 1, "risk_level": risk_level(0, True), "uncertain_criteria": ""})
            debug.update(self._origin_hashes(path, None))
        row["time_of_processing"] = round(time.perf_counter() - t0, 4)
        debug["time_of_processing"] = row["time_of_processing"]
        return row, debug

    # ---- бонус-выходы (визуализация / DICOM SR / ROI-автокоррекция) — опционально,
    # никогда не бросает исключение наружу (process_file уже внутри своего try, но эта
    # функция сама оборачивает каждый под-шаг отдельно, чтобы сбой одного бонуса не
    # тушил остальные и уж тем более не портил основную строку CSV).
    def _emit_bonus_outputs(self, path: Path, info, region: str, feats: Dict[str, Any],
                            crit_results: Dict[str, Dict[str, Any]], quality_class: int,
                            violations: List[str], quality_prob: float, debug: Dict[str, Any]) -> None:
        if not (self.visualize_dir or self.sr_dir or self.roi_autocorrect_dir):
            return
        stem = path.stem
        violation_type_str = self.cfg["output"]["violation_separator"].join(violations)

        if self.visualize_dir:
            try:
                from visualize_report import save_overlay_png
                out_png = self.visualize_dir / f"{stem}_overlay.png"
                save_overlay_png(str(out_png), info.img_u8, region, feats,
                                  crit_results=crit_results, quality_class=quality_class,
                                  violation_type=violation_type_str)
                debug["bonus_overlay_png"] = str(out_png)
            except Exception as e:  # noqa: BLE001
                LOG.warning("visualize_report failed for %s: %s", path.name, e)
                debug["bonus_overlay_error"] = str(e)

        if self.sr_dir:
            try:
                from dicom_sr import save_sr
                out_sr = self.sr_dir / f"{stem}_sr.dcm"
                save_sr(str(out_sr), info.ds, region, quality_class, violation_type_str,
                        quality_prob, feats)
                debug["bonus_sr_dcm"] = str(out_sr)
            except Exception as e:  # noqa: BLE001
                LOG.warning("dicom_sr failed for %s: %s", path.name, e)
                debug["bonus_sr_error"] = str(e)

        if self.roi_autocorrect_dir and region in ("right_hip", "left_hip"):
            try:
                from auto_roi import suggest_hip_roi, draw_roi_correction
                suggestion = suggest_hip_roi(info.img_u8, feats)
                debug["bonus_roi_needs_correction"] = suggestion.get("needs_correction")
                debug["bonus_roi_reason"] = suggestion.get("reason")
                debug["bonus_roi_deficit_mm"] = suggestion.get("deficit_mm")
                if suggestion.get("needs_correction"):
                    import cv2
                    out_png = self.roi_autocorrect_dir / f"{stem}_roi_correction.png"
                    overlay = draw_roi_correction(info.img_u8, suggestion)
                    cv2.imwrite(str(out_png), overlay)
                    debug["bonus_roi_png"] = str(out_png)
            except Exception as e:  # noqa: BLE001
                LOG.warning("auto_roi failed for %s: %s", path.name, e)
                debug["bonus_roi_error"] = str(e)

    def _path_str(self, path: Path, root: Optional[Path]) -> str:
        mode = self.cfg["output"].get("path_mode", "relative")
        try:
            if mode == "name":
                return path.name
            if mode == "absolute" or root is None:
                return str(path.resolve())
            shown = getattr(self, "_display_paths", {}).get(path.resolve())
            if shown:
                return shown
            return str(path.resolve().relative_to(Path(root).resolve()))
        except Exception:  # noqa: BLE001
            return str(path)

    @staticmethod
    def _uids_without_pixels(path: Path, rel: Optional[str] = None) -> Tuple[str, str]:
        """UID для строки Failure: из тегов, если читаются; иначе детерминированный хэш ОТНОСИТЕЛЬНОГО
        пути (path_to_study), одинаковый на любой машине и в контейнере (нужно для verify.sh / эталона)."""
        try:
            ds = pydicom.dcmread(str(path), force=True, stop_before_pixels=True)
            s, i = _tag(ds, "StudyInstanceUID"), _tag(ds, "SOPInstanceUID")
        except Exception:  # noqa: BLE001
            s, i = "", ""
        rel_p = Path(rel) if rel else path
        return (s or "hash-" + path_hash(rel_p.parent)), (i or "hash-" + path_hash(rel_p))

    @staticmethod
    def _origin_hashes(path: Path, ds) -> Dict[str, str]:
        """SHA-256 исходного файла (побайтово) и массива пикселей — хэш неизменности оригинала.
        Никогда не бросает исключение."""
        out = {"sha256_file": "", "sha256_pixels": ""}
        try:
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            out["sha256_file"] = h.hexdigest()
        except Exception:  # noqa: BLE001
            pass
        try:
            if ds is not None and "PixelData" in ds:
                out["sha256_pixels"] = hashlib.sha256(bytes(ds.PixelData)).hexdigest()
        except Exception:  # noqa: BLE001
            pass
        return out

    def write_study_sr(self, rows: List[Dict[str, Any]], debug_rows: List[Dict[str, Any]],
                       out_dir: Path) -> Dict[str, str]:
        """Один DICOM SR на каждое исследование партии (включая норму и Failure).
        Возвращает {study_uid: путь к SR}. Ошибка одного исследования не мешает остальным."""
        from dicom_sr import save_study_sr, study_sr_filename
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sep = self.cfg["output"]["violation_separator"]
        model_version = str(self.cfg.get("version", __version__))
        cfg_hash = config_hash(self.cfg)
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for i, r in enumerate(rows):
            dbg = debug_rows[i] if i < len(debug_rows) else {}
            groups.setdefault(str(r.get("study_uid") or ""), []).append({
                "image_uid": str(r.get("image_uid") or ""),
                "sop_class_uid": dbg.get("sop_class_uid") or "",
                "series_uid": dbg.get("series_uid") or "",
                "anatomical_region": str(r.get("anatomical_region") or ""),
                "quality_class": int(r.get("quality_class") or 0),
                "violations": [v.strip() for v in str(r.get("violation_type") or "").split(sep) if v.strip()],
                "quality_prob": r.get("quality_prob"),
                "processing_status": str(r.get("processing_status") or ""),
                "sha256_file": dbg.get("sha256_file") or "",
                "sha256_pixels": dbg.get("sha256_pixels") or "",
                "path_to_study": str(r.get("path_to_study") or ""),
            })
        written: Dict[str, str] = {}
        for study_uid, items in groups.items():
            if not study_uid:
                continue
            try:
                out = out_dir / study_sr_filename(study_uid)
                save_study_sr(str(out), study_uid, items, model_version, cfg_hash,
                              self._study_headers.get(study_uid, {}))
                written[study_uid] = str(out)
            except Exception as e:  # noqa: BLE001
                LOG.warning("study SR failed for %s: %s", study_uid, e)
        LOG.info("Study SR written: %d of %d studies -> %s", len(written), len([g for g in groups if g]), out_dir)
        return written

    # ---- пакет
    def run(self, input_path: Path, output_csv: Path, debug_csv: Optional[Path] = None,
            xlsx: bool = False, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        tmp_dirs: List[Path] = []
        t_start = time.perf_counter()
        rows: List[Dict[str, Any]] = []
        debug_rows: List[Dict[str, Any]] = []
        try:
            self._display_paths: Dict[Path, str] = {}
            root, files = discover_files(Path(input_path), tmp_dirs, self._display_paths)
            if limit:
                files = files[:limit]
            LOG.info("Found %d DICOM candidate files under %s", len(files), input_path)
            if not files:
                LOG.warning("No DICOM files found -> writing empty CSV with header")
            if files and (self.registry.emb or self.registry.any_emb):
                # прогрев всех нужных backbone вне замера time_of_processing
                for mb in list(self.registry.emb.values()) + list(self.registry.any_emb.values()):
                    self.embedder._init(str(mb.meta.get("emb_source", EmbeddingExtractor.DEFAULT_SOURCE)))
            extras_inputs: List[Optional[Dict[str, Any]]] = []  # [EXTRAS]
            for i, f in enumerate(files, 1):
                self._extras_input = None
                row, dbg = self.process_file(f, root)
                rows.append(row)
                debug_rows.append(dbg)
                extras_inputs.append(self._extras_input)
                if i % 25 == 0 or i == len(files):
                    LOG.info("  %d/%d processed (%.1fs)", i, len(files), time.perf_counter() - t_start)
        finally:
            for d in tmp_dirs:
                shutil.rmtree(d, ignore_errors=True)

        write_results(rows, Path(output_csv), self.cfg, xlsx=xlsx)
        self.last_debug_rows = debug_rows  # для API: детали по критериям без повторного чтения CSV
        self.last_study_sr = {}
        if self.sr_study:
            try:
                sr_dir = self.sr_study_dir or (Path(output_csv).with_suffix(".csv").parent / "sr")
                self.last_study_sr = self.write_study_sr(rows, debug_rows, sr_dir)
            except Exception as e:  # noqa: BLE001  — бонус не должен ломать основной выход
                LOG.warning("study SR not written: %s", e)
        if debug_csv:
            write_debug(debug_rows, Path(debug_csv))
        self.last_extras_rows = []
        if self.extras:  # [EXTRAS] отдельный файл <output>_extras.csv; основной CSV уже записан
            try:
                import extras as _extras
                self.last_extras_rows = _extras.compute_extras_for_rows(rows, debug_rows, extras_inputs,
                                                                        models_dir=self.registry.models_dir)
                extras_csv = Path(output_csv).with_suffix(".csv")
                extras_csv = extras_csv.with_name(extras_csv.stem + "_extras.csv")
                _extras.write_extras_csv(self.last_extras_rows, extras_csv)
                LOG.info("extras written -> %s", extras_csv)
            except Exception as e:  # noqa: BLE001  — extras не должны ломать основной выход
                LOG.warning("extras not written: %s", e)
        n_fail = sum(1 for r in rows if r["processing_status"] == self.cfg["output"]["status_failure"])
        LOG.info("DONE: %d rows, %d failures, %.1fs total -> %s", len(rows), n_fail,
                 time.perf_counter() - t_start, output_csv)
        return rows


# --------------------------------------------------------------------------- #
# Запись результатов
# --------------------------------------------------------------------------- #
def write_results(rows: List[Dict[str, Any]], output_csv: Path, cfg: Dict[str, Any], xlsx: bool = False):
    cols = cfg["output"]["columns"]
    output_csv = Path(output_csv)
    if output_csv.suffix.lower() == ".xlsx":       # запросили xlsx -> CSV пишем рядом, xlsx по указанному пути
        xlsx = True
        output_csv = output_csv.with_suffix(".csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding=cfg["output"].get("csv_encoding", "utf-8")) as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL,
                           lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    os.replace(tmp, output_csv)  # атомарная запись
    if xlsx:
        try:
            xlsx_path = output_csv.with_suffix(".xlsx")
            write_xlsx(rows, xlsx_path, cfg)
            LOG.info("XLSX written: %s", xlsx_path)
        except Exception as e:  # noqa: BLE001
            LOG.warning("XLSX not written (%s); CSV is the primary artifact", e)


def write_xlsx(rows: List[Dict[str, Any]], xlsx_path: Path, cfg: Dict[str, Any]) -> None:
    """XLSX для человека: лист «Результаты» с теми же колонками, что и CSV (закреплённая
    шапка, автофильтр, подсветка нарушений и сбоев), лист «Сводка» (счётчики по областям и
    типам нарушений) и лист «Исследования» (агрегат по study_uid). Официальный артефакт —
    CSV; XLSX — удобное представление тех же данных."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    cols = cfg["output"]["columns"]
    fail = cfg["output"]["status_failure"]
    sep = cfg["output"]["violation_separator"]
    head_font = Font(bold=True, color="FFFFFF")
    head_fill = PatternFill("solid", fgColor="01696F")
    fill_viol = PatternFill("solid", fgColor="FBE9F2")
    fill_fail = PatternFill("solid", fgColor="F3E4D5")
    fill_ok = PatternFill("solid", fgColor="E9F1E3")

    def _header(ws, names):
        ws.append(list(names))
        for c in ws[1]:
            c.font, c.fill = head_font, head_fill
            c.alignment = Alignment(vertical="center", wrap_text=True)
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(names))}{max(2, ws.max_row)}"

    def _autowidth(ws, minimum=8, maximum=60):
        for i, col in enumerate(ws.iter_cols(min_row=1, max_row=min(ws.max_row, 500)), 1):
            w = max((len(str(c.value)) for c in col if c.value is not None), default=minimum)
            ws.column_dimensions[get_column_letter(i)].width = max(minimum, min(maximum, w + 2))

    wb = Workbook()
    ws = wb.active
    ws.title = "Результаты"
    extra = ["Действие"]
    _header(ws, cols + extra)
    for r in rows:
        is_fail = r.get("processing_status") == fail
        is_viol = str(r.get("quality_class")) == "1"
        action = "Проверить вручную" if is_fail else ("Проверить / переснять" if is_viol else "Принять")
        ws.append([r.get(c, "") for c in cols] + [action])
        f = fill_fail if is_fail else (fill_viol if is_viol else fill_ok)
        for c in ws[ws.max_row]:
            c.fill = f
    ws.auto_filter.ref = f"A1:{get_column_letter(len(cols) + len(extra))}{max(2, ws.max_row)}"
    _autowidth(ws)

    # --- Сводка
    ws2 = wb.create_sheet("Сводка")
    n = len(rows)
    n_fail = sum(1 for r in rows if r.get("processing_status") == fail)
    n_viol = sum(1 for r in rows if str(r.get("quality_class")) == "1")
    n_ok = n - n_fail - n_viol
    t_sum = sum(float(r.get("time_of_processing") or 0) for r in rows)
    studies = {r.get("study_uid") for r in rows if r.get("study_uid")}
    _header(ws2, ["Показатель", "Значение"])
    for k, v in [
        ("Файлов обработано", n), ("Исследований", len(studies)),
        ("Без нарушений", n_ok), ("С нарушениями", n_viol),
        ("Доля с нарушениями", f"{(n_viol / n * 100):.1f} %" if n else "—"),
        ("Ошибок обработки (Failure)", n_fail),
        ("Суммарное время обработки, с", round(t_sum, 2)),
        ("Среднее время на файл, с", round(t_sum / n, 3) if n else "—"),
        ("Версия модели", cfg.get("version", __version__)),
    ]:
        ws2.append([k, v])
    ws2.append([])
    ws2.append(["Область", "Файлов", "С нарушениями", "Ошибок"])
    for c in ws2[ws2.max_row]:
        c.font = Font(bold=True)
    for reg in sorted({r.get("anatomical_region", "") for r in rows}):
        sub = [r for r in rows if r.get("anatomical_region", "") == reg]
        ws2.append([reg or "—", len(sub), sum(1 for r in sub if str(r.get("quality_class")) == "1"),
                    sum(1 for r in sub if r.get("processing_status") == fail)])
    ws2.append([])
    ws2.append(["Тип нарушения", "Файлов"])
    for c in ws2[ws2.max_row]:
        c.font = Font(bold=True)
    counts: Dict[str, int] = {}
    for r in rows:
        for v in str(r.get("violation_type") or "").split(sep):
            if v.strip():
                counts[v.strip()] = counts.get(v.strip(), 0) + 1
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        ws2.append([k, v])
    ws2.auto_filter.ref = None
    _autowidth(ws2, minimum=14)

    # --- Исследования
    ws3 = wb.create_sheet("Исследования")
    _header(ws3, ["study_uid", "Файлов", "С нарушениями", "Ошибок", "Макс. quality_prob", "Нарушения"])
    by_study: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_study.setdefault(str(r.get("study_uid") or ""), []).append(r)
    for uid, sub in sorted(by_study.items(), key=lambda kv: -max(float(x.get("quality_prob") or 0) for x in kv[1])):
        viols: List[str] = []
        for r in sub:
            for v in str(r.get("violation_type") or "").split(sep):
                if v.strip() and v.strip() not in viols:
                    viols.append(v.strip())
        nv = sum(1 for r in sub if str(r.get("quality_class")) == "1")
        ws3.append([uid, len(sub), nv, sum(1 for r in sub if r.get("processing_status") == fail),
                    round(max(float(x.get("quality_prob") or 0) for x in sub), 3), "; ".join(viols)])
        if nv:
            for c in ws3[ws3.max_row]:
                c.fill = fill_viol
    ws3.auto_filter.ref = f"A1:F{max(2, ws3.max_row)}"
    _autowidth(ws3, minimum=10)
    wb.save(xlsx_path)


def write_debug(debug_rows: List[Dict[str, Any]], path: Path):
    try:
        import pandas as pd
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(debug_rows).to_csv(path, index=False)
        LOG.info("Debug CSV written: %s", path)
    except Exception as e:  # noqa: BLE001
        LOG.warning("Debug CSV not written: %s", e)


def validate_output_csv(path: Path, cfg: Optional[Dict[str, Any]] = None,
                        schema_check: bool = True) -> List[str]:
    """Проверка выходного файла на соответствие официальному формату (правила из config.yaml
    + JSON Schema schema/results_row.schema.json). Возвращает список проблем."""
    cfg = cfg or load_config()
    problems: List[str] = []
    rows_for_schema: List[Dict[str, str]] = []
    allowed_regions = {cfg["regions"]["spine"], cfg["regions"]["hip"]}
    allowed_viol = set(cfg["violations"].values())
    sep = cfg["output"]["violation_separator"]
    with open(path, "r", encoding=cfg["output"].get("csv_encoding", "utf-8"), newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != cfg["output"]["columns"]:
            problems.append(f"columns mismatch: {reader.fieldnames}")
        for i, r in enumerate(reader, 2):
            if r["anatomical_region"] not in allowed_regions:
                problems.append(f"line {i}: bad anatomical_region '{r['anatomical_region']}'")
            if r["quality_class"] not in ("0", "1"):
                problems.append(f"line {i}: quality_class must be 0/1, got '{r['quality_class']}'")
            try:
                p = float(r["quality_prob"])
                if not 0.0 <= p <= 1.0:
                    problems.append(f"line {i}: quality_prob out of [0,1]: {p}")
            except ValueError:
                problems.append(f"line {i}: quality_prob not float: '{r['quality_prob']}'")
            if r["processing_status"] not in (cfg["output"]["status_success"], cfg["output"]["status_failure"]):
                problems.append(f"line {i}: bad processing_status '{r['processing_status']}'")
            if r["violation_type"]:
                for v in r["violation_type"].split(sep):
                    if v.strip() not in allowed_viol:
                        problems.append(f"line {i}: unknown violation '{v}'")
                if r["quality_class"] != "1":
                    problems.append(f"line {i}: violations listed but quality_class != 1")
            elif r["quality_class"] == "1":
                problems.append(f"line {i}: quality_class=1 but violation_type empty")
            try:
                float(r["time_of_processing"])
            except ValueError:
                problems.append(f"line {i}: time_of_processing not float")
            rows_for_schema.append(r)
    # JSON Schema (schema/results_row.schema.json) — второй, независимый источник истины формата.
    # Проверка через пакет jsonschema, если он установлен, иначе встроенный мини-валидатор
    # (src/schema_check.py). Отсутствие файла схемы — предупреждение, не ошибка формата.
    if schema_check:
        try:
            from schema_check import load_schema, validate_results_rows
            problems.extend(validate_results_rows(rows_for_schema, load_schema("results_row")))
        except FileNotFoundError as e:
            LOG.warning("JSON Schema not found, schema check skipped: %s", e)
        except Exception as e:  # noqa: BLE001
            LOG.warning("JSON Schema check failed to run: %s", e)
    return problems


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def setup_logging(log_file: Optional[Path], verbose: bool):
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", handlers=handlers, force=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="DensitoAI — пакетный инференс качества DXA (DICOM -> CSV)")
    ap.add_argument("--input", "-i", default=None, help="Папка с исследованиями, один DICOM или zip-архив (обязателен, кроме --validate-only)")
    ap.add_argument("--output", "-o", required=True, help="Путь к выходному CSV (или .xlsx)")
    ap.add_argument("--models-dir", default=None, help=f"Папка с моделями (по умолчанию {MODELS_DIR})")
    ap.add_argument("--config", default=None, help=f"config.yaml (по умолчанию {CONFIG_PATH})")
    ap.add_argument("--debug-csv", nargs="?", const="auto", default=None,
                    help="Расширенный CSV с признаками/скорами (по умолчанию <output>_debug.csv)")
    ap.add_argument("--xlsx", action="store_true", help="Дополнительно записать .xlsx рядом с CSV")
    ap.add_argument("--no-embeddings", action="store_true", help="Отключить контур B (только геометрия)")
    ap.add_argument("--path-mode", choices=["relative", "absolute", "name"], default=None)
    ap.add_argument("--limit", type=int, default=None, help="Обработать только первые N файлов (отладка)")
    ap.add_argument("--log-file", default=None, help="Файл лога (по умолчанию <output>.log)")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--validate-only", action="store_true",
                    help="Только проверить формат уже существующего CSV (--output) и выйти")
    ap.add_argument("--visualize-dir", default=None, help="[BONUS] Papka dlya PNG-overlayev s geometricheskimi priznakami kachestva")
    ap.add_argument("--sr-dir", default=None, help="[BONUS] Papka dlya DICOM Structured Report (.dcm) s rezultatami")
    ap.add_argument("--roi-autocorrect-dir", default=None, help="[BONUS] Papka dlya diagnosticheskih PNG s predlozheniem korrektsii ROI (bedro)")
    ap.add_argument("--sr-study", action="store_true",
                    help="[BONUS] Один DICOM SR на исследование (включая норму) в <каталог CSV>/sr/<study_uid>_SR.dcm")
    ap.add_argument("--sr-study-dir", default=None, help="[BONUS] Каталог для SR на исследование (включает --sr-study)")
    ap.add_argument("--extras", action="store_true",
                    help="[EXTRAS] Дополнительно записать <output>_extras.csv (белые линии, OOD-gate, эндопротез, когерентность)")
    args = ap.parse_args(argv)

    output_csv = Path(args.output)
    if output_csv.suffix.lower() == ".xlsx":   # CSV — основной артефакт, xlsx пишется рядом
        args.xlsx = True
        output_csv = output_csv.with_suffix(".csv")
    setup_logging(Path(args.log_file) if args.log_file else output_csv.with_suffix(".log"), args.verbose)
    cfg = load_config(Path(args.config) if args.config else None)
    if args.path_mode:
        cfg["output"]["path_mode"] = args.path_mode

    if not args.validate_only and not args.input:
        ap.error("--input/-i обязателен (кроме режима --validate-only)")
    if args.validate_only:
        problems = validate_output_csv(output_csv, cfg)
        print("\n".join(problems) if problems else "OK: output format valid")
        return 1 if problems else 0

    try:
        engine = DensitoInference(cfg=cfg, models_dir=args.models_dir, use_embeddings=not args.no_embeddings,
                                   visualize_dir=args.visualize_dir, sr_dir=args.sr_dir,
                                   roi_autocorrect_dir=args.roi_autocorrect_dir,
                                   sr_study=args.sr_study, sr_study_dir=args.sr_study_dir, extras=args.extras)
        debug_csv = None
        if args.debug_csv:
            debug_csv = (output_csv.with_name(output_csv.stem + "_debug.csv") if args.debug_csv == "auto"
                         else Path(args.debug_csv))
        engine.run(Path(args.input), output_csv, debug_csv=debug_csv, xlsx=args.xlsx, limit=args.limit)
        problems = validate_output_csv(output_csv, cfg)
        if problems:
            LOG.error("Output format problems: %s", problems[:10])
        else:
            LOG.info("Output format check: OK")
        return 0
    except Exception as e:  # noqa: BLE001  — последний рубеж: файл с заголовком всё равно должен быть
        LOG.critical("Fatal error in batch: %s\n%s", e, traceback.format_exc())
        try:
            if not output_csv.exists():
                write_results([], output_csv, cfg)
        except Exception:  # noqa: BLE001
            pass
        return 2


if __name__ == "__main__":
    sys.exit(main())
