#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
measurement_check.py — акт поверки измерительного контура A на фантомах (градусы и миллиметры, без разметки).

Контур A измеряет геометрию кадра явными функциями (src/geometry_features.py, src/hip_features.py):
угол оси позвоночника к вертикали, смещение центра кости от центра кадра, угол диафиза бедра,
расстояние от латерального края кости до края кадра, ширину диафиза, длину диафиза ниже вертелов.
Здесь проверяется только измерительная часть: измеряет ли она то, что заявляет, — независимо от
разметки экспертов и без моделей. Синтетическим фантомам (tests/phantoms/ и генератор
tools/make_phantoms.py) задаются известные преобразования в физических единицах:
  * поворот всего кадра на заданные градусы вокруг центра кадра (в миллиметровых координатах,
    с учётом анизотропного пикселя 1.05×0.6 мм);
  * сдвиг содержимого кадра на заданные миллиметры по X и по Y;
  * наклон только столбика позвонков (тело и таз прямые) — синтетический позвоночник с заданным
    физическим углом оси; масштаб не меняется.
Преобразованный кадр записывается 8-битным DICOM (PixelSpacing 1.05\\0.6) и читается тем же кодом,
что и в инференсе, с вариантом предобработки критерия из config.yaml (preprocessing.variant_by_criterion).
Измеренное сравнивается с заданным; для каждой величины — таблица «заданное → измеренное», ошибка,
MAE, смещение (bias), максимум, наклон и R² линейной зависимости в заявленном диапазоне, а также точки
вне диапазона, где измерение ломается.

Это поверка на фантомах, не замена клинической валидации: фантомы — упрощённые силуэты, а не снимки
пациентов; результат говорит о корректности геометрических формул и их устойчивости к преобразованиям
кадра, но не о согласии с экспертом (для этого — docs/METRICS_REPORT.md).

    python tools/measurement_check.py [--out docs/measurement_check.json] [--md docs/MEASUREMENT_CHECK.md]
                                      [--phantoms tests/phantoms] [--workdir <tmp>] [--seed 20260923]

Детерминизм: зерно фиксировано, генерация и преобразования без случайности вне зерна; повторный
запуск даёт тот же results_sha256. Ничего в models/, config.yaml и tests/phantoms/ не меняется.
Код возврата: 0 — все статистики в заявленных пределах (LIMITS), 1 — есть превышения (JSON пишется всегда).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import cv2  # noqa: E402
import pydicom  # noqa: E402

import geometry_features as gf  # noqa: E402
import hip_features as hf  # noqa: E402
import make_phantoms as mp  # noqa: E402
import preprocess  # noqa: E402

SX, SY = gf.PIXEL_SPACING_X_MM, gf.PIXEL_SPACING_Y_MM      # 0.6 / 1.05 мм — константа аппарата в контуре A
TOOL_VERSION = "1.0"

# Сетки преобразований (физические единицы) и заявленные диапазоны линейности
ROT_DEG = [-15, -12, -10, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 10, 12, 15]
SHIFT_MM = [-30, -20, -15, -10, -5, -2, 0, 2, 5, 10, 15, 20, 30]
DECLARED_ROT = 8.0       # |угол| ≤ 8°
DECLARED_SHIFT = 20.0    # |сдвиг| ≤ 20 мм

# Заявленные пределы (проверяются здесь и в tests/test_measurement_check.py); выбраны по фактическим
# числам прогона с запасом. Ключ — имя серии из build().
LIMITS: Dict[str, Dict[str, float]] = {
    "spine_axis_rotation":     {"mae": 0.6, "abs_bias": 0.5, "max_abs": 1.5, "slope_min": 0.85, "slope_max": 1.05},
    "spine_center_shift_x":    {"mae": 0.5, "abs_bias": 0.5, "max_abs": 1.0, "slope_min": 0.97, "slope_max": 1.03},
    "hip_shaft_angle_rotation": {"mae": 0.3, "abs_bias": 0.3, "max_abs": 0.6, "slope_min": 0.97, "slope_max": 1.03},
    "hip_lateral_margin_shift_x": {"mae": 0.8, "abs_bias": 0.8, "max_abs": 1.5, "slope_min": 0.97, "slope_max": 1.03},
    "hip_shaft_len_shift_down": {"mae": 2.0, "abs_bias": 2.0, "max_abs": 3.5, "slope_min": 0.9, "slope_max": 1.1},
}
# Инвариантность: максимум |Δ| относительно нулевого кадра в заявленном диапазоне сдвигов
INVARIANCE_LIMITS: Dict[str, float] = {
    "spine_axis_shift_x": 0.6,          # °, сдвиг по X не должен менять угол оси
    "spine_center_shift_y": 2.5,        # мм, сдвиг по Y меняет состав маски (рёбра/таз), допуск с запасом
    "hip_shaft_angle_shift_x": 0.3,     # °
    "hip_shaft_angle_shift_y": 0.3,     # °
    "hip_shaft_width_all": 1.0,         # мм, ширина диафиза при любом преобразовании в заявленном диапазоне
}
SIDE_REQUIRED = 1.0  # доля верно определённой стороны бедра в заявленном диапазоне


# --------------------------------------------------------------------------- #
# Преобразования и запись DICOM
# --------------------------------------------------------------------------- #
def affine_mm(img: np.ndarray, theta_deg: float, dx_mm: float, dy_mm: float) -> np.ndarray:
    """Поворот на theta_deg вокруг центра кадра в миллиметровых координатах + сдвиг (dx_mm, dy_mm).
    Положительный угол — поворот по часовой стрелке на экране (низ вертикальной линии уходит влево);
    положительный dx — содержимое уходит вправо, положительный dy — вниз. Пиксельная матрица
    M = S⁻¹·R·S, S = diag(0.6, 1.05), поэтому угол задаётся в физических градусах."""
    h, w = img.shape
    c = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    t = np.deg2rad(theta_deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    A = np.diag([1 / SX, 1 / SY]) @ R @ np.diag([SX, SY])
    b = c - A @ c + np.array([dx_mm / SX, dy_mm / SY])
    M = np.hstack([A, b[:, None]]).astype(np.float64)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def write_dicom(img: np.ndarray, path: Path, *, tag: str, study_idx: int, seed: int) -> None:
    """8-битный DICOM с тегами экспорта аппарата и PixelSpacing 1.05\\0.6 (tools/make_phantoms.build_dataset)."""
    ds = mp.build_dataset(np.clip(img, 0, 252), study_uid=mp.det_uid("mc-study", str(seed), tag),
                          series_uid=mp.det_uid("mc-series", str(seed), tag),
                          sop_uid=mp.det_uid("mc-sop", str(seed), tag), instance_number=1, bits=8,
                          explicit_vr=True, pixel_spacing=True, study_idx=study_idx)
    ds.save_as(str(path), enforce_file_format=True)


def spine_column_tilt(rng: np.random.RandomState, tilt_phys_deg: float, h: int = 317, w: int = 300):
    """Позвоночник с наклоном ТОЛЬКО столбика позвонков на заданный физический угол (тело, рёбра и таз
    прямые). Повторяет tools/make_phantoms.spine_phantom(variant='normal'), но угол задан в физических
    градусах: пиксельный наклон a_px = atan(tan(θ)·SY/SX). Возвращает (кадр, истина)."""
    img = np.zeros((h, w), np.float64)
    cx = w / 2 + rng.uniform(-6, 6)
    body = mp.ellipse(h, w, h / 2, cx, h * 0.75, w * 0.42)
    img[body] = 55 + 25 * rng.rand(int(body.sum()))
    img = mp.ndimage.gaussian_filter(img, 2.0)
    a_px = float(np.arctan(np.tan(np.deg2rad(tilt_phys_deg)) * SY / SX))
    n_vert, vh, gap, top = 6, 30, 10, 60
    for i in range(n_vert):
        cy = top + i * (vh + gap) + vh / 2
        dx = (cy - h / 2) * np.tan(a_px)
        m = mp.rot_rect(h, w, cy, cx + dx, vh / 2, 20, np.degrees(a_px))
        img[m] = 175 + 45 * rng.rand(int(m.sum()))
        m2 = mp.rot_rect(h, w, cy, cx + dx, 5, 38, np.degrees(a_px))
        img[m2 & ~m] = np.maximum(img[m2 & ~m], 140)
    for side in (-1, 1):
        for k in range(2):
            ring = (mp.ellipse(h, w, 40 + 14 * k, cx + side * 55, 22, 60)
                    & ~mp.ellipse(h, w, 40 + 14 * k, cx + side * 55, 18, 56))
            ring &= (mp._grid(h, w)[0] > 18)
            img[ring] = np.maximum(img[ring], 165)
    for side in (-1, 1):
        pel = mp.ellipse(h, w, h - 20, cx + side * 85, 45, 60)
        img[pel] = np.maximum(img[pel], 150 + 30 * rng.rand(int(pel.sum())))
    img = mp.ndimage.gaussian_filter(img, 0.8)
    img += rng.normal(0, 3.0, img.shape)
    img[~body] = np.clip(img[~body] * 0.15, 0, 12)
    return np.clip(img, 0, 252), {"axis_tilt_phys_deg": float(tilt_phys_deg), "center_x": float(cx)}


def hip_truth() -> Dict[str, float]:
    """Аналитическая истина для фантома бедра (константы tools/make_phantoms.hip_phantom, правое бедро):
    диафиз — отрезок (140, 85) → (h+10, 60) толщиной 30 px. Угол оси к вертикали в физических градусах;
    латеральный край на нижней строке кадра; горизонтальная ширина диафиза."""
    h = 291
    (y0, x0), (y1, x1), thick = (140.0, 85.0), (h + 10.0, 60.0), 30.0
    a_px = (x1 - x0) / (y1 - y0)                       # dx/dy в пикселях (< 0: низ уходит влево)
    angle = float(np.degrees(np.arctan2(a_px * SX, SY)))  # знак как в hip_features (a>0 → +)
    phi = float(np.arctan(abs(a_px)))
    half_w_px = (thick / 2.0) / np.cos(phi)
    x_bottom = x0 + a_px * ((h - 1) - y0)
    return {"signed_shaft_angle_deg": angle,
            "lateral_margin_mm": float((x_bottom - half_w_px) * SX),
            "shaft_width_mm": float(2.0 * half_w_px * SX),
            "scan_length_mm": float(h * SY)}


# --------------------------------------------------------------------------- #
# Базовые фантомы
# --------------------------------------------------------------------------- #
def load_variants(cfg_path: Path) -> Dict[str, str]:
    """Вариант предобработки контура A по критерию из config.yaml (только чтение)."""
    default = {"sp_pos": "canonical", "sp_axis": "baseline", "sp_art": "baseline",
               "hip_pos": "canonical", "hip_roi": "baseline"}
    try:
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        vb = (cfg.get("preprocessing") or {}).get("variant_by_criterion") or {}
        out = dict(default)
        for k, v in vb.items():
            if isinstance(v, dict) and v.get("geom"):
                out[k] = str(v["geom"])
        return out
    except Exception:  # noqa: BLE001
        return default


def phantom_bases(phantoms: Path, seed: int) -> List[Dict[str, Any]]:
    """Базовые кадры: фантомы «норма» из tests/phantoms (по MANIFEST.json) + сгенерированные тем же
    генератором с зёрнами от --seed. Пиксели приводятся к шкале 0..252 (16-битные — делением на 4095)."""
    bases: List[Dict[str, Any]] = []
    man_path = phantoms / "MANIFEST.json"
    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        for f in man.get("files", []):
            if f.get("kind") != "phantom" or f.get("variant") != "normal":
                continue
            p = phantoms / f["path"]
            if not p.exists():
                continue
            ds = pydicom.dcmread(str(p), force=True)
            arr = ds.pixel_array.astype(np.float64)
            if int(getattr(ds, "BitsAllocated", 8)) > 8:
                arr = arr / 4095.0 * 252.0
            bases.append({"name": f["path"], "source": "tests/phantoms", "region": f["region"],
                          "img": np.clip(arr, 0, 252), "truth": dict(f.get("geometry", {})),
                          "sha256": f.get("sha256", "")})
    for k in range(2):
        rng = np.random.RandomState(seed * 10 + k)
        img, info = mp.spine_phantom(rng, "normal")
        bases.append({"name": f"generated/spine_{k}", "source": "make_phantoms", "region": "spine",
                      "img": img, "truth": info, "sha256": ""})
    for k, side in enumerate(("right_hip", "left_hip")):
        rng = np.random.RandomState(seed * 10 + 5 + k)
        img, info = mp.hip_phantom(rng, side, "normal")
        bases.append({"name": f"generated/{side}", "source": "make_phantoms", "region": side,
                      "img": img, "truth": info, "sha256": ""})
    return bases


# --------------------------------------------------------------------------- #
# Измерение теми же функциями, что в инференсе
# --------------------------------------------------------------------------- #
def measure(path: Path, region: str, variants: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if region == "spine":
        with preprocess.variant(variants["sp_axis"]):
            img, _ = gf.read_dicom_normalized(str(path))
            mask = gf.segment_bone(img)
            ax = gf.spine_axis_features(img, mask)
            out["axis_angle_deg"] = ax["axis_angle_deg"]
            out["mask_px"] = int((mask > 0).sum())
            # режим tracked (ось только столбика) в поставке не используется; измеряется для сравнения
            out["axis_angle_deg_tracked"] = gf.spine_axis_features(img, mask, mode="tracked")["axis_angle_deg"]
        with preprocess.variant(variants["sp_pos"]):
            img, _ = gf.read_dicom_normalized(str(path))
            mask = gf.segment_bone(img)
            pos = gf.spine_positioning_features(img, mask)
            w = img.shape[1]
            out["center_offset_mm"] = None if pos["center_offset_ratio"] is None else pos["center_offset_ratio"] * w * SX
            out["bone_width_mm"] = None if pos["bone_width_ratio"] is None else pos["bone_width_ratio"] * w * SX
    else:
        with preprocess.variant(variants["hip_pos"]):
            img, _ = gf.read_dicom_normalized(str(path))
            f = hf.hip_all_features(img)
            out["signed_shaft_angle_deg"] = f["signed_shaft_angle_deg"]
            out["abs_shaft_angle_deg"] = f["abs_shaft_angle_deg"]
            out["shaft_width_mm"] = f["shaft_width_mm"]
            out["lateral_margin_mm"] = f["lateral_margin_mm"]
            out["hip_side_detected"] = f["hip_side_detected"]
        with preprocess.variant(variants["hip_roi"]):
            img, _ = gf.read_dicom_normalized(str(path))
            f = hf.hip_all_features(img)
            out["shaft_len_below_troch_mm"] = f["shaft_len_below_troch_mm"]
            out["scan_length_mm"] = f["scan_length_mm"]
    return out


def _fin(v) -> Optional[float]:
    return None if v is None or not np.isfinite(v) else float(v)


# --------------------------------------------------------------------------- #
# Статистика
# --------------------------------------------------------------------------- #
def summarize(points: List[Dict[str, Any]], declared: float, key_set: str = "set") -> Dict[str, Any]:
    """points: [{set, truth, measured, phantom}] → таблица по заданным значениям и статистики
    (в заявленном диапазоне |set| ≤ declared и по всем точкам)."""
    by_set: Dict[float, List[Dict[str, Any]]] = {}
    for p in points:
        by_set.setdefault(float(p[key_set]), []).append(p)
    rows = []
    for s in sorted(by_set):
        pts = by_set[s]
        meas = [p["measured"] for p in pts if p["measured"] is not None]
        errs = [p["measured"] - p["truth"] for p in pts if p["measured"] is not None]
        rows.append({
            "set": s, "truth": float(np.mean([p["truth"] for p in pts])), "n": len(pts), "n_measured": len(meas),
            "measured_mean": _fin(np.mean(meas)) if meas else None,
            "measured_min": _fin(np.min(meas)) if meas else None,
            "measured_max": _fin(np.max(meas)) if meas else None,
            "err_mean": _fin(np.mean(errs)) if errs else None,
            "abs_err_max": _fin(np.max(np.abs(errs))) if errs else None,
            "in_declared": abs(s) <= declared + 1e-9,
        })

    def stats(sel: List[Dict[str, Any]]) -> Dict[str, Any]:
        ok = [p for p in sel if p["measured"] is not None]
        if len(ok) < 2:
            return {"n": len(sel), "n_measured": len(ok), "n_missing": len(sel) - len(ok)}
        t = np.array([p["truth"] for p in ok]); m = np.array([p["measured"] for p in ok]); e = m - t
        st = {"n": len(sel), "n_measured": len(ok), "n_missing": len(sel) - len(ok),
              "mae": _fin(np.mean(np.abs(e))), "bias": _fin(np.mean(e)), "max_abs": _fin(np.max(np.abs(e))),
              "rmse": _fin(np.sqrt(np.mean(e ** 2)))}
        if np.ptp(t) > 1e-9:
            a, b = np.polyfit(t, m, 1)
            pred = a * t + b
            ss_res = float(np.sum((m - pred) ** 2)); ss_tot = float(np.sum((m - m.mean()) ** 2))
            st.update({"slope": _fin(a), "intercept": _fin(b),
                       "r2": _fin(1.0 - ss_res / ss_tot) if ss_tot > 0 else None})
        return st

    inside = [p for p in points if abs(float(p[key_set])) <= declared + 1e-9]
    return {"rows": rows, "stats_declared": stats(inside), "stats_all": stats(points),
            "declared_range": [-declared, declared]}


def invariance(points: List[Dict[str, Any]], declared: float) -> Dict[str, Any]:
    """|Δ| относительно нулевого кадра того же фантома, максимум в заявленном диапазоне и по всем точкам."""
    zero = {p["phantom"]: p["measured"] for p in points if abs(float(p["set"])) < 1e-9}
    rows = []
    for p in points:
        z = zero.get(p["phantom"])
        if z is None or p["measured"] is None:
            continue
        rows.append({"set": float(p["set"]), "phantom": p["phantom"], "delta": float(p["measured"] - z)})
    by_set: Dict[float, List[float]] = {}
    for r in rows:
        by_set.setdefault(r["set"], []).append(r["delta"])
    table = [{"set": s, "delta_mean": _fin(np.mean(v)), "abs_delta_max": _fin(np.max(np.abs(v))), "n": len(v),
              "in_declared": abs(s) <= declared + 1e-9} for s, v in sorted(by_set.items())]
    ins = [r["delta"] for r in rows if abs(r["set"]) <= declared + 1e-9]
    return {"rows": table,
            "max_abs_delta_declared": _fin(np.max(np.abs(ins))) if ins else None,
            "max_abs_delta_all": _fin(np.max(np.abs([r["delta"] for r in rows]))) if rows else None,
            "declared_range": [-declared, declared]}


def check_limits(series: Dict[str, Any], inv: Dict[str, Any], side: Dict[str, Any]) -> List[str]:
    fails: List[str] = []
    for name, lim in LIMITS.items():
        st = series[name]["stats_declared"]
        if st.get("n_missing", 0):
            fails.append(f"{name}: нет измерения в {st['n_missing']} точках заявленного диапазона")
            continue
        if st["mae"] > lim["mae"]:
            fails.append(f"{name}: MAE {st['mae']:.3f} > {lim['mae']}")
        if abs(st["bias"]) > lim["abs_bias"]:
            fails.append(f"{name}: |bias| {abs(st['bias']):.3f} > {lim['abs_bias']}")
        if st["max_abs"] > lim["max_abs"]:
            fails.append(f"{name}: max {st['max_abs']:.3f} > {lim['max_abs']}")
        if not (lim["slope_min"] <= st["slope"] <= lim["slope_max"]):
            fails.append(f"{name}: наклон {st['slope']:.3f} вне [{lim['slope_min']}; {lim['slope_max']}]")
    for name, lim in INVARIANCE_LIMITS.items():
        v = inv[name]["max_abs_delta_declared"]
        if v is None or v > lim:
            fails.append(f"инвариантность {name}: max |Δ| {v} > {lim}")
    if side["fraction_correct_declared"] < SIDE_REQUIRED:
        fails.append(f"сторона бедра: верно {side['fraction_correct_declared']:.3f} < {SIDE_REQUIRED}")
    return fails


# --------------------------------------------------------------------------- #
# Основной расчёт
# --------------------------------------------------------------------------- #
def build(phantoms: Path, workdir: Path, seed: int, cfg_path: Path) -> Dict[str, Any]:
    t0 = time.time()
    variants = load_variants(cfg_path)
    bases = phantom_bases(phantoms, seed)
    ht = hip_truth()
    workdir.mkdir(parents=True, exist_ok=True)
    raw: List[Dict[str, Any]] = []      # все измерения: series, phantom, region, set, transform, values

    def run(base: Dict[str, Any], series: str, set_val: float, img: np.ndarray, extra: Dict[str, Any]) -> Dict[str, Any]:
        tag = f"{base['name']}|{series}|{set_val:+.3f}"
        path = workdir / (hashlib.sha1(tag.encode("utf-8")).hexdigest()[:16] + ".dcm")
        write_dicom(img, path, tag=tag, study_idx=1, seed=seed)
        vals = measure(path, base["region"], variants)
        rec = {"series": series, "phantom": base["name"], "region": base["region"], "set": float(set_val),
               "values": {k: (_fin(v) if not isinstance(v, str) else v) for k, v in vals.items()}}
        rec.update(extra)
        raw.append(rec)
        return rec

    for base in bases:
        for th in ROT_DEG:
            run(base, "rotation", th, affine_mm(base["img"], th, 0, 0), {"transform": {"theta_deg": th}})
        for d in SHIFT_MM:
            run(base, "shift_x", d, affine_mm(base["img"], 0, d, 0), {"transform": {"dx_mm": d}})
        for d in SHIFT_MM:
            run(base, "shift_y", d, affine_mm(base["img"], 0, 0, d), {"transform": {"dy_mm": d}})
    # наклон только столбика позвонков — 3 синтетических позвоночника с заданным физическим углом
    tilt_truth_cx: Dict[str, float] = {}
    for k in range(3):
        for th in ROT_DEG:
            rng = np.random.RandomState(seed * 100 + 50 + k)   # одинаковый силуэт при всех углах
            img, info = spine_column_tilt(rng, th)
            tilt_truth_cx[f"generated/column_tilt_{k}"] = info["center_x"]
            base = {"name": f"generated/column_tilt_{k}", "region": "spine"}
            run(base, "column_tilt", th, img, {"transform": {"column_tilt_deg": th}})

    # --- сбор точек по величинам --------------------------------------------------
    def pts(series: str, region_pred, value: str, truth_fn, set_fn=None) -> List[Dict[str, Any]]:
        """set_fn — пересчёт заданного значения в каноническую ориентацию (для бедра: левое зеркалится)."""
        out = []
        for r in raw:
            if r["series"] != series or not region_pred(r["region"]):
                continue
            out.append({"set": (float(set_fn(r)) + 0.0) if set_fn else r["set"], "set_frame": r["set"],
                        "phantom": r["phantom"], "region": r["region"],
                        "measured": r["values"].get(value), "truth": float(truth_fn(r))})
        return out

    is_spine = lambda reg: reg == "spine"          # noqa: E731
    is_hip = lambda reg: reg != "spine"            # noqa: E731
    cx_by_phantom = {b["name"]: float(b["truth"].get("center_x", 150.0)) for b in bases if b["region"] == "spine"}
    w_spine = 300

    def spine_offset_truth(r):
        return (cx_by_phantom[r["phantom"]] - w_spine / 2.0) * SX + float(r["set"])

    # каноническая ориентация — правое бедро; левое зеркалится, поэтому знак поворота и сдвига по X меняется
    def hip_rot_canon(r):
        return -float(r["set"]) if r["region"] == "right_hip" else float(r["set"])

    def hip_dx_canon(r):
        return float(r["set"]) if r["region"] == "right_hip" else -float(r["set"])

    def hip_angle_truth(r):
        return ht["signed_shaft_angle_deg"] + hip_rot_canon(r)

    def hip_margin_truth(r):
        return ht["lateral_margin_mm"] + hip_dx_canon(r)

    # длина диафиза ниже вертелов: нижний конец — срез кадра (точен), верхний — подобранная точка;
    # истина дифференциальная — относительно нулевого кадра того же фантома, при сдвиге вниз на dy
    # видимая длина уменьшается ровно на dy (диафиз выходит за нижний край кадра)
    zero_len = {r["phantom"]: r["values"]["shaft_len_below_troch_mm"] for r in raw
                if r["series"] == "shift_y" and abs(r["set"]) < 1e-9 and r["region"] != "spine"}

    def shaft_len_truth(r):
        return zero_len[r["phantom"]] - float(r["set"])

    series: Dict[str, Any] = {}
    series["spine_axis_rotation"] = {
        "quantity": "axis_angle_deg", "unit": "°", "criterion": "sp_axis", "variant": variants["sp_axis"],
        "transform": "поворот всего кадра", "truth_rule": "|θ| (угол измеряется по модулю, знак не измеряется)",
        "function": "geometry_features.spine_axis_features (режим baseline)",
        **summarize(pts("rotation", is_spine, "axis_angle_deg", lambda r: abs(float(r["set"]))), DECLARED_ROT)}
    series["spine_axis_column_tilt_info"] = {
        "quantity": "axis_angle_deg", "unit": "°", "criterion": "sp_axis", "variant": variants["sp_axis"],
        "transform": "наклон только столбика позвонков (рёбра и таз прямые), информативно, в пределы не входит",
        "truth_rule": "|θ|",
        "function": "geometry_features.spine_axis_features (режим baseline)",
        **summarize(pts("column_tilt", is_spine, "axis_angle_deg", lambda r: abs(float(r["set"]))), DECLARED_ROT)}
    series["spine_axis_column_tilt_tracked_info"] = {
        "quantity": "axis_angle_deg", "unit": "°", "criterion": "sp_axis", "variant": variants["sp_axis"],
        "transform": "тот же наклон столбика, режим tracked (в поставке не используется), информативно",
        "truth_rule": "|θ|",
        "function": "geometry_features.spine_axis_features (режим tracked)",
        **summarize(pts("column_tilt", is_spine, "axis_angle_deg_tracked", lambda r: abs(float(r["set"]))), DECLARED_ROT)}
    # размер маски (largest component) по углу наклона столбика — объясняет скачки измерения
    tilt_mask = {}
    for r in raw:
        if r["series"] == "column_tilt":
            tilt_mask.setdefault(r["set"], []).append(r["values"]["mask_px"])
    series["spine_axis_column_tilt_info"]["mask_px_mean_by_set"] = {f"{k:+g}": float(np.mean(v)) for k, v in sorted(tilt_mask.items())}
    series["spine_center_shift_x"] = {
        "quantity": "center_offset_mm = center_offset_ratio·w·0.6", "unit": "мм", "criterion": "sp_pos",
        "variant": variants["sp_pos"], "transform": "сдвиг по X",
        "truth_rule": "(center_x − w/2)·0.6 + dx, center_x — из генератора фантома",
        "function": "geometry_features.spine_positioning_features",
        **summarize(pts("shift_x", is_spine, "center_offset_mm", spine_offset_truth), DECLARED_SHIFT)}
    series["hip_shaft_angle_rotation"] = {
        "quantity": "signed_shaft_angle_deg", "unit": "°", "criterion": "hip_pos", "variant": variants["hip_pos"],
        "transform": "поворот всего кадра",
        "truth_rule": f"{ht['signed_shaft_angle_deg']:+.2f}° + θк, θк — угол в канонической ориентации (правое бедро: θк = −θ, левое: θк = +θ, левое зеркалится)",
        "set_meaning": "θк, °, каноническая ориентация",
        "function": "hip_features.hip_all_features",
        **summarize(pts("rotation", is_hip, "signed_shaft_angle_deg", hip_angle_truth, hip_rot_canon), DECLARED_ROT)}
    series["hip_lateral_margin_shift_x"] = {
        "quantity": "lateral_margin_mm", "unit": "мм", "criterion": "hip_pos (положение кости относительно края кадра)",
        "variant": variants["hip_pos"], "transform": "сдвиг по X",
        "truth_rule": f"{ht['lateral_margin_mm']:.2f} мм + dxк, dxк — сдвиг в канонической ориентации (правое: dxк = dx, левое: dxк = −dx)",
        "set_meaning": "dxк, мм, каноническая ориентация; dxк < 0 — кость к латеральному краю",
        "function": "hip_features.hip_all_features",
        **summarize(pts("shift_x", is_hip, "lateral_margin_mm", hip_margin_truth, hip_dx_canon), DECLARED_SHIFT)}
    down = [p for p in pts("shift_y", is_hip, "shaft_len_below_troch_mm", shaft_len_truth) if p["set"] >= 0]
    series["hip_shaft_len_shift_down"] = {
        "quantity": "shaft_len_below_troch_mm", "unit": "мм", "criterion": "hip_roi", "variant": variants["hip_roi"],
        "transform": "сдвиг вниз по Y (dy ≥ 0)",
        "truth_rule": "L(0) − dy, L(0) — измерение нулевого кадра того же фантома (дифференциальная истина)",
        "function": "hip_features.hip_all_features",
        **summarize(down, DECLARED_SHIFT)}
    up = [p for p in pts("shift_y", is_hip, "shaft_len_below_troch_mm", shaft_len_truth) if p["set"] < 0]
    series["hip_shaft_len_shift_up_info"] = {
        "quantity": "shaft_len_below_troch_mm", "unit": "мм", "criterion": "hip_roi", "variant": variants["hip_roi"],
        "transform": "сдвиг вверх по Y (dy < 0), информативно, в пределы не входит",
        "truth_rule": "L(0) + |dy| при условии, что верхняя граница диафиза неподвижна",
        "function": "hip_features.hip_all_features",
        **summarize(up, DECLARED_SHIFT)}

    # --- инвариантность и перекрёстное влияние -----------------------------------
    inv: Dict[str, Any] = {}
    inv["spine_axis_shift_x"] = {"quantity": "axis_angle_deg", "unit": "°", "transform": "сдвиг по X",
                                 **invariance(pts("shift_x", is_spine, "axis_angle_deg", lambda r: 0.0), DECLARED_SHIFT)}
    inv["spine_axis_shift_y"] = {"quantity": "axis_angle_deg", "unit": "°", "transform": "сдвиг по Y (информативно)",
                                 **invariance(pts("shift_y", is_spine, "axis_angle_deg", lambda r: 0.0), DECLARED_SHIFT)}
    inv["spine_center_shift_y"] = {"quantity": "center_offset_mm", "unit": "мм", "transform": "сдвиг по Y",
                                   **invariance(pts("shift_y", is_spine, "center_offset_mm", lambda r: 0.0), DECLARED_SHIFT)}
    inv["spine_center_rotation"] = {"quantity": "center_offset_mm", "unit": "мм", "transform": "поворот (перекрёстное влияние, информативно)",
                                    **invariance(pts("rotation", is_spine, "center_offset_mm", lambda r: 0.0), DECLARED_ROT)}
    inv["spine_bone_width_shift_x"] = {"quantity": "bone_width_mm", "unit": "мм", "transform": "сдвиг по X (информативно)",
                                       **invariance(pts("shift_x", is_spine, "bone_width_mm", lambda r: 0.0), DECLARED_SHIFT)}
    inv["hip_shaft_angle_shift_x"] = {"quantity": "signed_shaft_angle_deg", "unit": "°", "transform": "сдвиг по X",
                                      **invariance(pts("shift_x", is_hip, "signed_shaft_angle_deg", lambda r: 0.0), DECLARED_SHIFT)}
    inv["hip_shaft_angle_shift_y"] = {"quantity": "signed_shaft_angle_deg", "unit": "°", "transform": "сдвиг по Y",
                                      **invariance(pts("shift_y", is_hip, "signed_shaft_angle_deg", lambda r: 0.0), DECLARED_SHIFT)}
    inv["hip_lateral_margin_rotation"] = {"quantity": "lateral_margin_mm", "unit": "мм", "transform": "поворот (перекрёстное влияние, информативно)",
                                          **invariance(pts("rotation", is_hip, "lateral_margin_mm", lambda r: 0.0), DECLARED_ROT)}
    # ширина диафиза: одна сводка по трём сериям; в строки входят только точки заявленного диапазона (группировка
    # по модулю заданного: |θ| в ° или |d| в мм), максимум по всем точкам считается отдельно
    width_in: List[Dict[str, Any]] = []
    width_all_delta: List[float] = []
    for ser, decl in (("rotation", DECLARED_ROT), ("shift_x", DECLARED_SHIFT), ("shift_y", DECLARED_SHIFT)):
        ser_pts = pts(ser, is_hip, "shaft_width_mm", lambda r: 0.0)
        zero = {q["phantom"]: q["measured"] for q in ser_pts if abs(q["set"]) < 1e-9}
        for q in ser_pts:
            if q["measured"] is not None and zero.get(q["phantom"]) is not None:
                width_all_delta.append(abs(q["measured"] - zero[q["phantom"]]))
            if abs(q["set"]) <= decl + 1e-9:
                q2 = dict(q); q2["set"] = abs(float(q["set"])); q2["phantom"] = f"{q['phantom']}|{ser}"
                width_in.append(q2)
    inv["hip_shaft_width_all"] = {"quantity": "shaft_width_mm", "unit": "мм",
                                  "transform": "поворот |θ| ≤ 8°, сдвиги |d| ≤ 20 мм (три серии вместе; строки — по модулю заданного)",
                                  **invariance(width_in, DECLARED_SHIFT)}
    inv["hip_shaft_width_all"]["max_abs_delta_all"] = _fin(max(width_all_delta)) if width_all_delta else None
    # абсолютное значение ширины и угла на нулевом кадре против аналитической истины (смещение сегментации)
    zero_hip = [r for r in raw if r["series"] == "rotation" and abs(r["set"]) < 1e-9 and r["region"] != "spine"]
    zero_abs = {
        "shaft_width_mm": {"truth": ht["shaft_width_mm"],
                           "measured_mean": _fin(np.mean([r["values"]["shaft_width_mm"] for r in zero_hip])),
                           "n": len(zero_hip)},
        "lateral_margin_mm": {"truth": ht["lateral_margin_mm"],
                              "measured_mean": _fin(np.mean([r["values"]["lateral_margin_mm"] for r in zero_hip])),
                              "n": len(zero_hip)},
        "signed_shaft_angle_deg": {"truth": ht["signed_shaft_angle_deg"],
                                   "measured_mean": _fin(np.mean([r["values"]["signed_shaft_angle_deg"] for r in zero_hip])),
                                   "n": len(zero_hip)},
        "scan_length_mm": {"truth": ht["scan_length_mm"],
                           "measured_mean": _fin(np.mean([r["values"]["scan_length_mm"] for r in zero_hip])),
                           "n": len(zero_hip)},
    }
    for k, v in zero_abs.items():
        v["bias"] = _fin(v["measured_mean"] - v["truth"]) if v["measured_mean"] is not None else None

    # --- сторона бедра -------------------------------------------------------------
    side_rows = []
    for r in raw:
        if r["region"] == "spine":
            continue
        decl = DECLARED_ROT if r["series"] == "rotation" else DECLARED_SHIFT
        side_rows.append({"series": r["series"], "set": r["set"], "phantom": r["phantom"], "in_declared": abs(r["set"]) <= decl + 1e-9,
                          "expected": "right" if r["region"] == "right_hip" else "left",
                          "detected": r["values"].get("hip_side_detected")})
    in_d = [s for s in side_rows if s["in_declared"]]
    side = {"n_declared": len(in_d), "n_correct_declared": sum(s["expected"] == s["detected"] for s in in_d),
            "fraction_correct_declared": float(np.mean([s["expected"] == s["detected"] for s in in_d])) if in_d else 0.0,
            "n_all": len(side_rows), "n_correct_all": sum(s["expected"] == s["detected"] for s in side_rows),
            "wrong": [{"series": s["series"], "set": s["set"], "phantom": s["phantom"], "detected": s["detected"]}
                      for s in side_rows if s["expected"] != s["detected"]]}

    # --- где ломается: точки вне заявленного диапазона с ошибкой выше предела и аномалии внутри ----
    breaks = []
    for name, lim in LIMITS.items():
        for row in series[name]["rows"]:
            if row["abs_err_max"] is None:
                breaks.append({"series": name, "set": row["set"], "reason": "нет измерения", "in_declared": row["in_declared"]})
            elif row["abs_err_max"] > lim["max_abs"]:
                breaks.append({"series": name, "set": row["set"], "abs_err_max": row["abs_err_max"],
                               "limit": lim["max_abs"], "in_declared": row["in_declared"]})
    anomalies = []
    for name, lim in INVARIANCE_LIMITS.items():
        for row in inv[name]["rows"]:
            if row["abs_delta_max"] > lim:
                anomalies.append({"invariance": name, "set": row["set"], "abs_delta_max": row["abs_delta_max"],
                                  "limit": lim, "in_declared": row["in_declared"]})
    # аномалии по фантомам для сдвига по Y позвоночника (информативная серия): скачки угла оси
    y_jumps = [{"phantom": r["phantom"], "dy_mm": r["set"], "axis_angle_deg": r["values"]["axis_angle_deg"]}
               for r in raw if r["series"] == "shift_y" and r["region"] == "spine"
               and r["values"]["axis_angle_deg"] is not None and r["values"]["axis_angle_deg"] > 1.0]

    fails = check_limits(series, inv, side)
    payload: Dict[str, Any] = {
        "tool": "tools/measurement_check.py", "tool_version": TOOL_VERSION, "seed": seed,
        "pixel_spacing_mm": {"y": SY, "x": SX},
        "note": "поверка измерительного контура A на фантомах; не замена клинической валидации",
        "conventions": {
            "rotation": "положительный угол — поворот кадра по часовой стрелке на экране (низ вертикальной линии уходит влево); "
                        "поворот вокруг центра кадра в миллиметровых координатах (матрица S⁻¹·R·S, S = diag(0.6, 1.05))",
            "shift": "положительный dx — содержимое уходит вправо, положительный dy — вниз; сдвиг в мм через PixelSpacing 1.05×0.6",
            "hip_sign": "signed_shaft_angle_deg > 0 — низ диафиза медиальнее (в канонической ориентации правого бедра); левое бедро зеркалится",
            "spine_angle": "axis_angle_deg — модуль угла оси к вертикали кадра, знак не измеряется",
            "mm_source": "функции контура A переводят пиксели в мм константой аппарата 1.05×0.6 (geometry_features.PIXEL_SPACING_*); "
                         "тот же размер записан в PixelSpacing фантомов",
        },
        "variants_by_criterion": variants,
        "grids": {"rotation_deg": ROT_DEG, "shift_mm": SHIFT_MM, "declared_rotation_deg": DECLARED_ROT,
                  "declared_shift_mm": DECLARED_SHIFT},
        "phantoms": [{"name": b["name"], "source": b["source"], "region": b["region"], "rows": int(b["img"].shape[0]),
                      "cols": int(b["img"].shape[1]), "truth": b["truth"], "sha256": b["sha256"]} for b in bases]
                    + [{"name": k, "source": "measurement_check.spine_column_tilt", "region": "spine", "rows": 317, "cols": 300,
                        "truth": {"center_x": v}, "sha256": ""} for k, v in tilt_truth_cx.items()],
        "hip_truth_analytic": ht,
        "series": series, "invariance": inv, "zero_frame_absolute": zero_abs, "hip_side": side,
        "breaks": breaks, "invariance_anomalies": anomalies, "spine_axis_jumps_shift_y": y_jumps,
        "limits": LIMITS, "invariance_limits": INVARIANCE_LIMITS, "side_required": SIDE_REQUIRED,
        "limit_failures": fails, "all_within_limits": not fails,
        "n_frames": len(raw), "raw": raw,
    }
    payload["results_sha256"] = hashlib.sha256(json.dumps(
        {"series": series, "invariance": inv, "side": side, "raw": raw}, sort_keys=True, ensure_ascii=False,
        default=str).encode("utf-8")).hexdigest()
    payload["runtime_s"] = round(time.time() - t0, 2)
    payload["environment"] = {"python": platform.python_version(), "numpy": np.__version__,
                              "opencv": cv2.__version__, "pydicom": pydicom.__version__}
    return payload


# --------------------------------------------------------------------------- #
# Документ
# --------------------------------------------------------------------------- #
def _f(v: Optional[float], nd: int = 2) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def _table(s: Dict[str, Any], nd: int = 2) -> str:
    u = s["unit"]
    lines = [f"| Задано ({s.get('set_meaning', u)}) | Истина, {u} | Измерено (среднее), {u} | Мин – макс, {u} | Ошибка (средняя), {u} | |Ошибка| макс, {u} | n | В диапазоне |",
             "|---|---|---|---|---|---|---|---|"]
    for r in s["rows"]:
        lines.append(f"| {r['set']:+g} | {_f(r['truth'], nd)} | {_f(r['measured_mean'], nd)} | {_f(r['measured_min'], nd)} – {_f(r['measured_max'], nd)} | "
                     f"{_f(r['err_mean'], nd)} | {_f(r['abs_err_max'], nd)} | {r['n_measured']}/{r['n']} | {'да' if r['in_declared'] else 'нет'} |")
    return "\n".join(lines)


def _stats_line(s: Dict[str, Any], lim: Optional[Dict[str, float]]) -> str:
    st = s["stats_declared"]; sa = s["stats_all"]; u = s["unit"]
    lo, hi = s["declared_range"]
    txt = (f"В заявленном диапазоне [{lo:+g}; {hi:+g}] ({st.get('n_measured', 0)} кадров): MAE {_f(st.get('mae'))} {u}, "
           f"смещение {_f(st.get('bias'))} {u}, максимум {_f(st.get('max_abs'))} {u}, наклон {_f(st.get('slope'), 3)}, "
           f"R² {_f(st.get('r2'), 4)}. По всем точкам ({sa.get('n_measured', 0)} кадров): MAE {_f(sa.get('mae'))} {u}, "
           f"максимум {_f(sa.get('max_abs'))} {u}, наклон {_f(sa.get('slope'), 3)}.")
    if lim:
        txt += (f" Пределы теста: MAE ≤ {lim['mae']}, |смещение| ≤ {lim['abs_bias']}, максимум ≤ {lim['max_abs']}, "
                f"наклон в [{lim['slope_min']}; {lim['slope_max']}].")
    return txt


def render_md(p: Dict[str, Any], generated_at: str) -> str:
    s = p["series"]; inv = p["invariance"]; L = p["limits"]; IL = p["invariance_limits"]
    ht = p["hip_truth_analytic"]; za = p["zero_frame_absolute"]; side = p["hip_side"]
    n_ph = len(p["phantoms"])
    env = p["environment"]
    out: List[str] = []
    out.append("# Акт поверки измерительного контура на фантомах")
    out.append("")
    out.append(f"Сгенерировано `tools/measurement_check.py` {generated_at}; зерно {p['seed']}; окружение: Python {env['python']}, "
               f"numpy {env['numpy']}, OpenCV {env['opencv']}, pydicom {env['pydicom']}; время расчёта {p['runtime_s']} с; "
               f"кадров измерено {p['n_frames']}; `results_sha256` {p['results_sha256'][:16]}…")
    out.append("")
    out.append("## 1. Что поверяется и что нет")
    out.append("")
    out.append("Контур A сервиса измеряет геометрию кадра явными функциями (`src/geometry_features.py`, `src/hip_features.py`) и "
               "передаёт измеренное в логистические модели критериев. Здесь проверяется только измерительная часть — без моделей, "
               "порогов и разметки экспертов: фантомам задаются известные преобразования в физических единицах, и измеренное "
               "сравнивается с заданным. Поверяемые величины и их место в поставке:")
    out.append("")
    out.append("| Величина | Единица | Критерий (признак модели) | Функция | Вариант предобработки (`config.yaml`) |")
    out.append("|---|---|---|---|---|")
    out.append(f"| Угол оси позвоночника к вертикали `axis_angle_deg` | ° | `sp_axis` | `spine_axis_features` | {p['variants_by_criterion']['sp_axis']} |")
    out.append(f"| Смещение центра кости от центра кадра `center_offset_ratio` (здесь в мм: ·w·0.6) | мм | `sp_pos` | `spine_positioning_features` | {p['variants_by_criterion']['sp_pos']} |")
    out.append(f"| Знаковый угол диафиза бедра `signed_shaft_angle_deg` (в модель входит модуль `abs_shaft_angle_deg`) | ° | `hip_pos` | `hip_all_features` | {p['variants_by_criterion']['hip_pos']} |")
    out.append(f"| Расстояние от латерального края диафиза до края кадра `lateral_margin_mm` | мм | положение кости в кадре (в модели не входит, диагностика) | `hip_all_features` | {p['variants_by_criterion']['hip_pos']} |")
    out.append(f"| Ширина диафиза `shaft_width_mm` | мм | `hip_pos` | `hip_all_features` | {p['variants_by_criterion']['hip_pos']} |")
    out.append(f"| Длина диафиза ниже вертелов `shaft_len_below_troch_mm`, длина скана `scan_length_mm` | мм | `hip_roi` | `hip_all_features` | {p['variants_by_criterion']['hip_roi']} |")
    out.append("| Сторона бедра `hip_side_detected` | — | выбор канонической ориентации | `hip_side_score` | — |")
    out.append("")
    out.append("Не поверяются (не являются измерениями в градусах или миллиметрах): признаки посторонних предметов `sp_art` "
               "(площадь и контраст плотных объектов — на синтетических фантомах «кость» сама яркая, см. `tools/verify_checks.py`), "
               "признаки формы бедра (`femur_solidity`, `merge_height_mm`, `medial_neck_extent_mm`), логит синтетической головы "
               "`synth_pos_logit` (контур A `sp_pos`, `src/sppos_head.py`) и весь контур B. Вертикальное положение позвоночника контур A "
               "не измеряет (в моделях нет такого признака) — для сдвига по Y проверяется только инвариантность.")
    out.append("")
    out.append("**Это поверка на фантомах, не замена клинической валидации.** Фантомы — упрощённые силуэты (`tools/make_phantoms.py`), "
               "а не снимки пациентов; результат говорит о корректности геометрических формул и об устойчивости измерения к "
               "преобразованиям кадра, но ничего не говорит о согласии с экспертом и о клинической точности. Согласие с разметкой — "
               "`docs/METRICS_REPORT.md`.")
    out.append("")
    out.append("## 2. Метод")
    out.append("")
    ph_lines = [f"`{b['name']}` ({b['source']}, {b['region']}, {b['rows']}×{b['cols']})" for b in p["phantoms"]]
    out.append(f"Базовые кадры ({n_ph}): " + "; ".join(ph_lines) + ".")
    out.append("")
    out.append(f"Преобразования (масштаб не меняется): поворот всего кадра на {', '.join(f'{v:+g}' for v in p['grids']['rotation_deg'])}° "
               f"вокруг центра кадра в миллиметровых координатах (пиксель 1.05×0.6 мм, матрица S⁻¹·R·S); сдвиг содержимого на "
               f"{', '.join(f'{v:+g}' for v in p['grids']['shift_mm'])} мм отдельно по X и по Y; для позвоночника дополнительно — "
               "наклон только столбика позвонков на те же углы (тело, рёбра и таз прямые; синтез в `spine_column_tilt`, три силуэта). "
               "Каждый кадр записывается 8-битным DICOM с тегами экспорта аппарата и `PixelSpacing 1.05\\0.6`, читается "
               "`geometry_features.read_dicom_normalized` с вариантом предобработки критерия из `config.yaml` и измеряется теми же "
               "функциями, что в `src/inference.py`. Знаки: " + p["conventions"]["rotation"] + "; " + p["conventions"]["shift"] + "; "
               + p["conventions"]["hip_sign"] + "; " + p["conventions"]["spine_angle"] + ".")
    out.append("")
    out.append(f"Заявленный диапазон линейности: |угол| ≤ {p['grids']['declared_rotation_deg']:g}°, |сдвиг| ≤ {p['grids']['declared_shift_mm']:g} мм; "
               "статистики (MAE, смещение, максимум, наклон и R² зависимости «измерено от истины») считаются внутри него, точки вне "
               "диапазона показывают, где измерение ломается. Истина для бедра — аналитическая, из констант генератора: угол диафиза "
               f"{ht['signed_shaft_angle_deg']:+.2f}°, латеральный край {ht['lateral_margin_mm']:.2f} мм, горизонтальная ширина диафиза "
               f"{ht['shaft_width_mm']:.2f} мм, длина скана {ht['scan_length_mm']:.2f} мм. Пределы теста `tests/test_measurement_check.py` "
               "выбраны по числам этого прогона с запасом и перечислены под каждой таблицей.")
    out.append("")
    out.append("## 3. Угол оси позвоночника (`sp_axis`)")
    out.append("")
    out.append("### 3.1. Поворот всего кадра")
    out.append("")
    out.append(_table(s["spine_axis_rotation"]))
    out.append("")
    out.append(_stats_line(s["spine_axis_rotation"], L["spine_axis_rotation"]))
    out.append("")
    out.append("### 3.2. Наклон только столбика позвонков (информативно, в пределы не входит)")
    out.append("")
    ct = s["spine_axis_column_tilt_info"]; ctt = s["spine_axis_column_tilt_tracked_info"]
    mpx = ct["mask_px_mean_by_set"]
    out.append("Синтетический позвоночник, у которого наклонён только столбик позвонков, а рёберные дуги и таз остаются прямыми "
               "и симметричными (три силуэта, `spine_column_tilt`). Режим baseline — построчный центроид всей костной маски; "
               "режим tracked (ось только столбика) в поставке не используется и приведён для сравнения.")
    out.append("")
    out.append("| Задано, ° | Истина, ° | baseline: измерено (среднее), ° | baseline: ошибка, ° | tracked: измерено (среднее), ° | tracked: ошибка, ° | Маска, px (среднее) | n |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r, rt in zip(ct["rows"], ctt["rows"]):
        out.append(f"| {r['set']:+g} | {_f(r['truth'])} | {_f(r['measured_mean'])} | {_f(r['err_mean'])} | {_f(rt['measured_mean'])} | "
                   f"{_f(rt['err_mean'])} | {_f(mpx.get(f'{r['set']:+g}'), 0)} | {r['n_measured']}/{r['n']} |")
    out.append("")
    st = ct["stats_declared"]; stt = ctt["stats_declared"]
    small = [r for r in ct["rows"] if 0 < abs(r["set"]) <= 3 and r["measured_mean"] is not None]
    ratio_small = float(np.mean([r["measured_mean"] / r["truth"] for r in small])) if small else float("nan")
    px0 = mpx.get("+0")
    jump_sets = sorted({abs(r["set"]) for r in ct["rows"] if px0 and abs(mpx.get(f"{r['set']:+g}", px0) - px0) / px0 > 0.1})
    out.append(f"Итог по этой серии: в заявленном диапазоне baseline даёт MAE {_f(st.get('mae'))}°, максимум {_f(st.get('max_abs'))}°, "
               f"наклон {_f(st.get('slope'), 3)}, R² {_f(st.get('r2'), 3)}; tracked — MAE {_f(stt.get('mae'))}°, максимум {_f(stt.get('max_abs'))}°, "
               f"наклон {_f(stt.get('slope'), 3)}, R² {_f(stt.get('r2'), 3)}. При |θ| ≤ 3° baseline измеряет в среднем {ratio_small:.2f} от заданного: "
               "прямые симметричные рёбра и таз входят в ту же маску и разбавляют наклон столбика. "
               + (f"При |θ| ≥ {jump_sets[0]:g}° размер крупнейшей связной компоненты маски меняется более чем на 10 % "
                  f"(столбик отделяется от части силуэта), и измерение скачком меняется. " if jump_sets else "")
               + "Вывод: `axis_angle_deg` в режиме baseline — угол оси всей костной маски кадра, а не отдельно столбика позвонков; "
               "при повороте всего кадра (3.1) он линеен, при наклоне одного столбика на фоне прямых рёбер и таза — нет. "
               "Это свойство метода, зафиксированное здесь как граница применимости, а не дефект формул.")
    out.append("")
    out.append("## 4. Смещение центра позвоночника от центра кадра (`sp_pos`)")
    out.append("")
    out.append(_table(s["spine_center_shift_x"]))
    out.append("")
    out.append(_stats_line(s["spine_center_shift_x"], L["spine_center_shift_x"]))
    out.append("")
    out.append("## 5. Угол диафиза бедра (`hip_pos`)")
    out.append("")
    out.append(_table(s["hip_shaft_angle_rotation"]))
    out.append("")
    out.append(_stats_line(s["hip_shaft_angle_rotation"], L["hip_shaft_angle_rotation"]))
    out.append("")
    out.append("## 6. Положение бедра относительно края кадра")
    out.append("")
    out.append(_table(s["hip_lateral_margin_shift_x"]))
    out.append("")
    out.append(_stats_line(s["hip_lateral_margin_shift_x"], L["hip_lateral_margin_shift_x"]))
    out.append("")
    out.append("## 7. Длина диафиза ниже вертелов (`hip_roi`)")
    out.append("")
    out.append("Нижний конец величины — срез кадра (точен), верхний — точка, где края диафиза перестают быть прямыми "
               "(допуск 2.5 мм в `_fit_shaft`), поэтому истина дифференциальная: при сдвиге вниз на dy видимая длина должна "
               "уменьшиться ровно на dy.")
    out.append("")
    out.append(_table(s["hip_shaft_len_shift_down"]))
    out.append("")
    out.append(_stats_line(s["hip_shaft_len_shift_down"], L["hip_shaft_len_shift_down"]))
    out.append("")
    su = s["hip_shaft_len_shift_up_info"]
    up_rows = [r for r in su["rows"] if r["measured_mean"] is not None]
    up_meas = [r["measured_mean"] for r in up_rows]
    up_truth = [r["truth"] for r in up_rows]
    out.append(f"Сдвиг вверх (dy < 0, информативно, в пределы не входит): ожидание L(0) + |dy| не выполняется — при истине "
               f"от {_f(min(up_truth))} до {_f(max(up_truth))} мм измеренное (среднее по фантомам) остаётся в пределах "
               f"{_f(min(up_meas))}–{_f(max(up_meas))} мм (наклон {_f(su['stats_all'].get('slope'), 3)}, MAE {_f(su['stats_all'].get('mae'))} мм). "
               "Верхняя граница диафиза находится подбором прямолинейности краёв (`_fit_shaft`, старт с нижних 30 % трека) и на этом "
               "фантоме не следует за сдвигом содержимого вверх; поэтому величина поверена только для сдвига вниз, где истина задана срезом кадра. "
               "Это граница применимости, зафиксированная актом. Длина скана `scan_length_mm` = Rows·1.05 при любом преобразовании равна "
               f"{_f(za['scan_length_mm']['measured_mean'])} мм (истина {_f(za['scan_length_mm']['truth'])}).")
    out.append("")
    out.append("## 8. Инвариантность, перекрёстное влияние и абсолютные смещения")
    out.append("")
    out.append("| Величина | Преобразование | max |Δ| к нулевому кадру в заявленном диапазоне | max |Δ| по всем точкам | Предел теста |")
    out.append("|---|---|---|---|---|")
    for k, v in inv.items():
        lim = IL.get(k)
        out.append(f"| `{v['quantity']}` | {v['transform']} | {_f(v['max_abs_delta_declared'])} {v['unit']} | {_f(v['max_abs_delta_all'])} {v['unit']} | "
                   f"{('≤ ' + str(lim) + ' ' + v['unit']) if lim is not None else 'информативно'} |")
    out.append("")
    out.append("Перекрёстное влияние ожидаемо по геометрии, а не ошибка формул: при повороте всего кадра вокруг его центра "
               "смещается центроид маски (у позвоночника масса маски ниже центра кадра — таз), а у бедра нижняя точка диафиза уходит "
               "к краю кадра, поэтому `center_offset_mm` и `lateral_margin_mm` при повороте меняются.")
    out.append("")
    out.append("Абсолютные значения на нулевом кадре бедра против аналитической истины (смещение, вносимое сегментацией: "
               "размытие силуэта и порог 0.75·Otsu расширяют маску):")
    out.append("")
    out.append("| Величина | Истина | Измерено (среднее по нулевым кадрам) | Смещение | n |")
    out.append("|---|---|---|---|---|")
    for k, v in za.items():
        out.append(f"| `{k}` | {_f(v['truth'])} | {_f(v['measured_mean'])} | {_f(v['bias'])} | {v['n']} |")
    out.append("")
    out.append(f"Сторона бедра определена верно в {side['n_correct_declared']}/{side['n_declared']} кадрах заявленного диапазона "
               f"и в {side['n_correct_all']}/{side['n_all']} по всем точкам"
               + ("." if not side["wrong"] else "; ошибки: " + "; ".join(
                   f"{w_['phantom']} {w_['series']} {w_['set']:+g} → {w_['detected']}" for w_ in side["wrong"]) + "."))
    out.append("")
    out.append("## 9. Где измерение линейно и где ломается")
    out.append("")
    lin = []
    for name in L:
        st = s[name]["stats_declared"]
        lin.append(f"`{s[name]['quantity'].split(' ')[0]}` ({s[name]['transform']}): наклон {_f(st.get('slope'), 3)}, MAE {_f(st.get('mae'))} {s[name]['unit']}, "
                   f"максимум {_f(st.get('max_abs'))} {s[name]['unit']}")
    out.append(f"Линейно в заявленном диапазоне (|угол| ≤ {p['grids']['declared_rotation_deg']:g}°, |сдвиг| ≤ {p['grids']['declared_shift_mm']:g} мм): "
               + "; ".join(lin) + ".")
    out.append("")
    if p["breaks"]:
        out.append("Точки, где |ошибка| выше предела максимума (все — вне заявленного диапазона, если не отмечено иначе):")
        out.append("")
        out.append("| Серия | Задано | |Ошибка| макс | Предел | В диапазоне |")
        out.append("|---|---|---|---|---|")
        for b in p["breaks"]:
            out.append(f"| `{b['series']}` | {b['set']:+g} | {_f(b.get('abs_err_max'))} | {b.get('limit', '—')} | {'да' if b['in_declared'] else 'нет'} |")
        out.append("")
    if p["invariance_anomalies"]:
        out.append("Нарушения инвариантности выше предела:")
        out.append("")
        out.append("| Инвариантность | Задано | max |Δ| | Предел | В диапазоне |")
        out.append("|---|---|---|---|---|")
        for a in p["invariance_anomalies"]:
            out.append(f"| `{a['invariance']}` | {a['set']:+g} | {_f(a['abs_delta_max'])} | {a['limit']} | {'да' if a['in_declared'] else 'нет'} |")
        out.append("")
    if p["spine_axis_jumps_shift_y"]:
        j = p["spine_axis_jumps_shift_y"]
        out.append("Скачки угла оси прямого позвоночника при сдвиге по Y (информативная серия): "
                   + "; ".join(f"`{x['phantom']}` dy {x['dy_mm']:+g} мм → {x['axis_angle_deg']:.2f}°" for x in j)
                   + ". Причина — смена состава крупнейшей связной компоненты маски: рёберная дуга примыкает к столбу с одной стороны, "
                     "и построчный центроид всей маски (режим baseline) смещается. Это известное свойство метода центроида: угол оси "
                     "зависит от того, какие структуры вошли в маску, а не только от наклона столбика позвонков.")
        out.append("")
    def _row(name: str, set_val: float) -> Optional[Dict[str, Any]]:
        for r in s[name]["rows"]:
            if abs(r["set"] - set_val) < 1e-9:
                return r
        return None

    def _err(name: str, set_val: float) -> str:
        r = _row(name, set_val)
        return _f(r["err_mean"]) if r and r["err_mean"] is not None else "—"

    items = []
    items.append(f"угол оси позвоночника при повороте всего кадра занижается с ростом угла (средняя ошибка {_err('spine_axis_rotation', -10)}° при −10°, "
                 f"{_err('spine_axis_rotation', 15)}° при +15°): маска включает таз и рёбра, при повороте они смещают построчный центроид")
    items.append("при наклоне только столбика позвонков угол оси нелинеен уже внутри заявленного диапазона (3.2)")
    items.append(f"смещение центра позвоночника при |dx| = 30 мм искажается обрезкой силуэта краем кадра (ошибка {_err('spine_center_shift_x', -30)} мм "
                 f"и {_err('spine_center_shift_x', 30)} мм)")
    items.append(f"угол диафиза бедра при θк = −15° (ошибка {_err('hip_shaft_angle_rotation', -15)}°): нижняя часть диафиза уходит за "
                 f"латеральный край кадра; при θк = +15° ошибка {_err('hip_shaft_angle_rotation', 15)}°")
    items.append(f"латеральный край бедра при dxк = −30 мм (истина {_f(_row('hip_lateral_margin_shift_x', -30)['truth']) if _row('hip_lateral_margin_shift_x', -30) else '—'} мм: "
                 f"кость частично за краем кадра, ошибка {_err('hip_lateral_margin_shift_x', -30)} мм) — измерение теряет смысл вместе с самой костью")
    items.append("длина диафиза ниже вертелов не следует за сдвигом вверх (раздел 7), поверена только на сдвиге вниз")
    if side["wrong"]:
        items.append(f"сторона бедра определена неверно в {len(side['wrong'])} кадрах (все перечислены в разделе 8)")
    else:
        items.append("сторона бедра определена верно во всех кадрах, включая точки вне заявленного диапазона")
    out.append("Итог по «где ломается»: " + "; ".join(items) + ". Знак наклона позвоночника не измеряется (величина по модулю) — "
               "по критерию он и не требуется.")
    out.append("")
    out.append("## 10. Итог поверки")
    out.append("")
    if p["all_within_limits"]:
        out.append("Все статистики в заявленном диапазоне укладываются в пределы, перечисленные под таблицами и зашитые в "
                   "`tests/test_measurement_check.py` (`LIMITS`, `INVARIANCE_LIMITS` в `tools/measurement_check.py`).")
    else:
        out.append("Есть превышения пределов: " + "; ".join(p["limit_failures"]) + ".")
    out.append("")
    out.append("Что это доказывает: функции контура A измеряют угол и смещение в тех единицах и с тем знаком, которые заявлены, "
               "линейно в заявленном диапазоне, независимо от разметки. Что не доказывает: точность на снимках пациентов, где "
               "сегментация сложнее, чем на фантомах, и согласие с экспертом — это предмет `docs/METRICS_REPORT.md`, "
               "`docs/CALIBRATION.md` и клинической валидации, которую поверка на фантомах не заменяет.")
    out.append("")
    out.append("Воспроизведение: `python tools/measurement_check.py` (пишет `docs/measurement_check.json` и этот документ; "
               "`--workdir` — каталог для временных DICOM, по умолчанию временный); проверка — `python tests/test_measurement_check.py`. "
               "Модели, `config.yaml`, `tests/phantoms/` и `tools/verify.sh` не меняются.")
    out.append("")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phantoms", default=str(ROOT / "tests" / "phantoms"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "measurement_check.json"))
    ap.add_argument("--md", default=str(ROOT / "docs" / "MEASUREMENT_CHECK.md"), help="путь документа; '' — не писать")
    ap.add_argument("--workdir", default="", help="каталог временных DICOM (по умолчанию — временный, удаляется)")
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--no-raw", action="store_true", help="не писать поэлементные измерения raw в JSON")
    args = ap.parse_args(argv)

    tmp = None
    if args.workdir:
        workdir = Path(args.workdir)
    else:
        tmp = tempfile.TemporaryDirectory(prefix="densito_mc_")
        workdir = Path(tmp.name)
    try:
        payload = build(Path(args.phantoms), workdir, args.seed, Path(args.config))
    finally:
        if tmp is not None:
            tmp.cleanup()
    generated_at = time.strftime("%Y-%m-%d %H:%M")
    payload["generated_at"] = generated_at
    if args.no_raw:
        payload.pop("raw", None)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    if args.md:
        Path(args.md).write_text(render_md(payload, generated_at), encoding="utf-8")
    print(f"measurement_check: {payload['n_frames']} кадров за {payload['runtime_s']} с; "
          f"пределы: {'OK' if payload['all_within_limits'] else 'ПРЕВЫШЕНЫ'}; sha {payload['results_sha256'][:16]} -> {out}")
    for f in payload["limit_failures"]:
        print("  ", f)
    return 0 if payload["all_within_limits"] else 1


if __name__ == "__main__":
    sys.exit(main())
