#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_measurement_check.py — поверка измерительного контура A на фантомах (tools/measurement_check.py).

Вызывает инструмент с фиксированным зерном во временный каталог и проверяет:
  1. инструмент завершается кодом 0 и пишет валидный JSON с ожидаемыми ключами;
  2. all_within_limits истинно и limit_failures пуст;
  3. статистики каждой поверяемой серии в заявленном диапазоне (|угол| ≤ 8°, |сдвиг| ≤ 20 мм) не превышают
     пределов, зашитых здесь (копия LIMITS / INVARIANCE_LIMITS из инструмента; выбраны по фактическим числам
     прогона с запасом и совпадают с числами под таблицами docs/MEASUREMENT_CHECK.md);
  4. инвариантности не превышают своих пределов, сторона бедра определена верно во всех кадрах диапазона;
  5. все точки диапазона измерены (n_missing = 0), знак и единицы согласованы (наклон близок к 1);
  6. детерминизм: второй прогон даёт тот же results_sha256;
  7. время одного прогона не превышает бюджета (по умолчанию 300 с).

    python tests/test_measurement_check.py            # как скрипт: код 0 — ок, 1 — расхождения
    python -m pytest tests/test_measurement_check.py -q

Переменные окружения:
    DENSITO_ROOT             корень репозитория (по умолчанию — родитель каталога tests/)
    DENSITO_MC_BUDGET        лимит времени одного прогона в секундах, по умолчанию 300
    DENSITO_MC_SKIP_REPEAT   1 — не выполнять второй прогон (проверку детерминизма)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
TOOL = ROOT / "tools" / "measurement_check.py"
PHANTOMS = ROOT / "tests" / "phantoms"
BUDGET_S = float(os.environ.get("DENSITO_MC_BUDGET", "300"))
SEED = 20260923
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Копия пределов инструмента (tools/measurement_check.py: LIMITS, INVARIANCE_LIMITS, SIDE_REQUIRED).
LIMITS: Dict[str, Dict[str, float]] = {
    "spine_axis_rotation":      {"mae": 0.6, "abs_bias": 0.5, "max_abs": 1.5, "slope_min": 0.85, "slope_max": 1.05},
    "spine_center_shift_x":     {"mae": 0.5, "abs_bias": 0.5, "max_abs": 1.0, "slope_min": 0.97, "slope_max": 1.03},
    "hip_shaft_angle_rotation": {"mae": 0.3, "abs_bias": 0.3, "max_abs": 0.6, "slope_min": 0.97, "slope_max": 1.03},
    "hip_lateral_margin_shift_x": {"mae": 0.8, "abs_bias": 0.8, "max_abs": 1.5, "slope_min": 0.97, "slope_max": 1.03},
    "hip_shaft_len_shift_down": {"mae": 2.0, "abs_bias": 2.0, "max_abs": 3.5, "slope_min": 0.9, "slope_max": 1.1},
}
INVARIANCE_LIMITS: Dict[str, float] = {
    "spine_axis_shift_x": 0.6,
    "spine_center_shift_y": 2.5,
    "hip_shaft_angle_shift_x": 0.3,
    "hip_shaft_angle_shift_y": 0.3,
    "hip_shaft_width_all": 1.0,
}
SIDE_REQUIRED = 1.0
REQUIRED_KEYS = ["tool", "seed", "pixel_spacing_mm", "conventions", "variants_by_criterion", "grids", "phantoms",
                 "hip_truth_analytic", "series", "invariance", "zero_frame_absolute", "hip_side", "breaks",
                 "limits", "invariance_limits", "limit_failures", "all_within_limits", "n_frames", "results_sha256",
                 "runtime_s"]
INFO_SERIES = ["spine_axis_column_tilt_info", "spine_axis_column_tilt_tracked_info", "hip_shaft_len_shift_up_info"]


def run_tool(workdir: Path, tag: str) -> Dict[str, Any]:
    out_json = workdir / f"measurement_check_{tag}.json"
    cmd = [sys.executable, str(TOOL), "--phantoms", str(PHANTOMS), "--out", str(out_json), "--md", "",
           "--seed", str(SEED), "--no-raw"]
    env = dict(os.environ)
    env["DENSITO_ROOT"] = str(ROOT)
    env["OMP_NUM_THREADS"] = "1"
    t0 = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0
    if p.returncode != 0:
        raise AssertionError(f"measurement_check завершился кодом {p.returncode}:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    if "Traceback" in p.stderr:
        raise AssertionError("в stderr инструмента есть Traceback:\n" + p.stderr[-2000:])
    if not out_json.exists():
        raise AssertionError("инструмент не создал JSON")
    try:
        res = json.loads(out_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:  # noqa: PERF203
        raise AssertionError(f"JSON невалиден: {e}") from e
    res["_elapsed_s"] = elapsed
    return res


def check(res: Dict[str, Any]) -> List[str]:
    problems: List[str] = []
    for k in REQUIRED_KEYS:
        if k not in res:
            problems.append(f"в JSON нет ключа {k}")
    if problems:
        return problems
    if res["_elapsed_s"] > BUDGET_S:
        problems.append(f"время прогона {res['_elapsed_s']:.1f} с > {BUDGET_S:g} с")
    if not res["all_within_limits"] or res["limit_failures"]:
        problems.append("инструмент сообщает о превышении пределов: " + "; ".join(res["limit_failures"]))
    if res["limits"] != LIMITS:
        problems.append("LIMITS инструмента не совпадают с копией в тесте")
    if res["invariance_limits"] != INVARIANCE_LIMITS:
        problems.append("INVARIANCE_LIMITS инструмента не совпадают с копией в тесте")
    g = res["grids"]
    if g["declared_rotation_deg"] != 8.0 or g["declared_shift_mm"] != 20.0:
        problems.append(f"заявленный диапазон изменился: {g['declared_rotation_deg']}°, {g['declared_shift_mm']} мм")
    if res["pixel_spacing_mm"] != {"y": 1.05, "x": 0.6}:
        problems.append(f"PixelSpacing инструмента {res['pixel_spacing_mm']} != 1.05×0.6")

    series = res["series"]
    for name, lim in LIMITS.items():
        s = series.get(name)
        if s is None:
            problems.append(f"нет серии {name}")
            continue
        st = s["stats_declared"]
        if st.get("n_missing", 1) != 0:
            problems.append(f"{name}: не измерено {st.get('n_missing')} кадров в заявленном диапазоне")
        for key in ("mae", "bias", "max_abs", "slope"):
            if st.get(key) is None:
                problems.append(f"{name}: статистика {key} не вычислена")
        if None in (st.get("mae"), st.get("bias"), st.get("max_abs"), st.get("slope")):
            continue
        if st["mae"] > lim["mae"]:
            problems.append(f"{name}: MAE {st['mae']:.3f} > {lim['mae']}")
        if abs(st["bias"]) > lim["abs_bias"]:
            problems.append(f"{name}: |смещение| {abs(st['bias']):.3f} > {lim['abs_bias']}")
        if st["max_abs"] > lim["max_abs"]:
            problems.append(f"{name}: максимум {st['max_abs']:.3f} > {lim['max_abs']}")
        if not (lim["slope_min"] <= st["slope"] <= lim["slope_max"]):
            problems.append(f"{name}: наклон {st['slope']:.3f} вне [{lim['slope_min']}; {lim['slope_max']}]")
        rows_in = [r for r in s["rows"] if r["in_declared"]]
        if len(rows_in) < 5:
            problems.append(f"{name}: в заявленном диапазоне только {len(rows_in)} точек сетки")
    for name in INFO_SERIES:
        if name not in series:
            problems.append(f"нет информативной серии {name}")

    inv = res["invariance"]
    for name, lim in INVARIANCE_LIMITS.items():
        v = inv.get(name)
        if v is None:
            problems.append(f"нет инвариантности {name}")
            continue
        d = v.get("max_abs_delta_declared")
        if d is None:
            problems.append(f"{name}: max |Δ| не вычислен")
        elif d > lim:
            problems.append(f"инвариантность {name}: max |Δ| {d:.3f} > {lim}")

    side = res["hip_side"]
    if side["n_declared"] == 0 or side["fraction_correct_declared"] < SIDE_REQUIRED:
        problems.append(f"сторона бедра: верно {side['n_correct_declared']}/{side['n_declared']} < {SIDE_REQUIRED}")

    za = res["zero_frame_absolute"]
    if za["scan_length_mm"]["bias"] is None or abs(za["scan_length_mm"]["bias"]) > 1e-6:
        problems.append(f"scan_length_mm на нулевом кадре {za['scan_length_mm']}")
    if za["signed_shaft_angle_deg"]["bias"] is None or abs(za["signed_shaft_angle_deg"]["bias"]) > 0.3:
        problems.append(f"signed_shaft_angle_deg на нулевом кадре: смещение {za['signed_shaft_angle_deg']['bias']}")

    if res["n_frames"] < 400:
        problems.append(f"измерено кадров {res['n_frames']} < 400")
    return problems


def summary_line(res: Dict[str, Any]) -> str:
    parts = []
    for name in LIMITS:
        st = res["series"][name]["stats_declared"]
        u = res["series"][name]["unit"]
        parts.append(f"{name}: MAE {st['mae']:.2f} {u}, bias {st['bias']:+.2f} {u}, max {st['max_abs']:.2f} {u}, наклон {st['slope']:.3f}")
    return "\n".join("  " + p for p in parts)


def test_measurement_check() -> None:
    with tempfile.TemporaryDirectory(prefix="densito_mc_") as td:
        res = run_tool(Path(td), "a")
        problems = check(res)
        if os.environ.get("DENSITO_MC_SKIP_REPEAT") != "1":
            res2 = run_tool(Path(td), "b")
            if res2["results_sha256"] != res["results_sha256"]:
                problems.append("детерминизм: results_sha256 второго прогона отличается")
        assert not problems, "\n".join(problems)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="densito_mc_") as td:
        res = run_tool(Path(td), "a")
        problems = check(res)
        det = "пропущено"
        if os.environ.get("DENSITO_MC_SKIP_REPEAT") != "1":
            res2 = run_tool(Path(td), "b")
            same = res2["results_sha256"] == res["results_sha256"]
            det = "да" if same else "НЕТ"
            if not same:
                problems.append("детерминизм: results_sha256 второго прогона отличается")
        print(summary_line(res))
        side = res["hip_side"]
        print(f"  инвариантность: " + "; ".join(
            f"{k} {res['invariance'][k]['max_abs_delta_declared']:.2f} ≤ {v}" for k, v in INVARIANCE_LIMITS.items()))
        print(f"  сторона бедра {side['n_correct_declared']}/{side['n_declared']}; кадров {res['n_frames']}; "
              f"время {res['_elapsed_s']:.1f} с (бюджет {BUDGET_S:g}); детерминизм: {det}; sha {res['results_sha256'][:16]}")
        if problems:
            print("test_measurement_check: РАСХОЖДЕНИЯ")
            for p in problems:
                print("  -", p)
            return 1
        print("test_measurement_check: OK")
        return 0


if __name__ == "__main__":
    sys.exit(main())
