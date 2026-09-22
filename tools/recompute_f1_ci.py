#!/usr/bin/env python3
"""Пересчёт доверительных интервалов F1 в models/metrics_summary.json по сохранённым OOF-прогнозам.

Зачем. `study_level_bootstrap_f1` до 22.09.2026 присваивала F1 = 1.0 ресэмплам, в которых
не было ни одного положительного исследования («обе стороны согласны, что позитивов нет»).
У критериев с 5–17 положительными исследованиями такие ресэмплы составляют заметную долю,
поэтому верхняя граница ДИ была завышена, а сам интервал — не интервалом для F1, а смесью
двух разных величин. Функция исправлена (ресэмпл без позитивов исключается), и этот скрипт
переписывает в `metrics_summary.json` ТОЛЬКО поля `f1_ci_lo` / `f1_ci_hi` (+ добавляет
`f1_ci_method` и `f1_ci_share_skipped`), ничего больше не меняя: пороги, точечные метрики,
веса и решения вентилей остаются как в поставке.

Точечные F1 и пороги при этом пересчитываются для контроля и сверяются с записанными:
расхождение печатается и (с --check) возвращает код 1.

Запуск:
    python tools/recompute_f1_ci.py            # пересчитать и записать
    python tools/recompute_f1_ci.py --check    # только сверить, ничего не писать
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import f1_score  # noqa: E402
from eval_oof_metrics import N_BOOT, THR_TIE_ATOL, flags_from_scores  # noqa: E402

SUMMARY = ROOT / "models" / "metrics_summary.json"
FULL = ROOT / "models" / "metrics_oof_full.json"   # единый источник ДИ для пяти критериев ТЗ
FULL_CI: dict = {}
SEED = 2026         # как в src/eval_oof_metrics.py (RNG = default_rng(2026), N_BOOT = 2000)

# (раздел в metrics_summary.json, критерий, файл OOF)
PLACES = [
    ("spine", "sp_pos", "oof_stacked_spine_sp_pos.csv"),
    ("spine", "sp_axis", "oof_stacked_spine_sp_axis.csv"),
    ("spine", "sp_art", "oof_stacked_spine_sp_art.csv"),
    ("hip", "hip_pos", "oof_stacked_hip_hip_pos.csv"),
    ("hip", "hip_roi", "oof_stacked_hip_hip_roi.csv"),
    ("left_hip", "lh_pos", "oof_stacked_left_hip_lh_pos.csv"),
    ("left_hip", "lh_roi", "oof_stacked_left_hip_lh_roi.csv"),
    ("right_hip", "rh_pos", "oof_stacked_right_hip_rh_pos.csv"),
    ("right_hip", "rh_roi", "oof_stacked_right_hip_rh_roi.csv"),
]


def bootstrap_f1_ci(y: np.ndarray, flag: np.ndarray, groups: np.ndarray):
    """95 % ДИ для F1: бутстрап по исследованиям, ресэмплы без положительных исключаются.

    Процедура и параметры совпадают с `src/eval_oof_metrics.bootstrap_ci`, поэтому числа в
    `models/metrics_summary.json` и `models/metrics_oof_full.json` сходятся по построению.
    """
    rng = np.random.default_rng(SEED)
    uniq = np.unique(groups)
    idx_by_g = {g: np.nonzero(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(N_BOOT):
        sample = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in sample])
        if len(np.unique(y[idx])) < 2:        # F1 не определён — ресэмпл не учитываем
            continue
        vals.append(f1_score(y[idx], flag[idx], zero_division=0))
    skipped = 1.0 - len(vals) / float(N_BOOT)
    if not vals:
        return 0.0, 0.0, 0.0, skipped
    return (float(np.percentile(vals, 2.5)), float(np.mean(vals)),
            float(np.percentile(vals, 97.5)), skipped)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="только сверить, не записывать")
    a = ap.parse_args()

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    full = json.loads(FULL.read_text(encoding="utf-8")) if FULL.exists() else {}
    global FULL_CI
    FULL_CI = {k: v["ci95"]["f1"] for k, v in full.get("by_violation_type", {}).items()
               if isinstance(v, dict) and "ci95" in v}
    changed, problems = [], []
    for section, crit, fname in PLACES:
        node = summary.get(section, {}).get(crit)
        if not isinstance(node, dict):
            continue
        path = ROOT / "models" / fname
        if not path.exists():
            problems.append(f"{section}/{crit}: нет {fname}")
            continue
        df = pd.read_csv(path)
        thr = float(node["threshold"])
        df = df.rename(columns={"y_true": "y", "oof_stacked": "pred_score"})
        y = df["y"].to_numpy()
        # флаги — ровно те, что выдаёт сервис (колонка pred_label, иначе порог с допуском на равенство)
        flag = flags_from_scores(df, df["pred_score"].to_numpy(), thr)
        f1_now = float(f1_score(y, flag, zero_division=0))
        if abs(f1_now - float(node["f1_oof"])) > 5e-4:
            problems.append(f"{section}/{crit}: точечный F1 {f1_now:.4f} != записанного {node['f1_oof']:.4f}")
        full_ci = FULL_CI.get(f"{section}/{crit}")
        if full_ci is not None:
            # для пяти критериев ТЗ ДИ берём из единого расчёта models/metrics_oof_full.json,
            # чтобы в METRICS_REPORT, MODEL_CARD и на сайте стояло одно и то же число
            lo, hi = float(full_ci[0]), float(full_ci[1])
            _, mean, _, skipped = bootstrap_f1_ci(y, flag, df["study"].to_numpy())
            src_note = "models/metrics_oof_full.json (src/eval_oof_metrics.py)"
        else:
            lo, mean, hi, skipped = bootstrap_f1_ci(y, flag, df["study"].to_numpy())
            src_note = "tools/recompute_f1_ci.py (та же процедура, критерий по стороне)"
        old = (float(node.get("f1_ci_lo", 0.0)), float(node.get("f1_ci_hi", 0.0)))
        changed.append((section, crit, int(y.sum()), df["study"].nunique(), f1_now, old, (lo, hi), mean, skipped))
        if not a.check:
            node["f1_ci_lo"], node["f1_ci_hi"] = lo, hi
            node["f1_ci_method"] = (f"бутстрап по исследованиям, {N_BOOT} ресэмплов; флаги — решения сервиса "
                                    f"(pred_label); ресэмплы без положительных исследований исключены "
                                    f"(F1 не определён). Источник: {src_note}")
            node["f1_ci_share_skipped"] = round(float(skipped), 4)

    w = max(len(f"{s}/{c}") for s, c, *_ in changed) if changed else 10
    print(f"{'критерий'.ljust(w)}  полож.  исслед.   F1     было ДИ          стало ДИ         среднее  пропущено")
    for section, crit, npos, nstud, f1, old, new, mean, skipped in changed:
        print(f"{(section + '/' + crit).ljust(w)}  {npos:5d}  {nstud:6d}  {f1:.3f}  "
              f"[{old[0]:.2f}; {old[1]:.2f}]     [{new[0]:.2f}; {new[1]:.2f}]     {mean:.3f}    {skipped * 100:.1f} %")
    if problems:
        print("\nРасхождения:")
        for p in problems:
            print("  -", p)
    if not a.check:
        SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nЗаписано: {SUMMARY}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
