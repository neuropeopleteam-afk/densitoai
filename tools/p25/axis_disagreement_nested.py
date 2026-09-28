#!/usr/bin/env python3
"""2.5, пункт 3 («второе мнение по оси», консилиум 27.09, Kimi K3, 4.3): флаг «контуры разошлись — посмотрите ось»
при |p_geom - p_emb| > t для sp_axis. Проверка вложенная: порог t выбирается ТОЛЬКО на обучающей части внешнего
фолда, оценивается на тестовой. GroupKFold(5, shuffle=True, random_state=42+r), r = 0..19; группы — связные
компоненты графа «исследование — хэш пикселей» (как tools/emb_gate.py / tools/nested_gate.connected_groups).

Данные: models/oof_stacked_spine_sp_axis.csv (OOF контуров A и B и решения сервиса, 166 снимков, 17 ошибок).
Ошибка сервиса = pred_label != y_true.

Правило выбора порога на обучающей части (задано до расчёта): сетка t = 0.30, 0.35, ..., 0.80; берётся
наименьший t, при котором доля флагов среди ВЕРНЫХ решений обучающей части не больше 20 % (ограничение нагрузки
из 4.3, п. 6), — то есть наибольшее покрытие при этом ограничении.

Критерий включения в продукт (задан до расчёта; 4.3, п. 6, и постановка 2.5):
  (1) покрытие ошибок на тестовых частях повтора (сумма по 5 фолдам) не ниже 0.47 не менее чем в 70 % повторов;
  (2) средняя доля флагов среди всех снимков позвоночника не выше 25 %;
  (3) средняя доля флагов среди верных решений не выше 20 %.
Дополнительно: покрытие случайного флага с той же нагрузкой (= нагрузка) — для сравнения.

Запуск: python tools/p25/axis_disagreement_nested.py [--out docs/p25/axis_disagreement_nested.json]
Нужна outputs/p25/frames.csv (tools/p25/prep_frames.py) — хэши пикселей.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from nested_gate import connected_groups  # noqa: E402

GRID = np.round(np.arange(0.30, 0.8001, 0.05), 2)
LOAD_CAP_CORRECT = 0.20
N_REPEATS, N_OUTER = 20, 5
COVER_MIN, COVER_SHARE, LOAD_MAX = 0.47, 0.70, 0.25


def pick_threshold(d, err):
    for t in GRID:
        if np.mean(d[~err] > t) <= LOAD_CAP_CORRECT:
            return float(t)
    return float(GRID[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "p25" / "axis_disagreement_nested.json")
    a = ap.parse_args()
    o = pd.read_csv(ROOT / "models" / "oof_stacked_spine_sp_axis.csv")
    fr = pd.read_csv(ROOT / "outputs" / "p25" / "frames.csv")
    sha = dict(zip(fr.file_path, fr.pixel_sha1))
    groups = connected_groups(o.study.tolist(), [sha[f] for f in o.file_path])
    d = np.abs(o.oof_geom.values - o.oof_emb.values)
    err = (o.pred_label.values != o.y_true.values)
    reps = []
    for r in range(N_REPEATS):
        caught = n_err = flagged = n = flagged_ok = n_ok = 0
        ts = []
        for tr, te in GroupKFold(n_splits=N_OUTER, shuffle=True, random_state=42 + r).split(d, groups=groups):
            t = pick_threshold(d[tr], err[tr])
            ts.append(t)
            f = d[te] > t
            caught += int((f & err[te]).sum()); n_err += int(err[te].sum())
            flagged += int(f.sum()); n += len(te)
            flagged_ok += int((f & ~err[te]).sum()); n_ok += int((~err[te]).sum())
        reps.append({"repeat": r, "thresholds": ts, "caught": caught, "errors": n_err, "coverage": caught / n_err,
                     "load_all": flagged / n, "load_correct": flagged_ok / n_ok, "flagged": flagged})
    cov = np.array([x["coverage"] for x in reps]); la = np.array([x["load_all"] for x in reps])
    lc = np.array([x["load_correct"] for x in reps])
    share_ok = float(np.mean(cov >= COVER_MIN))
    beats_random = float(np.mean(cov > la))
    accept = bool(share_ok >= COVER_SHARE and la.mean() <= LOAD_MAX and lc.mean() <= LOAD_CAP_CORRECT)
    # в выборке целиком (как у Kimi, для сравнения, не для решения)
    insample = {str(t): {"caught": int(((d > t) & err).sum()), "flag_correct": int(((d > t) & ~err).sum()),
                         "load_all": round(float(np.mean(d > t)), 3)} for t in (0.4, 0.5)}
    from sklearn.metrics import roc_auc_score
    res = {"n": int(len(o)), "errors": int(err.sum()), "fp": int(((o.pred_label == 1) & (o.y_true == 0)).sum()),
           "fn": int(((o.pred_label == 0) & (o.y_true == 1)).sum()), "n_groups": int(len(set(groups))),
           "auc_disagreement_vs_error": round(float(roc_auc_score(err, d)), 3),
           "insample_kimi": insample, "grid": GRID.tolist(), "load_cap_correct_train": LOAD_CAP_CORRECT,
           "criteria": {"coverage_min": COVER_MIN, "coverage_share_repeats": COVER_SHARE, "load_all_max": LOAD_MAX,
                        "load_correct_max": LOAD_CAP_CORRECT},
           "summary": {"coverage_mean": round(float(cov.mean()), 3), "coverage_min": round(float(cov.min()), 3),
                       "coverage_max": round(float(cov.max()), 3), "share_repeats_cov_ge_047": round(share_ok, 3),
                       "share_repeats_cov_gt_load": round(beats_random, 3),
                       "load_all_mean": round(float(la.mean()), 3), "load_correct_mean": round(float(lc.mean()), 3),
                       "caught_mean": round(float(np.mean([x['caught'] for x in reps])), 2),
                       "thresholds_seen": sorted({t for x in reps for t in x["thresholds"]})},
           "accept": accept, "repeats": reps}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "repeats"}, ensure_ascii=False))
    print("->", a.out)


if __name__ == "__main__":
    main()
