#!/usr/bin/env python3
"""Таблица бейзлайнов: вклад архитектуры на тех же OOF-данных, без изменения чисел поставки.

Зачем. Замороженные метрики (`models/metrics_summary.json`) показывают, что делает боевой стек,
но не показывают, насколько это лучше тривиальных решений и каждого контура в отдельности.
Скрипт считает на ОДНИХ И ТЕХ ЖЕ строках `models/oof_stacked_<region>_<crit>.csv` (OOF, группировка
по исследованию) для каждого из пяти критериев ТЗ:

  1. тривиальные бейзлайны: «всегда норма», «всегда нарушение», «случайный по распространённости»
     (ожидание по 2000 розыгрышам, seed 0; флаги — то же правило порога, что у критерия);
  2. только контур A (`oof_geom`), только контур B (`oof_emb`) — порог по тому же правилу
     (`threshold_rule` из metrics_summary.json: `prevalence` / `prevalence_x1.4`), применённому
     к собственным OOF-скорам контура, как это делает обучение для стека;
  3. боевой стек (`oof_stacked`, флаги — `pred_label`, т. е. решения сервиса) — должен
     воспроизвести замороженные AUC/F1 (допуск 1e-3);
  4. «простая модель на всех признаках»: логрегрессия (class_weight=balanced, C=1) на всех
     геометрических признаках региона из `data/geometry_features_canonical.csv` без отбора,
     GroupKFold(5, shuffle, random_state=0) по исследованию, медианная импутация по обучающему фолду,
     порог — то же правило на её OOF-скорах. Признаки из эмбеддинга (`synth_pos_logit`), метки,
     идентификаторы и выход детектора стороны исключены (список — в JSON).

Метрики: ROC-AUC, PR-AUC, F1 при пороге правила, balanced accuracy; 95 % ДИ бутстрапом по
исследованиям (500 ресэмплов, seed 0; одинаковая последовательность ресэмплов для всех строк
таблицы) — только для ROC-AUC. Ресэмплы без обоих классов пропускаются.

Бинарная задача «есть нарушение» по области: метка — OR меток критериев; флаг — OR флагов
критериев (как quality_class в инференсе); скор — max скоров критериев.
Для стека это компонента `max_criteria_only` из `models/metrics_oof_full.json`, а не боевой
`quality_prob` (в нём ещё any-модель); F1 при этом совпадает с замороженным F1 бинарной задачи.

Ничего в models/ и config.yaml не меняется. Выход: docs/baselines.json, docs/BASELINES.md.

Запуск:
    python tools/baseline_table.py                 # -> docs/baselines.json, docs/BASELINES.md
    python tools/baseline_table.py --n-boot 50     # быстрый режим (тест)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))

N_BOOT = 500
SEED_BOOT = 0
SEED_CV = 0
SEED_RANDOM = 0
N_RANDOM = 2000
TOL_FROZEN = 1e-3
THR_TIE_ATOL = 1e-9          # как в src/eval_oof_metrics.py

REGION_NAME = {"spine": "Поясничный отдел позвоночника", "hip": "Проксимальный отдел бедра"}
VIOL_NAME = {"sp_pos": "Некорректная укладка", "sp_axis": "Не выравнена ось позвоночника",
             "sp_art": "Присутствуют посторонние предметы",
             "hip_pos": "Некорректная укладка", "hip_roi": "Некорректная область интереса"}
REGION_CRITERIA = {"spine": ["sp_pos", "sp_axis", "sp_art"], "hip": ["hip_pos", "hip_roi"]}
REGION_ROWS = {"spine": ["spine"], "hip": ["right_hip", "left_hip"]}

MODELS = [  # (ключ, подпись в таблице)
    ("always_normal", "всегда «норма»"),
    ("always_violation", "всегда «нарушение»"),
    ("random_prevalence", "случайный по распространённости (ожидание)"),
    ("contour_a", "только контур A (геометрия)"),
    ("contour_b", "только контур B (эмбеддинги)"),
    ("simple_all_geom", "простая модель на всех геометрических признаках"),
    ("stack", "боевой стек A+B (поставка)"),
]
# Колонки geometry_features_canonical.csv, которые НЕ являются геометрическими признаками
NON_FEATURE_COLS = {"study", "file_path", "sop_instance_uid", "instance_number", "rows", "cols", "region",
                    "applicable", "quality_class", "violation_list",
                    "sp_pos", "sp_axis", "sp_art", "rh_pos", "rh_roi", "lh_pos", "lh_roi",
                    "rh_pos_c", "rh_roi_c", "lh_pos_c", "lh_roi_c", "hip_pos_c", "hip_roi_c",
                    "hip_side_detected", "hip_side_score", "synth_pos_logit"}


def _ensure_src(root: Path) -> None:
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


# --------------------------------------------------------------------------- #
# Правило порога и метрики
# --------------------------------------------------------------------------- #
def rule_threshold(rule: str, y: np.ndarray, scores: np.ndarray) -> float:
    """Порог по правилу — формулы `threshold_by_rule` из src/calibration_utils.py.

    prevalence      -> квантиль скоров уровня 1 - p;
    prevalence_x<k> -> квантиль уровня 1 - min(0.999, k * p), p — доля позитивов.
    """
    p = float(np.mean(y))
    if p <= 0 or p >= 1:
        return 0.5
    rule = (rule or "prevalence").strip()
    if rule == "prevalence":
        return float(np.quantile(scores, 1 - p))
    if rule.startswith("prevalence_x"):
        k = float(rule[len("prevalence_x"):])
        return float(np.quantile(scores, 1 - min(0.999, k * p)))
    raise ValueError(f"правило порога «{rule}» не поддерживается таблицей бейзлайнов")


def flags_by_rule(rule: str, y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, float]:
    thr = rule_threshold(rule, y, scores)
    return (scores >= thr - THR_TIE_ATOL).astype(int), thr


def point_metrics(y: np.ndarray, s: np.ndarray | None, flag: np.ndarray) -> dict:
    two = len(np.unique(y)) == 2
    out = {"f1": float(f1_score(y, flag, zero_division=0)),
           "balanced_accuracy": float(balanced_accuracy_score(y, flag)) if two else float("nan"),
           "n_flag": int(flag.sum())}
    if s is None:
        out["roc_auc"] = float("nan"); out["pr_auc"] = float("nan")
    else:
        out["roc_auc"] = float(roc_auc_score(y, s)) if two else float("nan")
        out["pr_auc"] = float(average_precision_score(y, s)) if y.sum() > 0 else float("nan")
    return out


def bootstrap_auc(y: np.ndarray, s: np.ndarray, groups: np.ndarray, n_boot: int, seed: int) -> dict:
    """Перцентильный ДИ 95 % ROC-AUC, кластер = исследование; ресэмплы без обоих классов пропущены."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_g = {g: np.nonzero(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n_boot):
        sample = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in sample])
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        vals.append(float(roc_auc_score(yb, s[idx])))
    if not vals:
        return {"lo": float("nan"), "hi": float("nan"), "share_skipped": 1.0}
    return {"lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)),
            "share_skipped": 1.0 - len(vals) / float(n_boot)}


def fast_metrics(y: np.ndarray, s: np.ndarray, flag: np.ndarray) -> dict:
    """Те же метрики, что point_metrics, на numpy (для тысяч розыгрышей случайного бейзлайна)."""
    y = y.astype(bool); f = flag.astype(bool)
    tp = int((f & y).sum()); fp = int((f & ~y).sum()); fn = int((~f & y).sum()); tn = int((~f & ~y).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    ba = 0.5 * (tp / (tp + fn) + tn / (tn + fp))
    # ROC-AUC через ранги (со средними рангами при совпадениях)
    r = pd.Series(s).rank().to_numpy()
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    auc = (r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    # average precision как в sklearn: сумма (R_k - R_{k-1}) * P_k по убыванию скора, совпадения — одним блоком
    order = np.argsort(-s, kind="mergesort")
    ss, yy = s[order], y[order].astype(float)
    tp_c = np.cumsum(yy); k = np.arange(1, len(ss) + 1)
    last = np.r_[ss[1:] != ss[:-1], True]           # конец блока одинаковых скоров
    prec = tp_c[last] / k[last]; rec = tp_c[last] / n_pos
    ap = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
    return {"f1": float(f1), "balanced_accuracy": float(ba), "roc_auc": float(auc), "pr_auc": ap, "n_flag": int(f.sum())}


def random_expectation(y: np.ndarray, rule: str, n_draw: int, seed: int) -> dict:
    """Ожидание метрик случайного классификатора: скор ~ U(0,1), флаги — правило порога на этом скоре.

    Для правила prevalence это ровно «пометить случайные n·p кадров».
    AUC в ожидании 0.5 (записывается аналитически), остальные метрики — среднее по розыгрышам.
    """
    rng = np.random.default_rng(seed)
    acc = {"f1": [], "balanced_accuracy": [], "pr_auc": [], "roc_auc": [], "n_flag": []}
    for _ in range(n_draw):
        s = rng.random(len(y))
        flag, _ = flags_by_rule(rule, y, s)
        m = fast_metrics(y, s, flag)
        for k in acc:
            acc[k].append(m[k])
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out["roc_auc_mc"] = out["roc_auc"]
    out["roc_auc"] = 0.5
    out["n_draw"] = int(n_draw)
    return out


# --------------------------------------------------------------------------- #
# Простая модель на всех геометрических признаках
# --------------------------------------------------------------------------- #
def geometry_feature_cols(geom: pd.DataFrame, region: str) -> list[str]:
    d = geom[geom["region"].isin(REGION_ROWS[region])]
    cols = []
    for c in geom.columns:
        if c in NON_FEATURE_COLS or not pd.api.types.is_numeric_dtype(geom[c]):
            continue
        if d[c].notna().any():
            cols.append(c)
    return cols


def oof_simple_model(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    """OOF-вероятности логрегрессии на всех признаках: GroupKFold(5, shuffle) по исследованию."""
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in gkf.split(X, y, groups):
        if y[tr].sum() < 2 or y[tr].sum() == len(tr):
            continue
        med = np.nanmedian(X[tr], axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        Xtr = np.where(np.isnan(X[tr]), med, X[tr]); Xte = np.where(np.isnan(X[te]), med, X[te])
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(sc.transform(Xtr), y[tr])
        oof[te] = clf.predict_proba(sc.transform(Xte))[:, 1]
    if np.isnan(oof).any():          # пропущенный вырожденный фолд — нейтральный скор
        oof = np.where(np.isnan(oof), 0.5, oof)
    return oof


# --------------------------------------------------------------------------- #
# Сборка таблицы
# --------------------------------------------------------------------------- #
def build(root: Path = ROOT, n_boot: int = N_BOOT, seed_boot: int = SEED_BOOT, n_random: int = N_RANDOM,
          with_simple: bool = True) -> dict:
    _ensure_src(root)
    from eval_oof_metrics import flags_from_scores  # noqa: E402
    summary = json.loads((root / "models" / "metrics_summary.json").read_text(encoding="utf-8"))
    full_path = root / "models" / "metrics_oof_full.json"
    full = json.loads(full_path.read_text(encoding="utf-8")) if full_path.exists() else {}
    geom_path = root / "data" / "geometry_features_canonical.csv"
    geom = pd.read_csv(geom_path) if (with_simple and geom_path.exists()) else None

    out = {"version": summary.get("version"),
           "method": {
               "rows": "models/oof_stacked_<region>_<crit>.csv — те же OOF-строки, что у замороженных метрик",
               "threshold": "правило порога критерия (threshold_rule из metrics_summary.json) на OOF-скорах "
                            "самой модели; для стека — pred_label (решение сервиса)",
               "ci": f"ROC-AUC: перцентильный бутстрап 95 %, {n_boot} ресэмплов, seed {seed_boot}, "
                     "кластер = исследование, одна последовательность ресэмплов для всех строк; "
                     "ресэмплы без обоих классов пропущены",
               "random": f"скор U(0,1), флаги по правилу порога; ожидание по {n_random} розыгрышам, seed {SEED_RANDOM}; "
                         "AUC записан аналитически (0.5)",
               "simple_all_geom": "LogisticRegression(C=1, class_weight=balanced) на всех геометрических признаках региона "
                                  "из data/geometry_features_canonical.csv (без synth_pos_logit), медианная импутация по "
                                  f"обучающему фолду, StandardScaler, GroupKFold(5, shuffle=True, random_state={SEED_CV}) по исследованию",
               "region_binary": "метка и флаг — OR по критериям; скор — max скоров критериев (у стека скоры ранговые, "
                                "у контуров и простой модели — вероятности логрегрессии с balanced-весами); для стека это "
                                "компонента max_criteria_only из metrics_oof_full.json; боевой quality_prob содержит ещё "
                                "any-модель и здесь не пересобирается",
               "frozen_tolerance": TOL_FROZEN},
           "models": [{"key": k, "label": v} for k, v in MODELS],
           "criteria": {}, "region_binary": {}, "checks": [], "simple_model_features": {}}

    per_crit_cache: dict[str, dict] = {}
    for region, criteria in REGION_CRITERIA.items():
        feat_cols = geometry_feature_cols(geom, region) if geom is not None else []
        out["simple_model_features"][region] = feat_cols
        for crit in criteria:
            node = summary[region][crit]
            df = pd.read_csv(root / "models" / f"oof_stacked_{region}_{crit}.csv")
            y = df["y_true"].to_numpy().astype(int)
            g = df["study"].to_numpy()
            rule = str(node.get("threshold_rule") or node.get("threshold_method") or "prevalence")
            thr_frozen = float(node["threshold"])
            s_geom = df["oof_geom"].to_numpy(float)
            s_emb = df["oof_emb"].to_numpy(float)
            s_stk = df["oof_stacked"].to_numpy(float)
            flag_stk = flags_from_scores(df, s_stk, thr_frozen)

            rec = {"section": region, "violation_type": VIOL_NAME[crit], "n": int(len(y)), "n_pos": int(y.sum()),
                   "n_studies": int(len(np.unique(g))), "n_pos_studies": int(len(np.unique(g[y == 1]))),
                   "prevalence": float(y.mean()), "threshold_rule": rule, "threshold_frozen": thr_frozen,
                   "rows": {}}
            scores = {}

            # тривиальные
            zeros, ones = np.zeros(len(y), int), np.ones(len(y), int)
            m = point_metrics(y, np.zeros(len(y)), zeros); m["score"] = "константа"
            rec["rows"]["always_normal"] = m
            m = point_metrics(y, np.ones(len(y)), ones); m["score"] = "константа"
            rec["rows"]["always_violation"] = m
            m = random_expectation(y, rule, n_random, SEED_RANDOM); m["score"] = "U(0,1)"
            rec["rows"]["random_prevalence"] = m

            # контуры
            for key, s in (("contour_a", s_geom), ("contour_b", s_emb)):
                flag, thr = flags_by_rule(rule, y, s)
                m = point_metrics(y, s, flag); m["threshold"] = thr
                rec["rows"][key] = m; scores[key] = s

            # простая модель
            if geom is not None and feat_cols:
                gsub = geom.set_index("file_path").loc[df["file_path"].to_numpy()]
                X = gsub[feat_cols].to_numpy(np.float64)
                s_simple = oof_simple_model(X, y, g, SEED_CV)
                flag, thr = flags_by_rule(rule, y, s_simple)
                m = point_metrics(y, s_simple, flag); m["threshold"] = thr; m["n_features"] = len(feat_cols)
                rec["rows"]["simple_all_geom"] = m; scores["simple_all_geom"] = s_simple
            else:
                rec["rows"]["simple_all_geom"] = {"skipped": "нет data/geometry_features_canonical.csv"}

            # стек
            m = point_metrics(y, s_stk, flag_stk); m["threshold"] = thr_frozen
            m["threshold_by_rule_recomputed"] = rule_threshold(rule, y, s_stk)
            # в таблицах печатается опубликованное значение (одни и те же числа во всех документах);
            # пересчёт по OOF-файлу сравнивается с ним в разделе «Сверка»
            m["roc_auc_published"] = float(node["auc_stacked"])
            rec["rows"]["stack"] = m; scores["stack"] = s_stk

            # ДИ AUC
            for key, s in scores.items():
                rec["rows"][key]["roc_auc_ci"] = bootstrap_auc(y, s, g, n_boot, seed_boot)
            for key in ("always_normal", "always_violation"):
                rec["rows"][key]["roc_auc_ci"] = {"lo": 0.5, "hi": 0.5, "share_skipped": 0.0}

            # сверка с замороженными
            chk = {"criterion": crit, "n_ok": int(len(y)) == int(node["n_valid"]), "n_pos_ok": int(y.sum()) == int(node["n_pos"]),
                   "auc_stack_diff": abs(m["roc_auc"] - float(node["auc_stacked"])),
                   "f1_stack_diff": abs(m["f1"] - float(node["f1_oof"])),
                   "auc_geom_diff": abs(rec["rows"]["contour_a"]["roc_auc"] - float(node["auc_geom"])),
                   "auc_emb_diff": abs(rec["rows"]["contour_b"]["roc_auc"] - float(node["auc_emb"])),
                   "threshold_rule_diff": abs(m["threshold_by_rule_recomputed"] - thr_frozen),
                   "n_flag_ok": int(flag_stk.sum()) == int(node["n_flag_oof"])}
            fv = full.get("by_violation_type", {}).get(f"{region}/{crit}")
            if fv is not None:
                chk["auc_stack_diff_oof_full"] = abs(m["roc_auc"] - float(fv["roc_auc"]))
                chk["pr_auc_stack_diff_oof_full"] = abs(m["pr_auc"] - float(fv["pr_auc"]))
                chk["bal_acc_stack_diff_oof_full"] = abs(m["balanced_accuracy"] - float(fv["balanced_accuracy"]))
            chk["ok"] = bool(chk["n_ok"] and chk["n_pos_ok"] and chk["auc_stack_diff"] <= TOL_FROZEN
                             and chk["f1_stack_diff"] <= TOL_FROZEN and chk["auc_geom_diff"] <= TOL_FROZEN
                             and chk["auc_emb_diff"] <= TOL_FROZEN)
            rec["frozen"] = {"auc_geom": node["auc_geom"], "auc_emb": node["auc_emb"], "auc_stacked": node["auc_stacked"],
                             "f1_oof": node["f1_oof"], "n_flag_oof": node["n_flag_oof"],
                             "oof_full": ({"roc_auc": fv["roc_auc"], "pr_auc": fv["pr_auc"], "f1": fv["f1"],
                                           "balanced_accuracy": fv["balanced_accuracy"]} if fv else None)}
            rec["check"] = chk
            out["checks"].append(chk)
            out["criteria"][crit] = rec
            per_crit_cache[crit] = {"y": y, "g": g, "rule": rule, "scores": scores,
                                    "flags": {"stack": flag_stk,
                                              "contour_a": (s_geom >= rec["rows"]["contour_a"]["threshold"] - THR_TIE_ATOL).astype(int),
                                              "contour_b": (s_emb >= rec["rows"]["contour_b"]["threshold"] - THR_TIE_ATOL).astype(int),
                                              **({"simple_all_geom": (scores["simple_all_geom"] >= rec["rows"]["simple_all_geom"]["threshold"] - THR_TIE_ATOL).astype(int)}
                                                 if "simple_all_geom" in scores else {})},
                                    "file_path": df["file_path"].to_numpy()}

        # ---- бинарная задача по области ----
        base_fp = per_crit_cache[criteria[0]]["file_path"]
        for c in criteria[1:]:
            assert (per_crit_cache[c]["file_path"] == base_fp).all(), "OOF-файлы региона не выровнены"
        y_any = np.zeros(len(base_fp), int)
        for c in criteria:
            y_any = np.maximum(y_any, per_crit_cache[c]["y"])
        g = per_crit_cache[criteria[0]]["g"]
        rb = {"anatomical_region": REGION_NAME[region], "criteria": criteria, "n": int(len(y_any)), "n_pos": int(y_any.sum()),
              "n_studies": int(len(np.unique(g))), "n_pos_studies": int(len(np.unique(g[y_any == 1]))),
              "prevalence": float(y_any.mean()), "rows": {}}
        zeros, ones = np.zeros(len(y_any), int), np.ones(len(y_any), int)
        m = point_metrics(y_any, np.zeros(len(y_any)), zeros); m["roc_auc_ci"] = {"lo": 0.5, "hi": 0.5, "share_skipped": 0.0}
        rb["rows"]["always_normal"] = m
        m = point_metrics(y_any, np.ones(len(y_any)), ones); m["roc_auc_ci"] = {"lo": 0.5, "hi": 0.5, "share_skipped": 0.0}
        rb["rows"]["always_violation"] = m
        # случайный: независимый розыгрыш по каждому критерию, OR флагов, max скоров
        rng = np.random.default_rng(SEED_RANDOM)
        acc = {"f1": [], "balanced_accuracy": [], "pr_auc": [], "roc_auc": [], "n_flag": []}
        for _ in range(n_random):
            fl = np.zeros(len(y_any), int); sc = np.zeros(len(y_any))
            for c in criteria:
                cc = per_crit_cache[c]
                s = rng.random(len(y_any))
                f, _ = flags_by_rule(cc["rule"], cc["y"], s)
                fl = np.maximum(fl, f); sc = np.maximum(sc, s)
            mm = fast_metrics(y_any, sc, fl)
            for k in acc:
                acc[k].append(mm[k])
        m = {k: float(np.mean(v)) for k, v in acc.items()}; m["roc_auc_mc"] = m["roc_auc"]; m["roc_auc"] = 0.5; m["n_draw"] = int(n_random)
        rb["rows"]["random_prevalence"] = m
        for key in ("contour_a", "contour_b", "simple_all_geom", "stack"):
            if any(key not in per_crit_cache[c]["scores"] for c in criteria):
                rb["rows"][key] = {"skipped": "нет скоров по всем критериям"}
                continue
            fl = np.zeros(len(y_any), int); sc = np.zeros(len(y_any))
            for c in criteria:
                cc = per_crit_cache[c]
                fl = np.maximum(fl, cc["flags"][key]); sc = np.maximum(sc, cc["scores"][key])
            m = point_metrics(y_any, sc, fl)
            m["roc_auc_ci"] = bootstrap_auc(y_any, sc, g, n_boot, seed_boot)
            rb["rows"][key] = m
        fb = full.get("by_region_binary", {}).get(region)
        if fb:
            st = rb["rows"]["stack"]
            chk = {"criterion": f"{region}/binary", "n_ok": rb["n"] == int(fb["n"]), "n_pos_ok": rb["n_pos"] == int(fb["n_pos"]),
                   "f1_stack_diff": abs(st["f1"] - float(fb["f1"])),
                   "auc_max_criteria_diff": abs(st["roc_auc"] - float(fb["roc_auc_components"]["max_criteria_only"]))}
            chk["ok"] = bool(chk["n_ok"] and chk["n_pos_ok"] and chk["f1_stack_diff"] <= TOL_FROZEN
                             and chk["auc_max_criteria_diff"] <= TOL_FROZEN)
            rb["frozen"] = {"f1": fb["f1"], "roc_auc_quality_prob": fb["roc_auc"],
                            "roc_auc_max_criteria_only": fb["roc_auc_components"]["max_criteria_only"],
                            "balanced_accuracy": fb["balanced_accuracy"], "pr_auc_quality_prob": fb["pr_auc"]}
            rb["check"] = chk
            out["checks"].append(chk)
        out["region_binary"][region] = rb

    out["all_frozen_ok"] = all(c["ok"] for c in out["checks"])
    out["n_boot"] = int(n_boot); out["seed_boot"] = int(seed_boot); out["n_random"] = int(n_random)
    return out


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def _f(v, nd=3):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def _auc_ci(row):
    pt = row.get("roc_auc_published", row["roc_auc"])
    ci = row.get("roc_auc_ci")
    if not ci or np.isnan(ci.get("lo", float("nan"))):
        return _f(pt)
    return f"{pt:.3f} [{ci['lo']:.2f}; {ci['hi']:.2f}]"


def _table(rows: dict) -> list[str]:
    L = ["| Модель | ROC-AUC [95 % ДИ по исслед.] | PR-AUC | F1 при пороге правила | Bal. acc. | Помечено |",
         "|---|---|---|---|---|---|"]
    for key, label in MODELS:
        r = rows.get(key)
        if r is None or "skipped" in r:
            L.append(f"| {label} | — | — | — | — | {r.get('skipped', '—') if r else '—'} |")
            continue
        bold = key == "stack"
        b = "**" if bold else ""
        nflag = f"{r['n_flag']:.1f}" if key == "random_prevalence" else str(r["n_flag"])
        L.append(f"| {b}{label}{b} | {b}{_auc_ci(r)}{b} | {_f(r['pr_auc'])} | {b}{_f(r['f1'])}{b} | {_f(r['balanced_accuracy'])} | {nflag} |")
    return L


def to_markdown(p: dict) -> str:
    L = ["# Таблица бейзлайнов: вклад архитектуры на OOF-данных", "",
         f"Источник: `docs/baselines.json` (`python tools/baseline_table.py`; ROC-AUC ДИ — бутстрап по исследованиям, "
         f"{p['n_boot']} ресэмплов, seed {p['seed_boot']}; случайный бейзлайн — ожидание по {p['n_random']} розыгрышам). "
         "Все строки считаются на одних и тех же OOF-строках `models/oof_stacked_*.csv`; числа поставки не меняются, "
         "строка «боевой стек» — их воспроизведение.", "",
         "Порог для каждой строки — правило критерия (`prevalence` или `prevalence_x1.4`) на OOF-скорах самой модели, "
         "как это делается для стека; для стека взяты решения сервиса (`pred_label`). Для «всегда норма» / «всегда нарушение» "
         "F1 и balanced accuracy детерминированы, ROC-AUC = 0.5. «Простая модель» — логрегрессия на всех геометрических "
         "признаках региона без отбора (GroupKFold(5) по исследованию, seed 0); списки признаков — в JSON. Точечный ROC-AUC "
         "стека по критериям — опубликованное значение `models/metrics_summary.json`; пересчёт по OOF-файлу — в разделе «Сверка».", ""]
    for crit, c in p["criteria"].items():
        L += [f"## {c['violation_type']} (`{crit}`), {REGION_NAME[c['section']]}", "",
              f"n = {c['n']} кадров / {c['n_studies']} исследований, позитивов {c['n_pos']} / {c['n_pos_studies']} "
              f"(доля {c['prevalence']:.3f}); правило порога `{c['threshold_rule']}`"
              + (f"; простая модель — {c['rows']['simple_all_geom'].get('n_features', '—')} признаков." if "n_features" in c["rows"].get("simple_all_geom", {}) else "."),
              ""]
        L += _table(c["rows"])
        L.append("")
    L += ["## Бинарная задача «есть нарушение» по области", "",
          "Метка и флаг — OR по критериям области (как `quality_class`), скор — max скоров критериев (у стека — ранговые скоры, "
          "у контуров и простой модели — вероятности логрегрессии). "
          "Для стека это компонента `max_criteria_only` из `models/metrics_oof_full.json`; боевой `quality_prob` содержит "
          "ещё any-модель и здесь не пересобирается, поэтому его AUC в таблице нет. F1 при этом совпадает с замороженным F1 бинарной задачи.", ""]
    for region, rb in p["region_binary"].items():
        L += [f"### {rb['anatomical_region']}", "",
              f"n = {rb['n']} / {rb['n_studies']} исследований, позитивов {rb['n_pos']} / {rb['n_pos_studies']} (доля {rb['prevalence']:.3f})."
              + (f" Замороженные: F1 {rb['frozen']['f1']:.3f}, ROC-AUC `quality_prob` {rb['frozen']['roc_auc_quality_prob']:.3f}, "
                 f"`max_criteria_only` {rb['frozen']['roc_auc_max_criteria_only']:.3f}." if rb.get("frozen") else ""), ""]
        L += _table(rb["rows"])
        L.append("")
    L += ["## Сверка с замороженными числами", "",
          "| Критерий | n | позитивы | Δ AUC стека | Δ F1 стека | Δ AUC A | Δ AUC B | Δ порога по правилу | помечено совпало |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in p["checks"]:
        if "auc_stack_diff" in c:
            L.append(f"| `{c['criterion']}` | {'да' if c['n_ok'] else 'НЕТ'} | {'да' if c['n_pos_ok'] else 'НЕТ'} | "
                     f"{c['auc_stack_diff']:.5f} | {c['f1_stack_diff']:.5f} | {c['auc_geom_diff']:.5f} | {c['auc_emb_diff']:.5f} | "
                     f"{c['threshold_rule_diff']:.1e} | {'да' if c['n_flag_ok'] else 'НЕТ'} |")
        else:
            L.append(f"| `{c['criterion']}` | {'да' if c['n_ok'] else 'НЕТ'} | {'да' if c['n_pos_ok'] else 'НЕТ'} | "
                     f"{c['auc_max_criteria_diff']:.5f} (max по критериям) | {c['f1_stack_diff']:.5f} | — | — | — | — |")
    L.append("")
    sp = p["criteria"].get("sp_art", {}).get("check", {})
    L.append(("Все сверки в пределах допуска 1e-3." if p["all_frozen_ok"] else "ВНИМАНИЕ: есть расхождения сверх допуска 1e-3.")
             + (f" Для `sp_art` расхождение AUC стека {sp.get('auc_stack_diff', 0):.5f}: в `metrics_summary.json` записано 0.8227, "
                "по OOF-файлу и `metrics_oof_full.json` — 0.8237 (округление скоров при записи, см. `docs/EVIDENCE.md`); "
                "в документах используется 0.823." if sp.get("auc_stack_diff", 0) > 5e-4 else ""))
    L += ["", "## Что показывает и чего не доказывает", "",
          "- Таблица отвечает на один вопрос: насколько поставленный стек лучше на тех же OOF-строках, чем ничего не делать, "
          "чем угадывать по доле позитивов, чем каждый контур в отдельности и чем логрегрессия на всех признаках без отбора.",
          "- «Всегда нарушение» задаёт нижнюю планку F1, равную 2p/(1+p) (у `hip_pos` 0.39, у `sp_art` 0.35): F1 стека надо читать "
          "относительно неё, а не относительно нуля. У случайного классификатора при том же правиле порога F1 в ожидании близок к доле позитивов.",
          "- Стек не везде лучше сильнейшего контура по AUC (`sp_pos`; в 2.5.0 также бинарная задача позвоночника — контур A 0.831 против 0.825); его роль — устойчивость при заранее "
          "фиксированном весе 0.5/0.5, а не максимум AUC на этой выборке. Пример хрупкости одного контура: у `sp_pos` контур A "
          "имеет AUC 0.915, но при пороге по доле позитивов помечает 10 клонов одного отрицательного исследования и даёт F1 = 0; "
          "стек с тем же правилом даёт F1 0.50.",
          "- Простая модель на всех признаках показывает цену отбора признаков: при 10–35 позитивах десятки шумных признаков "
          "понижают AUC относительно физически осмысленного набора (мотивировка отбора — комментарий к `CRITERION_GEOMETRY_COLS` в `src/train_stacked.py`).",
          "- Не доказывает: обобщение на другие аппараты и разметчиков (одна клиника, один эксперт), устойчивость оценки к "
          "разбиению (см. `docs/METRICS_REPORT.md`, «Разброс по разбиениям») и качество на закрытом тесте организаторов. "
          "Интервалы по редким критериям (`sp_pos` — 6 положительных исследований, `hip_roi` — 6) широки и перекрываются с бейзлайнами.",
          "- Пороги для контуров A и B и для простой модели подобраны тем же правилом на их собственных OOF-скорах, поэтому F1 этих строк "
          "сопоставим с F1 стека; но правило применено на всей OOF-выборке, как и у стека, — это не nested-оценка.",
          "- С внешними ориентирами таблица не сравнивается; она описывает только внутренние различия на одной выборке."]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Таблица бейзлайнов на OOF-данных")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED_BOOT)
    ap.add_argument("--n-random", type=int, default=N_RANDOM)
    ap.add_argument("--no-simple", action="store_true", help="не обучать простую модель на всех признаках")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    p = build(root, a.n_boot, a.seed, a.n_random, not a.no_simple)
    out_json = Path(a.out_json) if a.out_json else root / "docs" / "baselines.json"
    out_md = Path(a.out_md) if a.out_md else root / "docs" / "BASELINES.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(p, ensure_ascii=False, indent=1, allow_nan=True), encoding="utf-8")
    md = to_markdown(p)
    out_md.write_text(md, encoding="utf-8")
    print(md)
    print(f"записано: {out_json}, {out_md}")
    return 0 if p["all_frozen_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
