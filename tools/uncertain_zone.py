#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Зона «не уверен»: кривая «покрытие → полнота» на OOF (идея 10 консилиума, 23.09.2026).

Вопрос: что даёт зона «не уверен» (строки на просмотр врачу) при разной ширине полосы вокруг порога,
и разумно ли текущее правило. Числа поставки (модели, пороги, запасы в config.yaml, 9 колонок CSV) НЕ меняются:
скрипт только считает и пишет отчёт; решение о смене правила принимается отдельно.

Текущее правило сервиса (src/inference.py, src/calibration_utils.py, config.yaml: uncertainty):
  критерий «не уверен» <=> |score − threshold| <= margin[crit], где score — стэкнутый ранговый скор
  (0.5 · ранг контура A + 0.5 · ранг контура B), margin — запас по критерию (config.yaml
  uncertainty.margin_by_criterion, подобран по квоте 5 % строк на критерий);
  строка региона «не уверен» (needs_review = 1, risk_level «средний») <=> не уверен хотя бы один критерий региона.
  Зона по quality_prob в сервисе НЕ используется; здесь она посчитана для сравнения.

Семейства правил, которые сравниваются на одном OOF (тот же файл, что и docs/METRICS_REPORT.md):
  current      — текущее правило (запасы по критериям из config.yaml / models/calibration.pkl);
  score_band   — единая полоса δ для всех критериев региона: |score − threshold| <= δ, δ = 0.00 … 0.30 шаг 0.01;
  qp_band      — полоса по quality_prob: |quality_prob − 0.5| <= δ (quality_prob на OOF воспроизводится как в
                 src/eval_oof_metrics.py: 0.5 · any-модель + 0.5 · max скоров критериев, согласование с классом).

Что считается для каждой точки (по строкам региона; метка «есть нарушение» = хотя бы один критерий = 1,
класс = хотя бы один флаг критерия; флаги — pred_label из OOF-файла, ровно как в сервисе):
  доля строк в зоне (1 − покрытие); среди уверенных строк — чувствительность (полнота), специфичность, F1,
  точность; доля положительных строк, попавших в зону; доля ошибок класса (FN + FP), попавших в зону, и отдельно доля
  пропусков (FN) в зоне; «полнота с учётом зоны» = (TP среди уверенных + положительные в зоне) / все положительные —
  верхняя оценка при условии, что зону смотрит врач. То же по каждому критерию (флаг критерия против y_true).
  ДИ 95 % — кластерный percentile-бутстрап по исследованиям (500 ресэмплов, зерно 0) для строковых кривых.

Проекция на выгрузку боевой сборки (dataset/regress_2_3_2_debug.csv, если файл передан через --regress):
  сколько строк из 499 было бы «не уверен» при каждом δ (по колонкам <crit>_margin = |score − threshold| того прогона).
  Оговорка: это скоры 2.3.2 (sp_pos в 2.4.0 переобучен, запас sp_pos 0.0422 → 0.0), поэтому проекция приближённая.

Выход (только агрегаты, без путей к кадрам и идентификаторов пациентов):
  docs/uncertain_zone.json, docs/uncertain_zone/coverage_recall_<spine|hip>.svg,
  docs/uncertain_zone/coverage_recall_criteria.svg; таблицы для docs/UNCERTAIN_ZONE.md — в stdout (--markdown).

Запуск:
  OMP_NUM_THREADS=1 python tools/uncertain_zone.py --regress ../dataset/regress_2_3_2_debug.csv --markdown
  python tools/uncertain_zone.py --out DIR --n-boot 50 --no-svg        # быстрый прогон (тест)
Переменные: DENSITO_ROOT (корень репозитория).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
MODELS_DIR = ROOT / "models"
DOCS_JSON = ROOT / "docs" / "uncertain_zone.json"
DOCS_SVG_DIR = ROOT / "docs" / "uncertain_zone"

N_BOOT = 500
BOOT_SEED = 0
DELTA_GRID = [round(0.01 * k, 2) for k in range(0, 31)]
KEY_DELTAS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30]
TIE_EPS = 1e-12
REVIEW_CAP = 0.20  # ориентир: не больше одной строки из пяти на просмотр врачу

REGION_CRITERIA = {"spine": ["sp_pos", "sp_axis", "sp_art"], "hip": ["hip_pos", "hip_roi"]}
REGION_NAME = {"spine": "Поясничный отдел позвоночника", "hip": "Проксимальный отдел бедра"}
CRIT_NAME = {"sp_pos": "Некорректная укладка (позвоночник)", "sp_axis": "Не выравнена ось позвоночника",
             "sp_art": "Присутствуют посторонние предметы", "hip_pos": "Некорректная укладка (бедро)",
             "hip_roi": "Некорректная область интереса"}
# в выгрузке боевой сборки бедро идёт как rh_*/lh_*
REGRESS_CRITS = {"spine": {"sp_pos": ["sp_pos"], "sp_axis": ["sp_axis"], "sp_art": ["sp_art"]},
                 "hip": {"hip_pos": ["rh_pos", "lh_pos"], "hip_roi": ["rh_roi", "lh_roi"]}}


# ----------------------------------------------------------------------------- загрузка
def load_yaml_config() -> dict:
    try:
        import yaml
        with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:  # noqa: BLE001
        return {}


def load_calibration_pkl() -> Optional[dict]:
    p = MODELS_DIR / "calibration.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        c = pickle.load(f)
    return c if isinstance(c, dict) and c.get("kind") == "densito_calibration" else None


def margins_from_config(cfg: dict, calib: Optional[dict]) -> Dict[str, float]:
    """Запас по критерию ровно как его читает src/inference.py: config.yaml, при null — calibration.pkl."""
    out = {}
    ucfg = cfg.get("uncertainty") or {}
    mbc = ucfg.get("margin_by_criterion") or {}
    enabled = ucfg.get("enabled", True) is not False
    for crit in [c for cs in REGION_CRITERIA.values() for c in cs]:
        v = mbc.get(crit) if enabled else None
        if v is None and enabled and calib is not None:
            v = (calib.get("margin_by_criterion") or {}).get(crit)
            if v is None:
                v = ((calib.get("criteria") or {}).get(crit) or {}).get("margin")
        out[crit] = float(v) if v is not None else 0.0
    return out


def flags_from(df, s: np.ndarray, thr: float) -> np.ndarray:
    """Флаг критерия ровно как в сервисе: pred_label из CSV, иначе сравнение с порогом с допуском."""
    import pandas as pd
    if "pred_label" in df.columns:
        pl = pd.to_numeric(df["pred_label"], errors="coerce")
        if not pl.isna().any():
            return pl.values.astype(int)
    return (s >= thr - 1e-9).astype(int)


def load_region(region: str, crits: Sequence[str], summary: dict, margins: Dict[str, float]) -> dict:
    """Строки региона: y (any), flag (any), quality_prob_oof, расстояния до порога по критериям, группы."""
    import pandas as pd
    sys.path.insert(0, str(ROOT / "src"))
    import eval_oof_metrics as eom  # OOF any-модели ровно как в отчёте метрик

    cfg = load_yaml_config()
    w_model = float(((cfg.get("stacking") or {}).get("any_blend_weight_model", 0.5)))
    consistent = bool(((cfg.get("stacking") or {}).get("consistent_quality_prob", True)))

    per_crit = {}
    fp = None
    for crit in crits:
        df = pd.read_csv(MODELS_DIR / f"oof_stacked_{region}_{crit}.csv")
        if fp is None:
            fp = df["file_path"].values
        assert (df["file_path"].values == fp).all(), "OOF files misaligned"
        thr = float(summary[region][crit]["threshold"])
        s = df["oof_stacked"].values.astype(float)
        per_crit[crit] = {"y": df["y_true"].values.astype(int), "s": s, "thr": thr,
                          "flag": flags_from(df, s, thr), "dist": np.abs(s - thr), "margin": margins[crit]}
    n = len(fp)
    y = np.zeros(n, int); flag = np.zeros(n, int); crit_max = np.zeros(n)
    for c in crits:
        y = np.maximum(y, per_crit[c]["y"]); flag = np.maximum(flag, per_crit[c]["flag"])
        crit_max = np.maximum(crit_max, per_crit[c]["s"])
    groups = pd.read_csv(MODELS_DIR / f"oof_stacked_{region}_{crits[0]}.csv")["study"].values
    qp = None
    try:
        anym = eom.oof_any_model(region, list(crits)).set_index("file_path").loc[fp]
        assert (anym["y_any"].values == y).all(), "any-label mismatch"
        raw = w_model * anym["any_model_oof"].values.astype(float) + (1.0 - w_model) * crit_max
        qp = np.where(flag == 1, 0.5 + 0.5 * raw, np.minimum(0.5 * raw, 0.499999)) if consistent else raw
    except Exception as e:  # noqa: BLE001
        print(f"[warn] OOF any-модели для {region} не воспроизведён ({e}); семейство qp_band пропущено", file=sys.stderr)
    return {"n": n, "y": y, "flag": flag, "groups": groups, "qp": qp, "crits": per_crit}


# ----------------------------------------------------------------------------- метрики одной точки
def counts(y: np.ndarray, flag: np.ndarray, unc: np.ndarray) -> np.ndarray:
    """Вектор счётчиков: [n, n_unc, TPc, FNc, FPc, TNc, pos_in_zone, fn_in_zone, fp_in_zone, n_pos, n_err]
    (индекс c — среди уверенных строк)."""
    k = ~unc
    tp = int((flag[k] & y[k]).sum()); fn = int(((1 - flag[k]) & y[k]).sum())
    fpn = int((flag[k] & (1 - y[k])).sum()); tn = int(((1 - flag[k]) & (1 - y[k])).sum())
    err = flag != y
    return np.array([len(y), int(unc.sum()), tp, fn, fpn, tn, int((unc & (y == 1)).sum()),
                     int((unc & (y == 1) & (flag == 0)).sum()), int((unc & (y == 0) & (flag == 1)).sum()),
                     int(y.sum()), int(err.sum())], dtype=np.int64)


def _safe(a, b):
    return float(a / b) if b > 0 else None


def metrics_from_counts(c: np.ndarray) -> dict:
    n, n_unc, tp, fn, fpn, tn, pos_z, fn_z, fp_z, n_pos, n_err = [int(v) for v in c]
    return {
        "n": n, "n_uncertain": n_unc, "share_uncertain": _safe(n_unc, n), "coverage": _safe(n - n_unc, n),
        "n_confident": n - n_unc,
        "sensitivity_confident": _safe(tp, tp + fn), "specificity_confident": _safe(tn, tn + fpn),
        "precision_confident": _safe(tp, tp + fpn), "f1_confident": _safe(2 * tp, 2 * tp + fn + fpn),
        "n_pos": n_pos, "n_pos_in_zone": pos_z, "share_pos_in_zone": _safe(pos_z, n_pos),
        "n_errors": n_err, "n_errors_in_zone": fn_z + fp_z, "share_errors_in_zone": _safe(fn_z + fp_z, n_err),
        "n_fn_total": fn + fn_z, "n_fn_in_zone": fn_z, "share_fn_in_zone": _safe(fn_z, fn + fn_z),
        "n_fp_total": fpn + fp_z, "n_fp_in_zone": fp_z,
        "error_rate_in_zone": _safe(fn_z + fp_z, n_unc), "error_rate_all": _safe(n_err, n),
        "recall_with_review": _safe(tp + pos_z, n_pos),
        # пропусков в зоне сверх того, что дал бы случайный выбор того же числа строк
        "fn_in_zone_excess": (float(fn_z - n_unc * (fn + fn_z) / n) if n else None),
    }


def zone_score_band(reg: dict, delta: Optional[float], margins: Optional[Dict[str, float]] = None) -> np.ndarray:
    """delta=None — текущее правило (запасы по критериям); иначе единая полоса delta."""
    unc = np.zeros(reg["n"], bool)
    for c, d in reg["crits"].items():
        m = (margins[c] if margins is not None else d["margin"]) if delta is None else delta
        unc |= d["dist"] <= float(m) + TIE_EPS
    return unc


def zone_qp_band(reg: dict, delta: float) -> np.ndarray:
    return np.abs(reg["qp"] - 0.5) <= delta + TIE_EPS


# ----------------------------------------------------------------------------- бутстрап по исследованиям
def bootstrap_curve(y, flag, groups, zones: List[np.ndarray], n_boot: int, seed: int) -> List[dict]:
    """Кластерный percentile-бутстрап: исследования выбираются с возвращением; счётчики суммируются по
    исследованиям, метрики пересчитываются из сумм. Возвращает ДИ для каждой точки."""
    ug, inv = np.unique(groups, return_inverse=True)
    G = len(ug)
    # матрица G x P x 11 счётчиков по исследованиям
    per_study = np.zeros((G, len(zones), 11), dtype=np.int64)
    for gi in range(G):
        m = inv == gi
        for pi, unc in enumerate(zones):
            per_study[gi, pi] = counts(y[m], flag[m], unc[m])
    rng = np.random.default_rng(seed)
    keys = ["share_uncertain", "sensitivity_confident", "specificity_confident", "f1_confident",
            "share_pos_in_zone", "share_errors_in_zone", "recall_with_review"]
    samples = {k: np.full((n_boot, len(zones)), np.nan) for k in keys}
    for b in range(n_boot):
        idx = rng.integers(0, G, G)
        tot = per_study[idx].sum(axis=0)  # P x 11
        for pi in range(len(zones)):
            mt = metrics_from_counts(tot[pi])
            for k in keys:
                if mt[k] is not None:
                    samples[k][b, pi] = mt[k]
    out = []
    for pi in range(len(zones)):
        ci = {}
        for k in keys:
            col = samples[k][:, pi]
            col = col[~np.isnan(col)]
            ci[k] = [float(np.percentile(col, 2.5)), float(np.percentile(col, 97.5))] if len(col) else None
        out.append(ci)
    return out


def paired_bootstrap_diff(y, flag, groups, zone_a: np.ndarray, zone_b: np.ndarray, n_boot: int, seed: int) -> dict:
    """Парный кластерный бутстрап разности метрик (b − a) на одних и тех же ресэмплах исследований:
    честнее сравнения двух отдельных ДИ. Возвращает точечную разность, ДИ 95 % и долю ресэмплов с разностью <= 0."""
    ug, inv = np.unique(groups, return_inverse=True)
    G = len(ug)
    per_study = np.zeros((G, 2, 11), dtype=np.int64)
    for gi in range(G):
        m = inv == gi
        per_study[gi, 0] = counts(y[m], flag[m], zone_a[m])
        per_study[gi, 1] = counts(y[m], flag[m], zone_b[m])
    keys = ["share_uncertain", "sensitivity_confident", "f1_confident", "recall_with_review", "share_fn_in_zone"]
    rng = np.random.default_rng(seed)
    diffs = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, G, G)
        tot = per_study[idx].sum(axis=0)
        ma, mb = metrics_from_counts(tot[0]), metrics_from_counts(tot[1])
        for k in keys:
            if ma[k] is not None and mb[k] is not None:
                diffs[k].append(mb[k] - ma[k])
    pa, pb = metrics_from_counts(per_study[:, 0].sum(axis=0)), metrics_from_counts(per_study[:, 1].sum(axis=0))
    out = {}
    for k in keys:
        d = np.asarray(diffs[k], float)
        out[k] = {"diff": (pb[k] - pa[k]) if (pa[k] is not None and pb[k] is not None) else None,
                  "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if len(d) else None,
                  "share_resamples_le_0": float((d <= 0).mean()) if len(d) else None}
    return out


# ----------------------------------------------------------------------------- сборка результата
def build(n_boot: int = N_BOOT, seed: int = BOOT_SEED, regress_path: Optional[Path] = None) -> dict:
    summary = json.load(open(MODELS_DIR / "metrics_summary.json", encoding="utf-8"))
    cfg = load_yaml_config()
    calib = load_calibration_pkl()
    margins = margins_from_config(cfg, calib)
    res = {"method": {
        "rule_current": "критерий «не уверен» <=> |score − threshold| <= margin[crit]; строка — если не уверен хотя бы "
                        "один критерий региона (src/inference.py, src/calibration_utils.py, config.yaml: uncertainty)",
        "rule_score_band": "единая полоса δ для всех критериев региона: |score − threshold| <= δ",
        "rule_qp_band": "полоса по quality_prob (OOF, как в src/eval_oof_metrics.py): |quality_prob − 0.5| <= δ",
        "delta_grid": DELTA_GRID, "n_boot": n_boot, "seed": seed, "ci": "percentile 95 %, кластерный бутстрап по исследованиям",
        "flags": "pred_label из models/oof_stacked_*.csv (как в сервисе); метка строки — хотя бы один критерий",
        "review_cap": REVIEW_CAP},
        "inputs": {"thresholds": {}, "margins": margins, "oof_files": []},
        "regions": {}, "criteria": {}, "regress": None, "recommendation": {}}

    regs = {}
    for region, crits in REGION_CRITERIA.items():
        reg = load_region(region, crits, summary, margins)
        regs[region] = reg
        for c in crits:
            res["inputs"]["thresholds"][c] = reg["crits"][c]["thr"]
            res["inputs"]["oof_files"].append(f"models/oof_stacked_{region}_{c}.csv")
        y, flag, g = reg["y"], reg["flag"], reg["groups"]
        # --- строки региона: текущее правило + полоса по скорам
        zones = [zone_score_band(reg, None)] + [zone_score_band(reg, d) for d in DELTA_GRID]
        labels = ["current"] + [f"{d:.2f}" for d in DELTA_GRID]
        pts = [metrics_from_counts(counts(y, flag, z)) for z in zones]
        cis = bootstrap_curve(y, flag, g, zones, n_boot, seed) if n_boot > 0 else [None] * len(zones)
        region_out = {"name": REGION_NAME[region], "n": int(reg["n"]), "n_pos": int(y.sum()),
                      "n_studies": int(len(np.unique(g))), "n_pos_studies": int(len(np.unique(g[y == 1]))),
                      "share_class1": float(flag.mean()),
                      "current": dict(pts[0], ci95=cis[0], margins={c: margins[c] for c in crits}),
                      "score_band": [dict(delta=DELTA_GRID[i], **pts[i + 1], ci95=cis[i + 1]) for i in range(len(DELTA_GRID))],
                      "qp_band": None}
        if reg["qp"] is not None:
            qz = [zone_qp_band(reg, d) for d in DELTA_GRID]
            qpts = [metrics_from_counts(counts(y, flag, z)) for z in qz]
            qcis = bootstrap_curve(y, flag, g, qz, n_boot, seed) if n_boot > 0 else [None] * len(qz)
            region_out["qp_band"] = [dict(delta=DELTA_GRID[i], **qpts[i], ci95=qcis[i]) for i in range(len(DELTA_GRID))]
            region_out["qp_gap"] = {"max_qp_class0": float(reg["qp"][flag == 0].max()) if (flag == 0).any() else None,
                                    "min_qp_class1": float(reg["qp"][flag == 1].min()) if (flag == 1).any() else None}
        # --- какие критерии дают зону при текущем правиле
        cur = zones[0]
        region_out["current"]["by_criterion_rows"] = {
            c: int((reg["crits"][c]["dist"] <= margins[c] + TIE_EPS).sum()) for c in crits}
        region_out["current"]["n_flag_in_zone"] = int((cur & (flag == 1)).sum())
        res["regions"][region] = region_out
        # --- по критериям
        for c in crits:
            d = reg["crits"][c]
            cz = [d["dist"] <= d["margin"] + TIE_EPS] + [d["dist"] <= dd + TIE_EPS for dd in DELTA_GRID]
            cp = [metrics_from_counts(counts(d["y"], d["flag"], z)) for z in cz]
            res["criteria"][c] = {"region": region, "name": CRIT_NAME[c], "threshold": d["thr"], "margin": d["margin"],
                                  "n": int(len(d["y"])), "n_pos": int(d["y"].sum()),
                                  "n_pos_studies": int(len(np.unique(g[d["y"] == 1]))),
                                  "current": cp[0],
                                  "score_band": [dict(delta=DELTA_GRID[i], **cp[i + 1]) for i in range(len(DELTA_GRID))]}
    res["recommendation"] = recommend(res)
    if n_boot > 0:
        for region, rc in res["recommendation"]["by_region"].items():
            reg = regs[region]
            cur_zone = zone_score_band(reg, None)
            for fam in ("score_band", "qp_band"):
                k = rc.get(fam)
                if not k:
                    continue
                zb = zone_score_band(reg, k["delta"]) if fam == "score_band" else zone_qp_band(reg, k["delta"])
                k["paired_vs_current"] = paired_bootstrap_diff(reg["y"], reg["flag"], reg["groups"], cur_zone, zb, n_boot, seed)
                pd_ = k["paired_vs_current"]["sensitivity_confident"]
                if pd_["ci95"] is not None and k["verdict"].startswith("прирост"):
                    k["verdict"] = ("прирост полноты среди уверенных: парный ДИ 95 % разности не содержит 0"
                                    if pd_["ci95"][0] > 0 else
                                    "прирост в пределах шума: парный ДИ 95 % разности содержит 0, данных для решения мало")
    if regress_path is not None and Path(regress_path).exists():
        res["regress"] = project_regress(Path(regress_path), margins)
    return res


def _candidate(cur: dict, best: dict, family: str) -> dict:
    d_rows = best["n_uncertain"] - cur["n_uncertain"]
    d_sens = (best["sensitivity_confident"] or 0.0) - (cur["sensitivity_confident"] or 0.0)
    d_rwr = (best["recall_with_review"] or 0.0) - (cur["recall_with_review"] or 0.0)
    ci_cur = (cur.get("ci95") or {}).get("sensitivity_confident")
    ci_best = (best.get("ci95") or {}).get("sensitivity_confident")
    overlap = None
    if ci_cur and ci_best:
        overlap = not (ci_best[0] > ci_cur[1] or ci_cur[0] > ci_best[1])
    if d_sens <= 1e-9 and d_rwr <= 1e-9:
        verdict = "текущее правило не хуже"
    elif overlap in (True, None):
        verdict = "прирост в пределах 95 % ДИ текущего правила: данных для решения мало"
    else:
        verdict = "прирост за пределами 95 % ДИ текущего правила"
    return {
        "family": family, "delta": best["delta"], "share_uncertain": best["share_uncertain"], "n_uncertain": best["n_uncertain"],
        "sensitivity_confident": best["sensitivity_confident"], "ci95_sensitivity_confident": ci_best,
        "specificity_confident": best["specificity_confident"], "f1_confident": best["f1_confident"],
        "recall_with_review": best["recall_with_review"], "n_fn_in_zone": best["n_fn_in_zone"],
        "fn_in_zone_excess": best["fn_in_zone_excess"],
        "current": {"share_uncertain": cur["share_uncertain"], "n_uncertain": cur["n_uncertain"],
                    "sensitivity_confident": cur["sensitivity_confident"], "ci95_sensitivity_confident": ci_cur,
                    "specificity_confident": cur["specificity_confident"], "f1_confident": cur["f1_confident"],
                    "recall_with_review": cur["recall_with_review"], "n_fn_in_zone": cur["n_fn_in_zone"],
                    "fn_in_zone_excess": cur["fn_in_zone_excess"]},
        "delta_rows_to_zone": int(d_rows), "delta_sensitivity_confident": float(d_sens),
        "delta_recall_with_review": float(d_rwr),
        "rows_per_pp_recall_with_review": (float(d_rows / (100 * d_rwr)) if d_rwr > 1e-9 and d_rows > 0 else None),
        "ci_overlap_with_current": overlap, "verdict": verdict,
    }


def recommend(res: dict) -> dict:
    """Кандидатные точки по формальному правилу (решение — за человеком).

    Критерий отбора: среди точек с долей «не уверен» <= review_cap берётся точка с наибольшим числом пропусков
    (FN) в зоне сверх случайного выбора того же числа строк (fn_in_zone_excess); при равенстве — меньшее δ.
    Смысл: зона полезна ровно настолько, насколько она собирает пропуски плотнее, чем случайная выборка строк.
    Считается для двух семейств: полоса по скорам критериев (как текущее правило) и полоса по quality_prob."""
    out = {"rule": f"среди δ с долей «не уверен» <= {REVIEW_CAP:.0%}: максимум пропусков в зоне сверх случайного выбора "
                   f"того же числа строк (fn_in_zone_excess), при равенстве — меньшее δ; цена — строки в зону, выигрыш — "
                   f"полнота среди уверенных и полнота с учётом зоны", "by_region": {}}
    for region, r in res["regions"].items():
        cur = r["current"]
        rec = {"current": {"n_uncertain": cur["n_uncertain"], "share_uncertain": cur["share_uncertain"],
                           "sensitivity_confident": cur["sensitivity_confident"],
                           "ci95_sensitivity_confident": (cur.get("ci95") or {}).get("sensitivity_confident"),
                           "recall_with_review": cur["recall_with_review"], "n_fn_in_zone": cur["n_fn_in_zone"],
                           "fn_in_zone_excess": cur["fn_in_zone_excess"]}}
        for fam in ("score_band", "qp_band"):
            pts = r.get(fam) or []
            cands = [p for p in pts if p["share_uncertain"] is not None and p["share_uncertain"] <= REVIEW_CAP + 1e-9
                     and p["fn_in_zone_excess"] is not None]
            if not cands:
                rec[fam] = None
                continue
            best = max(cands, key=lambda p: (round(p["fn_in_zone_excess"], 6), -p["delta"]))
            rec[fam] = _candidate(cur, best, fam)
        out["by_region"][region] = rec
    return out


def project_regress(path: Path, margins_240: Dict[str, float]) -> dict:
    """Проекция на выгрузку боевого прогона: сколько строк было бы «не уверен» при каждом δ.
    Читаются только колонки <crit>_margin (= |score − threshold| того прогона), needs_review и quality_prob."""
    import pandas as pd
    df = pd.read_csv(path)
    n = len(df)
    out = {"file": path.name, "n_rows": int(n), "note": "скоры и пороги боевой сборки того прогона; для sp_pos это модель "
                                                        "2.3.2 (в 2.4.0 переобучена), проекция приближённая",
           "n_needs_review_in_file": int(pd.to_numeric(df.get("needs_review"), errors="coerce").fillna(0).sum())
           if "needs_review" in df else None}
    # запас, фактически применённый в том прогоне: восстанавливаем по колонкам *_uncertain
    applied = {}
    dist_by_crit = {}
    for region, mp in REGRESS_CRITS.items():
        for base, cols in mp.items():
            d_all = np.full(n, np.nan)
            u_all = np.zeros(n, bool)
            for col in cols:
                if f"{col}_margin" not in df:
                    continue
                d = pd.to_numeric(df[f"{col}_margin"], errors="coerce").values
                m = ~np.isnan(d)
                d_all[m] = d[m]
                if f"{col}_uncertain" in df:
                    u_all |= (pd.to_numeric(df[f"{col}_uncertain"], errors="coerce").fillna(0).values.astype(int) == 1)
            dist_by_crit[base] = d_all
            has = ~np.isnan(d_all)
            lo = float(np.nanmax(d_all[u_all])) if u_all.any() else 0.0
            hi = float(np.nanmin(d_all[has & ~u_all])) if (has & ~u_all).any() else None
            applied[base] = {"n_rows": int(has.sum()), "n_uncertain": int(u_all.sum()),
                             "margin_bounds": [lo, hi], "margin_2_4_0": margins_240.get(base)}
    out["applied_in_file"] = applied

    def rows_for(margin_fn) -> dict:
        res = {}
        total = 0
        for region, mp in REGRESS_CRITS.items():
            unc = np.zeros(n, bool); has = np.zeros(n, bool)
            for base in mp:
                d = dist_by_crit.get(base)
                if d is None:
                    continue
                m = margin_fn(base)
                if m is None:
                    continue
                has |= ~np.isnan(d)
                unc |= (~np.isnan(d)) & (d <= m + TIE_EPS)
            res[region] = {"n_rows": int(has.sum()), "n_uncertain": int(unc.sum()),
                           "share_uncertain": _safe(int(unc.sum()), int(has.sum()))}
            total += int(unc.sum())
        res["total_uncertain"] = total
        res["total_share"] = _safe(total, n)
        return res

    # текущее правило, восстановленное из файла (нижняя граница запаса) — должно дать needs_review файла
    out["rule_in_file"] = rows_for(lambda b: applied[b]["margin_bounds"][0] if b in applied else None)
    out["rule_2_4_0_margins"] = rows_for(lambda b: margins_240.get(b))
    out["score_band"] = [dict(delta=d, **rows_for(lambda b, d=d: d)) for d in DELTA_GRID]
    qp = None
    if "quality_prob" in df:
        qp = pd.to_numeric(df["quality_prob"], errors="coerce").values
    else:
        # 9 колонок того же прогона лежат рядом: <stem без _debug>.csv, тот же порядок строк (сверка по имени файла)
        sib = path.with_name(path.name.replace("_debug", ""))
        if sib.exists() and sib != path:
            try:
                nine = pd.read_csv(sib, sep=None, engine="python")
                if len(nine) == n and "quality_prob" in nine and "file" in df:
                    same = (nine["path_to_study"].astype(str).str.split("/").str[-1].values
                            == df["file"].astype(str).str.split("/").str[-1].values).all()
                    if same:
                        qp = pd.to_numeric(nine["quality_prob"], errors="coerce").values
                        out["quality_prob_source"] = sib.name
            except Exception:  # noqa: BLE001
                qp = None
    if qp is not None:
        out["qp_band"] = [{"delta": d, "n_uncertain": int(np.nansum(np.abs(qp - 0.5) <= d + TIE_EPS))} for d in DELTA_GRID]
    return out


# ----------------------------------------------------------------------------- SVG без внешних библиотек
def _esc(t: str) -> str:
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _pct(v) -> str:
    return "—" if v is None else f"{100 * v:.1f} %"


def curve_svg(title: str, panels: Sequence[dict], footer: str, width_panel: int = 400) -> str:
    """Панели в ряд: x — доля строк в зоне «не уверен» (0…1), y — метрика (0…1). Каждая панель: список серий
    {name, color, pts:[(x, y)], band:[(x, lo, hi)] | None, dash}, точка текущего правила (cur) и подписи δ."""
    W, H = width_panel, 445
    pad_l, pad_t, plot = 62, 84, 260
    total_w = W * len(panels) + 20
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{H}" viewBox="0 0 {total_w} {H}" '
             'font-family="DejaVu Sans, Arial, sans-serif" font-size="11">',
             f'<rect x="0" y="0" width="{total_w}" height="{H}" fill="white"/>',
             f'<text x="10" y="20" font-size="14" font-weight="bold">{_esc(title)}</text>']
    for i, pn in enumerate(panels):
        ox = 10 + i * W + pad_l
        oy = pad_t
        sx = lambda v: ox + v * plot  # noqa: E731
        sy = lambda v: oy + plot - v * plot  # noqa: E731
        parts.append(f'<text x="{ox}" y="{oy - 44}" font-size="12">{_esc(pn["title"])}</text>')
        for li, line in enumerate(str(pn.get("subtitle", "")).split("\n")[:2]):
            parts.append(f'<text x="{ox}" y="{oy - 29 + 13 * li}" font-size="10" fill="#444">{_esc(line)}</text>')
        parts.append(f'<rect x="{ox}" y="{oy}" width="{plot}" height="{plot}" fill="none" stroke="#333"/>')
        for k in range(1, 5):
            v = k / 5
            parts.append(f'<line x1="{sx(v):.1f}" y1="{oy}" x2="{sx(v):.1f}" y2="{oy + plot}" stroke="#eee"/>')
            parts.append(f'<line x1="{ox}" y1="{sy(v):.1f}" x2="{ox + plot}" y2="{sy(v):.1f}" stroke="#eee"/>')
            parts.append(f'<text x="{sx(v):.1f}" y="{oy + plot + 12}" text-anchor="middle" font-size="9">{100 * v:.0f} %</text>')
            parts.append(f'<text x="{ox - 4}" y="{sy(v) + 3:.1f}" text-anchor="end" font-size="9">{v:.1f}</text>')
        parts.append(f'<text x="{ox - 4}" y="{sy(0) + 3:.1f}" text-anchor="end" font-size="9">0.0</text>')
        parts.append(f'<text x="{ox - 4}" y="{sy(1) + 3:.1f}" text-anchor="end" font-size="9">1.0</text>')
        # ориентир REVIEW_CAP
        parts.append(f'<line x1="{sx(REVIEW_CAP):.1f}" y1="{oy}" x2="{sx(REVIEW_CAP):.1f}" y2="{oy + plot}" '
                     'stroke="#999" stroke-dasharray="4 3"/>')
        for ser in pn["series"]:
            if ser.get("band"):
                up = " ".join(f'{sx(x):.1f},{sy(hi):.1f}' for x, lo, hi in ser["band"])
                dn = " ".join(f'{sx(x):.1f},{sy(lo):.1f}' for x, lo, hi in reversed(ser["band"]))
                parts.append(f'<polygon points="{up} {dn}" fill="{ser["color"]}" fill-opacity="0.12" stroke="none"/>')
            pts = [(x, yv) for x, yv in ser["pts"] if yv is not None]
            if pts:
                s = " ".join(f'{sx(x):.1f},{sy(yv):.1f}' for x, yv in pts)
                dash = ' stroke-dasharray="5 3"' if ser.get("dash") else ""
                parts.append(f'<polyline points="{s}" fill="none" stroke="{ser["color"]}" stroke-width="1.6"{dash}/>')
                for x, yv in pts:
                    parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(yv):.1f}" r="2" fill="{ser["color"]}"/>')
            for x, yv, lab in ser.get("labels", []):
                if yv is not None:
                    parts.append(f'<text x="{sx(x) + 4:.1f}" y="{sy(yv) - 4:.1f}" font-size="8" fill="{ser["color"]}">{_esc(lab)}</text>')
        cur = pn.get("cur")
        if cur and cur[1] is not None:
            parts.append(f'<rect x="{sx(cur[0]) - 5:.1f}" y="{sy(cur[1]) - 5:.1f}" width="10" height="10" fill="none" '
                         'stroke="#111" stroke-width="1.8"/>')
            parts.append(f'<text x="{sx(cur[0]) + 8:.1f}" y="{sy(cur[1]) + 12:.1f}" font-size="9" fill="#111">текущее правило</text>')
        parts.append(f'<text x="{ox + plot / 2:.1f}" y="{oy + plot + 26}" text-anchor="middle" font-size="10">'
                     'доля строк в зоне «не уверен» (1 − покрытие)</text>')
        parts.append(f'<text transform="translate({ox - 44},{oy + plot / 2:.1f}) rotate(-90)" text-anchor="middle" '
                     f'font-size="10">{_esc(pn.get("ylabel", "полнота среди уверенных строк"))}</text>')
        # легенда
        ly = oy + plot + 44
        lx = ox
        for ser in pn["series"]:
            parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 16}" y2="{ly}" stroke="{ser["color"]}" stroke-width="2"'
                         f'{" stroke-dasharray=\"5 3\"" if ser.get("dash") else ""}/>')
            parts.append(f'<text x="{lx + 20}" y="{ly + 3}" font-size="9">{_esc(ser["name"])}</text>')
            lx += 24 + 6 * len(ser["name"])
            if lx > ox + plot - 60:
                lx = ox; ly += 14
    parts.append(f'<text x="10" y="{H - 6}" font-size="9" fill="#666">{_esc(footer)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _rel(f: Path) -> str:
    try:
        return str(f.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(f)


def write_svgs(res: dict, out_dir: Path) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for region, r in res["regions"].items():
        sb = r["score_band"]
        cur = r["current"]
        xs = [p["share_uncertain"] for p in sb]
        lab_pts = [(p["share_uncertain"], p["sensitivity_confident"], f'δ={p["delta"]:.2f}') for p in sb
                   if p["delta"] in (0.05, 0.10, 0.20, 0.30)]
        band = [(p["share_uncertain"], p["ci95"]["sensitivity_confident"][0], p["ci95"]["sensitivity_confident"][1])
                for p in sb if p.get("ci95") and p["ci95"].get("sensitivity_confident")]
        p1 = {"title": "Строки региона: полоса по скорам критериев",
              "subtitle": f'n={r["n"]}, положительных {r["n_pos"]} ({r["n_pos_studies"]} исследований)\nполоса — ДИ 95 %, бутстрап по исследованиям',
              "series": [
                  {"name": "полнота среди уверенных", "color": "#1f6fb2",
                   "pts": list(zip(xs, [p["sensitivity_confident"] for p in sb])), "band": band, "labels": lab_pts},
                  {"name": "полнота с учётом зоны (врач смотрит зону)", "color": "#2a9d4b", "dash": True,
                   "pts": list(zip(xs, [p["recall_with_review"] for p in sb]))},
                  {"name": "специфичность среди уверенных", "color": "#d9541e", "dash": True,
                   "pts": list(zip(xs, [p["specificity_confident"] for p in sb]))}],
              "cur": (cur["share_uncertain"], cur["sensitivity_confident"])}
        p2 = {"title": "Строки региона: F1 среди уверенных и доля ошибок в зоне",
              "subtitle": f'ошибок класса на OOF {cur["n_errors"]} из {r["n"]}\nвертикальный пунктир — ориентир {REVIEW_CAP:.0%} строк на просмотр',
              "ylabel": "F1 среди уверенных / доля ошибок в зоне",
              "series": [
                  {"name": "F1 среди уверенных", "color": "#1f6fb2", "pts": list(zip(xs, [p["f1_confident"] for p in sb]))},
                  {"name": "доля всех ошибок, попавших в зону", "color": "#7b3fa0", "dash": True,
                   "pts": list(zip(xs, [p["share_errors_in_zone"] for p in sb]))},
                  {"name": "доля положительных в зоне", "color": "#b8860b", "dash": True,
                   "pts": list(zip(xs, [p["share_pos_in_zone"] for p in sb]))}],
              "cur": (cur["share_uncertain"], cur["f1_confident"])}
        panels = [p1, p2]
        if r.get("qp_band"):
            qb = r["qp_band"]
            panels.append({"title": "Полоса по quality_prob вокруг 0.5 (для сравнения)",
                           "subtitle": "в сервисе не используется\nсогласование с классом оставляет разрыв вокруг 0.5",
                           "series": [
                               {"name": "полнота среди уверенных", "color": "#1f6fb2",
                                "pts": [(p["share_uncertain"], p["sensitivity_confident"]) for p in qb],
                                "labels": [(p["share_uncertain"], p["sensitivity_confident"], f'δ={p["delta"]:.2f}')
                                           for p in qb if p["delta"] in (0.20, 0.25, 0.30)]},
                               {"name": "доля всех ошибок в зоне", "color": "#7b3fa0", "dash": True,
                                "pts": [(p["share_uncertain"], p["share_errors_in_zone"]) for p in qb]}]})
        svg = curve_svg(f"Зона «не уверен»: покрытие → полнота (OOF) — {r['name']}", panels,
                        "точки — δ от 0.00 до 0.30 шагом 0.01; квадрат — текущее правило (запасы по критериям из config.yaml); "
                        "полоса — 95 % ДИ полноты среди уверенных; только агрегаты, без кадров")
        f = out_dir / f"coverage_recall_{region}.svg"
        f.write_text(svg, encoding="utf-8"); files.append(_rel(f))
    # критерии: одна панель на критерий
    panels = []
    colors = {"sp_pos": "#1f6fb2", "sp_axis": "#d9541e", "sp_art": "#2a9d4b", "hip_pos": "#7b3fa0", "hip_roi": "#b8860b"}
    for crit, c in res["criteria"].items():
        sb = c["score_band"]
        xs = [p["share_uncertain"] for p in sb]
        panels.append({"title": f'{crit}: {c["name"]}',
                       "subtitle": f'n={c["n"]}, положительных {c["n_pos"]} ({c["n_pos_studies"]} исследований)\nтекущий запас {c["margin"]:.4g}',
                       "series": [
                           {"name": "полнота среди уверенных", "color": colors[crit],
                            "pts": list(zip(xs, [p["sensitivity_confident"] for p in sb])),
                            "labels": [(p["share_uncertain"], p["sensitivity_confident"], f'δ={p["delta"]:.2f}')
                                       for p in sb if p["delta"] in (0.05, 0.10, 0.20, 0.30)]},
                           {"name": "доля ошибок критерия в зоне", "color": "#555", "dash": True,
                            "pts": list(zip(xs, [p["share_errors_in_zone"] for p in sb]))}],
                       "cur": (c["current"]["share_uncertain"], c["current"]["sensitivity_confident"])})
    svg = curve_svg("Зона «не уверен» по критериям: покрытие → полнота (OOF)", panels,
                    "точки — δ от 0.00 до 0.30 шагом 0.01; квадрат — текущий запас критерия; только агрегаты, без кадров",
                    width_panel=370)
    f = out_dir / "coverage_recall_criteria.svg"
    f.write_text(svg, encoding="utf-8"); files.append(_rel(f))
    return files


# ----------------------------------------------------------------------------- таблицы для документа
def _f(v, nd=3) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def _ci(ci, key, nd=2) -> str:
    if not ci or not ci.get(key):
        return ""
    return f" [{ci[key][0]:.{nd}f}; {ci[key][1]:.{nd}f}]"


def markdown_tables(res: dict) -> str:
    L = []
    for region, r in res["regions"].items():
        L.append(f"### {r['name']} — строки региона (n = {r['n']}, положительных {r['n_pos']} в {r['n_pos_studies']} исследованиях)\n")
        L.append("| Правило | δ | «не уверен» | полнота среди уверенных [ДИ] | специфичность | F1 | положительных в зоне | ошибок в зоне | пропусков в зоне | полнота с учётом зоны |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")

        def row(name, d, p):
            L.append(f"| {name} | {d} | {p['n_uncertain']}/{p['n']} ({_pct(p['share_uncertain'])}) | "
                     f"{_f(p['sensitivity_confident'])}{_ci(p.get('ci95'), 'sensitivity_confident')} | "
                     f"{_f(p['specificity_confident'])} | {_f(p['f1_confident'])} | {p['n_pos_in_zone']}/{p['n_pos']} | "
                     f"{p['n_errors_in_zone']}/{p['n_errors']} | {p['n_fn_in_zone']}/{p['n_fn_total']} | {_f(p['recall_with_review'])} |")
        row("текущее (запасы по критериям)", "—", r["current"])
        for p in r["score_band"]:
            if p["delta"] in KEY_DELTAS:
                row("полоса по скорам", f"{p['delta']:.2f}", p)
        if r.get("qp_band"):
            for p in r["qp_band"]:
                if p["delta"] in (0.10, 0.20, 0.25, 0.30):
                    row("полоса по quality_prob", f"{p['delta']:.2f}", p)
        L.append("")
    L.append("### По критериям (флаг критерия против разметки)\n")
    L.append("| Критерий | правило | δ | «не уверен» | полнота среди уверенных | специфичность | F1 | ошибок критерия в зоне | пропусков в зоне |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for crit, c in res["criteria"].items():
        p = c["current"]
        L.append(f"| `{crit}` | текущий запас {c['margin']:.4g} | — | {p['n_uncertain']}/{p['n']} ({_pct(p['share_uncertain'])}) | "
                 f"{_f(p['sensitivity_confident'])} | {_f(p['specificity_confident'])} | {_f(p['f1_confident'])} | "
                 f"{p['n_errors_in_zone']}/{p['n_errors']} | {p['n_fn_in_zone']}/{p['n_fn_total']} |")
        for p in c["score_band"]:
            if p["delta"] in (0.02, 0.05, 0.10, 0.20):
                L.append(f"| `{crit}` | полоса | {p['delta']:.2f} | {p['n_uncertain']}/{p['n']} ({_pct(p['share_uncertain'])}) | "
                         f"{_f(p['sensitivity_confident'])} | {_f(p['specificity_confident'])} | {_f(p['f1_confident'])} | "
                         f"{p['n_errors_in_zone']}/{p['n_errors']} | {p['n_fn_in_zone']}/{p['n_fn_total']} |")
    L.append("")
    if res.get("regress"):
        rg = res["regress"]
        L.append(f"### Проекция на {rg['file']} (n = {rg['n_rows']})\n")
        L.append(f"needs_review в файле: {rg['n_needs_review_in_file']}; правило файла восстановлено: "
                 f"{rg['rule_in_file']['total_uncertain']}; запасы 2.4.0 на тех же скорах: {rg['rule_2_4_0_margins']['total_uncertain']}\n")
        L.append("| δ | позвоночник | бедро | всего | доля | по quality_prob |")
        L.append("|---|---|---|---|---|---|")
        qmap = {p["delta"]: p["n_uncertain"] for p in rg.get("qp_band", [])}
        for p in rg["score_band"]:
            if p["delta"] in KEY_DELTAS:
                L.append(f"| {p['delta']:.2f} | {p['spine']['n_uncertain']}/{p['spine']['n_rows']} | {p['hip']['n_uncertain']}/{p['hip']['n_rows']} | "
                         f"{p['total_uncertain']} | {_pct(p['total_share'])} | {qmap.get(p['delta'], '—')} |")
        L.append("")
    L.append("### Кандидатные точки по формальному правилу\n")
    L.append(res["recommendation"]["rule"] + "\n")
    fam_name = {"score_band": "полоса по скорам критериев", "qp_band": "полоса по quality_prob"}
    for region, rc in res["recommendation"]["by_region"].items():
        c = rc["current"]
        L.append(f"- {REGION_NAME[region]}, сейчас: «не уверен» {c['n_uncertain']} строк ({_pct(c['share_uncertain'])}), полнота среди "
                 f"уверенных {_f(c['sensitivity_confident'])}{_ci({'s': c['ci95_sensitivity_confident']}, 's')}, с учётом зоны "
                 f"{_f(c['recall_with_review'])}, пропусков в зоне {c['n_fn_in_zone']} (сверх случайного {c['fn_in_zone_excess']:+.1f}).")
        for fam in ("score_band", "qp_band"):
            k = rc.get(fam)
            if not k:
                continue
            L.append(f"  - {fam_name[fam]}: δ = {k['delta']:.2f} — «не уверен» {k['n_uncertain']} ({_pct(k['share_uncertain'])}), "
                     f"полнота среди уверенных {_f(k['sensitivity_confident'])}{_ci({'s': k['ci95_sensitivity_confident']}, 's')} "
                     f"({k['delta_sensitivity_confident']:+.3f}), F1 {_f(k['f1_confident'])}, с учётом зоны {_f(k['recall_with_review'])} "
                     f"({k['delta_recall_with_review']:+.3f}), пропусков в зоне {k['n_fn_in_zone']} (сверх случайного {k['fn_in_zone_excess']:+.1f}); "
                     f"цена {k['delta_rows_to_zone']:+d} строк"
                     + (f" ({k['rows_per_pp_recall_with_review']:.1f} строки за 1 п.п. полноты с учётом зоны)" if k['rows_per_pp_recall_with_review'] else "")
                     + (f"; парная разность полноты среди уверенных {k['paired_vs_current']['sensitivity_confident']['diff']:+.3f}"
                        f" [{k['paired_vs_current']['sensitivity_confident']['ci95'][0]:+.2f}; {k['paired_vs_current']['sensitivity_confident']['ci95'][1]:+.2f}]"
                        if k.get("paired_vs_current") else "")
                     + f"; вывод: {k['verdict']}.")
    return "\n".join(L)


def _round(o, nd=6):
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, list):
        return [_round(v, nd) for v in o]
    return o


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None, help="каталог для JSON и SVG (по умолчанию docs/uncertain_zone.json и docs/uncertain_zone/)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=BOOT_SEED)
    ap.add_argument("--regress", default=None, help="debug-CSV боевого прогона для проекции (например dataset/regress_2_3_2_debug.csv)")
    ap.add_argument("--no-svg", action="store_true")
    ap.add_argument("--markdown", action="store_true", help="напечатать таблицы для docs/UNCERTAIN_ZONE.md")
    a = ap.parse_args(argv)

    res = build(n_boot=a.n_boot, seed=a.seed, regress_path=Path(a.regress) if a.regress else None)
    if a.out:
        out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "uncertain_zone.json"; svg_dir = out_dir
    else:
        json_path = DOCS_JSON; svg_dir = DOCS_SVG_DIR
    res["svg_files"] = write_svgs(res, svg_dir) if not a.no_svg else []
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(_round(res), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if a.markdown:
        print(markdown_tables(res))
    print(f"записано: {json_path}; SVG: {len(res['svg_files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
