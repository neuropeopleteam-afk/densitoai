#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Отчёт о калибровке вероятностей на OOF-предсказаниях (идея «г», 23.09.2026).

Цель: честно показать, насколько `quality_prob` и скоры критериев можно читать как вероятности,
НЕ меняя чисел поставки (модели, пороги, config.yaml, 9 колонок CSV не затрагиваются).

Что оценивается
  Области (бинарная задача «есть нарушение», ровно как в src/eval_oof_metrics.py и src/inference.py):
    final      — quality_prob_oof: 0.5 * OOF any-модели (geom+emb, StratifiedGroupKFold(5), 3 сида)
                 + 0.5 * max OOF-скоров критериев, затем согласование с классом
                 (class=1 -> 0.5 + 0.5 * raw; class=0 -> min(0.5 * raw, 0.499999)).
    raw_blend  — та же смесь до согласования с классом.
  Критерии (sp_pos, sp_axis, sp_art, hip_pos, hip_roi), из models/oof_stacked_<region>_<crit>.csv:
    score          — стэкнутый ранговый скор `oof_stacked` (0.5 * rank_geom + 0.5 * rank_emb); он и
                     сравнивается с порогом, но по построению НЕ является вероятностью;
    p_cal          — Platt из models/calibration.pkl (`<crit>_p_cal` в debug-CSV и API details);
                     параметры подобраны на всём OOF, поэтому оценка на том же OOF оптимистична (in-sample);
    p_cal_crossfit — Platt, обученный cross-fit (GroupKFold(5) по исследованию, зерно 0): честная оценка.

Метрики на каждую пару (цель, вариант)
  диаграмма надёжности: 10 бинов равной ширины и 5 квантильных; ECE и MCE для обеих разбивок;
  Brier (и Brier константного предиктора = доля позитивов, skill = 1 - Brier/Brier_const);
  лог-лосс (обрезка 1e-6); наклон/сдвиг калибровки (логистическая рекалибровка y ~ a + b * logit(p),
  идеал b=1, a=0; IRLS с гребнем 1e-6); ROC-AUC для справки;
  95 % ДИ бутстрапом по исследованиям (500 повторов, зерно 0) для ECE(10) и Brier.
  Доля предсказаний в зоне «не уверен»: |score - threshold| <= margin, margin — config.yaml
  `uncertainty.margin_by_criterion` (при null — models/calibration.pkl); для областей — доля строк, у которых
  «не уверен» хотя бы один критерий.

Выход (без кадров пациентов; в JSON только агрегаты):
  docs/calibration/calibration.json, docs/calibration/*.svg (SVG пишется вручную, без matplotlib),
  docs/CALIBRATION.md формируется отдельно (таблицы копируются из JSON / stdout этого скрипта).

Запуск:
  OMP_NUM_THREADS=1 python tools/calibration_report.py             # полный отчёт
  python tools/calibration_report.py --out DIR --n-boot 100       # другой каталог / меньше бутстрапа
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
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
MODELS_DIR = ROOT / "models"
DOCS_OUT = ROOT / "docs" / "calibration"

N_BOOT = 500
BOOT_SEED = 0
N_BINS_EQ = 10
N_BINS_Q = 5
EPS = 1e-6

REGION_CRITERIA = {"spine": ["sp_pos", "sp_axis", "sp_art"], "hip": ["hip_pos", "hip_roi"]}
REGION_NAME = {"spine": "Поясничный отдел позвоночника", "hip": "Проксимальный отдел бедра"}
CRIT_NAME = {"sp_pos": "Некорректная укладка (позвоночник)", "sp_axis": "Не выравнена ось позвоночника",
             "sp_art": "Присутствуют посторонние предметы", "hip_pos": "Некорректная укладка (бедро)",
             "hip_roi": "Некорректная область интереса"}
VARIANT_NAME = {"final": "quality_prob (итоговый, после согласования с классом)",
                "raw_blend": "смесь any-модели и max критериев (до согласования)",
                "score": "ранговый скор (сравнивается с порогом)",
                "p_cal": "Platt из calibration.pkl (in-sample)",
                "p_cal_crossfit": "Platt cross-fit (GroupKFold 5, зерно 0)"}


# ----------------------------------------------------------------------------- базовые метрики
def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, float), EPS, 1.0 - EPS)


def logit(p: np.ndarray) -> np.ndarray:
    p = _clip(p)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float)
    return 1.0 / (1.0 + np.exp(-z))


def bin_edges_equal(n_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1)


def bin_edges_quantile(p: np.ndarray, n_bins: int) -> np.ndarray:
    """Границы квантильных бинов; дубли (связи в ранговых скорах) убираются, число бинов может уменьшиться."""
    p = np.asarray(p, float)
    qs = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    qs[0], qs[-1] = 0.0, 1.0
    return np.unique(qs)


def assign_bins(p: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Бин i: edges[i] <= p < edges[i+1]; последний бин включает правую границу."""
    p = np.asarray(p, float)
    n = len(edges) - 1
    return np.clip(np.digitize(p, edges[1:-1], right=False), 0, n - 1)


def reliability_table(y: np.ndarray, p: np.ndarray, edges: np.ndarray) -> List[dict]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = assign_bins(p, edges)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        rows.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()),
                     "n_pos": int(y[m].sum()) if m.any() else 0,
                     "mean_pred": float(p[m].mean()) if m.any() else None,
                     "frac_pos": float(y[m].mean()) if m.any() else None,
                     "gap": float(abs(y[m].mean() - p[m].mean())) if m.any() else None})
    return rows


def ece_mce(y: np.ndarray, p: np.ndarray, edges: np.ndarray) -> Tuple[float, float]:
    """ECE = сумма (n_b / n) * |frac_pos_b - mean_pred_b|; MCE = максимум |...| по непустым бинам."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = assign_bins(p, edges)
    n = len(p)
    ece, mce = 0.0, 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            gap = abs(float(y[m].mean()) - float(p[m].mean()))
            ece += m.sum() / n * gap
            mce = max(mce, gap)
    return float(ece), float(mce)


def ece_equal(y, p, n_bins: int = N_BINS_EQ) -> float:
    return ece_mce(y, p, bin_edges_equal(n_bins))[0]


def brier(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def log_loss(y, p) -> float:
    y, p = np.asarray(y, float), _clip(p)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def roc_auc(y, s) -> Optional[float]:
    """ROC-AUC через ранги (связи — средним рангом), без sklearn."""
    y, s = np.asarray(y, int), np.asarray(s, float)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def logistic_recalibration(y, p, ridge: float = 1e-6, max_iter: int = 100) -> Tuple[float, float]:
    """Наклон b и сдвиг a модели P(y=1) = sigmoid(a + b * logit(p)); IRLS (Ньютон) с малым гребнем.
    Идеально калиброванный предиктор: b = 1, a = 0. b < 1 — предиктор слишком уверен (переразброс),
    b > 1 — недоразброс; a != 0 при b = 1 — систематический сдвиг уровня."""
    y = np.asarray(y, float)
    x = logit(p)
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    for _ in range(max_iter):
        z = X @ w
        mu = sigmoid(z)
        W = mu * (1.0 - mu) + 1e-12
        H = X.T @ (X * W[:, None]) + ridge * np.eye(2)
        g = X.T @ (y - mu) - ridge * w
        step = np.linalg.solve(H, g)
        w = w + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(w[1]), float(w[0])


def fit_platt_1d(s, y, ridge: float = 1e-6) -> Tuple[float, float]:
    """Platt p = sigmoid(a * s + b) на самом скоре (не на logit): как в src/calibration_utils.fit_platt по смыслу."""
    y = np.asarray(y, float)
    X = np.column_stack([np.asarray(s, float), np.ones(len(y))])
    w = np.zeros(2)
    for _ in range(200):
        mu = sigmoid(X @ w)
        W = mu * (1.0 - mu) + 1e-12
        H = X.T @ (X * W[:, None]) + ridge * np.eye(2)
        g = X.T @ (y - mu) - ridge * w
        step = np.linalg.solve(H, g)
        w = w + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(w[0]), float(w[1])


def group_kfold_indices(groups: np.ndarray, n_splits: int, seed: int, y: Optional[np.ndarray] = None) -> List[np.ndarray]:
    """Детерминированный GroupKFold по исследованиям: исследования перемешиваются зерном и раскладываются по
    фолдам по кругу; при заданном y сначала идут исследования с положительной меткой (стратификация по
    исследованиям — чтобы при 6 положительных исследованиях ни один фолд не остался без позитивов в обучении)."""
    groups = np.asarray(groups)
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ug)
    if y is not None:
        y = np.asarray(y, float)
        pos_g = {g for g in ug if y[groups == g].max() > 0}
        perm = np.array([g for g in perm if g in pos_g] + [g for g in perm if g not in pos_g])
    fold_of = {g: i % n_splits for i, g in enumerate(perm)}
    f = np.array([fold_of[g] for g in groups])
    return [np.nonzero(f == k)[0] for k in range(n_splits)]


def platt_crossfit(s, y, groups, n_splits: int = 5, seed: int = BOOT_SEED) -> np.ndarray:
    s, y = np.asarray(s, float), np.asarray(y, float)
    out = np.zeros(len(s))
    for te in group_kfold_indices(np.asarray(groups), n_splits, seed, y=y):
        tr = np.setdiff1d(np.arange(len(s)), te)
        a, b = fit_platt_1d(s[tr], y[tr])
        out[te] = sigmoid(a * s[te] + b)
    return out


# ----------------------------------------------------------------------------- бутстрап по исследованиям
def bootstrap_ci(y, p, groups, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> Dict[str, List[float]]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    groups = np.asarray(groups)
    ug = np.unique(groups)
    idx_by_g = {g: np.nonzero(groups == g)[0] for g in ug}
    rng = np.random.default_rng(seed)
    eces, briers = [], []
    for _ in range(n_boot):
        pick = rng.choice(ug, len(ug), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        eces.append(ece_equal(y[idx], p[idx]))
        briers.append(brier(y[idx], p[idx]))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"ece_10": q(eces), "brier": q(briers)}


def evaluate(y, p, groups, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Все метрики калибровки для одного вектора вероятностей."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    groups = np.asarray(groups)
    e_eq = bin_edges_equal(N_BINS_EQ)
    e_q = bin_edges_quantile(p, N_BINS_Q)
    ece10, mce10 = ece_mce(y, p, e_eq)
    eceq, mceq = ece_mce(y, p, e_q)
    prev = float(y.mean())
    slope, intercept = logistic_recalibration(y, p)
    b = brier(y, p)
    b_const = brier(y, np.full_like(p, prev))
    res = {
        "n": int(len(y)), "n_pos": int(y.sum()), "prevalence": prev,
        "n_studies": int(len(np.unique(groups))),
        "n_pos_studies": int(len(np.unique(groups[y == 1]))),
        "brier": b, "brier_const": b_const,
        "brier_skill": float(1.0 - b / b_const) if b_const > 0 else None,
        "log_loss": log_loss(y, p), "log_loss_const": log_loss(y, np.full_like(p, prev)),
        "ece_10": ece10, "mce_10": mce10, "ece_q5": eceq, "mce_q5": mceq,
        "n_bins_q_effective": int(len(e_q) - 1),
        "slope": slope, "intercept": intercept,
        "mean_pred": float(p.mean()), "roc_auc": roc_auc(y.astype(int), p),
        "bins_equal_10": reliability_table(y, p, e_eq),
        "bins_quantile_5": reliability_table(y, p, e_q),
    }
    if n_boot > 0:
        res["ci95"] = bootstrap_ci(y, p, groups, n_boot=n_boot, seed=seed)
    return res


# ----------------------------------------------------------------------------- данные поставки
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
    out = {}
    mbc = ((cfg.get("uncertainty") or {}).get("margin_by_criterion") or {})
    for crit in [c for cs in REGION_CRITERIA.values() for c in cs]:
        v = mbc.get(crit)
        if v is None and calib is not None:
            v = (calib.get("margin_by_criterion") or {}).get(crit)
            if v is None:
                v = ((calib.get("criteria") or {}).get(crit) or {}).get("margin")
        out[crit] = float(v) if v is not None else 0.0
    return out


def load_oof_criterion(region: str, crit: str):
    import pandas as pd
    df = pd.read_csv(MODELS_DIR / f"oof_stacked_{region}_{crit}.csv")
    return df


def flags_from(df, s: np.ndarray, thr: float) -> np.ndarray:
    """Флаг нарушения ровно как в сервисе: pred_label из CSV, иначе сравнение с порогом с допуском."""
    import pandas as pd
    if "pred_label" in df.columns:
        pl = pd.to_numeric(df["pred_label"], errors="coerce")
        if not pl.isna().any():
            return pl.values.astype(int)
    return (s >= thr - 1e-9).astype(int)


def build_targets(n_boot: int) -> dict:
    """Собирает все цели: 2 области (final, raw_blend) и 5 критериев (score, p_cal, p_cal_crossfit)."""
    sys.path.insert(0, str(ROOT / "src"))
    import eval_oof_metrics as eom  # воспроизводит OOF any-модели ровно как отчёт метрик

    summary = json.load(open(MODELS_DIR / "metrics_summary.json", encoding="utf-8"))
    cfg = load_yaml_config()
    calib = load_calibration_pkl()
    margins = margins_from_config(cfg, calib)
    w_model = float(((cfg.get("stacking") or {}).get("any_blend_weight_model", 0.5)))
    consistent = bool(((cfg.get("stacking") or {}).get("consistent_quality_prob", True)))

    out = {"regions": {}, "criteria": {}, "uncertain_zone": {}, "inputs": {
        "oof_files": [], "thresholds": {}, "margins": margins, "platt": {},
        "any_blend_weight_model": w_model, "consistent_quality_prob": consistent}}

    for region, crits in REGION_CRITERIA.items():
        frames = {}
        for crit in crits:
            df = load_oof_criterion(region, crit)
            out["inputs"]["oof_files"].append(f"models/oof_stacked_{region}_{crit}.csv")
            thr = float(summary[region][crit]["threshold"])
            out["inputs"]["thresholds"][crit] = thr
            y = df["y_true"].values.astype(int)
            s = df["oof_stacked"].values.astype(float)
            g = df["study"].values
            flag = flags_from(df, s, thr)
            frames[crit] = (df, y, s, g, flag)
            # --- варианты критерия
            variants = {"score": s}
            if calib is not None and crit in (calib.get("criteria") or {}):
                pl = calib["criteria"][crit]["platt"]
                out["inputs"]["platt"][crit] = {"a": float(pl["a"]), "b": float(pl["b"])}
                variants["p_cal"] = sigmoid(float(pl["a"]) * s + float(pl["b"]))
            variants["p_cal_crossfit"] = platt_crossfit(s, y, g)
            folds = []
            for k, te in enumerate(group_kfold_indices(g, 5, BOOT_SEED, y=y)):
                tr = np.setdiff1d(np.arange(len(s)), te)
                a_k, b_k = fit_platt_1d(s[tr], y[tr])
                folds.append({"fold": k, "n_train_pos": int(y[tr].sum()), "n_test_pos": int(y[te].sum()),
                              "a": a_k, "b": b_k, "p_cal_at_threshold": float(sigmoid(a_k * thr + b_k))})
            out["criteria"][crit] = {"region": region, "name": CRIT_NAME[crit], "threshold": thr,
                                     "platt_crossfit_folds": folds,
                                     "variants": {k: evaluate(y, v, g, n_boot=n_boot) for k, v in variants.items()}}
            # --- зона «не уверен»
            dist = np.abs(s - thr)
            unc = dist <= margins[crit] + 1e-12
            wrong = flag != y
            out["uncertain_zone"][crit] = {
                "margin": margins[crit], "threshold": thr, "n": int(len(s)),
                "n_uncertain": int(unc.sum()), "share_uncertain": float(unc.mean()),
                "n_errors_total": int(wrong.sum()), "n_errors_in_zone": int((wrong & unc).sum()),
                "share_errors_covered": float((wrong & unc).sum() / wrong.sum()) if wrong.sum() else None,
                "p_cal_at_threshold": (float(sigmoid(out["inputs"]["platt"][crit]["a"] * thr
                                                     + out["inputs"]["platt"][crit]["b"]))
                                       if crit in out["inputs"]["platt"] else None)}
        # --- область: quality_prob_oof как в eval_oof_metrics
        base_df = frames[crits[0]][0]
        fp = base_df["file_path"].values
        flags = np.zeros(len(fp), int); ytrue = np.zeros(len(fp), int); crit_max = np.zeros(len(fp))
        unc_row = np.zeros(len(fp), bool)
        for crit in crits:
            df, y, s, g, flag = frames[crit]
            assert (df["file_path"].values == fp).all(), "OOF files misaligned"
            flags = np.maximum(flags, flag); ytrue = np.maximum(ytrue, y); crit_max = np.maximum(crit_max, s)
            unc_row |= np.abs(s - out["inputs"]["thresholds"][crit]) <= margins[crit] + 1e-12
        anym = eom.oof_any_model(region, crits).set_index("file_path").loc[fp]
        assert (anym["y_any"].values == ytrue).all(), "any-label mismatch"
        any_oof = anym["any_model_oof"].values.astype(float)
        raw = w_model * any_oof + (1.0 - w_model) * crit_max
        final = np.where(flags == 1, 0.5 + 0.5 * raw, np.minimum(0.5 * raw, 0.499999)) if consistent else raw
        g = base_df["study"].values
        out["regions"][region] = {
            "name": REGION_NAME[region],
            "variants": {"final": evaluate(ytrue, final, g, n_boot=n_boot),
                         "raw_blend": evaluate(ytrue, raw, g, n_boot=n_boot)},
            "components_roc_auc": {"any_model_only": roc_auc(ytrue, any_oof), "max_criteria_only": roc_auc(ytrue, crit_max)},
            "share_class1": float(flags.mean()),
            "row_uncertain_share": float(unc_row.mean()), "n_row_uncertain": int(unc_row.sum()),
        }
    return out


# ----------------------------------------------------------------------------- SVG без внешних библиотек
def _esc(t: str) -> str:
    return (str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def _fmt(v, nd=3) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def reliability_svg(panels: Sequence[dict], title: str, width_panel: int = 380) -> str:
    """Диаграмма надёжности: панели в ряд. Круги — бины равной ширины (радиус ~ sqrt(n)), квадраты с линией —
    квантильные бины; внизу — гистограмма предсказаний по 10 бинам. Только агрегаты, без кадров."""
    W, H = width_panel, 500
    pad_l, pad_t, plot = 60, 84, 250
    hist_h = 60
    total_w = W * len(panels) + 20
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{H}" viewBox="0 0 {total_w} {H}" '
             'font-family="DejaVu Sans, Arial, sans-serif" font-size="11">',
             f'<rect x="0" y="0" width="{total_w}" height="{H}" fill="white"/>',
             f'<text x="10" y="20" font-size="14" font-weight="bold">{_esc(title)}</text>']
    for i, pn in enumerate(panels):
        ox = 10 + i * W + pad_l
        oy = pad_t
        sx = lambda v: ox + v * plot
        sy = lambda v: oy + plot - v * plot
        parts.append(f'<text x="{ox}" y="{oy - 44}" font-size="12">{_esc(pn["title"])}</text>')
        for li, line in enumerate(pn.get("subtitle", "").split("\n")):
            parts.append(f'<text x="{ox}" y="{oy - 28 + 13 * li}" font-size="10" fill="#444">{_esc(line)}</text>')
        parts.append(f'<rect x="{ox}" y="{oy}" width="{plot}" height="{plot}" fill="none" stroke="#333"/>')
        for k in range(1, 5):
            v = k / 5
            parts.append(f'<line x1="{sx(v):.1f}" y1="{oy}" x2="{sx(v):.1f}" y2="{oy + plot}" stroke="#eee"/>')
            parts.append(f'<line x1="{ox}" y1="{sy(v):.1f}" x2="{ox + plot}" y2="{sy(v):.1f}" stroke="#eee"/>')
            parts.append(f'<text x="{sx(v):.1f}" y="{oy + plot + 12}" text-anchor="middle" font-size="9">{v:.1f}</text>')
            parts.append(f'<text x="{ox - 4}" y="{sy(v) + 3:.1f}" text-anchor="end" font-size="9">{v:.1f}</text>')
        parts.append(f'<line x1="{ox}" y1="{oy + plot}" x2="{ox + plot}" y2="{oy}" stroke="#999" stroke-dasharray="4 3"/>')
        # квантильные бины
        qb = [b for b in pn["bins_q"] if b["n"] > 0]
        if qb:
            pts = " ".join(f'{sx(b["mean_pred"]):.1f},{sy(b["frac_pos"]):.1f}' for b in qb)
            parts.append(f'<polyline points="{pts}" fill="none" stroke="#d9541e" stroke-width="1.5"/>')
            for b in qb:
                parts.append(f'<rect x="{sx(b["mean_pred"]) - 4:.1f}" y="{sy(b["frac_pos"]) - 4:.1f}" width="8" height="8" '
                             f'fill="#d9541e" stroke="white"/>')
        # бины равной ширины
        eb = [b for b in pn["bins_eq"] if b["n"] > 0]
        nmax = max([b["n"] for b in eb] + [1])
        for b in eb:
            r = 3.0 + 11.0 * (b["n"] / nmax) ** 0.5
            parts.append(f'<circle cx="{sx(b["mean_pred"]):.1f}" cy="{sy(b["frac_pos"]):.1f}" r="{r:.1f}" '
                         f'fill="#1f6fb2" fill-opacity="0.55" stroke="#1f6fb2"/>')
            parts.append(f'<text x="{sx(b["mean_pred"]) + r + 2:.1f}" y="{sy(b["frac_pos"]) + 3:.1f}" font-size="8" '
                         f'fill="#1f6fb2">{b["n"]}</text>')
        parts.append(f'<text x="{ox + plot / 2:.1f}" y="{oy + plot + 24}" text-anchor="middle" font-size="10">'
                     'средняя предсказанная вероятность</text>')
        parts.append(f'<text transform="translate({ox - 40},{oy + plot / 2:.1f}) rotate(-90)" text-anchor="middle" '
                     'font-size="10">доля положительных</text>')
        # гистограмма
        hy = oy + plot + 34
        allb = pn["bins_eq"]
        hmax = max([b["n"] for b in allb] + [1])
        bw = plot / len(allb)
        for b in allb:
            hh = hist_h * b["n"] / hmax
            parts.append(f'<rect x="{ox + b["bin"] * bw + 1:.1f}" y="{hy + hist_h - hh:.1f}" width="{bw - 2:.1f}" '
                         f'height="{hh:.1f}" fill="#888"/>')
        parts.append(f'<line x1="{ox}" y1="{hy + hist_h}" x2="{ox + plot}" y2="{hy + hist_h}" stroke="#333"/>')
        parts.append(f'<text x="{ox}" y="{hy + hist_h + 12}" font-size="9" fill="#444">'
                     f'{_esc(pn.get("footer", ""))}</text>')
    parts.append(f'<text x="10" y="{H - 6}" font-size="9" fill="#666">круги — 10 бинов равной ширины (число = n в бине); '
                 'квадраты — 5 квантильных бинов; пунктир — идеальная калибровка; внизу — распределение предсказаний</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _panel(name: str, ev: dict, n_boot: int) -> dict:
    ci = ev.get("ci95", {})
    ci_e = f' [{ci["ece_10"][0]:.2f}; {ci["ece_10"][1]:.2f}]' if ci else ""
    ci_b = f' [{ci["brier"][0]:.3f}; {ci["brier"][1]:.3f}]' if ci else ""
    return {"title": name,
            "subtitle": f'n={ev["n"]}, положительных {ev["n_pos"]} ({ev["n_pos_studies"]} исследований)\n'
                        f'ECE10 {ev["ece_10"]:.3f}{ci_e}; Brier {ev["brier"]:.3f}{ci_b}',
            "footer": f'наклон {_fmt(ev["slope"], 2)}, сдвиг {_fmt(ev["intercept"], 2)}; MCE10 {_fmt(ev["mce_10"])}; '
                      f'лог-лосс {_fmt(ev["log_loss"])}; AUC {_fmt(ev["roc_auc"])}',
            "bins_eq": ev["bins_equal_10"], "bins_q": ev["bins_quantile_5"]}


def _rel(f: Path) -> str:
    try:
        return str(f.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(f)


def write_svgs(res: dict, out_dir: Path, n_boot: int) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for region, r in res["regions"].items():
        panels = [_panel(VARIANT_NAME["final"], r["variants"]["final"], n_boot),
                  _panel(VARIANT_NAME["raw_blend"], r["variants"]["raw_blend"], n_boot)]
        svg = reliability_svg(panels, f"Надёжность quality_prob (OOF) — {r['name']}")
        f = out_dir / f"reliability_{region}.svg"
        f.write_text(svg, encoding="utf-8"); files.append(_rel(f))
    for crit, c in res["criteria"].items():
        order = [k for k in ("score", "p_cal", "p_cal_crossfit") if k in c["variants"]]
        panels = [_panel(VARIANT_NAME[k], c["variants"][k], n_boot) for k in order]
        svg = reliability_svg(panels, f"Надёжность скора критерия {crit} (OOF) — {c['name']}")
        f = out_dir / f"reliability_{crit}.svg"
        f.write_text(svg, encoding="utf-8"); files.append(_rel(f))
    return files


# ----------------------------------------------------------------------------- таблицы для документа
def _r(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def markdown_tables(res: dict) -> str:
    L = []
    L.append("| Цель | Вариант | n / pos (иссл.) | Brier [95 % ДИ] | Brier конст. | Лог-лосс | ECE10 [95 % ДИ] | MCE10 | ECE q5 | Наклон | Сдвиг | ROC-AUC |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    def row(tgt, var, ev):
        ci = ev.get("ci95", {})
        ce = f' [{ci["ece_10"][0]:.2f}; {ci["ece_10"][1]:.2f}]' if ci else ""
        cb = f' [{ci["brier"][0]:.3f}; {ci["brier"][1]:.3f}]' if ci else ""
        L.append(f'| {tgt} | {var} | {ev["n"]} / {ev["n_pos"]} ({ev["n_pos_studies"]}) | {ev["brier"]:.3f}{cb} | '
                 f'{ev["brier_const"]:.3f} | {ev["log_loss"]:.3f} | {ev["ece_10"]:.3f}{ce} | {ev["mce_10"]:.3f} | '
                 f'{ev["ece_q5"]:.3f} | {_r(ev["slope"], 2)} | {_r(ev["intercept"], 2)} | {_r(ev["roc_auc"])} |')
    for region, r in res["regions"].items():
        for k in ("final", "raw_blend"):
            row(r["name"], VARIANT_NAME[k], r["variants"][k])
    for crit, c in res["criteria"].items():
        for k in ("score", "p_cal", "p_cal_crossfit"):
            if k in c["variants"]:
                row(f"`{crit}`", VARIANT_NAME[k], c["variants"][k])
    L.append("")
    L.append("| Критерий | порог | запас | «не уверен» на OOF | ошибок в зоне / всего | p_cal на пороге |")
    L.append("|---|---|---|---|---|---|")
    for crit, u in res["uncertain_zone"].items():
        L.append(f'| `{crit}` | {u["threshold"]:.4f} | {u["margin"]:.6g} | {u["n_uncertain"]}/{u["n"]} '
                 f'({100 * u["share_uncertain"]:.1f} %) | {u["n_errors_in_zone"]}/{u["n_errors_total"]} | {_r(u["p_cal_at_threshold"])} |')
    L.append("")
    L.append("| Область | строк «не уверен» (любой критерий) | доля class=1 на OOF | доля позитивов |")
    L.append("|---|---|---|---|")
    for region, r in res["regions"].items():
        ev = r["variants"]["final"]
        L.append(f'| {r["name"]} | {r["n_row_uncertain"]}/{ev["n"]} ({100 * r["row_uncertain_share"]:.1f} %) | '
                 f'{r["share_class1"]:.3f} | {ev["prevalence"]:.3f} |')
    return "\n".join(L)


def bins_markdown(ev: dict) -> str:
    L = ["| бин | n | pos | средн. предсказание | доля положительных | разрыв |", "|---|---|---|---|---|---|"]
    for b in ev["bins_equal_10"]:
        L.append(f'| [{b["lo"]:.1f}; {b["hi"]:.1f}) | {b["n"]} | {b["n_pos"]} | {_r(b["mean_pred"])} | {_r(b["frac_pos"])} | {_r(b["gap"])} |')
    return "\n".join(L)


# ----------------------------------------------------------------------------- main
def _round(o, nd=6):
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, list):
        return [_round(v, nd) for v in o]
    return o


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(DOCS_OUT), help="каталог для calibration.json и SVG (по умолчанию docs/calibration)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--no-svg", action="store_true")
    ap.add_argument("--bins", action="store_true", help="печатать таблицы бинов равной ширины")
    a = ap.parse_args(argv)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = build_targets(a.n_boot)
    res["method"] = {
        "oof": "models/oof_stacked_<region>_<crit>.csv + OOF any-модели (src/eval_oof_metrics.oof_any_model)",
        "ece": f"{N_BINS_EQ} бинов равной ширины (ece_10/mce_10) и {N_BINS_Q} квантильных (ece_q5/mce_q5); последний бин включает 1.0",
        "ci": f"percentile bootstrap по исследованиям, {a.n_boot} повторов, зерно {BOOT_SEED}; для ece_10 и brier",
        "slope_intercept": "логистическая рекалибровка y ~ a + b*logit(p), p обрезан до [1e-6, 1-1e-6]; идеал b=1, a=0",
        "p_cal_crossfit": "Platt на скоре, GroupKFold(5) по исследованию, зерно 0",
        "uncertain": "|score - threshold| <= margin (config.yaml uncertainty.margin_by_criterion)",
        "numbers_of_delivery_changed": False,
    }
    files = write_svgs(res, out_dir, a.n_boot) if not a.no_svg else []
    res["svg_files"] = files
    (out_dir / "calibration.json").write_text(json.dumps(_round(res), ensure_ascii=False, indent=1, sort_keys=True),
                                              encoding="utf-8")
    print(markdown_tables(res))
    if a.bins:
        for region, r in res["regions"].items():
            print(f"\n### {r['name']} — quality_prob итоговый\n" + bins_markdown(r["variants"]["final"]))
        for crit, c in res["criteria"].items():
            for k in ("score", "p_cal_crossfit"):
                print(f"\n### {crit} — {VARIANT_NAME[k]}\n" + bins_markdown(c["variants"][k]))
    print(f"\nзаписано: {out_dir / 'calibration.json'}; SVG: {len(files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
