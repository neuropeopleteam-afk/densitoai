#!/usr/bin/env python3
"""Кривая чистой пользы (decision curve analysis, Vickers и Elkin, 2006) для решения «переснять».

Сравниваются стратегии на OOF поставки, по областям (позвоночник, бедро):
  * «по решению сервиса» — пересъёмка, если quality_class = 1 (ИЛИ флагов критериев, pred_label);
  * «переснимать всех» и «не переснимать никого»;
  * справочно — «quality_prob ≥ p_t» (классическая кривая по оценке; quality_prob откалиброван грубо,
    ECE 0.14–0.15, `docs/CALIBRATION.md`, поэтому это ориентир, а не главный результат).

Чистая польза на кадр: NB(p_t) = TP/n − FP/n · w, где w = p_t / (1 − p_t) — цена одной ложной
пересъёмки в долях пользы от одного пойманного нарушения (w = 1: ложная пересъёмка стоит столько же,
сколько пропущенный брак; w = 0.25: четыре ложные пересъёмки стоят одного пропуска).
Рабочая точка сервиса: порог класса quality_prob ≥ 0.5, то есть p_t = 0.5 (w = 1); на кривой по оценке
она совпадает с линией «по решению сервиса».
95 % ДИ — бутстрэп по исследованиям (2000 повторов). Модели и пороги не меняются.
quality_prob OOF воспроизводится ровно как в `src/eval_oof_metrics.py` (any-модель + max критериев,
согласование с классом); контроль — ROC-AUC 0.813 / 0.760.

Запуск: DENSITO_ROOT=$PWD ../venv/bin/python tools/decision_curve.py
Выход: docs/decision_curve.json, docs/decision_curve.svg
Никаких обещаний экономии: кривая показывает, при какой цене ошибки стратегия лучше альтернатив
на этой выборке, а не сколько пересъёмок отделение сэкономит.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
import eval_oof_metrics as E  # noqa: E402

N_BOOT = 2000
SEED = 2026
PT_GRID = np.round(np.arange(0.02, 0.905, 0.01), 3)
W_REPORT = [0.1, 0.25, 0.5, 1.0, 2.0]          # цены ложной пересъёмки для таблицы
REGION_RU = {"spine": "Поясничный отдел позвоночника", "hip": "Проксимальный отдел бедра"}


def rel_path(p: str) -> str:
    return p.split("Исследования/", 1)[1] if "Исследования/" in p else p


def region_frame(region: str) -> pd.DataFrame:
    summary = json.load(open(ROOT / "models" / "metrics_summary.json", encoding="utf-8"))
    criteria = E.REGION_CRITERIA[region]
    base = None
    flags = ytrue = cmax = None
    for crit in criteria:
        df = pd.read_csv(ROOT / "models" / f"oof_stacked_{region}_{crit}.csv")
        s = df["oof_stacked"].values.astype(float)
        f = E.flags_from_scores(df, s, summary[region][crit]["threshold"])
        if base is None:
            base = df[["study", "file_path"]].copy()
            flags = np.zeros(len(df), int); ytrue = np.zeros(len(df), int); cmax = np.zeros(len(df))
        assert (df["file_path"].values == base["file_path"].values).all()
        flags = np.maximum(flags, f); ytrue = np.maximum(ytrue, df["y_true"].values.astype(int))
        cmax = np.maximum(cmax, s)
    anym = E.oof_any_model(region, criteria).set_index("file_path").loc[base["file_path"].values]
    raw = 0.5 * anym["any_model_oof"].values + 0.5 * cmax
    prob = np.where(flags == 1, 0.5 + 0.5 * raw, np.minimum(0.5 * raw, 0.499999))
    base = base.assign(y=ytrue, flag=flags, prob=prob, rel_path=base["file_path"].map(rel_path))
    ph = ROOT / "docs" / "k5" / "pixel_hashes.csv"
    if ph.exists():
        h = pd.read_csv(ph)
        h["rel_path"] = h["file_path"].map(rel_path)
        base = base.merge(h[["rel_path", "pixel_hash"]], on="rel_path", how="left")
    return base


def nb(y, decide, w):
    n = len(y)
    tp = np.sum((decide == 1) & (y == 1)); fp = np.sum((decide == 1) & (y == 0))
    return tp / n - fp / n * w


def curves(y, flag, prob):
    ws = PT_GRID / (1 - PT_GRID)
    out = {"service": [], "all": [], "score": []}
    for pt, w in zip(PT_GRID, ws):
        out["service"].append(nb(y, flag, w))
        out["all"].append(nb(y, np.ones_like(y), w))
        out["score"].append(nb(y, (prob >= pt).astype(int), w))
    return {k: np.array(v) for k, v in out.items()}


def boot(df: pd.DataFrame, rng):
    g = df["study"].to_numpy(); ug = np.unique(g)
    idx = {s: np.nonzero(g == s)[0] for s in ug}
    y, f, p = df["y"].to_numpy(), df["flag"].to_numpy(), df["prob"].to_numpy()
    S, A, D, SC = [], [], [], []
    for _ in range(N_BOOT):
        ii = np.concatenate([idx[s] for s in rng.choice(ug, len(ug))])
        c = curves(y[ii], f[ii], p[ii])
        S.append(c["service"]); A.append(c["all"]); SC.append(c["score"])
        D.append(c["service"] - np.maximum(c["all"], 0.0))
    q = lambda M: (np.percentile(np.array(M), 2.5, axis=0), np.percentile(np.array(M), 97.5, axis=0))
    return {"service": q(S), "all": q(A), "score": q(SC), "delta": q(D)}


def table_at(y, flag, prob, ws):
    rows = []
    for w in ws:
        pt = w / (1 + w)
        rows.append({"w": w, "p_t": round(pt, 4),
                     "nb_service_per100": 100 * nb(y, flag, w),
                     "nb_all_per100": 100 * nb(y, np.ones_like(y), w),
                     "nb_score_per100": 100 * nb(y, (prob >= pt).astype(int), w)})
    return rows


def delta_boot_at(df, ws, rng):
    g = df["study"].to_numpy(); ug = np.unique(g)
    idx = {s: np.nonzero(g == s)[0] for s in ug}
    y, f = df["y"].to_numpy(), df["flag"].to_numpy()
    res = {w: {"svc": [], "d": []} for w in ws}
    for _ in range(N_BOOT):
        ii = np.concatenate([idx[s] for s in rng.choice(ug, len(ug))])
        for w in ws:
            s = nb(y[ii], f[ii], w); a = nb(y[ii], np.ones(len(ii)), w)
            res[w]["svc"].append(100 * s); res[w]["d"].append(100 * (s - max(a, 0.0)))
    return {w: {"nb_service_ci": [float(np.percentile(v["svc"], 2.5)), float(np.percentile(v["svc"], 97.5))],
                "delta_vs_best_alt_ci": [float(np.percentile(v["d"], 2.5)), float(np.percentile(v["d"], 97.5))],
                "p_delta_le_0": float(np.mean(np.array(v["d"]) <= 0))} for w, v in res.items()}


# ---------------- SVG без внешних зависимостей ----------------

def svg_panel(x0, y0, W, H, title, c, ci, ymin, ymax):
    ws = PT_GRID / (1 - PT_GRID)
    xmin, xmax = 0.0, 0.9
    X = lambda pt: x0 + 58 + (pt - xmin) / (xmax - xmin) * (W - 78)
    Y = lambda v: y0 + 30 + (ymax - v) / (ymax - ymin) * (H - 80)
    el = [f'<text x="{x0 + W / 2}" y="{y0 + 18}" text-anchor="middle" font-size="14" font-weight="600">{title}</text>']
    # оси и сетка
    for v in np.arange(ymin, ymax + 1e-9, 0.1):
        el.append(f'<line x1="{X(xmin)}" y1="{Y(v):.1f}" x2="{X(xmax)}" y2="{Y(v):.1f}" stroke="#e3e1dc"/>')
        el.append(f'<text x="{X(xmin) - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" font-size="10">{v:.1f}</text>')
    for pt in (0.1, 0.2, 0.333, 0.5, 0.667, 0.8, 0.9):
        el.append(f'<line x1="{X(pt):.1f}" y1="{Y(ymin)}" x2="{X(pt):.1f}" y2="{Y(ymax)}" stroke="#eeede9"/>')
        el.append(f'<text x="{X(pt):.1f}" y="{Y(ymin) + 14}" text-anchor="middle" font-size="10">{pt:.2f}</text>')
        el.append(f'<text x="{X(pt):.1f}" y="{Y(ymin) + 27}" text-anchor="middle" font-size="9" fill="#6b6a65">'
                  f'w={pt / (1 - pt):.2g}</text>')
    el.append(f'<text x="{x0 + W / 2}" y="{Y(ymin) + 42}" text-anchor="middle" font-size="10">'
              f'порог p_t; w = p_t/(1−p_t) — цена ложной пересъёмки в долях пользы пойманного нарушения</text>')
    el.append(f'<text x="{x0 + 10}" y="{y0 + H / 2}" font-size="10" transform="rotate(-90 {x0 + 10} {y0 + H / 2})" '
              f'text-anchor="middle">чистая польза на кадр</text>')
    el.append(f'<line x1="{X(xmin)}" y1="{Y(0):.1f}" x2="{X(xmax)}" y2="{Y(0):.1f}" stroke="#28251d" stroke-width="1.5"/>')

    def clip(v):
        return min(max(v, ymin), ymax)

    def poly(vals, color, width=2, dash=""):
        # точки ниже нижней границы шкалы не рисуем (линия обрывается у края)
        pts = " ".join(f"{X(pt):.1f},{Y(v):.1f}" for pt, v in zip(PT_GRID, vals) if ymin <= v <= ymax)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}"{d}/>'

    lo, hi = ci["service"]
    band = " ".join(f"{X(pt):.1f},{Y(clip(v)):.1f}" for pt, v in zip(PT_GRID, hi)) + " " + \
        " ".join(f"{X(pt):.1f},{Y(clip(v)):.1f}" for pt, v in zip(PT_GRID[::-1], lo[::-1]))
    el.append(f'<polygon points="{band}" fill="#01696f" fill-opacity="0.15" stroke="none"/>')
    el.append(poly(c["all"], "#b0522b", 2, "6 4"))
    el.append(poly(c["score"], "#6b6a65", 1.3, "2 3"))
    el.append(poly(c["service"], "#01696f", 2.5))
    # рабочая точка: p_t = 0.5
    i = int(np.argmin(np.abs(PT_GRID - 0.5)))
    el.append(f'<circle cx="{X(0.5):.1f}" cy="{Y(clip(c["service"][i])):.1f}" r="5" fill="#01696f" stroke="#fff"/>')
    el.append(f'<text x="{X(0.5) + 8:.1f}" y="{Y(clip(c["service"][i])) - 8:.1f}" font-size="10">'
              f'рабочая точка: quality_prob ≥ 0.5</text>')
    return el


def write_svg(path, panels):
    W, H = 560, 380
    el = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W * 2}" height="{H + 60}" '
          f'font-family="Noto Sans, DejaVu Sans, Arial, sans-serif" fill="#28251d">',
          f'<rect width="{W * 2}" height="{H + 60}" fill="#fff"/>']
    for k, (title, c, ci) in enumerate(panels):
        el += svg_panel(k * W, 0, W, H, title, c, ci, -0.3, 0.6)
    ly = H + 20
    xs_leg = [30, 420, 590, 800]
    for j, (col, dash, lab) in enumerate([("#01696f", "", "по решению сервиса (quality_class = 1), полоса — 95 % ДИ"),
                                          ("#b0522b", "6 4", "переснимать всех"),
                                          ("#28251d", "", "не переснимать никого (0)"),
                                          ("#6b6a65", "2 3", "справочно: quality_prob ≥ p_t")]):
        x = xs_leg[j]
        d = f' stroke-dasharray="{dash}"' if dash else ""
        el.append(f'<line x1="{x}" y1="{ly}" x2="{x + 28}" y2="{ly}" stroke="{col}" stroke-width="2.5"{d}/>')
        el.append(f'<text x="{x + 34}" y="{ly + 4}" font-size="10.5">{lab}</text>')
    el.append(f'<text x="30" y="{ly + 26}" font-size="9.5" fill="#6b6a65">OOF поставки 2.4.0, кадры организаторов '
              f'(499 файлов, 100 исследований); tools/decision_curve.py. Показывает, при какой цене ошибки стратегия '
              f'лучше альтернатив на этой выборке; экономию не измеряет.</text>')
    el.append("</svg>")
    Path(path).write_text("\n".join(el), encoding="utf-8")


def main() -> None:
    rng = np.random.default_rng(SEED)
    res = {"method": {"nb": "TP/n − FP/n · p_t/(1−p_t)", "ci": f"бутстрэп по исследованиям, {N_BOOT}, 95 %",
                      "service_flag": "quality_class = ИЛИ pred_label критериев (как в src/inference.py)",
                      "working_point": "p_t = 0.5 (quality_prob ≥ 0.5 ⇔ quality_class = 1)"}}
    panels = []
    for region in ("spine", "hip"):
        df = region_frame(region)
        y, f, p = df.y.to_numpy(), df.flag.to_numpy(), df.prob.to_numpy()
        c = curves(y, f, p)
        ci = boot(df, rng)
        better = (ci["delta"][0] > 0)
        wins = PT_GRID[better]
        r = {"n": int(len(df)), "n_pos": int(y.sum()), "n_flag": int(f.sum()), "n_studies": int(df.study.nunique()),
             "prevalence": float(y.mean()), "tp": int(((f == 1) & (y == 1)).sum()), "fp": int(((f == 1) & (y == 0)).sum()),
             "auc_quality_prob_check": float(roc_auc_score(y, p)),
             "pt_where_service_ci_above_best_alt": [float(wins.min()), float(wins.max())] if len(wins) else None,
             "pt_where_service_point_above_best_alt": [float(PT_GRID[c["service"] > np.maximum(c["all"], 0)].min()),
                                                        float(PT_GRID[c["service"] > np.maximum(c["all"], 0)].max())]
             if (c["service"] > np.maximum(c["all"], 0)).any() else None,
             "pt_service_nb_zero": float(PT_GRID[np.argmin(np.abs(c["service"]))]),
             "table": table_at(y, f, p, W_REPORT),
             "table_ci": {str(k): v for k, v in delta_boot_at(df, W_REPORT, rng).items()}}
        if "pixel_hash" in df and df.pixel_hash.notna().all():
            du = df.drop_duplicates("pixel_hash")
            r["unique_frames"] = {"n": int(len(du)), "n_pos": int(du.y.sum()),
                                  "table": table_at(du.y.to_numpy(), du.flag.to_numpy(), du.prob.to_numpy(), W_REPORT)}
        res[region] = r
        panels.append((REGION_RU[region] + f" (n = {len(df)}, нарушений {int(y.sum())})", c, ci))
    write_svg(ROOT / "docs" / "decision_curve.svg", panels)
    (ROOT / "docs" / "decision_curve.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
