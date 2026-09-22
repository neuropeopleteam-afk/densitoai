#!/usr/bin/env python3
"""Полный OOF-отчёт по метрикам в формате ТЗ §8.4 с 95 % бутстрап-ДИ.

Что считается (всё — на out-of-fold предсказаниях, группировка по исследованию,
т.е. снимки одного исследования никогда не попадают одновременно в train и в
validation):

  1. По каждому типу нарушения (5 официальных типов: 3 позвоночник + 2 бедро):
     чувствительность, специфичность, сбалансированная точность, F1, ROC-AUC,
     PR-AUC — из models/oof_stacked_<region>_<crit>.csv (стек геометрия+эмбеддинги,
     порог из models/metrics_summary.json, как в инференсе).
  2. По каждой анатомической области — бинарная задача «есть нарушение»
     (quality_class / quality_prob):
       quality_class_oof = OR флагов критериев;
       quality_prob_oof  = смесь OOF any-модели (geom+emb, StratifiedGroupKFold(5),
                           3 сида) и max OOF-оценки критериев, приведённая к
                           согласованности с классом — ровно как в src/inference.py.
  3. Macro-F1 по типам нарушений внутри области (второй этап последовательной
     оценки организаторов: сначала бинарная, потом типы).

95 % ДИ — percentile bootstrap по исследованиям (study-level), 2000 повторов.

Запуск:  python src/eval_oof_metrics.py  -> models/metrics_oof_full.json,
                                            docs/metrics_oof_full.md
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train_final_models import (CRITERION_GEOMETRY_COLS, DATA_DIR, OUT_DIR, REGION_CRITERIA,  # noqa: E402
                                REGION_ROWS, _impute, fit_geom, fit_emb)

N_BOOT = 2000
SEEDS = (0, 1, 2)
RNG = np.random.default_rng(2026)
VIOL_NAME = {"sp_pos": "Некорректная укладка", "sp_axis": "Не выравнена ось позвоночника",
             "sp_art": "Присутствуют посторонние предметы",
             "hip_pos": "Некорректная укладка", "hip_roi": "Некорректная область интереса"}
REGION_NAME = {"spine": "Поясничный отдел позвоночника", "hip": "Проксимальный отдел бедра"}


def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def safe_ap(y, s):
    return float(average_precision_score(y, s)) if y.sum() > 0 else float("nan")


def point_metrics(y, s, yhat):
    tp = int(((yhat == 1) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    return {
        "n": int(len(y)), "n_pos": int(y.sum()), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)) if len(np.unique(y)) == 2 else float("nan"),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "roc_auc": safe_auc(y, s), "pr_auc": safe_ap(y, s),
    }


def bootstrap_ci(y, s, yhat, groups, keys=("sensitivity", "specificity", "balanced_accuracy", "f1", "roc_auc", "pr_auc")):
    ug = np.unique(groups)
    idx_by_g = {g: np.nonzero(groups == g)[0] for g in ug}
    acc = {k: [] for k in keys}
    for _ in range(N_BOOT):
        sample = RNG.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in sample])
        yb, sb, hb = y[idx], s[idx], yhat[idx]
        if len(np.unique(yb)) < 2:
            continue
        m = point_metrics(yb, sb, hb)
        for k in keys:
            acc[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if v else [float("nan")] * 2
            for k, v in acc.items()}


def oof_any_model(region, criteria):
    """OOF-вероятность any-модели (geom+emb) — воспроизводит train_final_models, но с CV."""
    geom_all = pd.read_csv(DATA_DIR / "geometry_features.csv")
    emb_labels = pd.read_csv(DATA_DIR / "labels_for_embeddings.csv")
    embeddings = np.load(DATA_DIR / "embeddings.npy")
    rows = REGION_ROWS.get(region, [region])
    gdf = geom_all[geom_all["region"].isin(rows)].reset_index(drop=True)
    eidx = np.nonzero(emb_labels["region"].isin(rows).values)[0]
    E_all = embeddings[eidx]
    assert (emb_labels.loc[eidx, "file_path"].values == gdf["file_path"].values).all()
    from train_stacked import CRITERION_LABEL_COL
    any_label = np.zeros(len(gdf)); seen = np.zeros(len(gdf), bool)
    for crit in criteria:
        y = gdf[CRITERION_LABEL_COL.get(crit, crit)].values.astype(float)
        v = ~np.isnan(y); any_label[v] = np.maximum(any_label[v], y[v]); seen |= v
    ya = any_label.astype(int)
    cols_any = sorted({c for cr in criteria for c in CRITERION_GEOMETRY_COLS[cr]})
    Xraw = gdf[cols_any].values.astype(np.float64)
    groups = gdf["study"].values
    oof = np.zeros((len(SEEDS), len(gdf)))
    for si, seed in enumerate(SEEDS):
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, te in skf.split(Xraw, ya, groups):
            med = np.nanmedian(Xraw[tr], axis=0)
            dg = fit_geom(_impute(Xraw[tr], med), ya[tr], cols_any, med)
            de = fit_emb(E_all[tr], ya[tr])
            pg = dg["clf"].predict_proba(dg["scaler"].transform(_impute(Xraw[te], med)))[:, 1]
            pe = de["clf"].predict_proba(de["pca"].transform(de["scaler"].transform(E_all[te])))[:, 1]
            oof[si, te] = 0.5 * (pg + pe)
    return gdf[["study", "file_path"]].assign(y_any=ya, any_model_oof=oof.mean(0), seen=seen)


# Допуск сравнения с порогом. Оценки в CSV записаны текстом, и значение, равное порогу
# в памяти, после разбора может оказаться на 1e-16 ниже. Без допуска такие случаи
# перестают считаться нарушением, и отчёт показывает F1 выше, чем даёт сервис.
THR_TIE_ATOL = 1e-9


def flags_from_scores(df: pd.DataFrame, s: np.ndarray, thr: float) -> np.ndarray:
    """Флаги нарушения ровно те же, что выдаёт сервис.

    Приоритет — колонка `pred_label`, записанная обучением в момент, когда оценки были
    в памяти: это решение, которое принимает инференс. Если колонки нет, сравниваем с
    порогом с допуском THR_TIE_ATOL, чтобы совпадения «оценка == порог» не терялись
    из-за округления при записи CSV.
    """
    if "pred_label" in df.columns:
        pl = pd.to_numeric(df["pred_label"], errors="coerce")
        if not pl.isna().any():
            yhat = pl.values.astype(int)
            recomputed = (s >= thr - THR_TIE_ATOL).astype(int)
            n_diff = int((yhat != recomputed).sum())
            if n_diff:
                print(f"    внимание: pred_label и сравнение с порогом расходятся в {n_diff} случаях "
                      f"(совпадения с порогом); берём pred_label — это решение сервиса")
            return yhat
    return (s >= thr - THR_TIE_ATOL).astype(int)


def main():
    summary = json.load(open(OUT_DIR / "metrics_summary.json", encoding="utf-8"))
    out = {"method": {"cv": "StratifiedGroupKFold / repeated GroupKFold by study; OOF predictions only",
                      "ci": f"study-level percentile bootstrap, {N_BOOT} resamples, 95 %"},
           "by_violation_type": {}, "by_region_binary": {}, "macro_f1": {}}
    md = ["# Полный OOF-отчёт по метрикам (ТЗ §8.4)\n",
          "Все значения — на out-of-fold предсказаниях (группировка по исследованию), "
          f"95 % ДИ — бутстрап по исследованиям ({N_BOOT} повторов). "
          "Пороги — те же, что в `config.yaml`/`models/metrics_summary.json` и в инференсе.\n"]

    for region, criteria in REGION_CRITERIA.items():
        # ---- per-criterion ----
        crit_frames = {}
        md.append(f"\n## {REGION_NAME[region]} — по типам нарушений\n")
        md.append("| Тип нарушения | n | n_pos | Sens | Spec | Bal.Acc | F1 | ROC-AUC | PR-AUC |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for crit in criteria:
            df = pd.read_csv(OUT_DIR / f"oof_stacked_{region}_{crit}.csv")
            thr = summary[region][crit]["threshold"]
            y = df["y_true"].values.astype(int); s = df["oof_stacked"].values.astype(float)
            yhat = flags_from_scores(df, s, thr)
            g = df["study"].values
            pm = point_metrics(y, s, yhat); ci = bootstrap_ci(y, s, yhat, g)
            pm["threshold"] = float(thr); pm["ci95"] = ci
            out["by_violation_type"][f"{region}/{crit}"] = {"violation_type": VIOL_NAME[crit], **pm}
            crit_frames[crit] = df.assign(flag=yhat)
            f = lambda k: f"{pm[k]:.3f} [{ci[k][0]:.2f}; {ci[k][1]:.2f}]"
            md.append(f"| {VIOL_NAME[crit]} (`{crit}`) | {pm['n']} | {pm['n_pos']} | {f('sensitivity')} | {f('specificity')} | "
                      f"{f('balanced_accuracy')} | {f('f1')} | {f('roc_auc')} | {f('pr_auc')} |")

        # ---- region binary (quality_class / quality_prob) ----
        base = crit_frames[criteria[0]][["study", "file_path"]].copy()
        flags = np.zeros(len(base), int); ytrue = np.zeros(len(base), int); crit_max = np.zeros(len(base))
        for crit, df in crit_frames.items():
            assert (df["file_path"].values == base["file_path"].values).all(), "OOF files misaligned"
            flags = np.maximum(flags, df["flag"].values); ytrue = np.maximum(ytrue, df["y_true"].values.astype(int))
            crit_max = np.maximum(crit_max, df["oof_stacked"].values)
        anym = oof_any_model(region, criteria)
        anym = anym.set_index("file_path").loc[base["file_path"].values]
        assert (anym["y_any"].values == ytrue).all(), "any-label mismatch"
        raw = 0.5 * anym["any_model_oof"].values + 0.5 * crit_max
        prob = np.where(flags == 1, 0.5 + 0.5 * raw, np.minimum(0.5 * raw, 0.499999))
        g = base["study"].values
        pm = point_metrics(ytrue, prob, flags); ci = bootstrap_ci(ytrue, prob, flags, g)
        pm["ci95"] = ci
        pm["roc_auc_components"] = {"any_model_only": safe_auc(ytrue, anym["any_model_oof"].values),
                                    "max_criteria_only": safe_auc(ytrue, crit_max),
                                    "blend_raw": safe_auc(ytrue, raw), "blend_consistent(final)": safe_auc(ytrue, prob)}
        out["by_region_binary"][region] = {"anatomical_region": REGION_NAME[region], **pm}
        f = lambda k: f"{pm[k]:.3f} [{ci[k][0]:.2f}; {ci[k][1]:.2f}]"
        md.append(f"\n## {REGION_NAME[region]} — бинарная задача «есть нарушение» (quality_class / quality_prob)\n")
        md.append("| n | n_pos | Sens | Spec | Bal.Acc | F1 | ROC-AUC | PR-AUC |")
        md.append("|---|---|---|---|---|---|---|---|")
        md.append(f"| {pm['n']} | {pm['n_pos']} | {f('sensitivity')} | {f('specificity')} | {f('balanced_accuracy')} | "
                  f"{f('f1')} | {f('roc_auc')} | {f('pr_auc')} |")
        rc = pm["roc_auc_components"]
        md.append(f"\nROC-AUC компонентов quality_prob: any-модель {rc['any_model_only']:.3f}, max по критериям "
                  f"{rc['max_criteria_only']:.3f}, смесь {rc['blend_raw']:.3f}, смесь + согласование с классом "
                  f"(итоговая) **{rc['blend_consistent(final)']:.3f}**.\n")

        # ---- macro-F1 over violation types ----
        f1s = {crit: out["by_violation_type"][f"{region}/{crit}"]["f1"] for crit in criteria}
        # bootstrap macro-F1
        ug = np.unique(g); idx_by_g = {x: np.nonzero(g == x)[0] for x in ug}; boots = []
        Y = np.stack([crit_frames[c]["y_true"].values for c in criteria], 1)
        H = np.stack([crit_frames[c]["flag"].values for c in criteria], 1)
        for _ in range(N_BOOT):
            idx = np.concatenate([idx_by_g[x] for x in RNG.choice(ug, len(ug), replace=True)])
            boots.append(np.mean([f1_score(Y[idx, j], H[idx, j], zero_division=0) for j in range(len(criteria))]))
        out["macro_f1"][region] = {"macro_f1": float(np.mean(list(f1s.values()))), "per_type": f1s,
                                   "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
        m = out["macro_f1"][region]
        md.append(f"**Macro-F1 по типам нарушений ({REGION_NAME[region]}):** {m['macro_f1']:.3f} "
                  f"[{m['ci95'][0]:.2f}; {m['ci95'][1]:.2f}]\n")

    # overall
    allb = out["by_region_binary"]
    md.append("\n## Итого\n")
    md.append("| Область | Бинарная ROC-AUC | Бинарная F1 | Macro-F1 по типам |")
    md.append("|---|---|---|---|")
    for r in REGION_CRITERIA:
        b = allb[r]; m = out["macro_f1"][r]
        md.append(f"| {REGION_NAME[r]} | {b['roc_auc']:.3f} [{b['ci95']['roc_auc'][0]:.2f}; {b['ci95']['roc_auc'][1]:.2f}] | "
                  f"{b['f1']:.3f} [{b['ci95']['f1'][0]:.2f}; {b['ci95']['f1'][1]:.2f}] | "
                  f"{m['macro_f1']:.3f} [{m['ci95'][0]:.2f}; {m['ci95'][1]:.2f}] |")
    md.append("\nСкорость и доля успешных обработок измеряются отдельно (см. `docs/METRICS_REPORT.md`, раздел «Скорость»).\n")

    (OUT_DIR / "metrics_oof_full.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "docs" / "metrics_oof_full.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
