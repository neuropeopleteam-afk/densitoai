#!/usr/bin/env python
"""
Разброс OOF-метрик по разбиениям (проверка устойчивости оценки).

Зачем. В `src/train_stacked.py` внешний цикл `for repeat in range(N_REPEATS)` берёт
`GroupKFold(n_splits=5)` без `shuffle` и меняет только порядок строк (`perm`).
GroupKFold раскладывает группы по фолдам детерминированно — по размеру группы, —
поэтому перестановка строк разбиение не меняет: все повторы дают ОДНО И ТО ЖЕ
разбиение, а усреднение по повторам возвращает те же числа, что один прогон.
Скрипт делает две вещи:
  1) доказывает это (сравнивает карты «группа → фолд» по повторам);
  2) измеряет, насколько OOF AUC и F1 зависят от разбиения, если разбиение
     действительно менять — `StratifiedGroupKFold(n_splits=5, shuffle=True,
     random_state=s)`, s = 0..N-1 (группы соблюдаются, доля положительных
     по фолдам выравнивается — при 10–92 положительных это важно).

Скрипт ничего не переобучает в поставке и не меняет продакшен-числа: он только
оценивает их устойчивость. Результат — `docs/metrics_split_spread.json` и таблица
для раздела «Разброс по разбиениям» в `docs/METRICS_REPORT.md`.

Запуск:  python tools/split_spread.py [--splits 10]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train_stacked as ts  # noqa: E402

REGIONS = {"spine": ["sp_pos", "sp_axis", "sp_art"], "hip": ["hip_pos", "hip_roi"]}


def prepare(region, crit):
    """Готовит X_geom / X_emb / y / groups ровно как train_region_stacked."""
    region_rows = ts.REGION_ROWS.get(region, [region])
    geom_by_variant = {}
    for variant, fn in ts.GEOM_VARIANT_FILES.items():
        f = ts.DATA_DIR / fn
        if f.exists():
            g = pd.read_csv(f)
            geom_by_variant[variant] = g[g["region"].isin(region_rows)].reset_index(drop=True)
    geom_df = geom_by_variant["baseline"]

    emb_labels = pd.read_csv(ts.DATA_DIR / "labels_for_embeddings.csv")
    emb_by_source = ts.load_embeddings_by_source()
    emb_labels["emb_idx"] = np.arange(len(emb_labels))
    region_emb_idx = emb_labels[emb_labels["region"].isin(region_rows)]["emb_idx"].values
    region_emb_by_source = {k: v[region_emb_idx] for k, v in emb_by_source.items()}

    y = geom_df[ts.CRITERION_LABEL_COL.get(crit, crit)].values
    valid = ~pd.isna(y)
    y_valid = y[valid].astype(int)

    preproc = ts.preproc_for(crit)
    gv = preproc["geom"] if preproc["geom"] in geom_by_variant else "baseline"
    X_geom_raw = geom_by_variant[gv][ts.CRITERION_GEOMETRY_COLS[crit]].values.astype(np.float64)
    med = np.nanmedian(X_geom_raw, axis=0)
    for j in range(X_geom_raw.shape[1]):
        X_geom_raw[np.isnan(X_geom_raw[:, j]), j] = med[j]

    emb_matrix, emb_src, _ = ts.emb_matrix_for(crit, region_emb_by_source)
    return (X_geom_raw[valid], emb_matrix[valid], y_valid,
            geom_df["study"].values[valid], emb_src)


def oof_scores(X_geom, X_emb, y, splits):
    """OOF по заданному списку (train_idx, val_idx): контур A + контур B."""
    og, oe = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for tr, va in splits:
        if y[tr].sum() < 2 or y[tr].sum() == len(tr):
            continue
        sg = StandardScaler()
        clf_g = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        clf_g.fit(sg.fit_transform(X_geom[tr]), y[tr])
        og[va] = clf_g.predict_proba(sg.transform(X_geom[va]))[:, 1]

        n_comp = min(ts.PCA_COMPONENTS, len(tr) - 1, X_emb.shape[1])
        se = StandardScaler()
        pca = PCA(n_components=n_comp, random_state=42)
        Xe_tr = pca.fit_transform(se.fit_transform(X_emb[tr]))
        clf_e = LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced")
        clf_e.fit(Xe_tr, y[tr])
        oe[va] = clf_e.predict_proba(pca.transform(se.transform(X_emb[va])))[:, 1]
    return og, oe


def metrics(crit, og, oe, y):
    w = ts.weight_geom_for(crit)
    stacked = w * pd.Series(og).rank(pct=True).values + (1 - w) * pd.Series(oe).rank(pct=True).values
    ok = ~np.isnan(stacked)
    auc = roc_auc_score(y[ok], stacked[ok]) if len(np.unique(y[ok])) > 1 else None
    thr, _ = ts.threshold_by_rule(ts.threshold_rule_for(crit), y, stacked,
                                  ts.f1_optimal_threshold, ts.prevalence_threshold, min_pos_f1=15)
    f1 = f1_score(y, (stacked >= thr).astype(int), zero_division=0)
    return auc, float(f1), float(thr)


def fold_map(groups, splits):
    """Карта «группа → номер фолда» для сравнения разбиений."""
    m = {}
    for k, (_, va) in enumerate(splits):
        for gr in np.unique(groups[va]):
            m[str(gr)] = k
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=10, help="сколько разных разбиений мерить")
    args = ap.parse_args()

    out = {"n_splits_measured": args.splits, "criteria": {}, "repeats_are_identical": None}
    identical_flags = []

    for region, crits in REGIONS.items():
        for crit in crits:
            X_geom, X_emb, y, groups, emb_src = prepare(region, crit)

            # --- 1. продакшен-разбиение (GroupKFold без shuffle) и проверка «повторов»
            maps = []
            for repeat in range(ts.N_REPEATS):
                rng = np.random.default_rng(42 + repeat)
                perm = rng.permutation(len(y))
                gkf = GroupKFold(n_splits=ts.N_FOLDS)
                sp = [(perm[a], perm[b]) for a, b in gkf.split(X_geom[perm], groups=groups[perm])]
                maps.append(fold_map(groups, sp))
                if repeat == 0:
                    prod = sp
            same = all(m == maps[0] for m in maps[1:])
            identical_flags.append(same)

            auc_p, f1_p, thr_p = metrics(crit, *oof_scores(X_geom, X_emb, y, prod), y)

            # --- 2. настоящие разные разбиения
            aucs, f1s, thrs = [], [], []
            for s in range(args.splits):
                sgkf = StratifiedGroupKFold(n_splits=ts.N_FOLDS, shuffle=True, random_state=s)
                sp = list(sgkf.split(X_geom, y, groups=groups))
                a, f, t = metrics(crit, *oof_scores(X_geom, X_emb, y, sp), y)
                aucs.append(a)
                f1s.append(f)
                thrs.append(t)

            out["criteria"][crit] = {
                "region": region, "n": int(len(y)), "n_pos": int(y.sum()), "emb_source": emb_src,
                "repeats_identical_split": bool(same),
                "production": {"auc": auc_p, "f1": f1_p, "threshold": thr_p},
                "shuffled_splits": {
                    "auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs, ddof=1)),
                    "auc_min": float(np.min(aucs)), "auc_max": float(np.max(aucs)),
                    "f1_mean": float(np.mean(f1s)), "f1_sd": float(np.std(f1s, ddof=1)),
                    "f1_min": float(np.min(f1s)), "f1_max": float(np.max(f1s)),
                    "thr_min": float(np.min(thrs)), "thr_max": float(np.max(thrs)),
                    "auc_all": [float(a) for a in aucs], "f1_all": f1s,
                },
            }
            print(f"{crit}: продакшен AUC {auc_p:.3f} F1 {f1_p:.3f} | "
                  f"{args.splits} разбиений AUC {np.mean(aucs):.3f}±{np.std(aucs, ddof=1):.3f} "
                  f"[{np.min(aucs):.3f}; {np.max(aucs):.3f}] F1 {np.mean(f1s):.3f}±{np.std(f1s, ddof=1):.3f} "
                  f"[{np.min(f1s):.3f}; {np.max(f1s):.3f}] | повторы дают одно разбиение: {same}")

    out["repeats_are_identical"] = bool(all(identical_flags))
    dst = ROOT / "docs" / "metrics_split_spread.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nзаписано: {dst}")

    # таблица для METRICS_REPORT.md
    print("\n| Критерий | n / полож. | AUC продакшен | AUC по 10 разбиениям | F1 продакшен | F1 по 10 разбиениям |")
    print("|---|---|---|---|---|---|")
    for crit, d in out["criteria"].items():
        s = d["shuffled_splits"]
        print(f"| `{crit}` | {d['n']} / {d['n_pos']} | {d['production']['auc']:.3f} | "
              f"{s['auc_mean']:.3f} ± {s['auc_sd']:.3f} (от {s['auc_min']:.3f} до {s['auc_max']:.3f}) | "
              f"{d['production']['f1']:.3f} | {s['f1_mean']:.3f} ± {s['f1_sd']:.3f} "
              f"(от {s['f1_min']:.3f} до {s['f1_max']:.3f}) |")


if __name__ == "__main__":
    main()
