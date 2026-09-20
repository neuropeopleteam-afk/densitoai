"""К11: выбор варианта предобработки контура A ПО КРИТЕРИЮ — только внутри nested CV.

Зачем. `docs/ROBUSTNESS_REPORT.md`: решения переворачиваются при гамме (26 / 40 % кадров)
и шуме σ=3 % (28 %). Причина — контур A не инвариантен к экспозиции (см. src/preprocess.py).
Инвариантная предобработка эту чувствительность убирает, но у критериев «посторонние предметы»
и «ROI» абсолютная плотность — это СИГНАЛ, а не шум, и канонизация его стирает. Поэтому вариант
предобработки выбирается по критерию, и выбор делается внутри nested, а не глазами.

Варианты контура A (src/extract_all_features.py: VARIANTS):
  baseline  — как в 2.1.0: маска тела жёстким порогом по сырому кадру, без канонизации;
  mask      — маска тела по сглаженному кадру (только устойчивость к шуму);
  canonical — mask + канонизация экспозиции (один параметр γ к эталону обучения).

Протокол (тот же, что К2 `tools/nested_gate.py`, чтобы числа были сравнимы):
  внешний контур  GroupKFold(shuffle=True, random_state=42+r), 5 фолдов × N_REPEATS повторов,
                  группы = компоненты связности (study, pixel_hash) на СЫРОЙ предобработке;
  внутренний      GroupKFold 3 фолда на внешнем train -> inner-OOF geom (для каждого варианта)
                  и inner-OOF emb (для эмбеддингов своей ветки);
  выбор варианта  максимум AUC стэка на inner-OOF (ранги внутри inner-OOF, w=0.5);
                  при равенстве — консервативно baseline;
  порог           правило из config.yaml thresholds_rule (как в train_stacked), считается на inner-OOF;
  внешний тест    модели, обученные на всём внешнем train; ранги относительно inner-OOF-референса.

Две ветки:
  base — контур A `baseline` + эмбеддинги 2.1.0 (data/embeddings*_baselinepre.npy), т.е. ровно продакшен;
  gate — контур A выбирается внутри nested + эмбеддинги на канонизированных кадрах.

Приёмка (как К2): прирост AUC ≥ 0.03 в ≥ 7/10 повторов И без потери macro-F1. Отдельно печатается
«не хуже» (нижняя граница парного ДИ прироста ≥ −0.02) — для варианта, который берут ради
устойчивости, а не ради метрики.

Запуск: N_REPEATS=10 python tools/preproc_gate.py
Отчёт:  docs/PREPROC_GATE_REPORT.md, решения — models/preproc_gate_decisions.json
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from train_stacked import (DATA_DIR, REGION_CRITERIA, CRITERION_GEOMETRY_COLS,  # noqa: E402
                           CRITERION_LABEL_COL, REGION_ROWS, EMB_SOURCE_BY_CRITERION,
                           threshold_rule_for, f1_optimal_threshold, prevalence_threshold)
from calibration_utils import threshold_by_rule as _threshold_by_rule  # noqa: E402


from nested_gate import (fit_geom, fit_emb, pct_rank, ref_rank, stack, safe_auc,  # noqa: E402
                         macro_f1, boot_ci, connected_groups, impute, degenerate)


def threshold_by_rule(rule, y, scores, min_positives_for_f1=15):
    """Обёртка: те же функции порога, что в train_stacked."""
    return _threshold_by_rule(rule, y, scores, f1_optimal_threshold, prevalence_threshold,
                              min_pos_f1=min_positives_for_f1)

VARIANTS = ["baseline", "mask", "canonical"]
VARIANT_FILES = {"baseline": "geometry_features_baseline.csv",
                 "mask": "geometry_features_mask.csv",
                 "canonical": "geometry_features.csv"}
# Эмбеддинги: ветка base — как в 2.1.0, ветка gate — на канонизированных кадрах.
EMB_FILES_BASE = {"imagenet": "embeddings_baselinepre.npy", "densito": "embeddings_densito_baselinepre.npy"}
EMB_FILES_GATE = {"imagenet": "embeddings.npy", "densito": "embeddings_densito.npy"}

N_OUTER, N_INNER = 5, 3
N_REPEATS = int(os.environ.get("N_REPEATS", 10))
W = 0.5                      # продакшен-вес стэкинга (К2: вентиль не принят)
GAIN_MIN, GAIN_REPEATS = 0.03, 7
NONINFERIOR_MARGIN = -0.02   # «не хуже» для решения, принимаемого ради устойчивости

# Две стадии, чтобы решения по контурам не смешивались:
#   STAGE=a — выбор варианта контура A по критерию; эмбеддинги в обеих ветках одинаковые (2.1.0);
#   STAGE=b — вариант A зафиксирован решением стадии a в ОБЕИХ ветках, сравниваются
#             только эмбеддинги (сырые против канонизированных).
STAGE = os.environ.get("PREPROC_STAGE", "a").lower()
EMB_BRANCH = {"base": "base", "gate": "base" if STAGE == "a" else "gate"}
FIXED_VARIANTS = json.loads(os.environ.get("PREPROC_FIXED_VARIANTS", "{}"))
OUT = ROOT / "outputs" / f"preproc_gate_{STAGE}"
OUT.mkdir(parents=True, exist_ok=True)


def load_all(region):
    """Геометрия по вариантам + эмбеддинги по ветвям + группы. Порядок строк проверяется."""
    rows = REGION_ROWS[region]
    geoms = {}
    for v in VARIANTS:
        g = pd.read_csv(DATA_DIR / VARIANT_FILES[v])
        geoms[v] = g[g["region"].isin(rows)].reset_index(drop=True)
    ref = geoms["baseline"]
    for v in VARIANTS[1:]:
        assert (geoms[v]["file_path"].values == ref["file_path"].values).all(), f"порядок строк {v} != baseline"

    lab = pd.read_csv(DATA_DIR / "labels_for_embeddings.csv")
    eidx = np.nonzero(lab["region"].isin(rows).values)[0]
    assert (lab.loc[eidx, "file_path"].values == ref["file_path"].values).all(), "порядок строк эмбеддингов"
    embs = {}
    for branch, files in (("base", EMB_FILES_BASE), ("gate", EMB_FILES_GATE)):
        embs[branch] = {src: np.load(DATA_DIR / fn)[eidx] for src, fn in files.items()
                        if (DATA_DIR / fn).exists()}
        assert "imagenet" in embs[branch], f"нет эмбеддингов imagenet для ветки {branch}"

    hashes = pd.read_csv(ROOT / "outputs" / "pixel_hashes.csv")
    hmap = dict(zip(hashes["file_path"], hashes["pixel_hash"]))
    groups = connected_groups(ref["study"].values, ref["file_path"].map(hmap).values)
    return geoms, embs, groups


def inner_oof_multi(Xs, E, y, groups, seed):
    """inner-OOF: geom по каждому варианту (Xs: {вариант: X}) и emb один раз."""
    og = {v: np.full(len(y), np.nan) for v in Xs}
    oe = np.full(len(y), np.nan)
    for tr, va in GroupKFold(n_splits=N_INNER, shuffle=True, random_state=seed).split(next(iter(Xs.values())), groups=groups):
        if degenerate(y[tr]):
            continue
        for v, X in Xs.items():
            og[v][va] = fit_geom(X[tr], y[tr])(X[va])
        oe[va] = fit_emb(E[tr], y[tr])(E[va])
    return og, oe


def run_criterion(region, crit, geoms, embs, groups, log):
    label_col = CRITERION_LABEL_COL.get(crit, crit)
    y_all = geoms["baseline"][label_col].values
    valid = ~pd.isna(y_all)
    y = y_all[valid].astype(int)
    cols = CRITERION_GEOMETRY_COLS[crit]
    Xraw = {v: geoms[v][cols].values.astype(np.float64)[valid] for v in VARIANTS}
    src = EMB_SOURCE_BY_CRITERION.get(crit, "imagenet")
    Emb = {b: embs[EMB_BRANCH[b]].get(src, embs[EMB_BRANCH[b]]["imagenet"])[valid] for b in ("base", "gate")}
    g = groups[valid]
    studies = geoms["baseline"]["study"].values[valid]
    files = geoms["baseline"]["file_path"].values[valid]
    rule = threshold_rule_for(crit)
    n = len(y)
    # Стадия a: база = baseline, кандидаты = все три. Стадия b: вариант A зафиксирован в обеих ветках.
    base_variant = FIXED_VARIANTS.get(crit, "baseline")
    candidates = [base_variant] if STAGE == "b" else VARIANTS
    log(f"\n=== {region}/{crit}: n={n}, n_pos={int(y.sum())}, emb='{src}' "
        f"(base={EMB_BRANCH['base']}, gate={EMB_BRANCH['gate']}), правило порога '{rule}', "
        f"контур A: база '{base_variant}', кандидаты {candidates}, geom={cols}")

    score = {b: np.full((N_REPEATS, n), np.nan) for b in ("base", "gate")}
    pred = {b: np.full((N_REPEATS, n), np.nan) for b in ("base", "gate")}
    per_repeat, fold_rows = [], []

    for r in range(N_REPEATS):
        splitter = GroupKFold(n_splits=N_OUTER, shuffle=True, random_state=42 + r)
        for k, (tr, te) in enumerate(splitter.split(Xraw["baseline"], groups=g)):
            if degenerate(y[tr]):
                continue
            # медианы для импутации — только по внешнему train, отдельно на каждый вариант
            X = {v: impute(Xraw[v], np.nanmedian(Xraw[v][tr], axis=0)) for v in VARIANTS}
            n_pos_tr = int(y[tr].sum())

            # --- ветка base: контур A базовый для стадии, эмбеддинги 2.1.0
            og_b, oe_b = inner_oof_multi({base_variant: X[base_variant][tr]}, Emb["base"][tr], y[tr], g[tr],
                                         seed=1000 * r + k)
            # --- ветка gate: кандидаты контура A, эмбеддинги своей ветки
            og_g, oe_g = inner_oof_multi({v: X[v][tr] for v in candidates}, Emb["gate"][tr], y[tr], g[tr],
                                         seed=1000 * r + k)

            def inner_stack(og_v, oe_v):
                ok = ~np.isnan(og_v) & ~np.isnan(oe_v)
                if ok.sum() < 2 or len(np.unique(y[tr][ok])) < 2:
                    return None, None, None
                rg, re_ = pct_rank(og_v[ok]), pct_rank(oe_v[ok])
                return ok, stack(rg, re_, W), safe_auc(y[tr][ok], stack(rg, re_, W))

            ok_b, s_in_b, auc_in_b = inner_stack(og_b[base_variant], oe_b)
            inner_auc = {v: None for v in VARIANTS}
            for v in candidates:
                _, _, a = inner_stack(og_g[v], oe_g)
                inner_auc[v] = a
            if ok_b is None or all(a is None or not np.isfinite(a) for a in inner_auc.values()):
                continue
            best = np.nanmax([a for a in inner_auc.values() if a is not None])
            v_sel = sorted([v for v, a in inner_auc.items() if a is not None and a >= best - 1e-12],
                           key=lambda v: VARIANTS.index(v))[0]
            ok_g, s_in_g, _ = inner_stack(og_g[v_sel], oe_g)

            thr_b, m_b = threshold_by_rule(rule, y[tr][ok_b], s_in_b, min_positives_for_f1=15)
            thr_g, m_g = threshold_by_rule(rule, y[tr][ok_g], s_in_g, min_positives_for_f1=15)

            for branch, v_use, og_use, oe_use, thr in (("base", base_variant, og_b[base_variant], oe_b, thr_b),
                                                       ("gate", v_sel, og_g[v_sel], oe_g, thr_g)):
                pg = fit_geom(X[v_use][tr], y[tr])(X[v_use][te])
                pe = fit_emb(Emb[branch][tr], y[tr])(Emb[branch][te])
                s = stack(ref_rank(pg, og_use), ref_rank(pe, oe_use), W)
                score[branch][r, te] = s
                pred[branch][r, te] = (s >= thr).astype(int)

            fold_rows.append({"region": region, "criterion": crit, "repeat": r, "fold": k,
                              "n_train": len(tr), "n_pos_train": n_pos_tr, "n_test": len(te),
                              "variant_selected": v_sel, "thr_base": thr_b, "thr_gate": thr_g,
                              "thr_method": m_g,
                              **{f"inner_auc_{v}": inner_auc[v] for v in VARIANTS},
                              "inner_auc_base_branch": auc_in_b})

        m = ~np.isnan(score["base"][r]) & ~np.isnan(score["gate"][r])
        if m.sum() < 2:
            continue
        yb = y[m]
        row = {"region": region, "criterion": crit, "repeat": r, "n_scored": int(m.sum()),
               "auc_base": safe_auc(yb, score["base"][r][m]), "auc_gate": safe_auc(yb, score["gate"][r][m]),
               "f1pos_base": f1_score(yb, pred["base"][r][m], zero_division=0),
               "f1pos_gate": f1_score(yb, pred["gate"][r][m], zero_division=0),
               "macro_f1_base": macro_f1(yb, pred["base"][r][m]),
               "macro_f1_gate": macro_f1(yb, pred["gate"][r][m]),
               "variants_folds": " ".join(fr["variant_selected"] for fr in fold_rows
                                          if fr["criterion"] == crit and fr["repeat"] == r)}
        row["delta_auc"] = row["auc_gate"] - row["auc_base"]
        row["delta_macro_f1"] = row["macro_f1_gate"] - row["macro_f1_base"]
        row["delta_f1pos"] = row["f1pos_gate"] - row["f1pos_base"]
        per_repeat.append(row)
        log(f"  repeat {r}: AUC base={row['auc_base']:.3f} gate={row['auc_gate']:.3f} d={row['delta_auc']:+.3f} "
            f"| F1(+) {row['f1pos_base']:.3f}->{row['f1pos_gate']:.3f} | macroF1 d={row['delta_macro_f1']:+.3f} "
            f"| {row['variants_folds']}")

    rep = pd.DataFrame(per_repeat)
    folds = pd.DataFrame(fold_rows)
    sb = np.nanmean(score["base"], axis=0)
    sg = np.nanmean(score["gate"], axis=0)
    pb = (np.nanmean(pred["base"], axis=0) >= 0.5).astype(int)
    pg_ = (np.nanmean(pred["gate"], axis=0) >= 0.5).astype(int)
    delta_ci = boot_ci(studies, lambda idx: safe_auc(y[idx], sg[idx]) - safe_auc(y[idx], sb[idx]))
    counts = folds["variant_selected"].value_counts().reindex(VARIANTS, fill_value=0)
    n_gain = int((rep["delta_auc"] >= GAIN_MIN).sum()) if len(rep) else 0
    mean_d = float(rep["delta_auc"].mean()) if len(rep) else float("nan")
    accepted = bool(len(rep) and n_gain >= GAIN_REPEATS
                    and rep["macro_f1_gate"].mean() >= rep["macro_f1_base"].mean() - 1e-12)
    noninferior = bool(np.isfinite(delta_ci[0]) and delta_ci[0] >= NONINFERIOR_MARGIN)
    decision = {"region": region, "criterion": crit, "n": n, "n_pos": int(y.sum()), "emb_source": src,
                "threshold_rule": rule,
                "variant_mode": str(counts.idxmax()), **{f"freq_{v}": int(counts[v]) for v in VARIANTS},
                "n_repeats": len(rep), "n_repeats_gain_ge_0.03": n_gain,
                "mean_delta_auc": mean_d,
                "mean_delta_macro_f1": float(rep["delta_macro_f1"].mean()) if len(rep) else float("nan"),
                "mean_delta_f1pos": float(rep["delta_f1pos"].mean()) if len(rep) else float("nan"),
                "auc_base_pooled": float(safe_auc(y, sb)), "auc_gate_pooled": float(safe_auc(y, sg)),
                "delta_auc_pooled_ci": list(delta_ci),
                "macro_f1_base_pooled": float(macro_f1(y, pb)), "macro_f1_gate_pooled": float(macro_f1(y, pg_)),
                "accepted_gain_rule": accepted, "noninferior": noninferior}
    log(f"  --> вариант mode={decision['variant_mode']} freq={dict(counts)}; "
        f"прирост ≥0.03 в {n_gain}/{len(rep)}, средний ΔAUC={mean_d:+.3f}, "
        f"парный ДИ [{delta_ci[0]:+.3f}; {delta_ci[2]:+.3f}] => "
        f"{'ПРИНЯТ по приросту' if accepted else ('не хуже' if noninferior else 'ХУЖЕ')}")
    pd.DataFrame({"study": studies, "file_path": files, "y_true": y,
                  "score_base_mean": sb, "score_gate_mean": sg,
                  "pred_base_vote": pb, "pred_gate_vote": pg_}).to_csv(
        OUT / f"preproc_oof_{region}_{crit}.csv", index=False)
    return rep, folds, decision


def main():
    t0 = time.time()
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    reps, all_folds, decisions = [], [], []
    for region, criteria in REGION_CRITERIA.items():
        geoms, embs, groups = load_all(region)
        for crit in criteria:
            rep, folds, dec = run_criterion(region, crit, geoms, embs, groups, log)
            reps.append(rep)
            all_folds.append(folds)
            decisions.append(dec)

    pd.concat(reps).to_csv(OUT / "per_repeat.csv", index=False)
    pd.concat(all_folds).to_csv(OUT / "per_fold.csv", index=False)
    (ROOT / "models" / f"preproc_gate_decisions_{STAGE}.json").write_text(
        json.dumps({"protocol": {"stage": STAGE, "outer": N_OUTER, "inner": N_INNER, "repeats": N_REPEATS,
                                 "w_stacking": W, "gain_min": GAIN_MIN, "gain_repeats": GAIN_REPEATS,
                                 "noninferior_margin": NONINFERIOR_MARGIN,
                                 "emb_branches": EMB_BRANCH, "fixed_variants": FIXED_VARIANTS,
                                 "groups": "connected components (study, pixel_hash) на сырой предобработке",
                                 "variants": VARIANTS},
                    "decisions": decisions}, ensure_ascii=False, indent=2))
    log(f"\nстадия {STAGE}: готово за {time.time() - t0:.0f} с; "
        f"решения -> models/preproc_gate_decisions_{STAGE}.json")
    (OUT / "log.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
