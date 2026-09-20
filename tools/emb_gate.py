"""К13: выбор источника эмбеддингов контура B ПО КРИТЕРИЮ — только внутри nested CV.

Зачем. GPU-эксперимент К13 (приватный отчёт `K13_backbones`) обучил два новых бэкбона
EfficientNet-B0 с loss инвариантности эмбеддинга к гамме и шуму: `densito_inv` (полный пул
15 633 плитки) и `densito_inv_free` (9 893 плитки, без данных с несвободной лицензией).
На отдельной парной оценке контура B они дали крупный прирост на `sp_axis` и `hip_pos`.
Эта оценка считалась ОДНИМ протоколом (PCA-32 + логрегрессия, 10 повторов GroupKFold),
то есть выбор бэкбона делался «на тех же данных, на которых потом мерили». Внедрять по такой
цифре нельзя — источник выбирается здесь, внутри nested CV, на полном стэке (контур A + контур B),
и принимается только по правилу проекта.

Протокол (тот же, что К2 `tools/nested_gate.py` и К11 `tools/preproc_gate.py`, чтобы числа были сравнимы):
  внешний контур  GroupKFold(shuffle=True, random_state=42+r), 5 фолдов × N_REPEATS повторов,
                  группы = компоненты связности (study, pixel_hash);
  внутренний      GroupKFold 3 фолда на внешнем train -> inner-OOF geom (один раз, вариант
                  предобработки продакшена) и inner-OOF emb для КАЖДОГО источника-кандидата;
  выбор источника максимум AUC стэка на inner-OOF (ранги внутри inner-OOF, w=0.5);
                  при равенстве — консервативно продакшен-источник;
  порог           правило из config.yaml thresholds_rule (как в train_stacked), на inner-OOF;
  внешний тест    модели, обученные на всём внешнем train; ранги относительно inner-OOF-референса.

Ветки: base  — источник продакшена (config.yaml embeddings.source_by_criterion);
       gate  — источник выбирается внутри nested из кандидатов (честная оценка процедуры выбора);
       fixed:<источник> — источник задан жёстко на всех фолдах (оценка ровно того изменения,
               которое пойдёт в прод, на тех же фолдах и с тем же порогом-правилом).

Приёмка (как К2/К11): прирост AUC ≥ 0.03 в ≥ 7/10 повторов И без потери macro-F1, И источник-кандидат
выбран в большинстве фолдов. Отдельно печатается «не хуже» (нижняя граница парного ДИ ≥ −0.02) —
для источника, который берут ради устойчивости, а не ради метрики.

Запуск: N_REPEATS=10 python tools/emb_gate.py
Отчёт:  docs/EMB_GATE_REPORT.md (пишется отдельно), решения — models/emb_gate_decisions.json
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
                           CRITERION_LABEL_COL, REGION_ROWS, EMB_FILES,
                           GEOM_VARIANT_FILES, preproc_for, emb_source_config, emb_file_for,
                           threshold_rule_for, f1_optimal_threshold, prevalence_threshold)
from calibration_utils import threshold_by_rule as _threshold_by_rule  # noqa: E402
from nested_gate import (fit_geom, fit_emb, pct_rank, ref_rank, stack, safe_auc,  # noqa: E402
                         macro_f1, boot_ci, connected_groups, impute, degenerate)


def threshold_by_rule(rule, y, scores, min_positives_for_f1=15):
    """Обёртка: те же функции порога, что в train_stacked."""
    return _threshold_by_rule(rule, y, scores, f1_optimal_threshold, prevalence_threshold,
                              min_pos_f1=min_positives_for_f1)


# Порядок = порядок разрешения ничьих после продакшен-источника (консервативно: сначала старое).
SOURCES = ["imagenet", "densito", "densito_inv", "densito_inv_free"]
N_OUTER, N_INNER = 5, 3
N_REPEATS = int(os.environ.get("N_REPEATS", 10))
# Базовый источник по критерию можно задать явно (JSON): нужно, чтобы после внедрения
# решения воспроизводить тот же прогон с исходной базой (иначе базой станет уже новый источник).
BASE_OVERRIDE = json.loads(os.environ.get("EMB_GATE_BASE", "{}"))
W = 0.5                      # продакшен-вес стэкинга (К2: вентиль не принят)
# Правило К2/К11: прирост ≥ 0.03 в ≥ 7 из 10 повторов, т.е. в ≥70 % повторов при любом N_REPEATS.
GAIN_MIN, GAIN_FRACTION = 0.03, 0.7
GAIN_REPEATS = int(-(-GAIN_FRACTION * N_REPEATS // 1))
NONINFERIOR_MARGIN = -0.02   # «не хуже» для решения, принимаемого ради устойчивости
OUT = ROOT / "outputs" / "emb_gate"
OUT.mkdir(parents=True, exist_ok=True)


def load_region(region):
    """Геометрия (все варианты), эмбеддинги по (источник, вариант), группы. Порядок строк проверяется."""
    rows = REGION_ROWS[region]
    geoms = {}
    for v, fn in GEOM_VARIANT_FILES.items():
        g = pd.read_csv(DATA_DIR / fn)
        geoms[v] = g[g["region"].isin(rows)].reset_index(drop=True)
    ref = geoms["baseline"]
    for v in geoms:
        assert (geoms[v]["file_path"].values == ref["file_path"].values).all(), f"порядок строк {v} != baseline"

    lab = pd.read_csv(DATA_DIR / "labels_for_embeddings.csv")
    eidx = np.nonzero(lab["region"].isin(rows).values)[0]
    assert (lab.loc[eidx, "file_path"].values == ref["file_path"].values).all(), "порядок строк эмбеддингов"
    embs = {}
    for src in SOURCES:
        for variant in GEOM_VARIANT_FILES:
            f = DATA_DIR / emb_file_for(src, variant)
            if f.exists():
                embs[(src, variant)] = np.load(f)[eidx]
    assert ("imagenet", "baseline") in embs, "нет data/embeddings.npy"

    hashes = pd.read_csv(ROOT / "outputs" / "pixel_hashes.csv")
    hmap = dict(zip(hashes["file_path"], hashes["pixel_hash"]))
    groups = connected_groups(ref["study"].values, ref["file_path"].map(hmap).values)
    return geoms, embs, groups


def inner_oof_multi(X, Es, y, groups, seed):
    """inner-OOF: geom один раз (вариант продакшена) и emb по каждому источнику-кандидату."""
    og = np.full(len(y), np.nan)
    oe = {s: np.full(len(y), np.nan) for s in Es}
    for tr, va in GroupKFold(n_splits=N_INNER, shuffle=True, random_state=seed).split(X, groups=groups):
        if degenerate(y[tr]):
            continue
        og[va] = fit_geom(X[tr], y[tr])(X[va])
        for s, E in Es.items():
            oe[s][va] = fit_emb(E[tr], y[tr])(E[va])
    return og, oe


def run_criterion(region, crit, geoms, embs, groups, log):
    label_col = CRITERION_LABEL_COL.get(crit, crit)
    y_all = geoms["baseline"][label_col].values
    valid = ~pd.isna(y_all)
    y = y_all[valid].astype(int)
    cols = CRITERION_GEOMETRY_COLS[crit]
    pp = preproc_for(crit)
    geom_variant = pp["geom"] if pp["geom"] in geoms else "baseline"
    emb_variant = pp["emb"]
    Xraw = geoms[geom_variant][cols].values.astype(np.float64)[valid]

    base_src = BASE_OVERRIDE.get(crit) or emb_source_config().get(crit, "imagenet")
    if (base_src, emb_variant) not in embs:
        log(f"  [warn] {crit}: нет эмбеддингов {base_src}/{emb_variant} — база imagenet/baseline")
        base_src, emb_variant = "imagenet", "baseline"
    # кандидаты: все источники, у которых есть файл эмбеддингов В ТОМ ЖЕ варианте предобработки
    avail = [s for s in SOURCES if (s, emb_variant) in embs]
    candidates = [base_src] + [s for s in avail if s != base_src]
    Emb = {s: embs[(s, emb_variant)][valid] for s in candidates}
    g = groups[valid]
    studies = geoms["baseline"]["study"].values[valid]
    files = geoms["baseline"]["file_path"].values[valid]
    rule = threshold_rule_for(crit)
    n = len(y)
    log(f"\n=== {region}/{crit}: n={n}, n_pos={int(y.sum())}, предобработка geom={geom_variant} emb={emb_variant}, "
        f"база '{base_src}', кандидаты {candidates}, правило порога '{rule}', geom={cols}")

    branches = ["base", "gate"] + [f"fixed:{s}" for s in candidates]
    score = {b: np.full((N_REPEATS, n), np.nan) for b in branches}
    pred = {b: np.full((N_REPEATS, n), np.nan) for b in branches}
    per_repeat, fold_rows = [], []

    for r in range(N_REPEATS):
        splitter = GroupKFold(n_splits=N_OUTER, shuffle=True, random_state=42 + r)
        for k, (tr, te) in enumerate(splitter.split(Xraw, groups=g)):
            if degenerate(y[tr]):
                continue
            X = impute(Xraw, np.nanmedian(Xraw[tr], axis=0))   # медианы — только по внешнему train
            n_pos_tr = int(y[tr].sum())
            og, oe = inner_oof_multi(X[tr], {s: Emb[s][tr] for s in candidates}, y[tr], g[tr],
                                     seed=1000 * r + k)

            def inner_stack(oe_v):
                ok = ~np.isnan(og) & ~np.isnan(oe_v)
                if ok.sum() < 2 or len(np.unique(y[tr][ok])) < 2:
                    return None, None, None
                rg, re_ = pct_rank(og[ok]), pct_rank(oe_v[ok])
                s = stack(rg, re_, W)
                return ok, s, safe_auc(y[tr][ok], s)

            inner_auc = {}
            for s in candidates:
                _, _, a = inner_stack(oe[s])
                inner_auc[s] = a
            if all(a is None or not np.isfinite(a) for a in inner_auc.values()):
                continue
            best = np.nanmax([a for a in inner_auc.values() if a is not None])
            # ничья -> первый по списку candidates (там продакшен-источник первым)
            s_sel = [s for s in candidates if inner_auc[s] is not None and inner_auc[s] >= best - 1e-12][0]

            ok_b, s_in_b, _ = inner_stack(oe[base_src])
            ok_g, s_in_g, _ = inner_stack(oe[s_sel])
            if ok_b is None or ok_g is None:
                continue
            thr_b, m_b = threshold_by_rule(rule, y[tr][ok_b], s_in_b, min_positives_for_f1=15)
            thr_g, m_g = threshold_by_rule(rule, y[tr][ok_g], s_in_g, min_positives_for_f1=15)

            pg = fit_geom(X[tr], y[tr])(X[te])
            rg_te = ref_rank(pg, og)
            # кэш внешних предсказаний контура B по источнику (используется и веткой gate, и fixed)
            pe_cache, thr_cache = {}, {}
            for s in candidates:
                pe_cache[s] = fit_emb(Emb[s][tr], y[tr])(Emb[s][te])
                ok_s, s_in_s, _ = inner_stack(oe[s])
                thr_cache[s] = threshold_by_rule(rule, y[tr][ok_s], s_in_s,
                                                 min_positives_for_f1=15)[0] if ok_s is not None else np.nan
            branch_src = [("base", base_src, thr_b), ("gate", s_sel, thr_g)]
            branch_src += [(f"fixed:{s}", s, thr_cache[s]) for s in candidates]
            for branch, s_use, thr in branch_src:
                if not np.isfinite(thr):
                    continue
                s = stack(rg_te, ref_rank(pe_cache[s_use], oe[s_use]), W)
                score[branch][r, te] = s
                pred[branch][r, te] = (s >= thr).astype(int)

            fold_rows.append({"region": region, "criterion": crit, "repeat": r, "fold": k,
                              "n_train": len(tr), "n_pos_train": n_pos_tr, "n_test": len(te),
                              "source_selected": s_sel, "thr_base": thr_b, "thr_gate": thr_g,
                              "thr_method": m_g,
                              **{f"inner_auc_{s}": inner_auc.get(s) for s in SOURCES}})

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
               "sources_folds": " ".join(fr["source_selected"] for fr in fold_rows
                                         if fr["criterion"] == crit and fr["repeat"] == r)}
        row["delta_auc"] = row["auc_gate"] - row["auc_base"]
        row["delta_macro_f1"] = row["macro_f1_gate"] - row["macro_f1_base"]
        row["delta_f1pos"] = row["f1pos_gate"] - row["f1pos_base"]
        for s in candidates:
            b = f"fixed:{s}"
            mf = ~np.isnan(score["base"][r]) & ~np.isnan(score[b][r])
            if mf.sum() < 2:
                continue
            yf = y[mf]
            row[f"auc_{b}"] = safe_auc(yf, score[b][r][mf])
            row[f"macro_f1_{b}"] = macro_f1(yf, pred[b][r][mf])
            row[f"f1pos_{b}"] = f1_score(yf, pred[b][r][mf], zero_division=0)
            row[f"delta_auc_{b}"] = row[f"auc_{b}"] - safe_auc(yf, score["base"][r][mf])
            row[f"delta_macro_f1_{b}"] = row[f"macro_f1_{b}"] - macro_f1(yf, pred["base"][r][mf])
        per_repeat.append(row)
        log(f"  repeat {r}: AUC base={row['auc_base']:.3f} gate={row['auc_gate']:.3f} d={row['delta_auc']:+.3f} "
            f"| F1(+) {row['f1pos_base']:.3f}->{row['f1pos_gate']:.3f} | macroF1 d={row['delta_macro_f1']:+.3f} "
            f"| {row['sources_folds']}")

    rep = pd.DataFrame(per_repeat)
    folds = pd.DataFrame(fold_rows)
    sb = np.nanmean(score["base"], axis=0)
    sg = np.nanmean(score["gate"], axis=0)
    pb = (np.nanmean(pred["base"], axis=0) >= 0.5).astype(int)
    pg_ = (np.nanmean(pred["gate"], axis=0) >= 0.5).astype(int)
    delta_ci = boot_ci(studies, lambda idx: safe_auc(y[idx], sg[idx]) - safe_auc(y[idx], sb[idx]))
    auc_base_repeats = rep["auc_base"].dropna() if len(rep) else pd.Series(dtype=float)
    counts = folds["source_selected"].value_counts().reindex(SOURCES, fill_value=0)
    n_folds = int(counts.sum())
    n_gain = int((rep["delta_auc"] >= GAIN_MIN).sum()) if len(rep) else 0
    mean_d = float(rep["delta_auc"].mean()) if len(rep) else float("nan")
    mode_src = str(counts.idxmax())
    majority = bool(n_folds and counts[mode_src] > n_folds / 2)
    accepted = bool(len(rep) and n_gain >= GAIN_REPEATS and majority and mode_src != base_src
                    and rep["macro_f1_gate"].mean() >= rep["macro_f1_base"].mean() - 1e-12)
    noninferior = bool(np.isfinite(delta_ci[0]) and delta_ci[0] >= NONINFERIOR_MARGIN)
    decision = {"region": region, "criterion": crit, "n": n, "n_pos": int(y.sum()),
                "base_source": base_src, "candidates": candidates,
                "preproc_geom": geom_variant, "preproc_emb": emb_variant, "threshold_rule": rule,
                "source_mode": mode_src, "mode_majority": majority,
                **{f"freq_{s}": int(counts[s]) for s in SOURCES},
                "n_folds": n_folds, "n_repeats": len(rep), "n_repeats_gain_ge_0.03": n_gain,
                "mean_delta_auc": mean_d,
                "mean_delta_macro_f1": float(rep["delta_macro_f1"].mean()) if len(rep) else float("nan"),
                "mean_delta_f1pos": float(rep["delta_f1pos"].mean()) if len(rep) else float("nan"),
                "auc_base_mean_over_repeats": float(auc_base_repeats.mean()) if len(auc_base_repeats) else float("nan"),
                "auc_base_pooled": float(safe_auc(y, sb)), "auc_gate_pooled": float(safe_auc(y, sg)),
                "delta_auc_pooled_ci": list(delta_ci),
                "macro_f1_base_pooled": float(macro_f1(y, pb)), "macro_f1_gate_pooled": float(macro_f1(y, pg_)),
                "accepted_gain_rule": accepted, "noninferior": noninferior,
                "recommended_source": mode_src if accepted else base_src}

    # --- жёстко заданный источник: ровно то изменение, которое пойдёт в прод
    fixed_stats = {}
    for s in candidates:
        b = f"fixed:{s}"
        col_d, col_m = f"delta_auc_{b}", f"delta_macro_f1_{b}"
        if not len(rep) or col_d not in rep:
            continue
        d = rep[col_d].dropna()
        sf = np.nanmean(score[b], axis=0)
        pf = (np.nanmean(pred[b], axis=0) >= 0.5).astype(int)
        ci = boot_ci(studies, lambda idx, sf=sf: safe_auc(y[idx], sf[idx]) - safe_auc(y[idx], sb[idx]))
        n_gain_f = int((d >= GAIN_MIN).sum())
        dm = rep[col_m].dropna()
        auc_col = rep[f"auc_fixed:{s}"].dropna() if f"auc_fixed:{s}" in rep else pd.Series(dtype=float)
        auc_ci = boot_ci(studies, lambda idx, sf=sf: safe_auc(y[idx], sf[idx]))
        fixed_stats[s] = {"auc_mean_over_repeats": float(auc_col.mean()) if len(auc_col) else float("nan"),
                          "auc_sd_over_repeats": float(auc_col.std()) if len(auc_col) > 1 else float("nan"),
                          "auc_ci": list(auc_ci),
                          "mean_delta_auc": float(d.mean()) if len(d) else float("nan"),
                          "n_repeats": int(len(d)), "n_repeats_gain_ge_0.03": n_gain_f,
                          "n_repeats_positive": int((d > 0).sum()),
                          "mean_delta_macro_f1": float(dm.mean()) if len(dm) else float("nan"),
                          "auc_pooled": float(safe_auc(y, sf)), "macro_f1_pooled": float(macro_f1(y, pf)),
                          "delta_auc_pooled_ci": list(ci),
                          "accepted_gain_rule": bool(len(d) and n_gain_f >= GAIN_REPEATS
                                                     and len(dm) and dm.mean() >= -1e-12 and s != base_src),
                          "noninferior": bool(np.isfinite(ci[0]) and ci[0] >= NONINFERIOR_MARGIN)}
        log(f"      fixed {s:17s}: ΔAUC={fixed_stats[s]['mean_delta_auc']:+.3f} "
            f"(≥0.03 в {n_gain_f}/{len(d)}, в плюсе {fixed_stats[s]['n_repeats_positive']}/{len(d)}), "
            f"Δmacro-F1={fixed_stats[s]['mean_delta_macro_f1']:+.3f}, "
            f"AUC pooled={fixed_stats[s]['auc_pooled']:.3f}, ДИ [{ci[0]:+.3f}; {ci[2]:+.3f}]"
            f"{' => ПРОХОДИТ ПРАВИЛО' if fixed_stats[s]['accepted_gain_rule'] else ''}")
    decision["fixed"] = fixed_stats
    log(f"  --> источник mode={mode_src} freq={dict(counts)} (большинство: {majority}); "
        f"прирост ≥0.03 в {n_gain}/{len(rep)}, средний ΔAUC={mean_d:+.3f}, "
        f"Δmacro-F1={decision['mean_delta_macro_f1']:+.3f}, парный ДИ [{delta_ci[0]:+.3f}; {delta_ci[2]:+.3f}] => "
        f"{'ПРИНЯТ ' + mode_src if accepted else ('не хуже, но не принят' if noninferior else 'НЕ ПРИНЯТ')}")
    pd.DataFrame({"study": studies, "file_path": files, "y_true": y,
                  "score_base_mean": sb, "score_gate_mean": sg,
                  "pred_base_vote": pb, "pred_gate_vote": pg_}).to_csv(
        OUT / f"emb_oof_{region}_{crit}.csv", index=False)
    return rep, folds, decision


def main():
    t0 = time.time()
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    reps, all_folds, decisions = [], [], []
    for region, criteria in REGION_CRITERIA.items():
        geoms, embs, groups = load_region(region)
        for crit in criteria:
            rep, folds, dec = run_criterion(region, crit, geoms, embs, groups, log)
            reps.append(rep)
            all_folds.append(folds)
            decisions.append(dec)

    pd.concat(reps).to_csv(OUT / "per_repeat.csv", index=False)
    pd.concat(all_folds).to_csv(OUT / "per_fold.csv", index=False)
    (ROOT / "models" / "emb_gate_decisions.json").write_text(
        json.dumps({"protocol": {"outer": N_OUTER, "inner": N_INNER, "repeats": N_REPEATS,
                                 "w_stacking": W, "gain_min": GAIN_MIN, "gain_repeats": GAIN_REPEATS,
                                 "noninferior_margin": NONINFERIOR_MARGIN,
                                 "sources": SOURCES, "emb_files": EMB_FILES,
                                 "groups": "connected components (study, pixel_hash)",
                                 "base": "config.yaml embeddings.source_by_criterion"},
                    "decisions": decisions}, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\nготово за {time.time() - t0:.0f} с; решения -> models/emb_gate_decisions.json")
    (OUT / "log.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
