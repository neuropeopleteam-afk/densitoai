"""Nested-проверка кандидатов «синтетика для sp_pos» (только критерий sp_pos, позвоночник).

Протокол — как tools/emb_gate.py / tools/nested_gate.py: внешний GroupKFold(5, shuffle, random_state=42+r)
x N_REPEATS повторов, группы = компоненты (study, pixel_hash); внутренний GroupKFold(3) на внешнем train ->
inner-OOF каждой ветки; ранги внешнего теста относительно inner-OOF-референса (inference.percentile_rank);
порог — правило config.yaml thresholds_rule (sp_pos: prevalence) на inner-OOF стэке.

База: контур A = geom canonical [center_offset_ratio, bone_width_ratio] (LR C=1, balanced),
      контур B = эмбеддинги densito/canonical (PCA 32 -> LR C=0.1), стэк рангов 0.5/0.5.
Кандидаты (каждый — на тех же фолдах, что база):
  K1   — контур B на эмбеддингах synth (бэкбон, дообученный синтетикой) вместо densito;
  K2   — контур A + скалярный скор головы «дефект» (логит) как третий признак геометрии; контур B — densito;
  K2b  — контур A + предсказанное смещение окна вверх (pred_oy; в v1/v2 — pred_bottom) как третий признак; контур B — densito;
  K3   — стэк из трёх рангов: geom, densito, synth (по 1/3);
  K3s  — стэк из трёх рангов: geom, densito, ранг скора головы (по 1/3).
Суффикс `_oof` — эмбеддинги/скор от бэкбона фолда GroupKFold(5), который не видел этот кадр
(честная версия: финальный бэкбон/голова обучены на синтетике из всех кадров позвоночника, включая тестовые,
хотя и без меток ТЗ). Справочно: те же K1/K1_oof для sp_axis и sp_art (не ухудшает ли новый бэкбон соседей).

Выход (--out): per_repeat_<cand>.csv (repeat, row_id, y, group, score_base, score_cand), summary.json,
repeats_<crit>.csv (метрики по повторам).
Запуск: OMP_NUM_THREADS=1 DENSITO_ROOT=<repo> python tools/sppos_nested.py --synth-dir outputs/finetune --out outputs/nested_finetune
"""
import argparse
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

ROOT = Path(os.environ.get("DENSITO_ROOT", "/home/user/workspace/densito/src/densito_rebuild"))
os.environ.setdefault("NESTED_GATE_WORK", "/tmp/sppos_nested_work")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from train_stacked import (DATA_DIR, CRITERION_GEOMETRY_COLS, preproc_for, emb_source_config,  # noqa: E402
                           emb_file_for, threshold_rule_for, f1_optimal_threshold, prevalence_threshold)
from calibration_utils import threshold_by_rule as _tbr  # noqa: E402
from nested_gate import (fit_geom, fit_emb, pct_rank, ref_rank, safe_auc, macro_f1, boot_ci,  # noqa: E402
                         connected_groups, impute, degenerate)

N_OUTER, N_INNER = 5, 3
GAIN_MIN, GAIN_FRACTION = 0.03, 0.7


def thr_rule(rule, y, s):
    return _tbr(rule, y, s, f1_optimal_threshold, prevalence_threshold, min_pos_f1=15)[0]


def inner_oof_multi(fitters, y, groups, seed):
    """fitters: {name: (X, fit_fn)}; возвращает {name: inner-OOF предсказания}."""
    out = {n: np.full(len(y), np.nan) for n in fitters}
    any_X = next(iter(fitters.values()))[0]
    for tr, va in GroupKFold(n_splits=N_INNER, shuffle=True, random_state=seed).split(any_X, groups=groups):
        if degenerate(y[tr]):
            continue
        for n, (X, fit) in fitters.items():
            out[n][va] = fit(X[tr], y[tr])(X[va])
    return out


def run(crit, geom, embs, scores, groups, hashes_ok, n_repeats, out_dir, log, candidates):
    y_all = geom[crit].values
    valid = ~pd.isna(y_all)
    y = y_all[valid].astype(int)
    cols = CRITERION_GEOMETRY_COLS[crit]
    pp = preproc_for(crit)
    Xraw = geom[cols].values.astype(np.float64)[valid]
    base_src = emb_source_config().get(crit, "imagenet")
    emb_variant = pp["emb"]
    E_base = embs[(base_src, emb_variant)][valid]
    E_synth = embs[("synth", "canonical")][valid]
    E_synth_oof = embs[("synth_oof", "canonical")][valid]
    sc = {k: v[valid] for k, v in scores.items()}
    g = groups[valid]
    studies = geom["study"].values[valid]
    rule = threshold_rule_for(crit)
    n = len(y)
    log(f"\n=== {crit}: n={n}, n_pos={int(y.sum())}, geom={pp['geom']} {cols}, emb база {base_src}/{emb_variant}, "
        f"порог '{rule}', кандидаты {candidates}")

    branches = ["base"] + candidates
    score = {b: np.full((n_repeats, n), np.nan) for b in branches}
    pred = {b: np.full((n_repeats, n), np.nan) for b in branches}
    fold_id = np.full((n_repeats, n), -1)
    rows = []
    t0 = time.time()
    for r in range(n_repeats):
        for k, (tr, te) in enumerate(GroupKFold(n_splits=N_OUTER, shuffle=True, random_state=42 + r).split(Xraw, groups=g)):
            if degenerate(y[tr]):
                continue
            med = np.nanmedian(Xraw[tr], axis=0)
            X = impute(Xraw, med)
            fold_id[r, te] = k
            # геометрия + скор головы (третий признак)
            Xs = {"logit": np.column_stack([X, sc["logit"]]), "logit_oof": np.column_stack([X, sc["logit_oof"]]),
                  "bottom": np.column_stack([X, sc["bottom"]]), "bottom_oof": np.column_stack([X, sc["bottom_oof"]])}
            fitters = {"geom": (X, fit_geom), "emb_base": (E_base, fit_emb),
                       "emb_synth": (E_synth, fit_emb), "emb_synth_oof": (E_synth_oof, fit_emb)}
            for nm, Xk in Xs.items():
                fitters[f"geom+{nm}"] = (Xk, fit_geom)
            fitters_tr = {nm: (Xk[tr], fit) for nm, (Xk, fit) in fitters.items()}
            io = inner_oof_multi(fitters_tr, y[tr], g[tr], seed=1000 * r + k)
            # скор головы как самостоятельный "контур" (ранг без обучения)
            io["score_logit"] = sc["logit"][tr].astype(float)
            io["score_logit_oof"] = sc["logit_oof"][tr].astype(float)
            # внешние предсказания
            te_pred = {nm: fit(Xk[tr], y[tr])(Xk[te]) for nm, (Xk, fit) in fitters.items()}
            te_pred["score_logit"] = sc["logit"][te].astype(float)
            te_pred["score_logit_oof"] = sc["logit_oof"][te].astype(float)

            def combo(parts):
                """parts: [(имя, вес)] -> (inner-стэк, маска ok, внешний стэк)."""
                ok = np.ones(len(tr), bool)
                for nm, _ in parts:
                    ok &= ~np.isnan(io[nm])
                if ok.sum() < 2 or len(np.unique(y[tr][ok])) < 2:
                    return None
                s_in = sum(w * pct_rank(io[nm][ok]) for nm, w in parts)
                s_te = sum(w * ref_rank(te_pred[nm], io[nm]) for nm, w in parts)
                return ok, s_in, s_te

            defs = {"base": [("geom", .5), ("emb_base", .5)],
                    "K1": [("geom", .5), ("emb_synth", .5)],
                    "K1_oof": [("geom", .5), ("emb_synth_oof", .5)],
                    "K2": [("geom+logit", .5), ("emb_base", .5)],
                    "K2_oof": [("geom+logit_oof", .5), ("emb_base", .5)],
                    "K2b": [("geom+bottom", .5), ("emb_base", .5)],
                    "K2b_oof": [("geom+bottom_oof", .5), ("emb_base", .5)],
                    "K3": [("geom", 1 / 3), ("emb_base", 1 / 3), ("emb_synth", 1 / 3)],
                    "K3_oof": [("geom", 1 / 3), ("emb_base", 1 / 3), ("emb_synth_oof", 1 / 3)],
                    "K3s": [("geom", 1 / 3), ("emb_base", 1 / 3), ("score_logit", 1 / 3)],
                    "K3s_oof": [("geom", 1 / 3), ("emb_base", 1 / 3), ("score_logit_oof", 1 / 3)],
                    "S_only_oof": [("score_logit_oof", 1.0)],
                    "B_synth_only_oof": [("emb_synth_oof", 1.0)],
                    "B_base_only": [("emb_base", 1.0)]}
            for b in branches:
                res = combo(defs[b])
                if res is None:
                    continue
                ok, s_in, s_te = res
                thr = thr_rule(rule, y[tr][ok], s_in)
                score[b][r, te] = s_te
                pred[b][r, te] = (s_te >= thr).astype(int)
            rows.append(dict(repeat=r, fold=k, n_train=len(tr), n_pos_train=int(y[tr].sum()), n_test=len(te),
                             **{f"outer_auc_{b}": safe_auc(y[te], score[b][r, te]) for b in branches}))
        if r == 0:
            log(f"  повтор 0 за {time.time() - t0:.0f} с")

    # метрики по повторам
    rep = []
    for r in range(n_repeats):
        m = ~np.isnan(score["base"][r])
        if m.sum() < 2:
            continue
        row = dict(repeat=r, auc_base=safe_auc(y[m], score["base"][r][m]), macro_f1_base=macro_f1(y[m], pred["base"][r][m]),
                   f1pos_base=f1_score(y[m], pred["base"][r][m], zero_division=0))
        for b in candidates:
            mb = m & ~np.isnan(score[b][r])
            row[f"auc_{b}"] = safe_auc(y[mb], score[b][r][mb])
            row[f"delta_auc_{b}"] = row[f"auc_{b}"] - safe_auc(y[mb], score["base"][r][mb])
            row[f"macro_f1_{b}"] = macro_f1(y[mb], pred[b][r][mb])
            row[f"delta_macro_f1_{b}"] = row[f"macro_f1_{b}"] - macro_f1(y[mb], pred["base"][r][mb])
            row[f"f1pos_{b}"] = f1_score(y[mb], pred[b][r][mb], zero_division=0)
        rep.append(row)
    rep = pd.DataFrame(rep)
    rep.to_csv(out_dir / f"repeats_{crit}.csv", index=False)
    sb = np.nanmean(score["base"], axis=0)
    pb = (np.nanmean(pred["base"], axis=0) >= 0.5).astype(int)
    summ = dict(criterion=crit, n=n, n_pos=int(y.sum()), n_repeats=int(len(rep)), threshold_rule=rule,
                base=dict(auc_mean=float(rep["auc_base"].mean()), auc_sd=float(rep["auc_base"].std()),
                          macro_f1_mean=float(rep["macro_f1_base"].mean()), f1pos_mean=float(rep["f1pos_base"].mean()),
                          auc_pooled=float(safe_auc(y, sb)), macro_f1_pooled=float(macro_f1(y, pb)),
                          auc_pooled_ci=list(boot_ci(studies, lambda idx: safe_auc(y[idx], sb[idx])))),
                candidates={})
    need = int(-(-GAIN_FRACTION * len(rep) // 1))
    for b in candidates:
        d = rep[f"delta_auc_{b}"].dropna()
        dm = rep[f"delta_macro_f1_{b}"].dropna()
        sf = np.nanmean(score[b], axis=0)
        pf = (np.nanmean(pred[b], axis=0) >= 0.5).astype(int)
        ci = boot_ci(studies, lambda idx, sf=sf: safe_auc(y[idx], sf[idx]) - safe_auc(y[idx], sb[idx]))
        n_gain = int((d >= GAIN_MIN).sum())
        summ["candidates"][b] = dict(
            auc_mean=float(rep[f"auc_{b}"].mean()), auc_sd=float(rep[f"auc_{b}"].std()),
            mean_delta_auc=float(d.mean()), sd_delta_auc=float(d.std()),
            delta_auc_repeats_ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            n_repeats_gain_ge_0_03=n_gain, n_repeats_positive=int((d > 0).sum()), n_repeats=int(len(d)),
            frac_gain_ge_0_03=float(n_gain / max(1, len(d))),
            mean_delta_macro_f1=float(dm.mean()), macro_f1_mean=float(rep[f"macro_f1_{b}"].mean()),
            f1pos_mean=float(rep[f"f1pos_{b}"].mean()),
            auc_pooled=float(safe_auc(y, sf)), macro_f1_pooled=float(macro_f1(y, pf)),
            delta_auc_pooled_ci_by_study=list(ci),
            accepted_old_rule=bool(n_gain >= need and dm.mean() >= -1e-12))
        c = summ["candidates"][b]
        log(f"  {b:16s} AUC {c['auc_mean']:.3f}±{c['auc_sd']:.3f} ΔAUC={c['mean_delta_auc']:+.3f} "
            f"(≥0.03 в {n_gain}/{len(d)}, в плюсе {c['n_repeats_positive']}/{len(d)}) Δmacro-F1={c['mean_delta_macro_f1']:+.3f} "
            f"pooled ΔAUC ДИ [{ci[0]:+.3f}; {ci[2]:+.3f}]{' => ПРОХОДИТ СТАРОЕ ПРАВИЛО' if c['accepted_old_rule'] else ''}")
        # per-repeat внешние OOF-скоры базы и кандидата (формат парного гейта идеи 11)
        recs = []
        row_ids = np.nonzero(valid)[0]
        for r in range(n_repeats):
            m = ~np.isnan(score["base"][r]) & ~np.isnan(score[b][r])
            recs.append(pd.DataFrame(dict(repeat=r, row_id=row_ids[m], y=y[m], group=g[m],
                                          score_base=score["base"][r][m], score_cand=score[b][r][m],
                                          fold=fold_id[r][m], study=studies[m],
                                          pred_base=pred["base"][r][m].astype(int), pred_cand=pred[b][r][m].astype(int))))
        pd.concat(recs).to_csv(out_dir / f"per_repeat_{crit}_{b}.csv", index=False)
    log(f"  база: AUC {summ['base']['auc_mean']:.3f}±{summ['base']['auc_sd']:.3f}, macro-F1 {summ['base']['macro_f1_mean']:.3f}, "
        f"pooled AUC {summ['base']['auc_pooled']:.3f} ДИ {np.round(summ['base']['auc_pooled_ci'], 3).tolist()}")
    pd.DataFrame(rows).to_csv(out_dir / f"folds_{crit}.csv", index=False)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", required=True, help="каталог с embeddings_synth*_canonical.npy и scores_synth.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=int(os.environ.get("N_REPEATS", 20)))
    ap.add_argument("--hashes", default=str(Path(__file__).resolve().parents[1] / "outputs" / "pixel_hashes.csv"))
    ap.add_argument("--criteria", default="sp_pos,sp_axis,sp_art")
    a = ap.parse_args()
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True); lines.append(s)

    lab = pd.read_csv(DATA_DIR / "labels_for_embeddings.csv")
    geom_all = {v: pd.read_csv(DATA_DIR / fn) for v, fn in
                {"baseline": "geometry_features.csv", "canonical": "geometry_features_canonical.csv"}.items()}
    for v, gdf in geom_all.items():
        assert (gdf["file_path"].values == lab["file_path"].values).all()
    sidx = np.nonzero((lab["region"] == "spine").values)[0]
    hashes = pd.read_csv(a.hashes)
    hmap = dict(zip(hashes["file_path"], hashes["pixel_hash"]))
    groups_all = connected_groups(lab["study"].values, lab["file_path"].map(hmap).values)
    sd = Path(a.synth_dir)
    embs = {}
    for src in ("imagenet", "densito", "densito_inv"):
        for variant in ("baseline", "canonical"):
            f = DATA_DIR / emb_file_for(src, variant)
            if f.exists():
                embs[(src, variant)] = np.load(f)[sidx]
    embs[("synth", "canonical")] = np.load(sd / "embeddings_synth_canonical.npy")[sidx]
    embs[("synth_oof", "canonical")] = np.load(sd / "embeddings_synth_oof_canonical.npy")[sidx]
    sc = pd.read_csv(sd / "scores_synth.csv")
    assert (sc["file_path"].values == lab["file_path"].values).all()
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    oy_col = "pred_oy" if "pred_oy" in sc else "pred_bottom"      # v3: смещение окна вверх; v1/v2: обрезка снизу
    scores = {"logit": logit(sc["p_defect"].values)[sidx], "logit_oof": logit(sc["p_defect_oof"].values)[sidx],
              "bottom": sc[oy_col].values[sidx], "bottom_oof": sc[oy_col + "_oof"].values[sidx]}
    groups = groups_all[sidx]
    summary = {"n_repeats": a.repeats, "synth_dir": str(sd), "protocol": "outer GroupKFold 5 x repeats, inner 3, groups=(study,pixel_hash)"}
    for crit in a.criteria.split(","):
        pp = preproc_for(crit)
        geom = geom_all[pp["geom"] if pp["geom"] in geom_all else "baseline"].iloc[sidx].reset_index(drop=True)
        if crit == "sp_pos":
            cands = ["K1", "K1_oof", "K2", "K2_oof", "K2b", "K2b_oof", "K3", "K3_oof", "K3s", "K3s_oof",
                     "S_only_oof", "B_synth_only_oof", "B_base_only"]
        else:
            cands = ["K1", "K1_oof"]
        summary[crit] = run(crit, geom, embs, scores, groups, None, a.repeats, out_dir, log, cands)
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=1, ensure_ascii=False)
    (out_dir / "log.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
