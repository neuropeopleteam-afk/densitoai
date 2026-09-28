"""2.5: позиционный признак посторонних предметов (sp_art) — вложенная проверка по протоколу проекта.

Предрегистрация: work/exp_spart/PREREG_spart_position.md (сетка, ветки и правило записаны до запуска).
Протокол — как tools/emb_gate.py (К13), чтобы база совпала с docs/EMB_GATE_REPORT.md бит в бит:
  внешний GroupKFold(shuffle, random_state=42+r) 5 фолдов × N_REPEATS (20) повторов, группы — компоненты
  связности (study, pixel_hash); внутренний GroupKFold 3 фолда на внешнем train (seed 1000*r+k);
  контур A = StandardScaler -> LogReg(C=1, balanced); контур B = imagenet, StandardScaler -> PCA(32) -> LogReg(C=0.1);
  стек 0.5/0.5 по рангам относительно inner-OOF; порог — правило config.yaml thresholds_rule.sp_art на inner-OOF;
  медианы пропусков — по внешнему train.
Ветки: base, fixed:<кандидат>, gate (выбор отсечки c внутри фолда, семейство band<c>), gate3 (семейство band<c>n),
       gate_b (как gate, но base тоже кандидат).
Выход: outputs/spart_position/{per_repeat.csv, per_fold.csv, summary.json, gate_input_<ветка>.csv, oof_<ветка>.csv, log.txt}

Запуск: OMP_NUM_THREADS=1 nice python tools/spart_position/spart_position_gate.py
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

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from train_stacked import (DATA_DIR, CRITERION_GEOMETRY_COLS, REGION_ROWS, emb_file_for,  # noqa: E402
                           threshold_rule_for, emb_source_config, preproc_for)
from nested_gate import (fit_geom, fit_emb, pct_rank, ref_rank, stack, safe_auc,  # noqa: E402
                         macro_f1, boot_ci, connected_groups, impute, degenerate)
from emb_gate import threshold_by_rule  # noqa: E402
from paired_gate import write_gate_input  # noqa: E402

CRIT, REGION = "sp_art", "spine"
BANDS = (50, 60, 70, 80, 100)
N_OUTER, N_INNER = 5, 3
N_REPEATS = int(os.environ.get("N_REPEATS", 20))
W = 0.5
GAIN_MIN, GAIN_FRACTION = 0.03, 0.7
GAIN_REPEATS = int(-(-GAIN_FRACTION * N_REPEATS // 1))
OUT = ROOT / "outputs" / "spart_position"
OUT.mkdir(parents=True, exist_ok=True)

BASE_COLS = list(CRITERION_GEOMETRY_COLS[CRIT])
CANDS = {"base": BASE_COLS,
         "ctrl_log": ["metal_metal_area_log", "metal_metal_max_intensity_gap"]}
for b in BANDS:
    CANDS[f"band{b}"] = [f"metal_metal_band{b}_area_log", f"metal_metal_band{b}_max_gap"]
for b in BANDS:
    CANDS[f"band{b}n"] = [f"metal_metal_band{b}_area_log", f"metal_metal_band{b}_max_gap", f"metal_metal_band{b}_n"]
# порядок разрешения ничьих внутри семейства: большая c первой (меньше фильтрации — ближе к базе)
FAMILY = {"gate": [f"band{b}" for b in sorted(BANDS, reverse=True)],
          "gate3": [f"band{b}n" for b in sorted(BANDS, reverse=True)],
          "gate_b": ["base"] + [f"band{b}" for b in sorted(BANDS, reverse=True)]}


def load():
    g = pd.read_csv(DATA_DIR / "geometry_features.csv")
    g = g[g["region"].isin(REGION_ROWS[REGION])].reset_index(drop=True)
    lab = pd.read_csv(DATA_DIR / "labels_for_embeddings.csv")
    eidx = np.nonzero(lab["region"].isin(REGION_ROWS[REGION]).values)[0]
    assert (lab.loc[eidx, "file_path"].values == g["file_path"].values).all(), "порядок строк эмбеддингов"
    src = emb_source_config().get(CRIT, "imagenet")
    pp = preproc_for(CRIT)
    assert pp == {"geom": "baseline", "emb": "baseline"}, pp
    E = np.load(DATA_DIR / emb_file_for(src, pp["emb"]))[eidx]
    hashes = pd.read_csv(ROOT / "outputs" / "pixel_hashes.csv")
    hmap = dict(zip(hashes["file_path"], hashes["pixel_hash"]))
    groups = connected_groups(g["study"].values, g["file_path"].map(hmap).values)
    return g, E, groups, src


def main():
    t0 = time.time()
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    g, E_all, groups_all, src = load()
    y_all = g[CRIT].values
    valid = ~pd.isna(y_all)
    y = y_all[valid].astype(int)
    E = E_all[valid]
    grp = groups_all[valid]
    studies = g["study"].values[valid]
    files = g["file_path"].values[valid]
    X_raw = {k: g[c].values.astype(np.float64)[valid] for k, c in CANDS.items()}
    rule = threshold_rule_for(CRIT)
    n = len(y)
    log(f"=== {REGION}/{CRIT}: n={n}, n_pos={int(y.sum())}, групп={len(np.unique(grp))}, "
        f"исследований с позитивом={len(np.unique(studies[y == 1]))}, emb='{src}', правило порога '{rule}', "
        f"повторов {N_REPEATS}, приёмка ≥{GAIN_MIN} в ≥{GAIN_REPEATS}")
    for k, c in CANDS.items():
        log(f"  {k}: {c}")

    branches = list(CANDS) + list(FAMILY)
    score = {b: np.full((N_REPEATS, n), np.nan) for b in branches}
    pred = {b: np.full((N_REPEATS, n), np.nan) for b in branches}
    geomp = {b: np.full((N_REPEATS, n), np.nan) for b in branches}   # сырой выход контура A (внешний тест)
    fold_id = np.full((N_REPEATS, n), np.nan)
    fold_rows = []

    for r in range(N_REPEATS):
        for k, (tr, te) in enumerate(GroupKFold(n_splits=N_OUTER, shuffle=True, random_state=42 + r).split(E, groups=grp)):
            if degenerate(y[tr]):
                continue
            Xc = {c: impute(X_raw[c], np.nanmedian(X_raw[c][tr], axis=0)) for c in CANDS}
            # inner-OOF: контур B один раз, контур A по каждому кандидату (те же inner-фолды)
            oe = np.full(len(tr), np.nan)
            og = {c: np.full(len(tr), np.nan) for c in CANDS}
            for itr, iva in GroupKFold(n_splits=N_INNER, shuffle=True, random_state=1000 * r + k).split(E[tr], groups=grp[tr]):
                if degenerate(y[tr][itr]):
                    continue
                oe[iva] = fit_emb(E[tr][itr], y[tr][itr])(E[tr][iva])
                for c in CANDS:
                    og[c][iva] = fit_geom(Xc[c][tr][itr], y[tr][itr])(Xc[c][tr][iva])
            inner = {}
            for c in CANDS:
                ok = ~np.isnan(og[c]) & ~np.isnan(oe)
                s_in = stack(pct_rank(og[c][ok]), pct_rank(oe[ok]), W)
                thr = threshold_by_rule(rule, y[tr][ok], s_in, min_positives_for_f1=15)[0]
                inner[c] = dict(auc=safe_auc(y[tr][ok], s_in), thr=thr, auc_geom=safe_auc(y[tr][ok], og[c][ok]))
            sel = {}
            for fam, members in FAMILY.items():
                best = np.nanmax([inner[c]["auc"] for c in members])
                sel[fam] = [c for c in members if inner[c]["auc"] >= best - 1e-12][0]
            pe = fit_emb(E[tr], y[tr])(E[te])
            re_te = ref_rank(pe, oe)
            outer = {}
            for c in CANDS:
                pg = fit_geom(Xc[c][tr], y[tr])(Xc[c][te])
                s = stack(ref_rank(pg, og[c]), re_te, W)
                outer[c] = (s, pg)
            for b in branches:
                c = sel.get(b, b)
                s, pg = outer[c]
                score[b][r, te] = s
                geomp[b][r, te] = pg
                pred[b][r, te] = (s >= inner[c]["thr"]).astype(int)
            fold_id[r, te] = k
            fold_rows.append({"repeat": r, "fold": k, "n_train": len(tr), "n_pos_train": int(y[tr].sum()),
                              "n_test": len(te), "n_pos_test": int(y[te].sum()),
                              **{f"sel_{f}": sel[f] for f in FAMILY},
                              **{f"inner_auc_{c}": inner[c]["auc"] for c in CANDS}})
        log(f"  повтор {r}: " + " ".join(f"{b}={safe_auc(y, score[b][r]):.3f}" for b in ("base", "ctrl_log", "gate", "gate3", "gate_b"))
            + " | выбор gate: " + " ".join(fr["sel_gate"] for fr in fold_rows if fr["repeat"] == r))

    # ---- по повторам
    rows = []
    for r in range(N_REPEATS):
        row = {"repeat": r}
        for b in branches:
            row[f"auc_{b}"] = safe_auc(y, score[b][r])
            row[f"auc_geom_{b}"] = safe_auc(y, geomp[b][r])
            row[f"f1pos_{b}"] = f1_score(y, pred[b][r], zero_division=0)
            row[f"macro_f1_{b}"] = macro_f1(y, pred[b][r])
            row[f"n_flag_{b}"] = int(np.nansum(pred[b][r]))
        for b in branches:
            if b == "base":
                continue
            row[f"d_auc_{b}"] = row[f"auc_{b}"] - row["auc_base"]
            row[f"d_f1pos_{b}"] = row[f"f1pos_{b}"] - row["f1pos_base"]
            row[f"d_macro_f1_{b}"] = row[f"macro_f1_{b}"] - row["macro_f1_base"]
        rows.append(row)
    rep = pd.DataFrame(rows)
    rep.to_csv(OUT / "per_repeat.csv", index=False)
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(OUT / "per_fold.csv", index=False)

    summary = {"protocol": {"outer": N_OUTER, "inner": N_INNER, "repeats": N_REPEATS, "w": W, "emb_source": src,
                            "threshold_rule": rule, "gain_min": GAIN_MIN, "gain_repeats": GAIN_REPEATS,
                            "groups": "connected components (study, pixel_hash)", "n": n, "n_pos": int(y.sum()),
                            "n_groups": int(len(np.unique(grp))),
                            "n_pos_studies": int(len(np.unique(studies[y == 1])))},
               "candidates": CANDS, "branches": {}}
    sb = np.nanmean(score["base"], axis=0)
    for b in branches:
        d = rep[f"d_auc_{b}"] if b != "base" else pd.Series(np.zeros(N_REPEATS))
        dm = rep[f"d_macro_f1_{b}"] if b != "base" else pd.Series(np.zeros(N_REPEATS))
        df1 = rep[f"d_f1pos_{b}"] if b != "base" else pd.Series(np.zeros(N_REPEATS))
        sm = np.nanmean(score[b], axis=0)
        ci = boot_ci(studies, lambda idx, sm=sm: safe_auc(y[idx], sm[idx]) - safe_auc(y[idx], sb[idx]))
        n_gain = int((d >= GAIN_MIN).sum())
        rec = {"auc_mean": float(rep[f"auc_{b}"].mean()), "auc_sd": float(rep[f"auc_{b}"].std()),
               "auc_geom_mean": float(rep[f"auc_geom_{b}"].mean()), "auc_geom_sd": float(rep[f"auc_geom_{b}"].std()),
               "f1pos_mean": float(rep[f"f1pos_{b}"].mean()), "macro_f1_mean": float(rep[f"macro_f1_{b}"].mean()),
               "n_flag_mean": float(rep[f"n_flag_{b}"].mean()),
               "d_auc_mean": float(d.mean()), "d_auc_sd": float(d.std()), "d_auc_min": float(d.min()), "d_auc_max": float(d.max()),
               "n_pos_repeats": int((d > 0).sum()), "n_gain_repeats": n_gain,
               "d_f1pos_mean": float(df1.mean()), "n_f1pos_not_worse": int((df1 >= -1e-12).sum()),
               "d_macro_f1_mean": float(dm.mean()), "n_macro_f1_not_worse": int((dm >= -1e-12).sum()),
               "auc_pooled_mean_score": float(safe_auc(y, sm)), "d_auc_pooled_ci95": list(ci),
               "old_rule_pass": bool(b != "base" and n_gain >= GAIN_REPEATS and dm.mean() >= -1e-12)}
        if b in FAMILY:
            rec["selection_counts"] = folds[f"sel_{b}"].value_counts().to_dict()
        summary["branches"][b] = rec
        if b != "base":
            write_gate_input(OUT / f"gate_input_{b}.csv", y, grp, score["base"], score[b],
                             pred_base=pred["base"], pred_cand=pred[b], fold=fold_id, study=studies, file_path=files)
        pd.DataFrame({"study": studies, "file_path": files, "y_true": y, "score_mean": sm,
                      "pred_vote": (np.nanmean(pred[b], axis=0) >= 0.5).astype(int)}).to_csv(OUT / f"oof_{b}.csv", index=False)
        log(f"  {b:10s}: стек AUC {rec['auc_mean']:.3f} ± {rec['auc_sd']:.3f} | A {rec['auc_geom_mean']:.3f} | "
            f"ΔAUC {rec['d_auc_mean']:+.3f} ± {rec['d_auc_sd']:.3f} (мин {rec['d_auc_min']:+.3f}), в плюсе {rec['n_pos_repeats']}/{N_REPEATS}, "
            f"≥0.03 {n_gain}/{N_REPEATS} | ΔF1(+) {rec['d_f1pos_mean']:+.3f} | Δmacro-F1 {rec['d_macro_f1_mean']:+.3f} "
            f"| ДИ95 [{ci[0]:+.3f}; {ci[2]:+.3f}]" + (" | ПРАВИЛО ПРОЙДЕНО" if rec["old_rule_pass"] else "")
            + (f" | выбор {rec['selection_counts']}" if b in FAMILY else ""))
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"готово за {time.time() - t0:.0f} с")
    (OUT / "log.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
