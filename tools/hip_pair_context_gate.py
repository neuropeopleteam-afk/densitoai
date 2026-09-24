"""
Б1 (24.09.2026): «контекст другого бедра того же визита» для hip_pos — полный nested-харнесс с ПЕРЕОБУЧЕНИЕМ.
Опубликованная проверка гипотезы; кандидаты и правило приёмки записаны заранее в docs/k5/B1_PREREGISTRATION.md.

База = стек поставки 2.4.0 для hip_pos: контур A — геометрия canonical (5 признаков) -> LogReg(C=1, balanced),
контур B — imagenet-эмбеддинги -> PCA(32) -> LogReg(C=0.1, balanced), стэк 0.5/0.5 по рангам (ранги test —
по inner-OOF референсу, как percentile_rank инференса), порог prevalence на inner-OOF стэке внешнего train.
В каждом внешнем фолде заново обучаются hip_pos, hip_roi и any-модель бедра (для бинарной задачи).

Скор другой стороны: для строк train — inner-OOF (оба бедра исследования в одном inner-фолде, группы =
компоненты study+pixel_hash); для строк test — модели внешнего фолда. R(s) — ранг стэка относительно inner-OOF.
Кандидаты (u — на шкале рангов, затем отображается обратно на шкалу стэка квантилем inner-OOF стэка, чтобы
строки без второй стороны сохраняли свой скор и max(hip_pos, hip_roi) в бинарной задаче не менял масштаб):
  w_inner      : u = (1-w)·R(s_own) + w·mean R(s_other), w ∈ {0,.1,..,.5} по AUC на inner-OOF (равенство -> меньший w);
  w_fixed_0.3  : то же, w = 0.3;
  grok_solidity: u = 0.5·R(s_own) + 0.5·R(solidity другой стороны), R — по распределению свидетеля во внешнем train;
справочно: k5_rule_mean = 0.5·s_own + 0.5·mean s_other на шкале стэка (правило К5 19.09).
Нет другой стороны -> u = R(s_own).

Окружение: GEOM=canonical|baseline (контур A hip_pos), THR_RULE=prevalence|f1_optimal, N_REPEATS (20),
B1_OUT (каталог результатов). Запуск: OMP_NUM_THREADS=1 nice -n 10 python tools/hip_pair_context_gate.py
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys, json, time, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

ROOT = Path(os.environ.get('DENSITO_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'tools'))
from nested_gate import (fit_geom, fit_emb, connected_groups, impute, safe_auc, pct_rank, ref_rank, stack,  # noqa: E402
                         inner_oof, macro_f1, degenerate, N_OUTER, N_INNER)
from train_stacked import CRITERION_GEOMETRY_COLS, prevalence_threshold, f1_optimal_threshold  # noqa: E402
from paired_gate import write_gate_input  # noqa: E402

N_REPEATS = int(os.environ.get('N_REPEATS', 20))
GEOM = os.environ.get('GEOM', 'canonical')
THR_RULE = os.environ.get('THR_RULE', 'prevalence')
OUT = Path(os.environ.get('B1_OUT', ROOT / 'outputs' / 'b1')); OUT.mkdir(parents=True, exist_ok=True)
W_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
CANDS = ['w_inner', 'w_fixed_0.3', 'grok_solidity', 'k5_rule_mean']
GAIN_MIN, GAIN_SHARE = 0.03, 0.70


def thr_of(y, s):
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if THR_RULE == 'f1_optimal' and y.sum() >= 15:
        t = f1_optimal_threshold(y, s)
        if t is not None:
            return float(t)
    return float(prevalence_threshold(y, s))


def load():
    gfile = {'canonical': 'geometry_features_canonical.csv', 'baseline': 'geometry_features.csv'}[GEOM]
    gpos = pd.read_csv(ROOT / 'data' / gfile)
    gbase = pd.read_csv(ROOT / 'data' / 'geometry_features.csv')
    assert (gpos.file_path.values == gbase.file_path.values).all()
    lab = pd.read_csv(ROOT / 'data' / 'labels_for_embeddings.csv')
    E = np.load(ROOT / 'data' / 'embeddings.npy')
    m = gbase.region.isin(['right_hip', 'left_hip']).values
    em = lab.region.isin(['right_hip', 'left_hip']).values
    hip, hpos, Eh = gbase[m].reset_index(drop=True), gpos[m].reset_index(drop=True), E[np.nonzero(em)[0]]
    assert (lab.file_path.values[em] == hip.file_path.values).all()
    h = pd.read_csv(ROOT / 'docs' / 'k5' / 'pixel_hashes.csv')
    hip['pixel_hash'] = hip.file_path.map(dict(zip(h.file_path, h.pixel_hash)))
    assert hip.pixel_hash.notna().all()
    valid = (hip.hip_pos_c.notna() & hip.hip_roi_c.notna()).values
    hip, hpos, Eh = hip[valid].reset_index(drop=True), hpos[valid].reset_index(drop=True), Eh[valid]
    hip['group'] = connected_groups(hip.study.values, hip.pixel_hash.values)
    return hip, hpos, Eh


def other_index(studies, sides):
    out = []
    for i in range(len(studies)):
        out.append(np.nonzero((studies == studies[i]) & (sides != sides[i]))[0])
    return out


def mean_other(v, oth):
    o = np.full(len(v), np.nan)
    for i, idx in enumerate(oth):
        if len(idx):
            vv = v[idx]
            if np.isfinite(vv).any():
                o[i] = np.nanmean(vv)
    return o


def local_other(sub, oth):
    pos = {g: i for i, g in enumerate(sub)}
    return [np.array([pos[j] for j in oth[g] if j in pos], int) for g in sub]


def blend(R, mo, w):
    return np.where(np.isnan(mo), R, (1 - w) * R + w * mo)


def back_to_stack(u, ref):
    """ранг u -> шкала стэка (линейная квантиль inner-OOF стэка); монотонно, строки без контекста ~ свой скор."""
    ref = np.sort(ref[~np.isnan(ref)])
    return np.where(np.isnan(u), np.nan, np.quantile(ref, np.clip(np.nan_to_num(u), 0, 1)))


def consistent(raw, flag):
    return np.where(flag == 1, 0.5 + 0.5 * raw, np.minimum(0.5 * raw, 0.499999))


def main():
    t0 = time.time()
    hip, hpos, Eh = load()
    y = hip.hip_pos_c.values.astype(int); yr = hip.hip_roi_c.values.astype(int); yb = np.maximum(y, yr)
    groups, studies, sides = hip.group.values, hip.study.values, hip.hip_side_detected.values
    Xp_raw = hpos[CRITERION_GEOMETRY_COLS['hip_pos']].values.astype(float)
    Xr_raw = hip[CRITERION_GEOMETRY_COLS['hip_roi']].values.astype(float)
    cols_any = sorted(set(CRITERION_GEOMETRY_COLS['hip_pos']) | set(CRITERION_GEOMETRY_COLS['hip_roi']))
    Xa_raw = hip[cols_any].values.astype(float)
    oth = other_index(studies, sides)
    sol_other = mean_other(hip.femur_solidity.values.astype(float), oth)
    n = len(y); has_other = np.array([len(o) > 0 for o in oth])
    print(f'GEOM={GEOM} THR_RULE={THR_RULE} R={N_REPEATS}: n={n} pos={y.sum()} roi_pos={yr.sum()} bin_pos={yb.sum()} '
          f'rows with other side={has_other.sum()} witness={np.isfinite(sol_other).sum()} groups={len(np.unique(groups))}', flush=True)
    names = ['base'] + CANDS
    S = {v: np.full((N_REPEATS, n), np.nan) for v in names}; P = {v: np.full((N_REPEATS, n), np.nan) for v in names}
    SB = {v: np.full((N_REPEATS, n), np.nan) for v in names}; PB = {v: np.full((N_REPEATS, n), np.nan) for v in names}
    S_roi = np.full((N_REPEATS, n), np.nan); P_roi = np.full((N_REPEATS, n), np.nan)
    FOLD = np.full((N_REPEATS, n), np.nan)
    fold_rows, rep_rows = [], []
    for r in range(N_REPEATS):
        for k, (tr, te) in enumerate(GroupKFold(N_OUTER, shuffle=True, random_state=42 + r).split(Xp_raw, groups=groups)):
            if degenerate(y[tr]):
                continue
            FOLD[r, te] = k
            seed = 1000 * r + k
            ot, oe_ = local_other(tr, oth), local_other(te, oth)
            # --- hip_pos: база (переобучение)
            Xp = impute(Xp_raw, np.nanmedian(Xp_raw[tr], 0))
            og, oe = inner_oof(Xp[tr], Eh[tr], y[tr], groups[tr], seed)
            ok = ~np.isnan(og) & ~np.isnan(oe)
            s_in = np.full(len(tr), np.nan); s_in[ok] = stack(pct_rank(og[ok]), pct_rank(oe[ok]), 0.5)
            pg, pe = fit_geom(Xp[tr], y[tr])(Xp[te]), fit_emb(Eh[tr], y[tr])(Eh[te])
            s_te = stack(ref_rank(pg, og), ref_rank(pe, oe), 0.5)
            R_in = np.full(len(tr), np.nan); R_in[ok] = pct_rank(s_in[ok]); R_te = ref_rank(s_te, s_in)
            mo_in, mo_te = mean_other(R_in, ot), mean_other(R_te, oe_)
            yt = y[tr]
            aucs = {w: safe_auc(yt[ok], blend(R_in, mo_in, w)[ok]) for w in W_GRID}
            best = np.nanmax(list(aucs.values()))
            w_sel = min(w for w, a in aucs.items() if a >= best - 1e-12)
            # свидетель solidity (без обучения); ранг — по распределению свидетеля во внешнем train
            wt, wte = sol_other[tr], sol_other[te]
            refw = wt[np.isfinite(wt)]
            Rw_in = np.where(np.isfinite(wt), ref_rank(np.nan_to_num(wt), refw), np.nan)
            Rw_te = np.where(np.isfinite(wte), ref_rank(np.nan_to_num(wte), refw), np.nan)
            u_in = {'w_inner': blend(R_in, mo_in, w_sel), 'w_fixed_0.3': blend(R_in, mo_in, 0.3),
                    'grok_solidity': np.where(np.isnan(Rw_in), R_in, 0.5 * R_in + 0.5 * Rw_in)}
            u_te = {'w_inner': blend(R_te, mo_te, w_sel), 'w_fixed_0.3': blend(R_te, mo_te, 0.3),
                    'grok_solidity': np.where(np.isnan(Rw_te), R_te, 0.5 * R_te + 0.5 * Rw_te)}
            sc_in = {'base': s_in, 'k5_rule_mean': np.where(np.isnan(mean_other(s_in, ot)), s_in, 0.5 * s_in + 0.5 * mean_other(s_in, ot))}
            sc_te = {'base': s_te, 'k5_rule_mean': np.where(np.isnan(mean_other(s_te, oe_)), s_te, 0.5 * s_te + 0.5 * mean_other(s_te, oe_))}
            for v in ('w_inner', 'w_fixed_0.3', 'grok_solidity'):
                sc_in[v] = np.where(ok, back_to_stack(u_in[v], s_in), np.nan)
                sc_te[v] = back_to_stack(u_te[v], s_in)
            thr = {v: thr_of(yt, sc_in[v]) for v in names}
            # --- hip_roi (не меняется кандидатами) и any-модель: переобучение
            Xr = impute(Xr_raw, np.nanmedian(Xr_raw[tr], 0))
            rg, re_ = inner_oof(Xr[tr], Eh[tr], yr[tr], groups[tr], seed)
            okr = ~np.isnan(rg) & ~np.isnan(re_)
            sr_in = np.full(len(tr), np.nan); sr_in[okr] = stack(pct_rank(rg[okr]), pct_rank(re_[okr]), 0.5)
            prg, pre = fit_geom(Xr[tr], yr[tr])(Xr[te]), fit_emb(Eh[tr], yr[tr])(Eh[te])
            sr_te = stack(ref_rank(prg, rg), ref_rank(pre, re_), 0.5)
            thr_r = thr_of(yr[tr], sr_in)
            fr = (sr_te >= thr_r).astype(int)
            S_roi[r, te], P_roi[r, te] = sr_te, fr
            Xa = impute(Xa_raw, np.nanmedian(Xa_raw[tr], 0))
            p_any = 0.5 * (fit_geom(Xa[tr], yb[tr])(Xa[te]) + fit_emb(Eh[tr], yb[tr])(Eh[te]))
            for v in names:
                fp = (sc_te[v] >= thr[v]).astype(int)
                S[v][r, te], P[v][r, te] = sc_te[v], fp
                fb = np.maximum(fp, fr)
                SB[v][r, te] = consistent(0.5 * p_any + 0.5 * np.maximum(sc_te[v], sr_te), fb); PB[v][r, te] = fb
            fold_rows.append(dict(repeat=r, fold=k, n_train=len(tr), n_test=len(te), n_pos_test=int(y[te].sum()), w_selected=w_sel,
                                  **{f'inner_auc_w{w}': aucs[w] for w in W_GRID}, **{f'thr_{v}': thr[v] for v in names}))
        row = dict(repeat=r, auc_roi=safe_auc(yr, S_roi[r]), f1_roi=f1_score(yr, P_roi[r], zero_division=0))
        for v in names:
            row[f'auc_{v}'] = safe_auc(y, S[v][r]); row[f'mf1_{v}'] = macro_f1(y, P[v][r])
            row[f'f1pos_{v}'] = f1_score(y, P[v][r], zero_division=0); row[f'nflag_{v}'] = int(P[v][r].sum())
            row[f'auc_bin_{v}'] = safe_auc(yb, SB[v][r]); row[f'f1_bin_{v}'] = f1_score(yb, PB[v][r], zero_division=0)
            row[f'mf1_bin_{v}'] = macro_f1(yb, PB[v][r])
        rep_rows.append(row)
        print(f"repeat {r}: base {row['auc_base']:.3f} | " + ' | '.join(f"{v} {row[f'auc_{v}']-row['auc_base']:+.3f} mF1 {row[f'mf1_{v}']-row['mf1_base']:+.3f} bin {row[f'auc_bin_{v}']-row['auc_bin_base']:+.3f}" for v in CANDS)
              + f" | w {[fr_['w_selected'] for fr_ in fold_rows if fr_['repeat'] == r]} ({time.time()-t0:.0f}s)", flush=True)
    tag = f'{GEOM}_{THR_RULE}_R{N_REPEATS}'
    rep = pd.DataFrame(rep_rows); rep.to_csv(OUT / f'per_repeat_{tag}.csv', index=False)
    folds = pd.DataFrame(fold_rows); folds.to_csv(OUT / f'per_fold_{tag}.csv', index=False)
    np.savez_compressed(OUT / f'scores_{tag}.npz', y=y, yr=yr, yb=yb, groups=groups, studies=studies, sides=sides,
                        has_other=has_other, fold=FOLD, S_roi=S_roi, P_roi=P_roi,
                        **{f'S_{v}': S[v] for v in names}, **{f'P_{v}': P[v] for v in names},
                        **{f'SB_{v}': SB[v] for v in names}, **{f'PB_{v}': PB[v] for v in names})
    dec = []
    for v in CANDS:
        d = rep[f'auc_{v}'] - rep['auc_base']; n_gain = int((d >= GAIN_MIN).sum())
        dm = rep[f'mf1_{v}'] - rep['mf1_base']; db = rep[f'auc_bin_{v}'] - rep['auc_bin_base']
        acc = bool(n_gain >= GAIN_SHARE * N_REPEATS and dm.mean() >= -1e-12 and db.mean() >= -1e-12)
        dec.append(dict(candidate=v, geom=GEOM, thr_rule=THR_RULE, n_repeats=N_REPEATS, n_gain_ge_0_03=n_gain,
                        mean_delta_auc=float(d.mean()), min_delta_auc=float(d.min()), max_delta_auc=float(d.max()),
                        n_delta_pos=int((d > 0).sum()), auc_base_mean=float(rep['auc_base'].mean()), auc_cand_mean=float(rep[f'auc_{v}'].mean()),
                        mf1_base_mean=float(rep['mf1_base'].mean()), mf1_cand_mean=float(rep[f'mf1_{v}'].mean()), mean_delta_mf1=float(dm.mean()),
                        n_mf1_not_worse=int((dm >= -1e-12).sum()), f1pos_base=float(rep['f1pos_base'].mean()), f1pos_cand=float(rep[f'f1pos_{v}'].mean()),
                        auc_roi_mean=float(rep['auc_roi'].mean()), delta_auc_roi=0.0,
                        auc_bin_base=float(rep['auc_bin_base'].mean()), auc_bin_cand=float(rep[f'auc_bin_{v}'].mean()), mean_delta_auc_bin=float(db.mean()),
                        f1_bin_base=float(rep['f1_bin_base'].mean()), f1_bin_cand=float(rep[f'f1_bin_{v}'].mean()),
                        accepted_rule_k2=acc))
        print(dec[-1], flush=True)
        write_gate_input(OUT / f'gate_input_hip_pos_{v}_{tag}.csv', y, groups, S['base'], S[v], pred_base=P['base'], pred_cand=P[v],
                         fold=FOLD, study=studies, file_path=hip.file_path.values)
        write_gate_input(OUT / f'gate_input_hip_bin_{v}_{tag}.csv', yb, groups, SB['base'], SB[v], pred_base=PB['base'], pred_cand=PB[v],
                         fold=FOLD, study=studies, file_path=hip.file_path.values)
    pd.DataFrame(dec).to_csv(OUT / f'decisions_{tag}.csv', index=False)
    if 'w_selected' in folds:
        print('w_selected freq:', folds.w_selected.value_counts().sort_index().to_dict())
    print(f'done {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
