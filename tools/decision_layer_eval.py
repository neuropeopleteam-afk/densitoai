"""
Б2: метрики и решения по гипотезам решающего слоя (протокол — docs/b2/B2_PROTOCOL.md).

Вход — NPZ из tools/decision_layer_nested.py (b2_<tag>_<region>.npz, tag = all | unique).
Выход — <out>/b2_<tag>_summary.json, <out>/b2_<tag>_per_repeat.csv, входы paired_gate
(<out>/gate_<tag>_*.npz) и печать сводки.

Варианты:
  base          — действующее правило (score >= thr, thr по thresholds_rule на inner-OOF);
  tie           — (а) порог = середина между соседними уровнями inner-OOF (decision_rules.tie_midpoint_threshold);
  excl_base     — (б) поверх base, tau выбран на inner-OOF внешнего train;
  excl_tie      — (б) поверх tie;
  excl_inf_base / excl_inf_tie — (б) с фиксированным tau = inf (правило без параметров, для справки).

Запуск: DENSITO_ROOT=$PWD ../venv/bin/python tools/decision_layer_eval.py --inp outputs/b2 --tag all
"""
import os, sys, json, math, argparse, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ.get('DENSITO_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from decision_rules import exclusive_types_array  # noqa: E402

TAUS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, math.inf]   # 0.0 = «выкл» (все флаги сохраняются)
N_BOOT = 2000
REGIONS = ['spine', 'hip']


def f1w(y, f, w=None):
    """F1(+) с весами строк; y, f — (n,), w — (n,) или (B, n)."""
    y = y.astype(float); f = f.astype(float)
    if w is None:
        tp = (y * f).sum(); fp = ((1 - y) * f).sum(); fn = (y * (1 - f)).sum()
    else:
        tp = w @ (y * f); fp = w @ ((1 - y) * f); fn = w @ (y * (1 - f))
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), np.nan)


def consistent(p, cls):
    p = np.clip(p, 0, 1)
    return np.where(cls == 1, 0.5 + 0.5 * p, np.minimum(0.5 * p, 0.499999))


def auc(y, s):
    m = np.isfinite(s)
    return float(roc_auc_score(y[m], s[m])) if len(np.unique(y[m])) == 2 else np.nan


def region_macro(Y, F, rows=None):
    if rows is not None:
        Y, F = Y[rows], F[rows]
    return float(np.nanmean([f1w(Y[:, j], F[:, j]) for j in range(Y.shape[1])]))


def choose_tau(Y_in, S_in, T):
    """tau с максимальной macro-F1 области на inner-OOF; при равенстве — ближайшая к «выкл»."""
    F0 = (S_in >= T[None, :]).astype(int)
    best, best_tau = -1.0, 0.0
    for tau in TAUS:
        F = F0 if tau == 0.0 else exclusive_types_array(F0, S_in, T, tau)
        m = region_macro(Y_in, F)
        if m > best + 1e-12:
            best, best_tau = m, tau
    return best_tau


def load(inp, tag, region):
    d = np.load(Path(inp) / f'b2_{tag}_{region}.npz', allow_pickle=True)
    return {k: d[k] for k in d.files}


def region_predictions(D):
    """Флаги всех вариантов по повторам: dict variant -> (R, n, C); плюс выбранные tau по фолдам."""
    Y, S, FOLD, THR, THR_TIE, SIN = D['Y'], D['S'], D['FOLD'], D['THR'], D['THR_TIE'], D['SIN']
    R, n, C = S.shape
    out = {v: np.zeros((R, n, C), int) for v in ('base', 'tie', 'excl_base', 'excl_tie', 'excl_inf_base', 'excl_inf_tie')}
    taus = {'excl_base': np.full((R, 5), np.nan), 'excl_tie': np.full((R, 5), np.nan)}
    for r in range(R):
        for k in range(5):
            te = FOLD[r] == k
            for thr_name, T_all, ex_name in (('base', THR, 'excl_base'), ('tie', THR_TIE, 'excl_tie')):
                T = T_all[r, k]
                # вырожденный фолд критерия: порог NaN, скор NaN -> флаг 0; такие строки исключаются из метрик повтора
                F = (S[r][te] >= T[None, :]).astype(int)
                out[thr_name][r][te] = F
                sin = SIN[r, k]; ok = np.isfinite(sin).all(1)
                tau = choose_tau(Y[ok], sin[ok], T) if (ok.sum() > 0 and np.isfinite(T).all()) else 0.0
                taus[ex_name][r, k] = tau
                out[ex_name][r][te] = F if tau == 0.0 else exclusive_types_array(F, S[r][te], T, tau)
                out['excl_inf_' + thr_name][r][te] = exclusive_types_array(F, S[r][te], T, math.inf)
    return out, taus


def per_repeat_metrics(D, P, criteria):
    Y, S, ANY = D['Y'], D['S'], D['ANY']
    y_any = Y.max(1)
    rows = []
    for r in range(S.shape[0]):
        ok = np.isfinite(S[r]).all(1)
        for v, F in P.items():
            Fr = F[r]
            rec = {'repeat': r, 'variant': v, 'n': int(ok.sum())}
            for j, c in enumerate(criteria):
                rec[f'f1_{c}'] = float(f1w(Y[ok, j], Fr[ok, j]))
                rec[f'nflag_{c}'] = int(Fr[ok, j].sum())
                rec[f'auc_{c}'] = auc(Y[ok, j], S[r][ok, j])
            rec['macro_region'] = float(np.nanmean([rec[f'f1_{c}'] for c in criteria]))
            cls = Fr.max(1)
            rec['f1_binary'] = float(f1w(y_any[ok], cls[ok]))
            raw = 0.5 * ANY[r] + 0.5 * np.nanmax(S[r], axis=1)
            qp = consistent(raw, cls)
            rec['auc_qprob'] = auc(y_any[ok & np.isfinite(qp)], qp[ok & np.isfinite(qp)])
            rec['auc_raw'] = auc(y_any[ok & np.isfinite(raw)], raw[ok & np.isfinite(raw)])
            rec['n_multi'] = int((Fr[ok].sum(1) >= 2).sum())
            rows.append(rec)
    return pd.DataFrame(rows)


def study_weights(studies_all, n_boot, seed=0):
    """Мультиномиальные веса исследований (общие для областей)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(studies_all)
    counts = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)), size=n_boot).astype(float)
    return uniq, counts


def boot_delta(D, P, criteria, cand, base, uniq, counts):
    """Бутстрап-распределение Δ (среднее по повторам): macro-F1 области, F1 по критериям, бинарная F1.
    Возвращает dict name -> (B,) и суммы TP/FP/FN для 5-критериального macro."""
    Y, S = D['Y'], D['S']
    pos = pd.Series(np.arange(len(uniq)), index=uniq).loc[D['study']].values
    W = counts[:, pos]                     # (B, n)
    R = S.shape[0]; C = len(criteria)
    y_any = Y.max(1)
    d_crit = np.zeros((counts.shape[0], C)); d_bin = np.zeros(counts.shape[0])
    for r in range(R):
        ok = np.isfinite(S[r]).all(1).astype(float)
        Wr = W * ok[None, :]
        for j in range(C):
            a = f1w(Y[:, j], P[cand][r][:, j], Wr); b = f1w(Y[:, j], P[base][r][:, j], Wr)
            d_crit[:, j] += np.nan_to_num(a - b) / R
        d_bin += np.nan_to_num(f1w(y_any, P[cand][r].max(1), Wr) - f1w(y_any, P[base][r].max(1), Wr)) / R
    return {'d_crit': d_crit, 'd_macro_region': d_crit.mean(1), 'd_binary': d_bin}


def ci(x, level=90):
    x = np.asarray(x); x = x[np.isfinite(x)]
    a = (100 - level) / 2
    return [round(float(np.percentile(x, a)), 4), round(float(np.percentile(x, 100 - a)), 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inp', default=str(ROOT / 'outputs' / 'b2'))
    ap.add_argument('--tag', default='all')
    ap.add_argument('--out', default=None)
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    a = ap.parse_args()
    out = Path(a.out or a.inp); out.mkdir(parents=True, exist_ok=True)
    Ds, Ps, TAU, REP, CR = {}, {}, {}, [], {}
    for region in REGIONS:
        D = load(a.inp, a.tag, region); crit = [str(c) for c in D['criteria']]
        P, taus = region_predictions(D)
        # B3: тождество OR флагов (quality_class, quality_prob, бинарная F1 не меняются)
        for thr_name in ('base', 'tie'):
            for ex in ('excl_' + thr_name, 'excl_inf_' + thr_name):
                assert (P[ex].max(2) == P[thr_name].max(2)).all(), f'{region}: OR флагов изменился ({ex})'
                assert (P[ex] <= P[thr_name]).all(), f'{region}: (б) добавил флаг ({ex})'
        rep = per_repeat_metrics(D, P, crit); rep.insert(0, 'region', region); REP.append(rep)
        Ds[region], Ps[region], TAU[region], CR[region] = D, P, taus, crit
    rep = pd.concat(REP, ignore_index=True)
    rep.to_csv(out / f'b2_{a.tag}_per_repeat.csv', index=False)
    studies_all = np.concatenate([Ds[r]['study'] for r in REGIONS])
    uniq, counts = study_weights(studies_all, a.n_boot, seed=0)
    comparisons = {'a_tie_vs_base': ('tie', 'base'), 'b_excl_vs_base': ('excl_base', 'base'),
                   'b_excl_tie_vs_tie': ('excl_tie', 'tie'), 'ab_excl_tie_vs_base': ('excl_tie', 'base'),
                   'b_inf_vs_base': ('excl_inf_base', 'base'), 'b_inf_tie_vs_tie': ('excl_inf_tie', 'tie')}
    summary = {'tag': a.tag, 'n_rows': {r: int(len(Ds[r]['Y'])) for r in REGIONS},
               'n_studies': int(len(uniq)), 'n_repeats': int(Ds['spine']['S'].shape[0]), 'n_boot': a.n_boot,
               'means': {}, 'comparisons': {}, 'tau_choice': {}}
    for region in REGIONS:
        rr = rep[rep.region == region]
        summary['means'][region] = rr.groupby('variant').mean(numeric_only=True).drop(columns=['repeat']).round(4).to_dict(orient='index')
        for name, t in TAU[region].items():
            v = t[np.isfinite(t)]
            summary['tau_choice'].setdefault(name, {})[region] = {
                'share_off': round(float((v == 0.0).mean()), 3),
                'counts': {('inf' if math.isinf(x) else str(x)): int((v == x).sum()) for x in TAUS}}
    for name, (cand, base) in comparisons.items():
        comp = {}
        boots = {region: boot_delta(Ds[region], Ps[region], CR[region], cand, base, uniq, counts) for region in REGIONS}
        # macro по 5 критериям: среднее пяти ΔF1 (одинаковые веса исследований в обеих областях)
        d5_boot = np.concatenate([boots[r]['d_crit'] for r in REGIONS], 1).mean(1)
        d5_rep = []
        for r in range(summary['n_repeats']):
            f_c = [];
            f_b = []
            for region in REGIONS:
                rr = rep[(rep.region == region) & (rep.repeat == r)].set_index('variant')
                for c in CR[region]:
                    f_c.append(rr.loc[cand, f'f1_{c}']); f_b.append(rr.loc[base, f'f1_{c}'])
            d5_rep.append(np.nanmean(f_c) - np.nanmean(f_b))
        d5_rep = np.array(d5_rep)
        comp['macro5'] = {'base_mean': None, 'delta_mean': round(float(d5_rep.mean()), 4),
                          'delta_min': round(float(d5_rep.min()), 4), 'delta_max': round(float(d5_rep.max()), 4),
                          'n_repeats_ge_0': int((d5_rep >= -1e-12).sum()), 'n_repeats_gt_0': int((d5_rep > 1e-12).sum()),
                          'ci90_cluster': ci(d5_boot), 'p_boot_le_0': round(float((d5_boot <= 0).mean()), 4)}
        f5b = []
        for region in REGIONS:
            rr = rep[rep.region == region]
            b = rr[rr.variant == base].set_index('repeat'); c = rr[rr.variant == cand].set_index('repeat')
            dm = c['macro_region'] - b['macro_region']; db = c['f1_binary'] - b['f1_binary']
            dq = c['auc_qprob'] - b['auc_qprob']
            reg = {'macro_base': round(float(b['macro_region'].mean()), 4), 'macro_cand': round(float(c['macro_region'].mean()), 4),
                   'd_macro_mean': round(float(dm.mean()), 4), 'd_macro_min': round(float(dm.min()), 4),
                   'n_repeats_d_macro_ge_0': int((dm >= -1e-12).sum()),
                   'd_macro_ci90_cluster': ci(boots[region]['d_macro_region']),
                   'd_macro_p_boot_le_0': round(float((boots[region]['d_macro_region'] <= 0).mean()), 4),
                   'f1_binary_base': round(float(b['f1_binary'].mean()), 4), 'd_binary_mean': round(float(db.mean()), 4),
                   'd_binary_ci90_cluster': ci(boots[region]['d_binary']),
                   'auc_qprob_base': round(float(b['auc_qprob'].mean()), 4), 'd_auc_qprob_mean': round(float(dq.mean()), 4),
                   'd_auc_qprob_min': round(float(dq.min()), 4),
                   'n_multi_base': round(float(b['n_multi'].mean()), 2), 'n_multi_cand': round(float(c['n_multi'].mean()), 2),
                   'per_criterion': {}}
            for j, cc in enumerate(CR[region]):
                f5b.append(float(b[f'f1_{cc}'].mean()))
                reg['per_criterion'][cc] = {'f1_base': round(float(b[f'f1_{cc}'].mean()), 4), 'f1_cand': round(float(c[f'f1_{cc}'].mean()), 4),
                                            'd_f1_ci90_cluster': ci(boots[region]['d_crit'][:, j]),
                                            'nflag_base': round(float(b[f'nflag_{cc}'].mean()), 2), 'nflag_cand': round(float(c[f'nflag_{cc}'].mean()), 2),
                                            'auc': round(float(b[f'auc_{cc}'].mean()), 4),
                                            'auc_identical': bool(np.allclose(b[f'auc_{cc}'], c[f'auc_{cc}'], equal_nan=True))}
            comp[region] = reg
        comp['macro5']['base_mean'] = round(float(np.mean(f5b)), 4)
        summary['comparisons'][name] = comp
    # решения по протоколу
    A = summary['comparisons']['a_tie_vs_base']
    dec_a = {'A1_macro5_mean_gt0_and_14of20': bool(A['macro5']['delta_mean'] > 0 and A['macro5']['n_repeats_ge_0'] >= 14),
             'A2_region_macro_ge_-0.01': bool(all(A[r]['d_macro_mean'] >= -0.01 for r in REGIONS)),
             'A3_region_binary_ge_-0.01': bool(all(A[r]['d_binary_mean'] >= -0.01 for r in REGIONS)),
             'A4_region_auc_qprob_ge_-0.005': bool(all(A[r]['d_auc_qprob_mean'] >= -0.005 for r in REGIONS)),
             'A5_macro5_ci_low_ge_-0.01': bool(A['macro5']['ci90_cluster'][0] >= -0.01)}
    summary['decision_a_conditions'] = dec_a
    dec_b = {}
    for name, off_key in (('b_excl_vs_base', 'excl_base'), ('b_excl_tie_vs_tie', 'excl_tie')):
        Bc = summary['comparisons'][name]
        dec_b[name] = {'B1_regions_ci_low_gt0': [r for r in REGIONS if Bc[r]['d_macro_ci90_cluster'][0] > 0],
                       'B2_no_region_below_-0.01': bool(all(Bc[r]['d_macro_mean'] >= -0.01 for r in REGIONS)),
                       'B3_binary_identity': True,
                       'B4_off_ge_50pct_both': bool(all(summary['tau_choice'][off_key][r]['share_off'] >= 0.5 for r in REGIONS))}
    summary['decision_b_conditions'] = dec_b
    json.dump(summary, open(out / f'b2_{a.tag}_summary.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    # входы paired_gate: область (score = quality_prob, pred = класс) и критерии (score одинаковый, pred = флаги)
    from paired_gate import write_gate_input
    for region in REGIONS:
        D, P = Ds[region], Ps[region]
        y_any = D['Y'].max(1)
        raw = 0.5 * D['ANY'] + np.where(np.isfinite(D['S']).all(2), 0.5 * np.nanmax(D['S'], axis=2), np.nan)
        cb, ct = P['base'].max(2), P['tie'].max(2)
        # NaN (вырожденный фолд в повторе) paired_gate исключает сам; строки не отбрасываются
        write_gate_input(out / f'gate_{a.tag}_{region}_qprob_tie.npz', y_any, D['groups'],
                         consistent(raw, cb), consistent(raw, ct), pred_base=cb, pred_cand=ct,
                         fold=D['FOLD'], study=D['study'])
        for j, c in enumerate(CR[region]):
            s = D['S'][:, :, j]
            write_gate_input(out / f'gate_{a.tag}_{c}_tie.npz', D['Y'][:, j], D['groups'], s, s,
                             pred_base=P['base'][:, :, j], pred_cand=P['tie'][:, :, j], fold=D['FOLD'],
                             study=D['study'])
    print(json.dumps({k: summary[k] for k in ('tag', 'n_rows', 'decision_a_conditions', 'decision_b_conditions', 'tau_choice')},
                     ensure_ascii=False, indent=1))
    for name, comp in summary['comparisons'].items():
        print(f"\n== {name}: macro5 {comp['macro5']}")
        for region in REGIONS:
            c = comp[region]
            print(f"  {region}: macro {c['macro_base']} -> {c['macro_cand']} (Δ {c['d_macro_mean']}, ДИ90 {c['d_macro_ci90_cluster']}, "
                  f"{c['n_repeats_d_macro_ge_0']}/20 >= 0) | binF1 {c['f1_binary_base']} Δ {c['d_binary_mean']} | "
                  f"AUC qprob {c['auc_qprob_base']} Δ {c['d_auc_qprob_mean']} | multi {c['n_multi_base']} -> {c['n_multi_cand']}")
            for cc, pc in c['per_criterion'].items():
                print(f"     {cc}: F1 {pc['f1_base']} -> {pc['f1_cand']} ДИ90 {pc['d_f1_ci90_cluster']} | флагов {pc['nflag_base']} -> {pc['nflag_cand']} | AUC {pc['auc']} same={pc['auc_identical']}")


if __name__ == '__main__':
    main()
