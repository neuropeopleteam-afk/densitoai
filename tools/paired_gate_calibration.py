"""
Калибровка правил приёмки (идея 11) на реальной структуре данных: реальные метки, группы
(компоненты (study, pixel_hash)) и OOF-скоры базы из models/oof_stacked_<region>_<crit>.csv.

Модель повторов: base_r = base + eps_r, cand_r = base + effect + eps'_r, eps ~ N(0, sigma_split^2),
sigma_split подбирается по критерию так, чтобы sd(ΔAUC_r) по повторам совпадал с наблюдаемым в
docs/NESTED_GATE_REPORT.md (0.033–0.041). Пороги — кросс-фит по фолдам GroupKFold(5, shuffle) повтора
по правилу train_stacked (F1-опт при >= 15 позитивов в train, иначе prevalence).

Сценарии нуля:
  null_noise_rep(sigma)  — кандидат = база + свежий шум в каждом повторе (кандидат чуть хуже базы);
  null_fixed(sigma)      — кандидат = база + фиксированное для выборки отклонение eta (одно на все
                           повторы) + шум разбиений: «другая, но не лучшая модель» — реалистичный ноль,
                           истинный эффект <= 0, а по повторам ΔAUC_r почти постоянен;
  null_binormal_equal(rho) — база и кандидат — две модели с ОДИНАКОВОЙ популяционной AUC (бинормальная
                           модель, откалиброванная на реальном AUC базы, корреляция скоров rho по объектам):
                           истинный эффект ровно 0, отклонение фиксировано для выборки — главный ноль;
  null_perm_improve      — кандидат = база + улучшение (+0.05 сдвиг позитивов), случайно переставленное
                           между объектами внутри кластеров (+ шум); реализованный ΔAUC приводится.
Сценарии мощности, e ∈ {0.02, 0.03, 0.05, 0.08}:
  effect_shift(e)    — кандидат = база + сдвиг скоров позитивов на c(e) (c откалибровано: ожидаемый ΔAUC = e);
                       кандидат почти полностью коррелирован с базой — оптимистичный случай (пост-обработка);
  effect_binormal(e) — бинормальная пара с корреляцией 0.8 и популяционной AUC кандидата = AUC базы + e —
                       реалистичный случай «другая модель, которая действительно лучше».

Правила: старое (ΔAUC >= 0.03 в >= 70 % повторов и средняя macro-F1 не хуже), новое (paired_gate,
alpha=0.05 односторонний, minimal_effect=0.02) в вариантах, а также справочно парный t-test по повторам
(p < 0.05 и ΔAUC >= 0.02). Колонки результата:
  rate_new              — бутстрап-ДИ + minimal_effect + Δmacro-F1 голосования, ДИ90 (буквальный вариант постановки);
  rate_new_boot_perm    — бутстрап-ДИ + sign-flip p < alpha + minimal_effect (AUC-часть итогового правила);
  rate_new_boot_perm_f1_mean_ci60 — итоговое правило (умолчания paired_gate: + среднее по повторам Δmacro-F1, ДИ60);
  остальные rate_new_*_f1_* — прочие варианты условия по F1. Сведение в таблицы — tools/paired_gate_tables.py.

Вызов: python tools/paired_gate_calibration.py --out-dir work/idea11_gate/outputs --n-sim 200 --n-boot 500
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paired_gate as pg  # noqa: E402

ROOT = Path(os.environ.get('DENSITO_ROOT', HERE.parent))
CRITERIA = [('spine', 'sp_pos'), ('spine', 'sp_axis'), ('spine', 'sp_art'), ('hip', 'hip_pos'), ('hip', 'hip_roi')]
OBSERVED_SD = {'sp_pos': 0.0351, 'sp_axis': 0.0366, 'sp_art': 0.0412, 'hip_pos': 0.0333, 'hip_roi': 0.0327}  # NESTED_GATE_REPORT
EFFECTS = [0.02, 0.03, 0.05, 0.08]
NOISE_LEVELS = [0.05, 0.10, 0.20]
BINORMAL_RHO = [0.5, 0.8]
POWER_RHO = 0.8


def connected_groups(studies, hashes):
    """Компоненты связности графа study—pixel_hash (как в tools/nested_gate.py)."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s, h in zip(studies, hashes):
        ra, rb = find(('s', s)), find(('h', h))
        if ra != rb:
            parent[ra] = rb
    roots = [find(('s', s)) for s in studies]
    ids = {r: i for i, r in enumerate(dict.fromkeys(roots))}
    return np.array([ids[r] for r in roots])


def load_criterion(region, crit, hashes_csv):
    o = pd.read_csv(ROOT / 'models' / f'oof_stacked_{region}_{crit}.csv')
    y = o['y_true'].values.astype(int)
    base = o['oof_stacked'].values.astype(float)
    if hashes_csv and Path(hashes_csv).exists():
        h = pd.read_csv(hashes_csv)
        hmap = dict(zip(h['file_path'], h['pixel_hash']))
        ph = o['file_path'].map(hmap)
        if ph.isna().any():
            ph = ph.fillna(o['file_path'])
        group = connected_groups(o['study'].values, ph.values)
    else:
        group = pd.factorize(o['study'])[0]
    return dict(region=region, crit=crit, y=y, base=base, group=group, study=o['study'].values, n=len(y),
                n_pos=int(y.sum()), n_groups=int(group.max() + 1), auc_base=pg.fast_auc(y, base))


def make_folds(group, R, seed0=42):
    n = len(group)
    F = np.zeros((R, n), dtype=int)
    for r in range(R):
        for k, (_, te) in enumerate(GroupKFold(n_splits=5, shuffle=True, random_state=seed0 + r).split(np.zeros(n), groups=group)):
            F[r, te] = k
    return F


def binormal_pair(D, rho, rng, effect=0.0):
    """Две модели с одинаковой популяционной AUC (= AUC базы), коррелированные по объектам (rho):
    z = d*y + sqrt(rho)*u + sqrt(1-rho)*v_k, d = sqrt(2)*Phi^-1(AUC). Масштаб — sd реального скора базы."""
    from scipy.stats import norm
    y = D['y'].astype(float)
    d = np.sqrt(2) * norm.ppf(np.clip(D['auc_base'], 0.5001, 0.9999))
    dc = np.sqrt(2) * norm.ppf(np.clip(D['auc_base'] + effect, 0.5001, 0.9999))
    u, v, w = rng.normal(size=(3, D['n']))
    scale = float(np.std(D['base']))
    b = (d * y + np.sqrt(rho) * u + np.sqrt(1 - rho) * v) * scale
    c = (dc * y + np.sqrt(rho) * u + np.sqrt(1 - rho) * w) * scale
    return b, c


def simulate_scores(D, R, sigma_split, rng, effect_vec=None, eta=None, extra_rep_sigma=0.0, base=None, cand=None):
    base = D['base'] if base is None else base
    cand = base if cand is None else cand
    n = D['n']
    sb = base[None, :] + rng.normal(0, sigma_split, (R, n))
    sc = cand[None, :] + rng.normal(0, sigma_split, (R, n))
    if extra_rep_sigma > 0:
        sc = sc + rng.normal(0, extra_rep_sigma, (R, n))
    if eta is not None:
        sc = sc + eta[None, :]
    if effect_vec is not None:
        sc = sc + effect_vec[None, :]
    return sb, sc


def delta_auc_repeats(D, sb, sc):
    return np.array([pg.fast_auc(D['y'], sc[r]) - pg.fast_auc(D['y'], sb[r]) for r in range(sb.shape[0])])


def calibrate_sigma_split(D, target_sd, rng, R=20, n_draw=30):
    """sigma_split, при котором sd(ΔAUC_r) по повторам ~ target_sd (кандидат = база, без эффекта)."""
    grid = np.array([0.01, 0.02, 0.03, 0.045, 0.06, 0.08, 0.10, 0.13, 0.17, 0.22, 0.3])
    sds = []
    for s in grid:
        v = [delta_auc_repeats(D, *simulate_scores(D, R, s, rng)).std(ddof=1) for _ in range(n_draw)]
        sds.append(np.mean(v))
    sds = np.array(sds)
    return float(np.interp(target_sd, sds, grid)), dict(grid=grid.tolist(), sd=sds.tolist())


def calibrate_shift(D, effect, sigma_split, rng, R=20, n_draw=40):
    """c такое, что E[ΔAUC] = effect для кандидата base + c*y (+ шум разбиений)."""
    y = D['y'].astype(float)

    def mean_delta(c):
        vals = []
        for _ in range(n_draw):
            sb, sc = simulate_scores(D, R, sigma_split, rng, effect_vec=c * y)
            vals.append(delta_auc_repeats(D, sb, sc).mean())
        return float(np.mean(vals))

    lo, hi = 0.0, 1.5
    for _ in range(18):
        mid = (lo + hi) / 2
        if mean_delta(mid) < effect:
            lo = mid
        else:
            hi = mid
    c = (lo + hi) / 2
    return float(c), mean_delta(c)


def permute_within_groups(vec, group, rng):
    out = vec.copy()
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        if len(idx) > 1:
            out[idx] = vec[rng.permutation(idx)]
    return out


def run_one(D, sb, sc, folds, n_boot, seed, n_perm=0, alpha=0.05, minimal_effect=0.02, f1_floor=-0.01, f1_ci=90.0):
    A = dict(y=D['y'], group=D['group'], n=D['n'], R=sb.shape[0], score_base=sb, score_cand=sc, fold=folds)
    res = pg.evaluate_gate(A, alpha=alpha, minimal_effect=minimal_effect, f1_floor=f1_floor, f1_ci=f1_ci, n_boot=n_boot,
                           n_perm=n_perm, seed=seed, threshold_source='crossfit', with_jackknife=False)
    d = res['delta_auc']; c = res['conditions']; t = res['ttest_repeats_reference_only']
    f1 = res['delta_macro_f1']
    row = dict(f1_vote_lo90=f1['low_bounds']['vote'][90], f1_vote_lo80=f1['low_bounds']['vote'][80], f1_vote_lo60=f1['low_bounds']['vote'][60],
               f1_mean_lo90=f1['low_bounds']['mean'][90], f1_mean_lo80=f1['low_bounds']['mean'][80], f1_mean_lo60=f1['low_bounds']['mean'][60],
               f1_point_mean=f1['point_mean_over_repeats'],
               old_accepted=res['old_rule']['accepted'], old_n_gain=res['old_rule']['n_repeats_gain'],
               old_auc_only=bool(res['old_rule']['n_repeats_gain'] >= res['old_rule']['needed']),
               new_accepted=res['accepted'], new_ci_low_gt0=c['ci_low_gt_0'], new_point_ok=c['point_ge_minimal_effect'],
               new_f1_ok=c['f1_ci_low_ge_floor'], new_auc_only=bool(c['ci_low_gt_0'] and c['point_ge_minimal_effect']),
               ttest_p=t['p_one_sided'], ttest_accepted=bool(t['p_one_sided'] < alpha and d['point_mean_over_repeats'] >= minimal_effect),
               delta_auc=d['point_mean_over_repeats'], sd_repeats=d['sd_over_repeats'], boot_se=d['boot_se'],
               ci90_low=d['ci90'][0], p_boot=d['p_boot_one_sided'], delta_f1_vote=res['delta_macro_f1']['point_vote'],
               f1_ci_low=res['delta_macro_f1']['ci_f1_low'])
    if n_perm:
        row['perm_p'] = res['signflip_permutation']['p_one_sided']
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', default=str(HERE.parent / 'outputs'))
    ap.add_argument('--criteria', nargs='*', default=None, help='подмножество критериев, например sp_pos hip_roi')
    ap.add_argument('--n-sim', type=int, default=200)
    ap.add_argument('--n-boot', type=int, default=500)
    ap.add_argument('--repeats', nargs='*', type=int, default=[10, 20])
    ap.add_argument('--n-perm', type=int, default=0, help='перестановки в каждой симуляции (0 — не считать)')
    ap.add_argument('--scenarios', nargs='*', default=None, help='подмножество сценариев')
    ap.add_argument('--hashes', default=str(HERE.parent / 'outputs' / 'pixel_hashes.csv'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='')
    args = ap.parse_args(argv)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows, calib = [], {}
    for region, crit in CRITERIA:
        if args.criteria and crit not in args.criteria:
            continue
        D = load_criterion(region, crit, args.hashes)
        rng = np.random.default_rng(args.seed + 17 * len(calib))
        sigma_split, sd_curve = calibrate_sigma_split(D, OBSERVED_SD[crit], rng)
        shifts = {e: calibrate_shift(D, e, sigma_split, rng) for e in EFFECTS}
        calib[crit] = dict(region=region, n=D['n'], n_pos=D['n_pos'], n_groups=D['n_groups'],
                           n_pos_groups=int(len(np.unique(D['group'][D['y'] == 1]))), auc_base=D['auc_base'],
                           sigma_split=sigma_split, target_sd_repeats=OBSERVED_SD[crit], sd_curve=sd_curve,
                           shift_for_effect={str(e): dict(c=c, realized=m) for e, (c, m) in shifts.items()})
        print(f"[{crit}] n={D['n']} n_pos={D['n_pos']} групп={D['n_groups']} AUC базы={D['auc_base']:.3f} sigma_split={sigma_split:.3f} "
              + ' '.join(f"c({e})={c:.3f}->{m:+.3f}" for e, (c, m) in shifts.items()), flush=True)
        scenarios = []
        for s in NOISE_LEVELS:
            scenarios.append((f'null_noise_rep_{s:.2f}', 'null', dict(extra_rep_sigma=s)))
        for s in NOISE_LEVELS:
            scenarios.append((f'null_fixed_{s:.2f}', 'null', dict(eta_sigma=s)))
        for rho in BINORMAL_RHO:
            scenarios.append((f'null_binormal_equal_{rho:.1f}', 'null', dict(binormal_rho=rho)))
        scenarios.append(('null_perm_improve_0.05', 'null_diluted', dict(perm_effect=shifts[0.05][0])))
        for e in EFFECTS:
            scenarios.append((f'effect_shift_{e:.2f}', 'power_shift', dict(shift=shifts[e][0], effect=e)))
        for e in EFFECTS:
            scenarios.append((f'effect_binormal_{e:.2f}', 'power_binormal', dict(binormal_rho=POWER_RHO, effect=e, binormal_effect=e)))
        if args.scenarios:
            scenarios = [s for s in scenarios if s[0] in args.scenarios]
        for R in args.repeats:
            folds = make_folds(D['group'], R)
            for name, kind, spec in scenarios:
                srng = np.random.default_rng(abs(hash((crit, name, R, args.seed))) % (2**32))
                for i in range(args.n_sim):
                    eta = srng.normal(0, spec['eta_sigma'], D['n']) if 'eta_sigma' in spec else None
                    eff = None
                    if 'shift' in spec:
                        eff = spec['shift'] * D['y']
                    if 'perm_effect' in spec:
                        eff = permute_within_groups(spec['perm_effect'] * D['y'].astype(float), D['group'], srng)
                    bvec = cvec = None
                    if 'binormal_rho' in spec:
                        bvec, cvec = binormal_pair(D, spec['binormal_rho'], srng, effect=spec.get('binormal_effect', 0.0))
                    sb, sc = simulate_scores(D, R, sigma_split, srng, effect_vec=eff, eta=eta, extra_rep_sigma=spec.get('extra_rep_sigma', 0.0),
                                             base=bvec, cand=cvec)
                    row = run_one(D, sb, sc, folds, args.n_boot, seed=int(srng.integers(2**31)), n_perm=args.n_perm)
                    row.update(criterion=crit, region=region, scenario=name, kind=kind, n_repeats=R, sim=i,
                               true_effect=spec.get('effect', 0.0), n_pos=D['n_pos'])
                    rows.append(row)
                sub = pd.DataFrame([r for r in rows if r['criterion'] == crit and r['scenario'] == name and r['n_repeats'] == R])
                print(f"  {crit} R={R} {name:24s} old={sub['old_accepted'].mean():.3f} new={sub['new_accepted'].mean():.3f} "
                      f"new_auc_only={sub['new_auc_only'].mean():.3f} ttest={sub['ttest_accepted'].mean():.3f} "
                      f"dAUC={sub['delta_auc'].mean():+.4f} sd_rep={sub['sd_repeats'].mean():.4f} boot_se={sub['boot_se'].mean():.4f} "
                      f"[{time.time() - t0:.0f} с]", flush=True)
    raw = pd.DataFrame(rows)
    tag = f'_{args.tag}' if args.tag else ''
    raw.to_csv(out / f'gate_calibration_raw{tag}.csv', index=False)
    agg_cols = dict(rate_old=('old_accepted', 'mean'), rate_old_auc_only=('old_auc_only', 'mean'), rate_new=('new_accepted', 'mean'),
                    rate_new_auc_only=('new_auc_only', 'mean'), rate_new_ci_low_gt0=('new_ci_low_gt0', 'mean'),
                    rate_new_f1_ok=('new_f1_ok', 'mean'), rate_ttest=('ttest_accepted', 'mean'),
                    mean_delta_auc=('delta_auc', 'mean'), sd_delta_auc_between_sims=('delta_auc', 'std'),
                    mean_sd_repeats=('sd_repeats', 'mean'), mean_boot_se=('boot_se', 'mean'), n_sim=('sim', 'count'))
    for var, col in (('f1_vote_ci90', 'f1_vote_lo90'), ('f1_vote_ci80', 'f1_vote_lo80'), ('f1_vote_ci60', 'f1_vote_lo60'),
                     ('f1_mean_ci90', 'f1_mean_lo90'), ('f1_mean_ci80', 'f1_mean_lo80'), ('f1_mean_ci60', 'f1_mean_lo60'),
                     ('f1_mean_point', 'f1_point_mean')):
        raw[f'new_{var}'] = raw['new_auc_only'] & (raw[col] >= -0.01)
        agg_cols[f'rate_new_{var}'] = (f'new_{var}', 'mean')
    if 'perm_p' in raw:
        raw['perm_accepted'] = raw['perm_p'] < 0.05
        agg_cols['rate_perm_p_lt_0.05'] = ('perm_accepted', 'mean')
        raw['new_boot_perm'] = raw['new_auc_only'] & raw['perm_accepted']
        agg_cols['rate_new_boot_perm'] = ('new_boot_perm', 'mean')
        for var, col in (('f1_vote_ci90', 'f1_vote_lo90'), ('f1_vote_ci80', 'f1_vote_lo80'), ('f1_mean_ci80', 'f1_mean_lo80'),
                         ('f1_mean_ci60', 'f1_mean_lo60'), ('f1_mean_point', 'f1_point_mean')):
            raw[f'new_boot_perm_{var}'] = raw['new_boot_perm'] & (raw[col] >= -0.01)
            agg_cols[f'rate_new_boot_perm_{var}'] = (f'new_boot_perm_{var}', 'mean')
    table = raw.groupby(['criterion', 'n_pos', 'kind', 'scenario', 'true_effect', 'n_repeats'], as_index=False).agg(**agg_cols)
    table.to_csv(out / f'gate_calibration{tag}.csv', index=False)
    with open(out / f'gate_calibration{tag}.json', 'w', encoding='utf-8') as f:
        json.dump(pg._json_ready(dict(params=vars(args), rules=dict(
            old='ΔAUC >= 0.03 в >= 70 % повторов и средняя macro-F1 кандидата не ниже базы',
            new='нижняя граница кластерного ДИ90 ΔAUC > 0, ΔAUC >= 0.02, нижняя граница ДИ90 Δmacro-F1 >= -0.01',
            new_auc_only='то же без условия по macro-F1', ttest='парный t-test по повторам p < 0.05 и ΔAUC >= 0.02 (справочно)'),
            calibration=calib, table=table.to_dict(orient='records'))), f, indent=2, ensure_ascii=False)
    print(f'готово за {time.time() - t0:.0f} с; {out}')
    return table


if __name__ == '__main__':
    main()
