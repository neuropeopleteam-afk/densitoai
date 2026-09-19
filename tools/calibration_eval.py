"""
К3: калибровка cross-fit, три правила порога внутри nested, бэггинг порогов hip_pos, зона «не уверен».
Все числа отчёта docs/METRICS_REPORT.md (раздел «Калибровка и зона не уверен») и work/G/REPORT.md
воспроизводятся этим скриптом. OMP_NUM_THREADS=1.

Стадии (--stage):
  calib   — калибровка cross-fit на OOF (models/oof_stacked_*.csv + OOF any-модели как в eval_oof_metrics):
            any-blend региона -> isotonic; стэкнутый скор критерия -> Platt. Brier / ECE(10 бинов) до/после,
            reliability diagram (PNG). Калибратор обучается только на OOF чужих фолдов (GroupKFold по study,
            5 фолдов × N_CAL_REPEATS повторов), оценка — на своём фолде.
  nested  — три правила порога (a) текущее train_stacked (F1-опт при >=15 позитивов в train, иначе prevalence),
            (b) prevalence ×1.0, (c) prevalence ×1.4 — внутри nested 5×10 / 3 (tools/nested_gate.py, w=0.5).
            Порог выбирается на inner-OOF, метрики — на внешнем фолде; regret = лучший F1(+) на фолде − F1(+) правила.
            Плюс бэггинг порога (медиана F1-опт порогов по 200 бутстрап-ресемплам исследований inner-OOF).
  uncert  — зона «не уверен» по запасу |score − порог| на OOF: risk–coverage, выбор запаса при доле отказов
            <= MAX_REJECT по критерию и по строкам региона; проверка на стресс-тесте (гамма 0.7/1.4, шум 3 %).
  all     — всё подряд.
Выход: <OUT>/calibration/*.csv|png, <OUT>/nested_thr/*.csv, <OUT>/uncertainty/*.csv, <OUT>/k3_summary.json.
Переменные окружения: DENSITO_ROOT (корень проекта), K3_OUT (каталог результатов, по умолчанию outputs/k3),
NESTED_GATE_WORK (где лежит results/pixel_hashes.csv для групп study+pixel_hash).
"""
import os, sys, json, time, argparse, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT = Path(os.environ.get('DENSITO_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
OUT = Path(os.environ.get('K3_OUT', ROOT / 'outputs' / 'k3'))
OUT.mkdir(parents=True, exist_ok=True)
from train_stacked import (DATA_DIR, OUT_DIR as MODELS_DIR, REGION_CRITERIA, CRITERION_GEOMETRY_COLS,  # noqa: E402
                           CRITERION_LABEL_COL, REGION_ROWS, prevalence_threshold, f1_optimal_threshold)

N_BINS = 10
N_CAL_REPEATS = 5
from calibration_utils import select_margins as cu_select_margins, row_reject_rates as cu_row_reject_rates, is_uncertain  # noqa: E402
MAX_REJECT = 0.05
N_BAG = 200
MIN_POS_F1 = 15
RULES = ['f1opt_current', 'prev_x1.0', 'prev_x1.4']


# ----------------------------------------------------------------------------- метрики калибровки
def ece(y, p, n_bins=N_BINS):
    """Expected calibration error, равные по ширине бины [0,1]."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def reliability_bins(y, p, n_bins=N_BINS):
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({'bin': b, 'lo': edges[b], 'hi': edges[b + 1], 'n': int(m.sum()),
                     'mean_pred': float(p[m].mean()) if m.any() else np.nan,
                     'frac_pos': float(y[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def safe_auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan


def brier(y, p):
    return float(brier_score_loss(np.asarray(y, int), np.clip(np.asarray(p, float), 0, 1)))


def fit_platt(s, y):
    """Platt: логистическая регрессия на скоре (один признак), без регуляризации-подавления (C большой)."""
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(s, float).reshape(-1, 1), np.asarray(y, int))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt(s, a, b):
    z = a * np.asarray(s, float) + b
    return 1.0 / (1.0 + np.exp(-z))


def fit_iso(s, y):
    return IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0).fit(np.asarray(s, float), np.asarray(y, int))


def crossfit(s, y, groups, fitter, applier, n_splits=5, repeats=N_CAL_REPEATS):
    """Cross-fit калибратора: на каждом повторе GroupKFold по study калибратор обучается на train-фолдах
    и применяется к test-фолду; возвращает усреднённые по повторам калиброванные вероятности."""
    s, y = np.asarray(s, float), np.asarray(y, int)
    acc = np.zeros((repeats, len(y)))
    for r in range(repeats):
        for tr, te in GroupKFold(n_splits=n_splits, shuffle=True, random_state=100 + r).split(s, groups=groups):
            if len(np.unique(y[tr])) < 2:
                acc[r, te] = y[tr].mean()
                continue
            model = fitter(s[tr], y[tr])
            acc[r, te] = applier(model, s[te])
    return acc.mean(axis=0)


def consistent(p, cls):
    p = np.clip(p, 0, 1)
    return np.where(cls == 1, 0.5 + 0.5 * p, np.minimum(0.5 * p, 0.499999))


def clip_consistent(p, cls):
    """Калиброванная вероятность, приведённая к инварианту class=1 <=> prob>=0.5 обрезкой (монотонно внутри класса)."""
    p = np.clip(p, 0, 1)
    return np.where(cls == 1, np.maximum(0.5, p), np.minimum(0.499999, p))


# ----------------------------------------------------------------------------- OOF any-модели (как eval_oof_metrics)
def oof_any_blend(region, criteria):
    """Сырой any-blend региона на OOF: 0.5*mean(p_any_geom, p_any_emb) + 0.5*max(oof_stacked критериев),
    ровно как inference.any_violation_prob; any-модель — OOF StratifiedGroupKFold(5) × 3 сида (eval_oof_metrics)."""
    from eval_oof_metrics import oof_any_model
    frames = {c: pd.read_csv(MODELS_DIR / f'oof_stacked_{region}_{c}.csv') for c in criteria}
    summary = json.load(open(MODELS_DIR / 'metrics_summary.json', encoding='utf-8'))
    base = frames[criteria[0]][['study', 'file_path']].copy()
    flags = np.zeros(len(base), int); ytrue = np.zeros(len(base), int); cmax = np.zeros(len(base))
    for c, df in frames.items():
        assert (df['file_path'].values == base['file_path'].values).all()
        thr = summary[region][c]['threshold']
        flags = np.maximum(flags, (df['oof_stacked'].values >= thr).astype(int))
        ytrue = np.maximum(ytrue, df['y_true'].values.astype(int))
        cmax = np.maximum(cmax, df['oof_stacked'].values)
    anym = oof_any_model(region, criteria).set_index('file_path').loc[base['file_path'].values]
    assert (anym['y_any'].values == ytrue).all()
    raw = 0.5 * anym['any_model_oof'].values + 0.5 * cmax
    return base.assign(y=ytrue, cls=flags, raw=raw, any_model=anym['any_model_oof'].values, crit_max=cmax)


# ----------------------------------------------------------------------------- стадия calib
def stage_calib(log):
    od = OUT / 'calibration'; od.mkdir(exist_ok=True)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows, final = [], {'any': {}, 'criteria': {}}
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel(); ax_i = 0
    # --- any-blend региона: isotonic
    for region, criteria in REGION_CRITERIA.items():
        df = oof_any_blend(region, criteria)
        df.to_csv(od / f'oof_any_blend_{region}.csv', index=False)
        y, cls, raw, g = df['y'].values, df['cls'].values, df['raw'].values, df['study'].values
        cal = crossfit(raw, y, g, fit_iso, lambda m, s: m.predict(s))
        cal_platt = crossfit(raw, y, g, lambda s, yy: fit_platt(s, yy), lambda m, s: apply_platt(s, *m))
        cur = consistent(raw, cls)
        # калибровка внутри класса: отдельный isotonic для строк class=1 (на [0.5,1]) и class=0 (на [0,0.5)) —
        # монотонно по any-скору внутри класса, инвариант class=1 <=> prob>=0.5 сохраняется обрезкой
        cal_pc = np.zeros(len(y))
        for k in (0, 1):
            m = cls == k
            cal_pc[m] = crossfit(raw[m], y[m], g[m], fit_iso, lambda mm, s: mm.predict(s))
        cal_pc = clip_consistent(cal_pc, cls)
        cal_cur_iso = clip_consistent(crossfit(cur, y, g, fit_iso, lambda mm, s: mm.predict(s)), cls)
        cal_cur_platt = clip_consistent(crossfit(cur, y, g, lambda s, yy: fit_platt(s, yy), lambda mm, s: apply_platt(s, *mm)), cls)
        variants = {
            'raw_blend': raw,
            'constant_prevalence': np.full(len(y), y.mean()),
            'current_consistent': cur,
            'isotonic_crossfit': cal,
            'platt_crossfit': cal_platt,
            'platt_then_consistent': consistent(cal_platt, cls),
            'isotonic_then_consistent': consistent(cal, cls),
            'isotonic_clip_consistent': clip_consistent(cal, cls),
            'per_class_isotonic_clip': cal_pc,
            'consistent_then_isotonic_clip': cal_cur_iso,
            'consistent_then_platt_clip': cal_cur_platt,
            'consistent_then_platt_noclip': crossfit(cur, y, g, lambda s, yy: fit_platt(s, yy), lambda mm, s: apply_platt(s, *mm)),
        }
        for name, p in variants.items():
            rows.append({'target': f'{region}/any', 'variant': name, 'n': len(y), 'n_pos': int(y.sum()),
                         'brier': brier(y, p), 'ece': ece(y, p), 'auc': safe_auc(y, p),
                         'invariant_ok': bool(np.all((p >= 0.5) == (cls == 1)))})
        log(f"[{region}/any] n={len(y)} pos={int(y.sum())} | " + ' | '.join(
            f"{r['variant']}: Brier {r['brier']:.4f} ECE {r['ece']:.4f} AUC {r['auc']:.3f}"
            for r in rows if r['target'] == f'{region}/any'))
        rb = reliability_bins(y, raw); rb_cal = reliability_bins(y, cal); rb_cur = reliability_bins(y, consistent(raw, cls))
        rb.to_csv(od / f'reliability_{region}_any_raw.csv', index=False)
        rb_cal.to_csv(od / f'reliability_{region}_any_isotonic.csv', index=False)
        ax = axes[ax_i]; ax_i += 1
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.plot(rb['mean_pred'], rb['frac_pos'], 'o-', label='any-blend сырой')
        ax.plot(rb_cur['mean_pred'], rb_cur['frac_pos'], 's-', label='текущий quality_prob')
        ax.plot(rb_cal['mean_pred'], rb_cal['frac_pos'], '^-', label='isotonic cross-fit')
        ax.set_title(f'{region}/any (n={len(y)}, pos={int(y.sum())})'); ax.set_xlabel('предсказанная вероятность')
        ax.set_ylabel('доля позитивов'); ax.legend(fontsize=7); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        # финальный калибратор на всём OOF (для внедрения): isotonic -> таблица (x, y)
        iso = fit_iso(raw, y)
        final['any'][region] = {'method': 'isotonic', 'x': [float(v) for v in iso.X_thresholds_],
                                'y': [float(v) for v in iso.y_thresholds_], 'n': int(len(y)), 'n_pos': int(y.sum())}
    # --- критерии: Platt на стэкнутом скоре
    for region, criteria in REGION_CRITERIA.items():
        summary = json.load(open(MODELS_DIR / 'metrics_summary.json', encoding='utf-8'))
        for c in criteria:
            df = pd.read_csv(MODELS_DIR / f'oof_stacked_{region}_{c}.csv')
            y, s, g = df['y_true'].values.astype(int), df['oof_stacked'].values.astype(float), df['study'].values
            thr = float(summary[region][c]['threshold'])
            cal = crossfit(s, y, g, lambda ss, yy: fit_platt(ss, yy), lambda m, ss: apply_platt(ss, *m))
            cal_iso = crossfit(s, y, g, fit_iso, lambda m, ss: m.predict(ss))
            for name, p in {'rank_score_as_prob': s, 'platt_crossfit': cal, 'isotonic_crossfit': cal_iso,
                            'constant_prevalence': np.full(len(y), y.mean())}.items():
                rows.append({'target': f'{region}/{c}', 'variant': name, 'n': len(y), 'n_pos': int(y.sum()),
                             'brier': brier(y, p), 'ece': ece(y, p), 'auc': safe_auc(y, p), 'invariant_ok': None})
            log(f"[{region}/{c}] n={len(y)} pos={int(y.sum())} | " + ' | '.join(
                f"{r['variant']}: Brier {r['brier']:.4f} ECE {r['ece']:.4f}"
                for r in rows if r['target'] == f'{region}/{c}'))
            rb = reliability_bins(y, s); rb_cal = reliability_bins(y, cal)
            rb.to_csv(od / f'reliability_{region}_{c}_raw.csv', index=False)
            rb_cal.to_csv(od / f'reliability_{region}_{c}_platt.csv', index=False)
            ax = axes[ax_i]; ax_i += 1
            ax.plot([0, 1], [0, 1], 'k--', lw=1)
            ax.plot(rb['mean_pred'], rb['frac_pos'], 'o-', label='ранговый скор')
            ax.plot(rb_cal['mean_pred'], rb_cal['frac_pos'], '^-', label='Platt cross-fit')
            ax.axvline(thr, color='grey', ls=':', lw=1)
            ax.set_title(f'{c} (n={len(y)}, pos={int(y.sum())})'); ax.set_xlabel('скор / вероятность')
            ax.set_ylabel('доля позитивов'); ax.legend(fontsize=7); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            a, b = fit_platt(s, y)
            final['criteria'][c] = {'method': 'platt', 'a': a, 'b': b, 'threshold': thr,
                                    'p_at_threshold': float(apply_platt(np.array([thr]), a, b)[0]),
                                    'n': int(len(y)), 'n_pos': int(y.sum())}
    for j in range(ax_i, len(axes)):
        axes[j].axis('off')
    fig.suptitle('Reliability diagrams (OOF, калибратор cross-fit GroupKFold по исследованию, 10 бинов)')
    fig.tight_layout(); fig.savefig(od / 'reliability_diagrams.png', dpi=130); plt.close(fig)
    tab = pd.DataFrame(rows); tab.to_csv(od / 'calibration_metrics.csv', index=False)
    json.dump(final, open(od / 'calibrators_full_oof.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    return tab, final


# ----------------------------------------------------------------------------- стадия nested (правила порога)
def fast_f1_opt(y, s):
    """То же, что train_stacked.f1_optimal_threshold (без проверки числа позитивов), но векторно:
    F1 по всем уникальным порогам (preds = s >= t), плато-усреднение равных максимумов."""
    y, s = np.asarray(y, int), np.asarray(s, float)
    order = np.argsort(-s, kind='mergesort'); ss, yy = s[order], y[order]
    tp, fp = np.cumsum(yy), np.cumsum(1 - yy); P = y.sum()
    last = np.r_[ss[1:] != ss[:-1], True]
    tp_u, fp_u, thr_u = tp[last], fp[last], ss[last]
    denom = 2 * tp_u + fp_u + (P - tp_u)
    f1 = np.where(denom > 0, 2 * tp_u / np.maximum(denom, 1), 0.0)
    best = f1.max()
    return float(np.mean(thr_u[np.isclose(f1, best, rtol=0, atol=1e-12)]))


def rule_thresholds(y, s, n_pos_train_full):
    """Три правила на inner-OOF стэке (train внешнего фолда)."""
    prev = y.mean()
    out = {}
    if len(np.unique(y)) < 2:
        q = float(np.quantile(s, 0.9)); return {r: q for r in RULES}
    out['f1opt_current'] = fast_f1_opt(y, s) if n_pos_train_full >= MIN_POS_F1 else float(prevalence_threshold(y, s))
    out['prev_x1.0'] = float(prevalence_threshold(y, s))
    out['prev_x1.4'] = float(np.quantile(s, 1 - min(0.999, 1.4 * prev)))
    return out


def bagged_threshold(y, s, groups, n_bag=N_BAG, seed=0):
    """Медиана F1-опт порогов по бутстрап-ресемплам исследований (групп) inner-OOF."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups); by = {g: np.nonzero(groups == g)[0] for g in uniq}
    ts = []
    for _ in range(n_bag):
        idx = np.concatenate([by[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        if y[idx].sum() < 2 or y[idx].sum() == len(idx):
            continue
        ts.append(fast_f1_opt(y[idx], s[idx]))
    return float(np.median(ts)) if ts else float(fast_f1_opt(y, s)), (float(np.std(ts)) if ts else np.nan)


def fold_metrics(y, pred):
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    tp = int(((y == 1) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    return {'f1pos': f1_score(y, pred, zero_division=0), 'macro_f1': f1_score(y, pred, average='macro', zero_division=0),
            'sens': tp / (tp + fn) if tp + fn else np.nan, 'spec': tn / (tn + fp) if tn + fp else np.nan,
            'n_flag': int(pred.sum()), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def stage_nested(log):
    import nested_gate as ng
    od = OUT / 'nested_thr'; od.mkdir(exist_ok=True)
    fold_rows, rep_rows = [], []
    for region, criteria in REGION_CRITERIA.items():
        geom, E = ng.load_region(region)
        for crit in criteria:
            label_col = CRITERION_LABEL_COL.get(crit, crit)
            y_all = geom[label_col].values; valid = ~pd.isna(y_all); y = y_all[valid].astype(int)
            cols = CRITERION_GEOMETRY_COLS[crit]
            Xraw = geom[cols].values.astype(np.float64)[valid]
            src = ng.emb_source_for(crit, E); Emb = E[src][valid]
            groups = geom['group'].values[valid]
            n = len(y)
            log(f"\n=== {region}/{crit}: n={n}, n_pos={int(y.sum())}")
            variants = RULES + (['bagged_f1opt'] if crit == 'hip_pos' else [])
            score = np.full((ng.N_REPEATS, n), np.nan)
            preds = {v: np.full((ng.N_REPEATS, n), np.nan) for v in variants}
            t0 = time.time()
            for r in range(ng.N_REPEATS):
                for k, (tr, te) in enumerate(GroupKFold(n_splits=ng.N_OUTER, shuffle=True, random_state=42 + r).split(Xraw, groups=groups)):
                    if ng.degenerate(y[tr]):
                        continue
                    med = np.nanmedian(Xraw[tr], axis=0); X = ng.impute(Xraw, med)
                    og, oe = ng.inner_oof(X[tr], Emb[tr], y[tr], groups[tr], seed=1000 * r + k)
                    ok = ~np.isnan(og) & ~np.isnan(oe)
                    if ok.sum() < 2 or len(np.unique(y[tr][ok])) < 2:
                        continue
                    s_in = ng.stack(ng.pct_rank(og[ok]), ng.pct_rank(oe[ok]), ng.W_BASE)
                    y_in, g_in = y[tr][ok], groups[tr][ok]
                    thr = rule_thresholds(y_in, s_in, int(y[tr].sum()))
                    bag_sd = np.nan
                    if crit == 'hip_pos':
                        thr['bagged_f1opt'], bag_sd = bagged_threshold(y_in, s_in, g_in, seed=1000 * r + k)
                    pg = ng.fit_geom(X[tr], y[tr])(X[te]); pe = ng.fit_emb(Emb[tr], y[tr])(Emb[te])
                    s_te = ng.stack(ng.ref_rank(pg, og), ng.ref_rank(pe, oe), ng.W_BASE)
                    score[r, te] = s_te
                    row = {'region': region, 'criterion': crit, 'repeat': r, 'fold': k, 'n_train': len(tr),
                           'n_pos_train': int(y[tr].sum()), 'prev_train': float(y_in.mean()), 'n_test': len(te),
                           'n_pos_test': int(y[te].sum()), 'bag_thr_sd': bag_sd}
                    for v in variants:
                        p = (s_te >= thr[v]).astype(int); preds[v][r, te] = p
                        m = fold_metrics(y[te], p)
                        row[f'thr_{v}'] = thr[v]
                        for kk, vv in m.items():
                            row[f'{kk}_{v}'] = vv
                    fold_rows.append(row)
                # per-repeat pooled
                m_all = ~np.isnan(score[r])
                rr = {'region': region, 'criterion': crit, 'repeat': r, 'n_scored': int(m_all.sum()),
                      'auc': safe_auc(y[m_all], score[r][m_all])}
                for v in variants:
                    m = fold_metrics(y[m_all], preds[v][r][m_all])
                    for kk in ('f1pos', 'macro_f1', 'sens', 'spec', 'n_flag'):
                        rr[f'{kk}_{v}'] = m[kk]
                rep_rows.append(rr)
                log(f"  repeat {r}: " + ' | '.join(f"{v}: F1+ {rr[f'f1pos_{v}']:.3f} mF1 {rr[f'macro_f1_{v}']:.3f} "
                                                    f"sens {rr[f'sens_{v}']:.2f} spec {rr[f'spec_{v}']:.2f}" for v in variants))
            log(f"  {time.time() - t0:.0f} с")
            np.savez(od / f'outer_preds_{crit}.npz', y=y, score=score, file_path=geom['file_path'].values[valid].astype(str),
                     group=groups.astype(str), **{f'pred_{v}': preds[v] for v in variants})
    folds = pd.DataFrame(fold_rows); reps = pd.DataFrame(rep_rows)
    folds.to_csv(od / 'per_fold.csv', index=False); reps.to_csv(od / 'per_repeat.csv', index=False)
    # regret по фолдам: лучший F1(+) среди трёх правил на этом фолде минус F1(+) правила (фолды с позитивами в test)
    summ = []
    for (region, crit), f in folds.groupby(['region', 'criterion'], sort=False):
        variants = RULES + (['bagged_f1opt'] if crit == 'hip_pos' else [])
        fpos = f[f['n_pos_test'] > 0]
        best_f1 = fpos[[f'f1pos_{v}' for v in RULES]].max(axis=1)
        best_mf1 = fpos[[f'macro_f1_{v}' for v in RULES]].max(axis=1)
        rep = reps[(reps['region'] == region) & (reps['criterion'] == crit)]
        for v in variants:
            reg = best_f1 - fpos[f'f1pos_{v}']
            regm = best_mf1 - fpos[f'macro_f1_{v}']
            summ.append({'region': region, 'criterion': crit, 'rule': v, 'n_folds': len(fpos),
                         'regret_f1pos_mean': float(reg.mean()), 'regret_f1pos_sd': float(reg.std()),
                         'regret_f1pos_max': float(reg.max()), 'regret_macro_f1_mean': float(regm.mean()),
                         'n_folds_best': int((reg <= 1e-12).sum()),
                         'fold_f1pos_mean': float(fpos[f'f1pos_{v}'].mean()), 'fold_macro_f1_mean': float(fpos[f'macro_f1_{v}'].mean()),
                         'fold_sens_mean': float(fpos[f'sens_{v}'].mean()), 'fold_spec_mean': float(fpos[f'spec_{v}'].mean()),
                         'thr_mean': float(f[f'thr_{v}'].mean()), 'thr_sd': float(f[f'thr_{v}'].std()),
                         'rep_f1pos_mean': float(rep[f'f1pos_{v}'].mean()), 'rep_f1pos_sd': float(rep[f'f1pos_{v}'].std()),
                         'rep_macro_f1_mean': float(rep[f'macro_f1_{v}'].mean()), 'rep_sens_mean': float(rep[f'sens_{v}'].mean()),
                         'rep_spec_mean': float(rep[f'spec_{v}'].mean()), 'rep_nflag_mean': float(rep[f'n_flag_{v}'].mean())})
    summ = pd.DataFrame(summ); summ.to_csv(od / 'summary.csv', index=False)
    # выбор: минимальный средний regret F1(+) среди трёх правил; при равенстве (<0.005) — текущее правило
    decisions = {}
    for (region, crit), s in summ.groupby(['region', 'criterion'], sort=False):
        s3 = s[s['rule'].isin(RULES)].set_index('rule')
        cur = s3.loc['f1opt_current', 'regret_f1pos_mean']
        best = s3['regret_f1pos_mean'].idxmin()
        chosen = best if s3.loc[best, 'regret_f1pos_mean'] < cur - 0.005 else 'f1opt_current'
        decisions[crit] = {'chosen_rule': chosen, 'regret_by_rule': s3['regret_f1pos_mean'].round(4).to_dict(),
                           'macro_f1_by_rule': s3['rep_macro_f1_mean'].round(4).to_dict(),
                           'change_from_current': chosen != 'f1opt_current'}
        if crit == 'hip_pos':
            b = s.set_index('rule')
            decisions[crit]['bagging'] = {'regret_point': float(b.loc['f1opt_current', 'regret_f1pos_mean']),
                                          'regret_bagged': float(b.loc['bagged_f1opt', 'regret_f1pos_mean']),
                                          'thr_sd_point': float(b.loc['f1opt_current', 'thr_sd']),
                                          'thr_sd_bagged': float(b.loc['bagged_f1opt', 'thr_sd']),
                                          'rep_f1pos_point': float(b.loc['f1opt_current', 'rep_f1pos_mean']),
                                          'rep_f1pos_bagged': float(b.loc['bagged_f1opt', 'rep_f1pos_mean']),
                                          'accepted': bool(b.loc['bagged_f1opt', 'regret_f1pos_mean'] <= b.loc['f1opt_current', 'regret_f1pos_mean'] + 1e-12
                                                           and b.loc['bagged_f1opt', 'thr_sd'] < b.loc['f1opt_current', 'thr_sd'])}
    # уровень региона: quality_class = OR флагов критериев; сравнение комбинаций правил по повторам (paired)
    region_rows = []
    for region, criteria in REGION_CRITERIA.items():
        packs = {c: np.load(od / f'outer_preds_{c}.npz', allow_pickle=True) for c in criteria}
        # общий индекс строк региона: строки, у которых есть метка по всем критериям (по file_path)
        fps = [set(packs[c]['file_path']) for c in criteria]; common = sorted(set.intersection(*fps))
        idx = {c: pd.Series(np.arange(len(packs[c]['file_path'])), index=packs[c]['file_path']).loc[common].values for c in criteria}
        y_any = np.zeros(len(common), int)
        for c in criteria:
            y_any |= packs[c]['y'][idx[c]].astype(int)
        combos = {'current': {c: 'f1opt_current' for c in criteria},
                  'chosen_nested': {c: decisions[c]['chosen_rule'] for c in criteria},
                  'all_prev_x1.0': {c: 'prev_x1.0' for c in criteria},
                  'all_prev_x1.4': {c: 'prev_x1.4' for c in criteria}}
        if 'hip_pos' in criteria:
            combos['bagged_hip_pos'] = {c: ('bagged_f1opt' if c == 'hip_pos' else 'f1opt_current') for c in criteria}
        for r in range(ng.N_REPEATS):
            ok = np.ones(len(common), bool)
            for c in criteria:
                ok &= ~np.isnan(packs[c]['score'][r][idx[c]])
            row = {'region': region, 'repeat': r, 'n': int(ok.sum()), 'n_pos_any': int(y_any[ok].sum())}
            for name, combo in combos.items():
                p_any = np.zeros(len(common), int)
                for c in criteria:
                    p_any |= np.nan_to_num(packs[c][f"pred_{combo[c]}"][r][idx[c]]).astype(int)
                m = fold_metrics(y_any[ok], p_any[ok])
                for kk in ('f1pos', 'macro_f1', 'sens', 'spec', 'n_flag'):
                    row[f'{kk}_{name}'] = m[kk]
            region_rows.append(row)
    regdf = pd.DataFrame(region_rows); regdf.to_csv(od / 'region_level_per_repeat.csv', index=False)
    reg_summary = {}
    for region, f in regdf.groupby('region', sort=False):
        names = [c[len('f1pos_'):] for c in f.columns if c.startswith('f1pos_')]
        reg_summary[region] = {}
        for name in names:
            d_f1 = f[f'f1pos_{name}'] - f['f1pos_current']; d_m = f[f'macro_f1_{name}'] - f['macro_f1_current']
            reg_summary[region][name] = {'f1pos_mean': round(float(f[f'f1pos_{name}'].mean()), 4), 'f1pos_sd': round(float(f[f'f1pos_{name}'].std()), 4),
                                         'macro_f1_mean': round(float(f[f'macro_f1_{name}'].mean()), 4),
                                         'sens_mean': round(float(f[f'sens_{name}'].mean()), 4), 'spec_mean': round(float(f[f'spec_{name}'].mean()), 4),
                                         'd_f1pos_vs_current_mean': round(float(d_f1.mean()), 4), 'd_f1pos_min': round(float(d_f1.min()), 4), 'd_f1pos_max': round(float(d_f1.max()), 4),
                                         'n_repeats_not_worse': int((d_f1 >= -1e-12).sum()),
                                         'd_macro_f1_vs_current_mean': round(float(d_m.mean()), 4)}
    decisions['_region_level'] = reg_summary
    json.dump(decisions, open(od / 'decisions.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    log('\n' + summ[['criterion', 'rule', 'regret_f1pos_mean', 'regret_f1pos_sd', 'n_folds_best', 'rep_f1pos_mean',
                     'rep_macro_f1_mean', 'rep_sens_mean', 'rep_spec_mean', 'thr_mean', 'thr_sd']].to_string(index=False))
    log(json.dumps(decisions, indent=1, ensure_ascii=False))
    return summ, decisions


# ----------------------------------------------------------------------------- стадия uncert
def risk_coverage(y, s, thr, margins_grid):
    rows = []
    m = np.abs(s - thr)
    for q in margins_grid:
        keep = ~is_uncertain(m, q)
        pred = (s[keep] >= thr).astype(int)
        rows.append({'margin': float(q), 'reject_rate': float(1 - keep.mean()), 'n_keep': int(keep.sum()),
                     'f1pos_keep': f1_score(y[keep], pred, zero_division=0) if keep.any() else np.nan,
                     'acc_keep': float((pred == y[keep]).mean()) if keep.any() else np.nan,
                     'n_pos_keep': int(y[keep].sum()),
                     'err_in_rejected': float(((s[~keep] >= thr).astype(int) != y[~keep]).mean()) if (~keep).any() else np.nan})
    return pd.DataFrame(rows)


def stage_uncert(log, run_stress=True):
    od = OUT / 'uncertainty'; od.mkdir(exist_ok=True)
    summary = json.load(open(MODELS_DIR / 'metrics_summary.json', encoding='utf-8'))
    data, rc_all, margins = {}, [], {}
    for region, criteria in REGION_CRITERIA.items():
        for c in criteria:
            df = pd.read_csv(MODELS_DIR / f'oof_stacked_{region}_{c}.csv')
            thr = float(summary[region][c]['threshold'])
            y, s = df['y_true'].values.astype(int), df['oof_stacked'].values.astype(float)
            data[c] = (region, df, y, s, thr)
            grid = np.unique(np.r_[0.0, np.quantile(np.abs(s - thr), np.linspace(0.005, 0.3, 60))])
            rc = risk_coverage(y, s, thr, grid).assign(region=region, criterion=c)
            rc_all.append(rc)
    rc_all = pd.concat(rc_all, ignore_index=True); rc_all.to_csv(od / 'risk_coverage.csv', index=False)

    # подбор запасов — ровно той же функцией, что в train_stacked.py (calibration_utils.select_margins):
    # квота q по региону вниз от MAX_REJECT, «не уверен» <=> |s - thr| <= margin (включительно)
    mdata = {c: np.abs(s - thr) for c, (region, df, y, s, thr) in data.items()}
    for region, criteria in REGION_CRITERIA.items():
        base = data[criteria[0]][1]['file_path'].values
        for c in criteria:
            assert (data[c][1]['file_path'].values == base).all()
    margins, q = cu_select_margins(mdata, REGION_CRITERIA, max_reject=MAX_REJECT)

    def row_reject(mg):
        return cu_row_reject_rates(mdata, REGION_CRITERIA, mg)
    per_crit = []
    for c, (region, df, y, s, thr) in data.items():
        m = np.abs(s - thr); rej = is_uncertain(m, margins[c]); keep = ~rej
        pred = (s >= thr).astype(int)
        per_crit.append({'region': region, 'criterion': c, 'threshold': thr, 'margin': margins[c], 'n': len(y),
                         'n_reject': int(rej.sum()), 'reject_rate': float(rej.mean()),
                         'f1pos_all': f1_score(y, pred, zero_division=0), 'f1pos_keep': f1_score(y[keep], pred[keep], zero_division=0),
                         'acc_all': float((pred == y).mean()), 'acc_keep': float((pred[keep] == y[keep]).mean()),
                         'err_rate_rejected': float((pred[rej] != y[rej]).mean()) if rej.any() else np.nan,
                         'n_pos_rejected': int(y[rej].sum()), 'n_errors_rejected': int((pred[rej] != y[rej]).sum()),
                         'n_errors_all': int((pred != y).sum())})
    per_crit = pd.DataFrame(per_crit); per_crit.to_csv(od / 'margins_per_criterion.csv', index=False)
    rows_rej = row_reject(margins)
    log(f"квота по регионам q={q}; доля строк «не уверен» по регионам: {rows_rej}")
    log(per_crit.to_string(index=False))
    res = {'quota_by_region': q, 'max_reject_rate': MAX_REJECT, 'margin_by_criterion': margins,
           'row_reject_rate_by_region': rows_rej, 'per_criterion': per_crit.to_dict(orient='records')}
    if run_stress:
        res['stress'] = stress_test(log, margins, od)
    json.dump(res, open(od / 'uncertainty_summary.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    return res


def stress_test(log, margins, od):
    """Гамма 0.7 / 1.4 и шум 3 % на выборке tests/robustness_suite.pick_sample(60) (81 файл): переворачивается ли класс,
    и попадает ли файл (до или после искажения) в зону «не уверен» по запасу критерия."""
    import tempfile, shutil
    sys.path.insert(0, str(ROOT / 'tests'))
    import robustness_suite as rs
    from inference import DensitoInference, load_config
    cfg = load_config()
    engine = DensitoInference(cfg=cfg)
    sample = rs.pick_sample(60); files = list(sample['file_path'])
    crits_by_region = cfg['criteria_by_region']

    def hip_key(c):
        return 'hip_' + c.split('_', 1)[1] if c.startswith(('rh_', 'lh_')) else c

    def run(paths, root):
        out = []
        for p in paths:
            row, dbg = engine.process_file(Path(p), root)
            region = dbg.get('internal_region')
            # запас/зона берутся из движка (inference.score_criterion, calibration.pkl) — то же правило, что в API
            unc = bool(dbg.get('needs_review', 0)); mins = np.inf
            for c in crits_by_region.get(region, []):
                d = dbg.get(f'{c}_margin')
                if d is not None:
                    mins = min(mins, float(d))
            out.append({'file': str(p), 'status': row['processing_status'], 'cls': int(row['quality_class']),
                        'prob': float(row['quality_prob']), 'region': region, 'uncertain': unc, 'min_margin': mins,
                        'risk_level': dbg.get('risk_level')})
        return pd.DataFrame(out)

    base = run(files, None)
    tmp = Path(tempfile.mkdtemp(prefix='k3_stress_'))
    results, rows = {}, []
    for name, fn in {'гамма 0.7': rs.d_bright_up, 'гамма 1.4': rs.d_bright_down, 'шум 3 %': rs.d_noise}.items():
        d = tmp / name.replace(' ', '_').replace('%', 'p'); d.mkdir()
        outs = []
        for i, f in enumerate(files):
            o = d / f'{i:03d}.dcm'; fn(Path(f), o); outs.append(o)
        dist = run(outs, d)
        ok = (base['status'] == 'Success').values & (dist['status'] == 'Success').values
        flip = ok & (base['cls'].values != dist['cls'].values)
        in_zone = base['uncertain'].values | dist['uncertain'].values
        r = {'distortion': name, 'n': int(ok.sum()), 'n_flip': int(flip.sum()), 'flip_rate': float(flip[ok].mean()),
             'uncertain_rate_base': float(base['uncertain'].values[ok].mean()),
             'uncertain_rate_distorted': float(dist['uncertain'].values[ok].mean()),
             'flips_in_zone_base': int((flip & base['uncertain'].values).sum()),
             'flips_in_zone_either': int((flip & in_zone).sum()),
             'share_flips_in_zone_base': float((flip & base['uncertain'].values).sum() / max(1, flip.sum())),
             'share_flips_in_zone_either': float((flip & in_zone).sum() / max(1, flip.sum())),
             'nonflip_in_zone_base_rate': float((base['uncertain'].values & ok & ~flip).sum() / max(1, (ok & ~flip).sum()))}
        results[name] = r
        log(f"стресс {name}: перевороты {r['n_flip']}/{r['n']} ({r['flip_rate']:.3f}); в зоне до искажения "
            f"{r['flips_in_zone_base']} ({r['share_flips_in_zone_base']:.2f}), до или после {r['flips_in_zone_either']} "
            f"({r['share_flips_in_zone_either']:.2f}); доля «не уверен» на исходных {r['uncertain_rate_base']:.3f}")
        rows.append(pd.DataFrame({'distortion': name, 'file': base['file'], 'cls_base': base['cls'], 'cls_dist': dist['cls'],
                                  'flip': flip, 'uncertain_base': base['uncertain'], 'uncertain_dist': dist['uncertain'],
                                  'min_margin_base': base['min_margin'], 'min_margin_dist': dist['min_margin']}))
    pd.concat(rows, ignore_index=True).to_csv(od / 'stress_per_file.csv', index=False)
    shutil.rmtree(tmp, ignore_errors=True)
    return results


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all', choices=['calib', 'nested', 'uncert', 'all'])
    ap.add_argument('--no-stress', action='store_true')
    a = ap.parse_args()
    lines = []

    def log(s):
        print(s, flush=True); lines.append(str(s))

    summary = {}
    t0 = time.time()
    if a.stage in ('calib', 'all'):
        tab, final = stage_calib(log); summary['calibration'] = tab.to_dict(orient='records')
    if a.stage in ('nested', 'all'):
        summ, dec = stage_nested(log); summary['threshold_rules'] = dec
    if a.stage in ('uncert', 'all'):
        summary['uncertainty'] = stage_uncert(log, run_stress=not a.no_stress)
    p = OUT / f'k3_summary_{a.stage}.json'
    json.dump(summary, open(p, 'w', encoding='utf-8'), indent=1, ensure_ascii=False, default=float)
    (OUT / f'log_{a.stage}.txt').write_text('\n'.join(lines), encoding='utf-8')
    log(f"готово за {time.time() - t0:.0f} с -> {p}")


if __name__ == '__main__':
    main()
