"""
Идеи 12–15 бэклога: кандидаты против боевой конфигурации 2.3.2 на одних и тех же фолдах
nested repeated GroupKFold (на основе tools/nested_gate.py; сам nested_gate.py не меняется).

Протокол (как в nested_gate.py):
  внешний контур: GroupKFold(5, shuffle=True, random_state=42+r), r = 0..N_REPEATS-1;
                  группы = компоненты связности (study, pixel_hash) — дубликаты не расходятся по фолдам;
  внутренний контур: GroupKFold(3, shuffle=True, random_state=1000*r+k) на внешнем train -> inner-OOF geom/emb;
  модели ровно как в src/train_stacked.py: geom = StandardScaler -> LogReg(C=1, balanced);
                                           emb  = StandardScaler -> PCA(32) -> LogReg(C=0.1, balanced);
  внешний test: модели на всём внешнем train; ранги относительно inner-OOF-референса
  (inference.ModelRegistry.percentile_rank); стэк = w*rank_geom + (1-w)*rank_emb.

Отличия от nested_gate.py — всё «как в бою» (config.yaml):
  * контур A читает geometry_features_<variant>.csv по preprocessing.variant_by_criterion[crit].geom;
  * контур B читает embeddings_<source>[_<variant>].npy по embeddings.source_by_criterion и
    preprocessing.variant_by_criterion[crit].emb (train_stacked.emb_matrix_for);
  * порог — по правилу thresholds_rule.by_criterion (calibration_utils.threshold_by_rule) на inner-OOF стэке;
  * база — w_geom из config.yaml (stacking.weights_by_criterion / weight_geom, сейчас 0.5 везде);
  * строки региона — те, у кого валидны метки ВСЕХ критериев региона (spine 166/166, hip 329/329), поэтому
    внешние разбиения общие для критериев региона и для уровня региона (quality_prob);
  * уровень региона (quality_prob), как inference.any_violation_prob + consistent_quality_prob:
      any_model = mean(p_any_geom, p_any_emb) (any-модели: OR критериев, geom baseline, emb imagenet baseline);
      crit_agg  = max(score_crit) (боевое) или 1 - prod(1 - score_crit) (noisy_or);
      quality_prob_raw = 0.5*any_model + 0.5*crit_agg;
      quality_class = OR(score_crit >= thr_crit); quality_prob = 0.5+0.5*raw при class=1, иначе min(0.5*raw, 0.499999).

Кандидаты (--candidate):
  tz_weights      — фиксированные априорные веса w_geom по критерию (docs/NESTED_GATE_REPORT.md, часть 3);
  noisy_or        — та же модель по критериям, другая агрегация в quality_prob; оценка на уровне региона;
  dupweight       — sample_weight = 1/кратность pixel_hash в обучении geom/emb логрегрессий и StandardScaler
                    (PCA sklearn веса не принимает — обучается без весов); оценка без весов;
  hip_side_router — бедро: правая сторона (hip_side_detected == right) w_geom=1.0, левая — 0.0; порог общий
                    (prevalence на объединённом inner-OOF), оценка на объединённом уровне hip_pos / hip_roi.

Выход (--out DIR): <crit>_scores.csv (repeat,row_id,y,group,study,fold,score_base,score_cand,threshold_base,
threshold_cand,pred_base,pred_cand), region_<region>_scores.csv (уровень quality_prob), per_repeat.csv,
per_repeat_region.csv, summary.json, log.txt.

Запуск: DENSITO_ROOT=<репозиторий> PIXEL_HASHES=<csv> python tools/backlog_gate.py --candidate tz_weights --repeats 20 --out <dir>
"""
import os, sys, json, time, argparse, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(os.environ.get('DENSITO_ROOT', Path(__file__).resolve().parents[1]))
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))
from train_stacked import (DATA_DIR, REGION_CRITERIA, CRITERION_GEOMETRY_COLS, CRITERION_LABEL_COL,  # noqa: E402
                           REGION_ROWS, GEOM_VARIANT_FILES, load_embeddings_by_source, emb_matrix_for,
                           preproc_for, threshold_rule_for, weight_geom_for, PCA_COMPONENTS,
                           f1_optimal_threshold, prevalence_threshold)
from calibration_utils import threshold_by_rule  # noqa: E402

N_OUTER, N_INNER = 5, 3
GEOM_C, EMB_C = 1.0, 0.1
GAIN_MIN, GAIN_SHARE = 0.03, 0.70          # старое правило приёмки: ΔAUC >= 0.03 в >= 70 % повторов без потери macro-F1
N_BOOT = 1000
CANDIDATES = ('tz_weights', 'noisy_or', 'dupweight', 'hip_side_router')
# Идея 13: априорные веса из смысла критерия ТЗ (зафиксированы до запуска, см. TZ_WEIGHTS_PRIOR.md)
TZ_WEIGHTS = {'sp_axis': 0.75, 'hip_roi': 0.75, 'sp_art': 0.25, 'sp_pos': 0.5, 'hip_pos': 0.5}
# Идея 15: вес контура A по обнаруженной стороне бедра
ROUTER_W = {'right': 1.0, 'left': 0.0}
ANY_BLEND = 0.5  # config stacking.any_blend_weight_model


# ----------------------------------------------------------------------------- модели (как train_stacked.py)
def fit_geom(X, y, sw=None):
    sc = StandardScaler().fit(X, sample_weight=sw)
    clf = LogisticRegression(max_iter=1000, C=GEOM_C, class_weight='balanced').fit(sc.transform(X), y, sample_weight=sw)
    return lambda Z: clf.predict_proba(sc.transform(Z))[:, 1]


def fit_emb(E, y, sw=None):
    sc = StandardScaler().fit(E, sample_weight=sw)
    n_comp = min(PCA_COMPONENTS, len(y) - 1, E.shape[1])
    pca = PCA(n_components=n_comp, random_state=42).fit(sc.transform(E))   # PCA: sample_weight не поддерживается
    clf = LogisticRegression(max_iter=1000, C=EMB_C, class_weight='balanced').fit(pca.transform(sc.transform(E)), y, sample_weight=sw)
    return lambda Z: clf.predict_proba(pca.transform(sc.transform(Z)))[:, 1]


def degenerate(y):
    return y.sum() < 2 or y.sum() == len(y)


def inner_oof(X, E, y, groups, seed, sw=None):
    og, oe = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for tr, va in GroupKFold(n_splits=N_INNER, shuffle=True, random_state=seed).split(X, groups=groups):
        if degenerate(y[tr]):
            continue
        s = None if sw is None else sw[tr]
        og[va] = fit_geom(X[tr], y[tr], s)(X[va])
        oe[va] = fit_emb(E[tr], y[tr], s)(E[va])
    return og, oe


def pct_rank(v):
    return pd.Series(v).rank(pct=True).values


def ref_rank(scores, ref):
    """inference.ModelRegistry.percentile_rank: доля референса <= score."""
    ref = np.sort(ref[~np.isnan(ref)])
    return np.searchsorted(ref, scores, side='right') / len(ref)


def stack(rg, re_, w):
    """w — скаляр или вектор по строкам (роутер по стороне)."""
    return w * rg + (1.0 - np.asarray(w)) * re_


def choose_threshold(rule, y, s):
    if len(np.unique(y)) < 2:
        return float(np.quantile(s, 0.9)), 'degenerate_q90'
    t, m = threshold_by_rule(rule, y, s, f1_optimal_threshold, prevalence_threshold, min_pos_f1=15)
    return float(t), m


def safe_auc(y, s):
    m = ~np.isnan(s)
    if m.sum() == 0 or len(np.unique(y[m])) < 2:
        return np.nan
    return roc_auc_score(y[m], s[m])


def macro_f1(y, p):
    return f1_score(y, p, average='macro', zero_division=0)


def boot_ci(studies, fn, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(studies)
    by_study = {s: np.nonzero(studies == s)[0] for s in uniq}
    vals = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_study[s] for s in samp])
        v = fn(idx)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.mean(vals)), float(np.percentile(vals, 97.5)))


def consistent_qp(prob, cls):
    """inference.consistent_quality_prob (векторно)."""
    p = np.clip(prob, 0, 1)
    return np.where(cls == 1, 0.5 + 0.5 * p, np.minimum(0.5 * p, 0.499999))


# ----------------------------------------------------------------------------- данные
def connected_groups(studies, hashes):
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for s, h in zip(studies, hashes):
        union(('s', s), ('h', h))
    roots = [find(('s', s)) for s in studies]
    ids = {r: i for i, r in enumerate(dict.fromkeys(roots))}
    return np.array([ids[r] for r in roots])


def impute(Xraw, medians):
    X = Xraw.copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j])
        X[m, j] = medians[j]
    return X


def load_region(region, hashes_path):
    rows = REGION_ROWS[region]
    geom_by_variant = {}
    for variant, fn in GEOM_VARIANT_FILES.items():
        f = DATA_DIR / fn
        if f.exists():
            g = pd.read_csv(f)
            geom_by_variant[variant] = g[g['region'].isin(rows)].reset_index(drop=True)
    geom = geom_by_variant['baseline']
    for v, g in geom_by_variant.items():
        assert (g['file_path'].values == geom['file_path'].values).all(), f"порядок строк варианта {v}"
    lab = pd.read_csv(DATA_DIR / 'labels_for_embeddings.csv')
    emb_by_source = load_embeddings_by_source()
    eidx = np.nonzero(lab['region'].isin(rows).values)[0]
    assert (lab.loc[eidx, 'file_path'].values == geom['file_path'].values).all()
    E = {k: v[eidx] for k, v in emb_by_source.items()}
    hashes = pd.read_csv(hashes_path)
    hmap = dict(zip(hashes['file_path'], hashes['pixel_hash']))
    geom['pixel_hash'] = geom['file_path'].map(hmap)
    assert geom['pixel_hash'].notna().all(), 'нет pixel_hash для части файлов'
    geom['group'] = connected_groups(geom['study'].values, geom['pixel_hash'].values)
    # кратность = размер группы одинаковых pixel_hash по всему набору файлов
    mult = hashes['pixel_hash'].value_counts()
    geom['multiplicity'] = geom['pixel_hash'].map(mult).astype(int)
    return geom, geom_by_variant, E


# ----------------------------------------------------------------------------- основной прогон по региону
def run_region(region, criteria, candidate, n_repeats, hashes_path, out, log):
    geom, geom_by_variant, E = load_region(region, hashes_path)
    labels = np.column_stack([geom[CRITERION_LABEL_COL.get(c, c)].values.astype(float) for c in criteria])
    full = ~np.isnan(labels).any(axis=1)
    if (~full).sum():
        log(f"[{region}] строк без полного набора меток: {(~full).sum()} — исключены")
    g = geom[full].reset_index(drop=True)
    idx_full = np.nonzero(full)[0]
    n = len(g)
    groups, studies = g['group'].values, g['study'].values
    mult = g['multiplicity'].values.astype(float)
    sw_all = 1.0 / mult
    side = g['hip_side_detected'].values.astype(str) if region == 'hip' else None
    y_any = (np.nanmax(labels[full], axis=1) > 0).astype(int)
    log(f"[{region}] n={n}, групп={g['group'].nunique()}, исследований={g['study'].nunique()}, "
        f"кратность: {dict(pd.Series(mult.astype(int)).value_counts().sort_index())}")

    # данные по критериям (как в бою)
    crit_data = {}
    for c in criteria:
        pp = preproc_for(c)
        gv = pp['geom'] if pp['geom'] in geom_by_variant else 'baseline'
        cols = CRITERION_GEOMETRY_COLS[c]
        Xraw = geom_by_variant[gv][cols].values.astype(np.float64)[idx_full]
        Emb, src, ev = emb_matrix_for(c, E)
        Emb = Emb[idx_full]
        y = g[CRITERION_LABEL_COL.get(c, c)].values.astype(int)
        rule = threshold_rule_for(c)
        w_base = weight_geom_for(c)
        if candidate == 'tz_weights':
            w_cand = TZ_WEIGHTS[c]
        elif candidate == 'hip_side_router' and region == 'hip':
            w_cand = np.array([ROUTER_W[s] for s in side])
        else:
            w_cand = w_base
        crit_data[c] = dict(Xraw=Xraw, Emb=Emb, y=y, rule=rule, w_base=w_base, w_cand=w_cand, cols=cols,
                            geom_variant=gv, emb_source=src, emb_variant=ev)
        log(f"  {c}: n_pos={int(y.sum())}, контур A '{gv}' {cols}, контур B '{src}'/'{ev}', порог '{rule}', "
            f"w_base={w_base}, w_cand={'по стороне ' + str(ROUTER_W) if isinstance(w_cand, np.ndarray) else w_cand}")
    # any-модели: baseline geom (объединение колонок), imagenet baseline emb (как train_final_models.py)
    cols_any = sorted({cc for cr in criteria for cc in CRITERION_GEOMETRY_COLS[cr]})
    Xraw_any = geom_by_variant['baseline'][cols_any].values.astype(np.float64)[idx_full]
    E_any = E['imagenet'][idx_full]

    S = {c: dict(base=np.full((n_repeats, n), np.nan), cand=np.full((n_repeats, n), np.nan),
                 thr_b=np.full((n_repeats, n), np.nan), thr_c=np.full((n_repeats, n), np.nan),
                 fold=np.full((n_repeats, n), -1)) for c in criteria}
    R = dict(any_model=np.full((n_repeats, n), np.nan), fold=np.full((n_repeats, n), -1))
    fold_rows = []
    t0 = time.time()
    for r in range(n_repeats):
        for k, (tr, te) in enumerate(GroupKFold(n_splits=N_OUTER, shuffle=True, random_state=42 + r).split(np.zeros((n, 1)), groups=groups)):
            sw_tr = sw_all[tr]
            for c in criteria:
                d = crit_data[c]
                y = d['y']
                if degenerate(y[tr]):
                    continue
                med = np.nanmedian(d['Xraw'][tr], axis=0)
                X = impute(d['Xraw'], med)
                Emb = d['Emb']
                # --- база: модели без весов
                og, oe = inner_oof(X[tr], Emb[tr], y[tr], groups[tr], seed=1000 * r + k)
                ok = ~np.isnan(og) & ~np.isnan(oe)
                if ok.sum() < 2:
                    continue
                rg_in, re_in = np.full(len(tr), np.nan), np.full(len(tr), np.nan)
                rg_in[ok], re_in[ok] = pct_rank(og[ok]), pct_rank(oe[ok])
                pg = fit_geom(X[tr], y[tr])(X[te]); pe = fit_emb(Emb[tr], y[tr])(Emb[te])
                rg_te, re_te = ref_rank(pg, og), ref_rank(pe, oe)
                wb = d['w_base']
                thr_b, mb = choose_threshold(d['rule'], y[tr][ok], stack(rg_in[ok], re_in[ok], wb))
                sb = stack(rg_te, re_te, wb)
                # --- кандидат
                if candidate == 'dupweight':
                    ogw, oew = inner_oof(X[tr], Emb[tr], y[tr], groups[tr], seed=1000 * r + k, sw=sw_tr)
                    okw = ~np.isnan(ogw) & ~np.isnan(oew)
                    rgw, rew = np.full(len(tr), np.nan), np.full(len(tr), np.nan)
                    rgw[okw], rew[okw] = pct_rank(ogw[okw]), pct_rank(oew[okw])
                    pgw = fit_geom(X[tr], y[tr], sw_tr)(X[te]); pew = fit_emb(Emb[tr], y[tr], sw_tr)(Emb[te])
                    thr_c, mc = choose_threshold(d['rule'], y[tr][okw], stack(rgw[okw], rew[okw], wb))
                    sc = stack(ref_rank(pgw, ogw), ref_rank(pew, oew), wb)
                    auc_g_c, auc_e_c = safe_auc(y[te], pgw), safe_auc(y[te], pew)
                else:
                    wc = d['w_cand']
                    wc_tr = wc[tr] if isinstance(wc, np.ndarray) else wc
                    wc_te = wc[te] if isinstance(wc, np.ndarray) else wc
                    s_in = stack(rg_in, re_in, wc_tr)
                    thr_c, mc = choose_threshold(d['rule'], y[tr][ok], s_in[ok])
                    sc = stack(rg_te, re_te, wc_te)
                    auc_g_c, auc_e_c = np.nan, np.nan
                S[c]['base'][r, te], S[c]['cand'][r, te] = sb, sc
                S[c]['thr_b'][r, te], S[c]['thr_c'][r, te] = thr_b, thr_c
                S[c]['fold'][r, te] = k
                fold_rows.append({'region': region, 'criterion': c, 'repeat': r, 'fold': k, 'n_train': len(tr),
                                  'n_pos_train': int(y[tr].sum()), 'n_test': len(te), 'n_pos_test': int(y[te].sum()),
                                  'thr_base': thr_b, 'thr_cand': thr_c, 'thr_method_base': mb, 'thr_method_cand': mc,
                                  'outer_auc_geom': safe_auc(y[te], pg), 'outer_auc_emb': safe_auc(y[te], pe),
                                  'outer_auc_geom_cand': auc_g_c, 'outer_auc_emb_cand': auc_e_c,
                                  'outer_auc_base': safe_auc(y[te], sb), 'outer_auc_cand': safe_auc(y[te], sc)})
            # any-модель региона (в quality_prob одинакова у базы и кандидата; при dupweight тоже без весов —
            # идея 12 касается логрегрессий критериев; отдельно фиксируем это в отчёте)
            med = np.nanmedian(Xraw_any[tr], axis=0)
            Xa = impute(Xraw_any, med)
            if not degenerate(y_any[tr]):
                pga = fit_geom(Xa[tr], y_any[tr])(Xa[te]); pea = fit_emb(E_any[tr], y_any[tr])(E_any[te])
                R['any_model'][r, te] = 0.5 * pga + 0.5 * pea
                R['fold'][r, te] = k
        log(f"  [{region}] повтор {r} готов ({time.time() - t0:.0f} с)")

    # ------------------------------------------------------------------ метрики по критериям
    per_repeat, decisions = [], []
    for c in criteria:
        y = crit_data[c]['y']
        sb, sc = S[c]['base'], S[c]['cand']
        pb = (sb >= S[c]['thr_b']).astype(float); pc = (sc >= S[c]['thr_c']).astype(float)
        pb[np.isnan(sb)] = np.nan; pc[np.isnan(sc)] = np.nan
        rows = []
        for r in range(n_repeats):
            m = ~np.isnan(sb[r]) & ~np.isnan(sc[r])
            yb = y[m]
            row = {'region': region, 'criterion': c, 'repeat': r, 'n_scored': int(m.sum()),
                   'auc_base': safe_auc(yb, sb[r][m]), 'auc_cand': safe_auc(yb, sc[r][m]),
                   'f1pos_base': f1_score(yb, pb[r][m], zero_division=0), 'f1pos_cand': f1_score(yb, pc[r][m], zero_division=0),
                   'macro_f1_base': macro_f1(yb, pb[r][m]), 'macro_f1_cand': macro_f1(yb, pc[r][m]),
                   'n_flag_base': int(np.nansum(pb[r][m])), 'n_flag_cand': int(np.nansum(pc[r][m]))}
            if region == 'hip':
                for sd in ('right', 'left'):
                    ms = m & (side == sd)
                    row[f'auc_base_{sd}'] = safe_auc(y[ms], sb[r][ms]); row[f'auc_cand_{sd}'] = safe_auc(y[ms], sc[r][ms])
                    row[f'f1pos_base_{sd}'] = f1_score(y[ms], pb[r][ms], zero_division=0)
                    row[f'f1pos_cand_{sd}'] = f1_score(y[ms], pc[r][ms], zero_division=0)
            row['delta_auc'] = row['auc_cand'] - row['auc_base']
            row['delta_macro_f1'] = row['macro_f1_cand'] - row['macro_f1_base']
            rows.append(row)
        rep = pd.DataFrame(rows); per_repeat.append(rep)
        sbm, scm = np.nanmean(sb, axis=0), np.nanmean(sc, axis=0)
        pbv, pcv = (np.nanmean(pb, axis=0) >= 0.5).astype(int), (np.nanmean(pc, axis=0) >= 0.5).astype(int)
        n_gain = int((rep['delta_auc'] >= GAIN_MIN).sum())
        dec = {'region': region, 'criterion': c, 'candidate': candidate, 'n': n, 'n_pos': int(y.sum()),
               'geom_variant': crit_data[c]['geom_variant'], 'emb_source': crit_data[c]['emb_source'],
               'emb_variant': crit_data[c]['emb_variant'], 'threshold_rule': crit_data[c]['rule'],
               'w_base': crit_data[c]['w_base'],
               'w_cand': (str(ROUTER_W) if isinstance(crit_data[c]['w_cand'], np.ndarray) else crit_data[c]['w_cand']),
               'n_repeats': n_repeats,
               'auc_base_mean_over_repeats': float(rep['auc_base'].mean()), 'auc_cand_mean_over_repeats': float(rep['auc_cand'].mean()),
               'auc_base_sd': float(rep['auc_base'].std(ddof=1)) if n_repeats > 1 else np.nan,
               'auc_cand_sd': float(rep['auc_cand'].std(ddof=1)) if n_repeats > 1 else np.nan,
               'mean_delta_auc': float(rep['delta_auc'].mean()), 'min_delta_auc': float(rep['delta_auc'].min()),
               'max_delta_auc': float(rep['delta_auc'].max()),
               'n_repeats_gain_ge_0.03': n_gain, 'share_repeats_gain_ge_0.03': n_gain / n_repeats,
               'n_repeats_delta_pos': int((rep['delta_auc'] > 1e-12).sum()),
               'macro_f1_base_mean': float(rep['macro_f1_base'].mean()), 'macro_f1_cand_mean': float(rep['macro_f1_cand'].mean()),
               'mean_delta_macro_f1': float(rep['delta_macro_f1'].mean()),
               'f1pos_base_mean': float(rep['f1pos_base'].mean()), 'f1pos_cand_mean': float(rep['f1pos_cand'].mean()),
               'auc_base_pooled': float(safe_auc(y, sbm)), 'auc_cand_pooled': float(safe_auc(y, scm)),
               'auc_base_ci': boot_ci(studies, lambda idx: safe_auc(y[idx], sbm[idx])),
               'auc_cand_ci': boot_ci(studies, lambda idx: safe_auc(y[idx], scm[idx])),
               'delta_auc_pooled_ci': boot_ci(studies, lambda idx: safe_auc(y[idx], scm[idx]) - safe_auc(y[idx], sbm[idx])),
               'macro_f1_base_pooled': float(macro_f1(y, pbv)), 'macro_f1_cand_pooled': float(macro_f1(y, pcv)),
               'delta_macro_f1_pooled_ci': boot_ci(studies, lambda idx: macro_f1(y[idx], pcv[idx]) - macro_f1(y[idx], pbv[idx]))}
        if region == 'hip':
            for sd in ('right', 'left'):
                dec[f'auc_base_mean_{sd}'] = float(rep[f'auc_base_{sd}'].mean()); dec[f'auc_cand_mean_{sd}'] = float(rep[f'auc_cand_{sd}'].mean())
                dec[f'n_pos_{sd}'] = int(y[side == sd].sum()); dec[f'n_{sd}'] = int((side == sd).sum())
        dec['accepted_old_rule'] = bool(n_gain >= np.ceil(GAIN_SHARE * n_repeats) - 1e-9
                                        and dec['macro_f1_cand_mean'] >= dec['macro_f1_base_mean'] - 1e-12)
        decisions.append(dec)
        log(f"  --> {c}: AUC база {dec['auc_base_mean_over_repeats']:.3f} кандидат {dec['auc_cand_mean_over_repeats']:.3f} "
            f"ΔAUC {dec['mean_delta_auc']:+.3f} [{dec['min_delta_auc']:+.3f}; {dec['max_delta_auc']:+.3f}], ≥0.03 в {n_gain}/{n_repeats}, "
            f"Δmacro-F1 {dec['mean_delta_macro_f1']:+.3f}, ДИ ΔAUC(pooled) [{dec['delta_auc_pooled_ci'][0]:+.3f}; {dec['delta_auc_pooled_ci'][2]:+.3f}] "
            f"=> {'принят (старое правило)' if dec['accepted_old_rule'] else 'не принят (старое правило)'}")
        # per-repeat скоры в длинном формате (для paired_gate)
        recs = []
        for r in range(n_repeats):
            m = ~np.isnan(sb[r])
            recs.append(pd.DataFrame({'repeat': r, 'row_id': np.nonzero(m)[0], 'y': y[m], 'group': groups[m], 'study': studies[m],
                                      'fold': S[c]['fold'][r][m], 'score_base': sb[r][m], 'score_cand': sc[r][m],
                                      'threshold_base': S[c]['thr_b'][r][m], 'threshold_cand': S[c]['thr_c'][r][m],
                                      'pred_base': pb[r][m].astype(int), 'pred_cand': pc[r][m].astype(int),
                                      **({'side': side[m]} if region == 'hip' else {})}))
        pd.concat(recs, ignore_index=True).to_csv(out / f'{c}_scores.csv', index=False, float_format='%.10g')

    # ------------------------------------------------------------------ уровень региона: quality_prob
    reg_rows, recs = [], []
    QB, QC = np.full((n_repeats, n), np.nan), np.full((n_repeats, n), np.nan)
    for r in range(n_repeats):
        SB = np.column_stack([S[c]['base'][r] for c in criteria]); SC = np.column_stack([S[c]['cand'][r] for c in criteria])
        PB = np.column_stack([(S[c]['base'][r] >= S[c]['thr_b'][r]) for c in criteria])
        PC = np.column_stack([(S[c]['cand'][r] >= S[c]['thr_c'][r]) for c in criteria])
        m = ~np.isnan(SB).any(axis=1) & ~np.isnan(SC).any(axis=1) & ~np.isnan(R['any_model'][r])
        agg_b = np.max(SB, axis=1)
        agg_c = (1.0 - np.prod(1.0 - SC, axis=1)) if candidate == 'noisy_or' else np.max(SC, axis=1)
        cls_b, cls_c = PB.any(axis=1).astype(int), PC.any(axis=1).astype(int)
        raw_b = ANY_BLEND * R['any_model'][r] + (1 - ANY_BLEND) * agg_b
        raw_c = ANY_BLEND * R['any_model'][r] + (1 - ANY_BLEND) * agg_c
        qp_b, qp_c = consistent_qp(raw_b, cls_b), consistent_qp(raw_c, cls_c)
        QB[r, m], QC[r, m] = qp_b[m], qp_c[m]
        ya = y_any[m]
        row = {'region': region, 'repeat': r, 'n_scored': int(m.sum()), 'n_pos_any': int(ya.sum()),
               'auc_any_model': safe_auc(ya, R['any_model'][r][m]),
               'auc_agg_base': safe_auc(ya, agg_b[m]), 'auc_agg_cand': safe_auc(ya, agg_c[m]),
               'auc_qp_raw_base': safe_auc(ya, raw_b[m]), 'auc_qp_raw_cand': safe_auc(ya, raw_c[m]),
               'auc_qp_base': safe_auc(ya, qp_b[m]), 'auc_qp_cand': safe_auc(ya, qp_c[m]),
               'f1_class_base': f1_score(ya, cls_b[m], zero_division=0), 'f1_class_cand': f1_score(ya, cls_c[m], zero_division=0),
               'macro_f1_class_base': macro_f1(ya, cls_b[m]), 'macro_f1_class_cand': macro_f1(ya, cls_c[m]),
               'n_class_differs': int((cls_b[m] != cls_c[m]).sum())}
        row['delta_auc_qp'] = row['auc_qp_cand'] - row['auc_qp_base']
        row['delta_auc_qp_raw'] = row['auc_qp_raw_cand'] - row['auc_qp_raw_base']
        reg_rows.append(row)
        recs.append(pd.DataFrame({'repeat': r, 'row_id': np.nonzero(m)[0], 'y': ya, 'group': groups[m], 'study': studies[m],
                                  'fold': R['fold'][r][m], 'score_base': qp_b[m], 'score_cand': qp_c[m],
                                  'threshold_base': 0.5, 'threshold_cand': 0.5, 'pred_base': cls_b[m], 'pred_cand': cls_c[m],
                                  'qp_raw_base': raw_b[m], 'qp_raw_cand': raw_c[m], 'any_model': R['any_model'][r][m],
                                  'crit_agg_base': agg_b[m], 'crit_agg_cand': agg_c[m]}))
    pd.concat(recs, ignore_index=True).to_csv(out / f'region_{region}_scores.csv', index=False, float_format='%.10g')
    rep_reg = pd.DataFrame(reg_rows)
    qbm, qcm = np.nanmean(QB, axis=0), np.nanmean(QC, axis=0)
    n_gain = int((rep_reg['delta_auc_qp'] >= GAIN_MIN).sum())
    dec_reg = {'region': region, 'candidate': candidate, 'level': 'quality_prob', 'n': n, 'n_pos_any': int(y_any.sum()),
               'n_repeats': n_repeats,
               'auc_any_model_mean': float(rep_reg['auc_any_model'].mean()),
               'auc_agg_base_mean': float(rep_reg['auc_agg_base'].mean()), 'auc_agg_cand_mean': float(rep_reg['auc_agg_cand'].mean()),
               'auc_qp_raw_base_mean': float(rep_reg['auc_qp_raw_base'].mean()), 'auc_qp_raw_cand_mean': float(rep_reg['auc_qp_raw_cand'].mean()),
               'auc_qp_base_mean': float(rep_reg['auc_qp_base'].mean()), 'auc_qp_cand_mean': float(rep_reg['auc_qp_cand'].mean()),
               'mean_delta_auc_qp': float(rep_reg['delta_auc_qp'].mean()), 'mean_delta_auc_qp_raw': float(rep_reg['delta_auc_qp_raw'].mean()),
               'n_repeats_gain_ge_0.03': n_gain, 'share_repeats_gain_ge_0.03': n_gain / n_repeats,
               'f1_class_base_mean': float(rep_reg['f1_class_base'].mean()), 'f1_class_cand_mean': float(rep_reg['f1_class_cand'].mean()),
               'macro_f1_class_base_mean': float(rep_reg['macro_f1_class_base'].mean()), 'macro_f1_class_cand_mean': float(rep_reg['macro_f1_class_cand'].mean()),
               'n_class_differs_total': int(rep_reg['n_class_differs'].sum()),
               'auc_qp_base_ci': boot_ci(studies, lambda idx: safe_auc(y_any[idx], qbm[idx])),
               'auc_qp_cand_ci': boot_ci(studies, lambda idx: safe_auc(y_any[idx], qcm[idx])),
               'delta_auc_qp_pooled_ci': boot_ci(studies, lambda idx: safe_auc(y_any[idx], qcm[idx]) - safe_auc(y_any[idx], qbm[idx]))}
    dec_reg['accepted_old_rule'] = bool(n_gain >= np.ceil(GAIN_SHARE * n_repeats) - 1e-9
                                        and dec_reg['macro_f1_class_cand_mean'] >= dec_reg['macro_f1_class_base_mean'] - 1e-12)
    log(f"  --> [{region}] quality_prob: AUC база {dec_reg['auc_qp_base_mean']:.3f} кандидат {dec_reg['auc_qp_cand_mean']:.3f} "
        f"Δ {dec_reg['mean_delta_auc_qp']:+.3f} (raw Δ {dec_reg['mean_delta_auc_qp_raw']:+.3f}), ≥0.03 в {n_gain}/{n_repeats}; "
        f"F1 класса {dec_reg['f1_class_base_mean']:.3f} -> {dec_reg['f1_class_cand_mean']:.3f}; строк с иным классом: {dec_reg['n_class_differs_total']}")
    return pd.concat(per_repeat, ignore_index=True), pd.DataFrame(fold_rows), decisions, rep_reg, dec_reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', required=True, choices=CANDIDATES)
    ap.add_argument('--repeats', type=int, default=int(os.environ.get('N_REPEATS', 20)))
    ap.add_argument('--regions', nargs='*', default=list(REGION_CRITERIA))
    ap.add_argument('--out', default=None)
    ap.add_argument('--hashes', default=os.environ.get('PIXEL_HASHES', str(ROOT / 'docs' / 'k5' / 'pixel_hashes.csv')))
    a = ap.parse_args()
    out = Path(a.out) if a.out else ROOT / 'outputs' / 'backlog_gate' / a.candidate
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True); lines.append(s)

    t0 = time.time()
    log(f"кандидат {a.candidate}, повторов {a.repeats}, ROOT={ROOT}, hashes={a.hashes}, out={out}")
    reps, folds, decs, rreps, rdecs = [], [], [], [], []
    for region in a.regions:
        rep, fold, dec, rrep, rdec = run_region(region, REGION_CRITERIA[region], a.candidate, a.repeats, a.hashes, out, log)
        reps.append(rep); folds.append(fold); decs += dec; rreps.append(rrep); rdecs.append(rdec)
    pd.concat(reps, ignore_index=True).to_csv(out / 'per_repeat.csv', index=False)
    pd.concat(folds, ignore_index=True).to_csv(out / 'per_fold.csv', index=False)
    pd.concat(rreps, ignore_index=True).to_csv(out / 'per_repeat_region.csv', index=False)
    summary = {'candidate': a.candidate, 'n_repeats': a.repeats, 'criteria': decs, 'region_quality_prob': rdecs,
               'protocol': {'outer': f'GroupKFold(shuffle=True, random_state=42+r) {N_OUTER} x {a.repeats}',
                            'inner': f'GroupKFold(shuffle=True, random_state=1000*r+k) {N_INNER}',
                            'groups': 'connected components (study, pixel_hash)',
                            'base': 'config.yaml 2.3.2: w_geom по критерию (0.5), thresholds_rule, preprocessing.variant_by_criterion, embeddings.source_by_criterion',
                            'old_rule': f'dAUC>={GAIN_MIN} in >={int(GAIN_SHARE * 100)}% repeats and mean macro-F1 not worse',
                            'tz_weights': TZ_WEIGHTS, 'router_w': ROUTER_W},
               'elapsed_s': round(time.time() - t0, 1)}
    with open(out / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    (out / 'log.txt').write_text('\n'.join(lines), encoding='utf-8')
    log(f"готово за {time.time() - t0:.0f} с; результаты в {out}")


if __name__ == '__main__':
    main()
