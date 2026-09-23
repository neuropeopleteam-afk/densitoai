"""
К2: вентильный стэкинг по критерию, выбор веса w_geom ∈ {0, .25, .5, .75, 1} ТОЛЬКО внутри nested CV.

Протокол:
  внешний контур: GroupKFold(shuffle=True, random_state=42+r) 5 фолдов × N_REPEATS повторов.
                  ВНИМАНИЕ: перестановка строк, как в train_stacked.py, НЕ меняет разбиение GroupKFold
                  (группы раскладываются по размеру), поэтому здесь настоящий шаффл групп (sklearn>=1.6);
                  группы = компоненты связности (study, pixel_hash),
                  т.е. дубликаты снимков никогда не расходятся по фолдам;
  внутренний контур: GroupKFold 3 фолда на внешнем train -> inner-OOF geom/emb;
  модели ровно как в train_stacked.py: geom = StandardScaler -> LogReg(C=1, balanced);
                                       emb  = StandardScaler -> PCA(32) -> LogReg(C=0.1, balanced);
  выбор w по AUC стэка на inner-OOF (ранги pct внутри inner-OOF), порог — тоже на inner-OOF стэке
  (F1-опт при >=15 позитивов в train, иначе prevalence — правило train_stacked);
  внешний test: модели, обученные на всём внешнем train; ранги относительно inner-OOF-референса
  (inference.percentile_rank: доля референса <= score); стэк = w*rank_g + (1-w)*rank_e.
  Сравнение: база w=0.5 против вентиля. Per-repeat скоры/предсказания экспортируются в
  results/gate_input_<region>_<crit>.csv для калиброванного гейта tools/paired_gate.py (идея 11). Метрики по повторам, парный прирост, macro-F1,
  бутстрап-ДИ по исследованиям для усреднённого по повторам outer-OOF скора.
Часть 2 (иерархия any -> типы): quality_prob региона = wb*any_model + (1-wb)*max(crit),
  any_model = среднее вероятностей geom/emb any-моделей (как inference.any_violation_prob);
  вариант «вентиль»: критерии с вентилем + wb и стэкинг any-модели по рангам с выбором внутри nested.
"""
import os, sys, json, time, warnings
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_gate import write_gate_input  # noqa: E402  (идея 11: экспорт per-repeat скоров)
from train_stacked import (DATA_DIR, REGION_CRITERIA, CRITERION_GEOMETRY_COLS, CRITERION_LABEL_COL,  # noqa: E402
                           REGION_ROWS, EMB_SOURCE_BY_CRITERION, load_embeddings_by_source, emb_source_for,
                           PCA_COMPONENTS, f1_optimal_threshold, prevalence_threshold)

WORK = Path(os.environ.get('NESTED_GATE_WORK', ROOT / 'outputs' / 'nested_gate'))
RES = WORK / 'results'
RES.mkdir(parents=True, exist_ok=True)
W_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
W_BASE = 0.5
N_OUTER, N_INNER, N_REPEATS = 5, 3, int(os.environ.get('N_REPEATS', 10))
GEOM_C, EMB_C = 1.0, 0.1
MIN_POS_F1 = 15
GAIN_MIN, GAIN_REPEATS = 0.03, 7
N_BOOT = 1000


# ----------------------------------------------------------------------------- модели
def fit_geom(X, y):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, C=GEOM_C, class_weight='balanced').fit(sc.transform(X), y)
    return lambda Z: clf.predict_proba(sc.transform(Z))[:, 1]


def fit_emb(E, y):
    sc = StandardScaler().fit(E)
    n_comp = min(PCA_COMPONENTS, len(y) - 1, E.shape[1])
    pca = PCA(n_components=n_comp, random_state=42).fit(sc.transform(E))
    clf = LogisticRegression(max_iter=1000, C=EMB_C, class_weight='balanced').fit(pca.transform(sc.transform(E)), y)
    return lambda Z: clf.predict_proba(pca.transform(sc.transform(Z)))[:, 1]


def degenerate(y):
    return y.sum() < 2 or y.sum() == len(y)


def inner_oof(X, E, y, groups, seed):
    """inner-OOF предсказания geom/emb на train-выборке (GroupKFold N_INNER)."""
    og, oe = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for tr, va in GroupKFold(n_splits=N_INNER, shuffle=True, random_state=seed).split(X, groups=groups):
        if degenerate(y[tr]):
            continue
        og[va] = fit_geom(X[tr], y[tr])(X[va])
        oe[va] = fit_emb(E[tr], y[tr])(E[va])
    return og, oe


def pct_rank(v):
    return pd.Series(v).rank(pct=True).values


def ref_rank(scores, ref):
    """inference.ModelRegistry.percentile_rank: доля референса <= score."""
    ref = np.sort(ref[~np.isnan(ref)])
    return np.searchsorted(ref, scores, side='right') / len(ref)


def stack(rg, re_, w):
    return w * rg + (1.0 - w) * re_


def choose_threshold(y, s, n_pos_rule):
    if len(np.unique(y)) < 2:
        return float(np.quantile(s, 0.9)), 'degenerate_q90'
    if n_pos_rule >= MIN_POS_F1:
        t = f1_optimal_threshold(y, s)
        return float(t), 'f1_optimal'
    return float(prevalence_threshold(y, s)), 'prevalence'


def safe_auc(y, s):
    m = ~np.isnan(s)
    if m.sum() == 0 or len(np.unique(y[m])) < 2:
        return np.nan
    return roc_auc_score(y[m], s[m])


def macro_f1(y, p):
    return f1_score(y, p, average='macro', zero_division=0)


def boot_ci(studies, fn, n_boot=N_BOOT, seed=0):
    """Бутстрап по исследованиям: fn(idx) -> метрика. Возвращает (lo, mean, hi)."""
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


# ----------------------------------------------------------------------------- данные
def load_region(region):
    geom = pd.read_csv(DATA_DIR / 'geometry_features.csv')
    rows = REGION_ROWS[region]
    geom = geom[geom['region'].isin(rows)].reset_index(drop=True)
    lab = pd.read_csv(DATA_DIR / 'labels_for_embeddings.csv')
    emb_by_source = load_embeddings_by_source()
    eidx = np.nonzero(lab['region'].isin(rows).values)[0]
    assert (lab.loc[eidx, 'file_path'].values == geom['file_path'].values).all()
    E = {k: v[eidx] for k, v in emb_by_source.items()}
    hashes = pd.read_csv(RES / 'pixel_hashes.csv')
    hmap = dict(zip(hashes['file_path'], hashes['pixel_hash']))
    geom['pixel_hash'] = geom['file_path'].map(hmap)
    geom['group'] = connected_groups(geom['study'].values, geom['pixel_hash'].values)
    return geom, E


def connected_groups(studies, hashes):
    """Компоненты связности графа study—pixel_hash (union-find). Дубликаты одного снимка
    в разных исследованиях склеивают исследования в одну группу."""
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


# ----------------------------------------------------------------------------- часть 1: по критериям
def run_criterion(region, crit, geom, E, log):
    label_col = CRITERION_LABEL_COL.get(crit, crit)
    y_all = geom[label_col].values
    valid = ~pd.isna(y_all)
    y = y_all[valid].astype(int)
    cols = CRITERION_GEOMETRY_COLS[crit]
    Xraw = geom[cols].values.astype(np.float64)[valid]
    src = emb_source_for(crit, E)
    Emb = E[src][valid]
    groups = geom['group'].values[valid]
    studies = geom['study'].values[valid]
    files = geom['file_path'].values[valid]
    n = len(y)
    log(f"\n=== {region}/{crit}: n={n}, n_pos={int(y.sum())}, emb='{src}', geom={cols}")

    per_repeat, fold_rows = [], []
    score_base = np.full((N_REPEATS, n), np.nan)
    score_gate = np.full((N_REPEATS, n), np.nan)
    pred_base = np.full((N_REPEATS, n), np.nan)
    pred_gate = np.full((N_REPEATS, n), np.nan)
    fold_id = np.full((N_REPEATS, n), np.nan)          # идея 11: номер внешнего фолда для экспорта в paired_gate
    # для части 2: сохраняем внешние предсказания geom/emb и inner-OOF референсы по фолдам
    fold_cache = {}
    n_degenerate_inner = [0]

    for r in range(N_REPEATS):
        for k, (tr, te) in enumerate(GroupKFold(n_splits=N_OUTER, shuffle=True, random_state=42 + r).split(Xraw, groups=groups)):
            if degenerate(y[tr]):
                continue
            med = np.nanmedian(Xraw[tr], axis=0)            # медианы только по внешнему train
            X = impute(Xraw, med)
            og, oe = inner_oof(X[tr], Emb[tr], y[tr], groups[tr], seed=1000 * r + k)
            ok = ~np.isnan(og) & ~np.isnan(oe)
            rg_in, re_in = np.full(len(tr), np.nan), np.full(len(tr), np.nan)
            rg_in[ok], re_in[ok] = pct_rank(og[ok]), pct_rank(oe[ok])
            inner_auc = {w: safe_auc(y[tr][ok], stack(rg_in[ok], re_in[ok], w)) for w in W_GRID}
            if ok.sum() < 2 or len(np.unique(y[tr][ok])) < 2 or np.all(np.isnan(list(inner_auc.values()))):
                # inner-OOF выродился (все inner-train без позитивов) — вентиль не выбираем, фолд как база
                inner_auc = {w: np.nan for w in W_GRID}
                w_sel = W_BASE
                n_degenerate_inner[0] += 1
            else:
                best = np.nanmax(list(inner_auc.values()))
                # выбор w: максимум inner-AUC; при равенстве — ближайший к 0.5 (консервативно)
                w_sel = sorted([w for w, a in inner_auc.items() if a >= best - 1e-12], key=lambda w: abs(w - 0.5))[0]
            if ok.sum() == 0:
                continue
            n_pos_tr = int(y[tr].sum())
            thr = {}
            for w in (W_BASE, w_sel):
                thr[w] = choose_threshold(y[tr][ok], stack(rg_in[ok], re_in[ok], w), n_pos_tr)
            # внешние модели на всём train
            pg = fit_geom(X[tr], y[tr])(X[te])
            pe = fit_emb(Emb[tr], y[tr])(Emb[te])
            rg_te, re_te = ref_rank(pg, og), ref_rank(pe, oe)
            sb, sg = stack(rg_te, re_te, W_BASE), stack(rg_te, re_te, w_sel)
            score_base[r, te], score_gate[r, te] = sb, sg
            fold_id[r, te] = k
            pred_base[r, te] = (sb >= thr[W_BASE][0]).astype(int)
            pred_gate[r, te] = (sg >= thr[w_sel][0]).astype(int)
            fold_rows.append({'region': region, 'criterion': crit, 'repeat': r, 'fold': k, 'n_train': len(tr),
                              'n_pos_train': n_pos_tr, 'n_test': len(te), 'n_pos_test': int(y[te].sum()),
                              'w_selected': w_sel, **{f'inner_auc_w{w}': inner_auc[w] for w in W_GRID},
                              'thr_base': thr[W_BASE][0], 'thr_gate': thr[w_sel][0], 'thr_rule': thr[w_sel][1],
                              'outer_auc_geom': safe_auc(y[te], pg), 'outer_auc_emb': safe_auc(y[te], pe),
                              'outer_auc_base': safe_auc(y[te], sb), 'outer_auc_gate': safe_auc(y[te], sg)})
            fold_cache[(r, k)] = dict(te=te, tr=tr, pg=pg, pe=pe, og=og, oe=oe, w_sel=w_sel,
                                      thr_base=thr[W_BASE][0], thr_gate=thr[w_sel][0])
        m = ~np.isnan(score_base[r])
        yb = y[m]
        row = {'region': region, 'criterion': crit, 'repeat': r, 'n_scored': int(m.sum()),
               'auc_base': safe_auc(yb, score_base[r][m]), 'auc_gate': safe_auc(yb, score_gate[r][m]),
               'f1pos_base': f1_score(yb, pred_base[r][m], zero_division=0),
               'f1pos_gate': f1_score(yb, pred_gate[r][m], zero_division=0),
               'macro_f1_base': macro_f1(yb, pred_base[r][m]), 'macro_f1_gate': macro_f1(yb, pred_gate[r][m]),
               'w_selected_folds': ' '.join(str(fr['w_selected']) for fr in fold_rows
                                            if fr['criterion'] == crit and fr['repeat'] == r)}
        row['delta_auc'] = row['auc_gate'] - row['auc_base']
        row['delta_macro_f1'] = row['macro_f1_gate'] - row['macro_f1_base']
        per_repeat.append(row)
        log(f"  repeat {r}: AUC base={row['auc_base']:.3f} gate={row['auc_gate']:.3f} "
            f"d={row['delta_auc']:+.3f} | macroF1 base={row['macro_f1_base']:.3f} gate={row['macro_f1_gate']:.3f} "
            f"| w: {row['w_selected_folds']}")

    rep = pd.DataFrame(per_repeat)
    folds = pd.DataFrame(fold_rows)
    # усреднённый по повторам outer-OOF скор и голосование по предсказаниям
    sb_mean, sg_mean = np.nanmean(score_base, axis=0), np.nanmean(score_gate, axis=0)
    pb_vote, pg_vote = (np.nanmean(pred_base, axis=0) >= 0.5).astype(int), (np.nanmean(pred_gate, axis=0) >= 0.5).astype(int)
    ci = {}
    for name, s, p in (('base', sb_mean, pb_vote), ('gate', sg_mean, pg_vote)):
        ci[f'auc_{name}_mean_over_repeats'] = float(rep[f'auc_{name}'].mean())
        ci[f'auc_{name}_pooled'] = float(safe_auc(y, s))
        ci[f'auc_{name}_ci'] = boot_ci(studies, lambda idx, s=s: safe_auc(y[idx], s[idx]))
        ci[f'macro_f1_{name}_pooled'] = float(macro_f1(y, p))
        ci[f'macro_f1_{name}_ci'] = boot_ci(studies, lambda idx, p=p: macro_f1(y[idx], p[idx]))
        ci[f'f1pos_{name}_pooled'] = float(f1_score(y, p, zero_division=0))
    # парный ДИ прироста AUC на усреднённом скоре
    ci['delta_auc_pooled_ci'] = boot_ci(studies, lambda idx: safe_auc(y[idx], sg_mean[idx]) - safe_auc(y[idx], sb_mean[idx]))
    w_counts = folds['w_selected'].value_counts().reindex(W_GRID, fill_value=0)
    n_gain = int((rep['delta_auc'] >= GAIN_MIN).sum())
    accepted = bool(n_gain >= GAIN_REPEATS and rep['macro_f1_gate'].mean() >= rep['macro_f1_base'].mean() - 1e-12)
    decision = {'region': region, 'criterion': crit, 'n': n, 'n_pos': int(y.sum()), 'emb_source': src,
                'w_mode': float(w_counts.idxmax()), **{f'freq_w{w}': int(w_counts[w]) for w in W_GRID},
                'n_repeats_gain_ge_0.03': n_gain, 'n_repeats': N_REPEATS, 'n_outer_folds_degenerate_inner': n_degenerate_inner[0],
                'mean_delta_auc': float(rep['delta_auc'].mean()), 'min_delta_auc': float(rep['delta_auc'].min()),
                'mean_delta_macro_f1': float(rep['delta_macro_f1'].mean()), 'accepted': accepted, **ci}
    log(f"  --> w mode={decision['w_mode']} freq={ {k: int(v) for k, v in w_counts.items()} } gain>=0.03 in {n_gain}/{N_REPEATS}, "
        f"mean dAUC={decision['mean_delta_auc']:+.3f}, d macroF1={decision['mean_delta_macro_f1']:+.3f} "
        f"=> {'ПРИНЯТ' if accepted else 'не принят'}")
    oof = pd.DataFrame({'study': studies, 'file_path': files, 'y_true': y, 'score_base_mean': sb_mean,
                        'score_gate_mean': sg_mean, 'pred_base_vote': pb_vote, 'pred_gate_vote': pg_vote})
    oof.to_csv(RES / f'nested_oof_{region}_{crit}.csv', index=False)
    # идея 11: per-repeat вход для калиброванного гейта (tools/paired_gate.py)
    write_gate_input(RES / f'gate_input_{region}_{crit}.csv', y, groups, score_base, score_gate,
                     pred_base=pred_base, pred_cand=pred_gate, fold=fold_id, study=studies, file_path=files)
    return rep, folds, decision, dict(y=y, valid=valid, cache=fold_cache, studies=studies)


# ----------------------------------------------------------------------------- часть 2: any -> типы
def run_any(region, criteria, geom, E, crit_state, log):
    """quality_prob региона: база = 0.5*mean(p_any_geom, p_any_emb) + 0.5*max(crit stack w=0.5);
    вентиль = wb*any_rank_stack(wa) + (1-wb)*max(crit stack w_sel), wa/wb выбраны по inner-OOF AUC.
    Строки региона: все, у кого есть хоть одна метка критерия (any_seen, как в train_final_models)."""
    labels = np.column_stack([geom[CRITERION_LABEL_COL.get(c, c)].values.astype(float) for c in criteria])
    seen = ~np.isnan(labels).all(axis=1)
    # для простоты используем строки, где все метки критериев валидны (иначе max по критериям неполный)
    full = ~np.isnan(labels).any(axis=1)
    if (seen & ~full).sum():
        log(f"  [any] строк с частично отсутствующими метками: {(seen & ~full).sum()} — исключены из части 2")
    idx_all = np.nonzero(full)[0]
    y = (np.nanmax(labels[full], axis=1) > 0).astype(int)
    cols_any = sorted({c for cr in criteria for c in CRITERION_GEOMETRY_COLS[cr]})
    Xraw = geom[cols_any].values.astype(np.float64)[full]
    Emb = E['imagenet'][full]
    groups, studies = geom['group'].values[full], geom['study'].values[full]
    n = len(y)
    log(f"\n=== {region}/any: n={n}, n_pos={int(y.sum())}, geom={cols_any}")
    # индексы критериев: crit_state[crit]['valid'] — маска по строкам региона; нужна карта строка->позиция
    pos_in_crit = {}
    for c in criteria:
        vpos = np.full(len(geom), -1); vpos[np.nonzero(crit_state[c]['valid'])[0]] = np.arange(crit_state[c]['valid'].sum())
        pos_in_crit[c] = vpos[idx_all]
        assert (pos_in_crit[c] >= 0).all()

    WB_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
    per_repeat, fold_rows = [], []
    sc_base, sc_gate, sc_any_only, sc_max_base = (np.full((N_REPEATS, n), np.nan) for _ in range(4))
    for r in range(N_REPEATS):
        # те же внешние разбиения, что и у критериев, невозможно гарантировать (разные valid-маски),
        # поэтому any-иерархия использует свои разбиения по тем же группам с тем же сидом
        for k, (tr, te) in enumerate(GroupKFold(n_splits=N_OUTER, shuffle=True, random_state=42 + r).split(Xraw, groups=groups)):
            med = np.nanmedian(Xraw[tr], axis=0)
            X = impute(Xraw, med)
            # inner-OOF any-моделей
            og, oe = inner_oof(X[tr], Emb[tr], y[tr], groups[tr], seed=1000 * r + k + 7)
            ok = ~np.isnan(og) & ~np.isnan(oe)
            # inner-OOF критериев на тех же inner-разбиениях: считаем заново (дёшево)
            crit_in_base, crit_in_gate, crit_te_base, crit_te_gate = [], [], [], []
            for c in criteria:
                yc = crit_state[c]['y'][pos_in_crit[c]]
                colsc = CRITERION_GEOMETRY_COLS[c]
                Xc = impute(geom[colsc].values.astype(np.float64)[full], np.nanmedian(geom[colsc].values.astype(np.float64)[full][tr], axis=0))
                Ec = E[emb_source_for(c, E)][full]
                ogc, oec = inner_oof(Xc[tr], Ec[tr], yc[tr], groups[tr], seed=1000 * r + k)
                okc = ~np.isnan(ogc) & ~np.isnan(oec)
                rg, re_ = np.full(len(tr), np.nan), np.full(len(tr), np.nan)
                rg[okc], re_[okc] = pct_rank(ogc[okc]), pct_rank(oec[okc])
                aucs = {w: safe_auc(yc[tr][okc], stack(rg[okc], re_[okc], w)) for w in W_GRID}
                if okc.sum() < 2 or np.all(np.isnan(list(aucs.values()))):
                    w_sel = W_BASE
                else:
                    best = np.nanmax(list(aucs.values()))
                    w_sel = sorted([w for w, a in aucs.items() if a >= best - 1e-12], key=lambda w: abs(w - 0.5))[0]
                crit_in_base.append(stack(rg, re_, W_BASE)); crit_in_gate.append(stack(rg, re_, w_sel))
                pgc = fit_geom(Xc[tr], yc[tr])(Xc[te]); pec = fit_emb(Ec[tr], yc[tr])(Ec[te])
                rgt, ret = ref_rank(pgc, ogc), ref_rank(pec, oec)
                crit_te_base.append(stack(rgt, ret, W_BASE)); crit_te_gate.append(stack(rgt, ret, w_sel))
            max_in_base = np.nanmax(np.column_stack(crit_in_base), axis=1)
            max_in_gate = np.nanmax(np.column_stack(crit_in_gate), axis=1)
            max_te_base = np.nanmax(np.column_stack(crit_te_base), axis=1)
            max_te_gate = np.nanmax(np.column_stack(crit_te_gate), axis=1)
            # any-модель: база — среднее вероятностей (как inference); вентиль — ранговый стэк с wa
            any_in_base = 0.5 * og + 0.5 * oe
            rg_in, re_in = np.full(len(tr), np.nan), np.full(len(tr), np.nan)
            rg_in[ok], re_in[ok] = pct_rank(og[ok]), pct_rank(oe[ok])
            okk = ok & ~np.isnan(max_in_gate)
            if okk.sum() < 2 or len(np.unique(y[tr][okk])) < 2:
                okk[:] = False
            best, sel = -1, (0.5, 0.5)
            for wa in W_GRID:
                any_in = stack(rg_in, re_in, wa)
                for wb in WB_GRID:
                    a = safe_auc(y[tr][okk], (wb * any_in + (1 - wb) * max_in_gate)[okk]) if okk.any() else np.nan
                    if np.isnan(a):
                        continue
                    if a > best + 1e-12 or (abs(a - best) <= 1e-12 and abs(wa - .5) + abs(wb - .5) < abs(sel[0] - .5) + abs(sel[1] - .5)):
                        best, sel = a, (wa, wb)
            wa, wb = sel
            pg, pe = fit_geom(X[tr], y[tr])(X[te]), fit_emb(Emb[tr], y[tr])(Emb[te])
            any_te_base = 0.5 * pg + 0.5 * pe
            any_te_gate = stack(ref_rank(pg, og), ref_rank(pe, oe), wa)
            sc_base[r, te] = 0.5 * any_te_base + 0.5 * max_te_base
            sc_gate[r, te] = wb * any_te_gate + (1 - wb) * max_te_gate
            sc_any_only[r, te] = any_te_base
            sc_max_base[r, te] = max_te_base
            fold_rows.append({'region': region, 'repeat': r, 'fold': k, 'wa_selected': wa, 'wb_selected': wb,
                              'inner_auc_best': best,
                              'outer_auc_base': safe_auc(y[te], sc_base[r, te]), 'outer_auc_gate': safe_auc(y[te], sc_gate[r, te])})
        row = {'region': region, 'repeat': r, 'auc_base': safe_auc(y, sc_base[r]), 'auc_gate': safe_auc(y, sc_gate[r]),
               'auc_any_model_only': safe_auc(y, sc_any_only[r]), 'auc_max_crit_base': safe_auc(y, sc_max_base[r]),
               'wa_wb_folds': ' '.join(f"{fr['wa_selected']}/{fr['wb_selected']}" for fr in fold_rows if fr['repeat'] == r)}
        row['delta_auc'] = row['auc_gate'] - row['auc_base']
        per_repeat.append(row)
        log(f"  [any] repeat {r}: AUC base={row['auc_base']:.3f} gate={row['auc_gate']:.3f} d={row['delta_auc']:+.3f} "
            f"(any-only {row['auc_any_model_only']:.3f}, max-crit {row['auc_max_crit_base']:.3f}) | wa/wb: {row['wa_wb_folds']}")
    rep, folds = pd.DataFrame(per_repeat), pd.DataFrame(fold_rows)
    sbm, sgm = np.nanmean(sc_base, axis=0), np.nanmean(sc_gate, axis=0)
    n_gain = int((rep['delta_auc'] >= GAIN_MIN).sum())
    dec = {'region': region, 'n': n, 'n_pos': int(y.sum()), 'n_repeats_gain_ge_0.03': n_gain,
           'mean_delta_auc': float(rep['delta_auc'].mean()),
           'auc_base_mean_over_repeats': float(rep['auc_base'].mean()), 'auc_gate_mean_over_repeats': float(rep['auc_gate'].mean()),
           'auc_base_ci': boot_ci(studies, lambda idx: safe_auc(y[idx], sbm[idx])),
           'auc_gate_ci': boot_ci(studies, lambda idx: safe_auc(y[idx], sgm[idx])),
           'delta_auc_pooled_ci': boot_ci(studies, lambda idx: safe_auc(y[idx], sgm[idx]) - safe_auc(y[idx], sbm[idx])),
           'wa_mode': float(folds['wa_selected'].mode()[0]), 'wb_mode': float(folds['wb_selected'].mode()[0]),
           'accepted': bool(n_gain >= GAIN_REPEATS)}
    log(f"  [any] --> gain>=0.03 in {n_gain}/{N_REPEATS}, mean dAUC={dec['mean_delta_auc']:+.3f}, "
        f"wa mode={dec['wa_mode']} wb mode={dec['wb_mode']} => {'ПРИНЯТ' if dec['accepted'] else 'не принят'}")
    return rep, folds, dec


# ----------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    log_lines = []

    def log(s):
        print(s, flush=True); log_lines.append(s)

    hashes = pd.read_csv(RES / 'pixel_hashes.csv')
    vc = hashes['pixel_hash'].value_counts()
    log(f"pixel_hash: файлов {len(hashes)}, уникальных {hashes['pixel_hash'].nunique()}, "
        f"групп дубликатов {(vc > 1).sum()}, файлов в них {int(vc[vc > 1].sum())}, "
        f"групп с >1 исследованием {(hashes.groupby('pixel_hash')['study'].nunique() > 1).sum()}")
    all_rep, all_folds, decisions, any_rep, any_folds, any_dec = [], [], [], [], [], []
    for region, criteria in REGION_CRITERIA.items():
        geom, E = load_region(region)
        log(f"[{region}] групп (study+pixel_hash): {geom['group'].nunique()} при {geom['study'].nunique()} исследованиях")
        state = {}
        for crit in criteria:
            rep, folds, dec, st = run_criterion(region, crit, geom, E, log)
            all_rep.append(rep); all_folds.append(folds); decisions.append(dec); state[crit] = st
        if os.environ.get('SKIP_ANY', '0') != '1':
            rep, folds, dec = run_any(region, criteria, geom, E, state, log)
            any_rep.append(rep); any_folds.append(folds); any_dec.append(dec)
        log(f"[{region}] готово за {time.time() - t0:.0f} с")

    rep = pd.concat(all_rep, ignore_index=True); folds = pd.concat(all_folds, ignore_index=True)
    dec = pd.DataFrame(decisions)
    rep.to_csv(RES / 'nested_gate_per_repeat.csv', index=False)
    folds.to_csv(RES / 'nested_gate_per_fold.csv', index=False)
    dec.to_csv(RES / 'nested_gate_decisions.csv', index=False)
    if any_rep:
        pd.concat(any_rep, ignore_index=True).to_csv(RES / 'nested_any_per_repeat.csv', index=False)
        pd.concat(any_folds, ignore_index=True).to_csv(RES / 'nested_any_per_fold.csv', index=False)
        pd.DataFrame(any_dec).to_csv(RES / 'nested_any_decisions.csv', index=False)
    with open(RES / 'nested_gate_decisions.json', 'w', encoding='utf-8') as f:
        json.dump({'criteria': decisions, 'any': any_dec, 'protocol': {
            'outer': f'GroupKFold(shuffle=True, random_state=42+r) {N_OUTER} x {N_REPEATS} repeats',
            'inner': f'GroupKFold(shuffle=True, random_state=1000*r+k) {N_INNER}',
            'groups': 'connected components (study, pixel_hash)', 'w_grid': W_GRID,
            'acceptance': f'dAUC>={GAIN_MIN} in >={GAIN_REPEATS}/{N_REPEATS} repeats and mean macro-F1 not worse'}},
            f, indent=2, ensure_ascii=False)
    (RES / 'nested_gate_log.txt').write_text('\n'.join(log_lines), encoding='utf-8')
    log(f"\nвсего {time.time() - t0:.0f} с; результаты в {RES}")


if __name__ == '__main__':
    main()
