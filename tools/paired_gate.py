"""
Калиброванный гейт приёмки (идея 11): парный кластерный тест «кандидат против базы»
по внешним OOF-скорам nested repeated GroupKFold. Инструмент не зависит от конкретного
кандидата — он принимает уже посчитанные скоры и решает, принят ли кандидат.

Формат входа (пишется харнессом на базе tools/nested_gate.py; см. write_gate_input()):

  CSV (длинный формат), одна строка = (повтор, объект):
    repeat      int    номер внешнего повтора (0..R-1)
    row_id      int    индекс объекта (0..n-1), одинаковый во всех повторах
    y           int    метка 0/1
    group       int|str кластер = компонента связности (study, pixel_hash) — единица ресэмплинга
    score_base  float  внешний OOF-скор базы в этом повторе (NaN, если фолд вырожден)
    score_cand  float  внешний OOF-скор кандидата в этом повторе на ТЕХ ЖЕ фолдах
    необязательные: fold (номер внешнего фолда), study, file_path,
                    pred_base, pred_cand (0/1 — решения по порогам, выбранным харнессом на inner-OOF).
  NPZ: y (n), group (n), score_base (R×n), score_cand (R×n) [+ pred_base, pred_cand (R×n), fold (R×n),
       row_id, study].

Если pred_* нет, пороги строятся по правилу train_stacked (F1-оптимальный при >= 15 позитивов
в обучающей части, иначе prevalence): при наличии колонки fold — кросс-фит (порог для фолда k
подбирается на OOF-скорах остальных фолдов того же повтора), иначе — по всем OOF-скорам повтора
(оптимистично, помечается в результате как threshold_source='insample').

Статистика:
  * ΔAUC_r и Δmacro-F1_r парные по повторам (одни и те же объекты и фолды);
  * кластерный бутстрап по группам (ресэмпл групп с возвращением, по умолчанию 4000) для
    ΔAUC, усреднённого по повторам, и для Δmacro-F1 (голосование предсказаний по повторам);
    ДИ 90 % и 95 % (перцентильные), одностороннее p_boot = доля бутстрап-ΔAUC <= 0;
  * перестановочный тест со случайной сменой ролей база/кандидат внутри кластера (sign-flip по группам);
  * парный t-test по повторам приводится только для справки: повторы делят одни и те же объекты,
    поэтому их разброс отражает изменчивость разбиений, а не выборки, и p-значение t-теста
    систематически завышает уверенность (см. tools/paired_gate_calibration.py).

Правило принятия (уровень alpha односторонний, по умолчанию 0.05 = нижняя граница 90 % ДИ);
все условия должны выполняться одновременно:
  1) нижняя граница кластерного ДИ ΔAUC > 0 (эквивалентно p_boot < alpha_adj);
  2) p перестановочного sign-flip теста по группам < alpha_adj (точный тест при обменности групп;
     страхует перцентильный бутстрап, который при n_pos ~ 10 слегка либерален); отключается --no-require-perm;
  3) точечная оценка ΔAUC (среднее по повторам) >= minimal_effect (по умолчанию 0.02);
  4) не хуже по macro-F1: нижняя граница ДИ Δmacro-F1 (--f1-stat mean = среднее по повторам Δmacro-F1_r,
     --f1-ci 60, то есть 20-й процентиль кластерного бутстрапа) >= f1_floor (по умолчанию -0.01).
     Вариант «голосование по повторам, ДИ 90 %» (--f1-stat vote --f1-ci 90) в калибровке вдвое
     снижал мощность при том же уровне ложных принятий, поэтому не является умолчанием;
  5) поправка на множественность в партии: --family holm|bh|none и --n-hypotheses m.
     Для одного кандидата в партии из m гипотез уровень alpha_adj = alpha/m (первый шаг Холма —
     то, что должен выдержать кандидат, чтобы быть принятым в партии из m); при нескольких входах
     за один вызов применяется полная процедура Холма (step-down) или Бенджамини–Хохберга
     по p = max(p_boot, p_perm) всех кандидатов.

Вызов:
  python tools/paired_gate.py INPUT [INPUT ...] [--alpha 0.05] [--minimal-effect 0.02]
        [--f1-floor -0.01] [--f1-stat mean] [--f1-ci 60] [--n-boot 4000] [--n-perm 2000]
        [--n-hypotheses 1] [--family holm] [--threshold-source auto] [--seed 0] [--out result.json]
Старое правило («ΔAUC >= 0.03 в >= 70 % повторов, средняя macro-F1 не хуже») считается рядом для сравнения.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get('DENSITO_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / 'src'))
try:  # правило порога — ровно как в train_stacked (torch не требуется)
    from train_stacked import f1_optimal_threshold as _ts_f1_optimal_threshold, prevalence_threshold as _ts_prevalence_threshold
    THRESHOLD_SOURCE_MODULE = 'src/train_stacked.py'
except Exception:  # noqa: BLE001
    _ts_f1_optimal_threshold = _ts_prevalence_threshold = None
    THRESHOLD_SOURCE_MODULE = 'local_copy'

MIN_POS_F1 = 15          # train_stacked: F1-оптимальный порог только при >= 15 позитивов
OLD_GAIN_MIN = 0.03      # старое правило
OLD_GAIN_SHARE = 0.7
DEFAULTS = dict(alpha=0.05, minimal_effect=0.02, f1_floor=-0.01, f1_ci=60.0, f1_stat='mean', n_boot=4000, n_perm=2000,
                n_hypotheses=1, family='holm', seed=0)


# ----------------------------------------------------------------------------- пороги (train_stacked)
def prevalence_threshold(y_train, scores_train):
    """Копия train_stacked.prevalence_threshold: квантиль по доле позитивов."""
    y_train = np.asarray(y_train)
    prevalence = y_train.mean()
    if prevalence <= 0 or prevalence >= 1:
        return 0.5
    return float(np.quantile(np.asarray(scores_train), 1 - prevalence))


def f1_optimal_threshold_fast(y_true, scores):
    """F1-оптимальный порог с усреднением плато, численно равный train_stacked.f1_optimal_threshold
    (проверяется тестом), но O(n log n): F1 для порогов t = уникальные значения скоров."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind='stable')
    s_sorted, y_sorted = scores[order], y_true[order]
    # для порога t = s: предсказано положительно всё с score >= s -> берём последнюю позицию каждого значения
    uniq, last_idx = np.unique(-s_sorted, return_index=True)  # -s возрастает => позиции первых вхождений по убыванию s
    # индекс последнего вхождения значения в порядке убывания
    counts = np.diff(np.append(last_idx, len(s_sorted)))
    last = last_idx + counts - 1
    tp = np.cumsum(y_sorted)[last]
    n_pred = last + 1
    n_pos = y_true.sum()
    f1 = np.where(tp > 0, 2 * tp / (n_pred + n_pos), 0.0)
    best = f1.max()
    thr_values = -uniq[np.isclose(f1, best, rtol=0, atol=1e-12)]
    return float(np.mean(thr_values))


def choose_threshold(y_train, s_train, n_pos_rule=None):
    """Правило train_stacked/nested_gate: F1-опт при >= MIN_POS_F1 позитивов, иначе prevalence."""
    y_train = np.asarray(y_train)
    if len(np.unique(y_train)) < 2:
        return float(np.quantile(s_train, 0.9)), 'degenerate_q90'
    n_pos_rule = int(y_train.sum()) if n_pos_rule is None else int(n_pos_rule)
    if n_pos_rule >= MIN_POS_F1:
        return f1_optimal_threshold_fast(y_train, s_train), 'f1_optimal'
    return prevalence_threshold(y_train, s_train), 'prevalence'


# ----------------------------------------------------------------------------- быстрые метрики
def fast_auc(y, s):
    """AUC = статистика Манна–Уитни с учётом связей (совпадает с roc_auc_score)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=float)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fast_auc_rows(y, S):
    """AUC для каждой строки матрицы S (k × n) при общих метках y; NaN, если класс один."""
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return np.full(S.shape[0], np.nan)
    r = rankdata(S, axis=1)
    return (r[:, y].sum(axis=1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def macro_f1(y, p):
    """macro-F1 двух классов без sklearn (zero_division=0)."""
    y = np.asarray(y).astype(int); p = np.asarray(p).astype(int)
    tp = np.sum((y == 1) & (p == 1)); fp = np.sum((y == 0) & (p == 1)); fn = np.sum((y == 1) & (p == 0)); tn = np.sum((y == 0) & (p == 0))
    f1_pos = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    f1_neg = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) > 0 else 0.0
    return float((f1_pos + f1_neg) / 2)


def weighted_auc(W, y, s):
    """AUC для каждого набора весов объектов W (B × n) — кластерный бутстрап без циклов.
    Учитывает связи (0.5). NaN, если у ресэмпла нет одного из классов."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=float)
    order = np.argsort(s, kind='stable')
    s_sorted = s[order]
    starts = np.flatnonzero(np.r_[True, np.diff(s_sorted) != 0])
    inv = np.zeros(len(s), dtype=int)                       # индекс уникального значения для каждого объекта
    inv[order] = np.repeat(np.arange(len(starts)), np.diff(np.append(starts, len(s))))
    Wn_sorted = (W * (~y)[None, :])[:, order]
    neg_by_value = np.add.reduceat(Wn_sorted, starts, axis=1)      # B × n_u
    neg_below = np.cumsum(neg_by_value, axis=1) - neg_by_value
    pos_idx = np.flatnonzero(y)
    contrib = W[:, pos_idx] * (neg_below[:, inv[pos_idx]] + 0.5 * neg_by_value[:, inv[pos_idx]])
    w_pos = W[:, pos_idx].sum(axis=1)
    w_neg = (W * (~y)[None, :]).sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        auc = contrib.sum(axis=1) / (w_pos * w_neg)
    auc[(w_pos == 0) | (w_neg == 0)] = np.nan
    return auc


def weighted_macro_f1(W, y, p):
    """macro-F1 для каждого набора весов W (B × n)."""
    y = np.asarray(y).astype(float); p = np.asarray(p).astype(float)
    cells = np.stack([y * p, (1 - y) * p, y * (1 - p), (1 - y) * (1 - p)], axis=1)   # tp fp fn tn
    C = W @ cells
    tp, fp, fn, tn = C[:, 0], C[:, 1], C[:, 2], C[:, 3]
    with np.errstate(invalid='ignore', divide='ignore'):
        f1p = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f1n = np.where(2 * tn + fn + fp > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
    return (f1p + f1n) / 2


# ----------------------------------------------------------------------------- ввод-вывод
def write_gate_input(path, y, group, score_base, score_cand, pred_base=None, pred_cand=None,
                     fold=None, study=None, row_id=None, file_path=None):
    """Записать вход гейта из массивов харнесса: y (n), group (n), score_* (R × n), pred_*/fold (R × n).
    Расширение .npz — сжатый NPZ, иначе CSV длинного формата."""
    path = Path(path)
    score_base, score_cand = np.asarray(score_base, float), np.asarray(score_cand, float)
    R, n = score_base.shape
    row_id = np.arange(n) if row_id is None else np.asarray(row_id)
    if path.suffix == '.npz':
        arrays = dict(y=np.asarray(y).astype(int), group=np.asarray(group), score_base=score_base,
                      score_cand=score_cand, row_id=row_id)
        for k, v in (('pred_base', pred_base), ('pred_cand', pred_cand), ('fold', fold), ('study', study), ('file_path', file_path)):
            if v is not None:
                arrays[k] = np.asarray(v)
        np.savez_compressed(path, **arrays)
        return path
    frames = []
    for r in range(R):
        d = pd.DataFrame({'repeat': r, 'row_id': row_id, 'y': np.asarray(y).astype(int), 'group': np.asarray(group),
                          'score_base': score_base[r], 'score_cand': score_cand[r]})
        if fold is not None:
            d['fold'] = np.asarray(fold)[r]
        if study is not None:
            d['study'] = np.asarray(study)
        if file_path is not None:
            d['file_path'] = np.asarray(file_path)
        if pred_base is not None:
            d['pred_base'] = np.asarray(pred_base)[r]
        if pred_cand is not None:
            d['pred_cand'] = np.asarray(pred_cand)[r]
        frames.append(d)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path


def load_gate_input(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == '.npz':
        z = np.load(path, allow_pickle=True)
        R, n = z['score_base'].shape
        rows = []
        for r in range(R):
            d = pd.DataFrame({'repeat': r, 'row_id': z['row_id'] if 'row_id' in z else np.arange(n),
                              'y': z['y'], 'group': z['group'], 'score_base': z['score_base'][r], 'score_cand': z['score_cand'][r]})
            for k in ('pred_base', 'pred_cand', 'fold'):
                if k in z:
                    d[k] = z[k][r]
            for k in ('study', 'file_path'):
                if k in z:
                    d[k] = z[k]
            rows.append(d)
        df = pd.concat(rows, ignore_index=True)
    else:
        df = pd.read_csv(path)
    need = {'repeat', 'row_id', 'y', 'group', 'score_base', 'score_cand'}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f'во входе нет колонок {sorted(missing)}')
    return df


def to_arrays(df: pd.DataFrame):
    """Длинный формат -> словарь массивов: y (n), group_idx (n), score_* (R × n) с NaN на пропусках."""
    df = df.copy()
    row_ids = np.sort(df['row_id'].unique())
    pos = {rid: i for i, rid in enumerate(row_ids)}
    n, repeats = len(row_ids), np.sort(df['repeat'].unique())
    R = len(repeats)
    ridx = df['repeat'].map({r: i for i, r in enumerate(repeats)}).values
    cidx = df['row_id'].map(pos).values
    first = df.drop_duplicates('row_id').set_index('row_id').loc[row_ids]
    y = first['y'].values.astype(int)
    grp_codes, grp_uniques = pd.factorize(first['group'].values)
    out = dict(y=y, group=grp_codes, group_labels=np.asarray(grp_uniques), n=n, R=R,
               study=first['study'].values if 'study' in first else None)
    for k in ('score_base', 'score_cand', 'pred_base', 'pred_cand', 'fold'):
        if k in df.columns:
            M = np.full((R, n), np.nan)
            M[ridx, cidx] = df[k].values.astype(float)
            out[k] = M
    # согласованность меток/групп между повторами
    chk = df.groupby('row_id')[['y', 'group']].nunique()
    if (chk > 1).any().any():
        raise ValueError('метка или группа объекта различаются между повторами')
    return out


# ----------------------------------------------------------------------------- предсказания по порогам
def derive_predictions(A: dict, source: str = 'auto'):
    """Заполнить pred_base/pred_cand, если харнесс их не записал.
    source: 'pred' (взять из входа), 'crossfit' (порог фолда k — на остальных фолдах повтора),
    'insample' (порог на всех OOF-скорах повтора), 'auto' = pred > crossfit > insample."""
    if source == 'auto':
        source = 'pred' if 'pred_base' in A and 'pred_cand' in A else ('crossfit' if 'fold' in A else 'insample')
    if source == 'pred':
        if 'pred_base' not in A or 'pred_cand' not in A:
            raise ValueError('threshold-source=pred, но колонок pred_base/pred_cand нет')
        return A, 'pred'
    y, R, n = A['y'], A['R'], A['n']
    rules = set()
    for key in ('score_base', 'score_cand'):
        P = np.full((R, n), np.nan)
        for r in range(R):
            s = A[key][r]
            m = ~np.isnan(s)
            if source == 'crossfit' and 'fold' in A:
                folds = A['fold'][r]
                for k in np.unique(folds[m]):
                    te = m & (folds == k)
                    tr = m & (folds != k)
                    if tr.sum() == 0:
                        continue
                    n_pos_rule = int(y[tr].sum())
                    t, rule = choose_threshold(y[tr], s[tr], n_pos_rule)
                    rules.add(rule)
                    P[r, te] = (s[te] >= t).astype(float)
            else:
                t, rule = choose_threshold(y[m], s[m])
                rules.add(rule)
                P[r, m] = (s[m] >= t).astype(float)
        A['pred_base' if key == 'score_base' else 'pred_cand'] = P
    return A, f"{source}:{'+'.join(sorted(rules))}"


# ----------------------------------------------------------------------------- статистика
def per_repeat_table(A: dict) -> pd.DataFrame:
    rows = []
    for r in range(A['R']):
        m = ~np.isnan(A['score_base'][r]) & ~np.isnan(A['score_cand'][r])
        yb = A['y'][m]
        ab, ac = fast_auc(yb, A['score_base'][r][m]), fast_auc(yb, A['score_cand'][r][m])
        fb, fc = macro_f1(yb, A['pred_base'][r][m]), macro_f1(yb, A['pred_cand'][r][m])
        rows.append(dict(repeat=r, n_scored=int(m.sum()), n_pos=int(yb.sum()), auc_base=ab, auc_cand=ac, delta_auc=ac - ab,
                         macro_f1_base=fb, macro_f1_cand=fc, delta_macro_f1=fc - fb))
    return pd.DataFrame(rows)


def vote(P):
    """Голосование предсказаний по повторам (как в nested_gate): доля 1 >= 0.5."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        v = np.nanmean(P, axis=0)
    return (np.nan_to_num(v, nan=0.0) >= 0.5).astype(int)


def make_group_weights(group, n_boot, rng):
    """Веса объектов при ресэмпле групп с возвращением: W (B × n)."""
    G = group.max() + 1
    draws = rng.integers(0, G, size=(n_boot, G))
    counts = np.zeros((n_boot, G), dtype=float)
    np.add.at(counts, (np.repeat(np.arange(n_boot), G), draws.ravel()), 1.0)
    return counts[:, group]


def cluster_bootstrap(A: dict, n_boot=4000, seed=0):
    """Кластерный бутстрап по группам. Возвращает распределения:
    d_auc — ΔAUC, усреднённый по повторам (ΔAUC_r на ресэмпле, потом среднее по r);
    d_auc_pooled — ΔAUC скора, усреднённого по повторам (как delta_auc_pooled_ci в nested_gate);
    d_macro_f1 — Δmacro-F1 голосования предсказаний по повторам;
    d_macro_f1_mean — среднее по повторам Δmacro-F1_r."""
    rng = np.random.default_rng(seed)
    y, group, R = A['y'], A['group'], A['R']
    W = make_group_weights(group, n_boot, rng)
    d_auc = np.zeros(n_boot); cnt = np.zeros(n_boot)
    d_f1_mean = np.zeros(n_boot)
    for r in range(R):
        m = ~np.isnan(A['score_base'][r]) & ~np.isnan(A['score_cand'][r])
        Wm = W[:, m]
        ab = weighted_auc(Wm, y[m], A['score_base'][r][m])
        ac = weighted_auc(Wm, y[m], A['score_cand'][r][m])
        d = ac - ab
        ok = ~np.isnan(d)
        d_auc[ok] += d[ok]; cnt[ok] += 1
        d_f1_mean += weighted_macro_f1(Wm, y[m], A['pred_cand'][r][m]) - weighted_macro_f1(Wm, y[m], A['pred_base'][r][m])
    with np.errstate(invalid='ignore'):
        d_auc = d_auc / cnt
    d_f1_mean /= R
    sb, sc = np.nanmean(A['score_base'], axis=0), np.nanmean(A['score_cand'], axis=0)
    m = ~np.isnan(sb) & ~np.isnan(sc)
    d_pooled = weighted_auc(W[:, m], y[m], sc[m]) - weighted_auc(W[:, m], y[m], sb[m])
    pb, pc = vote(A['pred_base']), vote(A['pred_cand'])
    d_f1 = weighted_macro_f1(W, y, pc) - weighted_macro_f1(W, y, pb)
    return dict(d_auc=d_auc, d_auc_pooled=d_pooled, d_macro_f1=d_f1, d_macro_f1_mean=d_f1_mean,
                n_boot=n_boot, n_groups=int(group.max() + 1),
                share_undefined=float(np.isnan(d_auc).mean()))


def signflip_permutation(A: dict, n_perm=2000, seed=0):
    """Перестановочный тест: для случайного подмножества групп база и кандидат меняются ролями
    (во всех повторах сразу). Статистика — среднее по повторам ΔAUC. p односторонний: P(T* >= T_obs)."""
    rng = np.random.default_rng(seed + 1)
    y, group, R = A['y'], A['group'], A['R']
    G = group.max() + 1
    flips = rng.integers(0, 2, size=(n_perm, G)).astype(bool)[:, group]      # P × n
    T = np.zeros(n_perm); cnt = np.zeros(n_perm); t_obs = 0.0; c_obs = 0
    for r in range(R):
        sb, sc = A['score_base'][r], A['score_cand'][r]
        m = ~np.isnan(sb) & ~np.isnan(sc)
        sb, sc, ym, F = sb[m], sc[m], y[m], flips[:, m]
        d_obs = fast_auc(ym, sc) - fast_auc(ym, sb)
        if np.isfinite(d_obs):
            t_obs += d_obs; c_obs += 1
        Sb = np.where(F, sc[None, :], sb[None, :])
        Sc = np.where(F, sb[None, :], sc[None, :])
        d = fast_auc_rows(ym, Sc) - fast_auc_rows(ym, Sb)
        ok = np.isfinite(d)
        T[ok] += d[ok]; cnt[ok] += 1
    with np.errstate(invalid='ignore'):
        T = T / cnt
    t_obs = t_obs / max(c_obs, 1)
    p = float((np.sum(T >= t_obs - 1e-12) + 1) / (n_perm + 1))
    return dict(t_obs=float(t_obs), p_one_sided=p, n_perm=n_perm, null_sd=float(np.nanstd(T)))


def jackknife_groups(A: dict):
    """Leave-one-group-out для ΔAUC усреднённого по повторам скора: min/max/число смен знака."""
    y, group = A['y'], A['group']
    sb, sc = np.nanmean(A['score_base'], axis=0), np.nanmean(A['score_cand'], axis=0)
    m = ~np.isnan(sb) & ~np.isnan(sc)
    vals = []
    for g in range(group.max() + 1):
        k = m & (group != g)
        d = fast_auc(y[k], sc[k]) - fast_auc(y[k], sb[k])
        if np.isfinite(d):
            vals.append(d)
    vals = np.array(vals)
    full = fast_auc(y[m], sc[m]) - fast_auc(y[m], sb[m])
    return dict(full=float(full), min=float(vals.min()), max=float(vals.max()),
                n_groups_flip_sign=int(np.sum(np.sign(vals) != np.sign(full))) if full != 0 else None)


def ttest_repeats(rep: pd.DataFrame):
    """Парный t-test по повторам — только для справки (повторы не независимы)."""
    d = rep['delta_auc'].dropna().values
    if len(d) < 2 or np.allclose(d.std(), 0):
        return dict(t=np.nan, p_one_sided=np.nan, sd_repeats=float(d.std(ddof=1)) if len(d) > 1 else np.nan)
    t, p2 = stats.ttest_1samp(d, 0.0)
    p1 = p2 / 2 if t > 0 else 1 - p2 / 2
    return dict(t=float(t), p_one_sided=float(p1), sd_repeats=float(d.std(ddof=1)), n_repeats=int(len(d)))


def pct_ci(x, level):
    x = x[np.isfinite(x)]
    lo, hi = (100 - level) / 2, 100 - (100 - level) / 2
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))]


def adjusted_alpha(alpha, n_hypotheses, family):
    if family == 'none' or n_hypotheses <= 1:
        return alpha
    return alpha / n_hypotheses      # первый шаг Холма = Бонферрони; для одного кандидата в партии


def old_rule(rep: pd.DataFrame, gain_min=OLD_GAIN_MIN, share=OLD_GAIN_SHARE):
    n_gain = int((rep['delta_auc'] >= gain_min).sum())
    need = int(np.ceil(share * len(rep) - 1e-9))
    ok = bool(n_gain >= need and rep['macro_f1_cand'].mean() >= rep['macro_f1_base'].mean() - 1e-12)
    return dict(accepted=ok, n_repeats_gain=n_gain, n_repeats=int(len(rep)), needed=need,
                mean_delta_macro_f1=float(rep['delta_macro_f1'].mean()))


def evaluate_gate(A: dict, alpha=0.05, minimal_effect=0.02, f1_floor=-0.01, f1_ci=60.0, n_boot=4000,
                  n_perm=2000, n_hypotheses=1, family='holm', seed=0, threshold_source='auto',
                  with_jackknife=True, f1_stat='mean', require_perm=True):
    """Полная оценка одного кандидата. Возвращает словарь (сериализуемый в JSON).
    f1_stat: 'vote' — Δmacro-F1 голосования предсказаний по повторам; 'mean' — среднее по повторам Δmacro-F1_r
    (менее шумная статистика, рекомендована по калибровке). require_perm: требовать и p sign-flip < alpha_adj."""
    A, thr_src = derive_predictions(A, threshold_source)
    rep = per_repeat_table(A)
    boot = cluster_bootstrap(A, n_boot=n_boot, seed=seed)
    a_adj = adjusted_alpha(alpha, n_hypotheses, family)
    d = boot['d_auc']
    point = float(rep['delta_auc'].mean())
    p_boot = float((np.sum(d[np.isfinite(d)] <= 0) + 1) / (np.isfinite(d).sum() + 1))
    ci_level_adj = 100 * (1 - 2 * a_adj)
    lo_adj = float(np.percentile(d[np.isfinite(d)], 100 * a_adj))
    f1_dist = boot['d_macro_f1'] if f1_stat == 'vote' else boot['d_macro_f1_mean']
    f1_lo = float(np.percentile(f1_dist, (100 - f1_ci) / 2))
    cond = dict(ci_low_gt_0=bool(lo_adj > 0), point_ge_minimal_effect=bool(point >= minimal_effect),
                f1_ci_low_ge_floor=bool(f1_lo >= f1_floor))
    perm = signflip_permutation(A, n_perm=n_perm, seed=seed) if (n_perm and n_perm > 0) else None
    if perm is not None and require_perm:
        cond['perm_p_lt_alpha'] = bool(perm['p_one_sided'] < a_adj)
    p_combined = max(p_boot, perm['p_one_sided']) if (perm is not None and require_perm) else p_boot
    res = dict(
        n=int(A['n']), n_pos=int(A['y'].sum()), n_groups=boot['n_groups'], n_repeats=int(A['R']),
        n_pos_groups=int(len(np.unique(A['group'][A['y'] == 1]))),
        threshold_source=thr_src, threshold_rule_module=THRESHOLD_SOURCE_MODULE,
        params=dict(alpha=alpha, alpha_adjusted=a_adj, n_hypotheses=n_hypotheses, family=family,
                    minimal_effect=minimal_effect, f1_floor=f1_floor, f1_ci=f1_ci, f1_stat=f1_stat, require_perm=require_perm,
                    n_boot=n_boot, n_perm=n_perm, seed=seed),
        p_combined_one_sided=float(p_combined),
        delta_auc=dict(point_mean_over_repeats=point, min_over_repeats=float(rep['delta_auc'].min()),
                       sd_over_repeats=float(rep['delta_auc'].std(ddof=1)) if len(rep) > 1 else np.nan,
                       boot_mean=float(np.nanmean(d)), boot_se=float(np.nanstd(d)),
                       ci90=pct_ci(d, 90), ci95=pct_ci(d, 95), ci_adjusted=pct_ci(d, ci_level_adj), ci_adjusted_level=ci_level_adj,
                       p_boot_one_sided=p_boot, share_undefined_resamples=boot['share_undefined']),
        delta_auc_pooled=dict(ci90=pct_ci(boot['d_auc_pooled'], 90), ci95=pct_ci(boot['d_auc_pooled'], 95)),
        delta_macro_f1=dict(point_vote=float(macro_f1(A['y'], vote(A['pred_cand'])) - macro_f1(A['y'], vote(A['pred_base']))),
                            point_mean_over_repeats=float(rep['delta_macro_f1'].mean()),
                            ci90=pct_ci(boot['d_macro_f1'], 90), ci95=pct_ci(boot['d_macro_f1'], 95),
                            ci_f1_level=f1_ci, ci_f1_low=f1_lo, f1_stat=f1_stat,
                            mean_over_repeats_ci90=pct_ci(boot['d_macro_f1_mean'], 90),
                            low_bounds={'vote': {lv: float(np.percentile(boot['d_macro_f1'], (100 - lv) / 2)) for lv in (50, 60, 80, 90, 95)},
                                        'mean': {lv: float(np.percentile(boot['d_macro_f1_mean'], (100 - lv) / 2)) for lv in (50, 60, 80, 90, 95)}}),
        auc_base_mean=float(rep['auc_base'].mean()), auc_cand_mean=float(rep['auc_cand'].mean()),
        macro_f1_base_mean=float(rep['macro_f1_base'].mean()), macro_f1_cand_mean=float(rep['macro_f1_cand'].mean()),
        ttest_repeats_reference_only=ttest_repeats(rep),
        conditions=cond, accepted=bool(all(cond.values())),
        old_rule=old_rule(rep),
        per_repeat=rep.round(6).to_dict(orient='records'),
    )
    if perm is not None:
        res['signflip_permutation'] = perm
    if with_jackknife:
        res['jackknife_groups'] = jackknife_groups(A)
    return res


def holm_bh_family(results: list, alpha: float, family: str):
    """Поправка по партии для нескольких кандидатов по p = max(p_boot, p_perm): Холм (step-down) или BH (step-up).
    Обновляет accepted с учётом семейной поправки (остальные условия сохраняются)."""
    p = np.array([r.get('p_combined_one_sided', r['delta_auc']['p_boot_one_sided']) for r in results])
    m = len(p)
    order = np.argsort(p)
    passed = np.zeros(m, dtype=bool)
    if family == 'holm':
        for k, i in enumerate(order):
            if p[i] <= alpha / (m - k):
                passed[i] = True
            else:
                break
    elif family == 'bh':
        thr = alpha * (np.arange(1, m + 1)) / m
        ok = p[order] <= thr
        if ok.any():
            kmax = np.max(np.flatnonzero(ok))
            passed[order[:kmax + 1]] = True
    else:
        passed = p <= alpha
    for r, ok in zip(results, passed):
        r['family_adjustment'] = dict(method=family, m=m, passed_family_step=bool(ok))
        r['conditions']['ci_low_gt_0'] = bool(ok)
        if 'perm_p_lt_alpha' in r['conditions']:
            r['conditions']['perm_p_lt_alpha'] = bool(ok)
        r['accepted'] = bool(all(r['conditions'].values()))
    return results


def _json_ready(o):
    if isinstance(o, dict):
        return {k: _json_ready(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_ready(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _json_ready(o.tolist())
    return o


def summarize(res: dict, name: str) -> str:
    d, f = res['delta_auc'], res['delta_macro_f1']
    lines = [f"[{name}] n={res['n']}, n_pos={res['n_pos']}, групп={res['n_groups']} (с позитивами {res['n_pos_groups']}), повторов={res['n_repeats']}, пороги: {res['threshold_source']}",
             f"  AUC база {res['auc_base_mean']:.3f} -> кандидат {res['auc_cand_mean']:.3f}; ΔAUC={d['point_mean_over_repeats']:+.4f} "
             f"(по повторам sd={d['sd_over_repeats']:.4f}, min={d['min_over_repeats']:+.4f}); бутстрап SE={d['boot_se']:.4f}",
             f"  ΔAUC ДИ90=[{d['ci90'][0]:+.4f}; {d['ci90'][1]:+.4f}], ДИ95=[{d['ci95'][0]:+.4f}; {d['ci95'][1]:+.4f}], "
             f"p_boot={d['p_boot_one_sided']:.4f} (alpha_adj={res['params']['alpha_adjusted']:.4f})",
             f"  Δmacro-F1 (голосование)={f['point_vote']:+.4f}, ДИ{int(f['ci_f1_level'])} низ={f['ci_f1_low']:+.4f}; средн. по повторам={f['point_mean_over_repeats']:+.4f}"]
    if 'signflip_permutation' in res:
        s = res['signflip_permutation']
        lines.append(f"  sign-flip по группам: p={s['p_one_sided']:.4f} (sd нуля {s['null_sd']:.4f})")
    if 'jackknife_groups' in res:
        j = res['jackknife_groups']
        lines.append(f"  jackknife по группам (усреднённый скор): ΔAUC {j['full']:+.4f}, диапазон [{j['min']:+.4f}; {j['max']:+.4f}], смен знака {j['n_groups_flip_sign']}")
    t = res['ttest_repeats_reference_only']
    lines.append(f"  парный t-test по повторам (справочно, завышает уверенность): p={t['p_one_sided']:.2e}")
    o = res['old_rule']
    lines.append(f"  старое правило: ΔAUC>=0.03 в {o['n_repeats_gain']}/{o['n_repeats']} (нужно {o['needed']}), Δmacro-F1 средн. {o['mean_delta_macro_f1']:+.4f} -> {'ПРИНЯТ' if o['accepted'] else 'не принят'}")
    c = res['conditions']
    lines.append(f"  новое правило: ДИ>0 {c['ci_low_gt_0']}, sign-flip {c.get('perm_p_lt_alpha', 'не требуется')}, ΔAUC>=min_effect {c['point_ge_minimal_effect']}, "
                 f"F1 не хуже ({res['delta_macro_f1']['f1_stat']}) {c['f1_ci_low_ge_floor']} -> {'ПРИНЯТ' if res['accepted'] else 'не принят'}")
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', help='CSV/NPZ с per-repeat OOF-скорами базы и кандидата (один файл = одна гипотеза)')
    ap.add_argument('--alpha', type=float, default=DEFAULTS['alpha'], help='односторонний уровень (0.05 = нижняя граница 90 %% ДИ)')
    ap.add_argument('--minimal-effect', type=float, default=DEFAULTS['minimal_effect'])
    ap.add_argument('--f1-floor', type=float, default=DEFAULTS['f1_floor'])
    ap.add_argument('--f1-ci', type=float, default=DEFAULTS['f1_ci'], help='уровень ДИ для условия по Δmacro-F1')
    ap.add_argument('--f1-stat', choices=['vote', 'mean'], default=DEFAULTS['f1_stat'], help='статистика Δmacro-F1: голосование по повторам или среднее по повторам')
    ap.add_argument('--no-require-perm', action='store_true', help='не требовать p sign-flip < alpha (только бутстрап-ДИ)')
    ap.add_argument('--n-boot', type=int, default=DEFAULTS['n_boot'])
    ap.add_argument('--n-perm', type=int, default=DEFAULTS['n_perm'], help='0 — без перестановочного теста')
    ap.add_argument('--n-hypotheses', type=int, default=DEFAULTS['n_hypotheses'], help='размер партии гипотез (если входов меньше)')
    ap.add_argument('--family', choices=['holm', 'bh', 'none'], default=DEFAULTS['family'])
    ap.add_argument('--threshold-source', choices=['auto', 'pred', 'crossfit', 'insample'], default='auto')
    ap.add_argument('--seed', type=int, default=DEFAULTS['seed'])
    ap.add_argument('--no-jackknife', action='store_true')
    ap.add_argument('--out', type=str, default=None, help='JSON с результатами')
    args = ap.parse_args(argv)

    results, names = [], []
    m = max(args.n_hypotheses, len(args.inputs))
    for path in args.inputs:
        A = to_arrays(load_gate_input(path))
        res = evaluate_gate(A, alpha=args.alpha, minimal_effect=args.minimal_effect, f1_floor=args.f1_floor, f1_ci=args.f1_ci,
                            n_boot=args.n_boot, n_perm=args.n_perm, n_hypotheses=m, family=args.family, seed=args.seed,
                            threshold_source=args.threshold_source, with_jackknife=not args.no_jackknife,
                            f1_stat=args.f1_stat, require_perm=not args.no_require_perm)
        res['input'] = str(path)
        results.append(res); names.append(Path(path).stem)
    if len(results) > 1 and args.family != 'none':
        if len(results) < m:
            # партия больше числа входов: неизвестные p считаем наихудшими -> Бонферрони alpha/m уже применён
            pass
        else:
            holm_bh_family(results, args.alpha, args.family)
    for res, name in zip(results, names):
        print(summarize(res, name))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(_json_ready(results if len(results) > 1 else results[0]), f, indent=2, ensure_ascii=False)
        print(f'результат: {args.out}')
    return results


if __name__ == '__main__':
    main()
