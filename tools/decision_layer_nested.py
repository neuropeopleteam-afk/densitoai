"""
Б2 (24.09.2026): решающий слой — вложенная повторная GroupKFold для двух гипотез.

  (а) decision.threshold_tie_rule — порог как середина между соседними уникальными уровнями
      референсной выборки (inner-OOF стэк), эквивалент строгого «>» на референсе;
  (б) decision.exclusive_types — одно нарушение на строку: при >= 2 флагах области оставить тип
      с наибольшим относительным запасом (score - thr)/(1 - thr); второй тип сохраняется, только если
      его запас >= tau. Политика tau (включая «выключено») выбирается на внутренних фолдах.

Этот скрипт — только расчёт (тяжёлый): для каждого повтора и внешнего фолда обучает модели
контуров A/B ровно как train_stacked.py (признаки contour_a_cols, варианты предобработки К11,
источник эмбеддингов К13, LogReg/PCA с теми же гиперпараметрами, стэк 0.5/0.5 по рангам),
считает inner-OOF (3 фолда) на внешнем train, пороги по правилу config.yaml thresholds_rule
на inner-OOF, скоры внешнего test (ранги относительно inner-OOF-референса, как
inference.percentile_rank) и any-модель области (для quality_prob). Всё сохраняется в NPZ;
метрики и решения считает tools/decision_layer_eval.py.

Протокол: внешний GroupKFold(5, shuffle, random_state=42+r) × N_REPEATS (20), один и тот же для всех
критериев области (нужно для (б)); группы = компоненты связности (study, pixel_hash);
внутренний GroupKFold(3, shuffle, seed=1000*r+k). Медианы импутации — по внешнему train.

  --unique : один кадр на pixel_hash (дедупликация клонов; 252 кадра из 499) — проверка
             устойчивости к клонам (находка 1 консилиума).

Запуск (из корня проекта): nice -n 10 env OMP_NUM_THREADS=1 DENSITO_ROOT=$PWD \
    ../venv/bin/python tools/decision_layer_nested.py [--unique] [--repeats 20] --out outputs/b2
"""
import os, sys, time, argparse, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(os.environ.get('DENSITO_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
import nested_gate as ng  # noqa: E402  (fit_geom/fit_emb/pct_rank/ref_rank/stack/connected_groups/impute)
from train_stacked import (DATA_DIR, REGION_CRITERIA, CRITERION_GEOMETRY_COLS, CRITERION_LABEL_COL,  # noqa: E402
                           REGION_ROWS, GEOM_VARIANT_FILES, preproc_for, contour_a_cols, emb_matrix_for,
                           load_embeddings_by_source, threshold_rule_for, f1_optimal_threshold,
                           prevalence_threshold)
from calibration_utils import threshold_by_rule  # noqa: E402
from decision_rules import tie_midpoint_threshold  # noqa: E402

N_OUTER, N_INNER = 5, 3


def load_region(region, unique=False):
    rows = REGION_ROWS[region]
    base = pd.read_csv(DATA_DIR / GEOM_VARIANT_FILES['baseline'])
    keep = base['region'].isin(rows).values
    geoms = {}
    for v, fn in GEOM_VARIANT_FILES.items():
        f = DATA_DIR / fn
        if f.exists():
            g = pd.read_csv(f)
            assert (g['file_path'].values == base['file_path'].values).all()
            geoms[v] = g[keep].reset_index(drop=True)
    lab = pd.read_csv(DATA_DIR / 'labels_for_embeddings.csv')
    eidx = np.nonzero(lab['region'].isin(rows).values)[0]
    assert (lab.loc[eidx, 'file_path'].values == geoms['baseline']['file_path'].values).all()
    emb_all = load_embeddings_by_source()
    E = {k: v[eidx] for k, v in emb_all.items()}
    g0 = geoms['baseline']
    h = pd.read_csv(ROOT / 'docs' / 'k5' / 'pixel_hashes.csv')
    hmap = dict(zip(h['file_path'], h['pixel_hash']))
    ph = g0['file_path'].map(hmap)
    assert ph.notna().all(), 'нет pixel_hash для части кадров'
    criteria = REGION_CRITERIA[region]
    valid = np.ones(len(g0), bool)
    for c in criteria:
        valid &= g0[CRITERION_LABEL_COL.get(c, c)].notna().values
    if unique:
        first = ~ph.duplicated().values
        valid &= first
    idx = np.nonzero(valid)[0]
    groups = ng.connected_groups(g0['study'].values[idx], ph.values[idx])
    return geoms, E, idx, groups, ph.values[idx]


def run(out, repeats, unique, log):
    out.mkdir(parents=True, exist_ok=True)
    for region, criteria in REGION_CRITERIA.items():
        geoms, E, idx, groups, ph = load_region(region, unique)
        g0 = geoms['baseline'].iloc[idx].reset_index(drop=True)
        n = len(idx)
        Y = np.stack([g0[CRITERION_LABEL_COL.get(c, c)].values.astype(int) for c in criteria], 1)
        Xraw, Emb, rule = {}, {}, {}
        for c in criteria:
            pp = preproc_for(c)
            gv = pp['geom'] if pp['geom'] in geoms else 'baseline'
            Xraw[c] = geoms[gv][contour_a_cols(c)].values.astype(np.float64)[idx]
            Em, src, ev = emb_matrix_for(c, E)
            Emb[c] = Em[idx]
            rule[c] = threshold_rule_for(c)
            log(f"{region}/{c}: n={n} pos={int(Y[:, criteria.index(c)].sum())} A='{gv}' {contour_a_cols(c)} "
                f"B='{src}'/'{ev}' rule={rule[c]}")
        any_cols = sorted({cc for cr in criteria for cc in CRITERION_GEOMETRY_COLS[cr]})
        Xany_raw = geoms['baseline'][any_cols].values.astype(np.float64)[idx]
        Eany = E['imagenet'][idx]
        y_any = Y.max(1)
        C = len(criteria)
        S = np.full((repeats, n, C), np.nan)            # внешний test-скор критерия
        FOLD = np.full((repeats, n), -1, int)
        THR = np.full((repeats, N_OUTER, C), np.nan)     # порог по действующему правилу (inner-OOF)
        THR_TIE = np.full((repeats, N_OUTER, C), np.nan) # (а): середина между уровнями
        ANY = np.full((repeats, n), np.nan)              # any-модель (среднее p geom/emb) на test
        SIN = np.full((repeats, N_OUTER, n, C), np.nan)  # inner-OOF стэк на строках внешнего train
        t0 = time.time()
        for r in range(repeats):
            splits = list(GroupKFold(n_splits=N_OUTER, shuffle=True, random_state=42 + r).split(np.zeros(n), groups=groups))
            for k, (tr, te) in enumerate(splits):
                FOLD[r, te] = k
                for j, c in enumerate(criteria):
                    y = Y[:, j]
                    if ng.degenerate(y[tr]):
                        continue
                    med = np.nanmedian(Xraw[c][tr], axis=0)
                    X = ng.impute(Xraw[c], med)
                    og, oe = ng.inner_oof(X[tr], Emb[c][tr], y[tr], groups[tr], seed=1000 * r + k)
                    ok = ~np.isnan(og) & ~np.isnan(oe)
                    if ok.sum() < 2 or len(np.unique(y[tr][ok])) < 2:
                        continue
                    s_in = ng.stack(ng.pct_rank(og[ok]), ng.pct_rank(oe[ok]), ng.W_BASE)
                    y_in = y[tr][ok]
                    t, _ = threshold_by_rule(rule[c], y_in, s_in, f1_optimal_threshold, prevalence_threshold)
                    THR[r, k, j] = t
                    THR_TIE[r, k, j] = tie_midpoint_threshold(t, s_in)
                    SIN[r, k, tr[ok], j] = s_in
                    pg = ng.fit_geom(X[tr], y[tr])(X[te]); pe = ng.fit_emb(Emb[c][tr], y[tr])(Emb[c][te])
                    S[r, te, j] = ng.stack(ng.ref_rank(pg, og[ok]), ng.ref_rank(pe, oe[ok]), ng.W_BASE)
                if not ng.degenerate(y_any[tr]):
                    med = np.nanmedian(Xany_raw[tr], axis=0); Xa = ng.impute(Xany_raw, med)
                    pa = ng.fit_geom(Xa[tr], y_any[tr])(Xa[te]); pb = ng.fit_emb(Eany[tr], y_any[tr])(Eany[te])
                    ANY[r, te] = 0.5 * (pa + pb)
            log(f"  {region} repeat {r}: {time.time() - t0:.0f} с")
        tag = 'unique' if unique else 'all'
        np.savez_compressed(out / f'b2_{tag}_{region}.npz', criteria=np.array(criteria), Y=Y, S=S, FOLD=FOLD, THR=THR,
                            THR_TIE=THR_TIE, ANY=ANY, SIN=SIN, groups=groups, study=g0['study'].values.astype(str),
                            file_path=g0['file_path'].values.astype(str), pixel_hash=ph.astype(str))
        log(f"  сохранено {out / f'b2_{tag}_{region}.npz'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--unique', action='store_true')
    ap.add_argument('--repeats', type=int, default=20)
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'b2'))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / f"run_{'unique' if a.unique else 'all'}.log", 'a', encoding='utf-8')

    def log(s):
        print(s, flush=True); logf.write(s + '\n'); logf.flush()
    run(out, a.repeats, a.unique, log)
    log('DONE')


if __name__ == '__main__':
    main()
