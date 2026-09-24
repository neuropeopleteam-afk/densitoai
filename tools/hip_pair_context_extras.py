"""
Б1: дополнительные проверки к tools/hip_pair_context_gate.py (читает scores_<tag>.npz).
  1) исследования с разной меткой hip_pos на сторонах: ошибки базы и кандидатов;
  2) строки без второго бедра: меняется ли скор и флаг;
  3) устойчивость к выбросу одного исследования (leave-one-study-out по повторам);
  4) те же правила на сохранённом OOF поставки (models/oof_stacked_hip_hip_pos.csv) — откуда +0.049/+0.071.
Запуск: B1_OUT=... TAG=canonical_prevalence_R20 python tools/hip_pair_context_extras.py
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ.get('DENSITO_ROOT', Path(__file__).resolve().parents[1]))
OUT = Path(os.environ.get('B1_OUT', ROOT / 'outputs' / 'b1'))
TAG = os.environ.get('TAG', 'canonical_prevalence_R20')
CANDS = ['w_inner', 'w_fixed_0.3', 'grok_solidity', 'k5_rule_mean']


def auc_rows(y, S):
    return np.array([roc_auc_score(y, s) for s in S])


def main():
    Z = np.load(OUT / f'scores_{TAG}.npz', allow_pickle=True)
    y, studies, sides, has_other = Z['y'], Z['studies'], Z['sides'], Z['has_other']
    Sb, Pb = Z['S_base'], Z['P_base']
    res = {'tag': TAG}
    # 1) расхождение меток сторон
    df = pd.DataFrame({'study': studies, 'side': sides, 'y': y})
    lab = df.groupby(['study', 'side']).y.max().unstack()
    both = lab.dropna()
    disc = both.index[both.left != both.right].tolist()
    conc_pos = both.index[(both.left == 1) & (both.right == 1)].tolist()
    md = np.isin(studies, disc)
    res['discordant'] = {'n_studies_both_sides': int(len(both)), 'n_discordant': len(disc), 'n_concordant_pos': len(conc_pos),
                         'n_rows': int(md.sum()), 'n_pos_rows': int(y[md].sum())}
    err_b = ((Pb[:, md] != y[md]).sum(1)).mean()
    res['discordant']['errors_base_mean'] = float(err_b)
    for v in CANDS:
        S, P = Z[f'S_{v}'], Z[f'P_{v}']
        err = ((P[:, md] != y[md]).sum(1)).mean()
        # сдвиг ранга скора внутри повтора (доля строк ниже) для позитивов и негативов
        rb = np.array([pd.Series(s).rank(pct=True).values for s in Sb]); rc = np.array([pd.Series(s).rank(pct=True).values for s in S])
        dpos = (rc - rb)[:, md & (y == 1)].mean(); dneg = (rc - rb)[:, md & (y == 0)].mean()
        res['discordant'][v] = {'errors_mean': float(err), 'rank_shift_pos': float(dpos), 'rank_shift_neg': float(dneg),
                                'rank_shift_pos_concordant_pos': float((rc - rb)[:, np.isin(studies, conc_pos) & (y == 1)].mean())}
    # 2) без второго бедра
    mn = ~has_other
    res['no_second_hip'] = {'n_rows': int(mn.sum()), 'n_studies': int(len(np.unique(studies[mn]))), 'n_pos': int(y[mn].sum())}
    for v in CANDS:
        S, P = Z[f'S_{v}'], Z[f'P_{v}']
        res['no_second_hip'][v] = {'max_abs_score_diff': float(np.nanmax(np.abs(S[:, mn] - Sb[:, mn]))),
                                   'flag_flips_total_over_repeats': int((P[:, mn] != Pb[:, mn]).sum()),
                                   'row_repeats': int(mn.sum() * len(Sb))}
    # 3) leave-one-study-out
    ab = auc_rows(y, Sb)
    uniq = np.unique(studies)
    res['loso'] = {}
    for v in CANDS:
        S = Z[f'S_{v}']
        d_full = auc_rows(y, S) - ab
        cnts, means = [], []
        for s in uniq:
            k = studies != s
            d = auc_rows(y[k], S[:, k]) - auc_rows(y[k], Sb[:, k])
            cnts.append(int((d >= 0.03).sum())); means.append(float(d.mean()))
        cnts, means = np.array(cnts), np.array(means)
        i_max, i_min = int(np.argmax(cnts)), int(np.argmin(means))
        res['loso'][v] = {'n_gain_full': int((d_full >= 0.03).sum()), 'mean_delta_full': float(d_full.mean()),
                          'n_gain_min': int(cnts.min()), 'n_gain_max': int(cnts.max()),
                          'study_max_gain': str(uniq[i_max]), 'mean_delta_min': float(means.min()), 'mean_delta_max': float(means.max()),
                          'study_min_delta': str(uniq[i_min]), 'n_studies_reaching_14': int((cnts >= 14).sum())}
    # 4) сохранённый OOF поставки: база 0.725 и те же правила без переобучения
    o = pd.read_csv(ROOT / 'models' / 'oof_stacked_hip_hip_pos.csv')
    s = o.oof_stacked.values; yo = o.y_true.values.astype(int); R = pd.Series(s).rank(pct=True).values
    oth = [np.nonzero((o.study.values == o.study.values[i]) & (o.hip_side_detected.values != o.hip_side_detected.values[i]))[0] for i in range(len(o))]
    mo = np.array([R[j].mean() if len(j) else np.nan for j in oth])
    g = pd.read_csv(ROOT / 'data' / 'geometry_features.csv').set_index('file_path')
    sol = g.loc[o.file_path.values, 'femur_solidity'].values
    wit = np.array([sol[j].mean() if len(j) else np.nan for j in oth])
    Rw = pd.Series(wit).rank(pct=True).values
    so = {'auc_base': roc_auc_score(yo, s)}
    for w in (0.3, 0.4, 0.5):
        so[f'auc_w{w}'] = roc_auc_score(yo, np.where(np.isnan(mo), R, (1 - w) * R + w * mo))
    so['auc_grok'] = roc_auc_score(yo, np.where(np.isnan(Rw), R, 0.5 * R + 0.5 * Rw))
    # усреднённый по 20 повторам скор харнесса (ближе к сохранённому OOF, который усреднён по 5 повторам)
    for v in ('w_fixed_0.3', 'grok_solidity'):
        so[f'harness_mean_over_repeats_delta_{v}'] = float(roc_auc_score(y, np.nanmean(Z[f'S_{v}'], 0)) - roc_auc_score(y, np.nanmean(Sb, 0)))
    so['harness_auc_base_mean_score'] = float(roc_auc_score(y, np.nanmean(Sb, 0)))
    res['saved_oof'] = {k: float(v) for k, v in so.items()}
    (OUT / f'extras_{TAG}.json').write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
