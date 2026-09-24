"""Б2, чувствительность (не объявлена заранее): вариант (а) «только на уровне».

Основной вариант (tie_midpoint_threshold) переносит порог в середину промежутка всегда, в том числе
когда квантиль лёг между уровнями (на референсе решения не меняются, на новых данных — меняются).
Здесь порог переносится, только если он совпадает с уровнем inner-OOF (|thr - уровень| <= TIE_EPS),
то есть ровно случай связки, где строгий «>» и «>=» различаются. Пишет b2_<tag>_tieonly_<region>.npz
с заменённым THR_TIE; дальше tools/decision_layer_eval.py --tag <tag>_tieonly.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_rules import TIE_EPS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--inp', required=True)
ap.add_argument('--tag', required=True)
a = ap.parse_args()
for region in ('spine', 'hip'):
    src = Path(a.inp) / f'b2_{a.tag}_{region}.npz'
    D = dict(np.load(src, allow_pickle=True))
    T, TT, SIN = D['THR'], D['THR_TIE'].copy(), D['SIN']
    R, K, C = T.shape
    moved = 0
    for r in range(R):
        for k in range(K):
            for j in range(C):
                s = SIN[r, k][:, j]
                s = s[np.isfinite(s)]
                if not np.isfinite(T[r, k, j]) or s.size == 0:
                    continue
                on_level = np.min(np.abs(s - T[r, k, j])) <= TIE_EPS
                if not on_level:
                    TT[r, k, j] = T[r, k, j]
                else:
                    moved += 1
    D['THR_TIE'] = TT
    np.savez_compressed(Path(a.inp) / f'b2_{a.tag}_tieonly_{region}.npz', **D)
    print(region, 'порогов на уровне (перенесены):', moved, 'из', R * K * C)
