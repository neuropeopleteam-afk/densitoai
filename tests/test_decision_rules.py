"""Тесты правил Б2 (tools/decision_rules.py): середина между уровнями эквивалентна строгому «>»
на референсе; «одно нарушение на строку» не меняет OR флагов (quality_class, quality_prob).
Гипотезы отвергнуты (docs/NESTED_GATE_REPORT.md, часть 6), тест фиксирует поведение опубликованных скриптов."""
import math
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "tools"))
from decision_rules import (exclusive_types, exclusive_types_array, relative_margin,  # noqa: E402
                            tie_midpoint_threshold)


def test_midpoint_equals_strict_on_reference():
    rng = np.random.default_rng(0)
    for _ in range(200):
        ref = np.round(rng.random(60), 2)          # много связок
        thr = float(rng.choice(ref))               # порог ровно на уровне
        t2 = tie_midpoint_threshold(thr, ref)
        assert np.array_equal(ref >= t2, ref > thr)
        assert t2 > thr


def test_off_level_and_edges():
    ref = np.array([0.1, 0.2, 0.2, 0.5])
    assert tie_midpoint_threshold(0.3, ref) == 0.35         # основной вариант: середина промежутка и вне уровня
    assert np.array_equal(ref >= 0.35, ref >= 0.3)          # на референсе решения те же
    assert tie_midpoint_threshold(0.05, ref) == 0.05        # ниже всех уровней
    assert tie_midpoint_threshold(0.5, ref) == 0.75         # верхний уровень -> середина до 1.0
    assert tie_midpoint_threshold(0.2, ref) == 0.35
    assert tie_midpoint_threshold(0.2, None) == 0.2


def test_one_ulp_tolerance():
    thr = 0.8614457831325302
    ref = np.array([0.5, 0.8614457831325301, 0.9])          # уровень на 1 ulp ниже порога
    assert tie_midpoint_threshold(thr, ref) == (0.8614457831325301 + 0.9) / 2


def test_production_oof_tie_blocks():
    """На сохранённых OOF правило (а) воспроизводит строгий «>» (числа находки 1 совета)."""
    exp = {'spine_sp_pos': (0.8614457831325302, 18, 8), 'spine_sp_axis': (0.7771084337349398, 26, 23),
           'spine_sp_art': (0.6239385251850776, 49, 49), 'hip_hip_pos': (0.6580547112462006, 83, 78),
           'hip_hip_roi': (0.9179331306990881, 19, 15)}
    for k, (thr, n0, n1) in exp.items():
        p = ROOT / 'models' / f'oof_stacked_{k}.csv'
        if not p.exists():
            print(f'SKIP {k}: нет {p}')
            continue
        s = pd.read_csv(p, float_precision='round_trip')['oof_stacked'].values
        assert int((s >= thr).sum()) == n0, k
        assert int((s >= tie_midpoint_threshold(thr, s)).sum()) == n1, k


def test_exclusive_keeps_or_identity():
    rng = np.random.default_rng(1)
    T = np.array([0.86, 0.78, 0.60])
    for tau in (0.0, 0.1, 0.3, math.inf):
        S = rng.random((500, 3))
        F = (S >= T).astype(int)
        G = exclusive_types_array(F, S, T, tau)
        assert np.array_equal(G.max(1), F.max(1))           # quality_class и quality_prob не меняются
        assert (G <= F).all()                               # флаги только снимаются
        if tau == math.inf:
            assert (G.sum(1) <= 1).all()
        if tau == 0.0:
            assert np.array_equal(G, F)


def test_exclusive_dict_picks_largest_relative_margin():
    thr = {'sp_pos': 0.86, 'sp_axis': 0.78, 'sp_art': 0.60}
    sc = {'sp_pos': 0.90, 'sp_axis': 0.95, 'sp_art': 0.99}
    fl = {k: int(sc[k] >= thr[k]) for k in thr}
    out = exclusive_types(fl, sc, thr)
    best = max(thr, key=lambda k: relative_margin(sc[k], thr[k]))
    assert out == {k: int(k == best) for k in thr}
    assert max(out.values()) == max(fl.values())


if __name__ == '__main__':
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'OK   {name}')
            except Exception:  # noqa: BLE001
                failed += 1
                print(f'FAIL {name}')
                traceback.print_exc()
    print('test_decision_rules:', 'FAIL' if failed else 'OK', f'(failed {failed})')
    sys.exit(1 if failed else 0)
