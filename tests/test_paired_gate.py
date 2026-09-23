"""Тесты калиброванного гейта (tools/paired_gate.py): детерминизм, принятие известного эффекта,
контроль ошибки первого рода на нуле, совпадение быстрых метрик с sklearn/train_stacked. Быстрая версия (< 30 с)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
try:
    import pytest
except ModuleNotFoundError:  # в образе pytest нет: минимальная замена, чтобы файл запускался как скрипт
    class _Approx:
        def __init__(self, v, rel=1e-6, abs_=1e-12): self.v, self.rel, self.abs = v, rel, abs_
        def __eq__(self, other): return abs(other - self.v) <= max(self.rel * abs(self.v), self.abs)

    class _Skip(Exception):
        pass

    class pytest:  # noqa: N801
        skip = staticmethod(lambda msg='': (_ for _ in ()).throw(_Skip(msg)))
        approx = staticmethod(lambda v, rel=1e-6, abs=1e-12: _Approx(v, rel, abs))
        _Skip = _Skip
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'src'))
import paired_gate as pg  # noqa: E402


def synth(seed, n=200, n_groups=80, prevalence=0.15, R=8, effect=0.0, rho=0.8, split_sigma=0.08):
    """Синтетика с известным эффектом: бинормальная пара база/кандидат с корреляцией rho,
    AUC базы ~0.75, кандидата ~0.75+effect; группы разного размера; шум разбиений по повторам."""
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    group = np.sort(rng.integers(0, n_groups, n))
    y = (rng.random(n) < prevalence).astype(int)
    d_b, d_c = np.sqrt(2) * norm.ppf(0.75), np.sqrt(2) * norm.ppf(0.75 + effect)
    u, v, w = rng.normal(size=(3, n))
    base = 0.25 * (d_b * y + np.sqrt(rho) * u + np.sqrt(1 - rho) * v)
    cand = 0.25 * (d_c * y + np.sqrt(rho) * u + np.sqrt(1 - rho) * w)
    sb = base[None, :] + rng.normal(0, split_sigma, (R, n))
    sc = cand[None, :] + rng.normal(0, split_sigma, (R, n))
    fold = rng.integers(0, 5, (R, n))
    return dict(y=y, group=group, n=n, R=R, score_base=sb, score_cand=sc, fold=fold)


def test_fast_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = int(rng.integers(30, 150))
        y = (rng.random(n) < 0.25).astype(int)
        if y.sum() == 0 or y.sum() == n:
            continue
        s = np.round(rng.random(n) * 7) / 7          # связи
        assert abs(pg.fast_auc(y, s) - roc_auc_score(y, s)) < 1e-12
        W = rng.integers(0, 3, size=(5, n)).astype(float)
        wa = pg.weighted_auc(W, y, s)
        p = (s >= np.median(s)).astype(int)
        wf = pg.weighted_macro_f1(W, y, p)
        for b in range(5):
            idx = np.repeat(np.arange(n), W[b].astype(int))
            if len(np.unique(y[idx])) < 2:
                assert np.isnan(wa[b])
                continue
            assert abs(wa[b] - roc_auc_score(y[idx], s[idx])) < 1e-10
            assert abs(wf[b] - f1_score(y[idx], p[idx], average='macro', zero_division=0)) < 1e-10


def test_threshold_rules_match_train_stacked():
    try:
        import train_stacked as ts
    except Exception:  # noqa: BLE001
        pytest.skip('train_stacked недоступен')
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = int(rng.integers(60, 200))
        y = (rng.random(n) < 0.3).astype(int)
        s = np.round(rng.random(n) * 20) / 20
        if y.sum() >= 15:
            assert abs(pg.f1_optimal_threshold_fast(y, s) - ts.f1_optimal_threshold(y, s)) < 1e-12
        assert abs(pg.prevalence_threshold(y, s) - ts.prevalence_threshold(y, s)) < 1e-12


def test_io_roundtrip(tmp_path):
    A = synth(3, R=3)
    for ext in ('csv', 'npz'):
        p = pg.write_gate_input(tmp_path / f'x.{ext}', A['y'], A['group'], A['score_base'], A['score_cand'], fold=A['fold'])
        B = pg.to_arrays(pg.load_gate_input(p))
        assert B['n'] == A['n'] and B['R'] == A['R']
        assert np.allclose(B['score_base'], A['score_base']) and np.allclose(B['score_cand'], A['score_cand'])
        assert (B['y'] == A['y']).all()
        assert np.array_equal(B['fold'], A['fold'])


def test_determinism_with_seed():
    A = synth(5, effect=0.05)
    r1 = pg.evaluate_gate(dict(A), n_boot=300, n_perm=200, seed=11)
    r2 = pg.evaluate_gate(dict(A), n_boot=300, n_perm=200, seed=11)
    assert r1['delta_auc'] == r2['delta_auc']
    assert r1['signflip_permutation'] == r2['signflip_permutation']
    assert r1['accepted'] == r2['accepted']
    r3 = pg.evaluate_gate(dict(A), n_boot=300, n_perm=200, seed=12)
    assert r3['delta_auc']['ci90'] != r1['delta_auc']['ci90']      # другой seed — другой бутстрап


def test_accepts_known_large_effect():
    acc = 0
    for seed in range(6):
        A = synth(100 + seed, effect=0.15, n=300, n_groups=120)
        r = pg.evaluate_gate(dict(A), n_boot=300, n_perm=200, seed=seed, f1_ci=60.0)
        acc += int(r['conditions']['ci_low_gt_0'] and r['conditions']['point_ge_minimal_effect'] and r['signflip_permutation']['p_one_sided'] < 0.05)
    assert acc >= 5


def test_type_one_error_on_null_within_bounds():
    """Ноль: две модели равной AUC (эффект 0). Доля ложных принятий AUC-части правила при 40 симуляциях
    должна быть в допустимых пределах для alpha=0.05 (биномиальная граница ~ 0.15)."""
    false_acc, tt = 0, 0
    n_sim = 40
    for seed in range(n_sim):
        A = synth(1000 + seed, effect=0.0, n=250, n_groups=100)
        r = pg.evaluate_gate(dict(A), n_boot=250, n_perm=0, seed=seed, with_jackknife=False)
        false_acc += int(r['conditions']['ci_low_gt_0'] and r['conditions']['point_ge_minimal_effect'])
        tt += int(r['ttest_repeats_reference_only']['p_one_sided'] < 0.05 and r['delta_auc']['point_mean_over_repeats'] >= 0.02)
    assert false_acc <= 6, f'ложных принятий {false_acc}/{n_sim}'
    # парный t-test по повторам на том же нуле принимает чаще: повторы не независимы
    assert tt >= false_acc


def test_old_rule_and_family_adjustment():
    rep = pd.DataFrame({'delta_auc': [0.04] * 7 + [0.0] * 3, 'macro_f1_cand': [0.6] * 10, 'macro_f1_base': [0.6] * 10, 'delta_macro_f1': [0.0] * 10})
    assert pg.old_rule(rep)['accepted']
    rep.loc[0, 'delta_auc'] = 0.0
    assert not pg.old_rule(rep)['accepted']
    assert pg.adjusted_alpha(0.05, 5, 'holm') == pytest.approx(0.01)
    assert pg.adjusted_alpha(0.05, 5, 'none') == 0.05
    res = [{'delta_auc': {'p_boot_one_sided': p}, 'conditions': {'ci_low_gt_0': True, 'point_ge_minimal_effect': True, 'f1_ci_low_ge_floor': True}, 'accepted': True}
           for p in (0.001, 0.03, 0.04)]
    out = pg.holm_bh_family(res, 0.05, 'holm')
    assert [r['accepted'] for r in out] == [True, False, False]
    out = pg.holm_bh_family([dict(r, conditions=dict(r['conditions'])) for r in res], 0.05, 'bh')
    assert [r['accepted'] for r in out] == [True, True, True]


def test_cli_smoke(tmp_path):
    A = synth(7, effect=0.10, R=4)
    p = pg.write_gate_input(tmp_path / 'c.csv', A['y'], A['group'], A['score_base'], A['score_cand'], fold=A['fold'])
    out = tmp_path / 'r.json'
    res = pg.main([str(p), '--n-boot', '200', '--n-perm', '100', '--n-hypotheses', '3', '--out', str(out)])
    assert out.exists() and res[0]['params']['alpha_adjusted'] == pytest.approx(0.05 / 3)


if __name__ == '__main__':  # запуск без pytest: python tests/test_paired_gate.py
    import inspect
    import tempfile
    import traceback
    failed = 0
    for name, fn in sorted(globals().items()):
        if not (name.startswith('test_') and inspect.isfunction(fn)):
            continue
        try:
            if 'tmp_path' in inspect.signature(fn).parameters:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f'OK   {name}')
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ == '_Skip' or (hasattr(pytest, 'skip') and 'Skipped' in type(e).__name__):
                print(f'SKIP {name}: {e}')
                continue
            failed += 1
            print(f'FAIL {name}')
            traceback.print_exc()
    print('test_paired_gate:', 'FAIL' if failed else 'OK', f'(failed {failed})')
    sys.exit(1 if failed else 0)
