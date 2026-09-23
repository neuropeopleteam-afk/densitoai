#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тест отчёта о калибровке (tools/calibration_report.py).

Проверяется:
  1. ECE идеально калиброванного синтетического предиктора (p ~ U(0,1), y ~ Bernoulli(p), n = 200 000) близок к 0
     для обеих разбивок; наклон рекалибровки близок к 1, сдвиг — к 0;
  2. ECE константного предиктора c равен |mean(y) - c| точно (обе разбивки), Brier — mean((c - y)^2);
  3. детерминизм: два вызова evaluate() с бутстрапом дают побайтно одинаковый JSON;
  4. бины: сумма n по бинам равна n, квантильные границы монотонны, ECE <= MCE;
  5. ROC-AUC через ранги совпадает со sklearn (если sklearn доступен);
  6. SVG — корректный XML, без внешних зависимостей, без запрещённых слов;
  7. при наличии OOF-файлов поставки: build_targets() детерминирован (два прогона, 20 бутстрап-повторов),
     ROC-AUC итогового quality_prob совпадает с models/metrics_oof_full.json, in-sample Platt из calibration.pkl даёт
     наклон 1 и сдвиг 0 (тождество логистической рекалибровки), JSON не содержит путей к кадрам.

Запуск: python tests/test_calibration_report.py   (или pytest -q tests/test_calibration_report.py)
"""
from __future__ import annotations

import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import calibration_report as cr  # noqa: E402

FORBIDDEN = ["Grad" + "-CAM", "автокоррекц" + "ия ROI", "ЕР" + "ИС", "сколи" + "оз", "угол К" + "обба"]


def _synthetic(n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0, 1.0, n)
    y = (rng.uniform(0.0, 1.0, n) < p).astype(int)
    g = rng.integers(0, 500, n)
    return y, p, g


def test_ece_perfectly_calibrated_near_zero():
    y, p, _ = _synthetic()
    e10, m10 = cr.ece_mce(y, p, cr.bin_edges_equal(10))
    eq, _ = cr.ece_mce(y, p, cr.bin_edges_quantile(p, 5))
    assert e10 < 0.01, e10
    assert eq < 0.01, eq
    assert m10 < 0.03, m10
    slope, intercept = cr.logistic_recalibration(y, p)
    assert abs(slope - 1.0) < 0.05, slope
    assert abs(intercept) < 0.05, intercept


def test_ece_constant_predictor_equals_abs_gap():
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=1000) < 0.3).astype(int)
    for c in (0.0, 0.1, 0.3, 0.5, 0.77, 1.0):
        p = np.full(len(y), c)
        for edges in (cr.bin_edges_equal(10), cr.bin_edges_quantile(p, 5)):
            e, m = cr.ece_mce(y, p, edges)
            assert abs(e - abs(y.mean() - c)) < 1e-12, (c, e)
            assert abs(m - abs(y.mean() - c)) < 1e-12, (c, m)
        assert abs(cr.brier(y, p) - np.mean((c - y) ** 2)) < 1e-12


def test_determinism_evaluate():
    y, p, g = _synthetic(n=3000, seed=2)
    a = json.dumps(cr._round(cr.evaluate(y, p, g, n_boot=50)), sort_keys=True)
    b = json.dumps(cr._round(cr.evaluate(y, p, g, n_boot=50)), sort_keys=True)
    assert a == b
    ev = json.loads(a)
    assert "ci95" in ev and len(ev["ci95"]["ece_10"]) == 2 and len(ev["ci95"]["brier"]) == 2
    assert ev["ci95"]["ece_10"][0] <= ev["ci95"]["ece_10"][1]


def test_bins_consistency():
    y, p, _ = _synthetic(n=5000, seed=3)
    for edges in (cr.bin_edges_equal(10), cr.bin_edges_quantile(p, 5)):
        assert np.all(np.diff(edges) > 0)
        rows = cr.reliability_table(y, p, edges)
        assert sum(r["n"] for r in rows) == len(p)
        e, m = cr.ece_mce(y, p, edges)
        assert e <= m + 1e-12
    # ранговый скор со связями: квантильные границы не дублируются
    p_ties = np.repeat([0.2, 0.5, 0.9], 100)
    edges = cr.bin_edges_quantile(p_ties, 5)
    assert np.all(np.diff(edges) > 0) and edges[0] == 0.0 and edges[-1] == 1.0


def test_roc_auc_matches_sklearn():
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:  # noqa: BLE001
        return
    rng = np.random.default_rng(4)
    y = (rng.uniform(size=400) < 0.2).astype(int)
    s = np.round(rng.uniform(size=400) + 0.3 * y, 1)  # со связями
    assert abs(cr.roc_auc(y, s) - roc_auc_score(y, s)) < 1e-12
    assert cr.roc_auc(np.zeros(10, int), s[:10]) is None


def test_svg_is_valid_xml():
    y, p, g = _synthetic(n=2000, seed=5)
    ev = cr.evaluate(y, p, g, n_boot=0)
    svg = cr.reliability_svg([cr._panel("синтетика <тест> & проверка", ev, 0)], "Проверка SVG")
    root = ET.fromstring(svg.encode("utf-8"))
    assert root.tag.endswith("svg")
    assert "<script" not in svg and "xlink:href" not in svg
    for w in FORBIDDEN:
        assert w not in svg


def test_delivery_oof_if_present():
    needed = [ROOT / "models" / "oof_stacked_spine_sp_pos.csv", ROOT / "models" / "metrics_summary.json",
              ROOT / "data" / "geometry_features.csv", ROOT / "data" / "embeddings.npy",
              ROOT / "data" / "labels_for_embeddings.csv"]
    if not all(f.exists() for f in needed):
        return
    r1 = json.dumps(cr._round(cr.build_targets(20)), sort_keys=True)
    r2 = json.dumps(cr._round(cr.build_targets(20)), sort_keys=True)
    assert r1 == r2, "build_targets не детерминирован"
    res = json.loads(r1)
    assert ".dcm" not in r1 and "file_path" not in r1, "в JSON не должно быть путей к кадрам"
    for w in FORBIDDEN:
        assert w not in r1
    full = ROOT / "models" / "metrics_oof_full.json"
    if full.exists():
        m = json.load(open(full, encoding="utf-8"))
        for region in ("spine", "hip"):
            ours = res["regions"][region]["variants"]["final"]["roc_auc"]
            ref = m["by_region_binary"][region]["roc_auc"]
            assert abs(ours - ref) < 1e-6, (region, ours, ref)
            assert res["regions"][region]["variants"]["final"]["n_pos"] == m["by_region_binary"][region]["n_pos"]
    for crit, c in res["criteria"].items():
        if "p_cal" in c["variants"]:
            ev = c["variants"]["p_cal"]
            assert abs(ev["slope"] - 1.0) < 0.02 and abs(ev["intercept"]) < 0.02, (crit, ev["slope"], ev["intercept"])
        sc = c["variants"]["score"]
        assert 0.0 <= sc["ece_10"] <= 1.0 and 0.0 <= sc["brier"] <= 1.0
    for crit, u in res["uncertain_zone"].items():
        assert 0.0 <= u["share_uncertain"] <= 1.0
    # quality_prob согласован с классом: class=1 <=> prob >= 0.5 — доля class=1 равна массе бинов >= 0.5
    for region in ("spine", "hip"):
        ev = res["regions"][region]["variants"]["final"]
        upper = sum(b["n"] for b in ev["bins_equal_10"] if b["lo"] >= 0.5)
        assert abs(upper / ev["n"] - res["regions"][region]["share_class1"]) < 1e-5  # share_class1 округлён до 6 знаков


if __name__ == "__main__":
    t0 = time.time()
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}  ({time.time() - t0:.1f} с)")
    print(f"все проверки пройдены за {time.time() - t0:.1f} с")
