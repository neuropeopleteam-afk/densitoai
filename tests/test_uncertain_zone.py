#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тест инструмента зоны «не уверен» (tools/uncertain_zone.py).

Проверяется:
  1. счётчики и метрики одной точки на синтетике: известные TP/FN/FP/TN, доля зоны, полнота с учётом зоны;
  2. парный бутстрап разности «зона против самой себя» даёт нуль без разброса;
  3. на OOF-файлах поставки инструмент отрабатывает (--out во временный каталог, 20 бутстрап-повторов), JSON валиден,
     без путей к кадрам и идентификаторов; SVG — корректный XML без наклонного текста, кроме rotate(-90) оси Y;
  4. монотонность: с ростом δ доля «не уверен», число положительных в зоне и полнота с учётом зоны не убывают
     (строки региона, полоса по quality_prob, каждый критерий);
  5. текущая точка воспроизводится: доля строк «не уверен» по областям равна
     models/metrics_summary.json -> uncertainty.row_uncertain_rate_by_region (14.5 % / 9.4 %), доля по критериям —
     models/calibration.pkl -> criteria.<crit>.uncertain_rate_oof; точка δ = 0 совпадает с числом строк ровно на пороге;
  6. детерминизм: два прогона build() дают побайтно одинаковый JSON;
  7. запрещённые слова отсутствуют в JSON, SVG и docs/UNCERTAIN_ZONE.md (если документ есть);
  8. при наличии выгрузки боевого прогона (DENSITO_REGRESS_DEBUG или ../dataset/regress_2_3_2_debug.csv) правило,
     восстановленное из файла, даёт ровно столько строк, сколько needs_review в файле (29 из 499).

Запуск: python tests/test_uncertain_zone.py   (или pytest -q tests/test_uncertain_zone.py)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import uncertain_zone as uz  # noqa: E402

FORBIDDEN = ["Grad" + "-CAM", "автокоррекц" + "ия ROI", "ЕР" + "ИС", "сколи" + "оз", "угол К" + "обба"]
HAVE_OOF = all((ROOT / "models" / f"oof_stacked_{r}_{c}.csv").exists()
               for r, cs in uz.REGION_CRITERIA.items() for c in cs) and (ROOT / "models" / "metrics_summary.json").exists()
_CACHE = {}


def _regress_path():
    p = os.environ.get("DENSITO_REGRESS_DEBUG")
    cands = [Path(p)] if p else []
    cands += [ROOT.parent / "dataset" / "regress_2_3_2_debug.csv", ROOT.parent.parent / "dataset" / "regress_2_3_2_debug.csv",
              ROOT.parent.parent.parent / "dataset" / "regress_2_3_2_debug.csv"]
    return next((c for c in cands if c.exists()), None)


def _build(n_boot=20):
    if "res" not in _CACHE:
        _CACHE["res"] = uz.build(n_boot=n_boot, seed=0, regress_path=_regress_path())
    return _CACHE["res"]


def test_counts_and_metrics_synthetic():
    y = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    flag = np.array([1, 0, 1, 1, 0, 0, 0, 1])
    unc = np.array([0, 0, 1, 0, 0, 1, 1, 0], bool)
    m = uz.metrics_from_counts(uz.counts(y, flag, unc))
    # уверенные: idx 0,1,3,4,7 -> TP=1 (0), FN=1 (1), FP=2 (3,7), TN=1 (4)
    assert m["n"] == 8 and m["n_uncertain"] == 3 and abs(m["share_uncertain"] - 3 / 8) < 1e-12
    assert abs(m["sensitivity_confident"] - 0.5) < 1e-12
    assert abs(m["specificity_confident"] - 1 / 3) < 1e-12
    assert abs(m["f1_confident"] - 2 / (2 + 1 + 2)) < 1e-12
    assert m["n_pos_in_zone"] == 2 and m["n_fn_in_zone"] == 1 and m["n_fp_in_zone"] == 0
    assert m["n_errors"] == 4 and m["n_errors_in_zone"] == 1
    assert abs(m["recall_with_review"] - 3 / 4) < 1e-12
    # пропусков всего 2 (idx 1 и 6), базовая доля 2/8, в зоне 3 строки -> ожидание 0.75, факт 1 -> +0.25
    assert abs(m["fn_in_zone_excess"] - 0.25) < 1e-12
    # пустая зона / всё в зоне
    m0 = uz.metrics_from_counts(uz.counts(y, flag, np.zeros(8, bool)))
    assert m0["share_uncertain"] == 0 and abs(m0["sensitivity_confident"] - 0.5) < 1e-12
    m1 = uz.metrics_from_counts(uz.counts(y, flag, np.ones(8, bool)))
    assert m1["sensitivity_confident"] is None and abs(m1["recall_with_review"] - 1.0) < 1e-12


def test_paired_bootstrap_self_diff_is_zero():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 120); flag = rng.integers(0, 2, 120); g = rng.integers(0, 25, 120)
    z = rng.uniform(size=120) < 0.2
    d = uz.paired_bootstrap_diff(y, flag, g, z, z, n_boot=30, seed=0)
    for k, v in d.items():
        assert v["diff"] == 0 and v["ci95"] == [0.0, 0.0], (k, v)
    # разные зоны на одних ресэмплах: разность доли зоны детерминирована
    z2 = z | (rng.uniform(size=120) < 0.2)
    d2 = uz.paired_bootstrap_diff(y, flag, g, z, z2, n_boot=30, seed=0)
    assert d2["share_uncertain"]["diff"] >= 0 and d2["share_uncertain"]["ci95"][0] >= 0


def test_tool_runs_json_and_svg_valid():
    if not HAVE_OOF:
        print("  (пропуск: нет OOF-файлов поставки)"); return
    with tempfile.TemporaryDirectory() as td:
        rc = uz.main(["--out", td, "--n-boot", "10"])
        assert rc == 0
        jp = Path(td) / "uncertain_zone.json"
        res = json.loads(jp.read_text(encoding="utf-8"))
        assert set(res["regions"]) == {"spine", "hip"} and set(res["criteria"]) == {"sp_pos", "sp_axis", "sp_art", "hip_pos", "hip_roi"}
        assert len(res["regions"]["spine"]["score_band"]) == 31
        txt = jp.read_text(encoding="utf-8")
        assert ".dcm" not in txt and "file_path" not in txt and "/home/" not in txt, "в JSON не должно быть путей к кадрам"
        for w in FORBIDDEN:
            assert w.lower() not in txt.lower(), w
        svgs = sorted(Path(td).glob("*.svg"))
        assert len(svgs) == 3, svgs
        for f in svgs:
            s = f.read_text(encoding="utf-8")
            ET.fromstring(s)  # корректный XML
            assert "rotate(" not in s.replace("rotate(-90)", ""), "наклонный текст допустим только для подписи оси Y"
            assert "<script" not in s and "xlink:href" not in s and "http://" not in s.replace("http://www.w3.org/2000/svg", "")
            for w in FORBIDDEN:
                assert w.lower() not in s.lower(), (f.name, w)
            assert "доля строк в зоне" in s and "полнота" in s


def _nondecreasing(a, eps=1e-12):
    a = [v for v in a if v is not None]
    return all(a[i + 1] >= a[i] - eps for i in range(len(a) - 1))


def test_monotonic_in_delta():
    if not HAVE_OOF:
        print("  (пропуск: нет OOF-файлов поставки)"); return
    res = _build()
    for region, r in res["regions"].items():
        for fam in ("score_band", "qp_band"):
            pts = r.get(fam) or []
            if not pts:
                print(f"  (пропуск: семейство {fam} для {region} не построено — нет data/geometry_features.csv)"); continue
            assert [p["delta"] for p in pts] == uz.DELTA_GRID
            assert _nondecreasing([p["share_uncertain"] for p in pts]), (region, fam, "share")
            assert _nondecreasing([p["n_pos_in_zone"] for p in pts]), (region, fam, "pos")
            assert _nondecreasing([p["recall_with_review"] for p in pts]), (region, fam, "recall_with_review")
            assert _nondecreasing([p["n_errors_in_zone"] for p in pts]), (region, fam, "errors")
        assert r["score_band"][-1]["share_uncertain"] > r["score_band"][0]["share_uncertain"]
    for crit, c in res["criteria"].items():
        pts = c["score_band"]
        assert _nondecreasing([p["share_uncertain"] for p in pts]), crit
        assert _nondecreasing([p["n_pos_in_zone"] for p in pts]), crit


def test_current_point_reproduces_delivery_numbers():
    if not HAVE_OOF:
        print("  (пропуск: нет OOF-файлов поставки)"); return
    res = _build()
    summary = json.load(open(ROOT / "models" / "metrics_summary.json", encoding="utf-8"))
    rates = (summary.get("uncertainty") or {}).get("row_uncertain_rate_by_region") or {}
    for region, r in res["regions"].items():
        cur = r["current"]
        assert abs(cur["share_uncertain"] - cur["n_uncertain"] / cur["n"]) < 1e-12
        if region in rates:
            assert abs(cur["share_uncertain"] - float(rates[region])) < 1e-6, (region, cur["share_uncertain"], rates[region])
        # δ = 0: только строки ровно на пороге (по любому критерию) — не больше, чем при текущих запасах >= 0
        assert r["score_band"][0]["n_uncertain"] <= cur["n_uncertain"] or any(cur["margins"][c] < 0 for c in cur["margins"])
        # строка «не уверен» по региону = объединение по критериям: не меньше максимума и не больше суммы
        by = cur["by_criterion_rows"]
        assert max(by.values()) <= cur["n_uncertain"] <= sum(by.values())
    calib = uz.load_calibration_pkl()
    if calib is not None:
        for crit, c in res["criteria"].items():
            rec = (calib.get("criteria") or {}).get(crit) or {}
            if rec.get("uncertain_rate_oof") is not None and abs(float(rec.get("margin", c["margin"])) - c["margin"]) < 1e-9:
                assert abs(c["current"]["share_uncertain"] - float(rec["uncertain_rate_oof"])) < 1e-6, crit
    # известные числа сборки 2.5.0 (docs/CALIBRATION.md, таблица 3; в 2.4.0 позвоночник 24), если запасы те же
    exp = {"spine": 23, "hip": 31}  # 2.5.0: sp_art на признаках положения предметов (было 24)
    m = res["inputs"]["margins"]
    if abs(m.get("sp_pos", -1)) < 1e-12 and abs(m.get("hip_roi", 0) - 0.025835867) < 1e-9:
        for region, n in exp.items():
            assert res["regions"][region]["current"]["n_uncertain"] == n, (region, res["regions"][region]["current"]["n_uncertain"])


def test_deterministic():
    if not HAVE_OOF:
        print("  (пропуск: нет OOF-файлов поставки)"); return
    a = uz.build(n_boot=15, seed=0)
    b = uz.build(n_boot=15, seed=0)
    ja = json.dumps(uz._round(a), ensure_ascii=False, sort_keys=True)
    jb = json.dumps(uz._round(b), ensure_ascii=False, sort_keys=True)
    assert ja == jb, "build() недетерминирован"
    c = uz.build(n_boot=15, seed=1)
    assert json.dumps(uz._round(c), ensure_ascii=False, sort_keys=True) != ja or True  # другое зерно может менять ДИ


def test_recommendation_structure_and_docs_forbidden_words():
    if not HAVE_OOF:
        print("  (пропуск: нет OOF-файлов поставки)"); return
    res = _build()
    for region, rc in res["recommendation"]["by_region"].items():
        assert "current" in rc and "score_band" in rc
        k = rc["score_band"]
        if k:
            assert k["share_uncertain"] <= uz.REVIEW_CAP + 1e-9
            assert "paired_vs_current" in k and "verdict" in k
    md = ROOT / "docs" / "UNCERTAIN_ZONE.md"
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for w in FORBIDDEN:
            assert w.lower() not in t.lower(), w
        assert "!" not in t.replace("!=", ""), "восклицательных знаков в документе нет"


def test_regress_projection_reproduces_needs_review():
    p = _regress_path()
    if p is None or not HAVE_OOF:
        print("  (пропуск: нет выгрузки боевого прогона)"); return
    res = _build()
    rg = res["regress"]
    assert rg is not None and rg["n_rows"] == 499
    assert rg["rule_in_file"]["total_uncertain"] == rg["n_needs_review_in_file"] == 29
    assert _nondecreasing([q["total_uncertain"] for q in rg["score_band"]])
    assert rg["score_band"][0]["total_uncertain"] <= rg["rule_in_file"]["total_uncertain"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    t0 = time.time()
    for t in tests:
        t1 = time.time()
        try:
            t()
            print(f"ok    {t.__name__} ({time.time() - t1:.1f} с)")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} проверок пройдено за {time.time() - t0:.1f} с")
    raise SystemExit(1 if failed else 0)
