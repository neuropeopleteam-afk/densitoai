#!/usr/bin/env python3
"""Тест паспорта выборки: детерминизм и согласование с замороженными метриками.

Быстрый режим: 60 бутстрап-ресэмплов (около 12 с) (параметр n_boot), quality_prob области пересобирается
(любой сбой импорта не валит тест — в паспорте остаётся F1 области). Хэши пикселей берутся
из outputs/pixel_hashes.csv или пересчитываются по DICOM, если каталог данных доступен
(DENSITO_DATASET_DIR); без них уникальные кадры равны None, остальные проверки не зависят от хэшей.

Запуск: python tests/test_sample_passport.py   (или pytest -q tests/test_sample_passport.py)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "tools"))

import sample_passport as sp  # noqa: E402

N_BOOT_FAST = 60
DATASET_DIR = os.environ.get("DENSITO_DATASET_DIR")


_CACHE: dict = {}


def _build(fresh: bool = False):
    """Один паспорт на все проверки (кэш); fresh=True — независимый повторный расчёт для детерминизма."""
    if fresh or "p" not in _CACHE:
        p = sp.build_passport(ROOT, n_boot=N_BOOT_FAST, seed=sp.SEED,
                              dataset_dir=Path(DATASET_DIR) if DATASET_DIR else None)
        if fresh:
            return p
        _CACHE["p"] = p
    return _CACHE["p"]


def _dump(p):
    return json.dumps(p, ensure_ascii=False, sort_keys=True, allow_nan=True)


def test_deterministic():
    a, b = _build(), _build(fresh=True)
    assert _dump(a) == _dump(b), "паспорт не детерминирован при одинаковом seed"


def test_points_match_frozen_metrics():
    p = _build()
    summary = json.loads((ROOT / "models" / "metrics_summary.json").read_text(encoding="utf-8"))
    for section, crit, _ in sp.PLACES:
        node = summary.get(section, {}).get(crit)
        if node is None:
            continue
        c = p["criteria"][crit]
        assert c["files"] == node["n_valid"], crit
        assert c["pos_files"] == node["n_pos"], crit
        assert abs(c["point"]["f1"] - node["f1_oof"]) <= sp.TOL_POINT, crit
        # AUC: совпадает либо с metrics_summary.json, либо с metrics_oof_full.json (описанное расхождение sp_art)
        chk = c["check"]
        assert chk["auc_ok"] or chk.get("auc_ok_oof_full"), (crit, chk["auc_diff"])
    # Пять критериев ТЗ: замороженный ДИ F1 (2000 ресэмплов) должен лежать рядом с быстрым (60 ресэмплов)
    for crit in ("sp_pos", "sp_axis", "sp_art", "hip_pos", "hip_roi"):
        lo, hi = p["criteria"][crit]["frozen"]["f1_ci"]
        mlo, mhi = p["criteria"][crit]["ci_by_study"]["f1_ci"]
        assert abs(lo - mlo) < 0.15 and abs(hi - mhi) < 0.15, (crit, (lo, hi), (mlo, mhi))
    assert p["all_points_match"], p["discrepancies"]


def test_structure_and_sanity():
    p = _build()
    t = p["total"]
    assert t["files"] == 499 and t["studies"] == 100
    assert t["studies_both_regions"] + t["studies_spine_only"] + t["studies_hip_only"] == t["studies"]
    assert t["hip_frames_by_label_side"]["right"] + t["hip_frames_by_label_side"]["left"] == \
        t["hip_frames_by_detected_side"]["right"] + t["hip_frames_by_detected_side"]["left"]
    if t["unique_frames"] is not None:
        assert t["unique_frames"] == 252 and t["hash_groups_spanning_studies"] == 0
    for crit, c in p["criteria"].items():
        de = c["design_effect"]
        assert 0.0 <= de["icc"] <= 1.0, crit
        assert de["deff"] >= 1.0 and c["studies"] <= de["n_eff"] + 1e-9 <= c["files"] + 1e-9, crit
        assert c["pos_studies"] <= c["pos_files"], crit
        cc = c["concentration"]
        assert 1 <= cc["studies_for_50pct"] <= cc["studies_for_80pct"] <= c["pos_studies"], crit
        # кластерные интервалы не уже наивных по файлам (по ширине; на малом числе ресэмплов — с запасом)
        a, b = c["ci_by_study"], c["ci_by_file"]
        assert (a["prevalence_ci"][1] - a["prevalence_ci"][0]) >= 0.8 * (b["prevalence_ci"][1] - b["prevalence_ci"][0]), crit
    for region in ("spine", "hip"):
        r = p["region_binary"][region]
        assert r["check"]["f1_ok"], region
    md = sp.to_markdown(p)
    assert "Паспорт выборки" in md and "`sp_pos`" in md


if __name__ == "__main__":
    t0 = time.time()
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}  ({time.time() - t0:.1f} с)")
    print(f"все проверки пройдены за {time.time() - t0:.1f} с")
