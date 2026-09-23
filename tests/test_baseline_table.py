#!/usr/bin/env python3
"""Тесты таблицы бейзлайнов (tools/baseline_table.py). Запуск: python tests/test_baseline_table.py (без pytest).

Проверяется:
  1. детерминизм — два вызова build() с одинаковыми параметрами дают побайтово одинаковый JSON;
  2. воспроизведение замороженных чисел стека: AUC/F1 по пяти критериям совпадают с
     models/metrics_summary.json в пределах 1e-3, с models/metrics_oof_full.json — в пределах 1e-6,
     число помеченных кадров совпадает с n_flag_oof; F1 бинарной задачи по области совпадает;
  3. тривиальные бейзлайны: ROC-AUC ровно 0.5, F1 «всегда норма» = 0, F1 «всегда нарушение» = 2p/(1+p),
     PR-AUC константного скора равен доле позитивов; порог по правилу воспроизводит порог поставки;
  4. документ docs/BASELINES.md и docs/baselines.json согласованы с текущим расчётом (если файлы есть).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

import baseline_table as bt  # noqa: E402

CRITERIA = [("spine", "sp_pos"), ("spine", "sp_axis"), ("spine", "sp_art"), ("hip", "hip_pos"), ("hip", "hip_roi")]
N_BOOT_TEST, N_RANDOM_TEST = 20, 50


def _build():
    return bt.build(ROOT, n_boot=N_BOOT_TEST, seed_boot=0, n_random=N_RANDOM_TEST, with_simple=True)


def test_determinism():
    a = json.dumps(_build(), sort_keys=True, ensure_ascii=False)
    b = json.dumps(_build(), sort_keys=True, ensure_ascii=False)
    assert a == b, "два запуска build() дали разный JSON"


def test_frozen_reproduced(p):
    summary = json.loads((ROOT / "models" / "metrics_summary.json").read_text(encoding="utf-8"))
    full = json.loads((ROOT / "models" / "metrics_oof_full.json").read_text(encoding="utf-8"))
    for region, crit in CRITERIA:
        node = summary[region][crit]
        st = p["criteria"][crit]["rows"]["stack"]
        assert abs(st["roc_auc"] - node["auc_stacked"]) <= 1e-3, (crit, st["roc_auc"], node["auc_stacked"])
        assert abs(st["f1"] - node["f1_oof"]) <= 1e-3, (crit, st["f1"], node["f1_oof"])
        assert st["n_flag"] == node["n_flag_oof"], (crit, st["n_flag"], node["n_flag_oof"])
        assert p["criteria"][crit]["n"] == node["n_valid"] and p["criteria"][crit]["n_pos"] == node["n_pos"]
        fv = full["by_violation_type"][f"{region}/{crit}"]
        for k_here, k_full in (("roc_auc", "roc_auc"), ("f1", "f1"), ("pr_auc", "pr_auc"), ("balanced_accuracy", "balanced_accuracy")):
            assert abs(st[k_here] - fv[k_full]) <= 1e-6, (crit, k_here, st[k_here], fv[k_full])
        # контуры в отдельности — auc_geom / auc_emb поставки
        assert abs(p["criteria"][crit]["rows"]["contour_a"]["roc_auc"] - node["auc_geom"]) <= 1e-6, crit
        assert abs(p["criteria"][crit]["rows"]["contour_b"]["roc_auc"] - node["auc_emb"]) <= 1e-6, crit
        # правило порога воспроизводит порог поставки
        assert abs(st["threshold_by_rule_recomputed"] - node["threshold"]) <= 1e-9, crit
    for region in ("spine", "hip"):
        rb = p["region_binary"][region]
        fb = full["by_region_binary"][region]
        assert rb["n"] == fb["n"] and rb["n_pos"] == fb["n_pos"], region
        assert abs(rb["rows"]["stack"]["f1"] - fb["f1"]) <= 1e-6, region
        assert abs(rb["rows"]["stack"]["roc_auc"] - fb["roc_auc_components"]["max_criteria_only"]) <= 1e-6, region
    assert p["all_frozen_ok"] is True
    # известное расхождение sp_art описано, а не подогнано: metrics_summary 0.8227 против OOF 0.8237
    d = p["criteria"]["sp_art"]["check"]["auc_stack_diff"]
    assert 5e-4 < d < 1e-3, d


def test_trivial_baselines(p):
    tasks = list(p["criteria"].values()) + list(p["region_binary"].values())
    for rec in tasks:
        prev = rec["n_pos"] / rec["n"]
        an, av, rnd = rec["rows"]["always_normal"], rec["rows"]["always_violation"], rec["rows"]["random_prevalence"]
        assert an["roc_auc"] == 0.5 and av["roc_auc"] == 0.5 and rnd["roc_auc"] == 0.5
        assert an["f1"] == 0.0 and an["n_flag"] == 0
        assert abs(av["f1"] - 2 * prev / (1 + prev)) <= 1e-12 and av["n_flag"] == rec["n"]
        assert abs(an["pr_auc"] - prev) <= 1e-12 and abs(av["pr_auc"] - prev) <= 1e-12
        assert abs(an["balanced_accuracy"] - 0.5) <= 1e-12 and abs(av["balanced_accuracy"] - 0.5) <= 1e-12
        assert 0.4 <= rnd["roc_auc_mc"] <= 0.6, rnd["roc_auc_mc"]
        assert an["roc_auc_ci"] == {"lo": 0.5, "hi": 0.5, "share_skipped": 0.0}
    # быстрые метрики для розыгрышей совпадают с sklearn
    rng = np.random.default_rng(3)
    y = (rng.random(150) < 0.2).astype(int); s = np.round(rng.random(150), 2); f = (s >= 0.8).astype(int)
    a, b = bt.fast_metrics(y, s, f), bt.point_metrics(y, s, f)
    for k in ("f1", "balanced_accuracy", "roc_auc", "pr_auc"):
        assert abs(a[k] - b[k]) <= 1e-12, k


def test_simple_model_present(p):
    for _, crit in CRITERIA:
        r = p["criteria"][crit]["rows"]["simple_all_geom"]
        assert "skipped" not in r, crit
        assert r["n_features"] >= 10 and 0.0 <= r["roc_auc"] <= 1.0
    feats = p["simple_model_features"]
    assert "synth_pos_logit" not in feats["spine"] and "hip_side_score" not in feats["hip"]
    assert "axis_angle_deg" in feats["spine"] and "scan_length_mm" in feats["hip"]


def test_docs_consistent(p):
    """docs/baselines.json (полный прогон) совпадает по точечным значениям с быстрым расчётом; markdown — из того же JSON."""
    jp, mp = ROOT / "docs" / "baselines.json", ROOT / "docs" / "BASELINES.md"
    if not (jp.exists() and mp.exists()):
        print("  (docs/baselines.json нет — пропуск сверки с документом)")
        return
    doc = json.loads(jp.read_text(encoding="utf-8"))
    for crit, rec in p["criteria"].items():
        for key in ("always_normal", "always_violation", "contour_a", "contour_b", "simple_all_geom", "stack"):
            for k in ("roc_auc", "f1", "pr_auc", "balanced_accuracy"):
                assert abs(rec["rows"][key][k] - doc["criteria"][crit]["rows"][key][k]) <= 1e-9, (crit, key, k)
    md = mp.read_text(encoding="utf-8")
    assert bt.to_markdown(doc) == md, "docs/BASELINES.md не соответствует docs/baselines.json — перезапустите tools/baseline_table.py"
    for bad in ("Grad" + "-CAM", "автокор" + "рекция ROI", "ЕР" + "ИС", "сколи" + "оз", "угол Ко" + "бба", "!"):
        assert bad not in md, bad


if __name__ == "__main__":
    print("test_determinism ...", end=" ", flush=True); test_determinism(); print("ok")
    P = _build()
    for name, fn in (("test_frozen_reproduced", test_frozen_reproduced), ("test_trivial_baselines", test_trivial_baselines),
                     ("test_simple_model_present", test_simple_model_present), ("test_docs_consistent", test_docs_consistent)):
        print(f"{name} ...", end=" ", flush=True); fn(P); print("ok")
    print("OK: 5 тестов")
