#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тест tools/organizer_metrics.py и согласованности стороны бедра (A2).

Проверяется:
  1. синтетика: F1 по типам, три трактовки macro-F1 (по областям, по 5 критериям, по 4 строкам словаря с общей
     «Некорректной укладкой»), уровень визита, дедупликация по хэшу пикселей;
  2. разбор violation_type: строки словаря через «;», чужая область не засчитывается;
  3. режим --oof воспроизводит models/metrics_oof_full.json (docs/METRICS_REPORT.md) по областям до 5e-4,
     метки разметки по стороне детектора совпадают с y_true OOF во всех ячейках; известные свойства:
     у sp_pos 18 флагов при 10 позитивах, 10 строк ровно на пороге — один кадр, все без нарушения;
  4. сторона в data/labels_full.csv = hip_side_detected в OOF и в data/geometry_features.csv (все 333 / 329 кадра),
     метки rh_/lh_ в labels_full соответствуют y_true OOF; порядок строк и file_path совпадают с
     data/labels_full_v1_density_side.csv (порядок эмбеддингов не сломан), отличаются только колонки стороны;
  5. режим --results на выгрузке сервиса (если есть ../work_2409/reg_local_baseline/results.csv) отрабатывает,
     все 499 строк прочитаны, размечено 495;
  6. models/metrics_summary.json: nested_auc_mean = nested_auc_production у всех пяти критериев, как и в генераторе
     src/train_stacked.nested_block_for; альтернатива — только в nested_auc_alternative*;
  7. запрещённые слова отсутствуют в выводе.

Запуск: python tests/test_organizer_metrics.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import organizer_metrics as om  # noqa: E402

FORBIDDEN = ["Grad" + "-CAM", "автокоррекц", "ЕР" + "ИС", "сколи" + "оз", "Коб" + "ба"]
MARKUP = ROOT.parent / "dataset" / "разметка.xlsx"
BASELINE = ROOT.parent / "work_2409" / "reg_local_baseline" / "results.csv"
_CACHE = {}


def _synthetic():
    rows = []
    # исследование A: позвоночник с sp_pos, два клона бедра (правое) с hip_pos; исследование B: всё норма
    def r(study, key, region, side, h, qc, qp, flags, ys):
        d = {"key": key, "study": study, "region": region, "side": side, "pixel_hash": h, "quality_class": qc,
             "quality_prob": qp, "status": "Success"}
        for c in om.CRITS:
            d[f"f_{c}"] = flags.get(c, 0)
            d[f"y_{c}"] = ys.get(c, np.nan)
        return d
    rows.append(r("A", "A/1", "spine", None, "h1", 1, 0.9, {"sp_pos": 1}, {"sp_pos": 1, "sp_axis": 0, "sp_art": 0}))
    rows.append(r("A", "A/2", "hip", "right", "h2", 1, 0.8, {"hip_pos": 1}, {"hip_pos": 1, "hip_roi": 0}))
    rows.append(r("A", "A/3", "hip", "right", "h2", 1, 0.8, {"hip_pos": 1}, {"hip_pos": 1, "hip_roi": 0}))
    rows.append(r("B", "B/1", "spine", None, "h3", 1, 0.7, {"sp_axis": 1}, {"sp_pos": 0, "sp_axis": 0, "sp_art": 0}))
    rows.append(r("B", "B/2", "hip", "left", "h4", 0, 0.2, {}, {"hip_pos": 0, "hip_roi": 1}))
    df = pd.DataFrame(rows)
    ys = df[[f"y_{c}" for c in om.CRITS]]
    df["labelled"] = ys.notna().any(axis=1)
    df["y_bin"] = (ys.fillna(0).max(axis=1) > 0).astype(int)
    return df


def test_synthetic_three_macro_readings():
    df = _synthetic()
    m = om.point_metrics(df)
    assert m["f1/sp_pos"] == 1.0 and m["f1/sp_axis"] == 0.0 and m["f1/sp_art"] == 0.0
    assert m["f1/hip_pos"] == 1.0 and m["f1/hip_roi"] == 0.0
    assert abs(m["macro/spine"] - 1 / 3) < 1e-12 and abs(m["macro/hip"] - 0.5) < 1e-12
    assert abs(m["macro/5_criteria"] - 0.4) < 1e-12
    # общая «Некорректная укладка»: TP 3, FP 0, FN 0 -> 1.0; остальные три строки словаря 0 -> среднее 0.25
    assert abs(m["macro/4_dictionary_rows"] - 0.25) < 1e-12
    v = om.visit_table(df)
    assert v.loc["A", "y"] == 1 and v.loc["B", "y"] == 1 and v.loc["B", "p"] == 1
    u = df.drop_duplicates("pixel_hash")
    assert len(u) == 4 and om.binary_block(u[u.region == "hip"].y_bin, u[u.region == "hip"].quality_class)["tp"] == 1


def test_parse_flags():
    f = om.parse_flags("spine", "Некорректная укладка;Присутствуют посторонние предметы")
    assert f == {"sp_pos": 1, "sp_axis": 0, "sp_art": 1}
    assert om.parse_flags("hip", "Некорректная область интереса") == {"hip_pos": 0, "hip_roi": 1}
    assert om.parse_flags("hip", "Не выравнена ось позвоночника") == {"hip_pos": 0, "hip_roi": 0}
    assert om.parse_flags("spine", "") == {"sp_pos": 0, "sp_axis": 0, "sp_art": 0}
    assert om.rel_key("/x/Исследования/1.2/s/CR DXA/CR000000.dcm") == "1.2/s/CR DXA/CR000000.dcm"


def _oof():
    if "oof" not in _CACHE:
        with tempfile.TemporaryDirectory() as d:
            code = om.main(["--oof", "--markup", str(MARKUP), "--boot", "30", "--out-json", f"{d}/o.json",
                            "--out-md", f"{d}/o.md"])
            _CACHE["oof"] = (code, json.loads(Path(f"{d}/o.json").read_text(encoding="utf-8")),
                             Path(f"{d}/o.md").read_text(encoding="utf-8"))
    return _CACHE["oof"]


def test_oof_reproduces_metrics_report():
    if not MARKUP.exists():
        print("  (пропуск: нет разметки)"); return
    code, res, _ = _oof()
    assert code == 0
    assert res["oof_check"]["ok"] and res["oof_check"]["max_abs_diff"] < 5e-4
    assert res["oof_label_mismatch_vs_markup"] == 0
    p = res["files"]["point"]
    assert abs(p["macro/5_criteria"] - 0.517) < 5e-4 and abs(p["macro/4_dictionary_rows"] - 0.524) < 5e-4
    assert res["files"]["n_rows"] == 495 and res["n_unique_frames"] == 249
    tb = res["tie_blocks"]["sp_pos"]
    assert tb["flags_total"] == 18 and tb["positives_total"] == 10
    assert tb["rows_on_threshold"] == 10 and tb["rows_on_threshold_positive"] == 0 and tb["unique_frames_on_threshold"] == 1
    v = res["files"]["visit"]
    assert v["n"] == 100 and v["n_pos"] == 48 and abs(v["sensitivity"] - 0.729) < 5e-4 and abs(v["specificity"] - 0.5) < 1e-9


def test_side_in_labels_equals_side_in_oof():
    lab = pd.read_csv(ROOT / "data" / "labels_full.csv", low_memory=False)
    old = pd.read_csv(ROOT / "data" / "labels_full_v1_density_side.csv", low_memory=False)
    assert list(lab["file_path"]) == list(old["file_path"]), "порядок строк labels_full.csv сохранён"
    same_cols = [c for c in lab.columns if c not in ("region", "applicable", "quality_class", "violation_list",
                                                     "rh_pos", "rh_roi", "lh_pos", "lh_roi")]
    assert lab[same_cols].equals(old[same_cols])
    assert (lab["region"] == "spine").equals(old["region"] == "spine")
    hip = lab[lab["region"].isin(["right_hip", "left_hip"])]
    assert len(hip) == 333
    geom = pd.read_csv(ROOT / "data" / "geometry_features.csv", low_memory=False, usecols=["file_path", "hip_side_detected"])
    g = hip.merge(geom, on="file_path", how="left")
    assert (g["region"].str.replace("_hip", "") == g["hip_side_detected"]).all()
    for crit, rc, lc in (("hip_pos", "rh_pos", "lh_pos"), ("hip_roi", "rh_roi", "lh_roi")):
        o = pd.read_csv(ROOT / "models" / f"oof_stacked_hip_{crit}.csv")
        m = o.merge(hip, on="file_path", how="left", validate="one_to_one")
        assert len(m) == 329 and m["region"].notna().all()
        assert (m["region"].str.replace("_hip", "") == m["hip_side_detected"]).all(), "сторона labels = сторона OOF"
        y = np.where(m["region"] == "right_hip", m[rc], m[lc])
        assert (y.astype(int) == m["y_true"].astype(int)).all(), f"метка {crit} labels = y_true OOF"
        assert m.loc[m["region"] == "left_hip", rc].isna().all() and m.loc[m["region"] == "right_hip", lc].isna().all()


def test_results_mode_on_service_csv():
    if not (BASELINE.exists() and MARKUP.exists()):
        print("  (пропуск: нет выгрузки сервиса)"); return
    with tempfile.TemporaryDirectory() as d:
        code = om.main(["--results", str(BASELINE), "--markup", str(MARKUP), "--boot", "0", "--out-json", f"{d}/r.json"])
        res = json.loads(Path(f"{d}/r.json").read_text(encoding="utf-8"))
    assert code == 0 and res["n_rows_input"] == 499 and res["n_rows_labelled"] == 495
    assert res["meta"]["in_sample_warning"] is True
    assert 0 <= res["files"]["point"]["bin_auc/pooled"] <= 1


def test_nested_fields_are_production():
    """nested_auc_mean = nested_auc_production у всех пяти критериев в файле и в генераторе (train_stacked)."""
    import refresh_nested_fields as rn
    import train_stacked as ts
    d = json.loads((ROOT / "models" / "metrics_summary.json").read_text(encoding="utf-8"))
    assert rn.consistent(d) == []
    for region, crit in rn.CRITS:
        b = d[region][crit]
        nb = ts.nested_block_for(crit, str(b.get("emb_source") or "imagenet"), ts.nested_decision_for(crit))
        assert nb["mean"] == nb["production"] == b["nested_auc_production"] == b["nested_auc_mean"], crit
        assert nb["ci"] == b["nested_auc_ci"] and nb["alternative_ci"] == b["nested_auc_alternative_ci"], crit
        assert b["nested_auc_alternative"] != b["nested_auc_production"], crit


def test_no_forbidden_words():
    if not MARKUP.exists():
        return
    _, res, md = _oof()
    txt = md + json.dumps(res, ensure_ascii=False)
    for w in FORBIDDEN:
        assert w.lower() not in txt.lower(), w


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
