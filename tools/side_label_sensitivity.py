#!/usr/bin/env python3
"""Чувствительность метрик бедра к выбору стороны при переносе метки (К6-методология).

Что проверяется. Разметка задана по исследованию отдельно для правого и левого бедра
(`rh_pos/rh_roi`, `lh_pos/lh_roi`). Какому снимку какая сторона принадлежит, решает
детектор стороны (`src/hip_features.detect_hip_side`), и `src/extract_all_features.py`
переносит на снимок метку ОБНАРУЖЕННОЙ стороны (`hip_pos_c`, `hip_roi_c`). Значит часть
мишени обучения и оценки зависит от нашего же шага конвейера: если детектор ошибся, снимок
получает метку другой стороны. Скрипт измеряет, насколько это влияет на итоговые числа:
считает те же AUC и F1 на двух вариантах мишени —

  A. «сторона детектора» (как в поставке): метка стороны, которую определил детектор;
  B. «сторона старой плотностной эвристики»: метка той стороны, которая была записана в
     `data/labels_full.csv` до К5 (файл сохранён как `data/labels_full_v1_density_side.csv`).
     Это не сторона, указанная рентгенологом: в разметке организаторов стороны снимка нет,
     есть только колонки «правое / левое бедро» по исследованию. С 24.09 (A2) колонка region
     в `data/labels_full.csv` перегенерирована анатомическим детектором и совпадает с A.

Разность A − B — цена доверия детектору стороны; она публикуется в
`docs/METRICS_REPORT.md` (раздел «Чувствительность к переносу метки по стороне»).

Запуск: python tools/side_label_sensitivity.py [--json docs/metrics_side_label.json]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
THR_TIE_ATOL = 1e-9

CRITERIA = [
    # (критерий, файл OOF, колонка правой стороны, колонка левой стороны)
    ("hip_pos", "oof_stacked_hip_hip_pos.csv", "rh_pos", "lh_pos"),
    ("hip_roi", "oof_stacked_hip_hip_roi.csv", "rh_roi", "lh_roi"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "docs" / "metrics_side_label.json"))
    a = ap.parse_args()

    labels = pd.read_csv(ROOT / "data" / "labels_full_v1_density_side.csv", low_memory=False)
    summary = json.loads((ROOT / "models" / "metrics_summary.json").read_text(encoding="utf-8"))
    out = {"note": "A — метка стороны, определённой детектором (как в поставке); "
                   "B — метка стороны по старой плотностной эвристике (data/labels_full_v1_density_side.csv, колонка region до К5); это не сторона, указанная рентгенологом",
           "criteria": {}}

    for crit, fname, rcol, lcol in CRITERIA:
        oof = pd.read_csv(ROOT / "models" / fname)
        thr = float(summary["hip"][crit]["threshold"])
        side = oof[["file_path", "y_true", "oof_stacked", "study"]].copy()
        if "hip_side_detected" in oof.columns:
            side["side_detected"] = oof["hip_side_detected"]
        lab = labels[["file_path", "region", rcol, lcol]].rename(columns={"region": "region_label"})
        m = side.merge(lab, on="file_path", how="left")
        m["y_label_side"] = np.where(m["region_label"] == "right_hip", m[rcol], m[lcol])
        n_nan = int(m["y_label_side"].isna().sum())
        m = m.dropna(subset=["y_label_side"])
        flags = (m["oof_stacked"].to_numpy() >= thr - THR_TIE_ATOL).astype(int)
        res = {}
        for key, y in (("A_side_detected", m["y_true"].astype(int).to_numpy()),
                       ("B_side_label", m["y_label_side"].astype(int).to_numpy())):
            res[key] = {"n": int(len(y)), "n_pos": int(y.sum()),
                        "auc": float(roc_auc_score(y, m["oof_stacked"])),
                        "f1": float(f1_score(y, flags, zero_division=0))}
        diff = m[m["y_true"].astype(int) != m["y_label_side"].astype(int)]
        res["n_rows_without_label"] = n_nan
        res["disagreements"] = {"n_frames": int(len(diff)), "n_studies": int(diff["study"].nunique()),
                                "studies": sorted(diff["study"].astype(str).unique().tolist())}
        res["delta_auc"] = res["A_side_detected"]["auc"] - res["B_side_label"]["auc"]
        res["delta_f1"] = res["A_side_detected"]["f1"] - res["B_side_label"]["f1"]
        out["criteria"][crit] = res
        print(f"{crit}: A pos={res['A_side_detected']['n_pos']} AUC={res['A_side_detected']['auc']:.4f} "
              f"F1={res['A_side_detected']['f1']:.4f} | B pos={res['B_side_label']['n_pos']} "
              f"AUC={res['B_side_label']['auc']:.4f} F1={res['B_side_label']['f1']:.4f} | "
              f"расхождений {res['disagreements']['n_frames']} кадров в "
              f"{res['disagreements']['n_studies']} исследовании(ях)")

    Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("записано:", a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
