#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""regen_labels_full.py — перегенерация колонки region в data/labels_full.csv по анатомическому детектору стороны.

Зачем. data/labels_full.csv собран до К5, когда сторона бедра определялась плотностной эвристикой
(hip_side_by_density, ошибается примерно на четверти снимков). src/build_dataset.py уже исправлен (К5 п. 1:
hip_side_anatomical), но сам файл не пересобирался, потому что build_dataset пишет новые абсолютные пути и
может поменять порядок строк, а порядок строк labels_full.csv совпадает с порядком data/embeddings*.npy
(src/extract_all_features.py, src/embeddings.py). Этот скрипт меняет только то, что зависит от стороны:
region, applicable, quality_class, violation_list, rh_pos/rh_roi/lh_pos/lh_roi. Порядок строк, file_path,
sop_instance_uid и остальные колонки сохраняются байт в байт.

Сторона — src/build_dataset.hip_side_anatomical (та же функция, что в исправленном генераторе) на DICOM из
--dicom-root; ключ — путь от папки исследования. Разметка — лист «Калибровка» (формат организаторов).
Прежний файл хранится как data/labels_full_v1_density_side.csv (для анализа чувствительности к стороне).

  python tools/regen_labels_full.py --dicom-root ../dataset/Исследования --markup ../dataset/разметка.xlsx
  python tools/regen_labels_full.py ... --check      # только сверка, без записи (код 1 при расхождении)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

LABELS = ROOT / "data" / "labels_full.csv"
SIDE_COLS = {"right_hip": ("rh_pos", "rh_roi"), "left_hip": ("lh_pos", "lh_roi")}


def rebuild(df: pd.DataFrame, dicom_root: Path, markup: pd.DataFrame) -> pd.DataFrame:
    import pydicom
    from build_dataset import VIOLATION_NAMES, hip_side_anatomical
    from organizer_metrics import rel_key
    out = df.copy()
    for i, r in out.iterrows():
        if r["region"] not in SIDE_COLS:
            continue
        ds = pydicom.dcmread(str(dicom_root / rel_key(r["file_path"])), force=True)
        region, _src = hip_side_anatomical(ds.pixel_array, ds)
        lab = markup.loc[str(r["study"])]
        crits = {k: lab[k] for k in SIDE_COLS[region]}
        vals = [v for v in crits.values() if pd.notna(v)]
        out.at[i, "region"] = region
        for k in ("rh_pos", "rh_roi", "lh_pos", "lh_roi"):
            out.at[i, k] = crits.get(k, np.nan)
        if vals:
            out.at[i, "applicable"] = True
            out.at[i, "quality_class"] = 1.0 if any(v == 1 for v in vals) else 0.0
            out.at[i, "violation_list"] = "; ".join(VIOLATION_NAMES[k] for k, v in crits.items() if pd.notna(v) and v == 1)
        else:
            out.at[i, "applicable"] = False
            out.at[i, "quality_class"] = np.nan
            out.at[i, "violation_list"] = np.nan
    return out


def main(argv=None) -> int:
    from organizer_metrics import load_markup
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dicom-root", type=Path, default=ROOT.parent / "dataset" / "Исследования")
    ap.add_argument("--markup", type=Path, default=ROOT.parent / "dataset" / "разметка.xlsx")
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    df = pd.read_csv(a.labels, low_memory=False)
    new = rebuild(df, a.dicom_root, load_markup(a.markup))
    hip = df["region"].isin(list(SIDE_COLS))
    changed = int((df.loc[hip, "region"] != new.loc[hip, "region"]).sum())
    print(f"строк {len(df)}, бедро {int(hip.sum())}, сторона изменится у {changed}")
    print(new["region"].value_counts().to_string())
    if a.check:
        return 1 if changed else 0
    new.to_csv(a.labels, index=False)
    print(f"записано: {a.labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
