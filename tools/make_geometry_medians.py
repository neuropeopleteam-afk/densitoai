#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Медианы геометрических признаков для импутации NaN -> models/geometry_medians.json.

Зачем: инференсу от `data/geometry_features.csv` нужны только медианы столбцов признаков.
Сам CSV — выгрузка обучающего набора заказчика: 499 строк с StudyInstanceUID,
SOPInstanceUID, путями к файлам и экспертными метками. В образ он попадать не должен.
Скрипт считает медианы тем же способом, что `DensitoInference._load_medians`
(`np.nanmedian` по столбцу), и пишет их в маленький JSON, который и едет в образ.

Запуск:
  python tools/make_geometry_medians.py            # записать models/geometry_medians.json
  python tools/make_geometry_medians.py --check    # сверить JSON с CSV (код 0/1)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "geometry_features.csv"
OUT = ROOT / "models" / "geometry_medians.json"


def feature_cols() -> list[str]:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    return sorted({c for cs in cfg["geometry_cols"].values() for c in cs})


def compute() -> dict:
    cols = feature_cols()
    df = pd.read_csv(CSV, usecols=lambda c: c in cols or c == "region")
    med = {}
    for c in cols:
        if c in df:
            med[c] = float(np.nanmedian(df[c].values.astype(np.float64)))
    return med


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if not CSV.exists():
        print(f"нет {CSV} — медианы считать не из чего", file=sys.stderr)
        return 2
    med = compute()
    if a.check:
        if not OUT.exists():
            print(f"нет {OUT}", file=sys.stderr)
            return 1
        have = json.loads(OUT.read_text(encoding="utf-8")).get("medians", {})
        bad = [c for c in med if repr(have.get(c)) != repr(med[c])]
        missing = [c for c in med if c not in have]
        if bad or missing:
            print(f"расходятся: {bad}; отсутствуют: {missing}", file=sys.stderr)
            return 1
        print(f"geometry medians check: OK ({len(med)} столбцов)")
        return 0
    payload = {
        "format_version": 1,
        "kind": "densito_geometry_medians",
        "source": "data/geometry_features.csv",
        "n_rows": int(pd.read_csv(CSV, usecols=["study"]).shape[0]),
        "note": ("Медианы столбцов признаков для импутации NaN. Только агрегаты: "
                 "ни путей, ни UID, ни меток."),
        "medians": med,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
                   encoding="utf-8")
    print(f"written {OUT} ({len(med)} столбцов)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
