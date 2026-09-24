#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refresh_nested_fields.py — обновить поля nested_* в models/metrics_summary.json без переобучения.

Поля nested_* пишет src/train_stacked.py (nested_block_for). До A2 (24.09) в ветке К2 поля nested_auc_mean /
nested_auc_ci брались от вентиля, который НЕ принят, поэтому у sp_art, hip_pos, hip_roi nested_auc_mean
равнялся отвергнутой альтернативе (0.809 / 0.681 / 0.852 вместо 0.817 / 0.709 / 0.873 у поставки). Генератор
исправлен; этот скрипт применяет ту же функцию nested_block_for к текущему файлу, чтобы не переобучать стэк
(пороги, OOF и веса не меняются — меняются только поля nested_*; порядок ключей и формат JSON сохраняются).

После правки: nested_auc_mean == nested_auc_production у всех критериев, nested_auc_ci — ДИ поставки,
nested_auc_alternative / nested_auc_alternative_ci — отвергнутая (заменённая) альтернатива.

  python tools/refresh_nested_fields.py          # записать
  python tools/refresh_nested_fields.py --check  # код 1, если файл не согласован
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
PATH = ROOT / "models" / "metrics_summary.json"
CRITS = [("spine", "sp_pos"), ("spine", "sp_axis"), ("spine", "sp_art"), ("hip", "hip_pos"), ("hip", "hip_roi")]


def refreshed(d: dict) -> dict:
    import train_stacked as ts
    for region, crit in CRITS:
        blk = d[region][crit]
        nb = ts.nested_block_for(crit, str(blk.get("emb_source") or "imagenet"), ts.nested_decision_for(crit))
        if nb["production"] is None:
            continue
        assert abs(float(nb["production"]) - float(blk["nested_auc_production"])) < 1e-12, crit
        new = {}
        for k, v in blk.items():
            if k == "nested_auc_alternative_ci":
                continue
            if k == "nested_auc_mean":
                v = nb["mean"]
            elif k == "nested_auc_ci":
                v = nb["ci"]
            new[k] = v
            if k == "nested_auc_ci":
                new["nested_auc_alternative_ci"] = nb["alternative_ci"]
        d[region][crit] = new
    return d


def consistent(d: dict) -> list[str]:
    bad = []
    for region, crit in CRITS:
        b = d[region][crit]
        if b.get("nested_auc_production") is not None and b.get("nested_auc_mean") != b.get("nested_auc_production"):
            bad.append(crit)
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    d = json.loads(PATH.read_text(encoding="utf-8"))
    bad = consistent(d)
    print("nested_auc_mean != nested_auc_production:", bad or "нет")
    if a.check:
        return 1 if bad else 0
    d = refreshed(d)
    PATH.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    for region, crit in CRITS:
        b = d[region][crit]
        print(f"{crit}: mean {b['nested_auc_mean']:.4f} production {b['nested_auc_production']:.4f} "
              f"alternative {b['nested_auc_alternative']:.4f} ({b['nested_alternative_what']})")
    return 1 if consistent(d) else 0


if __name__ == "__main__":
    raise SystemExit(main())
