#!/usr/bin/env python3
"""
build_actions_json.py — экспорт текстов карточки решения и режима лаборанта из config.yaml
(секции `actions`, `measurement_norms`) в web/assets/actions.json, чтобы тексты не дублировались
в HTML. Справочные диапазоны (ref_q95) считаются по размеченным нормальным кадрам
data/geometry_features.csv: 95 % квантиль модуля признака среди кадров с quality_class == 0
(для ref_low — 5 % квантиль). Допуски ТЗ (tz_max / tz_min) берутся из config.yaml как есть.

Запуск: python tools/web/build_actions_json.py [--root <densito_rebuild>] [--out web/assets/actions.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

REGION_OF = {
    "axis_angle_deg": "spine", "center_offset_ratio": "spine", "curvature": "spine",
    "metal_metal_area_mm2": None,  # считается отдельно для spine и hip
    "abs_shaft_angle_deg": "hip", "edge_distance_mm": "hip", "scan_length_mm": "hip",
    "shaft_len_below_troch_mm": "hip", "lesser_troch_prominence_mm": "hip",
}


def ref_quantile(df: pd.DataFrame, col: str, low: bool) -> float | None:
    if col not in df.columns:
        return None
    v = df[col].abs().dropna()
    if len(v) < 20:
        return None
    return round(float(v.quantile(0.05 if low else 0.95)), 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--config", default=None)
    ap.add_argument("--features", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    cfg = yaml.safe_load(open(a.config or root / "config.yaml", encoding="utf-8"))
    actions = cfg.get("actions") or {}
    norms = cfg.get("measurement_norms") or {}
    feats_path = Path(a.features or root / "data" / "geometry_features.csv")
    g = pd.read_csv(feats_path) if feats_path.exists() else pd.DataFrame()
    normal = g[g.get("quality_class", pd.Series(dtype=float)) == 0] if len(g) else g
    spine = normal[normal.region == "spine"] if len(normal) else normal
    hip = normal[normal.region.isin(["right_hip", "left_hip"])] if len(normal) else normal

    out_norms = {}
    for key, spec in norms.items():
        item = {k: v for k, v in spec.items() if k not in ("ref", "ref_col", "ref_low")}
        if spec.get("ref"):
            col = spec.get("ref_col", key)
            low = bool(spec.get("ref_low"))
            reg = REGION_OF.get(key)
            if reg is None:
                item["ref_q95_spine"] = ref_quantile(spine, col, low)
                item["ref_q95_hip"] = ref_quantile(hip, col, low)
            else:
                item["ref_q95"] = ref_quantile(spine if reg == "spine" else hip, col, low)
            item["ref_kind"] = "q05_normal" if low else "q95_normal"
            mult = float(spec.get("mult", 1))
            for k in ("ref_q95", "ref_q95_spine", "ref_q95_hip"):
                if item.get(k) is not None:
                    item[k] = round(item[k] * mult, 2)
        out_norms[key] = item

    payload = {
        "version": str(cfg.get("version", "")),
        "source": {
            "actions": "config.yaml: actions",
            "norms": "config.yaml: measurement_norms; справочные квантили — data/geometry_features.csv, "
                     f"нормальные кадры (spine n={len(spine)}, hip n={len(hip)})",
        },
        "actions": actions,
        "measurement_norms": out_norms,
    }
    out = Path(a.out or root / "web" / "assets" / "actions.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
