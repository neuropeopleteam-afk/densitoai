#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сверка списков признаков контура A (A1, 24.09.2026). Истина для боя — feature_cols, сохранённые
внутри models/model_*_geom.pkl (инференс берёт их оттуда). С ними обязаны совпадать:
  * config.yaml: geometry_cols (+ features) по критерию;
  * встроенные умолчания src/inference.py (DEFAULT_CONFIG);
  * src/train_stacked.py: CRITERION_GEOMETRY_COLS + CRITERION_EXTRA_COLS (разбор через ast, без импорта);
  * models/MODEL_CONTRACT.md (раздел «Признаки контура A»);
  * any-модели региона: отсортированное объединение признаков критериев без признаков эмбеддинга.
Отдельно: список колонок импутации (inference.IMPUTATION_MEDIAN_COLS) = tools/make_geometry_medians.py
= ключи models/geometry_medians.json. Тест падает при любом расхождении. Переобучения нет.

Запуск: python tests/test_feature_contract.py   (код возврата 0 — ок).
"""
import ast
import json
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import inference as inf  # noqa: E402

fails = []


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def load_pkl(p: Path):
    try:
        import joblib
        return joblib.load(p)
    except Exception:  # noqa: BLE001
        with open(p, "rb") as f:
            return pickle.load(f)


def pkl_cols(name: str):
    p = ROOT / "models" / name
    if not p.exists():
        return None
    obj = load_pkl(p)
    return list(obj.get("feature_cols") or []) if isinstance(obj, dict) else list(getattr(obj, "feature_cols", []) or [])


def module_dict(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


cfg = inf.load_config()
default = inf.DEFAULT_CONFIG
emb_feats = cfg.get("features") or {}

# критерий -> файл модели (у бедра одна модель на обе стороны)
MODEL_OF = {"sp_pos": "model_spine_sp_pos_geom.pkl", "sp_axis": "model_spine_sp_axis_geom.pkl",
            "sp_art": "model_spine_sp_art_geom.pkl", "rh_pos": "model_hip_pos_geom.pkl",
            "lh_pos": "model_hip_pos_geom.pkl", "rh_roi": "model_hip_roi_geom.pkl", "lh_roi": "model_hip_roi_geom.pkl"}
TRAIN_KEY = {"sp_pos": "sp_pos", "sp_axis": "sp_axis", "sp_art": "sp_art", "rh_pos": "hip_pos",
             "lh_pos": "hip_pos", "rh_roi": "hip_roi", "lh_roi": "hip_roi"}

train = ROOT / "src" / "train_stacked.py"
tr_geom = module_dict(train, "CRITERION_GEOMETRY_COLS")
tr_extra = module_dict(train, "CRITERION_EXTRA_COLS")
contract = (ROOT / "models" / "MODEL_CONTRACT.md").read_text(encoding="utf-8")

truth = {}
for crit, fname in MODEL_OF.items():
    cols = pkl_cols(fname)
    check(cols is not None and len(cols) > 0, f"{fname}: feature_cols есть ({cols})")
    if not cols:
        continue
    truth[crit] = cols
    extra = list(emb_feats.get(crit, []))
    check(list(cfg["geometry_cols"][crit]) + extra == cols,
          f"{crit}: config.yaml geometry_cols{'+features' if extra else ''} = pkl feature_cols")
    check(list(default["geometry_cols"][crit]) + list((default.get("features") or {}).get(crit, [])) == cols,
          f"{crit}: DEFAULT_CONFIG в inference.py = pkl feature_cols")
    k = TRAIN_KEY[crit]
    check(list(tr_geom.get(k, [])) + list(tr_extra.get(k, [])) == cols,
          f"{crit}: train_stacked CRITERION_GEOMETRY_COLS[{k}]+EXTRA = pkl feature_cols")

# MODEL_CONTRACT.md: строки вида «- `sp_pos`: `a, b` + `c` ...»
for label, crit in (("sp_pos", "sp_pos"), ("sp_axis", "sp_axis"), ("sp_art", "sp_art"),
                    ("rh_pos", "rh_pos"), ("rh_roi", "rh_roi")):
    m = re.search(r"^- `" + label + r"`[^:]*: (.+)$", contract, flags=re.MULTILINE)
    listed = []
    if m:
        for chunk in re.findall(r"`([^`]+)`", m.group(1)):
            if chunk.startswith("config.yaml") or chunk.endswith(".pkl"):
                continue
            listed += [c.strip() for c in chunk.split(",") if c.strip()]
    check(listed == truth.get(crit), f"MODEL_CONTRACT.md {label}: {listed} = pkl feature_cols")

# any-модели региона
for region, fname in (("spine", "model_spine_any_geom.pkl"), ("right_hip", "model_right_hip_any_geom.pkl"),
                      ("left_hip", "model_left_hip_any_geom.pkl")):
    cols = pkl_cols(fname)
    exp = sorted({c for cr in cfg["criteria_by_region"][region] for c in cfg["geometry_cols"][cr]})
    check(cols == exp, f"{fname}: feature_cols = объединение geometry_cols критериев региона")

# импутация медианами
_med_p = ROOT / "models" / "geometry_medians.json"
med_keys = sorted((json.loads(_med_p.read_text(encoding="utf-8")).get("medians") or {}).keys()) \
    if _med_p.exists() else None
mk = module_dict(ROOT / "tools" / "make_geometry_medians.py", "IMPUTATION_MEDIAN_COLS")
check(list(inf.IMPUTATION_MEDIAN_COLS) == list(mk), "IMPUTATION_MEDIAN_COLS: inference.py = make_geometry_medians.py")
if med_keys is not None:
    check(sorted(inf.IMPUTATION_MEDIAN_COLS) == med_keys, "IMPUTATION_MEDIAN_COLS = ключи models/geometry_medians.json")

print(f"\n{'ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ' if not fails else f'ПРОВАЛЕНО: {len(fails)}'}")
sys.exit(1 if fails else 0)
