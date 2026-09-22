#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Запасы зоны «не уверен»: кривая «отказ–польза» по OOF и выбор запаса по квоте.

Зачем: запас подбирался при общей квоте на регион (`uncertainty.max_reject_rate` = 0.05).
Пять критериев не укладываются в 5 % строк региона вместе, поэтому четырём критериям
достался запас 0.0 — зона срабатывала только на строках, совпавших с порогом до
последнего бита, то есть на практике не срабатывала. Скрипт показывает, что даёт
каждая квота, и записывает выбранные запасы в `config.yaml`.

Запас не меняет `quality_class` и девять колонок выгрузки: это отдельный признак
`needs_review` (строку смотрит врач). Порог, модель и признаки не затрагиваются,
поэтому протокол nested repeated GroupKFold здесь не применяется.

Оценки читаются из `models/oof_stacked_<region>_<crit>.csv`. Решение берётся из колонки
`pred_label` — это ровно то, что выдаёт сервис (значение, равное порогу, при чтении CSV
может оказаться на 1e-16 ниже, и пересчёт сравнением потерял бы такие случаи).

Запуск:
  python tools/uncertainty_margins.py                      # только таблица
  python tools/uncertainty_margins.py --quota 0.05         # таблица для одной квоты
  python tools/uncertainty_margins.py --quota 0.05 --write # + записать в config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
CONFIG = ROOT / "config.yaml"
TIE_ATOL = 1e-9

REGION_CRITERIA = {"spine": ["sp_pos", "sp_axis", "sp_art"], "hip": ["hip_pos", "hip_roi"]}
REGION_NAME = {"spine": "Поясничный отдел позвоночника", "hip": "Проксимальный отдел бедра"}


def load_criterion(region: str, crit: str):
    df = pd.read_csv(MODELS / f"oof_stacked_{region}_{crit}.csv")
    summary = json.loads((MODELS / "metrics_summary.json").read_text(encoding="utf-8"))
    thr = float(summary[region][crit]["threshold"])
    y = df["y_true"].values.astype(int)
    s = df["oof_stacked"].values.astype(float)
    if "pred_label" in df.columns and not pd.to_numeric(df["pred_label"], errors="coerce").isna().any():
        yhat = pd.to_numeric(df["pred_label"]).values.astype(int)
    else:
        yhat = (s >= thr - TIE_ATOL).astype(int)
    dist = np.abs(s - thr)
    return df, y, s, yhat, dist, thr


def curve(y, yhat, dist, margins: List[float]) -> pd.DataFrame:
    rows = []
    err = (y != yhat)
    n = len(y)
    for m in margins:
        unc = dist <= m + 1e-12
        keep = ~unc
        rows.append({
            "margin": m,
            "доля в зоне": unc.mean(),
            "ошибок в зоне": int(err[unc].sum()),
            "всего ошибок": int(err.sum()),
            "доля ошибок в зоне": (err[unc].sum() / err.sum()) if err.sum() else float("nan"),
            "F1 на оставшихся": f1_score(y[keep], yhat[keep], zero_division=0) if keep.sum() else float("nan"),
            "осталось строк": int(keep.sum()),
            "из": n,
        })
    return pd.DataFrame(rows)


def margin_for_quota(dist: np.ndarray, q: float) -> float:
    """Наибольший запас из наблюдаемых расстояний, при котором доля строк в зоне <= q."""
    cand = 0.0
    for v in np.unique(np.sort(dist)):
        if (dist <= v + 1e-12).mean() <= q + 1e-12:
            cand = float(v)
        else:
            break
    return cand


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quota", type=float, default=None,
                    help="квота по критерию (доля строк региона в зоне); без неё печатается сетка")
    ap.add_argument("--write", action="store_true", help="записать запасы в config.yaml")
    ap.add_argument("--from-config", action="store_true",
                    help="таблица для запасов, которые реально лежат в config.yaml (для документов)")
    a = ap.parse_args(argv)

    data = {}
    for region, crits in REGION_CRITERIA.items():
        for crit in crits:
            data[crit] = load_criterion(region, crit)

    if a.from_config:
        import yaml
        ucfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["uncertainty"]
        margins = {c: (0.0 if v is None else float(v))
                   for c, v in (ucfg.get("margin_by_criterion") or {}).items()}
        print("| Критерий | порог | запас | «не уверен» на OOF | ошибок в зоне | доля всех ошибок критерия | F1(+) без зоны → на оставшихся |")
        print("|---|---|---|---|---|---|---|")
        for region, crits in REGION_CRITERIA.items():
            for crit in crits:
                df, y, s, yhat, dist, thr = data[crit]
                m = margins.get(crit, 0.0)
                row = curve(y, yhat, dist, [m]).iloc[0]
                f1_all = f1_score(y, yhat, zero_division=0)
                n_unc = int(round(row["доля в зоне"] * len(y)))
                print(f"| {crit} | {thr:.4f} | {m:.6g} | {n_unc}/{len(y)} ({row['доля в зоне']:.1%}) | "
                      f"{row['ошибок в зоне']}/{row['всего ошибок']} | {row['доля ошибок в зоне']:.0%} | "
                      f"{f1_all:.3f} → {row['F1 на оставшихся']:.3f} |")
        for region, crits in REGION_CRITERIA.items():
            base = data[crits[0]][0]
            unc_any = np.zeros(len(base), bool)
            for crit in crits:
                df, y, s, yhat, dist, thr = data[crit]
                if len(df) != len(base):
                    continue
                unc_any |= (dist <= margins.get(crit, 0.0) + 1e-12)
            print(f"\n{REGION_NAME[region]}: строк с хотя бы одним «не уверен» — "
                  f"{unc_any.mean():.1%} ({int(unc_any.sum())}/{len(unc_any)})")
        return 0

    quotas = [a.quota] if a.quota is not None else [0.02, 0.05, 0.10, 0.15]
    print("# Зона «не уверен»: что даёт каждая квота\n")
    chosen: Dict[float, Dict[str, float]] = {}
    for q in quotas:
        margins = {c: margin_for_quota(d[4], q) for c, d in data.items()}
        chosen[q] = margins
        print(f"## Квота по критерию {q:.0%}\n")
        print("| Критерий | запас | доля в зоне | ошибок в зоне / всего | доля ошибок в зоне | F1 без зоны → на оставшихся |")
        print("|---|---|---|---|---|---|")
        for region, crits in REGION_CRITERIA.items():
            for crit in crits:
                df, y, s, yhat, dist, thr = data[crit]
                m = margins[crit]
                row = curve(y, yhat, dist, [m]).iloc[0]
                f1_all = f1_score(y, yhat, zero_division=0)
                print(f"| `{crit}` | {m:.6g} | {row['доля в зоне']:.1%} "
                      f"({int(row['доля в зоне'] * len(y))}/{len(y)}) | "
                      f"{row['ошибок в зоне']}/{row['всего ошибок']} | "
                      f"{row['доля ошибок в зоне']:.0%} | {f1_all:.3f} → {row['F1 на оставшихся']:.3f} |")
        # доля строк региона, где хотя бы один критерий «не уверен»
        for region, crits in REGION_CRITERIA.items():
            base = data[crits[0]][0]
            unc_any = np.zeros(len(base), bool)
            for crit in crits:
                df, y, s, yhat, dist, thr = data[crit]
                if len(df) != len(base):
                    continue
                unc_any |= (dist <= margins[crit] + 1e-12)
            print(f"\n{REGION_NAME[region]}: строк с хотя бы одним «не уверен» — "
                  f"{unc_any.mean():.1%} ({int(unc_any.sum())}/{len(unc_any)})")
        print()

    if a.write:
        if a.quota is None:
            print("--write требует --quota", file=sys.stderr)
            return 2
        import math
        import re
        margins = chosen[a.quota]
        # вверх до 1e-9 — так же, как в train_stacked.py, чтобы граничная строка не выпадала
        margins = {c: (0.0 if abs(v) < 1e-9 else math.ceil(float(v) * 1e9) / 1e9)
                   for c, v in margins.items()}
        text = CONFIG.read_text(encoding="utf-8")
        m_block = re.search(r"^(\s*)margin_by_criterion:\s*$", text, re.M)
        if not m_block:
            print("в config.yaml не найден блок margin_by_criterion — файл не изменён", file=sys.stderr)
            return 3
        indent = m_block.group(1) + "  "
        start = m_block.end()
        if text[start:start + 1] == "\n":
            start += 1                      # перевод строки после заголовка блока
        lines = text[start:].splitlines(keepends=True)
        taken = 0
        for ln in lines:
            if re.match(rf"^{re.escape(indent)}\S+:\s*\S*\s*$", ln):
                taken += 1
            else:
                break
        if taken != len(margins):
            print(f"в блоке margin_by_criterion {taken} строк, а критериев {len(margins)} — файл не изменён",
                  file=sys.stderr)
            return 3
        block_len = sum(len(x) for x in lines[:taken])
        new_block = "".join(f"{indent}{c}: {margins[c]:.9g}\n" for c in margins)
        CONFIG.write_text(text[:start] + new_block + text[start + block_len:], encoding="utf-8")
        print("config.yaml: uncertainty.margin_by_criterion ->",
              json.dumps(margins, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
