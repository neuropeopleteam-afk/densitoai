#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""uncertain_zone_md.py — таблицы docs/UNCERTAIN_ZONE.md (2.5.0) из docs/uncertain_zone.json.

Числа только из JSON, который пишет `tools/uncertain_zone.py`; скрипт ничего не считает заново, кроме форматирования.
Печатает markdown-блоки с маркерами `[//]: # (UZ:<имя>)`; `--write` подставляет их в docs/UNCERTAIN_ZONE.md между
строками `[//]: # (UZ:<имя>)` и `[//]: # (/UZ:<имя>)` (невидимы в markdown).

  python tools/uncertain_zone.py --regress <results_debug.csv 2.5.0> --markdown
  python tools/p25/uncertain_zone_md.py --write
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
J = ROOT / "docs" / "uncertain_zone.json"
MD = ROOT / "docs" / "UNCERTAIN_ZONE.md"
DELTAS = [0.00, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30]


def pct(x):
    return f"{100 * x:.1f} %"


def ci(v):
    return f"[{v[0]:.2f}; {v[1]:.2f}]"


def row_curve(label, p):
    return (f"| {label} | {p['n_uncertain']} ({pct(p['share_uncertain'])}) | {p['sensitivity_confident']:.3f} "
            f"{ci(p['ci95']['sensitivity_confident'])} | {p['specificity_confident']:.3f} | {p['f1_confident']:.3f} | "
            f"{p['n_pos_in_zone']}/{p['n_pos']} | {p['n_errors_in_zone']}/{p['n_errors']} | {p['n_fn_in_zone']}/{p['n_fn_total']} | "
            f"{p['recall_with_review']:.3f} |")


def blocks(d: dict) -> dict:
    out = {}
    R = d["regions"]
    lines = ["| Область | n | положительных | «не уверен» | полнота среди уверенных [ДИ 95 %] | специфичность | F1 | "
             "положительных в зоне | ошибок в зоне | пропусков в зоне | полнота с учётом зоны |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for reg in ("spine", "hip"):
        r, c = R[reg], R[reg]["current"]
        lines.append(f"| {r['name']} | {r['n']} | {r['n_pos']} ({r['n_pos_studies']} иссл.) | {c['n_uncertain']} "
                     f"({pct(c['share_uncertain'])}) | {c['sensitivity_confident']:.3f} {ci(c['ci95']['sensitivity_confident'])} | "
                     f"{c['specificity_confident']:.3f} | {c['f1_confident']:.3f} | {c['n_pos_in_zone']}/{c['n_pos']} | "
                     f"{c['n_errors_in_zone']}/{c['n_errors']} | {c['n_fn_in_zone']}/{c['n_fn_total']} | {c['recall_with_review']:.3f} |")
    out["current"] = "\n".join(lines)
    for reg in ("spine", "hip"):
        r = R[reg]
        lines = [f"{r['name']} (n = {r['n']}, положительных {r['n_pos']}):", "",
                 "| δ | «не уверен» | полнота среди уверенных [ДИ] | специфичность | F1 | положительных в зоне | ошибок в зоне | "
                 "пропусков в зоне | полнота с учётом зоны |", "|---|---|---|---|---|---|---|---|---|",
                 row_curve("текущее", r["current"])]
        for p in r["score_band"]:
            if any(abs(p["delta"] - x) < 1e-9 for x in DELTAS):
                lines.append(row_curve(f"{p['delta']:.2f}", p))
        out[f"curve_{reg}"] = "\n".join(lines)
    lines = ["| Область | δ | «не уверен» | полнота среди уверенных [ДИ] | специфичность | F1 | пропусков в зоне | полнота с учётом зоны |",
             "|---|---|---|---|---|---|---|---|"]
    for reg, name, ds in (("spine", "позвоночник", (0.20, 0.22, 0.23, 0.25)), ("hip", "бедро", (0.20, 0.25))):
        for p in R[reg]["qp_band"]:
            if any(abs(p["delta"] - x) < 1e-9 for x in ds):
                lines.append(f"| {name} | {p['delta']:.2f} | {p['n_uncertain']} ({pct(p['share_uncertain'])}) | "
                             f"{p['sensitivity_confident']:.3f} {ci(p['ci95']['sensitivity_confident'])} | {p['specificity_confident']:.3f} | "
                             f"{p['f1_confident']:.3f} | {p['n_fn_in_zone']}/{p['n_fn_total']} | {p['recall_with_review']:.3f} |")
    out["qp"] = "\n".join(lines)
    lines = ["| Область | Семья | δ | «не уверен» (Δ строк) | полнота среди уверенных (Δ) | с учётом зоны (Δ) | пропусков в зоне | "
             "парная разность [ДИ 95 %] | вывод |", "|---|---|---|---|---|---|---|---|---|"]
    rec = d["recommendation"]["by_region"]
    for reg, name in (("spine", "позвоночник"), ("hip", "бедро")):
        for fam, fname in (("score_band", "скоры критериев"), ("qp_band", "quality_prob")):
            k = rec[reg][fam]
            pv = k["paired_vs_current"]["sensitivity_confident"]
            verdict = "в пределах шума" if pv["ci95"][0] <= 0 else "прирост за пределами шума"
            lines.append(f"| {name} | {fname} | {k['delta']:.2f} | {k['n_uncertain']} ({k['delta_rows_to_zone']:+d}) | "
                         f"{k['sensitivity_confident']:.3f} ({k['delta_sensitivity_confident']:+.3f}) | {k['recall_with_review']:.3f} "
                         f"({k['delta_recall_with_review']:+.3f}) | {k['n_fn_in_zone']} (сверх случайного {k['fn_in_zone_excess']:+.1f}) | "
                         f"{pv['diff']:+.3f} [{pv['ci95'][0]:+.2f}; {pv['ci95'][1]:+.2f}] | {verdict} |")
    out["candidates"] = "\n".join(lines)
    rg = d.get("regress")
    if rg:
        lines = ["| δ | позвоночник (из 166) | бедро (из 333) | всего | доля |", "|---|---|---|---|---|"]
        for p in rg["score_band"]:
            if any(abs(p["delta"] - x) < 1e-9 for x in DELTAS):
                lines.append(f"| {p['delta']:.2f} | {p['spine']['n_uncertain']} | {p['hip']['n_uncertain']} | {p['total_uncertain']} | "
                             f"{pct(p['total_share'])} |")
        out["regress"] = "\n".join(lines)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    d = json.loads(J.read_text(encoding="utf-8"))
    b = blocks(d)
    if not a.write:
        for k, v in b.items():
            print(f"[//]: # (UZ:{k})\n{v}\n[//]: # (/UZ:{k})\n")
        return 0
    s = MD.read_text(encoding="utf-8")
    for k, v in b.items():
        pat = re.compile(rf"\[//\]: # \(UZ:{k}\)\n(?:.*?\n)?\[//\]: # \(/UZ:{k}\)", re.S)
        if not pat.search(s):
            raise SystemExit(f"нет маркера UZ:{k} в {MD}")
        s = pat.sub(lambda _m: f"[//]: # (UZ:{k})\n{v}\n[//]: # (/UZ:{k})", s)
    MD.write_text(s, encoding="utf-8")
    print("обновлено:", MD, ", ".join(b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
