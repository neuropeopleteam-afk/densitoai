#!/usr/bin/env python3
"""Таблица по повторам для отчёта: outputs/spart_position/per_repeat.csv -> per_repeat_table.md."""
from pathlib import Path
import pandas as pd
D = Path(__file__).resolve().parents[2] / "outputs" / "spart_position"
d = pd.read_csv(D / "per_repeat.csv")
L = ["| Повтор | Стек база | Стек gate | ΔAUC gate | Стек band70 (фикс.) | ΔAUC band70 | Стек gate3 | ΔAUC gate3 | ΔF1(+) gate | Δmacro-F1 gate | A база | A gate |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for _, r in d.iterrows():
    L.append(f"| {int(r['repeat'])} | {r.auc_base:.3f} | {r.auc_gate:.3f} | {r.auc_gate-r.auc_base:+.3f} | {r.auc_band70:.3f} | "
             f"{r.auc_band70-r.auc_base:+.3f} | {r.auc_gate3:.3f} | {r.auc_gate3-r.auc_base:+.3f} | {r.f1pos_gate-r.f1pos_base:+.3f} | "
             f"{r.macro_f1_gate-r.macro_f1_base:+.3f} | {r.auc_geom_base:.3f} | {r.auc_geom_gate:.3f} |")
s = lambda x: f"{x.mean():.3f} ± {x.std(ddof=1):.3f}"
L.append(f"| среднее ± sd | {s(d.auc_base)} | {s(d.auc_gate)} | {s(d.auc_gate-d.auc_base)} | {s(d.auc_band70)} | {s(d.auc_band70-d.auc_base)} | "
         f"{s(d.auc_gate3)} | {s(d.auc_gate3-d.auc_base)} | {s(d.f1pos_gate-d.f1pos_base)} | {s(d.macro_f1_gate-d.macro_f1_base)} | "
         f"{s(d.auc_geom_base)} | {s(d.auc_geom_gate)} |")
L.append("")
L.append("| Ветка | ΔAUC среднее ± sd | мин | в плюсе | ≥ 0.03 | ΔF1(+) | Δmacro-F1 | повторов с Δmacro-F1 < 0 |")
L.append("|---|---|---|---|---|---|---|---|")
for b in ["gate", "band70", "band80", "gate3", "gate_b", "ctrl_log", "band100"]:
    dd = d[f"auc_{b}"] - d.auc_base; dm = d[f"macro_f1_{b}"] - d.macro_f1_base
    L.append(f"| {b} | {dd.mean():+.3f} ± {dd.std(ddof=1):.3f} | {dd.min():+.3f} | {(dd>0).sum()}/20 | {(dd>=0.03).sum()}/20 | "
             f"{(d[f'f1pos_{b}']-d.f1pos_base).mean():+.3f} | {dm.mean():+.3f} | {(dm<0).sum()}/20 |")
(D / "per_repeat_table.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
