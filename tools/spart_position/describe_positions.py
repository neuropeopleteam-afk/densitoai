#!/usr/bin/env python3
"""Описательные числа для отчёта (не используются для выбора признака):
1) без меток — положение компонент по высоте (outputs/spart_position/component_positions.csv),
   совпадение полос между собой;
2) с метками — площадь в верхних 70 % и вне их у позитивов и негативов sp_art (только описание)."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
c = pd.read_csv(ROOT / "outputs" / "spart_position" / "component_positions.csv")
g = pd.read_csv(ROOT / "data" / "geometry_features.csv"); g = g[g.region == "spine"].reset_index(drop=True)
fr = c.drop_duplicates("file_path")
print(f"компонент {len(c)}, кадров с компонентами {c.file_path.nunique()}/{len(g)}")
print(f"маска кости на всю высоту кадра: {int(((fr.y0 == 0) & (fr.y1 == fr.h - 1)).sum())}/{len(fr)} кадров с компонентами")
h = np.histogram(c.cy_ratio, bins=np.linspace(0, 1, 11))[0]
print("центры компонент по высоте (доля протяжённости кости), децили:", dict(zip([f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)], h.tolist())))
for a, b in (("band50", "band60"), ("band60", "band70"), ("band70", "band80"), ("band80", "band100")):
    d = (g[f"metal_metal_{a}_area_mm2"] != g[f"metal_metal_{b}_area_mm2"]) | (g[f"metal_metal_{a}_max_gap"] != g[f"metal_metal_{b}_max_gap"])
    print(f"{a} vs {b}: различаются в {int(d.sum())}/{len(g)} строках")
top = g.metal_metal_band70_area_mm2; low = (g.metal_metal_area_mm2 - top).clip(lower=0)
for lab, name in ((1, "позитивы"), (0, "негативы")):
    m = g.sp_art == lab
    print(f"{name} (n={int(m.sum())}): верх70 > 3 мм² {int((top[m] > 3).sum())}, вне верх70 > 3 мм² {int((low[m] > 3).sum())}, "
          f"медиана верх70 {top[m].median():.1f}, медиана вне {low[m].median():.1f}, медиана всей площади {g.metal_metal_area_mm2[m].median():.1f}")
