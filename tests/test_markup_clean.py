#!/usr/bin/env python3
"""Очистка впечатанной разметки денситометра (src/markup_clean.py).

1. Кадр без разметки не меняется ни одним пикселем (в т.ч. кадры 499 из тестового набора, если доступны).
2. Синтетическая разметка (рамка L1–L4 с межпозвонковыми линиями; рамка бедра и повёрнутая рамка шейки)
   находится и закрашивается: средняя ошибка восстановления мала, ярких остатков почти нет.
3. Меньше MIN_SEGMENTS отрезков — очистка не включается (одиночная линия края поля не трогается).
Запуск:  python tests/test_markup_clean.py     (код 0 — пройдено)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import markup_clean as MC  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def frame(h=315, w=300, seed=0):
    import cv2
    rng = np.random.default_rng(seed)
    a = (rng.random((h, w)) * 90).astype(np.uint8)
    a = cv2.GaussianBlur(a, (9, 9), 0)
    cv2.ellipse(a, (w // 2, h // 2), (w // 6, h // 3), 0, 0, 360, 170, -1)  # «кость»
    return cv2.GaussianBlur(a, (5, 5), 0)


def main():
    import cv2
    for seed in range(5):
        a = frame(seed=seed)
        o, info = MC.clean(a)
        check(f"кадр без разметки не меняется (seed {seed})", (o == a).all() and not info["markup_cleaned"], str(info))
    # одиночная линия — ниже порога включения
    a = frame(seed=7); b = a.copy(); b[40, 30:270] = 255
    o, info = MC.clean(b)
    check("одна линия — очистка не включается", not info["markup_cleaned"] and (o == b).all(), str(info))
    # позвоночник: рамка + 3 линии
    a = frame(seed=1); b = a.copy()
    cv2.rectangle(b, (90, 57), (210, 270), 250, 1)
    for k in (1, 2, 3):
        y = int(57 + 213 * k / 4); cv2.line(b, (90, y), (210, y), 250, 1)
    o, info = MC.clean(b)
    err = np.abs(o.astype(int) - a.astype(int))
    check("позвоночник: разметка найдена", info["markup_cleaned"] and info["markup_segments"] >= 4, str(info))
    check("позвоночник: средняя ошибка восстановления < 1 уровня", err.mean() < 1.0, f"{err.mean():.3f}")
    check("позвоночник: ярких остатков линий < 1 %", (err > 40).sum() < 0.01 * (b != a).sum(), f"{(err > 40).sum()}")
    # бедро: рамка + повёрнутая рамка шейки
    a = frame(280, 280, seed=2); b = a.copy()
    cv2.rectangle(b, (50, 34), (230, 224), 250, 1)
    box = cv2.boxPoints(((134, 100), (50, 22), -45.0)).astype(np.int32); cv2.polylines(b, [box], True, 250, 1)
    o, info = MC.clean(b)
    err = np.abs(o.astype(int) - a.astype(int))
    check("бедро: разметка найдена, наклонные отрезки закрашены", info["markup_cleaned"] and info.get("markup_diag_pixels", 0) > 0, str(info))
    check("бедро: средняя ошибка восстановления < 1 уровня", err.mean() < 1.0, f"{err.mean():.3f}")
    # реальные кадры заказчика (если смонтированы): ни один не должен меняться
    ds_dir = Path("/ds/Исследования")
    if ds_dir.is_dir():
        import pydicom
        import inference
        n = bad = 0
        for f in sorted(ds_dir.rglob("*.dcm")):
            u8, _, _ = inference.normalize_pixels_ex(pydicom.dcmread(str(f)))
            _, k = MC.detect(u8)
            n += 1; bad += k >= MC.MIN_SEGMENTS
        check(f"кадры заказчика: очистка не срабатывает ни на одном из {n}", bad == 0, f"{bad}")
    else:
        print("  SKIP кадры заказчика не смонтированы (/ds/Исследования)")
    print("\nИТОГ:", "OK" if not FAILED else f"FAIL ({len(FAILED)})")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
