#!/usr/bin/env python3
"""Стресс-тест «разметка денситометра на изображении» (разъяснение организаторов 24.09.2026).

Организаторы: «Отдельной разметки анатомических областей в датасете нет. На части снимков такая разметка
денситометра есть прямо на изображении, но сохраняется она не всегда, поэтому оценивать её не требуется».
Значит, на закрытом тесте часть кадров может прийти с впечатанными контурами ROI сканера. На внешнем наборе
DEXA-Osteo (контуры впечатаны) детектор посторонних предметов срабатывал на 304 из 344 (docs/EXTERNAL_DXA.md).

Что делает: копирует 499 DICOM заказчика и впечатывает в пиксели тонкую (1 px) светлую разметку в стиле
отчёта денситометра — для позвоночника рамка L1–L4 с тремя межпозвонковыми линиями, для бедра рамка шейки и
контур области бедра. Геометрия берётся по кадру (центральная полоса / верхне-средняя зона), это
приближение, а не копия интерфейса сканера. Исходные файлы не меняются.

Запуск (на сервере):
  python tools/markup_stress.py make --src <499 DICOM> --dst /tmp/markup/in [--value 250] [--width 1]
  docker run --rm --network none -v /tmp/markup/in:/ds:ro -v /tmp/markup/out:/out --entrypoint python \
      densitoai:2.4.0 src/inference.py --input /ds --output /out/results.csv --debug-csv auto --path-mode relative
  python tools/markup_stress.py compare --base regress_2_4_0.csv --cand /tmp/markup/out/results.csv
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def draw_markup(a: np.ndarray, is_spine: bool, value: int, width: int) -> np.ndarray:
    import cv2
    out = a.copy()
    h, w = out.shape[:2]
    col = int(value)
    if is_spine:
        x0, x1 = int(w * 0.30), int(w * 0.70)
        y0, y1 = int(h * 0.18), int(h * 0.86)
        cv2.rectangle(out, (x0, y0), (x1, y1), col, width)
        for k in (1, 2, 3):
            y = int(y0 + (y1 - y0) * k / 4)
            cv2.line(out, (x0, y), (x1, y), col, width)
    else:
        # рамка шейки бедра (повёрнутый прямоугольник) и контур области проксимального отдела
        cx, cy = int(w * 0.48), int(h * 0.36)
        box = cv2.boxPoints(((cx, cy), (w * 0.18, h * 0.08), -45.0)).astype(np.int32)
        cv2.polylines(out, [box], True, col, width)
        cv2.rectangle(out, (int(w * 0.18), int(h * 0.12)), (int(w * 0.82), int(h * 0.80)), col, width)
    return out


def cmd_make(args) -> int:
    import pydicom
    src, dst = Path(args.src), Path(args.dst)
    n = 0
    for f in sorted(src.rglob("*.dcm")):
        ds = pydicom.dcmread(str(f))
        a = ds.pixel_array
        is_spine = a.shape[1] >= 290  # в данных заказчика ширина позвоночника 300 px, бедра 248/280 px
        m = draw_markup(a, is_spine, args.value, args.width).astype(a.dtype)
        ds.PixelData = m.tobytes()
        rel = f.relative_to(src)
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        ds.save_as(str(dst / rel))
        n += 1
    print(f"размечено файлов: {n} -> {dst}")
    return 0


def _read(p: Path):
    txt = p.read_text(encoding="utf-8-sig")
    dl = ";" if txt.split("\n", 1)[0].count(";") > txt.split("\n", 1)[0].count(",") else ","
    return {r["image_uid"]: r for r in csv.DictReader(txt.splitlines(), delimiter=dl)}


def cmd_compare(args) -> int:
    base, cand = _read(Path(args.base)), _read(Path(args.cand))
    common = [k for k in base if k in cand]
    cls_flip = Counter()
    viol_new, viol_lost = Counter(), Counter()
    fail = sum(1 for k in common if cand[k]["processing_status"] != "Success")
    by_region = Counter()
    for k in common:
        b, c = base[k], cand[k]
        reg = b["anatomical_region"]
        by_region[reg] += 1
        if b["quality_class"] != c["quality_class"]:
            cls_flip[(reg, b["quality_class"] + "→" + c["quality_class"])] += 1
        bv = set(filter(None, b["violation_type"].split(";")))
        cv = set(filter(None, c["violation_type"].split(";")))
        for v in cv - bv:
            viol_new[(reg, v)] += 1
        for v in bv - cv:
            viol_lost[(reg, v)] += 1
    res = {"n": len(common), "failures": fail, "by_region": dict(by_region),
           "class_flips": {f"{r} {t}": n for (r, t), n in cls_flip.items()},
           "violation_added": {f"{r}: {v}": n for (r, v), n in viol_new.items()},
           "violation_removed": {f"{r}: {v}": n for (r, v), n in viol_lost.items()},
           "share_class_changed": round(sum(cls_flip.values()) / max(len(common), 1), 4)}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("--src", required=True); m.add_argument("--dst", required=True)
    m.add_argument("--value", type=int, default=250); m.add_argument("--width", type=int, default=1)
    c = sub.add_parser("compare"); c.add_argument("--base", required=True); c.add_argument("--cand", required=True)
    c.add_argument("--json", default="")
    a = ap.parse_args()
    return cmd_make(a) if a.cmd == "make" else cmd_compare(a)


if __name__ == "__main__":
    sys.exit(main())
