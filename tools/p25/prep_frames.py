#!/usr/bin/env python3
"""2.5: служебная таблица для оценок пунктов 1 и 3 (outputs/p25/frames.csv, каталог outputs/ не в git).

По каждому из 499 файлов data/labels_full.csv: локальный путь, sha1 нормализованных пикселей (тот же способ, что
tools/review/select_frames.py — по нему находятся 40 кадров слепой проверки и строятся группы «исследование + хэш
пикселей» для вложенной проверки пункта 3). Таблица содержит ключ ослепления (хэш кадра -> файл), поэтому хранится только в outputs/.

Запуск: OMP_NUM_THREADS=1 python tools/p25/prep_frames.py [--data /путь/к/Исследования]
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pydicom  # noqa: E402

import inference  # noqa: E402


def local_path(fp: str, data: Path) -> Path:
    rel = fp.split("Исследования/", 1)[1]
    return data / rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/home/user/workspace/work/dataset/Исследования"))
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "p25" / "frames.csv")
    a = ap.parse_args()
    lab = pd.read_csv(ROOT / "data" / "labels_full.csv")
    rows = []
    for fp, region in zip(lab.file_path, lab.region):
        p = local_path(fp, a.data)
        ds = pydicom.dcmread(str(p), force=True)
        img = inference.normalize_pixels(ds)
        sha = hashlib.sha1(np.ascontiguousarray(img).tobytes() + str(img.shape).encode()).hexdigest()
        rec = {"file_path": fp, "local_path": str(p), "pixel_sha1": sha, "region": region}
        rows.append(rec)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("rows", len(rows), "->", a.out)


if __name__ == "__main__":
    main()
