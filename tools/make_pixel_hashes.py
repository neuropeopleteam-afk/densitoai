"""sha1 нормализованных пикселей для группировки дубликатов в nested-протоколах.

Группировка задаётся на СЫРОЙ предобработке (`--variant baseline`), чтобы состав групп
не зависел от сравниваемого варианта предобработки и все варианты сравнивались на
одинаковых разбиениях.

Запуск: python tools/make_pixel_hashes.py [--variant baseline|mask|canonical]
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import pydicom  # noqa: E402

import preprocess  # noqa: E402
from extract_all_features import VARIANTS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="baseline", choices=list(VARIANTS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    preprocess.flags().update(VARIANTS[a.variant])
    from inference import normalize_pixels  # noqa: E402  (импорт после установки флагов)

    lab = pd.read_csv(ROOT / "data" / "labels_for_embeddings.csv")
    hashes = []
    for p in lab["file_path"]:
        ds = pydicom.dcmread(p, force=True)
        img = normalize_pixels(ds)
        hashes.append(hashlib.sha1(img.tobytes() + str(img.shape).encode()).hexdigest())
    lab["pixel_hash"] = hashes
    out = Path(a.out) if a.out else ROOT / "outputs" / "pixel_hashes.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    lab[["study", "file_path", "region", "pixel_hash"]].to_csv(out, index=False)

    vc = lab["pixel_hash"].value_counts()
    dup = vc[vc > 1]
    print(f"вариант {a.variant}: файлов {len(lab)}, уникальных хэшей {lab['pixel_hash'].nunique()}, "
          f"групп дубликатов {len(dup)}, файлов в дубликатах {int(dup.sum())}")
    g = lab.groupby("pixel_hash")["study"].nunique()
    print(f"групп, пересекающих >1 исследование: {int((g > 1).sum())}")
    print(f"записан {out}")


if __name__ == "__main__":
    main()
