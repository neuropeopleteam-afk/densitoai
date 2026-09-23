"""Кэш кадров для GPU-эксперимента «синтетика для sp_pos».

Для каждой строки data/labels_for_embeddings.csv (порядок строк сохраняется) читает DICOM ровно тем
пайплайном, что src/embeddings.py при варианте предобработки canonical (read_dicom_normalized ->
canonicalize_exposure), и приводит к входу бэкбона (Resize (320, 192) PIL bilinear, как в
FrozenBackbone.tf). Сохраняет:
  <out>/frames_u8.npy   (N, 320, 192) uint8 — вход сети до нормализации ImageNet;
  <out>/frames_meta.csv  study, file_path, region, pixel_hash, group (компоненты (study, pixel_hash)),
                         center_offset_ratio / bone_width_ratio из geometry_features_canonical.csv (без меток ТЗ).
Метки ТЗ в кэш не пишутся — они нужны только nested-оценке.

Запуск (песочница):
  DENSITO_ROOT=/home/user/workspace/densito/src/densito_rebuild \
  DENSITO_DATASET=/home/user/workspace/densito/dataset/Исследования \
  python gpu/sppos_cache.py --out data/cache --hashes /path/to/pixel_hashes.csv
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(os.environ.get("DENSITO_ROOT", "/home/user/workspace/densito/src/densito_rebuild"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import preprocess  # noqa: E402
from geometry_features import read_dicom_normalized  # noqa: E402

H, W = 320, 192
OLD_PREFIX = "/home/user/workspace/dataset_extracted/Датасет/НД_для_обучения/Исследования"


def map_path(p: str, dataset_dir: str) -> str:
    if p.startswith(OLD_PREFIX):
        return dataset_dir.rstrip("/") + p[len(OLD_PREFIX):]
    return p


def connected_groups(studies, hashes):
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s, h in zip(studies, hashes):
        ra, rb = find(("s", s)), find(("h", h))
        if ra != rb:
            parent[ra] = rb
    roots = [find(("s", s)) for s in studies]
    ids = {r: i for i, r in enumerate(dict.fromkeys(roots))}
    return np.array([ids[r] for r in roots])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cache")
    ap.add_argument("--hashes", required=True, help="outputs/pixel_hashes.csv (tools/pixel_hash.py)")
    ap.add_argument("--dataset", default=os.environ.get("DENSITO_DATASET", "/home/user/workspace/densito/dataset/Исследования"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    preprocess.set_variant("canonical")
    lab = pd.read_csv(ROOT / "data" / "labels_for_embeddings.csv")
    geom = pd.read_csv(ROOT / "data" / "geometry_features_canonical.csv")
    assert (geom["file_path"].values == lab["file_path"].values).all(), "порядок строк geometry_features_canonical != labels"
    hashes = pd.read_csv(a.hashes)
    hmap = dict(zip(hashes["file_path"], hashes["pixel_hash"]))
    lab["pixel_hash"] = lab["file_path"].map(hmap)
    assert lab["pixel_hash"].notna().all(), "нет хэша для части файлов"
    lab["group"] = connected_groups(lab["study"].values, lab["pixel_hash"].values)

    frames = np.zeros((len(lab), H, W), dtype=np.uint8)
    native = []
    for i, p in enumerate(lab["file_path"]):
        img_u8, _ = read_dicom_normalized(map_path(p, a.dataset))
        native.append(img_u8.shape)
        # ровно как FrozenBackbone.tf: ToPILImage(3ch) -> Resize((320,192)) (PIL bilinear) -> ToTensor
        pil = Image.fromarray(np.stack([img_u8] * 3, axis=2))
        pil = pil.resize((W, H), Image.BILINEAR)
        frames[i] = np.asarray(pil)[:, :, 0]
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(lab)}")
    np.save(out / "frames_u8.npy", frames)
    meta = lab[["study", "file_path", "region", "pixel_hash", "group"]].copy()
    meta["native_rows"] = [s[0] for s in native]
    meta["native_cols"] = [s[1] for s in native]
    for c in ("center_offset_ratio", "bone_width_ratio", "top_margin_ratio", "bottom_margin_ratio"):
        meta[c] = geom[c].values
    meta.to_csv(out / "frames_meta.csv", index=False)
    print("кадры", frames.shape, "групп", meta["group"].nunique(),
          "spine", int((meta["region"] == "spine").sum()),
          "spine-групп", meta.loc[meta["region"] == "spine", "group"].nunique())


if __name__ == "__main__":
    main()
