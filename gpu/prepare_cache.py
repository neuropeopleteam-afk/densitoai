"""
Сборка кэша изображений для GPU-предобучения backbone DensitoAI.

Все источники (собственный DXA-датасет + открытые рентген/DXA наборы) приводятся к
единому виду, в котором работает `embeddings.FrozenBackbone`: серый uint8, 320x192
(H x W). Широкие/очень высокие снимки режутся на до 3 «вертикальных» плиток с
соотношением сторон 192/320, чтобы не было чёрных полей.

Выход: <out>/pretrain_u8.npy  (N, 320, 192) uint8
       <out>/pretrain_meta.csv (source, path, tile)
"""
import argparse
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

H, W = 320, 192
ASPECT = W / H  # 0.6

SRC_ROOT = Path("/workspace/data")
SOURCES = {
    # имя: (подкаталог, glob-маски)
    "own_dxa":  ("own_dataset", ("**/*.dcm",)),
    "arak_hip_dxa": ("kaggle/arak-bone-densitometry-center", ("**/*.png",)),
    "dexa_osteo_spine": ("kaggle/dexa-osteo", ("**/*.png", "**/*.PNG")),
    "aasce_spine": ("kaggle/aasce-miccai-2019-x-ray-dataset", ("**/*.jpg",)),
    "buu_lspine": ("BUU-LSPINE", ("**/*.jpg",)),
    "mtddh_pelvis": ("MTDDH", ("**/*.jpg",)),
    "fracatlas": ("FracAtlas", ("**/images/**/*.jpg", "**/*.jpg")),
    "cn_hip": ("chinese_osteoporosis/Hip", ("**/*.png", "**/*.jpg")),
    "cn_lumbar_ap": ("chinese_osteoporosis/LumbarP", ("**/*.png", "**/*.jpg")),
}


def read_gray(path: str):
    p = Path(path)
    if p.suffix.lower() == ".dcm":
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from geometry_features import read_dicom_normalized
        img, _ = read_dicom_normalized(str(p))
        return img
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    lo, hi = np.percentile(img, [1, 99])
    if hi > lo:
        img = np.clip((img.astype(np.float32) - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return img


def tiles_for(img: np.ndarray, is_dxa: bool):
    """DXA собственного формата: прямой resize (как в FrozenBackbone).
    Остальное: плитки с соотношением 0.6, максимум 3."""
    if is_dxa:
        return [cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)]
    h, w = img.shape[:2]
    out = []
    if w / h > ASPECT:  # шире, чем нужно -> режем по горизонтали
        tw = int(round(h * ASPECT))
        n = min(3, max(1, int(round(w / tw))))
        if n == 1:
            xs = [(w - tw) // 2]
        else:
            xs = np.linspace(0, w - tw, n).astype(int)
        for x in xs:
            out.append(cv2.resize(img[:, x:x + tw], (W, H), interpolation=cv2.INTER_AREA))
    else:  # выше, чем нужно -> режем по вертикали
        th = int(round(w / ASPECT))
        n = min(3, max(1, int(round(h / th))))
        if n == 1:
            ys = [(h - th) // 2]
        else:
            ys = np.linspace(0, h - th, n).astype(int)
        for y in ys:
            out.append(cv2.resize(img[y:y + th, :], (W, H), interpolation=cv2.INTER_AREA))
    return out


def process(args):
    src, path = args
    try:
        img = read_gray(path)
        if img is None or img.size == 0 or min(img.shape[:2]) < 32:
            return []
        return [(src, path, i, t) for i, t in enumerate(tiles_for(img, src == "own_dxa"))]
    except Exception as e:  # noqa: BLE001
        print("skip", path, e, file=sys.stderr)
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(SRC_ROOT))
    ap.add_argument("--out", default="/workspace/cache")
    ap.add_argument("--limit", type=int, default=0, help="для отладки: не более N файлов на источник")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    root = Path(a.root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for src, (sub, globs) in SOURCES.items():
        base = root / sub
        if not base.exists():
            print(f"[{src}] нет каталога {base} -> пропуск")
            continue
        files = []
        for g in globs:
            files += [str(p) for p in base.glob(g)]
        files = sorted(set(files))
        if a.limit:
            files = files[: a.limit]
        print(f"[{src}] файлов: {len(files)}")
        jobs += [(src, f) for f in files]

    arrs, meta = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for res in ex.map(process, jobs, chunksize=16):
            for src, path, i, t in res:
                arrs.append(t)
                meta.append((src, path, i))
    X = np.stack(arrs).astype(np.uint8)
    np.save(out / "pretrain_u8.npy", X)
    pd.DataFrame(meta, columns=["source", "path", "tile"]).to_csv(out / "pretrain_meta.csv", index=False)
    df = pd.DataFrame(meta, columns=["source", "path", "tile"])
    print("итого плиток:", X.shape, "\n", df["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
