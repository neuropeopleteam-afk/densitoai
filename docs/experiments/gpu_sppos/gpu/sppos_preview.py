"""Превью синтетических кадров (визуальная проверка заполнения фона и сдвигов): <out>/synth_preview.png."""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sppos_synth_train import synth_batch, MEAN, STD

ap = argparse.ArgumentParser(); ap.add_argument("--cache", default="/workspace/sppos/cache"); ap.add_argument("--out", default="/workspace/sppos")
a = ap.parse_args()
X = np.load(Path(a.cache) / "frames_u8.npy"); meta = pd.read_csv(Path(a.cache) / "frames_meta.csv")
spine = np.nonzero((meta["region"] == "spine").values)[0]
dev = torch.device("cuda"); gen = torch.Generator(device=dev); gen.manual_seed(3)
idx = torch.from_numpy(spine[:12]).to(dev)
Xg = torch.from_numpy(X).to(dev)
xb, ox, oy, ss, d = synth_batch(Xg[idx], gen)
img = (xb[:, :1] * STD.to(dev)[:, :1] + MEAN.to(dev)[:, :1]).clamp(0, 1)[:, 0].cpu().numpy()
orig = X[spine[:12]]
top = np.concatenate(list(orig), 1); bot = np.concatenate([(im * 255).astype(np.uint8) for im in img], 1)
Image.fromarray(np.concatenate([top, bot], 0)).save(Path(a.out) / "synth_preview.png")
print(pd.DataFrame(dict(ox=ox.cpu().numpy().round(3), oy=oy.cpu().numpy().round(3), s=ss.cpu().numpy().round(3), defect=d.cpu().numpy())).T.to_string())
