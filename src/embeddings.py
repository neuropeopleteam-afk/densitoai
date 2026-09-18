"""
Контур B — замороженный EfficientNet-B0 как экстрактор эмбеддингов
(НЕ fine-tuning), согласно рекомендации ревью Fable 5. Обучение поверх
эмбеддингов — логрегрессия с L2, устойчивая на малых данных.

Два источника весов (см. models/MODEL_CONTRACT.md, раздел «Бэкбоны»):
  * ``imagenet`` — torchvision EfficientNet-B0 IMAGENET1K_V1 (по умолчанию);
  * ``densito``  — тот же B0, предобученный нами на GPU на 15,6 тыс. рентген/DXA-фрагментах
    кости прокси-задачами (угол поворота, сдвиг, масштаб, синтетический металл),
    файл models/backbone_densito.pth (state_dict без classifier). На нашей разметке даёт
    устойчивый выигрыш только для критериев укладки (sp_pos, hip_pos) — для них и используется;
    для артефактов/оси/ROI остаётся ImageNet (gpu/eval_embeddings.py, 10 повторов GroupKFold).
"""
import warnings
warnings.filterwarnings("ignore")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, models

# отключает бэкенд NNPACK на уровне C++ — на CPU без поддержки специфичных
# инструкций torch иначе печатает предупреждение "Could not initialize NNPACK" на
# каждой свёртке; на корректность не влияет — просто выбирает другой backend для conv2d.
try:
    torch.backends.nnpack.set_flags(False)
except Exception:
    pass

from geometry_features import read_dicom_normalized

DEVICE = torch.device('cpu')


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BACKBONE_FILES = {"densito": "backbone_densito.pth"}
EMB_DIM = 1280


def backbone_path(source: str) -> Path | None:
    """Путь к файлу весов для источника (None для imagenet)."""
    if source == "imagenet":
        return None
    return MODELS_DIR / BACKBONE_FILES[source]


class FrozenBackbone:
    def __init__(self, source: str = "imagenet", weights=None):
        """source: 'imagenet' | 'densito'. weights — явный путь к state_dict (переопределяет source)."""
        self.source = source
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Identity()  # используем как чистый экстрактор (1280-d)
        wpath = Path(weights) if weights is not None else backbone_path(source)
        if wpath is not None:
            if not wpath.exists():
                raise FileNotFoundError(f"backbone weights not found: {wpath}")
            sd = torch.load(wpath, map_location="cpu")
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            if unexpected or any(not k.startswith("classifier") for k in missing):
                raise RuntimeError(f"backbone state_dict mismatch: missing={missing[:3]} unexpected={unexpected[:3]}")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            # паддинг до фиксированного прямоугольника с учётом анизотропии
            # пикселя (1.75 : 1), а не квадратный ресайз (рекомендация ревью)
            transforms.Resize((320, 192)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, img_u8):
        img_3ch = np.stack([img_u8, img_u8, img_u8], axis=2)
        tensor = self.tf(img_3ch).unsqueeze(0).to(DEVICE)
        emb = self.model(tensor)
        return emb.squeeze(0).numpy()


def build_embeddings_for_df(df, out_path, source: str = "imagenet"):
    backbone = FrozenBackbone(source)
    embs = []
    for i, row in df.iterrows():
        try:
            img_u8, _ = read_dicom_normalized(row['file_path'])
            emb = backbone.extract(img_u8)
        except Exception as e:
            print(f"ERROR on {row['file_path']}: {e}")
            emb = np.zeros(EMB_DIM, dtype=np.float32)
        embs.append(emb)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(df)} embeddings done")

    emb_arr = np.stack(embs)
    np.save(out_path, emb_arr)
    print(f"Saved embeddings {emb_arr.shape} to {out_path}")
    return emb_arr


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Построить эмбеддинги для labels_full.csv")
    ap.add_argument("--source", default="imagenet", choices=["imagenet"] + list(BACKBONE_FILES))
    a = ap.parse_args()
    DATA_DIR = Path(__file__).resolve().parent.parent / "data"
    df = pd.read_csv(DATA_DIR / "labels_full.csv")
    df = df[df['region'].isin(['spine', 'right_hip', 'left_hip'])].reset_index(drop=True)
    lab = DATA_DIR / "labels_for_embeddings.csv"
    if lab.exists():
        old = pd.read_csv(lab)
        assert (old['file_path'].values == df['file_path'].values).all(), "порядок строк labels_for_embeddings.csv изменился"
    else:
        df.to_csv(lab, index=False)
    out = DATA_DIR / ("embeddings.npy" if a.source == "imagenet" else f"embeddings_{a.source}.npy")
    build_embeddings_for_df(df, out, a.source)
