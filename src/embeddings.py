"""
Контур B — замороженный EfficientNet-B0 как экстрактор эмбеддингов
(НЕ fine-tuning), согласно рекомендации ревью Fable 5. Обучение поверх
эмбеддингов — логрегрессия с L2, устойчивая на малых данных.
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


class FrozenBackbone:
    def __init__(self):
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Identity()  # используем как чистый экстрактор (1280-d)
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


def build_embeddings_for_df(df, out_path):
    backbone = FrozenBackbone()
    embs = []
    for i, row in df.iterrows():
        try:
            img_u8, _ = read_dicom_normalized(row['file_path'])
            emb = backbone.extract(img_u8)
        except Exception as e:
            print(f"ERROR on {row['file_path']}: {e}")
            emb = np.zeros(1280, dtype=np.float32)
        embs.append(emb)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(df)} embeddings done")

    emb_arr = np.stack(embs)
    np.save(out_path, emb_arr)
    print(f"Saved embeddings {emb_arr.shape} to {out_path}")
    return emb_arr


if __name__ == '__main__':
    DATA_CSV = Path("/home/user/workspace/densito_rebuild/data/labels_full.csv")
    OUT_NPY = Path("/home/user/workspace/densito_rebuild/data/embeddings.npy")
    df = pd.read_csv(DATA_CSV)
    df = df[df['region'].isin(['spine', 'right_hip', 'left_hip'])].reset_index(drop=True)
    df.to_csv("/home/user/workspace/densito_rebuild/data/labels_for_embeddings.csv", index=False)
    build_embeddings_for_df(df, OUT_NPY)
