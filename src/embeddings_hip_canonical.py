"""
Канонические эмбеддинги бедра: левое бедро зеркалится по горизонтали так, чтобы
любой снимок выглядел как ПРАВОЕ бедро (диафиз слева-снизу, таз справа-сверху).
Сторона определяется ПО ИЗОБРАЖЕНИЮ (hip_features.detect_hip_side), а не по
метке region из labels_full.csv (там сторона определялась по плотности и
ошибалась в ~24% случаев).

Это позволяет обучать ОДНУ модель позиционирования/ROI на объединённых
left+right (n≈330 вместо 2×165) — рекомендация Fable 5 про зеркалирование.

Выход:
  data/embeddings_hip_canonical.npy  (N_hip x 1280, float32)
  data/labels_hip_canonical.csv      (file_path, study, hip_side_detected)
Порядок строк = порядок бедренных строк в data/labels_full.csv.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd

from geometry_features import read_dicom_normalized
from hip_features import detect_hip_side
from embeddings import FrozenBackbone

DATA = Path("/home/user/workspace/densito_rebuild/data")


def canonical_hip_image(img_u8, side=None):
    if side is None:
        side = detect_hip_side(img_u8)
    return (img_u8[:, ::-1].copy() if side == 'left' else img_u8), side


def main():
    df = pd.read_csv(DATA / 'labels_full.csv')
    df = df[df['region'].isin(['right_hip', 'left_hip'])].reset_index(drop=True)
    bb = FrozenBackbone()
    embs, sides = [], []
    for i, r in df.iterrows():
        img, _ = read_dicom_normalized(r['file_path'])
        canon, side = canonical_hip_image(img)
        embs.append(bb.extract(canon))
        sides.append(side)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(df)}")
    np.save(DATA / 'embeddings_hip_canonical.npy', np.stack(embs).astype(np.float32))
    pd.DataFrame({'file_path': df['file_path'], 'study': df['study'],
                  'hip_side_detected': sides}).to_csv(DATA / 'labels_hip_canonical.csv', index=False)
    print("saved", len(embs))


if __name__ == '__main__':
    main()
