"""
Обучение мультилейбл CNN (EfficientNet-B0) для DensitoAI по 7 критериям:
  spine: sp_pos, sp_axis, sp_art
  right_hip: rh_pos, rh_roi
  left_hip: lh_pos, lh_roi

Отдельная модель на регион (архитектура едина), голова — sigmoid по
применимым для региона критериям. Пропуски (NaN) маскируются в лоссе
(BCEWithLogits с per-sample mask), как явно подтвердили организаторы
(пропуск = неприменимый критерий, не 0 и не 1).

5-fold GroupKFold по study, OOF-предсказания сохраняются для честных метрик.
"""
import warnings
warnings.filterwarnings("ignore")
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import GroupKFold

torch.manual_seed(42)
np.random.seed(42)

DATA_CSV = Path("/home/user/workspace/densito_rebuild/data/labels_full.csv")
OUT_DIR = Path("/home/user/workspace/densito_rebuild/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = Path("/home/user/workspace/densito_rebuild/train.log")

IMG_SIZE = 224
N_FOLDS = 5
EPOCHS = 12
BATCH_SIZE = 8
LR = 1e-4
DEVICE = torch.device('cpu')

REGION_CRITERIA = {
    'spine': ['sp_pos', 'sp_axis', 'sp_art'],
    'right_hip': ['rh_pos', 'rh_roi'],
    'left_hip': ['lh_pos', 'lh_roi'],
}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


def read_dicom_img(path):
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array.astype(np.float32)
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
    else:
        arr = np.zeros_like(arr)
    return arr.astype(np.uint8)


class DXADataset(Dataset):
    def __init__(self, df, criteria, augment=False):
        self.df = df.reset_index(drop=True)
        self.criteria = criteria
        self.augment = augment
        self.base_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
        ])
        if augment:
            self.aug_tf = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.0),  # anatomy is side-specific; keep off by default
                transforms.RandomRotation(3),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ])
        else:
            self.aug_tf = None
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = read_dicom_img(row['file_path'])
        img_3ch = np.stack([img, img, img], axis=2)
        pil = self.base_tf(img_3ch)
        if self.aug_tf is not None:
            pil = self.aug_tf(pil)
        tensor = self.to_tensor(pil)

        labels = []
        mask = []
        for c in self.criteria:
            v = row[c]
            if pd.isna(v):
                labels.append(0.0)
                mask.append(0.0)
            else:
                labels.append(float(v))
                mask.append(1.0)
        return tensor, torch.tensor(labels, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


class MultiLabelCNN(nn.Module):
    def __init__(self, n_outputs):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_outputs),
        )

    def forward(self, x):
        return self.backbone(x)


def masked_bce_loss(logits, targets, mask, pos_weight=None):
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction='none', pos_weight=pos_weight
    )
    loss = loss * mask
    denom = mask.sum()
    if denom.item() == 0:
        return loss.sum() * 0.0
    return loss.sum() / denom


def train_region(region, df_region, criteria):
    log(f"=== Training region: {region} | criteria: {criteria} | n={len(df_region)} ===")
    groups = df_region['study'].values
    gkf = GroupKFold(n_splits=N_FOLDS)

    oof_preds = np.full((len(df_region), len(criteria)), np.nan)
    oof_mask = np.zeros((len(df_region), len(criteria)))

    # class balance -> pos_weight per criterion (computed globally, simple heuristic)
    pos_weights = []
    for c in criteria:
        vals = df_region[c].dropna()
        pos = (vals == 1).sum()
        neg = (vals == 0).sum()
        w = (neg / max(pos, 1)) if pos > 0 else 1.0
        w = min(w, 8.0)  # cap
        pos_weights.append(w)
    pos_weight_t = torch.tensor(pos_weights, dtype=torch.float32)
    log(f"pos_weights for {region}: {dict(zip(criteria, pos_weights))}")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_region, groups=groups)):
        t0 = time.time()
        train_df = df_region.iloc[train_idx]
        val_df = df_region.iloc[val_idx]

        train_ds = DXADataset(train_df, criteria, augment=True)
        val_ds = DXADataset(val_df, criteria, augment=False)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = MultiLabelCNN(len(criteria)).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_val_loss = float('inf')
        best_state = None

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            for x, y, m in train_loader:
                x, y, m = x.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
                optimizer.zero_grad()
                logits = model(x)
                loss = masked_bce_loss(logits, y, m, pos_weight=pos_weight_t)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.size(0)
            train_loss /= len(train_ds)
            scheduler.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y, m in val_loader:
                    x, y, m = x.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
                    logits = model(x)
                    loss = masked_bce_loss(logits, y, m, pos_weight=pos_weight_t)
                    val_loss += loss.item() * x.size(0)
            val_loss /= max(len(val_ds), 1)

            log(f"  fold{fold} epoch{epoch+1}/{EPOCHS} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # save best fold model
        torch.save({'model_state_dict': best_state, 'criteria': criteria},
                    OUT_DIR / f'cnn_ml_{region}_fold{fold}.pth')

        # OOF predictions with best model
        model.load_state_dict(best_state)
        model.eval()
        val_ds_noaug = DXADataset(val_df, criteria, augment=False)
        val_loader_noaug = DataLoader(val_ds_noaug, batch_size=BATCH_SIZE, shuffle=False)
        preds = []
        with torch.no_grad():
            for x, y, m in val_loader_noaug:
                x = x.to(DEVICE)
                logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.append(probs)
        preds = np.concatenate(preds, axis=0)
        oof_preds[val_idx] = preds
        for j, c in enumerate(criteria):
            oof_mask[val_idx, j] = (~df_region.iloc[val_idx][c].isna()).astype(float).values

        log(f"  fold{fold} done in {time.time()-t0:.1f}s, best_val_loss={best_val_loss:.4f}")

    # save OOF
    oof_df = df_region[['study', 'file_path'] + criteria].copy().reset_index(drop=True)
    for j, c in enumerate(criteria):
        oof_df[f'{c}_pred'] = oof_preds[:, j]
    oof_df.to_csv(OUT_DIR / f'oof_{region}.csv', index=False)
    log(f"=== Region {region} done. OOF saved. ===")
    return oof_df


def main():
    log("Loading dataset...")
    df = pd.read_csv(DATA_CSV)
    df = df[df['region'].isin(['spine', 'right_hip', 'left_hip'])].reset_index(drop=True)
    log(f"Total images: {len(df)}")

    all_oof = {}
    for region, criteria in REGION_CRITERIA.items():
        df_region = df[df['region'] == region].reset_index(drop=True)
        oof_df = train_region(region, df_region, criteria)
        all_oof[region] = oof_df

    log("ALL TRAINING DONE")


if __name__ == '__main__':
    main()
