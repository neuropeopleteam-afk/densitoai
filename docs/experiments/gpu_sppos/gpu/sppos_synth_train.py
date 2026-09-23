"""Синтетика для sp_pos: обучение головы «дефект укладки позвоночника» на GPU (версия 3, окно с масштабом,
независимым от метки).

Идея. Дефект укладки («позвоночник смещён от центра» / «поясничный отдел не полностью в кадре») —
это сдвиг или обрезка правильно уложенного кадра. История версий:
  v1 (gpu/_v1_fill_based_sppos_synth_train.py): сдвиг кадра с дорисовкой фона -> сеть выучила артефакты
     дорисовки (синтетический AUC 0.99, на реальных кадрах скор ~0 у всех).
  v2 (gpu/_v2_window_scale_confounded_sppos_synth_train.py): окно исходного кадра с масштабом s ~ U(0.65, 1),
     обрезка = 1 - s -> метка «дефект» почти детерминирована масштабом окна (крупнее позвонки = дефект);
     на реальных кадрах (s = 1) скор ~0 у всех.
  v3 (этот файл): сначала метка (p = 0.5), затем масштаб окна s ~ U(0.60, 0.80) НЕЗАВИСИМО от метки, затем
     положение окна: смещение центра окна относительно центра кадра ox, oy в долях ширины/высоты окна
     (oy > 0 — окно выше, обрезан низ = таз). Норма: |ox|, |oy| < 0.08; дефект: хотя бы одно > 0.12
     (порог TX_THR = TY_THR = 0.10: 99-й процентиль |center_offset_ratio| нормальных кадров
     geometry_features_canonical.csv = 0.10; таз занимает нижние ~10-15 % нормального кадра). Дорисовки нет.
     Отвлекающие факторы, не меняющие метку: масштаб, поворот +-3 градуса, гамма 0.8-1.25, контраст +-15 %,
     шум, размытие, горизонтальное отражение.
Голова: [logit дефекта, ox, oy, s] — абсолютное положение окна относительно границ исходного кадра.
Метки ТЗ (sp_pos и др.) в обучении НЕ используются; синтез применяется ко всем кадрам позвоночника.

Режимы: --mode finetune (бэкбон densito дообучается) | --mode head (бэкбон заморожен, учится только голова).
Протокол: GroupKFold(5) по группам (исследование, хэш пикселей) кадров позвоночника ->
  AUC синтетической валидации (кадры вне обучения x 16 фиксированных преобразований) и
  OOF-эмбеддинги/скор для реальных кадров (бэкбон фолда не видел эти кадры); затем финальное обучение на всех
  кадрах позвоночника -> models/backbone_densito_synth.pth, эмбеддинги 499 кадров, скор головы.

Метаморфика считается двумя способами: окно (без дорисовки, сдвиг до 0.15 / обрезка до 0.25) и
дорисовка фоном кадра (сдвиг до 0.25, как в постановке задачи; вне обучающего распределения).

Выход (<out>/):
  cv_metrics.json, fold_k/ (история), backbone_synth.pth, head_synth.pth,
  embeddings_synth_canonical.npy (499x1280, порядок = labels_for_embeddings.csv),
  embeddings_synth_oof_canonical.npy, scores_synth.csv (row_id, file_path, region, p_defect, pred_ox, pred_oy,
  pred_s, *_oof, fold), metamorphic.json, pipeline_check.json.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torchvision import models

H, W = 320, 192
S_MIN, S_MAX = 0.60, 0.80            # масштаб окна, одинаково распределён для обоих классов
OFF_MAX = 0.35                       # нормировка регрессии относительных смещений
TX_MAX, TY_MAX = 0.25, 0.25          # только для метаморфики с дорисовкой (версия 1)
TX_THR, TY_THR = 0.10, 0.10          # порог дефекта: смещение центра окна > 10 % его ширины/высоты
OK_MAX, DEF_MIN = 0.08, 0.12         # норма: |смещение| < 0.08; дефект: > 0.12 (полоса 0.08-0.12 не сэмплируется)
ROT_MAX = 3.0
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def log(msg, path):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(path, "a") as f:
        f.write(line + "\n")


# ----------------------------------------------------------------------------- синтез
def sample_params(B, gen, dev, tx=None, ty=None, rot=None):
    """Параметры преобразования батча. tx/ty/rot можно задать явно (метаморфика)."""
    def U(lo, hi, shape=(B,)):
        return torch.rand(shape, generator=gen, device=dev) * (hi - lo) + lo
    if tx is None:
        small = torch.rand(B, generator=gen, device=dev) < 0.5
        mag = torch.where(small, U(0.0, 0.06), U(0.06, TX_MAX))
        sign = torch.where(torch.rand(B, generator=gen, device=dev) < 0.5, -1.0, 1.0)
        tx = mag * sign
    if ty is None:
        small = torch.rand(B, generator=gen, device=dev) < 0.5
        mag = torch.where(small, U(0.0, 0.03), U(0.03, TY_MAX))
        sign = torch.where(torch.rand(B, generator=gen, device=dev) < 0.7, 1.0, -1.0)
        ty = mag * sign
    if rot is None:
        rot = U(-ROT_MAX, ROT_MAX)
    return tx, ty, rot


def apply_geometry(x, tx, ty, rot, fill_kind, gen):
    """x: (B,1,H,W) float [0,1]. tx — доля ширины (+ вправо), ty — доля высоты (+ вниз), rot — градусы.
    fill_kind: (B,) int 0=reflection, 1=border, 2=фон кадра + шум."""
    B, dev = x.shape[0], x.device
    phi = rot * math.pi / 180.0
    cos, sin = torch.cos(phi), torch.sin(phi)
    a11, a12 = cos, sin * (H / W)
    a21, a22 = -sin * (W / H), cos
    tnx, tny = 2 * tx, 2 * ty
    b1 = -(a11 * tnx + a12 * tny)
    b2 = -(a21 * tnx + a22 * tny)
    theta = torch.stack([torch.stack([a11, a12, b1], 1), torch.stack([a21, a22, b2], 1)], 1)
    grid = F.affine_grid(theta, (B, 1, H, W), align_corners=False)
    y_ref = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    y_bor = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    y_zero = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    inside = F.grid_sample(torch.ones_like(x), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    # фон кадра: квантиль 20 % пикселей + шум того же порядка, что разброс тёмных пикселей
    flat = x.flatten(1)
    bg = torch.quantile(flat, 0.20, dim=1).view(B, 1, 1, 1)
    bg_std = (torch.quantile(flat, 0.35, dim=1) - torch.quantile(flat, 0.05, dim=1)).view(B, 1, 1, 1) / 2
    noise = torch.randn(x.shape, generator=gen, device=dev) * bg_std
    y_bg = y_zero + (1 - inside) * (bg + noise).clamp(0, 1)
    k = fill_kind.view(B, 1, 1, 1)
    return torch.where(k == 0, y_ref, torch.where(k == 1, y_bor, y_bg))


def photometric(x, gen, gamma=None):
    B, dev = x.shape[0], x.device
    if gamma is None:
        gamma = torch.exp((torch.rand(B, generator=gen, device=dev) * 2 - 1) * 0.22)
    x = x.clamp(1e-4, 1) ** gamma.view(B, 1, 1, 1)
    c = 1 + (torch.rand(B, generator=gen, device=dev) * 2 - 1) * 0.15
    m = x.mean(dim=(2, 3), keepdim=True)
    x = ((x - m) * c.view(B, 1, 1, 1) + m).clamp(0, 1)
    noise = torch.randn(x.shape, generator=gen, device=dev) * (torch.rand(B, 1, 1, 1, generator=gen, device=dev) * 0.02)
    x = (x + noise).clamp(0, 1)
    blur = torch.rand(B, generator=gen, device=dev) < 0.2
    if blur.any():
        x[blur] = F.avg_pool2d(x[blur], 3, stride=1, padding=1)
    return x


def normalize3(x):
    x3 = x.repeat(1, 3, 1, 1)
    return (x3 - MEAN.to(x.device)) / STD.to(x.device)


def sample_window(B, gen, dev, s=None, ox=None, oy=None):
    """Версия 3. Сначала метка (p=0.5), затем масштаб s ~ U(S_MIN, S_MAX) НЕЗАВИСИМО от метки, затем положение окна.
    ox, oy — смещение центра окна относительно центра кадра в долях ширины/высоты ОКНА (oy > 0: окно выше,
    т.е. обрезан низ). Норма: |ox|, |oy| < OK_MAX; дефект: хотя бы одно > DEF_MIN (вертикальный — 50 %,
    горизонтальный — 25 %, оба — 25 %; вертикальный сдвиг вверх в 75 % случаев).
    Возвращает s, tx (доля ширины кадра), bottom, top (доли высоты кадра), ox, oy."""
    def U(lo, hi):
        return torch.rand(B, generator=gen, device=dev) * (hi - lo) + lo
    if s is None:
        s = U(S_MIN, S_MAX)
    half = (1.0 - s) / 2
    rmax = half / s                                   # максимально возможное относительное смещение
    if ox is None or oy is None:
        d = torch.rand(B, generator=gen, device=dev) < 0.5
        kind = torch.rand(B, generator=gen, device=dev)      # <0.5 вертикальный, <0.75 горизонтальный, иначе оба
        vert = d & ((kind < 0.5) | (kind >= 0.75))
        horz = d & (kind >= 0.5)
        mag_ok_x, mag_ok_y = U(0, OK_MAX), U(0, OK_MAX)
        mag_def_x = DEF_MIN + U(0, 1) * (rmax - DEF_MIN).clamp(min=0)
        mag_def_y = DEF_MIN + U(0, 1) * (rmax - DEF_MIN).clamp(min=0)
        sx = torch.where(torch.rand(B, generator=gen, device=dev) < 0.5, -1.0, 1.0)
        sy = torch.where(torch.rand(B, generator=gen, device=dev) < 0.75, 1.0, -1.0)
        ox_s = torch.where(horz, mag_def_x, mag_ok_x) * sx
        oy_s = torch.where(vert, mag_def_y, mag_ok_y) * sy
        ox = ox_s if ox is None else ox
        oy = oy_s if oy is None else oy
    ox = torch.clamp(ox, -rmax, rmax)
    oy = torch.clamp(oy, -rmax, rmax)
    tx = ox * s
    bottom = half + oy * s
    top = half - oy * s
    return s, tx, bottom, top, ox, oy


def apply_window(x, s, tx, bottom, top, rot):
    """Окно исходного кадра (без дорисовки; при повороте края — отражение) -> (B,1,H,W)."""
    B = x.shape[0]
    cy = 0.5 - s / 2 - bottom                        # центр окна по вертикали, доля высоты от центра кадра
    phi = rot * math.pi / 180.0
    cos, sin = torch.cos(phi), torch.sin(phi)
    a11, a12 = s * cos, s * sin * (H / W)
    a21, a22 = -s * sin * (W / H), s * cos
    theta = torch.stack([torch.stack([a11, a12, 2 * tx], 1), torch.stack([a21, a22, 2 * cy], 1)], 1)
    grid = F.affine_grid(theta, (B, 1, H, W), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def synth_batch(x_u8, gen, train=True, s=None, ox=None, oy=None, rot=None, gamma=None, photo=True, flip=True):
    """x_u8: (B,H,W) uint8 на GPU -> (B,3,H,W) вход сети, таргеты (ox, oy, s, defect)."""
    x = x_u8.float().unsqueeze(1) / 255.0
    B, dev = x.shape[0], x.device
    if flip:
        fl = torch.rand(B, generator=gen, device=dev) < 0.5
        x = torch.where(fl.view(B, 1, 1, 1), x.flip(-1), x)
    s, tx, bottom, top, ox, oy = sample_window(B, gen, dev, s, ox, oy)
    if rot is None:
        rot = (torch.rand(B, generator=gen, device=dev) * 2 - 1) * ROT_MAX
    x = apply_window(x, s, tx, bottom, top, rot)
    if photo:
        x = photometric(x, gen, gamma)
    defect = ((ox.abs() > TX_THR) | (oy.abs() > TY_THR)).float()
    return normalize3(x), ox, oy, s, defect


def synth_batch_fill(x_u8, gen, train=True, tx=None, ty=None, rot=None, gamma=None, fill=None, photo=True, flip=True):
    """Версия 1 (сдвиг с дорисовкой) — только для метаморфики вне обучающего распределения."""
    """x_u8: (B,H,W) uint8 на GPU -> (B,3,H,W) вход сети, таргеты (tx, ty, defect)."""
    x = x_u8.float().unsqueeze(1) / 255.0
    B, dev = x.shape[0], x.device
    if flip:
        fl = torch.rand(B, generator=gen, device=dev) < 0.5
        x = torch.where(fl.view(B, 1, 1, 1), x.flip(-1), x)
    tx, ty, rot = sample_params(B, gen, dev, tx, ty, rot)
    if fill is None:
        fill = torch.randint(0, 3, (B,), generator=gen, device=dev)
    x = apply_geometry(x, tx, ty, rot, fill, gen)
    if photo:
        x = photometric(x, gen, gamma)
    defect = ((tx.abs() > TX_THR) | (ty.abs() > TY_THR)).float()
    return normalize3(x), tx, ty, defect


def plain_batch(x_u8):
    """Кадр без преобразований — ровно вход FrozenBackbone (ToTensor + Normalize)."""
    x = x_u8.float().unsqueeze(1) / 255.0
    return normalize3(x)


# ----------------------------------------------------------------------------- модель
class SynthNet(nn.Module):
    def __init__(self, backbone_weights: str):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        self.backbone.classifier = nn.Identity()
        sd = torch.load(backbone_weights, map_location="cpu")
        missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.startswith("classifier") for k in missing), (missing[:3], unexpected[:3])
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 256), nn.GELU(), nn.Linear(256, 4))

    def forward(self, x):
        f = self.backbone(x)
        return f, self.head(f)


def backbone_state(model):
    return {k: v.detach().cpu() for k, v in model.backbone.state_dict().items() if not k.startswith("classifier")}


def loss_fn(out, ox, oy, s, defect):
    l_cls = F.binary_cross_entropy_with_logits(out[:, 0], defect)
    l_reg = (F.smooth_l1_loss(out[:, 1], ox / OFF_MAX, beta=0.1) + F.smooth_l1_loss(out[:, 2], oy / OFF_MAX, beta=0.1)
             + 0.5 * F.smooth_l1_loss(out[:, 3], (s - 0.7) / 0.1, beta=0.1))
    return l_cls + l_reg, l_cls, l_reg


def train_model(Xg, tr_idx, a, dev, logp, seed, val_fn=None):
    model = SynthNet(a.backbone).to(dev).to(memory_format=torch.channels_last)
    if a.mode == "head":
        for p in model.backbone.parameters():
            p.requires_grad = False
        params = [{"params": model.head.parameters(), "lr": a.lr_head}]
    else:
        params = [{"params": model.backbone.parameters(), "lr": a.lr_backbone},
                  {"params": model.head.parameters(), "lr": a.lr_head}]
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    total = a.epochs * a.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in params], total_steps=total,
                                                pct_start=0.1, div_factor=10, final_div_factor=50)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    rng = np.random.default_rng(seed)
    hist = []
    for ep in range(a.epochs):
        model.train()
        if a.mode == "head":
            model.backbone.eval()
        t0, tot, tc, tr_ = time.time(), 0.0, 0.0, 0.0
        for _ in range(a.steps):
            idx = torch.from_numpy(rng.choice(tr_idx, size=a.bs, replace=True)).to(dev)
            with torch.no_grad():
                xb, ox, oy, sc_, d = synth_batch(Xg[idx], gen, train=True)
            xb = xb.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if a.mode == "head":
                    with torch.no_grad():
                        f = model.backbone(xb)
                    out = model.head(f.float())
                else:
                    _, out = model(xb)
            out = out.float()
            loss, lc, lr_ = loss_fn(out, ox, oy, sc_, d)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step(); sched.step()
            tot += loss.item(); tc += lc.item(); tr_ += lr_.item()
        rec = dict(epoch=ep, loss=tot / a.steps, loss_cls=tc / a.steps, loss_reg=tr_ / a.steps, sec=time.time() - t0)
        if val_fn is not None and (ep % 5 == 4 or ep == a.epochs - 1):
            rec.update(val_fn(model))
        hist.append(rec)
        log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), logp)
    return model, hist


@torch.no_grad()
def eval_synth(model, Xg, idx, dev, n_rep=16, seed=7, bs=128):
    """Синтетическая валидация: каждый кадр x n_rep фиксированных преобразований."""
    model.eval()
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    P, OX, OY, SS, D, PX, PY, PS = [], [], [], [], [], [], [], []
    for _ in range(n_rep):
        for i in range(0, len(idx), bs):
            b = torch.from_numpy(idx[i:i + bs]).to(dev)
            xb, ox, oy, sc_, d = synth_batch(Xg[b], gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, out = model(xb.contiguous(memory_format=torch.channels_last))
            out = out.float()
            P.append(out[:, 0].cpu()); PX.append(out[:, 1].cpu() * OFF_MAX); PY.append(out[:, 2].cpu() * OFF_MAX)
            PS.append(out[:, 3].cpu() * 0.1 + 0.7)
            OX.append(ox.cpu()); OY.append(oy.cpu()); SS.append(sc_.cpu()); D.append(d.cpu())
    P, OX, OY, SS, D, PX, PY, PS = map(torch.cat, (P, OX, OY, SS, D, PX, PY, PS))
    res = dict(val_auc_defect=float(roc_auc_score(D.numpy(), P.numpy())),
               val_mae_ox=float((PX - OX).abs().mean()), val_mae_oy=float((PY - OY).abs().mean()),
               val_mae_s=float((PS - SS).abs().mean()), val_pos_rate=float(D.mean()),
               val_auc_defect_by_s_only=float(roc_auc_score(D.numpy(), -SS.numpy())))
    # главный реальный сценарий: только вертикальный сдвиг вверх (обрезан низ) при |ox| <= OK_MAX
    m = (OX.abs() <= OK_MAX) & (OY >= 0)
    if m.sum() > 10 and len(torch.unique(D[m])) == 2:
        res["val_auc_defect_bottom_only"] = float(roc_auc_score(D[m].numpy(), P[m].numpy()))
        res["val_auc_bottom_by_pred_oy"] = float(roc_auc_score(D[m].numpy(), PY[m].numpy()))
    return res


@torch.no_grad()
def infer_plain(model, Xg, idx, dev, bs=128):
    """Эмбеддинги (1280) и выходы головы для кадров без преобразований (ровно как FrozenBackbone)."""
    model.eval()
    E, O = [], []
    for i in range(0, len(idx), bs):
        b = torch.from_numpy(idx[i:i + bs]).to(dev)
        f, out = model(plain_batch(Xg[b]))      # fp32, без autocast — как в src/embeddings.py
        E.append(f.float().cpu()); O.append(out.float().cpu())
    return torch.cat(E).numpy(), torch.cat(O).numpy()


@torch.no_grad()
def _scores(model, Xg, idx, dev, make, bs=128):
    sc = []
    for i in range(0, len(idx), bs):
        b = torch.from_numpy(idx[i:i + bs]).to(dev)
        xb = make(Xg[b])
        _, o = model(xb)
        sc.append(torch.sigmoid(o[:, 0]).float().cpu())
    return torch.cat(sc).numpy()


def _mono_stats(S, grid, prefix, out):
    d = np.diff(S, axis=1)
    out[f"{prefix}_grid"] = [float(g) for g in grid]
    out[f"{prefix}_frac_monotone_nondecreasing"] = float(np.mean(np.all(d >= -0.01, axis=1)))
    out[f"{prefix}_frac_strictly_increasing"] = float(np.mean(np.all(d > 0, axis=1)))
    out[f"{prefix}_frac_end_above_start"] = float(np.mean(S[:, -1] > S[:, 0]))
    out[f"{prefix}_mean_score_by_grid"] = [float(v) for v in S.mean(0)]
    out[f"{prefix}_spearman_mean"] = float(np.nanmean([pd.Series(row).corr(pd.Series(grid), method="spearman") for row in S]))


@torch.no_grad()
def metamorphic(model, Xg, idx, dev):
    """Монотонность скора при сдвиге/обрезке и устойчивость к гамме.
    window_*: окно без дорисовки (как в обучении): s=0.7, относительное смещение центра 0..0.20;
    fill_*:   сдвиг 0..0.25 / обрезка снизу 0..0.20 с дорисовкой фоном кадра (постановка задачи; вне обучения)."""
    model.eval()
    gen = torch.Generator(device=dev); gen.manual_seed(11)
    out = {}
    z = lambda n, v=0.0: torch.full((n,), float(v), device=dev)
    # окно s=0.7 (как в обучении): относительное смещение центра 0..0.20 по горизонтали / вверх по вертикали
    grid_ox = np.arange(0, 0.201, 0.04)
    S = np.stack([_scores(model, Xg, idx, dev, lambda xb, v=v: synth_batch(
        xb, gen, s=z(len(xb), 0.7), ox=z(len(xb), v), oy=z(len(xb)), rot=z(len(xb)), photo=False, flip=False)[0])
        for v in grid_ox], 1)
    _mono_stats(S, grid_ox, "window_ox", out)
    grid_oy = np.arange(0, 0.201, 0.04)
    S = np.stack([_scores(model, Xg, idx, dev, lambda xb, v=v: synth_batch(
        xb, gen, s=z(len(xb), 0.7), ox=z(len(xb)), oy=z(len(xb), v), rot=z(len(xb)), photo=False, flip=False)[0])
        for v in grid_oy], 1)
    _mono_stats(S, grid_oy, "window_oy_up", out)
    # тот же сдвиг вверх при другом масштабе окна s=0.6 и s=0.8 (проверка, что скор не сводится к масштабу)
    for sv in (0.6, 0.8):
        S = np.stack([_scores(model, Xg, idx, dev, lambda xb, v=v, sv=sv: synth_batch(
            xb, gen, s=z(len(xb), sv), ox=z(len(xb)), oy=z(len(xb), v), rot=z(len(xb)), photo=False, flip=False)[0])
            for v in grid_oy], 1)
        _mono_stats(S, grid_oy, f"window_oy_up_s{sv}", out)
    # дорисовка фоном кадра (fill=2), без фотометрии
    grid_tx2 = np.arange(0, 0.251, 0.05)
    S = np.stack([_scores(model, Xg, idx, dev, lambda xb, v=v: synth_batch_fill(
        xb, gen, tx=z(len(xb), v), ty=z(len(xb)), rot=z(len(xb)), fill=torch.full((len(xb),), 2, device=dev, dtype=torch.long),
        photo=False, flip=False)[0]) for v in grid_tx2], 1)
    _mono_stats(S, grid_tx2, "fill_tx", out)
    grid_ty2 = np.arange(0, 0.201, 0.04)
    S = np.stack([_scores(model, Xg, idx, dev, lambda xb, v=v: synth_batch_fill(
        xb, gen, tx=z(len(xb)), ty=z(len(xb), v), rot=z(len(xb)), fill=torch.full((len(xb),), 2, device=dev, dtype=torch.long),
        photo=False, flip=False)[0]) for v in grid_ty2], 1)
    _mono_stats(S, grid_ty2, "fill_bottom", out)
    # гамма 0.8-1.2 без геометрии: доля переворотов решения при пороге = квантиль 0.9 скоров при гамме 1
    gam = [0.8, 0.9, 1.0, 1.1, 1.2]
    S = np.stack([_scores(model, Xg, idx, dev, lambda xb, gv=gv: normalize3((xb.float().unsqueeze(1) / 255.0).clamp(1e-4, 1) ** gv))
                  for gv in gam], 1)
    ref = S[:, 2]
    thr = float(np.quantile(ref, 0.9))
    flips = ((S >= thr) != (ref >= thr)[:, None])[:, [0, 1, 3, 4]].any(1)
    out["gamma_grid"] = gam
    out["gamma_frac_decision_flips_q90"] = float(flips.mean())
    out["gamma_mean_abs_score_change"] = float(np.abs(S - ref[:, None]).mean())
    out["gamma_max_abs_score_change_median_over_frames"] = float(np.median(np.abs(S - ref[:, None]).max(1)))
    out["score_real_frames_mean"] = float(ref.mean())
    out["score_real_frames_q90"] = thr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/workspace/sppos/cache")
    ap.add_argument("--backbone", default="/workspace/sppos/backbone_densito.pth")
    ap.add_argument("--out", default="/workspace/sppos/out_finetune")
    ap.add_argument("--mode", default="finetune", choices=["finetune", "head"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--ref-emb", default=None, help="data/embeddings_densito_canonical.npy для проверки пайплайна")
    ap.add_argument("--seed", type=int, default=0, help="смещение зерна инициализации/сэмплирования (проверка устойчивости)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    logp = out / "train.log"
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = False

    X = np.load(Path(a.cache) / "frames_u8.npy")
    meta = pd.read_csv(Path(a.cache) / "frames_meta.csv")
    Xg = torch.from_numpy(X).to(dev)
    spine = np.nonzero((meta["region"] == "spine").values)[0]
    groups = meta["group"].values[spine]
    log(f"кадров {len(meta)}, позвоночник {len(spine)}, групп {len(np.unique(groups))}, режим {a.mode}, "
        f"порог |tx|>{TX_THR} cut>{TY_THR}, окно s in [{S_MIN},{S_MAX}]", logp)

    # 0. проверка пайплайна: исходный бэкбон на кэше == data/embeddings_densito_canonical.npy?
    if a.ref_emb and Path(a.ref_emb).exists():
        m0 = SynthNet(a.backbone).to(dev)
        E0, _ = infer_plain(m0, Xg, np.arange(len(meta)), dev)
        ref = np.load(a.ref_emb)
        diff = np.abs(E0 - ref)
        cos = (E0 * ref).sum(1) / (np.linalg.norm(E0, axis=1) * np.linalg.norm(ref, axis=1) + 1e-9)
        chk = dict(max_abs_diff=float(diff.max()), mean_abs_diff=float(diff.mean()), min_cosine=float(cos.min()),
                   ref_abs_mean=float(np.abs(ref).mean()))
        json.dump(chk, open(out / "pipeline_check.json", "w"), indent=1)
        log(f"проверка пайплайна против {Path(a.ref_emb).name}: {chk}", logp)
        del m0

    # 1. GroupKFold: синтетическая валидация + OOF-эмбеддинги/скоры
    n_all = len(meta)
    E_oof = np.zeros((n_all, 1280), dtype=np.float32)
    O_oof = np.zeros((n_all, 4), dtype=np.float32)
    fold_of = np.full(n_all, -1)
    cv = []
    meta_all = []
    gkf = GroupKFold(n_splits=a.n_folds, shuffle=True, random_state=42)
    for k, (tr, va) in enumerate(gkf.split(spine, groups=groups)):
        tr_idx, va_idx = spine[tr], spine[va]
        log(f"--- фолд {k}: train {len(tr_idx)} кадров, val {len(va_idx)}", logp)
        val_fn = lambda m, vi=va_idx: eval_synth(m, Xg, vi, dev)
        model, hist = train_model(Xg, tr_idx, a, dev, logp, seed=100 + k + 1000 * a.seed, val_fn=val_fn)
        res = eval_synth(model, Xg, va_idx, dev)
        res_tr = eval_synth(model, Xg, tr_idx, dev, n_rep=4, seed=8)
        res.update({f"train_{k2}": v for k2, v in res_tr.items()})
        res["fold"] = k
        cv.append(res)
        log(f"фолд {k}: {json.dumps({k2: round(v, 4) for k2, v in res.items()})}", logp)
        E, O = infer_plain(model, Xg, va_idx, dev)
        E_oof[va_idx], O_oof[va_idx], fold_of[va_idx] = E, O, k
        if k == 0:   # для строк бедра OOF-значений нет по смыслу; заполняем моделью фолда 0
            other = np.setdiff1d(np.arange(n_all), spine)
            E_oof[other], O_oof[other] = infer_plain(model, Xg, other, dev)
        meta_all.append(metamorphic(model, Xg, va_idx, dev))
        (out / f"fold_{k}").mkdir(exist_ok=True)
        json.dump(hist, open(out / f"fold_{k}" / "history.json", "w"), indent=1)
        del model; torch.cuda.empty_cache()
    cvdf = pd.DataFrame(cv)
    summary = {c: dict(mean=float(cvdf[c].mean()), std=float(cvdf[c].std())) for c in cvdf.columns if c != "fold"}
    json.dump(dict(folds=cv, summary=summary, mode=a.mode, epochs=a.epochs, steps=a.steps, bs=a.bs,
                   tx_thr=TX_THR, ty_thr=TY_THR), open(out / "cv_metrics.json", "w"), indent=1)
    log("CV summary: " + json.dumps({k2: round(v["mean"], 4) for k2, v in summary.items()}), logp)
    # метаморфика OOF-моделей (каждый кадр — моделью, которая его не видела): усреднение по фолдам
    mm_oof = {k2: float(np.mean([m[k2] for m in meta_all])) for k2 in meta_all[0] if isinstance(meta_all[0][k2], float)}
    np.save(out / "embeddings_synth_oof_canonical.npy", E_oof)

    # 2. финальная модель на всех кадрах позвоночника
    log("--- финальное обучение на всех кадрах позвоночника", logp)
    model, hist = train_model(Xg, spine, a, dev, logp, seed=999 + 1000 * a.seed)
    json.dump(hist, open(out / "history_final.json", "w"), indent=1)
    torch.save(backbone_state(model), out / "backbone_synth.pth")
    torch.save({k2: v.cpu() for k2, v in model.head.state_dict().items()}, out / "head_synth.pth")
    E, O = infer_plain(model, Xg, np.arange(n_all), dev)
    np.save(out / "embeddings_synth_canonical.npy", E.astype(np.float32))
    sc = pd.DataFrame(dict(row_id=np.arange(n_all), file_path=meta["file_path"], region=meta["region"],
                           p_defect=1 / (1 + np.exp(-O[:, 0])), pred_ox=O[:, 1] * OFF_MAX, pred_oy=O[:, 2] * OFF_MAX,
                           pred_s=O[:, 3] * 0.1 + 0.7,
                           p_defect_oof=1 / (1 + np.exp(-O_oof[:, 0])), pred_ox_oof=O_oof[:, 1] * OFF_MAX,
                           pred_oy_oof=O_oof[:, 2] * OFF_MAX, pred_s_oof=O_oof[:, 3] * 0.1 + 0.7, fold=fold_of))
    sc.to_csv(out / "scores_synth.csv", index=False)
    res_final_train = eval_synth(model, Xg, spine, dev, n_rep=4, seed=8)
    mm_final = metamorphic(model, Xg, spine, dev)
    json.dump(dict(final_model_all_spine_frames=mm_final, oof_models_mean_over_folds=mm_oof,
                   final_train_synth=res_final_train), open(out / "metamorphic.json", "w"), indent=1)
    log("метаморфика (финальная модель): " + json.dumps({k2: (round(v, 3) if isinstance(v, float) else v)
                                                        for k2, v in mm_final.items() if "grid" not in k2 and "by_grid" not in k2}), logp)
    log("DONE", logp)


if __name__ == "__main__":
    main()
