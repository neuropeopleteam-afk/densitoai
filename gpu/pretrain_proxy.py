"""
Доменное предобучение backbone (EfficientNet-B0) на ~20k рентген/DXA-снимках
с СИНТЕТИЧЕСКИМИ геометрическими задачами — «прокси» критериев ТЗ:

  * поворот снимка   (аналог sp_axis / наклон оси кости)
  * сдвиг по X и Y   (аналог sp_pos / hip_pos — смещение объекта от центра)
  * масштаб          (ROI слишком мал/велик — hip_roi)
  * синтетический металлический артефакт (аналог sp_art)

Каждый батч: случайное аффинное преобразование через grid_sample (GPU), затем с
p=0.5 рисуются 1-3 «металлических» объекта (эллипс/стержень/кольцо). Сеть учится
предсказывать параметры преобразования и наличие артефакта => признаки backbone
становятся чувствительны ровно к тому, что нужно детектировать, вместо
ImageNet-инвариантности к поворотам/сдвигам.

Выход: ckpt/backbone_epXX.pth — state_dict EfficientNet-B0 без classifier
(drop-in для embeddings.FrozenBackbone), ckpt/last.pth — для возобновления.
"""
import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

H, W = 320, 192
ROT_MAX = 20.0      # градусов
SHIFT_MAX = 0.15    # доля ширины/высоты
SCALE_MIN, SCALE_MAX = 0.85, 1.20
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def log(msg, path):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(path, "a") as f:
        f.write(line + "\n")


# --------------------------------------------------------------------------- #
# Аугментации на GPU
# --------------------------------------------------------------------------- #
def random_affine(x: torch.Tensor, gen: torch.Generator):
    """x: (B,1,H,W) float [0,1]. Возвращает преобразованный тензор и таргеты
    (B,4): rot/ROT_MAX, tx/SHIFT_MAX, ty/SHIFT_MAX, (s-1)/(SCALE_MAX-1)."""
    B = x.shape[0]
    dev = x.device
    rot = (torch.rand(B, generator=gen, device=dev) * 2 - 1) * ROT_MAX
    tx = (torch.rand(B, generator=gen, device=dev) * 2 - 1) * SHIFT_MAX
    ty = (torch.rand(B, generator=gen, device=dev) * 2 - 1) * SHIFT_MAX
    s = SCALE_MIN + torch.rand(B, generator=gen, device=dev) * (SCALE_MAX - SCALE_MIN)

    phi = rot * math.pi / 180.0
    cos, sin = torch.cos(phi), torch.sin(phi)
    # физические координаты: X = x*W/2, Y = y*H/2 -> поворот корректен при анизотропии
    # p_in = D^-1 R(-phi) D (p_out - t) / s
    a11 = cos / s
    a12 = (sin * (H / W)) / s
    a21 = (-sin * (W / H)) / s
    a22 = cos / s
    tnx, tny = 2 * tx, 2 * ty  # доля ширины -> нормированные координаты [-1,1]
    b1 = -(a11 * tnx + a12 * tny)
    b2 = -(a21 * tnx + a22 * tny)
    theta = torch.stack([torch.stack([a11, a12, b1], 1), torch.stack([a21, a22, b2], 1)], 1)
    grid = F.affine_grid(theta, (B, 1, H, W), align_corners=False)
    y = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    tgt = torch.stack([rot / ROT_MAX, tx / SHIFT_MAX, ty / SHIFT_MAX, (s - 1) / (SCALE_MAX - 1)], 1)
    return y, tgt


_YY, _XX = None, None


def _grids(dev):
    global _YY, _XX
    if _YY is None or _YY.device != dev:
        _YY, _XX = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                                  torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    return _YY, _XX


def synth_metal_masks(B: int, gen: torch.Generator, dev) -> torch.Tensor:
    """Батч масок (B,H,W) float [0,1]: 1-3 «металлических» объекта на образец, всё векторно на GPU."""
    yy, xx = _grids(dev)
    K = 3
    def U(*shape, lo=0.0, hi=1.0):
        return torch.rand(*shape, generator=gen, device=dev) * (hi - lo) + lo
    n_obj = torch.randint(1, K + 1, (B,), generator=gen, device=dev)
    active = torch.arange(K, device=dev)[None, :] < n_obj[:, None]            # (B,K)
    kind = torch.multinomial(torch.tensor([0.35, 0.35, 0.15, 0.15], device=dev),
                             B * K, replacement=True, generator=gen).view(B, K)
    cx = U(B, K, lo=0.15 * W, hi=0.85 * W).view(B, K, 1, 1)
    cy = U(B, K, lo=0.1 * H, hi=0.9 * H).view(B, K, 1, 1)
    ang = U(B, K, lo=0.0, hi=math.pi).view(B, K, 1, 1)
    dx, dy = xx[None, None] - cx, yy[None, None] - cy                           # (B,K,H,W)
    u = dx * torch.cos(ang) + dy * torch.sin(ang)
    v = -dx * torch.sin(ang) + dy * torch.cos(ang)
    # параметры для всех типов
    a = U(B, K, lo=4, hi=22).view(B, K, 1, 1); b = U(B, K, lo=3, hi=14).view(B, K, 1, 1)
    ell = ((u / a) ** 2 + (v / b) ** 2) <= 1
    L = U(B, K, lo=25, hi=130).view(B, K, 1, 1); t = U(B, K, lo=1.5, hi=6).view(B, K, 1, 1)
    rod = (u.abs() <= L / 2) & (v.abs() <= t / 2)
    r = U(B, K, lo=6, hi=16).view(B, K, 1, 1); ri = r * U(B, K, lo=0.5, hi=0.8).view(B, K, 1, 1)
    d = torch.sqrt(u ** 2 + v ** 2)
    ring = (d <= r) & (d >= ri)
    Ls = U(B, K, lo=6, hi=20).view(B, K, 1, 1); ts = U(B, K, lo=2, hi=5).view(B, K, 1, 1)
    small = (u.abs() <= Ls / 2) & (v.abs() <= ts / 2)
    k = kind.view(B, K, 1, 1)
    sh = torch.where(k == 0, ell, torch.where(k == 1, rod, torch.where(k == 2, ring, small)))
    sh = sh & active.view(B, K, 1, 1)
    m = sh.any(dim=1).float()                                                   # (B,H,W)
    m = F.avg_pool2d(m[:, None], 3, stride=1, padding=1)[:, 0]
    return m


def augment_batch(x_u8: torch.Tensor, gen: torch.Generator, rng: np.random.Generator):
    """x_u8: (B,H,W) uint8 на GPU -> (B,3,H,W) нормированный, таргеты geom (B,4), art (B,)"""
    x = x_u8.float().unsqueeze(1) / 255.0
    B = x.shape[0]
    dev = x.device
    # инверсия (в открытых наборах кость бывает тёмной)
    inv = torch.rand(B, generator=gen, device=dev) < 0.2
    x = torch.where(inv.view(B, 1, 1, 1), 1 - x, x)
    # яркость/контраст/гамма
    gamma = torch.exp((torch.rand(B, generator=gen, device=dev) * 2 - 1) * 0.35).view(B, 1, 1, 1)
    x = x.clamp(1e-4, 1) ** gamma
    c = 1 + (torch.rand(B, generator=gen, device=dev) * 2 - 1) * 0.25
    bmean = x.mean(dim=(2, 3), keepdim=True)
    x = ((x - bmean) * c.view(B, 1, 1, 1) + bmean).clamp(0, 1)
    # геометрия
    x, geom = random_affine(x, gen)
    # артефакт
    art = (torch.rand(B, generator=gen, device=dev) < 0.5)
    m = synth_metal_masks(B, gen, dev) * art.view(B, 1, 1).float()             # (B,H,W)
    val = torch.where(inv, torch.rand(B, generator=gen, device=dev) * 0.12,
                      0.9 + torch.rand(B, generator=gen, device=dev) * 0.1).view(B, 1, 1)
    x[:, 0] = x[:, 0] * (1 - m) + m * val
    # шум + случайное размытие
    noise = torch.randn(x.shape, generator=gen, device=dev) * (torch.rand(B, 1, 1, 1, generator=gen, device=dev) * 0.04)
    x = (x + noise).clamp(0, 1)
    blur = torch.rand(B, generator=gen, device=dev) < 0.3
    if blur.any():
        xb = F.avg_pool2d(x[blur], 3, stride=1, padding=1)
        x[blur] = xb
    x3 = x.repeat(1, 3, 1, 1)
    x3 = (x3 - MEAN.to(dev)) / STD.to(dev)
    return x3, geom, art.float()


# --------------------------------------------------------------------------- #
class ProxyNet(nn.Module):
    def __init__(self, arch: str = "efficientnet_b0"):
        super().__init__()
        if arch == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            nf = self.backbone.classifier[1].in_features
        elif arch == "efficientnet_b2":
            self.backbone = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
            nf = self.backbone.classifier[1].in_features
        else:
            raise ValueError(arch)
        self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(nf, 256), nn.GELU(), nn.Linear(256, 5))

    def forward(self, x):
        return self.head(self.backbone(x))


def backbone_state(model: ProxyNet):
    return {k: v.detach().cpu() for k, v in model.backbone.state_dict().items() if not k.startswith("classifier")}


def sync_to_remote(ckpt_dir: Path, log_path):
    """Фоновый rsync чекпойнтов на сервер (pay-as-you-go: под может умереть)."""
    dest = os.environ.get("CKPT_REMOTE")  # например root@5.180.173.14:/opt/neuropeople/gpu_ckpt/
    pw = os.environ.get("CKPT_REMOTE_PASS")
    if not dest or not pw:
        return
    cmd = ["sshpass", "-p", pw, "rsync", "-a", "-e", "ssh -o StrictHostKeyChecking=no", str(ckpt_dir) + "/", dest]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(str(ckpt_dir / "rsync.err"), "a"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/workspace/cache")
    ap.add_argument("--out", default="/workspace/ckpt/proxy_b0")
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--exclude-own", action="store_true", help="не использовать собственный датасет")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logp = out / "train.log"
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    X = np.load(Path(a.cache) / "pretrain_u8.npy")
    meta = pd.read_csv(Path(a.cache) / "pretrain_meta.csv")
    if a.exclude_own:
        keep = (meta["source"] != "own_dxa").values
        X, meta = X[keep], meta[keep].reset_index(drop=True)
    N = len(X)
    rng = np.random.default_rng(42)
    perm = rng.permutation(N)
    n_val = max(256, int(0.05 * N))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xg = torch.from_numpy(X).to(dev)  # (N,H,W) uint8 целиком на GPU
    log(f"кэш {X.shape}, train {len(tr_idx)}, val {n_val}; источники: {meta['source'].value_counts().to_dict()}", logp)

    model = ProxyNet(a.arch).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(tr_idx) / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch,
                                                pct_start=0.08, div_factor=20, final_div_factor=100)
    start_ep = 0
    if a.resume and (out / "last.pth").exists():
        ck = torch.load(out / "last.pth", map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_ep = ck["epoch"] + 1
        log(f"resume с эпохи {start_ep}", logp)

    gen = torch.Generator(device=dev); gen.manual_seed(1234 + start_ep)
    hist = []
    for ep in range(start_ep, a.epochs):
        model.train()
        t0 = time.time()
        order = tr_idx[rng.permutation(len(tr_idx))]
        tot, n = 0.0, 0
        for i in range(0, len(order), a.bs):
            idx = torch.from_numpy(order[i:i + a.bs]).to(dev)
            with torch.no_grad():
                xb, geom, art = augment_batch(Xg[idx], gen, rng)
            xb = xb.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(xb)
            pred = pred.float()
            loss_g = F.smooth_l1_loss(pred[:, :4], geom, beta=0.1)
            loss_a = F.binary_cross_entropy_with_logits(pred[:, 4], art)
            loss = loss_g + 0.5 * loss_a
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step(); sched.step()
            tot += loss.item() * len(idx); n += len(idx)

        # валидация прокси-задач
        model.eval()
        vg, vp, va, vpa = [], [], [], []
        with torch.no_grad():
            gval = torch.Generator(device=dev); gval.manual_seed(7)
            rval = np.random.default_rng(7)
            for i in range(0, n_val, a.bs):
                idx = torch.from_numpy(val_idx[i:i + a.bs]).to(dev)
                xb, geom, art = augment_batch(Xg[idx], gval, rval)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(xb.contiguous(memory_format=torch.channels_last)).float()
                vg.append(geom.cpu()); vp.append(pred[:, :4].cpu()); va.append(art.cpu()); vpa.append(pred[:, 4].cpu())
        vg, vp, va, vpa = torch.cat(vg), torch.cat(vp), torch.cat(va), torch.cat(vpa)
        rot_mae = (vp[:, 0] - vg[:, 0]).abs().mean().item() * ROT_MAX
        sh_mae = ((vp[:, 1] - vg[:, 1]).abs().mean().item() * SHIFT_MAX * W + (vp[:, 2] - vg[:, 2]).abs().mean().item() * SHIFT_MAX * H) / 2
        sc_mae = (vp[:, 3] - vg[:, 3]).abs().mean().item() * (SCALE_MAX - 1)
        from sklearn.metrics import roc_auc_score
        art_auc = roc_auc_score(va.numpy(), vpa.numpy())
        rec = dict(epoch=ep, train_loss=tot / n, rot_mae_deg=rot_mae, shift_mae_px=sh_mae, scale_mae=sc_mae,
                   art_auc=art_auc, lr=sched.get_last_lr()[0], sec=time.time() - t0)
        hist.append(rec)
        log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), logp)

        torch.save(backbone_state(model), out / f"backbone_ep{ep:02d}.pth")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep},
                   out / "last.pth")
        with open(out / "history.json", "w") as f:
            json.dump(hist, f, indent=1)
        sync_to_remote(out, logp)

    log("DONE", logp)


if __name__ == "__main__":
    main()
