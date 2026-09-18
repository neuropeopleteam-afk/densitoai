"""
Честная оценка backbone для контура B: эмбеддинги -> StandardScaler -> PCA(32) ->
LogisticRegression(C=0.1, balanced), OOF по GroupKFold(5). Группы = study,
объединённые по одинаковым изображениям (в датасете 499 файлов / 252 уникальных
снимка — дубликаты не должны попадать в разные фолды).

Использование:
  python eval_embeddings.py --labels labels.csv --weights imagenet
  python eval_embeddings.py --labels labels.csv --weights ckpt/backbone_ep30.pth ckpt/backbone_ep59.pth
Печатает таблицу AUC по критериям для каждого набора весов и пишет json.
"""
import argparse
import os
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from geometry_features import read_dicom_normalized  # noqa: E402

CRITERIA = {
    "spine": ["sp_pos", "sp_axis", "sp_art"],
    "hip": ["hip_pos", "hip_roi"],
}


def load_backbone(weights: str, arch: str, dev):
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if weights == "imagenet" else None)
    else:
        m = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1 if weights == "imagenet" else None)
    m.classifier = nn.Identity()
    if weights != "imagenet":
        sd = torch.load(weights, map_location="cpu")
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        assert all(k.startswith("classifier") for k in missing), missing
    return m.eval().to(dev)


def build_groups(df: pd.DataFrame, imgs):
    """study + одинаковые изображения -> одна группа (union-find)."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (_, r), im in zip(df.iterrows(), imgs):
        h = "img:" + hashlib.md5(im.tobytes()).hexdigest()
        union("st:" + str(r["study"]), h)
    return np.array([find("st:" + str(s)) for s in df["study"]])


@torch.no_grad()
def embed(model, imgs, mirror, dev, bs=64):
    tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize((320, 192)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    out = []
    for i in range(0, len(imgs), bs):
        batch = []
        for im, mr in zip(imgs[i:i + bs], mirror[i:i + bs]):
            x = im[:, ::-1].copy() if mr else im
            batch.append(tf(np.stack([x, x, x], 2)))
        out.append(model(torch.stack(batch).to(dev)).float().cpu())
    return torch.cat(out).numpy()


def oof_auc(E, y, groups, n_pca=32, seeds=(0, 1, 2)):
    """Повторённый GroupKFold(5): усреднённый AUC и AP."""
    aucs, aps = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        ug = np.unique(groups)
        gmap = dict(zip(ug, rng.permutation(len(ug))))
        g = np.array([gmap[x] for x in groups])
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(E, y, g):
            if y[tr].sum() == 0:
                oof[te] = 0
                continue
            sc = StandardScaler().fit(E[tr])
            pca = PCA(n_components=min(n_pca, len(tr) - 1), random_state=42).fit(sc.transform(E[tr]))
            clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")
            clf.fit(pca.transform(sc.transform(E[tr])), y[tr])
            oof[te] = clf.predict_proba(pca.transform(sc.transform(E[te])))[:, 1]
        aucs.append(roc_auc_score(y, oof)); aps.append(average_precision_score(y, oof))
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(aps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--weights", nargs="+", default=["imagenet"])
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--out", default="eval_embeddings.json")
    ap.add_argument("--concat", action="store_true",
                    help="дополнительно оценить конкатенацию эмбеддингов первого набора весов (imagenet) с каждым следующим")
    ap.add_argument("--save_emb", default=None, help="папка для сохранения эмбеддингов (npy) по каждому набору весов")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(a.labels)
    imgs = [read_dicom_normalized(p)[0] for p in df["file_path"]]
    groups = build_groups(df, imgs)
    print("файлов", len(df), "групп", len(np.unique(groups)))
    mirror = (df["region"] == "right_hip").values
    sp = (df["region"] == "spine").values
    hip = df["region"].isin(["right_hip", "left_hip"]).values
    df["hip_pos"] = df["rh_pos"].fillna(df["lh_pos"])
    df["hip_roi"] = df["rh_roi"].fillna(df["lh_roi"])

    def evaluate(E):
        res = {}
        for region, crits in CRITERIA.items():
            mask = sp if region == "spine" else hip
            for c in crits:
                m = mask & df[c].notna().values
                y = df.loc[m, c].values.astype(int)
                auc, sd, apv = oof_auc(E[m], y, groups[m])
                res[c] = dict(auc=round(auc, 4), auc_sd=round(sd, 4), ap=round(apv, 4), n=int(m.sum()), pos=int(y.sum()))
        res["mean_auc"] = round(float(np.mean([v["auc"] for k, v in res.items() if k != "mean_auc"])), 4)
        return res

    results, embs = {}, {}
    for w in a.weights:
        model = load_backbone(w, a.arch, dev)
        E = embed(model, imgs, mirror, dev)
        embs[w] = E
        if a.save_emb:
            os.makedirs(a.save_emb, exist_ok=True)
            np.save(os.path.join(a.save_emb, os.path.basename(w).replace(".pth", "") + ".npy"), E)
        results[w] = evaluate(E)
        print(f"\n== {w}\n" + pd.DataFrame(results[w]).T.to_string())
        if a.concat and w != a.weights[0]:
            key = f"{a.weights[0]}+{w}"
            results[key] = evaluate(np.concatenate([embs[a.weights[0]], E], axis=1))
            print(f"\n== {key}\n" + pd.DataFrame(results[key]).T.to_string())
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
