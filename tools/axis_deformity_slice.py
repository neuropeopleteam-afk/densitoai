#!/usr/bin/env python3
"""Контрольный срез «деформация оси»: нормы, у которых разметчик отметил деформацию оси.

Что делает (модели, пороги и признаки не меняет; читает только файлы поставки):
  1. Из `разметка.xlsx` (лист «Калибровка», колонка «Комментарий») выбирает исследования, где комментарий
     разметчика говорит о деформации оси. Само слово комментария в коде не пишем (словарь проекта):
     сравниваем sha1 первого слова комментария в нижнем регистре с ключом AXIS_DEFORMITY_KEY.
  2. Сопоставляет исследования с кадрами позвоночника OOF поставки (`models/oof_stacked_spine_*.csv`,
     флаг = `pred_label`, то есть решение сервиса) и пишет срез в `data/slices/axis_deformity.csv`.
  3. Считает долю ложных срабатываний `sp_axis` и бинарного класса позвоночника (ИЛИ трёх флагов)
     на срезе против прочих норм; 95 % ДИ — кластерный бутстрэп по исследованиям (2000 повторов,
     отдельно внутри среза и внутри прочих норм); плюс перестановочный контроль «случайные 14
     исследований норм вместо среза».
  4. С ключом --sag — отчётная величина «стрела прогиба» центральной линии столба в мм (не признак
     модели, в CSV сервиса не выводится): максимум отклонения сглаженной линии центроидов строк столба
     от хорды между её концами. Столб выделяется так же, как в `tools/k5/axis_features.py`.

Запуск (из корня рабочей копии):
  DENSITO_ROOT=$PWD ../venv/bin/python tools/axis_deformity_slice.py [--sag]
Выход: data/slices/axis_deformity.csv, docs/axis_deformity_slice.json (числа для docs/AXIS_DEFORMITY_SLICE.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
DATASET = Path(os.environ.get("DENSITO_DATASET", ROOT.parent / "dataset"))
XLSX = DATASET / "разметка.xlsx"
MODELS = ROOT / "models"
N_BOOT = 2000
N_PERM = 5000
SEED = 2026
# sha1 (первые 16 знаков) первого слова комментария разметчика, обозначающего деформацию оси.
AXIS_DEFORMITY_KEY = "f4bca53872dc6d88"
CRITS = ("sp_pos", "sp_axis", "sp_art")
SX, SY = 0.6, 1.05  # мм/пиксель, как в src/hip_features.py и tools/k5/axis_features.py


def rel_path(p: str) -> str:
    """Путь относительно папки «Исследования» (в OOF записаны пути стенда обучения)."""
    return p.split("Исследования/", 1)[1] if "Исследования/" in p else p


def comment_key(c) -> str:
    w = re.split(r"[^\wё]+", str(c).strip().lower())[0]
    return hashlib.sha1(w.encode("utf-8")).hexdigest()[:16]


def slice_studies(xlsx: Path) -> pd.DataFrame:
    d = pd.read_excel(xlsx, sheet_name="Калибровка", header=None)
    hdr = [str(h) for h in d.iloc[1].tolist()]         # вторая строка — подписи колонок
    col_comment = next(i for i, h in enumerate(hdr) if h.strip() == "Комментарий")
    col_study = 1
    body = d.iloc[2:]
    rows = []
    for _, r in body.iterrows():
        c = r.iloc[col_comment]
        if pd.isna(c) or pd.isna(r.iloc[col_study]):
            continue
        if comment_key(c) == AXIS_DEFORMITY_KEY:
            rows.append({"study": str(r.iloc[col_study]).strip(),
                         "xlsx_sp_pos": int(r.iloc[2]), "xlsx_sp_axis": int(r.iloc[3]), "xlsx_sp_art": int(r.iloc[4])})
    return pd.DataFrame(rows)


def load_spine_oof() -> pd.DataFrame:
    base = None
    for crit in CRITS:
        df = pd.read_csv(MODELS / f"oof_stacked_spine_{crit}.csv")
        df = df.rename(columns={"y_true": f"y_{crit}", "oof_stacked": f"s_{crit}", "pred_label": f"f_{crit}"})
        df = df[["study", "file_path", f"y_{crit}", f"s_{crit}", f"f_{crit}"]]
        base = df if base is None else base.merge(df, on=["study", "file_path"], how="inner")
    base["rel_path"] = base["file_path"].map(rel_path)
    base["y_any"] = base[[f"y_{c}" for c in CRITS]].max(axis=1).astype(int)
    base["f_any"] = base[[f"f_{c}" for c in CRITS]].max(axis=1).astype(int)
    return base


def fpr(flags: np.ndarray) -> float:
    return float(flags.mean()) if len(flags) else float("nan")


def cluster_boot_diff(flag_a, grp_a, flag_b, grp_b, rng, n=N_BOOT):
    """Кластерный бутстрэп по исследованиям отдельно в двух наборах: ДИ для FPR_a, FPR_b и разности."""
    def idx_by(grp):
        u = np.unique(grp)
        return u, {g: np.nonzero(grp == g)[0] for g in u}
    ua, ia = idx_by(grp_a)
    ub, ib = idx_by(grp_b)
    va, vb, vd = [], [], []
    for _ in range(n):
        sa = np.concatenate([ia[g] for g in rng.choice(ua, len(ua))])
        sb = np.concatenate([ib[g] for g in rng.choice(ub, len(ub))])
        a, b = flag_a[sa].mean(), flag_b[sb].mean()
        va.append(a); vb.append(b); vd.append(a - b)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"ci_slice": q(va), "ci_other": q(vb), "ci_diff": q(vd), "p_diff_le_0": float(np.mean(np.array(vd) <= 0))}


def perm_null(df_norm: pd.DataFrame, flag_col: str, n_studies: int, observed_fp: int, rng) -> dict:
    """Случайные n_studies исследований норм вместо среза: сколько ложных срабатываний набирается случайно."""
    fp_by_study = df_norm.groupby("study")[flag_col].sum()
    studies = fp_by_study.index.to_numpy()
    vals = np.array([fp_by_study.loc[rng.choice(studies, n_studies, replace=False)].sum() for _ in range(N_PERM)])
    return {"null_mean_fp": float(vals.mean()), "null_p95_fp": float(np.percentile(vals, 95)),
            "p_value_ge_observed": float(np.mean(vals >= observed_fp))}


def compare(df: pd.DataFrame, norm_mask: np.ndarray, flag_col: str, score_col: str | None, rng) -> dict:
    d = df[norm_mask].copy()
    a, b = d[d.in_slice], d[~d.in_slice]
    out = {"n_slice_frames": int(len(a)), "n_slice_studies": int(a.study.nunique()),
           "n_other_frames": int(len(b)), "n_other_studies": int(b.study.nunique()),
           "fp_slice": int(a[flag_col].sum()), "fp_other": int(b[flag_col].sum()),
           "fp_total": int(d[flag_col].sum()),
           "fpr_slice": fpr(a[flag_col].to_numpy()), "fpr_other": fpr(b[flag_col].to_numpy()),
           "fp_studies_slice": int(a.groupby("study")[flag_col].max().sum()),
           "fp_studies_total": int(d.groupby("study")[flag_col].max().sum())}
    out.update(cluster_boot_diff(a[flag_col].to_numpy(), a.study.to_numpy(), b[flag_col].to_numpy(),
                                 b.study.to_numpy(), rng))
    # тот же расчёт на уникальных кадрах (один кадр на хэш пикселей): клоны не множат вес исследования
    if "pixel_hash" in d.columns:
        du = d.drop_duplicates("pixel_hash")
        au, bu = du[du.in_slice], du[~du.in_slice]
        out["unique_frames"] = {"n_slice": int(len(au)), "n_other": int(len(bu)),
                                "fp_slice": int(au[flag_col].sum()), "fp_other": int(bu[flag_col].sum()),
                                "fpr_slice": fpr(au[flag_col].to_numpy()), "fpr_other": fpr(bu[flag_col].to_numpy())}
    out["perm_random_studies"] = perm_null(d, flag_col, out["n_slice_studies"], out["fp_slice"], rng)
    if score_col:
        out["score_mean_slice"] = float(a[score_col].mean())
        out["score_mean_other"] = float(b[score_col].mean())
        out["auc_score_slice_vs_other_norms"] = float(roc_auc_score(d.in_slice.astype(int), d[score_col]))
    return out


# ---------- отчётная величина: стрела прогиба центральной линии столба, мм ----------

def sag_mm(path: str) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from geometry_features import read_dicom_normalized, segment_bone  # noqa: E402
    img, _ = read_dicom_normalized(path)
    mask = segment_bone(img)
    h = mask.shape[0]
    rows = [(y, xs.mean(), xs.max() - xs.min()) for y in range(h) for xs in [np.nonzero(mask[y])[0]] if len(xs) > 3]
    if len(rows) < 20:
        return {"sag_mm": np.nan, "col_len_mm": np.nan}
    R = np.array(rows, float)
    ys, cx, wd = R[:, 0], R[:, 1], R[:, 2]
    wmed = np.median(wd)
    idx = np.nonzero((wd >= 0.5 * wmed) & (wd <= 1.6 * wmed))[0]
    if len(idx) < 20:
        return {"sag_mm": np.nan, "col_len_mm": np.nan}
    seg = max(np.split(idx, np.nonzero(np.diff(idx) > 3)[0] + 1), key=len)
    y_mm, x_mm = ys[seg] * SY, cx[seg] * SX
    k = max(3, int(round(10.0 / SY)) | 1)  # сглаживание ~10 мм по вертикали
    xs = np.convolve(np.pad(x_mm, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    p0, p1 = np.array([xs[0], y_mm[0]]), np.array([xs[-1], y_mm[-1]])
    v = p1 - p0
    L = float(np.hypot(*v))
    if L < 1e-6:
        return {"sag_mm": np.nan, "col_len_mm": np.nan}
    dist = np.abs(v[0] * (y_mm - p0[1]) - v[1] * (xs - p0[0])) / L
    return {"sag_mm": float(dist.max()), "col_len_mm": L}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=str(XLSX))
    ap.add_argument("--sag", action="store_true", help="посчитать отчётную стрелу прогиба в мм")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)

    st = slice_studies(Path(a.xlsx))
    oof = load_spine_oof()
    ph_file = ROOT / "docs" / "k5" / "pixel_hashes.csv"
    if ph_file.exists():
        ph = pd.read_csv(ph_file)
        ph["rel_path"] = ph["file_path"].map(rel_path)
        oof = oof.merge(ph[["rel_path", "pixel_hash"]], on="rel_path", how="left")
    oof["in_slice"] = oof["study"].isin(set(st["study"]))
    sl = oof[oof.in_slice].copy()

    res = {"source": {"xlsx": "dataset/разметка.xlsx, лист «Калибровка», колонка «Комментарий»",
                      "oof": "models/oof_stacked_spine_{sp_pos,sp_axis,sp_art}.csv (pred_label = решение сервиса)",
                      "bootstrap": f"кластерный по исследованиям, {N_BOOT} повторов, 95 %, отдельно в срезе и в прочих нормах",
                      "perm": f"{N_PERM} случайных наборов из того же числа исследований норм"},
           "n_studies_xlsx": int(len(st)), "n_studies_with_spine_oof": int(sl.study.nunique()),
           "n_frames": int(len(sl)),
           "labels_in_slice": {c: int(sl[f"y_{c}"].sum()) for c in CRITS},
           "xlsx_labels_in_slice": {c: int(st[f"xlsx_{c}"].sum()) for c in CRITS}}

    # (1) sp_axis: нормы по оси и укладке (как определён срез)
    norm_ax = ((oof.y_sp_axis == 0) & (oof.y_sp_pos == 0)).to_numpy()
    res["sp_axis_norm_axis_pos"] = compare(oof, norm_ax, "f_sp_axis", "s_sp_axis", rng)
    # (1b) sp_axis: против всех норм по оси
    res["sp_axis_all_axis_norms"] = compare(oof, (oof.y_sp_axis == 0).to_numpy(), "f_sp_axis", "s_sp_axis", rng)
    res["sp_axis_fp_total_all_norms"] = int(oof.loc[oof.y_sp_axis == 0, "f_sp_axis"].sum())
    # (2) бинарный класс позвоночника: нормы по всем трём критериям
    res["binary_spine_norm_all"] = compare(oof, (oof.y_any == 0).to_numpy(), "f_any", None, rng)
    # (3) sp_pos и sp_art на срезе (для полноты)
    res["sp_pos_norm"] = compare(oof, (oof.y_sp_pos == 0).to_numpy(), "f_sp_pos", "s_sp_pos", rng)
    res["sp_art_norm"] = compare(oof, (oof.y_sp_art == 0).to_numpy(), "f_sp_art", "s_sp_art", rng)
    # что делает бинарный флаг на срезе: какие критерии срабатывают
    res["slice_flags_by_criterion"] = {c: int(sl[f"f_{c}"].sum()) for c in CRITS}
    res["slice_binary_flags"] = int(sl.f_any.sum())

    # геометрия (отчётно): угол оси, которым решает контур A sp_axis
    g = pd.read_csv(ROOT / "data" / "geometry_features.csv", usecols=["file_path", "region", "axis_angle_deg"])
    g = g[g.region == "spine"]
    g["rel_path"] = g["file_path"].map(rel_path)
    oof = oof.merge(g[["rel_path", "axis_angle_deg"]], on="rel_path", how="left")
    oof["abs_angle"] = oof["axis_angle_deg"].abs()
    grp = {"slice": oof[oof.in_slice & (oof.y_sp_axis == 0)], "other_norms": oof[~oof.in_slice & (oof.y_sp_axis == 0)],
           "axis_positives": oof[oof.y_sp_axis == 1]}
    res["abs_axis_angle_deg_median"] = {k: float(v.abs_angle.median()) for k, v in grp.items()}

    if a.sag:
        dsroot = DATASET / "Исследования"
        vals = [sag_mm(str(dsroot / p)) for p in oof["rel_path"]]
        oof["sag_mm"] = [v["sag_mm"] for v in vals]
        grp = {"slice": oof[oof.in_slice & (oof.y_sp_axis == 0)],
               "other_norms": oof[~oof.in_slice & (oof.y_sp_axis == 0)], "axis_positives": oof[oof.y_sp_axis == 1]}
        res["sag_mm"] = {k: {"median": float(v.sag_mm.median()), "q25": float(v.sag_mm.quantile(.25)),
                             "q75": float(v.sag_mm.quantile(.75)), "n": int(v.sag_mm.notna().sum())}
                         for k, v in grp.items()}
        yy = pd.concat([grp["slice"].assign(t=1), grp["other_norms"].assign(t=0)]).dropna(subset=["sag_mm"])
        res["sag_auc_slice_vs_other_norms"] = float(roc_auc_score(yy.t, yy.sag_mm))
        yy = pd.concat([grp["slice"].assign(t=1), grp["axis_positives"].assign(t=0)]).dropna(subset=["sag_mm"])
        res["sag_auc_slice_vs_axis_positives"] = float(roc_auc_score(yy.t, yy.sag_mm))
        res["sag_note"] = "отчётная величина, не признак модели; в CSV сервиса и в карточке не выводится"

    # срез в data/slices
    out_dir = ROOT / "data" / "slices"
    out_dir.mkdir(parents=True, exist_ok=True)
    sl = oof[oof.in_slice].copy()
    sl["slice"] = "деформация оси (комментарий разметчика)"
    cols = ["study", "rel_path", "slice", "y_sp_pos", "y_sp_axis", "y_sp_art", "f_sp_pos", "f_sp_axis", "f_sp_art",
            "s_sp_axis", "abs_angle"] + (["sag_mm"] if a.sag else []) + (["pixel_hash"] if "pixel_hash" in sl else [])
    sl = sl[cols].rename(columns={"rel_path": "file", "s_sp_axis": "oof_sp_axis", "abs_angle": "abs_axis_angle_deg"})
    sl.sort_values(["study", "file"]).to_csv(out_dir / "axis_deformity.csv", index=False, float_format="%.6g")
    (ROOT / "docs" / "axis_deformity_slice.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
