#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""organizer_metrics.py — метрики в схеме организаторов: наш CSV из 9 колонок + разметка по исследованиям.

Вход:
  --results results.csv          выгрузка сервиса (9 колонок ТЗ п. 2.5, порядок и имена не меняются);
  --markup  разметка.xlsx        разметка в формате организаторов: лист «Калибровка», строка = исследование,
                                 колонки «Позвоночник» (3 критерия), «правое бедро» и «левое бедро» (по 2 критерия);
  --side detector|label          как сопоставить снимок бедра со стороной в разметке:
                                 detector — анатомический детектор стороны (src/hip_features.detect_hip_side),
                                            как в сервисе и при обучении; без --dicom-root сторона берётся из
                                            data/labels_full.csv (перегенерирован тем же детектором);
                                 label    — готовая таблица «файл → сторона» (--side-map; по умолчанию
                                            data/labels_full_v1_density_side.csv, прежняя плотностная эвристика).
  --dicom-root DIR               папка исследований: если задана, сторона (detector) и хэш пикселей считаются
                                 по самим DICOM; иначе берутся из data/labels_full.csv и docs/k5/pixel_hashes.csv.
  --oof                          режим OOF: вместо --results читаются models/oof_stacked_*.csv; quality_prob
                                 собирается как в src/eval_oof_metrics.py (any-модель + max критериев, согласование
                                 с классом). Числа по областям обязаны совпасть с models/metrics_oof_full.json
                                 (docs/METRICS_REPORT.md) до третьего знака — проверяется, код возврата 1 при
                                 расхождении.

Выход (--out-md, --out-json):
  * бинарная задача «есть нарушение»: F1 по quality_class и ROC-AUC по quality_prob — по областям и общим пулом всех
    файлов (так, как посчитает жюри, если не делит по областям);
  * F1 по каждому из 5 типов;
  * macro-F1 в трёх трактовках: (1) по областям — среднее F1 типов внутри области; (2) по 5 критериям;
    (3) по 4 уникальным строкам словаря, где «Некорректная укладка» позвоночника и бедра — одна строка
    (флаги и метки двух областей складываются в один класс);
  * всё — по файлам и по уникальным кадрам (хэш пикселей: побайтовые клоны экспорта считаются один раз);
  * 95 % ДИ — кластерный бутстрап по исследованиям (ресэмплы без положительных исключаются для F1 и AUC);
  * уровень визита (исследования): визит «с нарушением», если нарушение есть хотя бы в одном размеченном снимке;
    «помечен», если хотя бы один снимок получил класс 1 — чувствительность, специфичность, точность, F1;
  * строки с несколькими типами: предсказано против истины; в режиме OOF — блоки связанных рангов на пороге.

Строки со статусом, отличным от Success, остаются в расчёте с тем классом и вероятностью, которые записаны в CSV
(так их увидит оценщик); их число печатается отдельно. Снимки без разметки (NaN в листе) не входят в метрики.

Важно: на 499 файлах заказчика выгрузка сервиса — in-sample (модели обучены на этих снимках), такие числа
завышены и годятся только для проверки конвейера. Честные числа — режим --oof.

Примеры:
  python tools/organizer_metrics.py --oof --markup ../dataset/разметка.xlsx
  python tools/organizer_metrics.py --results outputs/results.csv --markup ../dataset/разметка.xlsx \\
      --dicom-root ../dataset/Исследования --side detector
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))

REGION_RU = {"spine": "Поясничный отдел позвоночника", "hip": "Проксимальный отдел бедра"}
RU_REGION = {v: k for k, v in REGION_RU.items()}
# (критерий, область, строка словаря организаторов)
TYPES = [("sp_pos", "spine", "Некорректная укладка"),
         ("sp_axis", "spine", "Не выравнена ось позвоночника"),
         ("sp_art", "spine", "Присутствуют посторонние предметы"),
         ("hip_pos", "hip", "Некорректная укладка"),
         ("hip_roi", "hip", "Некорректная область интереса")]
CRITS = [t[0] for t in TYPES]
REGION_CRITS = {"spine": ["sp_pos", "sp_axis", "sp_art"], "hip": ["hip_pos", "hip_roi"]}
# 4 уникальные строки словаря: «Некорректная укладка» — общая для двух областей
DICT_ROWS = {"Некорректная укладка": ["sp_pos", "hip_pos"],
             "Не выравнена ось позвоночника": ["sp_axis"],
             "Присутствуют посторонние предметы": ["sp_art"],
             "Некорректная область интереса": ["hip_roi"]}
MARKUP_COLS = ["sp_pos", "sp_axis", "sp_art", "rh_pos", "rh_roi", "lh_pos", "lh_roi"]
THR_TIE_ATOL = 1e-9
DEFAULT_SIDE_MAP = ROOT / "data" / "labels_full_v1_density_side.csv"


# ----------------------------------------------------------------------------- ввод
def rel_key(p: str) -> str:
    """Ключ файла: путь от папки исследования (как path_to_study в CSV)."""
    s = str(p).replace("\\", "/")
    if "/Исследования/" in s:
        s = s.split("/Исследования/", 1)[1]
    return s.lstrip("./")


def load_markup(path: Path, sheet: str = "Калибровка") -> pd.DataFrame:
    """Лист «Калибровка»: две строки шапки, затем № | study | 3 критерия позвоночника | 2 правого | 2 левого бедра."""
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    body = raw.iloc[2:, 1:9].copy()
    body.columns = ["study"] + MARKUP_COLS
    body = body[body["study"].notna()]
    body["study"] = body["study"].astype(str).str.strip()
    body = body[body["study"].str.match(r"^\d")]
    for c in MARKUP_COLS:
        body[c] = pd.to_numeric(body[c], errors="coerce")
    return body.set_index("study")


def raw_pixel_hash(path: Path) -> str:
    import pydicom
    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array
    return hashlib.sha1(arr.tobytes() + str(arr.shape).encode()).hexdigest()


def detector_side(path: Path) -> str:
    from geometry_features import read_dicom_normalized
    from hip_features import detect_hip_side
    img, _ = read_dicom_normalized(str(path))
    return detect_hip_side(img)


def side_table_from_csv(path: Path) -> dict:
    """file → 'right'/'left' из CSV с колонками file_path|path_to_study и region|side."""
    t = pd.read_csv(path, low_memory=False)
    kcol = "file_path" if "file_path" in t.columns else "path_to_study"
    scol = "side" if "side" in t.columns else "region"
    out = {}
    for k, v in zip(t[kcol], t[scol]):
        v = str(v)
        if v.startswith("right"):
            out[rel_key(k)] = "right"
        elif v.startswith("left"):
            out[rel_key(k)] = "left"
    return out


def hash_table_default() -> dict:
    p = ROOT / "docs" / "k5" / "pixel_hashes.csv"
    if not p.exists():
        return {}
    t = pd.read_csv(p)
    return {rel_key(k): h for k, h in zip(t["file_path"], t["pixel_hash"])}


def parse_flags(region: str, vt) -> dict:
    names = [s.strip() for s in str(vt).split(";")] if isinstance(vt, str) and vt.strip() else []
    return {c: int(any(n == name for n in names)) for c, reg, name in TYPES if reg == region}


def attach_truth(df: pd.DataFrame, markup: pd.DataFrame) -> pd.DataFrame:
    """y_<crit> по разметке исследования и стороне снимка; NaN — не размечено."""
    for c in CRITS:
        df[f"y_{c}"] = np.nan
    for i, r in df.iterrows():
        st = str(r["study"])
        if st not in markup.index:
            continue
        m = markup.loc[st]
        if r["region"] == "spine":
            for c in REGION_CRITS["spine"]:
                df.at[i, f"y_{c}"] = m[c]
        elif r["region"] == "hip" and r["side"] in ("right", "left"):
            p = "rh" if r["side"] == "right" else "lh"
            df.at[i, "y_hip_pos"] = m[f"{p}_pos"]
            df.at[i, "y_hip_roi"] = m[f"{p}_roi"]
    ys = df[[f"y_{c}" for c in CRITS]]
    df["labelled"] = ys.notna().any(axis=1)
    df["y_bin"] = (ys.fillna(0).max(axis=1) > 0).astype(int)
    return df


def table_from_results(results: Path, side_mode: str, side_map: Path | None, dicom_root: Path | None) -> pd.DataFrame:
    r = pd.read_csv(results, dtype={"violation_type": str}, keep_default_na=False)
    need = ["path_to_study", "anatomical_region", "quality_class", "violation_type", "quality_prob", "processing_status"]
    miss = [c for c in need if c not in r.columns]
    if miss:
        raise SystemExit(f"в {results} нет колонок {miss}")
    rows = []
    if side_mode == "label":
        sides = side_table_from_csv(side_map or DEFAULT_SIDE_MAP)
    elif dicom_root is None:
        sides = side_table_from_csv(ROOT / "data" / "labels_full.csv")
    else:
        sides = {}
    hashes = {} if dicom_root is not None else hash_table_default()
    for _, x in r.iterrows():
        key = rel_key(x["path_to_study"])
        region = RU_REGION.get(str(x["anatomical_region"]), "unknown")
        side = None
        if region == "hip":
            if side_mode == "detector" and dicom_root is not None:
                try:
                    side = detector_side(dicom_root / key)
                except Exception:  # noqa: BLE001
                    side = None
            else:
                side = sides.get(key)
        if dicom_root is not None:
            try:
                h = raw_pixel_hash(dicom_root / key)
            except Exception:  # noqa: BLE001
                h = "nohash:" + key
        else:
            h = hashes.get(key, "nohash:" + key)
        qp = pd.to_numeric(x["quality_prob"], errors="coerce")
        row = {"key": key, "study": key.split("/")[0], "region": region, "side": side, "pixel_hash": h,
               "quality_class": int(pd.to_numeric(x["quality_class"], errors="coerce") == 1),
               "quality_prob": float(qp) if pd.notna(qp) else 0.5,
               "status": str(x["processing_status"])}
        fl = parse_flags(region, x["violation_type"]) if region in REGION_CRITS else {}
        for c in CRITS:
            row[f"f_{c}"] = fl.get(c, 0)
        rows.append(row)
    return pd.DataFrame(rows)


def table_from_oof(side_mode: str, side_map: Path | None) -> tuple[pd.DataFrame, dict]:
    """Псевдо-выгрузка из OOF: флаги pred_label, quality_prob как в src/eval_oof_metrics.py."""
    from eval_oof_metrics import oof_any_model
    summary = json.loads((ROOT / "models" / "metrics_summary.json").read_text(encoding="utf-8"))
    parts, scores = [], {}
    hashes = hash_table_default()
    for region, crits in REGION_CRITS.items():
        base = None
        for c in crits:
            o = pd.read_csv(ROOT / "models" / f"oof_stacked_{region}_{c}.csv")
            o = o.rename(columns={"y_true": f"yo_{c}", "pred_label": f"f_{c}", "oof_stacked": f"s_{c}"})
            keep = ["study", "file_path", f"yo_{c}", f"f_{c}", f"s_{c}"] + (["hip_side_detected"] if region == "hip" and base is None else [])
            base = o[keep] if base is None else base.merge(o[["file_path", f"yo_{c}", f"f_{c}", f"s_{c}"]], on="file_path", how="inner", validate="one_to_one")
            scores[c] = float(summary[region][c]["threshold"])
        flags = base[[f"f_{c}" for c in crits]].max(axis=1).values.astype(int)
        crit_max = base[[f"s_{c}" for c in crits]].max(axis=1).values
        anym = oof_any_model(region, crits).set_index("file_path").loc[base["file_path"].values]
        raw = 0.5 * anym["any_model_oof"].values + 0.5 * crit_max
        prob = np.where(flags == 1, 0.5 + 0.5 * raw, np.minimum(0.5 * raw, 0.499999))
        t = pd.DataFrame({"key": base["file_path"].map(rel_key), "study": base["study"].astype(str), "region": region,
                          "quality_class": flags, "quality_prob": prob, "status": "Success"})
        t["side"] = base["hip_side_detected"].values if region == "hip" else None
        for c in CRITS:
            t[f"f_{c}"] = base[f"f_{c}"].values.astype(int) if c in crits else 0
            t[f"s_{c}"] = base[f"s_{c}"].values if c in crits else np.nan
            t[f"yo_{c}"] = base[f"yo_{c}"].values if c in crits else np.nan
        parts.append(t)
    df = pd.concat(parts, ignore_index=True)
    df["pixel_hash"] = [hashes.get(k, "nohash:" + k) for k in df["key"]]
    if side_mode == "label":
        sides = side_table_from_csv(side_map or DEFAULT_SIDE_MAP)
        df.loc[df.region == "hip", "side"] = df.loc[df.region == "hip", "key"].map(sides)
    return df, scores


# ----------------------------------------------------------------------------- метрики
def _f1(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def binary_block(y, p, s=None):
    y = np.asarray(y).astype(int); p = np.asarray(p).astype(int)
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "n_flag": int(p.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
           "specificity": tn / (tn + fp) if tn + fp else float("nan"),
           "precision": tp / (tp + fp) if tp + fp else float("nan"),
           "f1": _f1(y, p)}
    if s is not None:
        out["roc_auc"] = _auc(y, np.asarray(s, float))
    return out


def point_metrics(df: pd.DataFrame) -> dict:
    """Все заголовочные числа на данном наборе строк (df — только размеченные строки)."""
    m = {}
    for reg in ("spine", "hip"):
        d = df[df.region == reg]
        m[f"bin_f1/{reg}"] = _f1(d.y_bin.values, d.quality_class.values)
        m[f"bin_auc/{reg}"] = _auc(d.y_bin.values, d.quality_prob.values)
    m["bin_f1/pooled"] = _f1(df.y_bin.values, df.quality_class.values)
    m["bin_auc/pooled"] = _auc(df.y_bin.values, df.quality_prob.values)
    f1 = {}
    for c, reg, _ in TYPES:
        d = df[(df.region == reg) & df[f"y_{c}"].notna()]
        f1[c] = _f1(d[f"y_{c}"].values.astype(int), d[f"f_{c}"].values)
        m[f"f1/{c}"] = f1[c]
    m["macro/spine"] = float(np.mean([f1[c] for c in REGION_CRITS["spine"]]))
    m["macro/hip"] = float(np.mean([f1[c] for c in REGION_CRITS["hip"]]))
    m["macro/regions_mean"] = (m["macro/spine"] + m["macro/hip"]) / 2
    m["macro/5_criteria"] = float(np.mean(list(f1.values())))
    d4 = {}
    for name, cs in DICT_ROWS.items():
        ys, ps = [], []
        for c in cs:
            reg = next(t[1] for t in TYPES if t[0] == c)
            d = df[(df.region == reg) & df[f"y_{c}"].notna()]
            ys.append(d[f"y_{c}"].values.astype(int)); ps.append(d[f"f_{c}"].values)
        d4[name] = _f1(np.concatenate(ys), np.concatenate(ps))
    m["f1_dict/Некорректная укладка (обе области)"] = d4["Некорректная укладка"]
    m["macro/4_dictionary_rows"] = float(np.mean(list(d4.values())))
    v = visit_table(df)
    m["visit/sensitivity"] = binary_block(v.y.values, v.p.values)["sensitivity"]
    m["visit/specificity"] = binary_block(v.y.values, v.p.values)["specificity"]
    return m


def visit_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("study").agg(y=("y_bin", "max"), p=("quality_class", "max"), n_files=("key", "size"))


def cluster_bootstrap(df: pd.DataFrame, n_boot: int, seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    studies = df["study"].unique()
    idx_by = {s: np.nonzero(df["study"].values == s)[0] for s in studies}
    acc: dict[str, list] = {}
    for _ in range(n_boot):
        samp = rng.choice(studies, size=len(studies), replace=True)
        idx = np.concatenate([idx_by[s] for s in samp])
        b = df.iloc[idx].copy()
        b["study"] = np.concatenate([[f"{s}#{j}"] * len(idx_by[s]) for j, s in enumerate(samp)])
        m = point_metrics_safe(b)
        for k, val in m.items():
            acc.setdefault(k, []).append(val)
    out = {}
    for k, vals in acc.items():
        a = np.array([x for x in vals if x is not None and not np.isnan(x)], float)
        out[k] = [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [float("nan")] * 2
        out[k + "#share_skipped"] = 1 - len(a) / len(vals)
    return out


def point_metrics_safe(b: pd.DataFrame) -> dict:
    """point_metrics, где F1 без положительных в ресэмпле не определён (NaN), как в eval_oof_metrics."""
    m = point_metrics(b)
    for c, reg, _ in TYPES:
        d = b[(b.region == reg) & b[f"y_{c}"].notna()]
        if d[f"y_{c}"].sum() == 0:
            m[f"f1/{c}"] = float("nan")
    for reg in ("spine", "hip"):
        if b[b.region == reg].y_bin.sum() == 0:
            m[f"bin_f1/{reg}"] = float("nan")
    return m


def multi_type_rows(df: pd.DataFrame) -> dict:
    out = {}
    for reg, cs in REGION_CRITS.items():
        d = df[df.region == reg]
        yt = d[[f"y_{c}" for c in cs]].fillna(0).sum(axis=1)
        pf = d[[f"f_{c}" for c in cs]].sum(axis=1)
        out[reg] = {"rows": int(len(d)), "true_rows_ge2_types": int((yt >= 2).sum()),
                    "pred_rows_ge2_types": int((pf >= 2).sum()),
                    "true_positive_rows": int((yt >= 1).sum()), "pred_flagged_rows": int((pf >= 1).sum())}
    return out


def tie_blocks(df: pd.DataFrame, thr: dict) -> dict:
    out = {}
    for c, reg, _ in TYPES:
        if f"s_{c}" not in df.columns:
            continue
        d = df[(df.region == reg) & df[f"y_{c}"].notna()]
        on = np.abs(d[f"s_{c}"].values - thr[c]) <= THR_TIE_ATOL
        sub = d[on]
        out[c] = {"threshold": thr[c], "rows_on_threshold": int(on.sum()),
                  "rows_on_threshold_positive": int(sub[f"y_{c}"].sum()),
                  "unique_frames_on_threshold": int(sub["pixel_hash"].nunique()),
                  "studies_on_threshold": int(sub["study"].nunique()),
                  "flags_total": int(d[f"f_{c}"].sum()), "positives_total": int(d[f"y_{c}"].sum())}
    return out


def evaluate(df: pd.DataFrame, n_boot: int) -> dict:
    lab = df[df["labelled"]].reset_index(drop=True)
    res = {"n_rows_input": int(len(df)), "n_rows_labelled": int(len(lab)),
           "n_rows_not_success": int((df["status"] != "Success").sum()),
           "n_studies": int(lab["study"].nunique()), "n_unique_frames": int(lab["pixel_hash"].nunique())}
    for name, d in (("files", lab), ("unique_frames", lab.drop_duplicates("pixel_hash").reset_index(drop=True))):
        blk = {"n_rows": int(len(d)), "point": point_metrics(d), "ci95": cluster_bootstrap(d, n_boot) if n_boot else {}}
        blk["binary"] = {reg: binary_block(d[d.region == reg].y_bin, d[d.region == reg].quality_class,
                                           d[d.region == reg].quality_prob) for reg in ("spine", "hip")}
        blk["binary"]["pooled"] = binary_block(d.y_bin, d.quality_class, d.quality_prob)
        blk["by_type"] = {}
        for c, reg, name_ in TYPES:
            dd = d[(d.region == reg) & d[f"y_{c}"].notna()]
            blk["by_type"][c] = {"violation_type": name_, "region": REGION_RU[reg],
                                 **binary_block(dd[f"y_{c}"].astype(int), dd[f"f_{c}"])}
        v = visit_table(d)
        blk["visit"] = binary_block(v.y.values, v.p.values)
        blk["multi_type_rows"] = multi_type_rows(d)
        res[name] = blk
    return res


# ----------------------------------------------------------------------------- вывод
def fmt(p, ci, k, nd=3):
    v = p.get(k, float("nan"))
    c = ci.get(k)
    s = f"{v:.{nd}f}"
    return s + (f" [{c[0]:.2f}; {c[1]:.2f}]" if c else "")


def to_markdown(res: dict, meta: dict) -> str:
    L = ["# Метрики в схеме организаторов (`tools/organizer_metrics.py`)", "",
         f"Источник предсказаний: {meta['source']}. Разметка: `{meta['markup']}`, лист «Калибровка». "
         f"Сторона бедра: {meta['side']}. Строк во входе {res['n_rows_input']}, размечено {res['n_rows_labelled']}, "
         f"исследований {res['n_studies']}, уникальных кадров {res['n_unique_frames']}; строк не Success: "
         f"{res['n_rows_not_success']}. 95 % ДИ — кластерный бутстрап по исследованиям, {meta['n_boot']} ресэмплов.", ""]
    if meta.get("in_sample_warning"):
        L += ["> Внимание: выгрузка на обучающих снимках — in-sample, числа завышены. Честные числа — режим `--oof`.", ""]
    for name, title in (("files", "по файлам"), ("unique_frames", "по уникальным кадрам (хэш пикселей)")):
        b = res[name]; p = b["point"]; ci = b["ci95"]
        L += [f"## {title[0].upper() + title[1:]} (строк {b['n_rows']})", "",
              "| Показатель | Позвоночник | Бедро | Общий пул |", "|---|---|---|---|",
              f"| Бинарная F1 (quality_class) | {fmt(p, ci, 'bin_f1/spine')} | {fmt(p, ci, 'bin_f1/hip')} | {fmt(p, ci, 'bin_f1/pooled')} |",
              f"| Бинарная ROC-AUC (quality_prob) | {fmt(p, ci, 'bin_auc/spine')} | {fmt(p, ci, 'bin_auc/hip')} | {fmt(p, ci, 'bin_auc/pooled')} |",
              f"| n / с нарушением | {b['binary']['spine']['n']} / {b['binary']['spine']['n_pos']} | "
              f"{b['binary']['hip']['n']} / {b['binary']['hip']['n_pos']} | {b['binary']['pooled']['n']} / {b['binary']['pooled']['n_pos']} |",
              "", "| Тип | n / с нарушением / флагов | F1 [95 % ДИ] | Чувств. | Спец. |", "|---|---|---|---|---|"]
        for c, reg, nm in TYPES:
            t = b["by_type"][c]
            L.append(f"| {nm} (`{c}`) | {t['n']} / {t['n_pos']} / {t['n_flag']} | {fmt(p, ci, 'f1/' + c)} | "
                     f"{t['sensitivity']:.3f} | {t['specificity']:.3f} |")
        L += ["", "| Трактовка macro-F1 | Значение [95 % ДИ] |", "|---|---|",
              f"| (1) по областям: позвоночник (3 типа) | {fmt(p, ci, 'macro/spine')} |",
              f"| (1) по областям: бедро (2 типа) | {fmt(p, ci, 'macro/hip')} |",
              f"| (1) по областям: среднее двух областей | {fmt(p, ci, 'macro/regions_mean')} |",
              f"| (2) по 5 критериям | {fmt(p, ci, 'macro/5_criteria')} |",
              f"| (3) по 4 строкам словаря (F1 общей «Некорректной укладки» {p['f1_dict/Некорректная укладка (обе области)']:.3f}) | {fmt(p, ci, 'macro/4_dictionary_rows')} |",
              ""]
        v = b["visit"]
        L += [f"Уровень визита: визитов {v['n']}, с нарушением {v['n_pos']}, помечено {v['n_flag']}; чувствительность "
              f"{fmt(p, ci, 'visit/sensitivity')}, специфичность {fmt(p, ci, 'visit/specificity')}, точность "
              f"{v['precision']:.3f}, F1 {v['f1']:.3f}.", ""]
        mt = b["multi_type_rows"]
        L += [f"Строки с двумя и более типами: позвоночник — истина {mt['spine']['true_rows_ge2_types']} из "
              f"{mt['spine']['true_positive_rows']} строк с нарушением, предсказано {mt['spine']['pred_rows_ge2_types']} из "
              f"{mt['spine']['pred_flagged_rows']} помеченных; бедро — истина {mt['hip']['true_rows_ge2_types']} из "
              f"{mt['hip']['true_positive_rows']}, предсказано {mt['hip']['pred_rows_ge2_types']} из {mt['hip']['pred_flagged_rows']}.", ""]
    if "tie_blocks" in res:
        L += ["## Блоки связанных рангов на пороге (OOF, строки с |score − порог| ≤ 1e-9; сервис сравнивает `>=`)", "",
              "| Критерий | Порог | Строк на пороге | Из них с нарушением | Уникальных кадров | Исследований | Флагов всего | С нарушением всего |",
              "|---|---|---|---|---|---|---|---|"]
        for c, t in res["tie_blocks"].items():
            L.append(f"| `{c}` | {t['threshold']:.4f} | {t['rows_on_threshold']} | {t['rows_on_threshold_positive']} | "
                     f"{t['unique_frames_on_threshold']} | {t['studies_on_threshold']} | {t['flags_total']} | {t['positives_total']} |")
        L.append("")
    if "oof_check" in res:
        ok = res["oof_check"]["ok"]
        L += [f"Сверка с `models/metrics_oof_full.json` (допуск 5e-4): {'совпадает' if ok else 'РАСХОЖДЕНИЕ'} "
              f"({len(res['oof_check']['items'])} величин, макс. |Δ| {res['oof_check']['max_abs_diff']:.2e}).", ""]
    return "\n".join(L)


def oof_check(res: dict) -> dict:
    ref = json.loads((ROOT / "models" / "metrics_oof_full.json").read_text(encoding="utf-8"))
    p = res["files"]["point"]
    items = []
    for reg in ("spine", "hip"):
        rb = ref["by_region_binary"][reg]
        items.append((f"bin_f1/{reg}", rb["f1"], p[f"bin_f1/{reg}"]))
        items.append((f"bin_auc/{reg}", rb["roc_auc"], p[f"bin_auc/{reg}"]))
        items.append((f"macro/{reg}", ref["macro_f1"][reg]["macro_f1"], p[f"macro/{reg}"]))
        for c in REGION_CRITS[reg]:
            items.append((f"f1/{c}", ref["by_violation_type"][f"{reg}/{c}"]["f1"], p[f"f1/{c}"]))
        items.append((f"n/{reg}", rb["n"], res["files"]["binary"][reg]["n"]))
        items.append((f"n_pos/{reg}", rb["n_pos"], res["files"]["binary"][reg]["n_pos"]))
    diffs = [abs(a - b) for _, a, b in items]
    return {"ok": bool(max(diffs) < 5e-4), "max_abs_diff": float(max(diffs)),
            "items": [{"what": k, "reference": a, "recomputed": b} for k, a, b in items]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path)
    ap.add_argument("--markup", type=Path, default=ROOT.parent / "dataset" / "разметка.xlsx")
    ap.add_argument("--sheet", default="Калибровка")
    ap.add_argument("--side", choices=["detector", "label"], default="detector")
    ap.add_argument("--side-map", type=Path, default=None)
    ap.add_argument("--dicom-root", type=Path, default=None)
    ap.add_argument("--oof", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out-md", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    a = ap.parse_args(argv)
    if not a.oof and a.results is None:
        ap.error("нужен --results или --oof")
    markup = load_markup(a.markup, a.sheet)
    thr = None
    if a.oof:
        df, thr = table_from_oof(a.side, a.side_map)
        source = "OOF (`models/oof_stacked_*.csv`; quality_prob — как в `src/eval_oof_metrics.py`)"
    else:
        df = table_from_results(a.results, a.side, a.side_map, a.dicom_root)
        source = f"`{a.results}`"
    df = attach_truth(df, markup)
    side_txt = ("анатомический детектор (`hip_features.detect_hip_side`)" if a.side == "detector"
                else f"таблица «файл → сторона» `{a.side_map or DEFAULT_SIDE_MAP.relative_to(ROOT)}`")
    res = evaluate(df, a.boot)
    if a.oof:
        lab = df[df.labelled]
        res["tie_blocks"] = tie_blocks(lab, thr)
        if a.side == "detector":
            mism = 0
            for c in CRITS:
                m = lab[f"yo_{c}"].notna()
                mism += int((lab.loc[m, f"yo_{c}"].astype(int) != lab.loc[m, f"y_{c}"].astype(int)).sum())
            res["oof_label_mismatch_vs_markup"] = mism
            res["oof_check"] = oof_check(res)
    in_sample = (not a.oof) and a.results is not None
    meta = {"source": source, "markup": a.markup.name, "side": side_txt, "n_boot": a.boot,
            "in_sample_warning": in_sample}
    res["meta"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in meta.items()}
    md = to_markdown(res, meta)
    if a.out_md:
        a.out_md.parent.mkdir(parents=True, exist_ok=True)
        a.out_md.write_text(md + "\n", encoding="utf-8")
    if a.out_json:
        a.out_json.parent.mkdir(parents=True, exist_ok=True)
        a.out_json.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(md)
    if a.oof and a.side == "detector":
        if res.get("oof_label_mismatch_vs_markup"):
            print(f"ОШИБКА: метки OOF и разметки расходятся в {res['oof_label_mismatch_vs_markup']} ячейках")
            return 1
        if not res["oof_check"]["ok"]:
            print("ОШИБКА: OOF-числа не совпали с models/metrics_oof_full.json")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
