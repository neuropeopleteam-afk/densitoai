#!/usr/bin/env python3
"""Паспорт выборки: объёмы в исследованиях, а не в кадрах, и кластерные доверительные интервалы.

Зачем. Во всех документах объёмы даны в кадрах («166 кадров, 10 позитивов»), тогда как
эффективная единица наблюдения — исследование: метка критерия задана организаторами на
исследование (для бедра — на исследование и сторону), внутри исследования кадры почти
всегда несут одну и ту же метку, а половина файлов — побайтовые клоны экспорта. Этот скрипт
строит один воспроизводимый источник таких чисел — `docs/sample_passport.json` и
`docs/sample_passport.md` — и НИЧЕГО не меняет в замороженных метриках: точечные AUC/F1
пересчитываются только для сверки с `models/metrics_summary.json` (допуск 5e-4).

Что считается (по каждому критерию ТЗ, по критериям сторон и по бинарному классу области):

  1. Объёмы: файлы, уникальные кадры по хэшу пикселей, исследования; то же для позитивов;
     доля позитивов по файлам и по исследованиям; концентрация — сколько исследований
     дают >= 50 % и >= 80 % всех положительных кадров.
  2. Эффект дизайна (design effect) кластерной выборки и эффективный размер выборки.
     Внутрикластерная корреляция rho оценивается ANOVA-оценкой для бинарного исхода
     (одно-факторный дисперсионный анализ, Fleiss–Cuzick):
         MSB = sum_i m_i (p_i - p)^2 / (k - 1),      MSW = sum_i m_i p_i (1 - p_i) / (N - k),
         m0  = (N - sum_i m_i^2 / N) / (k - 1),       rho = (MSB - MSW) / (MSB + (m0 - 1) MSW),
     где k — число исследований (кластеров), m_i — число валидных кадров исследования i,
     p_i — доля позитивов в нём, N — число кадров. rho обрезается в [0, 1].
     DEFF = 1 + (m_bar - 1) * rho, m_bar = N / k — средний размер кластера (формула Киша для
     равных кластеров; при неравных размерах она немного занижает DEFF, поэтому рядом даётся
     и вариант с m0). n_eff = N / DEFF. При rho = 1 (метка постоянна внутри исследования)
     n_eff = k: 166 кадров позвоночника несут информацию 99 исследований.
  3. Кластерный бутстрап по исследованиям (по умолчанию 2000 повторов, seed 2026 — те же
     параметры, что в `src/eval_oof_metrics.py` и `tools/recompute_f1_ci.py`) для доли
     позитивов, ROC-AUC и F1 замороженных OOF-скоров (`models/oof_stacked_*.csv`, флаги —
     `pred_label`, то есть решения сервиса при боевом пороге). Ресэмплы без обоих классов
     пропускаются (AUC и F1 не определены) — как в `tools/recompute_f1_ci.py`. Рядом —
     наивный бутстрап по файлам (кадры считаются независимыми), чтобы была видна разница.
     Процедура F1-бутстрапа воспроизводит `tools/recompute_f1_ci.bootstrap_f1_ci` (та же
     последовательность ресэмплов при том же seed), поэтому для критериев сторон интервалы
     сходятся с `metrics_summary.json` точно; для пяти критериев ТЗ в `metrics_summary.json`
     записаны интервалы из общего прогона `src/eval_oof_metrics.py` с одним генератором на
     все критерии — они совпадают с точностью бутстрапа, расхождение печатается.
  4. Состав исследований: обе области / только позвоночник / только бедро; кадры бедра по
     стороне из разметки (колонка region) и по детектору стороны (hip_side_detected).

Источники: `data/labels_for_embeddings.csv` (метки позвоночника, регион), `data/geometry_features.csv`
(метки бедра по обнаруженной стороне `hip_pos_c`/`hip_roi_c`, как в обучении), `models/oof_stacked_*.csv`,
`models/metrics_summary.json`, `models/metrics_oof_full.json` (бинарный класс области), хэш пикселей —
`outputs/pixel_hashes.csv` (tools/make_pixel_hashes.py) или пересчёт по DICOM тем же способом.

Запуск:
    python tools/sample_passport.py                       # -> docs/sample_passport.json, docs/sample_passport.md
    python tools/sample_passport.py --n-boot 200          # быстрый режим (тест)
    python tools/sample_passport.py --dataset-dir /path/к/Исследования   # хэши по DICOM из другого каталога
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))


def _ensure_src(root: Path) -> None:
    """Подключить src/ репозитория (flags_from_scores, normalize_pixels, oof_any_model)."""
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

N_BOOT = 2000
SEED = 2026
TOL_POINT = 5e-4          # допуск сверки точечных метрик с metrics_summary.json (третий знак)

REGION_NAME = {"spine": "Поясничный отдел позвоночника", "hip": "Проксимальный отдел бедра"}
VIOL_NAME = {"sp_pos": "Некорректная укладка", "sp_axis": "Не выравнена ось позвоночника",
             "sp_art": "Присутствуют посторонние предметы",
             "hip_pos": "Некорректная укладка", "hip_roi": "Некорректная область интереса",
             "rh_pos": "Некорректная укладка (правое)", "rh_roi": "Некорректная область интереса (правое)",
             "lh_pos": "Некорректная укладка (левое)", "lh_roi": "Некорректная область интереса (левое)"}
# (раздел metrics_summary.json, критерий, файл OOF) — как в tools/recompute_f1_ci.py
PLACES = [
    ("spine", "sp_pos", "oof_stacked_spine_sp_pos.csv"),
    ("spine", "sp_axis", "oof_stacked_spine_sp_axis.csv"),
    ("spine", "sp_art", "oof_stacked_spine_sp_art.csv"),
    ("hip", "hip_pos", "oof_stacked_hip_hip_pos.csv"),
    ("hip", "hip_roi", "oof_stacked_hip_hip_roi.csv"),
    ("right_hip", "rh_pos", "oof_stacked_right_hip_rh_pos.csv"),
    ("right_hip", "rh_roi", "oof_stacked_right_hip_rh_roi.csv"),
    ("left_hip", "lh_pos", "oof_stacked_left_hip_lh_pos.csv"),
    ("left_hip", "lh_roi", "oof_stacked_left_hip_lh_roi.csv"),
]
REGION_CRITERIA = {"spine": ["sp_pos", "sp_axis", "sp_art"], "hip": ["hip_pos", "hip_roi"]}
HASH_ANCHOR = "Исследования/"


# --------------------------------------------------------------------------- #
# Хэш пикселей
# --------------------------------------------------------------------------- #
def _remap(path: str, dataset_dir: Path | None) -> Path:
    if dataset_dir is None:
        return Path(path)
    i = path.find(HASH_ANCHOR)
    return dataset_dir / path[i + len(HASH_ANCHOR):] if i >= 0 else Path(path)


def load_or_compute_hashes(labels: pd.DataFrame, hashes_csv: Path | None, dataset_dir: Path | None):
    """sha1 нормализованных пикселей (tools/make_pixel_hashes.py, вариант baseline).

    Порядок: готовый CSV -> пересчёт по DICOM -> None (уникальные кадры не считаются).
    """
    if hashes_csv is not None and hashes_csv.exists():
        h = pd.read_csv(hashes_csv)
        if "pixel_hash" in h.columns and len(h) == len(labels):
            m = h.set_index("file_path")["pixel_hash"]
            if set(labels["file_path"]) <= set(m.index):
                return labels["file_path"].map(m).to_numpy(), f"файл {hashes_csv}"
    first = _remap(labels["file_path"].iloc[0], dataset_dir)
    if not first.exists():
        return None, "нет ни готового файла хэшей, ни DICOM — уникальные кадры не посчитаны"
    import pydicom  # noqa: E402
    from inference import normalize_pixels  # noqa: E402
    out = []
    for p in labels["file_path"]:
        ds = pydicom.dcmread(str(_remap(p, dataset_dir)), force=True)
        img = normalize_pixels(ds)
        out.append(hashlib.sha1(img.tobytes() + str(img.shape).encode()).hexdigest())
    return np.asarray(out), "пересчёт по DICOM (inference.normalize_pixels, baseline; как tools/make_pixel_hashes.py)"


# --------------------------------------------------------------------------- #
# Статистика кластеров
# --------------------------------------------------------------------------- #
def icc_anova_binary(y: np.ndarray, groups: np.ndarray) -> dict:
    """ANOVA-оценка внутрикластерной корреляции для бинарного исхода и эффект дизайна.

    См. докстринг модуля, п. 2. Возвращает rho, m_bar, m0, DEFF (по m_bar и по m0), n_eff.
    """
    df = pd.DataFrame({"y": y.astype(float), "g": groups})
    agg = df.groupby("g")["y"].agg(["size", "mean"])
    m = agg["size"].to_numpy(float)
    p_i = agg["mean"].to_numpy(float)
    N, k = float(m.sum()), len(m)
    p = float(df["y"].mean())
    res = {"n_frames": int(N), "n_clusters": int(k), "m_bar": N / k if k else float("nan")}
    if k < 2 or N <= k or p in (0.0, 1.0):
        rho = 1.0 if (k >= 2 and N > k) else float("nan")     # вырожденный случай
        res.update({"icc": rho, "m0": float("nan")})
    else:
        msb = float(np.sum(m * (p_i - p) ** 2) / (k - 1))
        msw = float(np.sum(m * p_i * (1 - p_i)) / (N - k))
        m0 = (N - np.sum(m ** 2) / N) / (k - 1)
        den = msb + (m0 - 1) * msw
        rho = (msb - msw) / den if den > 0 else 1.0
        rho = float(min(1.0, max(0.0, rho)))
        res.update({"icc": rho, "m0": float(m0)})
    rho = res["icc"]
    if np.isnan(rho):
        res.update({"deff": float("nan"), "deff_m0": float("nan"), "n_eff": float("nan")})
    else:
        deff = 1.0 + (res["m_bar"] - 1.0) * rho
        deff_m0 = 1.0 + (res["m0"] - 1.0) * rho if not np.isnan(res["m0"]) else deff
        res.update({"deff": float(deff), "deff_m0": float(deff_m0), "n_eff": float(N / deff)})
    return res


def concentration(y: np.ndarray, groups: np.ndarray) -> dict:
    """Сколько исследований дают >= 50 % и >= 80 % всех положительных кадров."""
    pos = pd.Series(y.astype(int)).groupby(pd.Series(groups)).sum().sort_values(ascending=False)
    total = int(pos.sum())
    if total == 0:
        return {"studies_for_50pct": 0, "studies_for_80pct": 0, "top1_share": 0.0, "positives_by_study": []}
    cum = pos.cumsum() / total
    return {"studies_for_50pct": int((cum < 0.5).sum() + 1),
            "studies_for_80pct": int((cum < 0.8).sum() + 1),
            "top1_share": float(pos.iloc[0] / total),
            "positives_by_study": [int(v) for v in pos[pos > 0].to_numpy()]}


def bootstrap(y: np.ndarray, s: np.ndarray | None, flag: np.ndarray | None, groups: np.ndarray,
              n_boot: int, seed: int, by_cluster: bool) -> dict:
    """Перцентильный бутстрап (2.5/97.5) доли позитивов, ROC-AUC и F1.

    by_cluster=True — ресэмпл исследований (np.unique(groups), rng.choice с возвратом — как в
    tools/recompute_f1_ci.py); False — наивный ресэмпл файлов. Ресэмплы без обоих классов
    пропускаются для AUC и F1 (доля позитивов считается всегда).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    if by_cluster:
        uniq = np.unique(groups)
        idx_by_g = {g: np.nonzero(groups == g)[0] for g in uniq}
    prev, aucs, f1s = [], [], []
    for _ in range(n_boot):
        if by_cluster:
            sample = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([idx_by_g[g] for g in sample])
        else:
            idx = rng.integers(0, n, size=n)
        yb = y[idx]
        prev.append(float(yb.mean()))
        if len(np.unique(yb)) < 2:
            continue
        if s is not None:
            aucs.append(float(roc_auc_score(yb, s[idx])))
        if flag is not None:
            f1s.append(float(f1_score(yb, flag[idx], zero_division=0)))

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [float("nan")] * 2

    out = {"unit": "исследование" if by_cluster else "файл", "n_boot": int(n_boot), "seed": int(seed),
           "prevalence_ci": ci(prev), "share_skipped": 1.0 - (len(f1s) if flag is not None else len(aucs)) / float(n_boot)}
    if s is not None:
        out["roc_auc_ci"] = ci(aucs)
    if flag is not None:
        out["f1_ci"] = ci(f1s)
        out["f1_boot_mean"] = float(np.mean(f1s)) if f1s else float("nan")
    return out


def volumes(y: np.ndarray, groups: np.ndarray, hashes: np.ndarray | None) -> dict:
    pos = y.astype(int) == 1
    d = {"files": int(len(y)), "studies": int(len(np.unique(groups))),
         "pos_files": int(pos.sum()), "pos_studies": int(len(np.unique(groups[pos])))}
    d["unique_frames"] = int(len(np.unique(hashes))) if hashes is not None else None
    d["pos_unique_frames"] = int(len(np.unique(hashes[pos]))) if hashes is not None else None
    d["prevalence_files"] = d["pos_files"] / d["files"] if d["files"] else float("nan")
    d["prevalence_studies"] = d["pos_studies"] / d["studies"] if d["studies"] else float("nan")
    return d


# --------------------------------------------------------------------------- #
# Сборка паспорта
# --------------------------------------------------------------------------- #
def build_passport(root: Path = ROOT, n_boot: int = N_BOOT, seed: int = SEED,
                   hashes_csv: Path | None = None, dataset_dir: Path | None = None,
                   with_region_binary: bool = True) -> dict:
    _ensure_src(root)
    from eval_oof_metrics import flags_from_scores  # noqa: E402
    labels = pd.read_csv(root / "data" / "labels_for_embeddings.csv")
    geom = pd.read_csv(root / "data" / "geometry_features.csv")
    assert (labels["file_path"].values == geom["file_path"].values).all(), "labels/geometry: порядок строк"
    summary = json.loads((root / "models" / "metrics_summary.json").read_text(encoding="utf-8"))
    full_path = root / "models" / "metrics_oof_full.json"
    full = json.loads(full_path.read_text(encoding="utf-8")) if full_path.exists() else {}

    if hashes_csv is None:
        hashes_csv = root / "outputs" / "pixel_hashes.csv"
    hashes, hash_src = load_or_compute_hashes(labels, hashes_csv, dataset_dir)
    labels = labels.assign(pixel_hash=hashes if hashes is not None else np.nan)
    hash_by_file = labels.set_index("file_path")["pixel_hash"] if hashes is not None else None

    out = {"version": summary.get("version", None), "n_boot": n_boot, "seed": seed,
           "pixel_hash_source": hash_src, "point_check_tolerance": TOL_POINT,
           "method": {
               "unit": "исследование (StudyInstanceUID); метка критерия задана на исследование, для бедра — на исследование и сторону",
               "unique_frames": "sha1 нормализованных пикселей (tools/make_pixel_hashes.py, baseline)",
               "icc": "ANOVA-оценка внутрикластерной корреляции для бинарного исхода (Fleiss–Cuzick), rho в [0; 1]",
               "deff": "1 + (m_bar - 1) * rho, m_bar = N / k; n_eff = N / DEFF",
               "ci": f"перцентильный бутстрап 95 %, {n_boot} ресэмплов, seed {seed}; кластер = исследование; "
                     "ресэмплы без обоих классов пропущены для AUC/F1; рядом наивный бутстрап по файлам",
               "flags": "pred_label из models/oof_stacked_*.csv (решение сервиса при боевом пороге)",
           }}

    # ---- 1. общая структура ----
    hip_rows = labels["region"].isin(["right_hip", "left_hip"])
    st = labels.groupby("study")["region"].agg(lambda r: set(r))
    has_sp = st.apply(lambda x: "spine" in x)
    has_hip = st.apply(lambda x: bool(x & {"right_hip", "left_hip"}))
    side_det = geom["hip_side_detected"]
    total = {
        "files": int(len(labels)), "studies": int(labels["study"].nunique()),
        "unique_frames": int(labels["pixel_hash"].nunique()) if hashes is not None else None,
        "duplicate_files": int(len(labels) - labels["pixel_hash"].nunique()) if hashes is not None else None,
        "hash_groups_spanning_studies": int((labels.groupby("pixel_hash")["study"].nunique() > 1).sum()) if hashes is not None else None,
        "files_by_region_label": {k: int(v) for k, v in labels["region"].value_counts().items()},
        "unique_frames_by_region_label": ({k: int(v) for k, v in labels.groupby("region")["pixel_hash"].nunique().items()}
                                          if hashes is not None else None),
        "studies_by_region": {"spine": int(has_sp.sum()), "hip": int(has_hip.sum())},
        "studies_both_regions": int((has_sp & has_hip).sum()),
        "studies_spine_only": int((has_sp & ~has_hip).sum()),
        "studies_hip_only": int((~has_sp & has_hip).sum()),
        "files_not_applicable": int((labels["applicable"] == False).sum()),  # noqa: E712
        "hip_frames_by_label_side": {"right": int((labels["region"] == "right_hip").sum()),
                                     "left": int((labels["region"] == "left_hip").sum())},
        "hip_frames_by_detected_side": {"right": int((side_det == "right").sum()), "left": int((side_det == "left").sum())},
        "hip_frames_side_disagree": int((hip_rows & side_det.notna() &
                                         (labels["region"].str.replace("_hip", "") != side_det)).sum()),
        "hip_files_with_side_label_valid": int(geom["hip_pos_c"].notna().sum()),
        "frames_per_study": {
            "spine": {k: int(v) for k, v in labels[labels.region == "spine"].groupby("study").size().value_counts().sort_index().items()},
            "hip": {k: int(v) for k, v in labels[hip_rows].groupby("study").size().value_counts().sort_index().items()},
        },
    }
    out["total"] = total

    # ---- 2. критерии ----
    crit_out, checks = {}, []
    for section, crit, fname in PLACES:
        node = summary.get(section, {}).get(crit)
        path = root / "models" / fname
        if node is None or not path.exists():
            continue
        df = pd.read_csv(path)
        y = df["y_true"].to_numpy().astype(int)
        s = df["oof_stacked"].to_numpy().astype(float)
        g = df["study"].to_numpy()
        thr = float(node["threshold"])
        flag = flags_from_scores(df, s, thr)
        h = hash_by_file.loc[df["file_path"]].to_numpy() if hash_by_file is not None else None

        vol = volumes(y, g, h)
        rec = {"section": section, "violation_type": VIOL_NAME.get(crit, crit), "threshold": thr,
               **vol, "concentration": concentration(y, g), "design_effect": icc_anova_binary(y, g)}
        auc = float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")
        f1 = float(f1_score(y, flag, zero_division=0))
        rec["point"] = {"roc_auc": auc, "f1": f1, "n_flag": int(flag.sum())}
        rec["frozen"] = {"n_valid": node["n_valid"], "n_pos": node["n_pos"], "auc_stacked": node["auc_stacked"],
                         "f1_oof": node["f1_oof"], "f1_ci": [node.get("f1_ci_lo"), node.get("f1_ci_hi")]}
        rec["ci_by_study"] = bootstrap(y, s, flag, g, n_boot, seed, by_cluster=True)
        rec["ci_by_file"] = bootstrap(y, s, flag, g, n_boot, seed, by_cluster=False)
        chk = {"criterion": crit,
               "n_valid_ok": int(vol["files"]) == int(node["n_valid"]),
               "n_pos_ok": int(vol["pos_files"]) == int(node["n_pos"]),
               "auc_diff": abs(auc - float(node["auc_stacked"])),
               "f1_diff": abs(f1 - float(node["f1_oof"]))}
        chk["auc_ok"] = chk["auc_diff"] <= TOL_POINT
        chk["f1_ok"] = chk["f1_diff"] <= TOL_POINT
        fv = full.get("by_violation_type", {}).get(f"{section}/{crit}")
        if fv is not None:
            # второй замороженный источник: models/metrics_oof_full.json (docs/METRICS_REPORT.md)
            chk["auc_diff_oof_full"] = abs(auc - float(fv["roc_auc"]))
            chk["f1_diff_oof_full"] = abs(f1 - float(fv["f1"]))
            chk["auc_ok_oof_full"] = chk["auc_diff_oof_full"] <= TOL_POINT
            rec["frozen"]["oof_full"] = {"roc_auc": fv["roc_auc"], "f1": fv["f1"], "roc_auc_ci": fv["ci95"]["roc_auc"],
                                         "f1_ci": fv["ci95"]["f1"]}
        if node.get("f1_ci_lo") is not None:
            chk["f1_ci_frozen"] = [float(node["f1_ci_lo"]), float(node["f1_ci_hi"])]
            chk["f1_ci_here"] = rec["ci_by_study"]["f1_ci"]
            chk["f1_ci_max_abs_diff"] = float(max(abs(a - b) for a, b in zip(chk["f1_ci_frozen"], chk["f1_ci_here"])))
        checks.append(chk)
        rec["check"] = chk
        crit_out[crit] = rec
    out["criteria"] = crit_out

    # ---- 3. бинарный класс области ----
    region_out = {}
    for region, criteria in REGION_CRITERIA.items():
        frames = {c: pd.read_csv(root / "models" / f"oof_stacked_{region}_{c}.csv") for c in criteria}
        base = frames[criteria[0]]
        for c in criteria[1:]:
            assert (frames[c]["file_path"].values == base["file_path"].values).all(), "OOF files misaligned"
        ytrue = np.zeros(len(base), int); flags = np.zeros(len(base), int); crit_max = np.zeros(len(base))
        for c, df in frames.items():
            thr = float(summary[region][c]["threshold"])
            flags = np.maximum(flags, flags_from_scores(df, df["oof_stacked"].to_numpy(float), thr))
            ytrue = np.maximum(ytrue, df["y_true"].to_numpy().astype(int))
            crit_max = np.maximum(crit_max, df["oof_stacked"].to_numpy(float))
        g = base["study"].to_numpy()
        h = hash_by_file.loc[base["file_path"]].to_numpy() if hash_by_file is not None else None
        prob = None
        prob_note = "quality_prob не пересобран"
        if with_region_binary:
            try:
                from eval_oof_metrics import oof_any_model  # noqa: E402
                anym = oof_any_model(region, criteria).set_index("file_path").loc[base["file_path"].values]
                assert (anym["y_any"].values == ytrue).all(), "any-label mismatch"
                raw = 0.5 * anym["any_model_oof"].values + 0.5 * crit_max
                prob = np.where(flags == 1, 0.5 + 0.5 * raw, np.minimum(0.5 * raw, 0.499999))
                prob_note = "quality_prob пересобран как в src/eval_oof_metrics.py (any-модель OOF + max по критериям, согласование с классом)"
            except Exception as e:  # noqa: BLE001
                prob_note = f"quality_prob не пересобран: {type(e).__name__}: {e}"
        vol = volumes(ytrue, g, h)
        rec = {"anatomical_region": REGION_NAME[region], **vol, "concentration": concentration(ytrue, g),
               "design_effect": icc_anova_binary(ytrue, g), "prob_note": prob_note}
        f1 = float(f1_score(ytrue, flags, zero_division=0))
        auc = float(roc_auc_score(ytrue, prob)) if prob is not None else float("nan")
        rec["point"] = {"f1": f1, "roc_auc": auc, "n_flag": int(flags.sum())}
        rec["ci_by_study"] = bootstrap(ytrue, prob, flags, g, n_boot, seed, by_cluster=True)
        rec["ci_by_file"] = bootstrap(ytrue, prob, flags, g, n_boot, seed, by_cluster=False)
        fb = full.get("by_region_binary", {}).get(region)
        if fb:
            rec["frozen"] = {"n": fb["n"], "n_pos": fb["n_pos"], "f1": fb["f1"], "roc_auc": fb["roc_auc"],
                             "f1_ci": fb["ci95"]["f1"], "roc_auc_ci": fb["ci95"]["roc_auc"]}
            chk = {"criterion": f"{region}/binary", "n_valid_ok": vol["files"] == fb["n"], "n_pos_ok": vol["pos_files"] == fb["n_pos"],
                   "f1_diff": abs(f1 - fb["f1"]), "auc_diff": abs(auc - fb["roc_auc"]) if prob is not None else None}
            chk["f1_ok"] = chk["f1_diff"] <= TOL_POINT
            chk["auc_ok"] = (chk["auc_diff"] <= TOL_POINT) if prob is not None else None
            rec["check"] = chk
            checks.append(chk)
        region_out[region] = rec
    out["region_binary"] = region_out
    out["checks"] = checks
    # Расхождения описываются, а не подгоняются: точечное значение считается воспроизведённым, если оно
    # совпадает хотя бы с одним из двух замороженных источников (metrics_summary.json или
    # metrics_oof_full.json); несовпадение с metrics_summary.json перечисляется отдельно.
    out["discrepancies"] = [
        {"criterion": c["criterion"], "what": "auc_stacked в metrics_summary.json", "abs_diff": c["auc_diff"],
         "matches_metrics_oof_full": c.get("auc_ok_oof_full")}
        for c in checks if c.get("auc_ok") is False
    ] + [
        {"criterion": c["criterion"], "what": "f1_oof в metrics_summary.json", "abs_diff": c["f1_diff"]}
        for c in checks if c.get("f1_ok") is False
    ]
    out["all_points_match"] = all(
        c["n_valid_ok"] and c["n_pos_ok"] and c["f1_ok"] and (c["auc_ok"] in (True, None) or c.get("auc_ok_oof_full"))
        for c in checks)
    out["all_points_match_metrics_summary"] = all(
        c["n_valid_ok"] and c["n_pos_ok"] and c["f1_ok"] and c["auc_ok"] in (True, None) for c in checks)
    return out


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def _ci(v, nd=2):
    if v is None or any(x is None or (isinstance(x, float) and np.isnan(x)) for x in v):
        return "—"
    return f"[{v[0]:.{nd}f}; {v[1]:.{nd}f}]"


def to_markdown(p: dict) -> str:
    t = p["total"]
    L = ["# Паспорт выборки", "",
         f"Источник: `docs/sample_passport.json` (`python tools/sample_passport.py`, бутстрап {p['n_boot']} ресэмплов, seed {p['seed']}). "
         "Единица наблюдения — исследование. Замороженные метрики не меняются; паспорт добавляет к ним объёмы в исследованиях и кластерные интервалы.", "",
         "## Состав", "",
         f"- Файлов {t['files']}, исследований {t['studies']}, уникальных кадров по хэшу пикселей "
         f"{t['unique_frames'] if t['unique_frames'] is not None else '—'} (дубликатов {t['duplicate_files'] if t['duplicate_files'] is not None else '—'}; "
         f"хэшей, встречающихся в двух исследованиях: {t['hash_groups_spanning_studies'] if t['hash_groups_spanning_studies'] is not None else '—'}).",
         f"- Исследований с обеими областями {t['studies_both_regions']}, только позвоночник {t['studies_spine_only']}, только бедро {t['studies_hip_only']}; "
         f"с кадрами позвоночника {t['studies_by_region']['spine']}, с кадрами бедра {t['studies_by_region']['hip']}.",
         f"- Кадров бедра по стороне из разметки: правое {t['hip_frames_by_label_side']['right']}, левое {t['hip_frames_by_label_side']['left']}; "
         f"по детектору стороны: правое {t['hip_frames_by_detected_side']['right']}, левое {t['hip_frames_by_detected_side']['left']}; "
         f"расходятся на {t['hip_frames_side_disagree']} кадрах. Кадров бедра с применимой разметкой стороны: {t['hip_files_with_side_label_valid']}.",
         f"- Кадров без применимой разметки: {t['files_not_applicable']}.", "",
         "## Объёмы по критериям", "",
         "| Критерий | Файлы | Уник. кадры | Исслед. | Позитивы: файлы / уник. / исслед. | Доля позит. (файлы / исслед.) | Исслед. на 50 % / 80 % позитивов | ICC | DEFF | n_eff |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    rows = list(p["criteria"].items()) + [(f"{r}: есть нарушение", v) for r, v in p["region_binary"].items()]
    for name, c in rows:
        de, cc = c["design_effect"], c["concentration"]
        uf = c["unique_frames"] if c["unique_frames"] is not None else "—"
        puf = c["pos_unique_frames"] if c["pos_unique_frames"] is not None else "—"
        L.append(f"| `{name}` | {c['files']} | {uf} | {c['studies']} | {c['pos_files']} / {puf} / {c['pos_studies']} | "
                 f"{c['prevalence_files']:.3f} / {c['prevalence_studies']:.3f} | {cc['studies_for_50pct']} / {cc['studies_for_80pct']} | "
                 f"{de['icc']:.2f} | {de['deff']:.2f} | {de['n_eff']:.0f} |")
    L += ["", "ICC — внутрикластерная корреляция метки внутри исследования (ANOVA-оценка), DEFF = 1 + (m − 1)·ICC, "
          "n_eff = файлы / DEFF — эффективный размер выборки в независимых наблюдениях.", "",
          "## Интервалы: по исследованиям и наивные по файлам", "",
          "| Критерий | Доля позитивов, ДИ по исслед. | ДИ по файлам | ROC-AUC | ДИ по исслед. | ДИ по файлам | F1 | ДИ по исслед. | ДИ по файлам | Замороженный ДИ F1 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, c in rows:
        a, b, pt = c["ci_by_study"], c["ci_by_file"], c["point"]
        fz = c.get("frozen", {})
        fro = fz.get("f1_ci")
        # Точечные AUC/F1 в таблице — опубликованные (замороженные) значения из metrics_summary.json;
        # пересчёт по OOF-файлу сравнивается с ними в разделе «Сверка».
        auc_pt = fz.get("auc_stacked", fz.get("roc_auc", pt["roc_auc"]))
        f1_pt = fz.get("f1_stacked", fz.get("f1", pt["f1"]))
        L.append(f"| `{name}` | {_ci(a['prevalence_ci'], 3)} | {_ci(b['prevalence_ci'], 3)} | {float(auc_pt):.3f} | {_ci(a.get('roc_auc_ci'))} | "
                 f"{_ci(b.get('roc_auc_ci'))} | {float(f1_pt):.3f} | {_ci(a['f1_ci'])} | {_ci(b['f1_ci'])} | {_ci(fro)} |")
    L += ["", "## Сверка с замороженными метриками", "",
          "| Критерий | n совпало | позитивы совпали | Δ AUC | Δ F1 | макс. Δ границ ДИ F1 |", "|---|---|---|---|---|---|"]
    for c in p["checks"]:
        d = c.get("f1_ci_max_abs_diff")
        auc_d = c["auc_diff"]
        L.append(f"| `{c['criterion']}` | {'да' if c['n_valid_ok'] else 'НЕТ'} | {'да' if c['n_pos_ok'] else 'НЕТ'} | "
                 f"{'—' if auc_d is None else f'{auc_d:.5f}'} | {c['f1_diff']:.5f} | {'—' if d is None else f'{d:.3f}'} |")
    L.append("")
    if p["all_points_match_metrics_summary"]:
        L.append("Все точечные значения совпадают с `models/metrics_summary.json` в пределах 5e-4.")
    else:
        for d in p["discrepancies"]:
            L.append(f"- `{d['criterion']}`: {d['what']} отличается на {d['abs_diff']:.5f}"
                     + ("; с `models/metrics_oof_full.json` совпадает." if d.get("matches_metrics_oof_full") else "."))
        L.append("")
        L.append("Остальные точечные значения совпадают с `models/metrics_summary.json` в пределах 5e-4."
                 if p["all_points_match"] else "ВНИМАНИЕ: есть невоспроизведённые точечные значения.")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Паспорт выборки")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--hashes", default=None, help="CSV с pixel_hash (по умолчанию outputs/pixel_hashes.csv)")
    ap.add_argument("--dataset-dir", default=os.environ.get("DENSITO_DATASET_DIR"),
                    help="каталог «Исследования» для пересчёта хэшей, если пути в labels устарели")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--no-region-binary-prob", action="store_true", help="не пересобирать quality_prob (быстрее)")
    ap.add_argument("--md-only", action="store_true", help="только перерисовать markdown из готового JSON")
    a = ap.parse_args()
    root = Path(a.root)
    if a.md_only:
        out_json = Path(a.out_json) if a.out_json else root / "docs" / "sample_passport.json"
        out_md = Path(a.out_md) if a.out_md else root / "docs" / "sample_passport.md"
        out_md.write_text(to_markdown(json.loads(out_json.read_text(encoding="utf-8"))), encoding="utf-8")
        print(f"записано: {out_md}")
        return 0
    p = build_passport(root, a.n_boot, a.seed, Path(a.hashes) if a.hashes else None,
                       Path(a.dataset_dir) if a.dataset_dir else None, not a.no_region_binary_prob)
    out_json = Path(a.out_json) if a.out_json else root / "docs" / "sample_passport.json"
    out_md = Path(a.out_md) if a.out_md else root / "docs" / "sample_passport.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(p, ensure_ascii=False, indent=1, allow_nan=True), encoding="utf-8")
    md = to_markdown(p)
    out_md.write_text(md, encoding="utf-8")
    print(md)
    print(f"записано: {out_json}, {out_md}")
    return 0 if p["all_points_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
