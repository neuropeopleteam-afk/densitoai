#!/usr/bin/env python3
"""Экзамен на подсказки (shortcut audit): может ли качество модели объясняться не изображением.

Две части, обе на кадрах организаторов (499 файлов, 100 исследований), модели и пороги не меняются.

A. Отрицательные контроли. Та же целевая переменная, что у сервиса (метки OOF поставки), но модель —
   логистическая регрессия (StandardScaler, class_weight='balanced', C = 1) только на «неизображенческих»
   признаках. Проверка: StratifiedGroupKFold(5) по группам «исследование + хэш пикселей»
   (объединение компонент: клоны одного кадра никогда не расходятся по фолдам), 10 повторов разбиения.
   Наборы признаков:
     C1  метаданные DICOM: SoftwareVersions (one-hot), SeriesNumber, InstanceNumber, наличие
         необязательных тегов (OperatorsName, PerformingPhysicianName), корень имени папки
         исследования (1.2.840… или 2.25…);
     C2  параметры экспозиции: TotalNumberOfExposures, ExposedArea (2 числа), EntranceDoseInmGy —
         физически связаны с длиной скана, это не чистый контроль (см. отчёт);
     C3  размер кадра: Rows, Columns;
     C4  число файлов в исследовании и число кадров этой области в исследовании;
     C5  порядок строк: номер строки в таблице признаков и номер файла CR00000X;
     C6  хэш UID: sha1 имени папки исследования и SOPInstanceUID, два числа в [0, 1).
   Для каждого контроля — нулевая полоса: те же признаки при метках, переставленных между уникальными
   кадрами (клоны сохраняют общую метку; 50 перестановок, 95-й перцентиль AUC).

B. Перестановка меток для стека изображения с входами поставки (контур A: признаки контура A критерия
   и вариант предобработки из config.yaml; контур B: боевой источник эмбеддингов, PCA 32; среднее двух
   вероятностей вместо рангового; тот же протокол CV):
     B1  метки на реальных данных;
     B2  метки переставлены между уникальными кадрами (клоны — одна метка), CV по группам
         «исследование + хэш» (20 перестановок) — ожидание AUC ≈ 0.5;
         если выше, значит протокол CV пропускает утечку;
     B3  метки переставлены внутри исследования (20 перестановок) — у позвоночника метка одна на
         исследование, перестановка ничего не меняет (контроль корректности); у бедра метка на сторону,
         и падение AUC показывает, что модель различает кадры внутри исследования, а не узнаёт исследование.
   Это упрощённая копия обучения (без рангового стэкинга и повторов OOF), поэтому B1 близок к боевому OOF,
   но не равен ему и метрикой сервиса не является.

Запуск: DENSITO_ROOT=$PWD ../venv/bin/python tools/shortcut_audit.py
Выход: docs/shortcut_audit.json (числа для docs/SHORTCUT_AUDIT.md), data/slices/shortcut_meta_499.csv.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
DATASET = Path(os.environ.get("DENSITO_DATASET", ROOT.parent / "dataset"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

N_REPEAT = 10
N_NULL = 50
N_PERM_MODEL = 20
SEED = 2026
REGIONS = {"spine": ["sp_pos", "sp_axis", "sp_art"], "hip": ["hip_pos", "hip_roi"]}
REGION_ROWS = {"spine": ["spine"], "hip": ["right_hip", "left_hip"]}


def rel_path(p: str) -> str:
    return p.split("Исследования/", 1)[1] if "Исследования/" in p else p


def h01(s: str) -> float:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16) / 2 ** 32


def read_meta(rels: list[str]) -> pd.DataFrame:
    import pydicom
    rows = []
    for r in rels:
        d = pydicom.dcmread(str(DATASET / "Исследования" / r), stop_before_pixels=True)
        area = list(getattr(d, "ExposedArea", [np.nan, np.nan]) or [np.nan, np.nan])
        study_dir = r.split("/", 1)[0]
        rows.append({"rel_path": r,
                     "SoftwareVersions": str(getattr(d, "SoftwareVersions", "")),
                     "SeriesNumber": float(getattr(d, "SeriesNumber", np.nan) or np.nan),
                     "InstanceNumber": float(getattr(d, "InstanceNumber", np.nan) or np.nan),
                     "has_OperatorsName": float("OperatorsName" in d),
                     "has_PerformingPhysicianName": float("PerformingPhysicianName" in d),
                     "uid_root_2_25": float(study_dir.startswith("2.25.")),
                     "TotalNumberOfExposures": float(getattr(d, "TotalNumberOfExposures", np.nan) or np.nan),
                     "ExposedArea_0": float(area[0]), "ExposedArea_1": float(area[1] if len(area) > 1 else np.nan),
                     "EntranceDoseInmGy": float(getattr(d, "EntranceDoseInmGy", np.nan) or np.nan),
                     "Rows": float(d.Rows), "Columns": float(d.Columns),
                     "file_no": float(Path(r).stem.replace("CR", "") or 0),
                     "uid_hash_study": h01(study_dir),
                     "uid_hash_sop": h01(str(getattr(d, "SOPInstanceUID", "")))})
    return pd.DataFrame(rows)


def groups_study_hash(study: np.ndarray, phash: np.ndarray) -> np.ndarray:
    """Компоненты связности графа «исследование — хэш пикселей»."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for s, h in zip(study, phash):
        a, b = find(("s", s)), find(("h", h))
        if a != b:
            parent[a] = b
    roots = [find(("s", s)) for s in study]
    u = {r: i for i, r in enumerate(dict.fromkeys(roots))}
    return np.array([u[r] for r in roots])


def cv_auc(X, y, groups, seed, model="lr"):
    oof = np.zeros(len(y))
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y, groups):
        med = np.nanmedian(X[tr], axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        Xtr = np.where(np.isnan(X[tr]), med, X[tr]); Xte = np.where(np.isnan(X[te]), med, X[te])
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced").fit(sc.transform(Xtr), y[tr])
        oof[te] = clf.predict_proba(sc.transform(Xte))[:, 1]
    return roc_auc_score(y, oof)


def perm_between(y, groups, rng):
    """Метки переставляются между единицами целиком (метка единицы = максимум по её кадрам).
    Единица — уникальный кадр (хэш пикселей): клоны получают одну метку, доля нарушений сохраняется."""
    ug = np.unique(groups)
    gl = np.array([y[groups == g].max() for g in ug])
    new = dict(zip(ug, rng.permutation(gl)))
    return np.array([new[g] for g in groups])


def perm_within(y, study, rng):
    out = y.copy()
    for s in np.unique(study):
        i = np.nonzero(study == s)[0]
        out[i] = rng.permutation(y[i])
    return out


def build_region(region: str, meta: pd.DataFrame):
    base = None
    for crit in REGIONS[region]:
        df = pd.read_csv(ROOT / "models" / f"oof_stacked_{region}_{crit}.csv")
        df = df.rename(columns={"y_true": f"y_{crit}", "oof_stacked": f"s_{crit}"})[["study", "file_path", f"y_{crit}", f"s_{crit}"]]
        base = df if base is None else base.merge(df, on=["study", "file_path"])
    base["y_binary"] = base[[f"y_{c}" for c in REGIONS[region]]].max(axis=1).astype(int)
    base["rel_path"] = base["file_path"].map(rel_path)
    base = base.merge(meta, on="rel_path", how="left")
    # C4: число файлов в исследовании (все области) и кадров этой области
    base["n_files_study"] = base["study"].map(meta.assign(study=meta.rel_path.str.split("/").str[0]).groupby("study").size())
    base["n_frames_region_study"] = base.groupby("study")["file_path"].transform("size")
    # C5: номер строки в таблице признаков (порядок, в котором файлы шли в обучение)
    g = pd.read_csv(ROOT / "data" / "geometry_features.csv")
    g = g[g.region.isin(REGION_ROWS[region])].reset_index(drop=True)
    g["row_no"] = np.arange(len(g))
    base = base.merge(g[["file_path", "row_no"]], on="file_path", how="left")
    ph = pd.read_csv(ROOT / "docs" / "k5" / "pixel_hashes.csv")
    ph["rel_path"] = ph["file_path"].map(rel_path)
    base = base.merge(ph[["rel_path", "pixel_hash"]].drop_duplicates("rel_path"), on="rel_path", how="left")
    return base


def control_colnames(sw_levels):
    return {"C1_metadata": [f"SoftwareVersions={v}" for v in sw_levels] +
            ["SeriesNumber", "InstanceNumber", "has_OperatorsName", "has_PerformingPhysicianName", "uid_root_2_25"],
            "C2_exposure": ["TotalNumberOfExposures", "ExposedArea_0", "ExposedArea_1", "EntranceDoseInmGy"],
            "C3_frame_size": ["Rows", "Columns"], "C4_files_in_study": ["n_files_study", "n_frames_region_study"],
            "C5_row_order": ["row_no", "file_no"], "C6_uid_hash": ["uid_hash_study", "uid_hash_sop"]}


def drill_down(X, names, y, score, study):
    """Разбор сработавшего контроля: одномерный AUC каждого столбца; для самого сильного — страты
    (значения или половины по медиане), доля нарушений и AUC боевой OOF-оценки внутри каждой страты."""
    uni = {}
    for j, n in enumerate(names):
        v = X[:, j]
        ok = ~np.isnan(v)
        uni[n] = float(roc_auc_score(y[ok], v[ok])) if len(np.unique(v[ok])) > 1 and len(np.unique(y[ok])) > 1 else 0.5
    top = max(uni, key=lambda k: abs(uni[k] - 0.5))
    v = X[:, names.index(top)]
    strata = v if len(np.unique(v)) <= 4 else (v > np.nanmedian(v)).astype(float)
    st = {}
    for u in np.unique(strata):
        m = strata == u
        st[str(u)] = {"n": int(m.sum()), "n_studies": int(len(np.unique(study[m]))), "n_pos": int(y[m].sum()),
                      "prevalence": float(y[m].mean()),
                      "production_auc_in_stratum": float(roc_auc_score(y[m], score[m])) if 0 < y[m].sum() < m.sum() else None}
    return {"univariate_auc": uni, "top_feature": top, "strata_of_top": st}


def control_sets(df: pd.DataFrame, sw_levels):
    sw = np.stack([(df.SoftwareVersions == v).astype(float).to_numpy() for v in sw_levels], 1)
    c = lambda cols: df[cols].to_numpy(float)
    return {
        "C1_metadata": np.hstack([sw, c(["SeriesNumber", "InstanceNumber", "has_OperatorsName",
                                           "has_PerformingPhysicianName", "uid_root_2_25"])]),
        "C2_exposure": c(["TotalNumberOfExposures", "ExposedArea_0", "ExposedArea_1", "EntranceDoseInmGy"]),
        "C3_frame_size": c(["Rows", "Columns"]),
        "C4_files_in_study": c(["n_files_study", "n_frames_region_study"]),
        "C5_row_order": c(["row_no", "file_no"]),
        "C6_uid_hash": c(["uid_hash_study", "uid_hash_sop"]),
    }


def prod_like_inputs(crit, df):
    """Признаки контура A и эмбеддинги контура B так, как их берёт поставка (src/train_stacked.py):
    вариант предобработки из config.yaml, колонки contour_a_cols (для sp_pos — с H2-признаком synth_pos_logit),
    источник эмбеддингов из embeddings.source_by_criterion. Строки выровнены с df по file_path."""
    import train_stacked as TS
    pre = TS.preproc_for(crit)
    region_rows = REGION_ROWS["spine" if crit.startswith("sp_") else "hip"]
    gv = pd.read_csv(ROOT / "data" / TS.GEOM_VARIANT_FILES[pre["geom"]])
    gv = gv[gv.region.isin(region_rows)].reset_index(drop=True)
    cols = TS.contour_a_cols(crit)
    missing = [c for c in cols if c not in gv.columns]
    if missing:  # H2-признак может лежать отдельно; тогда контур A без него (помечается в отчёте)
        cols = [c for c in cols if c in gv.columns]
    emb_labels = pd.read_csv(ROOT / "data" / "labels_for_embeddings.csv")
    eidx = np.nonzero(emb_labels["region"].isin(region_rows).values)[0]
    E_all, src, variant = TS.emb_matrix_for(crit, TS.load_embeddings_by_source())
    E = E_all[eidx]
    fp_e = emb_labels.loc[eidx, "file_path"].to_numpy()
    pos_e = pd.Series(np.arange(len(fp_e)), index=fp_e).loc[df.file_path].to_numpy()
    pos_g = pd.Series(np.arange(len(gv)), index=gv.file_path).loc[df.file_path].to_numpy()
    return (gv.iloc[pos_g][cols].to_numpy(float), E[pos_e],
            {"geom_variant": pre["geom"], "emb_source": src, "emb_variant": variant, "geom_cols": cols,
             "missing_cols": missing})


def stack_auc(y, groups, seed, inputs):
    """Контур A (логрегрессия C=1) + контур B (PCA 32, логрегрессия C=0.1) из train_final_models,
    среднее вероятностей (в поставке — ранговое среднее с весом 0.5; для AUC перестановки это не важно)."""
    from train_final_models import _impute, fit_geom, fit_emb
    Xg, E, info = inputs
    oof = np.zeros(len(y))
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(Xg, y, groups):
        med = np.nanmedian(Xg[tr], axis=0)
        dg = fit_geom(_impute(Xg[tr], med), y[tr], info["geom_cols"], med)
        de = fit_emb(E[tr], y[tr])
        pg = dg["clf"].predict_proba(dg["scaler"].transform(_impute(Xg[te], med)))[:, 1]
        pe = de["clf"].predict_proba(de["pca"].transform(de["scaler"].transform(E[te])))[:, 1]
        oof[te] = 0.5 * (pg + pe)
    return roc_auc_score(y, oof)


def main() -> None:
    rng = np.random.default_rng(SEED)
    ph = pd.read_csv(ROOT / "docs" / "k5" / "pixel_hashes.csv")
    rels = sorted(set(ph["file_path"].map(rel_path)))
    meta = read_meta(rels)
    out_dir = ROOT / "data" / "slices"; out_dir.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out_dir / "shortcut_meta_499.csv", index=False, float_format="%.6g")
    sw_levels = sorted(meta.SoftwareVersions.unique())

    import decision_curve as DC  # quality_prob OOF — эталон бинарного класса

    res = {"protocol": {"cv": "StratifiedGroupKFold(5), группы = исследование + хэш пикселей, "
                              f"{N_REPEAT} повторов разбиения", "model": "LogisticRegression C=1, balanced, StandardScaler",
                        "null": f"{N_NULL} перестановок меток между уникальными кадрами, 95-й перцентиль",
                        "software_versions": sw_levels,
                        "n_files": int(len(meta))}}
    for region, crits in REGIONS.items():
        df = build_region(region, meta)
        groups = groups_study_hash(df.study.to_numpy(), df.pixel_hash.to_numpy())
        units = pd.factorize(df.pixel_hash)[0]
        rr = {"n": int(len(df)), "n_studies": int(df.study.nunique()), "n_groups": int(len(np.unique(groups))),
              "groups_equal_studies": bool(len(np.unique(groups)) == df.study.nunique())}
        qp = DC.region_frame(region).set_index("file_path").loc[df.file_path]
        X_sets = control_sets(df, sw_levels)
        for target in ["binary"] + crits:
            y = df[f"y_{target}"].to_numpy().astype(int)
            prod = float(roc_auc_score(y, qp["prob"].to_numpy() if target == "binary" else df[f"s_{target}"]))
            t = {"n_pos": int(y.sum()), "production_oof_auc": prod, "controls": {}}
            for name, X in X_sets.items():
                aucs = [cv_auc(X, y, groups, s) for s in range(N_REPEAT)]
                null = [cv_auc(X, perm_between(y, units, rng), groups, 100 + k) for k in range(N_NULL)]
                t["controls"][name] = {"auc_mean": float(np.mean(aucs)), "auc_min": float(np.min(aucs)),
                                       "auc_max": float(np.max(aucs)), "null_mean": float(np.mean(null)),
                                       "null_p95": float(np.percentile(null, 95)),
                                       "above_null_p95": bool(np.mean(aucs) > np.percentile(null, 95))}
                # одномерный AUC каждого столбца (без обучения): AUC далеко от 0.5 в любую сторону — сигнал
                uni = {}
                for j, n in enumerate(control_colnames(sw_levels)[name]):
                    v = X[:, j]; ok = ~np.isnan(v)
                    uni[n] = float(roc_auc_score(y[ok], v[ok])) if len(np.unique(v[ok])) > 1 else 0.5
                t["controls"][name]["univariate_auc"] = uni
                t["controls"][name]["null_p05"] = float(np.percentile(null, 5))
                if t["controls"][name]["above_null_p95"]:
                    sc = qp["prob"].to_numpy() if target == "binary" else df[f"s_{target}"].to_numpy()
                    t["controls"][name]["drill_down"] = drill_down(X, control_colnames(sw_levels)[name], y, sc,
                                                                   df.study.to_numpy())
            if target != "binary":
                inputs = prod_like_inputs(target, df)
                real = [stack_auc(y, groups, s, inputs) for s in range(3)]
                pb = [stack_auc(perm_between(y, units, rng), groups, 200 + k, inputs) for k in range(N_PERM_MODEL)]
                pw = [stack_auc(perm_within(y, df.study.to_numpy(), rng), groups, 300 + k, inputs)
                      for k in range(N_PERM_MODEL)]
                t["stack_inputs"] = {k: v for k, v in inputs[2].items()}
                t["simplified_stack"] = {"real_auc_mean": float(np.mean(real)),
                                         "perm_between_groups_mean": float(np.mean(pb)),
                                         "perm_between_groups_p95": float(np.percentile(pb, 95)),
                                         "perm_between_groups_max": float(np.max(pb)),
                                         "perm_within_study_mean": float(np.mean(pw)),
                                         "perm_within_study_p95": float(np.percentile(pw, 95)),
                                         "labels_vary_within_study": int((df.groupby("study")[f"y_{target}"].nunique() > 1).sum())}
            if region == "hip" and target == "hip_roi":
                g = pd.read_csv(ROOT / "data" / "geometry_features.csv", usecols=["file_path", "scan_length_mm"])
                sl = g.set_index("file_path").loc[df.file_path, "scan_length_mm"].to_numpy()
                t["rows_vs_scan_length_mm_spearman"] = float(pd.Series(df.Rows.to_numpy()).corr(pd.Series(sl), method="spearman"))
            rr[target] = t
            print(region, target, json.dumps(t, ensure_ascii=False), flush=True)
        res[region] = rr
    (ROOT / "docs" / "shortcut_audit.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
