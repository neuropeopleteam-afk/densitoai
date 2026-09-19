#!/usr/bin/env python
"""
Отбор кадров для слепой ревизии рентгенолога (задача К6).

Выход (work/F/out/):
  review_key.csv        служебный ключ галереи (врач его НЕ видит)
  landmarks_key.csv     служебный ключ формы ориентиров
  pixel_hashes.csv      sha1 нормализованных пикселей всех 499 файлов
  frames_cache.npz      нормализованные изображения выбранных кадров (для сборки HTML)
  selection_summary.json сводка состава набора

Правила отбора:
  * уникальность — sha1(normalize_pixels(ds)) (как в inference.read_and_validate);
    из группы дубликатов остаётся первый по порядку geometry_features.csv;
  * 30 случайных уникальных кадров, стратифицированных по области:
    10 позвоночник / 20 бедро (пропорция как в данных: 166 / 333);
  * 20 «расхождений» — кадры, где OOF pred_label != y_true хотя бы по одному
    критерию (models/oof_stacked_<region>_<crit>.csv), не пересекающиеся со случайными;
    разнообразие: round-robin по критериям (sp_pos, sp_axis, sp_art, hip_pos, hip_roi);
  * 5 повторов (10 %) — из уже выбранных кадров под другими id, для intra-reader;
  * форма ориентиров — 40 уникальных кадров (20 позвоночник, 20 бедро): сначала
    из галереи, затем добавляются случайные уникальные.
Seed фиксирован: SEED = 20260919.
"""
import hashlib
import json
import os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

B = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(B / "src"))
from inference import normalize_pixels  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260919
N_RANDOM_SPINE, N_RANDOM_HIP = 10, 20
N_DISC = 20
N_REPEAT = 5
N_LM_SPINE, N_LM_HIP = 20, 20

CRITS = {
    "spine": ["sp_pos", "sp_axis", "sp_art"],
    "hip": ["hip_pos", "hip_roi"],
}
LABEL_COL = {"sp_pos": "sp_pos", "sp_axis": "sp_axis", "sp_art": "sp_art",
             "hip_pos": "hip_pos_c", "hip_roi": "hip_roi_c"}


def main():
    rng = np.random.default_rng(SEED)
    g = pd.read_csv(B / "data" / "geometry_features.csv")
    g["area"] = np.where(g.region == "spine", "spine", "hip")

    # ---- 1. хэши пикселей всех файлов -------------------------------------
    hashes, shapes, imgs = [], [], {}
    for i, fp in enumerate(g.file_path):
        ds = pydicom.dcmread(fp, force=True)
        img = normalize_pixels(ds)
        h = hashlib.sha1(np.ascontiguousarray(img).tobytes() + str(img.shape).encode()).hexdigest()
        hashes.append(h)
        shapes.append(img.shape)
        imgs[fp] = img
    g["pixel_sha1"] = hashes
    g["img_rows"] = [s[0] for s in shapes]
    g["img_cols"] = [s[1] for s in shapes]
    g[["file_path", "study", "region", "pixel_sha1", "img_rows", "img_cols"]].to_csv(
        OUT / "pixel_hashes.csv", index=False)
    dup_mask = g.duplicated("pixel_sha1", keep="first")
    n_dup = int(dup_mask.sum())
    n_groups = int((g.groupby("pixel_sha1").size() > 1).sum())
    uniq = g[~dup_mask].copy()
    print(f"файлов: {len(g)}, уникальных по пикселям: {len(uniq)}, дубликатов: {n_dup} "
          f"(групп с повторами: {n_groups})")

    # ---- 2. OOF-предсказания ------------------------------------------------
    oof = {}
    for area, crits in CRITS.items():
        for c in crits:
            d = pd.read_csv(B / "models" / f"oof_stacked_{area}_{c}.csv")
            oof[c] = d.set_index("file_path")[["y_true", "pred_label", "oof_stacked"]]
    for c in LABEL_COL:
        uniq[f"oof_{c}"] = uniq.file_path.map(oof[c]["pred_label"])
        uniq[f"oofscore_{c}"] = uniq.file_path.map(oof[c]["oof_stacked"])
        ytrue = uniq.file_path.map(oof[c]["y_true"])
        lab = uniq[LABEL_COL[c]]
        both = ytrue.notna() & lab.notna()
        assert (ytrue[both] == lab[both]).all(), f"y_true OOF != метка geometry_features для {c}"
        uniq[f"disc_{c}"] = (uniq[f"oof_{c}"].notna() & lab.notna() & (uniq[f"oof_{c}"] != lab))
    disc_cols = [f"disc_{c}" for c in LABEL_COL]
    uniq["any_disc"] = uniq[disc_cols].any(axis=1)
    uniq["n_disc"] = uniq[disc_cols].sum(axis=1)
    print("уникальных кадров с расхождением OOF:", int(uniq.any_disc.sum()),
          {c: int(uniq[f"disc_{c}"].sum()) for c in LABEL_COL})

    # ---- 3. случайные 30 (10 / 20) ----------------------------------------
    def sample(df, n):
        idx = rng.choice(df.index.to_numpy(), size=n, replace=False)
        return df.loc[sorted(idx)]

    rand_spine = sample(uniq[uniq.area == "spine"], N_RANDOM_SPINE)
    rand_hip = sample(uniq[uniq.area == "hip"], N_RANDOM_HIP)
    random_set = pd.concat([rand_spine, rand_hip])
    random_set = random_set.assign(is_discrepancy=False)

    # ---- 4. расхождения 20 (round-robin по критериям) ---------------------
    pool = uniq[uniq.any_disc & ~uniq.index.isin(random_set.index)]
    # чтобы не набрать половину кадров из одного исследования — не более 2 на study
    chosen, study_count = [], {}
    per_crit = {c: list(rng.permutation(pool[pool[f"disc_{c}"]].index.to_numpy())) for c in LABEL_COL}
    order = ["sp_pos", "sp_axis", "sp_art", "hip_pos", "hip_roi"]
    while len(chosen) < N_DISC and any(per_crit.values()):
        for c in order:
            while per_crit[c]:
                i = per_crit[c].pop(0)
                st = pool.loc[i, "study"]
                if i in chosen or study_count.get(st, 0) >= 2:
                    continue
                chosen.append(i)
                study_count[st] = study_count.get(st, 0) + 1
                break
            if len(chosen) >= N_DISC:
                break
    disc_set = pool.loc[chosen].assign(is_discrepancy=True)

    base = pd.concat([random_set, disc_set])
    assert base.pixel_sha1.is_unique
    base = base.assign(is_repeat=False, repeat_of_index=np.nan)

    # ---- 5. повторы 5 (10 %) ---------------------------------------------
    rep_idx = rng.choice(base.index.to_numpy(), size=N_REPEAT, replace=False)
    repeats = base.loc[rep_idx].copy()
    repeats["is_repeat"] = True
    repeats["repeat_of_index"] = rep_idx
    gallery = pd.concat([base, repeats])

    # перемешиваем порядок показа; повтор не должен идти сразу за оригиналом
    for _ in range(1000):
        perm = rng.permutation(len(gallery))
        show = gallery.iloc[perm].reset_index().rename(columns={"index": "src_index"})
        pos = {int(r.src_index): k for k, r in show[~show.is_repeat].iterrows()}
        ok = all(abs(k - pos[int(r.repeat_of_index)]) >= 8 for k, r in show[show.is_repeat].iterrows())
        if ok:
            break
    show["display_id"] = [f"R{k + 1:03d}" for k in range(len(show))]
    id_by_src = {int(r.src_index): r.display_id for _, r in show[~show.is_repeat].iterrows()}
    show["repeat_of_id"] = [id_by_src[int(r.repeat_of_index)] if r.is_repeat else "" for _, r in show.iterrows()]

    key_cols = ["display_id", "file_path", "study", "region", "area", "img_rows", "img_cols", "pixel_sha1",
                "sp_pos", "sp_axis", "sp_art", "hip_pos_c", "hip_roi_c",
                "oof_sp_pos", "oof_sp_axis", "oof_sp_art", "oof_hip_pos", "oof_hip_roi",
                "oofscore_sp_pos", "oofscore_sp_axis", "oofscore_sp_art", "oofscore_hip_pos", "oofscore_hip_roi",
                "disc_sp_pos", "disc_sp_axis", "disc_sp_art", "disc_hip_pos", "disc_hip_roi",
                "is_discrepancy", "is_repeat", "repeat_of_id"]
    key = show[key_cols].copy()
    key.to_csv(OUT / "review_key.csv", index=False)

    # ---- 6. форма ориентиров: 40 уникальных (20 / 20) ---------------------
    lm = []
    for area, n in (("spine", N_LM_SPINE), ("hip", N_LM_HIP)):
        in_gal = base[base.area == area]
        take = in_gal if len(in_gal) <= n else sample(in_gal, n)
        lm.append(take)
        need = n - len(take)
        if need > 0:
            extra_pool = uniq[(uniq.area == area) & ~uniq.index.isin(base.index)]
            lm.append(sample(extra_pool, need))
    lm = pd.concat(lm)
    assert lm.pixel_sha1.is_unique
    lm = lm.iloc[rng.permutation(len(lm))].reset_index().rename(columns={"index": "src_index"})
    lm["display_id"] = [f"L{k + 1:03d}" for k in range(len(lm))]
    lm["in_gallery"] = lm.src_index.isin(base.index)
    lm_key = lm[["display_id", "file_path", "study", "region", "area", "img_rows", "img_cols", "pixel_sha1",
                 "in_gallery"]]
    lm_key.to_csv(OUT / "landmarks_key.csv", index=False)

    # ---- 7. кэш изображений и сводка -------------------------------------
    need_fp = sorted(set(key.file_path) | set(lm_key.file_path))
    np.savez_compressed(OUT / "frames_cache.npz", **{hashlib.sha1(fp.encode()).hexdigest(): imgs[fp] for fp in need_fp},
                        __paths__=np.array(need_fp))

    summary = {
        "seed": SEED,
        "n_files": int(len(g)),
        "n_unique_pixels": int(len(uniq)),
        "n_duplicates": n_dup,
        "n_duplicate_groups": n_groups,
        "n_unique_with_oof_discrepancy": int(uniq.any_disc.sum()),
        "gallery": {
            "n_shown": int(len(key)),
            "n_unique": int(len(base)),
            "random": {"spine": int((random_set.area == "spine").sum()), "hip": int((random_set.area == "hip").sum())},
            "discrepancy": {"spine": int((disc_set.area == "spine").sum()), "hip": int((disc_set.area == "hip").sum())},
            "discrepancy_by_crit": {c: int(disc_set[f"disc_{c}"].sum()) for c in LABEL_COL},
            "repeats": int(N_REPEAT),
            "n_studies": int(base.study.nunique()),
            "positives_by_crit_labels": {c: int(base[LABEL_COL[c]].fillna(0).sum()) for c in LABEL_COL},
            "positives_by_crit_oof": {c: int(base[f"oof_{c}"].fillna(0).sum()) for c in LABEL_COL},
        },
        "landmarks": {
            "n": int(len(lm_key)),
            "spine": int((lm_key.area == "spine").sum()), "hip": int((lm_key.area == "hip").sum()),
            "in_gallery": int(lm_key.in_gallery.sum()),
            "regions": lm_key.region.value_counts().to_dict(),
        },
    }
    (OUT / "selection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
