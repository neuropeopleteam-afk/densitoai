#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Отбор облегчённого набора слепой ревизии (44 показа, одна страница).

Механическое правило с фиксированным зерном SEED = 20260922, без ручного вмешательства:
  * источник меток — data/labels_full.csv (метки организаторов);
    hip_pos = rh_pos или lh_pos, hip_roi = rh_roi или lh_roi (по области снимка);
  * уникальность кадров — sha1 нормализованных пикселей (inference.normalize_pixels),
    из группы дубликатов остаётся первый по порядку labels_full.csv;
  * 20 кадров с нарушением: по 4 на каждый критерий в порядке
    sp_pos, sp_axis, sp_art, hip_pos, hip_roi; внутри критерия сначала кадры,
    где положителен только этот критерий (чтобы пример был чистым),
    не более 2 кадров на одно исследование;
  * 20 нормальных кадров: 10 позвоночник (sp_pos = sp_axis = sp_art = 0),
    10 бедро (оба критерия бедра = 0), не более 2 на исследование;
  * 4 скрытых повтора — те же кадры под другим номером показа, расстояние
    от оригинала не менее 10 показов, повтор всегда позже оригинала; итого 44 показа;
  * короткий режим (?short=1): 24 показа — 11 уникальных с нарушением
    (sp_pos 2, sp_axis 2, sp_art 3, hip_pos 2, hip_roi 2), 11 уникальных нормальных
    (6 позвоночник, 5 бедро) и 2 повтора внутри этих 24 (один с нарушением, один нормальный),
    то есть 12 показов с нарушением и 12 нормальных.

Вердикты сервиса берутся из уже посчитанных OOF-файлов models/oof_stacked_*.csv
(инференс не запускается), измеренные величины — из data/geometry_features.csv.

Выход:
  tools/review/kit_light_manifest.json      список показов и его sha256
  tools/review/out/light/<имя>.png          кадры без наложений и подписей
  tools/review/out/light/payload_light.json данные для страницы (кадры + вердикты)
  tools/review/out/light/pixel_hashes_light.csv  sha1 пикселей всех файлов выборки

Запуск:  python tools/review/select_light.py [--no-frames]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

B = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(B / "src"))

SEED = 20260922
OUT = Path(__file__).resolve().parent / "out" / "light"
MANIFEST = Path(__file__).resolve().parent / "kit_light_manifest.json"

# корень датасета: путь в CSV снят на машине разметки, здесь он перебазируется
DATA_ROOTS = [
    "/opt/neuropeople/external_datasets/own_dataset",
    "/opt/neuropeople/densito_full_test/train_data",
]
PATH_MARK = "Исследования/"

CRITS = ["sp_pos", "sp_axis", "sp_art", "hip_pos", "hip_roi"]
N_PER_CRIT = 4
N_NORM_SPINE, N_NORM_HIP = 10, 10
N_REPEAT = 4
MIN_REPEAT_GAP = 10
SHORT_VIOL = {"sp_pos": 2, "sp_axis": 2, "sp_art": 3, "hip_pos": 2, "hip_roi": 2}
SHORT_NORM_SPINE, SHORT_NORM_HIP = 6, 5

# закрытый перечень официальных строк нарушений (config.yaml: violations) — не менять
VIOLATION_STR = {
    "sp_pos": "Некорректная укладка",
    "sp_axis": "Не выравнена ось позвоночника",
    "sp_art": "Присутствуют посторонние предметы",
    "hip_pos": "Некорректная укладка",
    "hip_roi": "Некорректная область интереса",
}
OOF_FILE = {
    "sp_pos": "oof_stacked_spine_sp_pos.csv",
    "sp_axis": "oof_stacked_spine_sp_axis.csv",
    "sp_art": "oof_stacked_spine_sp_art.csv",
    "hip_pos": "oof_stacked_hip_hip_pos.csv",
    "hip_roi": "oof_stacked_hip_hip_roi.csv",
}
# измеряемые величины и их названия — как в src/dicom_sr.py
MEASURES = {
    "spine": [
        ("axis_angle_deg", "Угол оси позвоночника к вертикали кадра", "град"),
        ("curvature", "Показатель кривизны центральной линии", ""),
        ("metal_metal_outside_bone_mm2", "Площадь посторонних объектов вне кости", "мм²"),
    ],
    "hip": [
        ("abs_shaft_angle_deg", "Угол диафиза бедренной кости к вертикали", "град"),
        ("lateral_margin_mm", "Отступ ROI от латерального края кадра", "мм"),
        ("shaft_len_below_troch_mm", "Длина диафиза ниже малого вертела в кадре", "мм"),
        ("merge_height_mm", "Высота слияния диафиза с тазом", "мм"),
    ],
}
CRIT_RU = {
    "sp_pos": "укладка позвоночника",
    "sp_axis": "ось позвоночника",
    "sp_art": "посторонние предметы",
    "hip_pos": "укладка бедра",
    "hip_roi": "область интереса бедра",
}


def rebase(path: str) -> str:
    """Перебазирует путь из CSV на локальный корень датасета."""
    if os.path.exists(path):
        return path
    i = path.find(PATH_MARK)
    tail = path[i:] if i >= 0 else os.path.basename(path)
    for root in DATA_ROOTS:
        cand = os.path.join(root, tail)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(path)


def f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def read_csv_rows(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def area_of(region: str) -> str:
    return "spine" if region == "spine" else "hip"


def labels_of(row: dict) -> dict:
    """Метки организаторов по пяти критериям для конкретного снимка."""
    area = area_of(row["region"])
    if area == "spine":
        return {c: f(row.get(c)) for c in ("sp_pos", "sp_axis", "sp_art")}
    side = "rh" if row["region"] == "right_hip" else "lh"
    return {"hip_pos": f(row.get(side + "_pos")), "hip_roi": f(row.get(side + "_roi"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-frames", action="store_true", help="не перерисовывать PNG кадров")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    rows = read_csv_rows(B / "data" / "labels_full.csv")
    geom = {r["file_path"]: r for r in read_csv_rows(B / "data" / "geometry_features.csv")}
    oof = {}
    for c, fn in OOF_FILE.items():
        oof[c] = {r["file_path"]: r for r in read_csv_rows(B / "models" / fn)}
    thr = {}
    ms = json.loads((B / "models" / "metrics_summary.json").read_text(encoding="utf-8"))
    for c in CRITS:
        block = ms.get("spine" if c.startswith("sp_") else "hip", {}).get(c, {})
        thr[c] = f(block.get("threshold"))

    # ---- уникальность по пикселям ------------------------------------------
    from inference import normalize_pixels  # noqa: E402
    import pydicom  # noqa: E402

    seen, uniq = set(), []
    hash_rows = []
    for r in rows:
        local = rebase(r["file_path"])
        ds = pydicom.dcmread(local, force=True)
        img = normalize_pixels(ds)
        h = hashlib.sha1(np.ascontiguousarray(img).tobytes() + str(img.shape).encode()).hexdigest()
        hash_rows.append({"study": r["study"], "region": r["region"], "pixel_sha1": h,
                          "img_rows": img.shape[0], "img_cols": img.shape[1]})
        if h in seen:
            continue
        seen.add(h)
        rec = dict(r)
        rec["local_path"] = local
        rec["pixel_sha1"] = h
        rec["area"] = area_of(r["region"])
        rec["labels"] = labels_of(r)
        rec["frame"] = "f" + h[:12]
        rec["shape"] = (int(img.shape[0]), int(img.shape[1]))
        rec["_img"] = img
        uniq.append(rec)
    with open(OUT / "pixel_hashes_light.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["study", "region", "pixel_sha1", "img_rows", "img_cols"])
        w.writeheader()
        w.writerows(hash_rows)
    print(f"файлов: {len(rows)}, уникальных по пикселям: {len(uniq)}")

    by_frame = {r["frame"]: r for r in uniq}
    chosen, study_count = [], {}

    def take(pool, n, tag):
        got = []
        for rec in pool:
            if len(got) >= n:
                break
            if rec["frame"] in chosen:
                continue
            if study_count.get(rec["study"], 0) >= 2:
                continue
            chosen.append(rec["frame"])
            study_count[rec["study"]] = study_count.get(rec["study"], 0) + 1
            rec["group"] = tag
            got.append(rec["frame"])
        if len(got) < n:
            raise RuntimeError(f"не хватило кадров для {tag}: {len(got)} из {n}")
        return got

    # ---- 20 кадров с нарушением (по 4 на критерий) -------------------------
    viol_by_crit = {}
    for c in CRITS:
        pool = [r for r in uniq if r["labels"].get(c) == 1.0]
        order = list(rng.permutation(len(pool)))
        pool = [pool[i] for i in order]
        # сначала кадры, где положителен только этот критерий
        pool.sort(key=lambda r: sum(1 for k, v in r["labels"].items() if k != c and v == 1.0))
        viol_by_crit[c] = take(pool, N_PER_CRIT, "viol:" + c)

    # ---- 20 нормальных кадров ---------------------------------------------
    def norm_pool(area):
        pool = [r for r in uniq if r["area"] == area
                and all(v == 0.0 for v in r["labels"].values())]
        order = list(rng.permutation(len(pool)))
        return [pool[i] for i in order]

    norm_spine = take(norm_pool("spine"), N_NORM_SPINE, "norm:spine")
    norm_hip = take(norm_pool("hip"), N_NORM_HIP, "norm:hip")

    unique_frames = list(chosen)
    assert len(unique_frames) == 40 and len(set(unique_frames)) == 40

    # ---- вердикты сервиса из OOF ------------------------------------------
    def service(rec):
        area = rec["area"]
        crits = ["sp_pos", "sp_axis", "sp_art"] if area == "spine" else ["hip_pos", "hip_roi"]
        fp = rec["file_path"]
        items, flags = [], []
        for c in crits:
            o = oof[c].get(fp)
            if not o:
                continue
            pred = f(o.get("pred_label"))
            score = f(o.get("oof_stacked"))
            if pred == 1.0:
                flags.append(c)
            items.append({"code": c, "name": CRIT_RU[c], "flag": pred == 1.0,
                          "score": None if score is None else round(score, 3),
                          "threshold": None if thr[c] is None else round(thr[c], 3)})
        g = geom.get(fp, {})
        meas = []
        for key, name, unit in MEASURES[area]:
            v = f(g.get(key))
            if v is not None:
                meas.append({"name": name, "value": round(v, 2), "unit": unit})
        return {
            "region": "Поясничный отдел позвоночника" if area == "spine" else "Проксимальный отдел бедра",
            "quality_class": 1 if flags else 0,
            "verdict": "Нарушение" if flags else "Норма",
            "violation_type": "; ".join(VIOLATION_STR[c] for c in flags) if flags else "нет",
            "criteria": items,
            "measures": meas,
        }

    # ---- последовательность показов: 40 уникальных + 4 повтора ------------
    def build_sequence(frames, n_repeat):
        frames = list(frames)
        rep_pos = sorted(rng.choice(len(frames), size=n_repeat, replace=False).tolist())
        repeats = [frames[i] for i in rep_pos]
        for _ in range(20000):
            order = [frames[i] for i in rng.permutation(len(frames)).tolist()]
            total = len(order) + n_repeat
            # места для повторов во второй половине последовательности
            slots = sorted(rng.choice(range(total // 2, total), size=n_repeat, replace=False).tolist())
            seq, ri, oi = [], 0, 0
            for k in range(total):
                if ri < n_repeat and k == slots[ri]:
                    seq.append((repeats[ri], True))
                    ri += 1
                else:
                    seq.append((order[oi], False))
                    oi += 1
            first = {}
            ok = True
            for k, (fr, is_rep) in enumerate(seq):
                if is_rep:
                    if fr not in first or k - first[fr] < MIN_REPEAT_GAP:
                        ok = False
                        break
                else:
                    if fr in first:
                        ok = False
                        break
                    first[fr] = k
            if ok:
                return seq, repeats
        raise RuntimeError("не удалось разложить повторы")

    seq_full, rep_full = build_sequence(unique_frames, N_REPEAT)

    # короткий режим: 11 с нарушением + 11 нормальных + 2 повтора = 24 показа
    short_viol = [fr for c in CRITS for fr in viol_by_crit[c][: SHORT_VIOL[c]]]
    short_norm = norm_spine[:SHORT_NORM_SPINE] + norm_hip[:SHORT_NORM_HIP]
    short_unique = short_viol + short_norm
    assert len(short_viol) == 11 and len(short_norm) == 11
    seq_short, rep_short = build_sequence(short_unique, 2)
    # один повтор с нарушением, один нормальный — иначе пересобираем с тем же зерном
    for _ in range(200):
        n_v = sum(1 for fr in rep_short if fr in short_viol)
        if n_v == 1:
            break
        seq_short, rep_short = build_sequence(short_unique, 2)
    else:
        raise RuntimeError("не удалось подобрать повторы короткого режима")

    def show_list(seq):
        return [{"idx": k + 1, "frame": by_frame[fr]["frame"], "file": by_frame[fr]["frame"] + ".png",
                 "is_repeat": bool(is_rep)} for k, (fr, is_rep) in enumerate(seq)]

    shows_full, shows_short = show_list(seq_full), show_list(seq_short)

    # ---- sha256 списка (фиксируется до начала ревизии) ---------------------
    kit_list = {
        "seed": SEED,
        "unique_frames": [by_frame[fr]["frame"] for fr in unique_frames],
        "full": [[s["idx"], s["file"], s["is_repeat"]] for s in shows_full],
        "short": [[s["idx"], s["file"], s["is_repeat"]] for s in shows_short],
    }
    kit_json = json.dumps(kit_list, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    kit_sha = hashlib.sha256(kit_json.encode("utf-8")).hexdigest()

    composition = {
        "violation_by_criterion": {c: len(viol_by_crit[c]) for c in CRITS},
        "normal_spine": len(norm_spine),
        "normal_hip": len(norm_hip),
        "unique_frames": len(unique_frames),
        "shows_full": len(shows_full),
        "repeats_full": N_REPEAT,
        "shows_short": len(shows_short),
        "repeats_short": 2,
        "studies": len(study_count),
    }
    manifest = {
        "kit": "light",
        "service_version": "2.3.2",
        "seed": SEED,
        "kit_sha256": kit_sha,
        "kit_list_serialization": "json, sort_keys=True, separators=(',',':'), ensure_ascii=False",
        "composition": composition,
        "frames": [
            {
                "frame": by_frame[fr]["frame"],
                "file": by_frame[fr]["frame"] + ".png",
                "area": by_frame[fr]["area"],
                "region": by_frame[fr]["region"],
                "group": by_frame[fr]["group"],
                "pixel_sha1": by_frame[fr]["pixel_sha1"],
                "labels": {k: (None if v is None else int(v)) for k, v in by_frame[fr]["labels"].items()},
            }
            for fr in unique_frames
        ],
        "shows_full": shows_full,
        "shows_short": shows_short,
        "repeat_frames_full": [by_frame[fr]["frame"] for fr in rep_full],
        "repeat_frames_short": [by_frame[fr]["frame"] for fr in rep_short],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- данные для страницы (вердикт сервиса раскрывается после ответа) --
    payload = {
        "kit_sha256": kit_sha,
        "service_version": "2.3.2",
        "shows": [{"idx": s["idx"], "file": s["file"]} for s in shows_full],
        "shows_short": [{"idx": s["idx"], "file": s["file"]} for s in shows_short],
        "service": {by_frame[fr]["frame"] + ".png": service(by_frame[fr]) for fr in unique_frames},
        "sizes": {by_frame[fr]["frame"] + ".png": list(by_frame[fr]["shape"]) for fr in unique_frames},
    }
    (OUT / "payload_light.json").write_text(json.dumps(payload, ensure_ascii=False,
                                                       separators=(",", ":")), encoding="utf-8")

    # ---- PNG кадров (без наложений и подписей) ----------------------------
    if not args.no_frames:
        import cv2  # noqa: E402
        for fr in unique_frames:
            rec = by_frame[fr]
            ok = cv2.imwrite(str(OUT / (rec["frame"] + ".png")), rec["_img"],
                             [cv2.IMWRITE_PNG_COMPRESSION, 9])
            assert ok, rec["frame"]

    print(json.dumps({"kit_sha256": kit_sha, "composition": composition}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
