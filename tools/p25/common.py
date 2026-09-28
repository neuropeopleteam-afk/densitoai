"""Общие загрузчики для оценок 2.5 (tools/p25/*): OOF-флаги сервиса, 40 кадров слепой проверки, ответы врачей."""
import json
import pickle
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DOCS_RAW = Path("/home/user/workspace/council_2709/doctors_raw")
# Учитываемые ответы (docs/REVIEW_DOCTORS.md): второй проход с компьютера врача 2 и повторная отправка сессии не в счёт.
DOCTORS = {"20260924_153814_931472.json": "врач 1", "20260924_162809_d1a12e.json": "врач 2",
           "20260924_191650_fc47e8.json": "врач 3"}
EXCLUDED = ("20260924_195838_76011c.json", "20260924_163153_25949f.json")
CRITS = {"spine": ["sp_pos", "sp_axis", "sp_art"], "hip": ["hip_pos", "hip_roi"]}


def load_oof():
    """file_path -> {area, study, flags, labels, uncertain, scores, p_geom, p_emb} по OOF текущих моделей
    (models/oof_stacked_*.csv; бедро — объединённые файлы hip_hip_*), зона «не уверен» — models/calibration.pkl."""
    cal = pickle.load(open(ROOT / "models" / "calibration.pkl", "rb"))["criteria"]
    rows = {}
    for area, crits in CRITS.items():
        for c in crits:
            o = pd.read_csv(ROOT / "models" / f"oof_stacked_{area}_{c}.csv")
            thr, m = cal[c]["threshold"], cal[c]["margin"]
            for r in o.itertuples():
                d = rows.setdefault(r.file_path, {"area": area, "study": r.study, "flags": {}, "labels": {},
                                                  "uncertain": False, "p_geom": {}, "p_emb": {}})
                d["flags"][c] = int(r.pred_label) == 1
                d["labels"][c] = int(r.y_true) == 1
                d["p_geom"][c] = float(r.oof_geom); d["p_emb"][c] = float(r.oof_emb)
                d["uncertain"] |= abs(float(r.oof_stacked) - thr) <= m
    return rows


def load_frames40(oof=None, check_page=None):
    """40 кадров слепой проверки: file -> {show, area, file_path, labels (по критериям), flags (OOF текущих моделей)}.
    Кадр сопоставляется с файлом по pixel_sha1 манифеста (outputs/p25/frames.csv, tools/p25/prep_frames.py).
    check_page — страница проверки с DATA.service; если задана, флаги OOF сверяются с ней (должны совпасть)."""
    oof = oof or load_oof()
    man = json.loads((ROOT / "tools" / "review" / "kit_light_manifest.json").read_text(encoding="utf-8"))
    order = {s["file"]: s["idx"] for s in man["shows_full"] if not s["is_repeat"]}
    fr = pd.read_csv(ROOT / "outputs" / "p25" / "frames.csv")
    by_sha = dict(zip(fr.pixel_sha1, fr.file_path))
    out = {}
    for f in man["frames"]:
        fp = by_sha[f["pixel_sha1"]]
        o = oof[fp]
        out[f["file"]] = {"show": order[f["file"]], "area": o["area"], "file_path": fp,
                          "labels": {k: int(v) == 1 for k, v in f["labels"].items()},
                          "flags": dict(o["flags"]), "uncertain": o["uncertain"]}
    if check_page:
        html = Path(check_page).read_text(encoding="utf-8")
        svc = json.loads(re.search(r"^const DATA = (.+);$", html, re.M).group(1))["service"]
        for fn, d in out.items():
            page_flags = {re.sub(r"^(rh|lh)_", "hip_", c["code"]): bool(c.get("flag")) for c in svc[fn]["criteria"]}
            assert page_flags == d["flags"], (d["show"], page_flags, d["flags"])
    return out


def load_first_answers():
    """врач -> {file: первый ответ (ok / bad / unsure)} по учитываемым ответам."""
    res = {}
    for fn, name in DOCTORS.items():
        d = json.loads((DOCS_RAW / fn).read_text(encoding="utf-8"))
        first = {}
        for a in d["answers"]:
            first.setdefault(a["file"], a["verdict"])
        res[name] = first
    return res
