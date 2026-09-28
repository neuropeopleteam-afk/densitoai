#!/usr/bin/env python3
"""Эксперимент 2.5 (sp_art, положение предметов): регрессия на 499 и влияние на 40 кадров слепой проверки.

Только OOF-предсказания (models/oof_stacked_*.csv): «до» — копия поставки в outputs/spart_position/before/,
«после» — models/ копии после src/train_stacked.py с новыми признаками. Сервисные прогоны (финальные модели,
обучены на всех 499, т.е. на своих же кадрах) сравниваются отдельно, справочно.

Выход: outputs/spart_position/regression_review.json, outputs/spart_position/review_page_new.html
(копия web/review/index.html, в которой DATA.service заменён на OOF новой версии — вход для
tools/review/compute_light_agreement.py --page).

Запуск (из корня копии B):  python tools/spart_position/regression_review.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "spart_position"
BEFORE = OUT / "before"
RUNS = ROOT.parent / "runs"
SPINE = ["sp_pos", "sp_axis", "sp_art"]
NAMES = {"sp_pos": "Некорректная укладка", "sp_axis": "Не выравнена ось позвоночника",
         "sp_art": "Присутствуют посторонние предметы"}


def load_oof(d: Path, crit: str) -> pd.DataFrame:
    df = pd.read_csv(d / f"oof_stacked_spine_{crit}.csv")
    return df.set_index("file_path")


def thr(d: Path, crit: str) -> float:
    return float(json.loads((d / "metrics_summary.json").read_text(encoding="utf-8"))["spine"][crit]["threshold"])


def margin(d: Path, crit: str) -> float:
    return float(json.loads((d / "metrics_summary.json").read_text(encoding="utf-8"))["spine"][crit]["margin"])


def metrics(y, s, yhat):
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    return {"auc": float(roc_auc_score(y, s)) if s is not None else None, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sens": tp / max(tp + fn, 1), "spec": tn / max(tn + fp, 1), "f1_pos": float(f1_score(y, yhat)),
            "macro_f1": float(f1_score(y, yhat, average="macro")), "flagged": int(yhat.sum())}


def main():
    res = {}
    old = {c: load_oof(BEFORE, c) for c in SPINE}
    new = {c: load_oof(ROOT / "models", c) for c in SPINE}
    idx = old["sp_art"].index
    for c in SPINE:
        assert (new[c].index == idx).all() and (old[c].index == idx).all()
    # 1. sp_art на OOF
    y = old["sp_art"]["y_true"].values.astype(int)
    so, sn = old["sp_art"]["oof_stacked"].values, new["sp_art"]["oof_stacked"].values
    po, pn = old["sp_art"]["pred_label"].values.astype(int), new["sp_art"]["pred_label"].values.astype(int)
    to, tn_ = thr(BEFORE, "sp_art"), thr(ROOT / "models", "sp_art")
    assert ((so >= to).astype(int) == po).all() and ((sn >= tn_).astype(int) == pn).all(), "pred_label != score>=thr"
    res["sp_art_oof"] = {"n": int(len(y)), "n_pos": int(y.sum()), "thr_old": to, "thr_new": tn_,
                         "old": metrics(y, so, po), "new": metrics(y, sn, pn),
                         "auc_geom_old": float(roc_auc_score(y, old["sp_art"]["oof_geom"])),
                         "auc_geom_new": float(roc_auc_score(y, new["sp_art"]["oof_geom"])),
                         "changed": int((po != pn).sum()), "0->1": int(((po == 0) & (pn == 1)).sum()),
                         "1->0": int(((po == 1) & (pn == 0)).sum()),
                         "changed_fixed_to_correct": int(((po != pn) & (pn == y)).sum()),
                         "changed_to_wrong": int(((po != pn) & (pn != y)).sum())}
    # «не уверен» по sp_art
    mo, mn = margin(BEFORE, "sp_art"), margin(ROOT / "models", "sp_art")
    uo, un = np.abs(so - to) <= mo, np.abs(sn - tn_) <= mn
    res["sp_art_oof"]["uncertain_old"], res["sp_art_oof"]["uncertain_new"] = int(uo.sum()), int(un.sum())
    # остальные критерии позвоночника не меняются
    for c in ("sp_pos", "sp_axis"):
        res[f"{c}_oof_identical"] = bool(np.allclose(old[c]["oof_stacked"], new[c]["oof_stacked"], atol=1e-12)
                                         and (old[c]["pred_label"] == new[c]["pred_label"]).all())
    # 2. «позвоночник: есть нарушение» = OR флагов критериев (как quality_class в сервисе)
    lab_any = np.maximum.reduce([old[c]["y_true"].values.astype(int) for c in SPINE])
    cls_old = np.maximum.reduce([old[c]["pred_label"].values.astype(int) for c in SPINE])
    cls_new = np.maximum.reduce([new[c]["pred_label"].values.astype(int) for c in SPINE])
    res["spine_class_oof"] = {"n": int(len(cls_old)), "label_pos": int(lab_any.sum()),
                              "old": metrics(lab_any, None, cls_old), "new": metrics(lab_any, None, cls_new),
                              "changed": int((cls_old != cls_new).sum()),
                              "0->1": int(((cls_old == 0) & (cls_new == 1)).sum()),
                              "1->0": int(((cls_old == 1) & (cls_new == 0)).sum()),
                              "changed_to_correct": int(((cls_old != cls_new) & (cls_new == lab_any)).sum())}
    # hip-файлы: OOF бедра не меняются
    hip_same = True
    for f in BEFORE.glob("oof_stacked_*hip*.csv"):
        a, b = pd.read_csv(f), pd.read_csv(ROOT / "models" / f.name)
        num = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c])]
        hip_same &= bool(np.allclose(a[num].fillna(-9).values, b[num].fillna(-9).values, atol=1e-12))
    res["hip_oof_identical"] = hip_same

    # 3. сервисные прогоны (финальные модели, справочно)
    svc = {}
    try:
        b = pd.concat([pd.read_csv(RUNS / f"base_{h}.csv") for h in ("h1", "h2")], ignore_index=True)
        n = pd.concat([pd.read_csv(RUNS / f"new2_{h}.csv") for h in ("h1", "h2")], ignore_index=True)
        bd = pd.concat([pd.read_csv(RUNS / f"base_{h}_debug.csv") for h in ("h1", "h2")], ignore_index=True)
        nd = pd.concat([pd.read_csv(RUNS / f"new2_{h}_debug.csv") for h in ("h1", "h2")], ignore_index=True)
        assert list(b.columns) == list(n.columns) and len(b) == len(n) == 499
        assert (b.image_uid.values == n.image_uid.values).all()
        sp = (b.anatomical_region.astype(str).str.contains("позвоночник", case=False)).values
        svc = {"rows": int(len(b)), "columns": list(b.columns), "n_columns": int(len(b.columns)),
               "spine_rows": int(sp.sum()),
               "quality_class_changed": int((b.quality_class != n.quality_class).sum()),
               "quality_class_changed_spine": int(((b.quality_class != n.quality_class) & sp).sum()),
               "quality_class_changed_hip": int(((b.quality_class != n.quality_class) & ~sp).sum()),
               "violation_type_changed": int((b.violation_type.fillna("") != n.violation_type.fillna("")).sum()),
               "quality_prob_changed_hip": int(((np.abs(b.quality_prob - n.quality_prob) > 1e-9) & ~sp).sum()),
               "quality_prob_changed_spine": int(((np.abs(b.quality_prob - n.quality_prob) > 1e-9) & sp).sum()),
               "other_cols_identical": {c: bool((b[c].astype(str) == n[c].astype(str)).all())
                                        for c in b.columns if c not in ("quality_class", "violation_type", "quality_prob")}}
        fo, fn_ = bd["sp_art_flag"], nd["sp_art_flag"]
        m = fo.notna() | fn_.notna()
        svc["sp_art_flag_changed"] = int((fo[m].astype(float) != fn_[m].astype(float)).sum())
        svc["sp_art_flag_old"], svc["sp_art_flag_new"] = int(fo[m].astype(float).sum()), int(fn_[m].astype(float).sum())
        for c in ("sp_pos", "sp_axis"):
            svc[f"{c}_flag_changed"] = int((bd[f"{c}_flag"][m].astype(float) != nd[f"{c}_flag"][m].astype(float)).sum())
        svc["sp_art_flag_0to1"] = int(((fo[m].astype(float) == 0) & (fn_[m].astype(float) == 1)).sum())
        svc["sp_art_flag_1to0"] = int(((fo[m].astype(float) == 1) & (fn_[m].astype(float) == 0)).sum())
        svc["class_0to1"] = int(((b.quality_class == 0) & (n.quality_class == 1)).sum())
        svc["class_1to0"] = int(((b.quality_class == 1) & (n.quality_class == 0)).sum())
        svc["spine_class1_old"], svc["spine_class1_new"] = int(b.quality_class[sp].sum()), int(n.quality_class[sp].sum())
        dq = (n.quality_prob - b.quality_prob)[sp].abs()
        svc["spine_quality_prob_absdiff_median"], svc["spine_quality_prob_absdiff_max"] = float(dq.median()), float(dq.max())
        svc["sp_art_uncertain_old"] = int(bd["sp_art_uncertain"][m].astype(float).sum())
        svc["sp_art_uncertain_new"] = int(nd["sp_art_uncertain"][m].astype(float).sum())
    except FileNotFoundError as e:  # noqa: F841
        svc = {"error": str(e)}
    res["service_runs"] = svc

    # 4. 40 кадров слепой проверки
    man = json.loads((ROOT / "tools" / "review" / "kit_light_manifest.json").read_text(encoding="utf-8"))
    ph = pd.read_csv(OUT.parent / "pixel_hashes.csv")
    html_p = ROOT / "web" / "review" / "index.html"
    html = html_p.read_text(encoding="utf-8")
    mm = re.search(r"^const DATA = (.+);$", html, re.M)
    data = json.loads(mm.group(1))
    service = data["service"]
    rows = []
    for fr in man["frames"]:
        if fr["area"] != "spine":
            continue
        fps = ph.loc[ph.pixel_hash == fr["pixel_sha1"], "file_path"].tolist()
        fps = [p for p in fps if p in idx]
        assert fps, fr["frame"]
        sv = service[fr["file"]]
        crit = {c["code"]: c for c in sv["criteria"]}
        r = {"frame": fr["frame"], "file": fr["file"], "group": fr["group"], "labels": fr["labels"],
             "n_files": len(fps)}
        for c in SPINE:
            # дубликаты по пикселям: набор брал первый файл группы (select_frames.py, keep="first")
            vo = old[c].loc[fps[:1], "oof_stacked"].values; vn = new[c].loc[fps[:1], "oof_stacked"].values
            r.setdefault("dup_spread_old", {})[c] = float(np.ptp(old[c].loc[fps, "oof_stacked"].values))
            assert abs(round(float(vo[0]), 3) - crit[c]["score"]) < 1.5e-3, (fr["frame"], c, vo[0], crit[c]["score"])
            r[f"{c}_old"], r[f"{c}_new"] = float(vo[0]), float(vn[0])
            r[f"{c}_flag_old"] = int(old[c].loc[fps[0], "pred_label"]); r[f"{c}_flag_new"] = int(new[c].loc[fps[0], "pred_label"])
            assert r[f"{c}_flag_old"] == int(bool(crit[c]["flag"])), (fr["frame"], c)
        r["class_old"] = int(sv["quality_class"])
        r["class_new"] = max(r[f"{c}_flag_new"] for c in SPINE)
        assert r["class_old"] == max(r[f"{c}_flag_old"] for c in SPINE)
        rows.append(r)
        # новая страница: DATA.service с OOF новой версии (только sp_art меняется)
        for c in sv["criteria"]:
            if c["code"] == "sp_art":
                c["score"], c["threshold"], c["flag"] = round(r["sp_art_new"], 3), round(tn_, 3), bool(r["sp_art_flag_new"])
        sv["quality_class"] = r["class_new"]
        flagged = [c["code"] for c in sv["criteria"] if c["flag"]]
        sv["verdict"] = "Нарушение" if flagged else "Норма"
        sv["violation_type"] = NAMES[flagged[0]] if flagged else ""
    order = {s["file"]: s["idx"] for s in man["shows_full"] if not s["is_repeat"]}
    for r in rows:
        r["show"] = order.get(r["file"])
    res["review_spine_frames"] = rows
    res["review_changed"] = [{"show": r["show"], "frame": r["frame"], "group": r["group"],
                              "sp_art_label": r["labels"]["sp_art"], "sp_art_old": round(r["sp_art_old"], 3),
                              "sp_art_new": round(r["sp_art_new"], 3), "flag_old": r["sp_art_flag_old"],
                              "flag_new": r["sp_art_flag_new"], "class_old": r["class_old"], "class_new": r["class_new"]}
                             for r in rows if r["sp_art_flag_old"] != r["sp_art_flag_new"] or r["class_old"] != r["class_new"]]
    new_html = html[:mm.start(1)] + json.dumps(data, ensure_ascii=False) + html[mm.end(1):]
    (OUT / "review_page_new.html").write_text(new_html, encoding="utf-8")
    (OUT / "regression_review.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "review_spine_frames"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
