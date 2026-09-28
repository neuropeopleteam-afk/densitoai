#!/usr/bin/env python3
"""2.5, пункт 1 («команда только с доказательством», консилиум 27.09, GPT 6 Sol, решение A): что правила
src/action_evidence.py делают на OOF 499 и на 40 кадрах слепой проверки врачей.

Модель не переобучается: флаги — pred_label из models/oof_stacked_*.csv (OOF текущих моделей, для 2.5.0 — новый
sp_art), зона «не уверен» — models/calibration.pkl. Измерения для правил — data/geometry_features.csv
(lateral_margin_mm; metal_metal_area_mm2 и metal_metal_band70_area_mm2 — те же признаки, что считает сервис).
Пороги правил заданы в src/action_evidence.py до этого расчёта и здесь не подбираются.

Команда 2.4.1 на OOF: «Переснять» = есть флаг и снимок не в зоне «не уверен» (как в карточке web/index.html:
зона «не уверен» даёт «Проверить»). Команда 2.5 = то же, но «Проверить», если ни у одного флага нет основания.
Нужна outputs/p25/frames.csv (tools/p25/prep_frames.py).
Запуск: python tools/p25/evidence_eval.py [--out docs/p25/evidence_eval.json]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "src"))
import pandas as pd  # noqa: E402

import action_evidence as AE  # noqa: E402
import common as C  # noqa: E402

CODE = {"sp_pos": "sp_pos", "sp_axis": "sp_axis", "sp_art": "sp_art", "hip_pos": "rh_pos", "hip_roi": "rh_roi"}


def evidence(area, flags, g):
    meas = {"lateral_margin_mm": g.get("lateral_margin_mm"), "metal_area_mm2": g.get("metal_metal_area_mm2"),
            "band70_area_mm2": g.get("metal_metal_band70_area_mm2"), "band70_n": g.get("metal_metal_band70_n")}
    if area == "hip":
        meas = {"lateral_margin_mm": meas["lateral_margin_mm"]}
    else:
        meas.pop("lateral_margin_mm")
    ev = AE.image_evidence("spine" if area == "spine" else "right_hip", {CODE[c]: v for c, v in flags.items()}, meas)
    if ev:
        for i in ev["items"]:
            i["criterion"] = AE.GROUP_OF[i["criterion"]]
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=C.ROOT / "docs" / "p25" / "evidence_eval.json")
    a = ap.parse_args()
    geo = pd.read_csv(C.ROOT / "data" / "geometry_features.csv")
    G = {r["file_path"]: r for r in geo.to_dict("records")}
    oof = C.load_oof()
    res = {"rules": {"hip_wide_field_mm": AE.HIP_WIDE_FIELD_MM, "art_zone": "metal_metal_band70_area_mm2 == 0",
                     "version": AE.EVIDENCE_VERSION}, "n_files": int(len(geo)), "n_files_oof": len(oof)}
    per = []
    for fp, d in oof.items():
        if not any(d["flags"].values()):
            continue
        ev = evidence(d["area"], d["flags"], G[fp])
        per.append({"file_path": fp, "area": d["area"], "cmd_241": "check" if d["uncertain"] else "retake",
                    "cmd_25": "check" if (d["uncertain"] or ev["command"] == "check") else "retake",
                    "items": {i["criterion"]: i["status"] for i in ev["items"]}, "labels": d["labels"],
                    "any_label": any(d["labels"].values()), "lm": G[fp].get("lateral_margin_mm")})
    df = pd.DataFrame(per)
    ret = df[df.cmd_241 == "retake"]; down = ret[ret.cmd_25 == "check"]; kept = ret[ret.cmd_25 == "retake"]
    o = {"flagged_images": int(len(df)), "retake_241": int(len(ret)), "retake_to_check": int(len(down)),
         "retake_25": int(len(kept)),
         "by_area": {ar: {"retake_241": int((ret.area == ar).sum()), "to_check": int((down.area == ar).sum())}
                     for ar in ("spine", "hip")},
         "images_to_check_any_violation": {"n": int(len(down)), "with_any_label": int(down.any_label.sum())},
         "images_kept_retake_any_violation": {"n": int(len(kept)), "with_any_label": int(kept.any_label.sum())}}
    rows = [{"criterion": c, "status": st, "label": r["labels"][c]} for r in per if r["cmd_241"] == "retake"
            for c, st in r["items"].items()]
    cr = pd.DataFrame(rows)
    o["flag_status_table"] = {f"{c}:{st}": {"n_flags": int(len(g)), "true_violation_of_criterion": int(g.label.sum()),
                                            "share_true": round(float(g.label.mean()), 3)}
                              for (c, st), g in cr.groupby(["criterion", "status"])}
    sens = {}
    sub = [r for r in per if r["cmd_241"] == "retake" and "hip_pos" in r["items"]]
    for thr in (51.0, 55.0):
        wide = [r for r in sub if float(r["lm"]) > thr]
        narrow = [r for r in sub if float(r["lm"]) <= thr]
        sens[str(thr)] = {"hip_pos_flags": len(sub), "wide": len(wide),
                          "wide_true_hip_pos": sum(r["labels"]["hip_pos"] for r in wide),
                          "narrow": len(narrow), "narrow_true_hip_pos": sum(r["labels"]["hip_pos"] for r in narrow)}
    o["hip_threshold_sensitivity"] = sens
    res["oof"] = o

    frames = C.load_frames40(oof)
    first = C.load_first_answers()
    shows = []
    for fn, fr in frames.items():
        g = G[fr["file_path"]]
        ev = evidence(fr["area"], fr["flags"], g) if any(fr["flags"].values()) else None
        shows.append({"show": fr["show"], "area": fr["area"], "flags": [c for c, v in fr["flags"].items() if v],
                      "labels": [c for c, v in fr["labels"].items() if v], "uncertain": fr["uncertain"],
                      "lateral_margin_mm": round(float(g["lateral_margin_mm"]), 1) if fr["area"] == "hip" else None,
                      "art_zone": AE.art_zone_summary(g["metal_metal_area_mm2"], g["metal_metal_band70_area_mm2"],
                                                      g["metal_metal_band70_n"]) if fr["area"] == "spine" else None,
                      "command_25": ev["command"] if ev else None,
                      "evidence": {i["criterion"]: i["status"] for i in ev["items"]} if ev else {},
                      "doctors": {k: v.get(fn) for k, v in first.items()}})
    shows.sort(key=lambda x: x["show"])
    res["doctors40"] = {"shows": shows, "service_class1": sum(bool(s["flags"]) for s in shows),
                        "to_check": [s["show"] for s in shows if s["command_25"] == "check"],
                        "to_check_fp": [s["show"] for s in shows if s["command_25"] == "check" and not s["labels"]],
                        "to_check_true": [s["show"] for s in shows if s["command_25"] == "check" and s["labels"]],
                        "retake_kept_fp": [s["show"] for s in shows if s["command_25"] == "retake" and not s["labels"]]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("OOF:", {k: v for k, v in o.items() if k != "flag_status_table"})
    for k, v in o["flag_status_table"].items():
        print("  ", k, v)
    d40 = res["doctors40"]
    print("40 кадров: флаг есть у", d40["service_class1"], "| «Проверить»:", d40["to_check"], "| из них норма по разметке:",
          d40["to_check_fp"], "| с нарушением:", d40["to_check_true"], "| «Переснять» на норме:", d40["retake_kept_fp"])
    for s in shows:
        if s["flags"] and (not s["labels"] or s["command_25"] == "check" or s["show"] in (5, 8, 14)):
            print("   показ", s["show"], s["area"], "флаги", s["flags"], "разметка", s["labels"], "поле",
                  s["lateral_margin_mm"], "зона", s["art_zone"], "->", s["command_25"], s["evidence"], s["doctors"])
    print("->", a.out)


if __name__ == "__main__":
    main()
