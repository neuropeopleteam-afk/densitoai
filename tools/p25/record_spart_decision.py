#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""record_spart_decision.py — запись решения владельца по `sp_art` (2.5.0) в models/nested_gate_decisions.json
и поля nested_* блока spine.sp_art в models/metrics_summary.json.

Источник чисел — эксперимент `exp_spart` (REPORT_spart_position.md, разделы 3.1–3.4):
  outputs/spart_position/summary.json          — ветки gate / band70 / base (5 x 20 x 3, n = 166);
  outputs/spart_position/paired_gate_gate.json — калиброванный гейт (tools/paired_gate.py, Холм, m = 3);
  outputs/spart_position/paired_gate_band70.json, paired_gate_gate_m1.json.
Путь к каталогу эксперимента — `--exp` (по умолчанию /home/user/workspace/work/exp_spart/B/outputs/spart_position).

Поля nested_* пишутся той же функцией, что и при обучении (`src/train_stacked.py: nested_block_for` с принятой
записью kind == 'feature'), поэтому повторный `train_stacked.py` даст те же поля. Пороги, OOF, веса и модели
не меняются. Повторный запуск идемпотентен (запись sp_art kind == 'feature' заменяется).

  python tools/p25/record_spart_decision.py [--exp DIR] [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DEC = ROOT / "models" / "nested_gate_decisions.json"
MS = ROOT / "models" / "metrics_summary.json"
EXP = Path("/home/user/workspace/work/exp_spart/B/outputs/spart_position")


def gate_block(g: dict, protocol: str) -> dict:
    da = g["delta_auc"]
    return {
        "accepted": bool(g["accepted"]),
        "protocol": protocol,
        "delta_auc_ci90": da["ci90"],
        "delta_auc_ci95": da["ci95"],
        "boot_se": da["boot_se"],
        "p_boot_one_sided": da["p_boot_one_sided"],
        "p_combined_one_sided": g["p_combined_one_sided"],
        "signflip_p_one_sided": g["signflip_permutation"]["p_one_sided"],
        "delta_macro_f1_vote": g["delta_macro_f1"]["point_vote"],
        "delta_macro_f1_ci90": g["delta_macro_f1"]["ci90"],
        "jackknife_groups": g.get("jackknife_groups"),
        "conditions": g["conditions"],
        "old_rule": g["old_rule"],
    }


def build_record(exp: Path) -> dict:
    s = json.loads((exp / "summary.json").read_text(encoding="utf-8"))
    br = s["branches"]
    g = json.loads((exp / "paired_gate_gate.json").read_text(encoding="utf-8"))
    g70 = json.loads((exp / "paired_gate_band70.json").read_text(encoding="utf-8"))
    g1 = json.loads((exp / "paired_gate_gate_m1.json").read_text(encoding="utf-8"))
    p = s["protocol"]
    base, gate, b70 = br["base"], br["gate"], br["band70"]
    holm = "tools/paired_gate.py (идея 11), партия 3 гипотез (gate, band70, gate3), поправка Холма"
    cg = gate_block(g, holm)
    cg_m1 = gate_block(g1, "tools/paired_gate.py, m = 1 (без поправки), справочно")
    rec = {
        "region": "spine",
        "criterion": "sp_art",
        "kind": "feature",
        "candidate": "spart_zone_band70",
        "candidate_what": ("признаки контура A `metal_metal_band70_area_log`, `metal_metal_band70_max_gap` — площадь (log1p, мм²) "
                           "и перепад яркости плотных участков вне кости в верхних 70 % вертикальной протяжённости маски кости "
                           "(зона измерения L1–L4); те же компоненты, что у `metal_metal_area_mm2` (src/geometry_features.py: "
                           "foreign_object_features)"),
        "alternative_what": "контур A 2.4.x: metal_metal_area_mm2 + metal_metal_max_intensity_gap (площадь по всему кадру)",
        "protocol_id": "nested_spart_zone_2_5",
        "protocol": ("tools/spart_position/spart_position_gate.py (exp_spart): GroupKFold 5 внешних x 20 повторов x 3 внутренних; "
                     "группы (study, pixel_hash); порог prevalence_x1.4 на inner-OOF; стек 0.5/0.5 с контуром B imagenet; "
                     "отсечка c из {0.5, 0.6, 0.7, 0.8, 1.0} выбирается только на inner-OOF внешнего train"),
        "n": p["n"],
        "n_pos": p["n_pos"],
        "n_groups": p["n_groups"],
        "n_pos_groups": p["n_pos_studies"],
        "emb_source": p["emb_source"],
        "preproc": {"geom": "baseline", "emb": "baseline"},
        "n_repeats": p["repeats"],
        "n_repeats_gain_ge_0.03": gate["n_gain_repeats"],
        "n_repeats_needed": p["gain_repeats"],
        "mean_delta_auc": gate["d_auc_mean"],
        "sd_delta_auc": gate["d_auc_sd"],
        "min_delta_auc": gate["d_auc_min"],
        "mean_delta_f1pos": gate["d_f1pos_mean"],
        "mean_delta_macro_f1": gate["d_macro_f1_mean"],
        "n_repeats_macro_f1_not_worse": gate["n_macro_f1_not_worse"],
        "auc_base_mean_over_repeats": base["auc_mean"],
        "auc_cand_mean_over_repeats": gate["auc_mean"],
        "auc_cand_ci": None,
        "auc_geom_base_mean": base["auc_geom_mean"],
        "auc_geom_cand_mean": gate["auc_geom_mean"],
        "macro_f1_base_mean_over_repeats": base["macro_f1_mean"],
        "macro_f1_cand_mean_over_repeats": gate["macro_f1_mean"],
        "selection_counts": gate.get("selection_counts"),
        "score_variant": "gate: отсечка выбирается внутри фолда (основная ветка; в продакшене — фиксированная 70 %, см. production_variant_control)",
        "accepted": True,
        "accepted_by": ("решение владельца (27.09.2026) по действующему правилу nested, как голова укладки H2 в 2.4.0: "
                        "dAUC >= 0.03 в >= 14/20 повторов без потери macro-F1 — пройдено 19/20 (фиксированная отсечка 70 % — 20/20), "
                        "Δmacro-F1 +0.074"),
        "calibrated_gate": cg,
        "calibrated_gate_m1": cg_m1,
        "calibrated_gate_note": ("калиброванный гейт tools/paired_gate.py НЕ пройден: ДИ95 кластерного бутстрапа ΔAUC [−0.029; +0.176], "
                                 "p_boot 0.072, sign-flip p 0.092 (не проходит и без поправки Холма). Положительных исследований 17, "
                                 "поэтому это не нехватка мощности (в отличие от H2, где их было 6): прирост неоднороден по исследованиям. "
                                 "На 40 кадрах слепой проверки врачами эффекта нет: совпадение с разметкой 27 → 28 из 40, найдено с верной "
                                 "причиной 13 из 20 без изменений, ложных тревог 6 из 20 без изменений, с большинством врачей 31 → 30 из 40. "
                                 "Механизм: прежний признак в основном считал предметами крылья подвздошных костей внизу кадра. "
                                 "Признак включён в 2.5.0 решением владельца с этой оговоркой."),
        "production_variant_control": {
            "what": "band70: фиксированная отсечка 70 % — ровно то, что стоит в продакшене (config.yaml geometry_cols.sp_art)",
            "n_repeats_gain_ge_0.03": b70["n_gain_repeats"],
            "mean_delta_auc": b70["d_auc_mean"],
            "sd_delta_auc": b70["d_auc_sd"],
            "min_delta_auc": b70["d_auc_min"],
            "mean_delta_macro_f1": b70["d_macro_f1_mean"],
            "auc_base_mean_over_repeats": base["auc_mean"],
            "auc_cand_mean_over_repeats": b70["auc_mean"],
            "calibrated_gate": gate_block(g70, holm),
        },
        "doctors_40": {
            "source": "exp_spart REPORT_spart_position.md, раздел 6 (OOF, кадры tools/review/kit_light_manifest.json)",
            "agreement_with_markup": [27, 28, 40],
            "found_with_correct_reason": [13, 13, 20],
            "false_alarms": [6, 6, 20],
            "majority_of_doctors_agrees": [31, 30, 40],
            "changed_frames": [23, 32, 44],
        },
        "sources": [
            "work/exp_spart/REPORT_spart_position.md (разделы 3–6, 9)",
            "work/exp_spart/B/outputs/spart_position/summary.json",
            "work/exp_spart/B/outputs/spart_position/paired_gate_gate.json",
            "work/exp_spart/B/outputs/spart_position/paired_gate_band70.json",
            "docs/NESTED_GATE_REPORT.md, часть 8",
        ],
        "date": "2026-09-27",
        "version": "2.5.0",
    }
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=str(EXP))
    ap.add_argument("--check", action="store_true", help="только сверить, что поля metrics_summary согласованы с записью")
    a = ap.parse_args(argv)
    import train_stacked as ts

    dec = json.loads(DEC.read_text(encoding="utf-8"))
    if not a.check:
        rec = build_record(Path(a.exp))
        crit = [r for r in dec["criteria"] if not (r.get("criterion") == "sp_art" and r.get("kind") == "feature")]
        crit.append(rec)
        dec["criteria"] = crit
        dec["protocol"]["feature_candidates"] = (
            "записи kind == 'feature' — кандидаты-признаки контура A (H2 sp_pos, 2.4.0: sppos_nested.py; sp_art 2.5.0: "
            "spart_position_gate.py; 5 x 20 x 3, правило dAUC >= 0.03 в >= 14/20); принятая запись имеет приоритет для "
            "nested_* в metrics_summary.json (train_stacked.nested_decision_for)")
        DEC.write_text(json.dumps(dec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ms = json.loads(MS.read_text(encoding="utf-8"))
    blk = ms["spine"]["sp_art"]
    nb = ts.nested_block_for("sp_art", str(blk.get("emb_source") or "imagenet"), ts.nested_decision_for("sp_art"))
    want = {"nested_protocol": nb["protocol"], "nested_auc_production": nb["production"],
            "nested_auc_alternative": nb["alternative"], "nested_alternative_what": nb["alternative_what"],
            "nested_auc_mean": nb["mean"], "nested_auc_ci": nb["ci"], "nested_auc_alternative_ci": nb["alternative_ci"],
            "nested_auc_base_mean": nb["base_mean"], "nested_repeats_gain_ge_0.03": nb["gain"]}
    diff = {k: (blk.get(k), v) for k, v in want.items() if blk.get(k) != v}
    if a.check:
        print("расхождений:", len(diff), diff)
        return 1 if diff else 0
    for k, v in want.items():
        blk[k] = v
    MS.write_text(json.dumps(ms, ensure_ascii=False, indent=1), encoding="utf-8")
    print("sp_art nested:", {k: blk[k] for k in want})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
