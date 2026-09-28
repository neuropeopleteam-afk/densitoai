#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест 2.5: команда только с доказательством, роли критериев, второе мнение по оси (src/action_evidence.py).

Без весов моделей и без DICOM:
  1. укладка бедра: боковой запас > 51 мм -> «Проверить укладку по малому вертелу: поле шире обычного» с мм;
     ровно 51 мм и меньше -> «Переснять» (основание измерено); запас не измерен -> «Проверить»;
  2. посторонние предметы: вся площадь вне верхних 70 % протяжённости кости -> «Проверить: плотный участок вне зоны
     L1–L4, на измерение может не влиять»; площадь в зоне > 0 -> «Переснять»; участков нет -> «Проверить»;
  3. снимок: хотя бы один флаг с основанием (или без правила: sp_pos, sp_axis, hip_roi) -> «Переснять»;
     Failure и снимок без флагов -> сертификата нет (отсутствие основания не маскирует Failure);
  4. роли: sp_pos, sp_axis, hip_roi — измерение; sp_art, hip_pos — подсказка, решает врач;
  5. второе мнение по оси: |p_geom - p_emb| > 0.45 -> флаг, ровно 0.45 -> нет, нет контура -> None;
  6. api_server._details: action_code review_evidence, criteria[].role/evidence, action_evidence,
     axis_second_opinion; класс строки не трогается;
  7. dicom_sr: evidence_command == check -> PA-EVIDENCE (код <= 16 символов, смысл <= 64), без него PA-RETAKE;
     в SR снимка есть текст ACTION-EVIDENCE и AXIS-2ND-OPINION; inference._criteria_for_priority передаёт
     evidence_command; SOP Instance UID от новых полей не зависит;
  8. extras.compute_extras_for_rows: колонки action_command / action_evidence / flag_roles /
     axis_contours_diverge, у Failure пусто; первые 11 колонок results_extras.csv прежние;
  9. пороги заданы до расчёта и воспроизводятся без меток: 51 мм — верхняя терциль lateral_margin_mm по 333
     снимкам бедра; 0.45 — правило tools/p25/axis_disagreement_nested.pick_threshold на OOF sp_axis;
 10. веб: карточка и лаборантский приоритет знают EVIDENCE-CHECK, блок основания, роль и второе мнение;
     норма «Площадь посторонних объектов в зоне измерения» есть в actions.json и в резервных текстах;
 11. тексты без запрещённых слов.

Запуск: python tests/test_action_evidence.py   (код возврата 0 — ок).
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "p25"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("DENSITO_ROOT", str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import action_evidence as AE  # noqa: E402

fails = []


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


# ---- 1. бедро ---------------------------------------------------------------------------------------------
e = AE.criterion_evidence("rh_pos", {"lateral_margin_mm": 61.8})
check(e["command"] == "check" and e["status"] == "wide_field", "бедро 61.8 мм -> Проверить (wide_field)")
check(e["text"].startswith("Проверить укладку по малому вертелу: поле шире обычного") and "62 мм" in e["text"]
      and "оценка менее надёжна" in e["text"], "текст про малый вертел с мм: " + e["text"])
check(AE.criterion_evidence("lh_pos", {"lateral_margin_mm": 51.0})["command"] == "retake", "ровно 51 мм -> Переснять")
check(AE.criterion_evidence("lh_pos", {"lateral_margin_mm": 40.0})["status"] == "confirmed", "40 мм -> основание измерено")
e = AE.criterion_evidence("rh_pos", {"lateral_margin_mm": None})
check(e["command"] == "check" and e["status"] == "no_witness", "запас не измерен -> Проверить")
check(AE.criterion_evidence("rh_pos", {"lateral_margin_mm": float("nan")})["status"] == "no_witness", "NaN -> не измерено")

# ---- 2. посторонние предметы --------------------------------------------------------------------------------
e = AE.criterion_evidence("sp_art", {"metal_area_mm2": 63.0, "band70_area_mm2": 0.0, "band70_n": 0})
check(e["command"] == "check" and e["status"] == "below_zone", "вся площадь ниже зоны -> Проверить (below_zone)")
check(e["text"].startswith("Проверить: плотный участок вне зоны L1–L4, на измерение может не влиять")
      and "63 мм²" in e["text"], "текст про зону L1–L4: " + e["text"])
e = AE.criterion_evidence("sp_art", {"metal_area_mm2": 323.2, "band70_area_mm2": 136.1, "band70_n": 2})
check(e["command"] == "retake" and "136 мм²" in e["text"], "площадь в зоне 136 мм² -> Переснять")
check(AE.criterion_evidence("sp_art", {"metal_area_mm2": 0.0, "band70_area_mm2": 0.0})["status"] == "no_witness",
      "участков нет -> измеримого основания нет")
check(AE.criterion_evidence("sp_art", {})["status"] == "not_applicable"
      and AE.criterion_evidence("sp_art", {})["command"] == "retake", "без измерений -> как в 2.4.1")
for c in ("sp_pos", "sp_axis", "rh_roi", "lh_roi"):
    check(AE.criterion_evidence(c, {})["command"] == "retake", f"{c}: правила нет -> Переснять как в 2.4.1")

# ---- 3. снимок ----------------------------------------------------------------------------------------------
m = {"metal_area_mm2": 63.0, "band70_area_mm2": 0.0}
ev = AE.image_evidence("spine", {"sp_pos": 0, "sp_axis": 0, "sp_art": 1}, m)
check(ev["command"] == "check" and ev["checks"] == ["sp_art"], "только sp_art вне зоны -> снимок Проверить")
ev = AE.image_evidence("spine", {"sp_pos": 0, "sp_axis": "1", "sp_art": True}, m)
check(ev["command"] == "retake" and ev["text"] == "", "sp_art вне зоны + ось -> Переснять (ось без правила)")
check(AE.image_evidence("spine", {"sp_art": 1}, m, is_failure=True) is None, "Failure -> сертификата нет")
check(AE.image_evidence("spine", {"sp_art": 0, "sp_pos": "", "sp_axis": None}, m) is None, "без флагов -> None")

# ---- 4. роли ------------------------------------------------------------------------------------------------
check([AE.role_of(c) for c in ("sp_pos", "sp_axis", "rh_roi", "lh_roi")] == ["measured"] * 4, "измерения: sp_pos, sp_axis, hip_roi")
check([AE.role_of(c) for c in ("sp_art", "rh_pos", "lh_pos")] == ["hint"] * 3, "подсказки: sp_art, hip_pos")
check(AE.ROLE_TEXT["hint"] == "подсказка, решает врач", "подпись подсказки")

# ---- 5. ось -------------------------------------------------------------------------------------------------
a = AE.axis_second_opinion(0.969, 0.056)
check(a["flag"] and a["text"].startswith("Контуры разошлись — посмотрите ось"), "0.969 / 0.056 -> флаг")
check(AE.axis_second_opinion(0.70, 0.25)["flag"] is False, "расхождение ровно 0.45 -> нет флага")
check(AE.axis_second_opinion(None, 0.3) is None, "нет контура -> None")
check(AE.axis_from_debug({"internal_region": "right_hip", "sp_axis_p_geom": 1, "sp_axis_p_emb": 0}) is None,
      "бедро -> второго мнения по оси нет")

# ---- 6. API _details ----------------------------------------------------------------------------------------
import api_server as A  # noqa: E402
import inference as I  # noqa: E402

cfg = I.load_config()
dbg = {"internal_region": "spine", "feat_metal_metal_area_mm2": 63.0, "feat_metal_metal_band70_area_mm2": 0.0,
       "feat_metal_metal_band70_n": 0, "sp_axis_p_geom": 0.9, "sp_axis_p_emb": 0.1, "needs_review": 0}
for c, f in (("sp_pos", 0), ("sp_axis", 0), ("sp_art", 1)):
    dbg.update({f"{c}_flag": f, f"{c}_score": 0.9 if f else 0.1, f"{c}_threshold": 0.6, f"{c}_uncertain": 0})
row = {"quality_class": 1, "processing_status": "Success", "violation_type": "Присутствуют посторонние предметы"}
row0 = dict(row)
d = A._details(row, dbg, cfg)
check(d["action_code"] == "review_evidence" and d["action"].startswith("Проверить: плотный участок вне зоны L1–L4"),
      "API: action_code review_evidence")
check(d["action_evidence"]["command"] == "check", "API: action_evidence.command == check")
cr = {c["code"]: c for c in d["criteria"]}
check(cr["sp_art"]["role"] == "hint" and cr["sp_axis"]["role"] == "measured" and cr["sp_pos"]["role_text"] == "измерение",
      "API: роли критериев")
check(cr["sp_art"]["evidence"]["status"] == "below_zone" and cr["sp_axis"]["evidence"] is None, "API: основание по флагу")
check(d["axis_second_opinion"]["flag"] is True, "API: второе мнение по оси")
check(row == row0, "API: строка выгрузки не меняется")
meas = {m_["key"]: m_ for m_ in d["measurements"]}
check("metal_metal_band70_area_mm2" in meas and meas["metal_metal_band70_area_mm2"]["title"]
      == "Площадь посторонних объектов в зоне измерения", "API: мера площади в зоне измерения")
dbg_u = dict(dbg, needs_review=1)
check(A._details(row, dbg_u, cfg)["action_code"] == "review_uncertain", "API: зона «не уверен» приоритетнее")
dbg_f = dict(dbg, error="boom")
d_f = A._details({"quality_class": 0, "processing_status": "Failure"}, dbg_f, cfg)
check(d_f["action_code"] == "manual" and d_f["action_evidence"] is None, "API: Failure -> manual, без сертификата")
mf = A._model_features({"sp_art_model_features": json.dumps({"variant": "baseline", "values": {
    "metal_metal_band70_area_log": float(np.log1p(136.1)), "metal_metal_band70_max_gap": 3.2}})}, "sp_art")
check(mf["items"][0]["key"] == "metal_metal_band70_area_mm2" and abs(mf["items"][0]["value"] - 136) <= 1,
      "API: признак модели в мм² (из логарифма)")

# ---- 7. SR и приоритет ----------------------------------------------------------------------------------------
import dicom_sr as S  # noqa: E402

crit = I._criteria_for_priority(cfg, dbg)
check({c["code"]: c.get("evidence_command") for c in crit}.get("sp_art") == "check", "inference: evidence_command")
base = {"image_uid": "1.2.3.4", "anatomical_region": cfg["regions"]["spine"], "quality_class": 1,
        "violations": ["Присутствуют посторонние предметы"], "quality_prob": 0.8, "processing_status": "Success",
        "internal_region": "spine"}
it_e = dict(base, criteria=crit, **I._evidence_texts_for_sr(cfg, dbg, "Success"))
it_r = dict(base, criteria=[{k: v for k, v in c.items() if k != "evidence_command"} for c in crit])
pe, pr = S.study_priority_action([it_e]), S.study_priority_action([it_r])
check(pe["action_code"] == "PA-EVIDENCE" and "основание для пересъёмки не измерено" in pe["text"], "SR: PA-EVIDENCE")
check(pr["action_code"] == "PA-RETAKE", "SR: без основания в данных -> PA-RETAKE как раньше")
check(len("PA-EVIDENCE") <= 16 and all(len(v[0]) <= 64 for v in S._PRIORITY_ACTIONS.values()), "SR: длины кодов")
ds = S.build_study_sr("1.2.3", [it_e], "2.5.0", "x")


def _walk(d, acc):
    for it_ in getattr(d, "ContentSequence", []) or []:
        acc.append((it_.ConceptNameCodeSequence[0].CodeValue, str(getattr(it_, "TextValue", ""))))
        _walk(it_, acc)
    return acc


items_sr = _walk(ds, [])
txt = " ".join(a + " " + b for a, b in items_sr)
check(all(len(a) <= 16 for a in ("ACTION-EVIDENCE", "AXIS-2ND-OPINION", "PA-EVIDENCE")), "SR: новые CodeValue <= 16 символов")
check("ACTION-EVIDENCE" in txt and "AXIS-2ND-OPINION" in txt and "вне зоны L1–L4" in txt, "SR: тексты основания и оси")
ds2 = S.build_study_sr("1.2.3", [dict(base)], "2.5.0", "x")
check(ds.SOPInstanceUID == ds2.SOPInstanceUID, "SR: SOP Instance UID не зависит от новых полей")
check(I._evidence_texts_for_sr(cfg, dbg, "Failure") == {"action_evidence_text": "", "axis_second_opinion_text": ""},
      "SR: у Failure нет текста основания")

# ---- 8. results_extras.csv ----------------------------------------------------------------------------------
import extras as X  # noqa: E402

rows = [dict(row, path_to_study="a.dcm", image_uid="1", study_uid="s"),
        {"quality_class": 0, "processing_status": "Failure", "path_to_study": "b.dcm", "image_uid": "2", "study_uid": "s"}]
xr = X.compute_extras_for_rows(rows, [dbg, dbg_f], [None, None])
check(list(xr[0].keys())[:len(X.EXTRAS_COLUMNS)] == X.EXTRAS_COLUMNS, "extras: первые колонки прежние")
check(xr[0]["action_command"] == "check" and xr[0]["action_evidence"] == "sp_art:below_zone"
      and xr[0]["flag_roles"] == "sp_art:подсказка, решает врач" and xr[0]["axis_contours_diverge"] == 1,
      "extras: новые колонки")
check(xr[1]["action_command"] == "" and xr[1]["axis_contours_diverge"] == "", "extras: у Failure пусто")

# ---- 9. пороги без меток ------------------------------------------------------------------------------------
g = pd.read_csv(ROOT / "data" / "geometry_features.csv")
lm = g[g.region != "spine"].lateral_margin_mm.dropna()
check(len(lm) == 333 and abs(float(np.quantile(lm, 2 / 3)) - AE.HIP_WIDE_FIELD_MM) < 0.05,
      f"51 мм = верхняя терциль бокового запаса ({np.quantile(lm, 2 / 3):.2f})")
import axis_disagreement_nested as N  # noqa: E402

o = pd.read_csv(ROOT / "models" / "oof_stacked_spine_sp_axis.csv")
t = N.pick_threshold(np.abs(o.oof_geom.values - o.oof_emb.values), (o.pred_label != o.y_true).values)
check(abs(t - AE.AXIS_DISAGREE_T) < 1e-9, f"порог оси = правилу вложенной проверки на всех 166 ({t})")

# ---- 10. веб ------------------------------------------------------------------------------------------------
html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
for s in ("'EVIDENCE-CHECK'", "function evidenceBlock", "function axisSecondOpinion", "подсказка, решает врач",
          "Основание для пересъёмки не измерено", "metal_metal_band70_area_mm2"):
    check(s in html, f"web/index.html: {s}")
aj = json.loads((ROOT / "web" / "assets" / "actions.json").read_text(encoding="utf-8"))
check(aj["measurement_norms"].get("metal_metal_band70_area_mm2", {}).get("label")
      == "Площадь посторонних объектов в зоне измерения", "actions.json: норма площади в зоне измерения")
_cmp = "луч" + "ше"
check((_cmp + " врач") not in html, "веб: без сравнения с врачом")

# ---- 11. запрещённые слова ----------------------------------------------------------------------------------
texts = [e_["text"] for e_ in (AE.criterion_evidence(c, mm) for c in ("rh_pos", "sp_art") for mm in (
    {"lateral_margin_mm": 60}, {"lateral_margin_mm": 40}, {"metal_area_mm2": 5, "band70_area_mm2": 0},
    {"metal_area_mm2": 5, "band70_area_mm2": 5}, {}))]
src = "\n".join(texts + [(ROOT / "src" / "action_evidence.py").read_text(encoding="utf-8")])
# список собирается из частей, чтобы сами слова не лежали в репозитории (как в test_card_measurements.py)
bad = ("grad" + "-cam", "grad" + "cam", "авто" + "коррекц", "авто-" + "коррекц", "сколи" + "оз", "коб" + "б")
CITY = "ЕР" + "ИС"  # регистрозависимо
check(not any(b in src.lower() for b in bad) and CITY not in src, "тексты без запрещённых слов")

print("\nALL CHECKS PASSED" if not fails else f"\nFAILED: {len(fails)}")
sys.exit(1 if fails else 0)
