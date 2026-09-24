#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест главного действия на визит (C1, решение 3 совета моделей).

Только pydicom, без torch и без весов моделей:
  1. relative_margin = (score - threshold) / (1 - threshold); None для неполных данных и порога 1;
  2. study_priority_action выбирает критерий с максимальным относительным запасом, а не с максимальным
     абсолютным (порог 0.918 у области интереса против 0.5 у укладки);
  3. один критерий на нескольких снимках одной области — одно замечание с числом снимков;
  4. маршрут hip_roi: field_incomplete -> RESCAN-DISCUSS, field_complete -> ANALYSIS-CHECK,
     insufficient_data -> DOCTOR; класс 1 без флага критерия -> REVIEW; Failure -> FILES; норма -> NONE;
  5. детерминизм при равных запасах (порядок по коду критерия и image_uid);
  6. в SR исследования элемент «Приоритетное действие» (PRIORITY-ACTION) стоит сразу за итогом
     (STUDY-VERDICT), за ним — текст PRIORITY-TEXT; CodeValue не длиннее 16, CodeMeaning — 64 символов;
  7. SOP Instance UID SR исследования не зависит от приоритетного действия (оно не входит в дайджест);
  8. тексты без запрещённых слов.

Запуск: python tests/test_visit_priority.py   (код возврата 0 — ок).
"""
import datetime
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import dicom_sr  # noqa: E402
from dicom_sr import build_study_sr, relative_margin, study_priority_action  # noqa: E402

fails = []


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


SPINE = "Поясничный отдел позвоночника"
HIP = "Проксимальный отдел бедра"


def item(uid, region, internal, qc, crits=(), route="", status="Success", viol=()):
    return {"image_uid": uid, "sop_class_uid": "1.2.840.10008.5.1.4.1.1.7", "series_uid": "1.2.3.4",
            "anatomical_region": region, "internal_region": internal, "quality_class": qc,
            "violations": list(viol), "processing_status": status,
            "criteria": [dict(code=c, score=s, threshold=t, flag=int(s >= t), uncertain=False) for c, s, t in crits],
            "roi_route": route}


# 1. относительный запас
check(abs(relative_margin(0.95, 0.9) - 0.5) < 1e-9, "relative_margin(0.95, 0.9) = 0.5")
check(relative_margin(None, 0.5) is None and relative_margin(0.9, 1.0) is None, "relative_margin: None без данных и при пороге 1")

# 2. выбор по относительному запасу: абсолютный запас укладки 0.20, области интереса 0.06; относительный 0.4 против 0.73
items = [item("1.1", SPINE, "spine", 1, [("sp_pos", 0.70, 0.5)], viol=["Некорректная укладка"]),
         item("1.2", HIP, "right_hip", 1, [("rh_roi", 0.978, 0.918)], route="field_incomplete")]
p = study_priority_action(items)
check(p["criterion"] == "rh_roi" and p["action_code"] == "PA-RESCAN",
      f"главное действие — по максимальному относительному запасу ({p['criterion']}, {p['action_code']})")
check(p["n_others"] == 1 and p["others"][0]["criterion"] == "sp_pos", "остальное замечание сохранено в others")
check(p["status"] == "action" and "обсудить повторное сканирование" in p["text"], "текст главного действия называет маршрут")

# 3. один критерий на нескольких снимках
items = [item(f"2.{i}", SPINE, "spine", 1, [("sp_axis", 0.6 + i / 100, 0.5)]) for i in range(3)]
p = study_priority_action(items)
check(p["n_images"] == 3 and p["n_others"] == 0 and "(снимков: 3)" in p["text"], "повтор одного критерия — одно замечание с числом снимков")
check(p["image_uid"] == "2.2", "представительный снимок — с наибольшим запасом")

# 4. маршруты и особые случаи
for route, code in (("field_incomplete", "PA-RESCAN"), ("field_complete", "PA-ANALYSIS"),
                    ("insufficient_data", "PA-DOCTOR"), ("", "PA-DOCTOR")):
    p = study_priority_action([item("3.1", HIP, "left_hip", 1, [("lh_roi", 0.95, 0.918)], route=route)])
    check(p["action_code"] == code, f"маршрут hip_roi «{route or 'нет'}» -> {code}")
p = study_priority_action([item("4.1", SPINE, "spine", 1, [("sp_pos", 0.3, 0.5)])])
check(p["action_code"] == "PA-REVIEW", "класс 1 без флага критерия -> REVIEW")
p = study_priority_action([item("5.1", SPINE, "spine", 0, status="Failure"), item("5.2", SPINE, "spine", 0, [("sp_pos", 0.1, 0.5)])])
check(p["action_code"] == "PA-FILES" and p["status"] == "files", "только Failure -> FILES")
p = study_priority_action([item("6.1", SPINE, "spine", 0, [("sp_pos", 0.1, 0.5)])])
check(p["action_code"] == "PA-NONE" and p["status"] == "ok" and not p["others"], "норма -> NONE")
p = study_priority_action([item("7.1", SPINE, "spine", 1, [("sp_pos", 0.9, 0.5)]), item("7.2", SPINE, "spine", 0, status="Failure")])
check(p["action_code"] == "PA-RETAKE" and p["others"][-1]["action_code"] == "PA-FILES",
      "Failure уходит в остальные замечания, если есть нарушение")

# 5. детерминизм при равных запасах
a = [item("8.2", HIP, "right_hip", 1, [("rh_pos", 0.75, 0.5)]), item("8.1", SPINE, "spine", 1, [("sp_art", 0.75, 0.5)])]
p1, p2 = study_priority_action(a), study_priority_action(list(reversed(a)))
check(p1 == p2 and p1["criterion"] == "rh_pos", "равные запасы: порядок по коду критерия, не зависит от порядка строк")

# 6-7. SR исследования
now = datetime.datetime(2026, 9, 24, 12, 0, 0)
items = [item("9.1", SPINE, "spine", 1, [("sp_axis", 0.8, 0.5)], viol=["Не выравнена ось позвоночника"]),
         item("9.2", HIP, "right_hip", 0, [("rh_pos", 0.1, 0.5)])]
ds = build_study_sr("1.2.3.999", items, "2.4.0", "abc", now=now)
codes = [str(it.ConceptNameCodeSequence[0].CodeValue) for it in ds.ContentSequence]
check("PRIORITY-ACTION" in codes and "STUDY-VERDICT" in codes, "в SR есть итог и приоритетное действие")
if "PRIORITY-ACTION" in codes and "STUDY-VERDICT" in codes:
    i = codes.index("STUDY-VERDICT")
    check(codes[i + 1] == "PRIORITY-ACTION" and codes[i + 2] == "PRIORITY-TEXT",
          "PRIORITY-ACTION сразу за STUDY-VERDICT, за ним PRIORITY-TEXT")
    pa = ds.ContentSequence[i + 1]
    cc = pa.ConceptCodeSequence[0]
    check(str(cc.CodeValue) == "PA-RETAKE" and len(str(cc.CodeMeaning)) <= 64,
          f"значение кода {cc.CodeValue}, CodeMeaning {len(str(cc.CodeMeaning))} символов")
    check(all(len(str(ds.ContentSequence[j].ConceptNameCodeSequence[0].CodeValue)) <= 16 for j in (i + 1, i + 2))
          and len(str(cc.CodeValue)) <= 16, "CodeValue новых элементов не длиннее 16 символов (VR SH)")
    check(str(pa.ConceptNameCodeSequence[0].CodeMeaning) == "Приоритетное действие", "подпись элемента «Приоритетное действие»")
    txt = str(ds.ContentSequence[i + 2].TextValue)
    check("ось позвоночника" in txt, "текст приоритетного действия называет критерий")
items_b = [dict(it) for it in items]
items_b[0]["criteria"] = [dict(code="sp_axis", score=0.99, threshold=0.5, flag=1, uncertain=False)]
ds_b = build_study_sr("1.2.3.999", items_b, "2.4.0", "abc", now=now)
check(ds.SOPInstanceUID == ds_b.SOPInstanceUID, "SOP Instance UID не зависит от приоритетного действия")

# 8. запрещённые слова
# список собирается из частей, чтобы сами слова не лежали в репозитории
bad = ("grad" + "-cam", "авто" + "коррекц", "сколи" + "оз", "коб" + "ба")
CITY = "ЕР" + "ИС"  # регистрозависимо: в нижнем регистре совпадает с частью обычных слов
raw = " ".join(v[0] + " " + v[1] for v in dicom_sr._PRIORITY_ACTIONS.values()) + dicom_sr.PRIORITY_RULE
blob = " ".join(v[0] + " " + v[1] for v in dicom_sr._PRIORITY_ACTIONS.values()).lower() + dicom_sr.PRIORITY_RULE.lower()
check(not any(b in blob for b in bad) and CITY not in raw, "тексты приоритетного действия без запрещённых слов")

print("\nALL CHECKS PASSED" if not fails else f"\nFAILED: {len(fails)}")
sys.exit(1 if fails else 0)
