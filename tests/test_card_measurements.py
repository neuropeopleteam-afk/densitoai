#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест данных карточки врача (C1: замечания владельца З2, З3, развилка hip_roi).

Без torch и без весов моделей (файлы моделей читаются только ради feature_cols):
  1. extras.study_warning_items: повторы областей из study_coherence сворачиваются в одно замечание
     с числом и расшифровкой, одинаковые строки не дублируются, пустой вход -> [];
  2. auto_roi.hip_field_status: короткий скан / мало диафиза / кость у края -> field_incomplete;
     всё в норме -> field_complete; нет длины скана или диафиза -> insufficient_data; ключи feat_* читаются;
  3. extras.hip_roi_reason: разные action_key и тексты для «поле неполное» и «поле полное»,
     подпись «предложение системы, исходное измерение не меняется»; без флага -> None;
     action_key есть в config.yaml actions;
  4. api_server._decision_source: оба контура за нарушение -> geom_and_image, только изображение ->
     image (текст «решение по изображению, измерения в норме — проверьте снимок визуально»),
     только измерения -> geom, без флага -> None, резервное правило -> rule;
  5. api_server._model_features: признаки из debug-поля <crit>_model_features, отношения в процентах;
  6. inference._model_features_json берёт список признаков из feature_cols файла модели
     (для hip_roi — длина скана и диафиз ниже вертела), значения — из варианта предобработки критерия;
  7. в config.yaml нет порога 5° и 20 мм как «ТЗ»: axis_angle_deg и edge_distance_mm — ориентиры (guide_*);
  8. тексты без запрещённых слов.

Запуск: python tests/test_card_measurements.py   (код возврата 0 — ок).
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("DENSITO_ROOT", str(ROOT))

import yaml  # noqa: E402

import extras  # noqa: E402
from auto_roi import hip_field_status  # noqa: E402

fails = []


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


# 1. предупреждения по исследованию
rows = [{"study_uid": "S", "image_uid": f"i{i}", "region": r, "pixel_hash": h}
        for i, (r, h) in enumerate([("spine", "a"), ("spine", "a"), ("spine", "b"),
                                    ("right_hip", "c"), ("right_hip", "c"), ("left_hip", "d"), ("left_hip", "e")])]
w = extras.study_coherence(rows)["S"]
items = extras.study_warning_items(w)
codes = [it["code"] for it in items]
check(len(w) >= 4 and len(items) == len(set(codes)), f"study_coherence дал {len(w)} строк -> {len(items)} уникальных замечаний")
rep = [it for it in items if it["code"] == "region_repeat"]
check(len(rep) == 1 and rep[0]["count"] == 3 and "позвоночник ×3" in rep[0]["detail"] and "правое бедро ×2" in rep[0]["detail"],
      "повторы областей — одно замечание: " + (rep[0]["detail"] if rep else "нет"))
check(sum(1 for c in codes if c == "pixel_duplicates") == 1, "дубликаты кадров — одно замечание")
items2 = extras.study_warning_items(" | ".join(w + w))
check([(i["code"], i["text"]) for i in items2] == [(i["code"], i["text"]) for i in items], "строка через « | » с повтором даёт те же уникальные замечания")
check(extras.study_warning_items("") == [] and extras.study_warning_items(None) == [], "пустой вход -> []")

# 2. полнота поля бедра
st = hip_field_status({"scan_length_mm": 180, "shaft_len_below_troch_mm": 80, "lateral_margin_mm": 30})
check(st["status"] == "field_incomplete" and st["reasons"] == ["scan_too_short"], "короткий скан -> field_incomplete")
st = hip_field_status({"feat_scan_length_mm": 260, "feat_shaft_len_below_troch_mm": 40, "feat_lateral_margin_mm": 30})
check(st["status"] == "field_incomplete" and "shaft_below_trochanter_too_short" in st["reasons"], "мало диафиза (ключи feat_*) -> field_incomplete")
st = hip_field_status({"scan_length_mm": 260, "shaft_len_below_troch_mm": 90, "lateral_margin_mm": 12})
check(st["status"] == "field_incomplete" and st["reasons"] == ["lateral_margin_below_threshold"], "кость у края кадра -> field_incomplete")
st = hip_field_status({"scan_length_mm": 260, "shaft_len_below_troch_mm": 90, "lateral_margin_mm": 30})
check(st["status"] == "field_complete" and st["reasons"] == [], "поле в норме -> field_complete")
st = hip_field_status({"scan_length_mm": 260})
check(st["status"] == "insufficient_data", "нет длины диафиза -> insufficient_data")
check(hip_field_status(None)["status"] == "insufficient_data", "пустой вход -> insufficient_data")

# 3. развилка hip_roi
cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
inc = extras.hip_roi_reason({"feat_scan_length_mm": 150, "feat_shaft_len_below_troch_mm": 30}, True)
com = extras.hip_roi_reason({"feat_scan_length_mm": 260, "feat_shaft_len_below_troch_mm": 90, "feat_lateral_margin_mm": 35}, True)
unk = extras.hip_roi_reason({}, True)
check(inc["code"] == "field_incomplete" and "повторное сканирование" in inc["action"], "поле неполное -> обсудить повторное сканирование")
check(com["code"] == "field_complete" and "повторно не облучать" in com["action"] and "анализ" in com["action"],
      "поле полное -> проверить анализ на аппарате, повторно не облучать")
check(len({inc["action_key"], com["action_key"], unk["action_key"]}) == 3 and inc["action"] != com["action"], "разные коды и тексты маршрутов")
check(all(x["note"] == "предложение системы, исходное измерение не меняется" for x in (inc, com, unk)), "подпись «предложение системы, исходное измерение не меняется»")
check(extras.hip_roi_reason({"feat_scan_length_mm": 150}, False) is None, "без флага hip_roi -> None")
acts = cfg.get("actions", {})
check(all(x["action_key"] in acts and "Предложение системы, исходное измерение не меняется" in acts[x["action_key"]]["hint"]
          for x in (inc, com, unk)), "action_key маршрутов есть в config.yaml actions, подсказки с подписью")
check("20 мм" not in acts.get("hip_roi", {}).get("hint", ""), "общая подсказка hip_roi не выдаёт 20 мм за требование")

# 4-5. источник решения и признаки модели в ответе API
import api_server  # noqa: E402

base = {"sp_axis_method": "stacked_rank_avg", "sp_axis_threshold": 0.6}
check(api_server._decision_source(dict(base, sp_axis_rank_geom=0.7, sp_axis_rank_emb=0.8), "sp_axis", True) == "geom_and_image", "оба контура -> geom_and_image")
check(api_server._decision_source(dict(base, sp_axis_rank_geom=0.4, sp_axis_rank_emb=0.9), "sp_axis", True) == "image", "только изображение -> image")
check(api_server._decision_source(dict(base, sp_axis_rank_geom=0.9, sp_axis_rank_emb=0.4), "sp_axis", True) == "geom", "только измерения -> geom")
check(api_server._decision_source(dict(base, sp_axis_rank_geom=0.9, sp_axis_rank_emb=0.9), "sp_axis", False) is None, "без флага -> None")
check(api_server._decision_source({"sp_axis_method": "fallback_rule"}, "sp_axis", True) == "rule", "резервное правило -> rule")
check(api_server.DECISION_SOURCE_TEXT["image"] == "решение по изображению, измерения в норме — проверьте снимок визуально", "текст источника «по изображению»")
mf = api_server._model_features({"sp_pos_model_features": json.dumps({"variant": "canonical", "values": {"center_offset_ratio": 0.0512, "axis_angle_deg": -2.34}})}, "sp_pos")
vals = {i["key"]: i["value"] for i in mf["items"]}
check(mf["variant"] == "canonical" and vals.get("center_offset_ratio") == 5.1 and vals.get("axis_angle_deg") == -2.3, f"признаки модели из debug: {vals}")
check(api_server._model_features({}, "sp_pos") == {"variant": None, "items": []}, "нет поля -> пустой список")

# 6. feature_cols из файлов моделей
try:
    import joblib  # noqa: E402
    import inference  # noqa: E402
    pth = ROOT / "models" / "model_hip_roi_geom.pkl"
    if not pth.exists():
        print("SKIP models/model_hip_roi_geom.pkl нет")
    else:
        mb = inference.ModelBundle(joblib.load(pth), pth.name)
        fc = list(mb.meta.get("feature_cols") or [])
        check("scan_length_mm" in fc and "shaft_len_below_troch_mm" in fc, f"{pth.name}: feature_cols hip_roi = {fc}")
        js = json.loads(inference._model_features_json(mb, {"scan_length_mm": 200.0, "shaft_len_below_troch_mm": float("nan"),
                                                           "lateral_margin_mm": 5.0}, "canonical"))
        check(js["variant"] == "canonical" and list(js["values"]) == fc and js["values"]["scan_length_mm"] == 200.0
              and js["values"]["shaft_len_below_troch_mm"] is None,
              "_model_features_json: только признаки feature_cols, NaN -> None (модель подставит медиану)")
        check(inference._model_features_json(None, {}, "canonical") == "", "нет модели -> пустая строка")
except ImportError as e:
    print(f"SKIP inference недоступен: {e}")

# 7. ориентиры вместо «порогов ТЗ»
norms = cfg.get("measurement_norms", {})
check("tz_max" not in norms.get("axis_angle_deg", {}) and norms.get("axis_angle_deg", {}).get("guide_max") == 5.0, "угол оси: 5° — ориентир (guide_max), не tz_max")
check("tz_min" not in norms.get("edge_distance_mm", {}) and norms.get("edge_distance_mm", {}).get("guide_min") == 20.0, "край кадра: 20 мм — ориентир (guide_min), не tz_min")
aj = json.loads((ROOT / "web" / "assets" / "actions.json").read_text(encoding="utf-8"))
check(all(k in aj["actions"] for k in ("hip_roi_field_incomplete", "hip_roi_field_complete", "hip_roi_insufficient_data")),
      "web/assets/actions.json пересобран: маршруты hip_roi есть")

# 8. запрещённые слова
# список собирается из частей, чтобы сами слова не лежали в репозитории
bad = ("grad" + "-cam", "авто" + "коррекц", "сколи" + "оз", "коб" + "ба")
CITY = "ЕР" + "ИС"  # регистрозависимо: в нижнем регистре совпадает с частью обычных слов
raw = json.dumps([extras.STUDY_WARNING_TEXTS, extras.HIP_ROI_ROUTES, api_server.DECISION_SOURCE_TEXT,
                   api_server.MODEL_FEATURE_TITLES, aj], ensure_ascii=False)
blob = json.dumps([extras.STUDY_WARNING_TEXTS, extras.HIP_ROI_ROUTES, api_server.DECISION_SOURCE_TEXT,
                   api_server.MODEL_FEATURE_TITLES, aj], ensure_ascii=False).lower()
check(not any(b in blob for b in bad) and CITY not in raw, "тексты карточки без запрещённых слов")

print("\nALL CHECKS PASSED" if not fails else f"\nFAILED: {len(fails)}")
sys.exit(1 if fails else 0)
