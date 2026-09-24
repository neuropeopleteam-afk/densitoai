#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тесты проверки поддерживаемой области (пакетный путь и API), маркера LATERAL и classify_region
при маркерах обеих областей. Запуск: python tests/test_region_support.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from region_support import check_tags  # noqa: E402

fails = []


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


# 1. То, что реально лежит в наших 499 файлах, должно проходить: тегов области нет,
#    ширина 248/280/300, высота 180–405.
real = {"SeriesDescription": "Изображения DXA", "ProtocolName": "Anonymized",
        "StudyDescription": "DXA Обследование", "Modality": "CR"}
for cols, rows in ((300, 405), (280, 300), (248, 180), (300, 180)):
    ok, reason = check_tags(real, cols=cols, rows=rows)
    check(ok, f"наш формат проходит ({cols}x{rows}) {reason}")
ok, _ = check_tags({"SeriesDescription": "DXA Images", "Modality": "CR"}, cols=248, rows=200)
check(ok, "англоязычное описание проходит")
ok, _ = check_tags({}, cols=300, rows=300)
check(ok, "пустые теги — не повод для отказа")
ok, _ = check_tags({"Modality": "CR"}, cols=0, rows=0)
check(ok, "неизвестная геометрия — не повод для отказа")

# 2. Область, для которой сервис не предназначен.
for tags, what in (
    ({"BodyPartExamined": "FOREARM", "Modality": "CR"}, "предплечье в BodyPartExamined"),
    ({"SeriesDescription": "Forearm DXA", "Modality": "CR"}, "предплечье в описании серии"),
    ({"StudyDescription": "Предплечье", "Modality": "CR"}, "предплечье по-русски"),
    ({"SeriesDescription": "Whole Body Composition", "Modality": "CR"}, "всё тело"),
    ({"ProtocolName": "Total Body", "Modality": "CR"}, "всё тело в протоколе"),
    ({"StudyDescription": "Состав тела", "Modality": "CR"}, "состав тела по-русски"),
    ({"ProtocolName": "Lateral Spine VFA", "Modality": "CR"}, "боковая проекция"),
    ({"SeriesDescription": "DVA", "Modality": "CR"}, "морфометрия позвонков"),
    ({"StudyDescription": "Колено", "Modality": "CR"}, "коленный сустав"),
    ({"BodyPartExamined": "CALCANEUS", "Modality": "CR"}, "пяточная кость"),
    ({"BodyPartExamined": "WRIST", "Modality": "CR"}, "запястье"),
):
    ok, reason = check_tags(tags, cols=300, rows=300)
    check((not ok) and "сервис оценивает только" in reason, f"отказ: {what}")

# 3. Другая модальность.
for mod in ("MR", "CT", "US", "PT"):
    ok, reason = check_tags({"Modality": mod}, cols=300, rows=300)
    check(not ok, f"отказ по модальности {mod}")
for mod in ("CR", "OT", "DX", "RG", "SC"):
    ok, _ = check_tags({"Modality": mod}, cols=300, rows=300)
    check(ok, f"модальность {mod} допускается")

# 4. Геометрия кадра вне наблюдаемого диапазона.
for cols, rows, what in ((1024, 1024, "снимок 1024x1024"), (2048, 2500, "большой рентген"),
                         (150, 300, "слишком узкий кадр"), (300, 60, "слишком низкий кадр"),
                         (300, 900, "слишком высокий кадр")):
    ok, reason = check_tags({"Modality": "CR"}, cols=cols, rows=rows)
    check(not ok, f"отказ по геометрии: {what}")
# и наоборот — нестандартная, но близкая геометрия не отсекается (другой экспорт того же аппарата)
for cols, rows in ((256, 320), (320, 420), (240, 200)):
    ok, reason = check_tags({"Modality": "CR"}, cols=cols, rows=rows)
    check(ok, f"близкая геометрия {cols}x{rows} проходит")

# 4б. Аппарат вне области применения (docs/EXTERNAL_DXA.md): отказ только при явно чужом производителе.
for tags, what in (({"Manufacturer": "HOLOGIC", "ManufacturerModelName": "Horizon A"}, "Hologic"),
                   ({"Manufacturer": "Norland"}, "Norland"), ({"Manufacturer": "external PNG", "Modality": "OT"}, "PNG-обёртка")):
    ok, reason = check_tags(tags, cols=300, rows=300)
    check((not ok) and "вне области применения" in reason, f"отказ по аппарату: {what}")
for tags, what in (({"Manufacturer": "GE Healthcare", "ManufacturerModelName": "Lunar Prodigy Advance", "Modality": "CR"}, "наш аппарат (все 499)"),
                   ({"Manufacturer": "GE MEDICAL SYSTEMS"}, "GE другим написанием"), ({"Manufacturer": "GE"}, "GE"),
                   ({"ManufacturerModelName": "Lunar iDXA"}, "Lunar без производителя"),
                   ({"Manufacturer": ""}, "пустой производитель"), ({"Manufacturer": "Anonymized"}, "обезличенный производитель"), ({}, "нет тега")):
    ok, reason = check_tags(tags, cols=300, rows=300)
    check(ok, f"аппарат допускается: {what} {reason}")

# 5. Пакетный путь (CLI и API, DensitoInference.process_file) вызывает фильтр до классификации
#    (A1, 24.09.2026): отказ — строка Failure; на формате 499 файлов заказчика правило не срабатывает
#    (раздел 1; прогон 499 сверяется compare_regressions). Сквозной CLI-тест — tests/test_output_contract.py.
src = (ROOT / "src" / "inference.py").read_text(encoding="utf-8")
check("from region_support import check_tags" in src and "region_support_tags_of(info)" in src
      and "region_support_geometry_of(info)" in src,
      "inference.py вызывает region_support в process_file (теги — отказ, геометрия — отметка в debug)")
api = (ROOT / "src" / "api_server.py").read_text(encoding="utf-8")
check("region_support" in api, "api_server.py использует region_support")

# 5б. «LATERAL» — только отдельным словом: BILATERAL / CONTRALATERAL не отказ.
for text, exp, what in (("DualFemur BILATERAL", True, "BILATERAL"), ("Contralateral hip", True, "CONTRALATERAL"),
                        ("BILATERAL_HIP", True, "BILATERAL_HIP"), ("LATERAL", False, "LATERAL"),
                        ("Lateral Spine", False, "Lateral Spine"), ("L-SPINE_LATERAL", False, "_LATERAL"),
                        ("AP/LATERAL", False, "AP/LATERAL"), ("Боковая проекция", False, "боковая по-русски")):
    ok, reason = check_tags({"SeriesDescription": text, "Modality": "CR"}, cols=280, rows=300)
    check(ok == exp, f"проекция «{text}»: {'проходит' if exp else 'отказ'} ({what}) {reason[:40]}")

# 5в. classify_region: маркеры обеих областей в описании -> решает ширина кадра (300 / 280 / 248).
import numpy as np  # noqa: E402
import inference as inf  # noqa: E402

cfg = inf.load_config()


def _info(cols, desc, lat=""):
    img = np.zeros((300, cols), np.uint8)
    img[40:260, cols // 2 - 20:cols // 2 + 20] = 200
    return inf.DicomInfo(ds=None, img_u8=img, rows=300, cols=cols, study_uid="1", image_uid="1",
                         pixel_spacing=(1.0, 1.0), tags={"SeriesDescription": desc, "Laterality": lat})


for cols, desc, lat, exp in ((300, "SPINE + HIP", "", "spine"), (280, "SPINE + HIP", "R", "right_hip"),
                             (248, "Позвоночник и бедро", "L", "left_hip"), (280, "L-SPINE / DualFemur HIP", "L", "left_hip"),
                             (300, "Lumbar spine", "", "spine"), (280, "Lumbar spine", "", "spine"),
                             (300, "HIP", "R", "right_hip")):
    reg, how = inf.classify_region(_info(cols, desc, lat), Path("x.dcm"), cfg)
    check(reg == exp, f"classify_region({cols} px, «{desc}») -> {reg} ({how}), ожидается {exp}")
reg, how = inf.classify_region(_info(280, "SPINE HIP", ""), Path("x.dcm"), cfg)
check(reg in ("right_hip", "left_hip") and how.startswith("dims"), f"обе области, 280 px -> бедро по ширине ({how})")

# 6. Запрещённые формулировки не просочились в тексты отказов.
# собирается из частей, чтобы сами слова не лежали в репозитории
banned = ("Grad" + "-CAM", "авто" + "коррекция ROI", "ЕР" + "ИС", "сколи" + "оз", "Коб" + "ба")
rs = (ROOT / "src" / "region_support.py").read_text(encoding="utf-8")
check(not any(b.lower() in rs.lower() for b in banned), "в модуле нет запрещённых формулировок")

print(f"\nВсего проверок: {len(fails)} провалов")
sys.exit(1 if fails else 0)
