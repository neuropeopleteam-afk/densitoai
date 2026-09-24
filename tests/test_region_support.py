#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тесты проверки поддерживаемой области (слой API) и гарантии, что пакетный путь не затронут.
Запуск: python tests/test_region_support.py — зависимостей кроме стандартной библиотеки нет.
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

# 5. Главная гарантия: пакетный путь (CSV для организаторов) проверку области не вызывает,
#    поэтому числа поставки не могут измениться из-за неё.
src = (ROOT / "src" / "inference.py").read_text(encoding="utf-8")
check("region_support" not in src, "inference.py не импортирует region_support")
api = (ROOT / "src" / "api_server.py").read_text(encoding="utf-8")
check("region_support" in api, "api_server.py использует region_support")

# 6. Запрещённые формулировки не просочились в тексты отказов.
# собирается из частей, чтобы сами слова не лежали в репозитории
banned = ("Grad" + "-CAM", "авто" + "коррекция ROI", "ЕР" + "ИС", "сколи" + "оз", "Коб" + "ба")
rs = (ROOT / "src" / "region_support.py").read_text(encoding="utf-8")
check(not any(b.lower() in rs.lower() for b in banned), "в модуле нет запрещённых формулировок")

print(f"\nВсего проверок: {len(fails)} провалов")
sys.exit(1 if fails else 0)
