# -*- coding: utf-8 -*-
"""Проверка поддерживаемой области исследования — только для слоя API и веба.

Пакетный путь (CLI, CSV для организаторов) этот модуль НЕ вызывает: формат, порядок колонок и
числа поставки не меняются по построению.

Зачем. `classify_region` в `inference.py` по построению всегда возвращает одну из трёх областей
(spine / right_hip / left_hip): широкий кадр считается позвоночником, узкий — бедром. Для
закрытого теста это верно, туда приходят только поясничный отдел и бедро. Но если пользователь
загрузит предплечье, «всё тело», боковую проекцию или снимок другой модальности, сервис выдаст
уверенный вердикт там, где он не обучался. Этот модуль отвечает на один вопрос: похоже ли
исследование на то, для которого сервис предназначен.

Правило консервативное: отказываем только при явном признаке, сомнение трактуем в пользу
обработки. Основание — что реально лежит в наших 499 файлах (измерено 22.09.2026):
  * Modality='CR', Manufacturer='GE Healthcare', Model='Lunar Prodigy Advance' — все 499;
  * BodyPartExamined пустой во всех 499, SeriesDescription — 'Изображения DXA' / 'DXA Images';
  * ширина кадра принимает только значения 248 / 280 / 300 px, высота 180–405 px.
То есть на поддерживаемых исследованиях тегов области может не быть вовсе, и отсутствие
тега никогда не является поводом для отказа.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

# Область исследования, для которой сервис не предназначен. Срабатывает независимо от прочих тегов.
UNSUPPORTED_AREA = (
    ("FOREARM", "предплечье"), ("RADIUS", "предплечье"), ("ULNA", "предплечье"),
    ("WRIST", "запястье"), ("ПРЕДПЛЕЧ", "предплечье"), ("ЛУЧЕВ", "предплечье"),
    ("ЗАПЯСТ", "запястье"), ("КИСТ", "кисть"), ("HAND", "кисть"),
    ("WHOLE BODY", "всё тело"), ("WHOLEBODY", "всё тело"), ("TOTAL BODY", "всё тело"),
    ("TOTALBODY", "всё тело"), ("BODY COMPOSITION", "состав тела"),
    ("СОСТАВ ТЕЛА", "состав тела"), ("ВСЕ ТЕЛО", "всё тело"), ("ВСЁ ТЕЛО", "всё тело"),
    ("KNEE", "коленный сустав"), ("КОЛЕН", "коленный сустав"),
    ("TIBIA", "голень"), ("ГОЛЕН", "голень"),
    ("CALCANEUS", "пяточная кость"), ("ПЯТОЧ", "пяточная кость"), ("HEEL", "пяточная кость"),
    ("ANKLE", "голеностоп"), ("SHOULDER", "плечевой сустав"), ("ПЛЕЧ", "плечевой сустав"),
    ("SKULL", "череп"), ("ЧЕРЕП", "череп"),
)

# Проекция, для которой сервис не предназначен: оценивались только прямые проекции
# поясничного отдела и проксимального отдела бедра.
UNSUPPORTED_PROJECTION = (
    ("LATERAL", "боковая проекция"), ("БОКОВ", "боковая проекция"),
    ("VFA", "морфометрия позвонков"), ("DVA", "морфометрия позвонков"),
    ("LVA", "морфометрия позвонков"),
)

ALLOWED_MODALITY = ("CR", "OT", "DX", "RG", "SC", "")

TEXT_TAGS = ("BodyPartExamined", "SeriesDescription", "ProtocolName", "StudyDescription")

# Аппарат вне области применения (24.09.2026, docs/EXTERNAL_DXA.md): на 904 PNG чужих аппаратов сервис без этого
# правила не отказывал ни разу и давал 70–92 % «нарушений» с quality_prob ~0,93–1,00. Отказ — только если тег
# Manufacturer ЗАПОЛНЕН и явно не GE/Lunar; пустой или обезличенный тег — не повод для отказа (сомнение — в пользу
# обработки). Все 499 файлов заказчика: Manufacturer = 'GE Healthcare' — правило на них не срабатывает.
SUPPORTED_VENDOR_MARKERS = ("GE ", "GE_", "GEHC", "GE HEALTHCARE", "GENERAL ELECTRIC", "LUNAR")
VENDOR_UNKNOWN = ("", "ANONYMIZED", "ANONYMOUS", "UNKNOWN", "NONE", "N/A", "-")

# Границы геометрии кадра. Наблюдаемое в наших данных: ширина 248–300, высота 180–405.
# Отказ — только при явном выходе за границы, чтобы не отсекать другой экспорт того же аппарата.
COLS_MIN, COLS_MAX = 200, 400
ROWS_MIN, ROWS_MAX = 120, 600

SUPPORTED_SCOPE = ("сервис оценивает только прямые проекции поясничного отдела позвоночника "
                   "и проксимального отдела бедра")


def _text_of(tags: Dict[str, Any]) -> str:
    low = {str(k).lower(): v for k, v in (tags or {}).items()}
    parts = []
    for t in TEXT_TAGS:
        v = low.get(t.lower())
        if v:
            parts.append(str(v))
    return " ".join(parts).upper()


def check_tags(tags: Dict[str, Any], cols: int = 0, rows: int = 0) -> Tuple[bool, str]:
    """Возвращает (поддерживается, причина). Причина заполнена только при отказе."""
    low = {str(k).lower(): v for k, v in (tags or {}).items()}
    text = _text_of(tags)

    for marker, human in UNSUPPORTED_AREA:
        if marker in text:
            return False, (f"в описании исследования указано «{marker}» ({human}); {SUPPORTED_SCOPE}")

    for marker, human in UNSUPPORTED_PROJECTION:
        if marker in text:
            return False, (f"в описании исследования указано «{marker}» ({human}); {SUPPORTED_SCOPE}")

    mod = str(low.get("modality") or "").strip().upper()
    if mod and mod not in ALLOWED_MODALITY:
        return False, (f"модальность снимка {mod} не соответствует рентгеновской "
                       f"денситометрии; {SUPPORTED_SCOPE}")

    vendor = " ".join(str(low.get(k) or "") for k in ("manufacturer", "manufacturermodelname")).strip().upper()
    if vendor not in VENDOR_UNKNOWN and vendor != "GE" and not any(m in vendor + " " for m in SUPPORTED_VENDOR_MARKERS):
        shown = " ".join(str(low.get(k) or "") for k in ("manufacturer", "manufacturermodelname")).strip()
        return False, (f"аппарат «{shown}» вне области применения: сервис настроен на GE Lunar Prodigy, "
                       f"на снимках других аппаратов и форматов результат не определён; {SUPPORTED_SCOPE}")

    try:
        c = int(cols or low.get("columns") or 0)
    except (TypeError, ValueError):
        c = 0
    try:
        r = int(rows or low.get("rows") or 0)
    except (TypeError, ValueError):
        r = 0
    if c and not (COLS_MIN <= c <= COLS_MAX):
        return False, (f"ширина кадра {c} px вне диапазона поддерживаемых исследований "
                       f"(наблюдаемые значения 248–300 px); {SUPPORTED_SCOPE}")
    if r and not (ROWS_MIN <= r <= ROWS_MAX):
        return False, (f"высота кадра {r} px вне диапазона поддерживаемых исследований "
                       f"(наблюдаемые значения 180–405 px); {SUPPORTED_SCOPE}")
    return True, ""


def check_file(path: Path, rows: int = 0, cols: int = 0) -> Tuple[bool, str]:
    """То же по файлу: читает только заголовок DICOM. При любой ошибке чтения возвращает
    (True, "") — решение об ошибке принимает сам движок, отказ по области здесь не выдумываем."""
    tags: Dict[str, Any] = {}
    try:
        import pydicom  # локальный импорт: модуль пригоден и без pydicom

        ds = pydicom.dcmread(str(path), force=True, stop_before_pixels=True)
        for t in TEXT_TAGS + ("Modality", "Rows", "Columns", "Manufacturer", "ManufacturerModelName"):
            v = getattr(ds, t, None)
            if v not in (None, ""):
                tags[t] = v
    except Exception:  # noqa: BLE001 — отсутствие тегов не повод для отказа
        pass
    if not cols:
        cols = int(tags.get("Columns") or 0)
    if not rows:
        rows = int(tags.get("Rows") or 0)
    return check_tags(tags, cols=cols, rows=rows)
