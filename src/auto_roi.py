#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI-автокоррекция (бонус ТЗ п.2.6, портирование v1 auto_roi.py в архитектуру
densito_rebuild).

Что делает v1 auto_roi.py (по описанию в HANDOVER): для 499 изображений
предлагал скорректированный прямоугольник ROI. В v1 это было эвристикой на
резком CNN-масштабе без физических единиц. Здесь делаем то же самое, но
опираясь на РЕАЛЬНЫЕ физические измерения Контура A (пиксель 1.05x0.6 мм,
подтверждён организаторами письменно, см. Raziasneniia-po-voprosam_V2.docx)
и клинический критерий ТЗ (ROI >= 2 см от края кадра, диафиз должен быть
виден минимум ~3 см ниже большого/малого вертела).

Два разных сценария коррекции, в зависимости от найденной причины
нарушения (см. hip_features.py, докстринг, "Наблюдение по данным"):

  1. ROI обрезана по краю кадра (lateral_margin_mm < 20 мм) — предлагаем
     новый bounding box, сдвинутый так, чтобы кость была центрирована с
     запасом >= 20 мм от каждого края (в пределах кадра — т.е. это
     "как СЛЕДОВАЛО БЫ снять", а не постобработка существующих пикселей,
     потому что если бедро физически обрезано кадром, недостающие пиксели
     отсутствуют и не могут быть восстановлены).
  2. Скан слишком короткий (shaft_len_below_troch_mm < 30 мм) — это
     ГЛАВНАЯ реальная причина rh_roi/lh_roi в данных: оператор остановил
     скан слишком рано. Автокоррекция здесь = явно вычисленный "дефицит"
     скана в мм/пикселях и предлагаемая рамка, которая показывает, СКОЛЬКО
     дополнительного кадра ниже текущего края нужно было бы захватить.
     Поскольку кадр обрезан физически, предложение — диагностическое (для
     эксперта/технолога), а не восстановление пикселей.

Для позвоночника (spine) автокоррекция не применяется — там нет понятия
"ROI-рамка" в ТЗ (критерий "ось" — это угол, не площадь).

Функция:
  suggest_hip_roi(img_u8, feats=None) -> dict с полями:
      needs_correction: bool
      reason: str | None
      suggested_box_px: (x0, y0, x1, y1) | None  — координаты в ИСХОДНОМ
          (не зеркальном) кадре
      deficit_mm: float | None  — на сколько мм не хватает диафиза/отступа
      note: str — человекочитаемое объяснение для README/эксперта
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from hip_features import (
    PIXEL_SPACING_X_MM, PIXEL_SPACING_Y_MM,
    segment_bone_hip, detect_hip_side, track_femur, hip_features_canonical,
)

ROI_LATERAL_MARGIN_MIN_MM = 20.0   # ТЗ: >= 2 см от края кадра (формальный порог, не основной сигнал в данных)
# Порог по shaft_len_below_troch_mm откалиброван на реальном распределении:
# 11 rh_roi/lh_roi позитивов имеют медиану 65мм (диапазон 57-82мм) против
# медианы 89мм у негативов (диапазон 28-183мм) — граница на 25-й перцентиль
# позитивов даёт разумный компромисс чувствительность/специфичность.
ROI_SHAFT_BELOW_TROCH_MIN_MM = 75.0
# scan_length_mm — более сильный и физически прямой сигнал того же дефекта
# ("скан слишком короткий"): позитивы медиана 217мм (189-274), негативы
# медиана 278мм (217-366) — почти без пересечения ниже 220мм.
ROI_SCAN_LENGTH_MIN_MM = 220.0


def suggest_hip_roi(img_u8: np.ndarray, feats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    h, w = img_u8.shape
    mask = segment_bone_hip(img_u8)
    side = detect_hip_side(img_u8, mask)
    mirrored = side != "right"
    mask_c = mask if not mirrored else np.ascontiguousarray(mask[:, ::-1])

    f = feats if feats is not None else hip_features_canonical(mask_c)
    tr = track_femur(mask_c)

    lateral_margin_mm = f.get("lateral_margin_mm")
    shaft_len_mm = f.get("shaft_len_below_troch_mm")
    scan_length_mm = f.get("scan_length_mm")

    out: Dict[str, Any] = {
        "needs_correction": False, "reason": None, "suggested_box_px": None,
        "deficit_mm": None, "note": "", "side_detected": side,
    }

    lateral_bad = lateral_margin_mm is not None and lateral_margin_mm < ROI_LATERAL_MARGIN_MIN_MM
    scan_short = scan_length_mm is not None and scan_length_mm < ROI_SCAN_LENGTH_MIN_MM
    shaft_bad = shaft_len_mm is not None and shaft_len_mm < ROI_SHAFT_BELOW_TROCH_MIN_MM

    if not lateral_bad and not scan_short and not shaft_bad:
        out["note"] = "ROI в норме: отступ от края, длина скана и диафиз ниже вертела соответствуют откалиброванным порогам."
        return out

    ys_nonzero, xs_nonzero = np.nonzero(mask)
    if len(ys_nonzero) < 5:
        out["note"] = "Недостаточно пикселей кости для расчёта коррекции ROI."
        return out
    x0_bone, x1_bone = int(xs_nonzero.min()), int(xs_nonzero.max())
    y0_bone, y1_bone = int(ys_nonzero.min()), int(ys_nonzero.max())

    # Приоритет: scan_too_short (самый сильный сигнал на реальных данных),
    # затем shaft_below_trochanter (дублирующий тот же дефект по-другому измерению),
    # затем lateral_margin (формальный критерий ТЗ, редко срабатывает сам по себе на этих данных).
    if scan_short:
        deficit_mm = ROI_SCAN_LENGTH_MIN_MM - scan_length_mm
        deficit_px = int(np.ceil(deficit_mm / PIXEL_SPACING_Y_MM))
        sy1_extended = y1_bone + deficit_px
        out.update({
            "needs_correction": True,
            "reason": "scan_too_short",
            "suggested_box_px": (x0_bone, y0_bone, x1_bone, y1_bone),
            "extended_box_px": (x0_bone, y0_bone, x1_bone, sy1_extended),
            "deficit_mm": round(float(deficit_mm), 1),
            "note": (f"Физическая длина скана {scan_length_mm:.0f} мм < откалиброванного порога "
                     f"{ROI_SCAN_LENGTH_MIN_MM:.0f} мм (не хватает {deficit_mm:.0f} мм = {deficit_px}px). "
                     f"Это основная причина нарушения ROI на бедре в обучающих данных "
                     f"(11/11 позитивов имеют scan_length_mm < 275мм) — оператор остановил скан слишком рано; "
                     f"диагностика, не постобработка (пикселей за пределами кадра физически нет)."),
        })
        return out

    if shaft_bad:
        deficit_mm = ROI_SHAFT_BELOW_TROCH_MIN_MM - shaft_len_mm
        deficit_px = int(np.ceil(deficit_mm / PIXEL_SPACING_Y_MM))
        sy1_extended = y1_bone + deficit_px
        out.update({
            "needs_correction": True,
            "reason": "shaft_below_trochanter_too_short",
            "suggested_box_px": (x0_bone, y0_bone, x1_bone, min(h - 1, y1_bone)),
            "extended_box_px": (x0_bone, y0_bone, x1_bone, sy1_extended),
            "deficit_mm": round(float(deficit_mm), 1),
            "note": (f"Диафиз ниже вертела в кадре {shaft_len_mm:.0f} мм < порога "
                     f"{ROI_SHAFT_BELOW_TROCH_MIN_MM:.0f} мм (не хватает {deficit_mm:.0f} мм = "
                     f"{deficit_px}px по вертикали). Скан остановлен слишком рано — "
                     f"вторичный признак того же дефекта, что и короткий скан; "
                     f"диагностика, не постобработка."),
        })
        return out

    # lateral_bad как единственное срабатывание (формальный критерий ТЗ, без данных об эффективности)
    deficit_mm = ROI_LATERAL_MARGIN_MIN_MM - lateral_margin_mm
    deficit_px = int(np.ceil(deficit_mm / PIXEL_SPACING_X_MM))
    margin_px = int(round(ROI_LATERAL_MARGIN_MIN_MM / PIXEL_SPACING_X_MM))
    sx0 = max(0, x0_bone - margin_px)
    sx1 = min(w - 1, x1_bone + margin_px)
    out.update({
        "needs_correction": True,
        "reason": "lateral_margin_below_threshold",
        "suggested_box_px": (sx0, y0_bone, sx1, y1_bone),
        "deficit_mm": round(float(deficit_mm), 1),
        "note": (f"Отступ ROI от края кадра {lateral_margin_mm:.0f} мм < формального порога ТЗ "
                 f"{ROI_LATERAL_MARGIN_MIN_MM:.0f} мм (не хватает {deficit_mm:.0f} мм = {deficit_px}px). "
                 f"Корректная рамка ROI показана с отступом {ROI_LATERAL_MARGIN_MIN_MM:.0f} мм; "
                 f"если кадр физически обрезан — требуется повторная укладка/скан."),
    })
    return out


def draw_roi_correction(img_u8: np.ndarray, suggestion: Dict[str, Any]) -> np.ndarray:
    """Визуализация предложенной коррекции: жёлтая рамка — предложенный ROI,
    красная (при выходе за кадр) — диагностическая зона недостающего скана."""
    import cv2
    col = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    h, w = img_u8.shape
    box = suggestion.get("suggested_box_px")
    if box:
        x0, y0, x1, y1 = box
        cv2.rectangle(col, (x0, y0), (x1, y1), (0, 220, 220), 2)
    ext = suggestion.get("extended_box_px")
    if ext:
        x0, y0, x1, y1 = ext
        y1_visible = min(y1, h - 1)
        if y1 > h - 1:
            # Полоса-плашка красная (диагностическая зона недостающего скана);
            # текст поверх неё должен контрастировать с красным, поэтому белый
            # с чёрной обводкой — красный текст на красном фоне был нечитаем.
            label = f"+{suggestion['deficit_mm']:.0f}мм"
            bar_h = 14
            y_bar0 = max(0, h - bar_h)
            cv2.rectangle(col, (x0, y_bar0), (x1, h - 1), (0, 0, 255), -1)
            text_y = h - 4
            cv2.putText(col, label, (x0 + 2, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(col, label, (x0 + 2, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return col


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from geometry_features import read_dicom_normalized

    fp = sys.argv[1] if len(sys.argv) > 1 else None
    if fp:
        img_u8, ds = read_dicom_normalized(fp)
        result = suggest_hip_roi(img_u8)
        print(result)
        if result["needs_correction"]:
            import cv2
            out = draw_roi_correction(img_u8, result)
            cv2.imwrite("/tmp/roi_correction_test.png", out)
            print("saved overlay -> /tmp/roi_correction_test.png")
