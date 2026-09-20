"""Предобработка, инвариантная к экспозиции и шуму (контуры A и B используют одно и то же).

Зачем: `docs/ROBUSTNESS_REPORT.md` показал, что решения переворачиваются при гамме
(25.9 / 39.5 % кадров) и гауссовом шуме σ=3 % (28.4 %), тогда как к сдвигам формата
(теги, MONOCHROME1, битность, UID, упаковка) система устойчива полностью. Две причины:

1. **Шум.** Маска тела бралась жёстким порогом по сырому кадру (`img_u8 > 8`). При
   σ=3 % (≈8 уровней из 255) воздух попадает в «тело», Otsu по телу смещается, порог
   кости уезжает. Решение — считать маску по сглаженному кадру (`GaussianBlur(5,5) > 8`):
   на чистых кадрах разница 0.3–0.7 % пикселей, при шуме воздух больше не протекает.

2. **Гамма.** Перцентильное окно 1–99 % компенсирует только линейные изменения
   экспозиции; порог Otsu и признаки интенсивности к степенному преобразованию не
   инвариантны. Решение — канонизация экспозиции: по квантилям пикселей тела
   оценивается один параметр γ, приводящий кадр к эталонному распределению
   (`models/exposure_reference.json`, медиана по 499 кадрам обучения), и применяется
   `x -> x^γ`.

   Почему это работает точно, а не приблизительно: монотонное преобразование
   действует на квантили так же, как на пиксели. Для чистого кадра с квантилями `q`
   оценка даёт `γ0`: `q^γ0 = ref`. Для того же кадра, искажённого гаммой `g`
   (квантили `q^g`), оценка даёт `γ1 = γ0 / g`, и после коррекции получается
   `(x^g)^(γ0/g) = x^γ0` — тот же кадр, что и из чистого входа. То есть коррекция
   убирает любое степенное искажение экспозиции, а не подгоняет его под порог.

Эталон строится по обучающей выборке, поэтому на ней γ ≈ 1 и канонизация почти не
меняет вход (медиана |γ − 1| = 0.03); на «чужой» экспозиции она возвращает кадр в
знакомый диапазон. Диагностика (`γ`, доля тела) уезжает в extras как
`exposure_gamma` — именно она объясняет врачу, почему кадр признан нестандартным.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

BODY_THRESHOLD = 8
BODY_BLUR = (5, 5)
QUANTILE_PROBS = (10.0, 25.0, 50.0, 75.0, 90.0)
GAMMA_CLIP = (0.5, 2.0)
# Оценка γ округляется до сетки: иначе микроскопическая разница входа
# (например, те же пиксели в 12 битах вместо 8) даёт другой LUT и лишний разброс решений.
GAMMA_GRID = 0.02

_ROOT = Path(__file__).resolve().parent.parent
_REF_PATH = _ROOT / "models" / "exposure_reference.json"
_REF_CACHE: Optional[np.ndarray] = None

# Варианты предобработки. ПО УМОЛЧАНИЮ активен `baseline` — бит в бит поведение 2.1.0.
# Кто хочет другой вариант — вызывает set_variant() явно (так делают extract_all_features.py,
# embeddings.py, tools/preproc_gate.py). Инференс не использует эти флаги: он считает оба варианта
# кадра и берёт по критерию тот, который выбрал nested (config.yaml: preprocessing.variant_by_criterion).
VARIANTS = {
    "baseline":  {"noise_robust_body_mask": False, "exposure_canonicalization": False},
    "mask":      {"noise_robust_body_mask": True,  "exposure_canonicalization": False},
    "canonical": {"noise_robust_body_mask": True,  "exposure_canonicalization": True},
}
# ContextVar, а не глобальный dict: FastAPI выполняет синхронные ручки в пуле потоков,
# и вариант предобработки одного запроса не должен протекать в соседний.
_VARIANT: contextvars.ContextVar[str] = contextvars.ContextVar("densito_preproc_variant", default="baseline")


def current_variant() -> str:
    return _VARIANT.get()


def flags() -> dict:
    """Флаги текущего варианта (копия; менять вариант — через set_variant / variant())."""
    return dict(VARIANTS[_VARIANT.get()])


def set_variant(name: str) -> str:
    """Установить вариант предобработки до конца контекста (для скриптов)."""
    if name not in VARIANTS:
        raise ValueError(f"неизвестный вариант предобработки: {name} (есть {list(VARIANTS)})")
    _VARIANT.set(name)
    return name


@contextlib.contextmanager
def variant(name: str) -> Iterator[str]:
    """Временно переключить вариант предобработки (инференс считает оба варианта)."""
    if name not in VARIANTS:
        raise ValueError(f"неизвестный вариант предобработки: {name} (есть {list(VARIANTS)})")
    token = _VARIANT.set(name)
    try:
        yield name
    finally:
        _VARIANT.reset(token)


def load_reference() -> Optional[np.ndarray]:
    """Эталонные квантили пикселей тела в [0,1] (None — файла нет, канонизация выключена)."""
    global _REF_CACHE
    if not VARIANTS[_VARIANT.get()]["exposure_canonicalization"]:
        return None
    if _REF_CACHE is None:
        if not _REF_PATH.exists():
            return None
        payload = json.loads(_REF_PATH.read_text())
        probs = tuple(float(p) for p in payload["quantile_probs"])
        if probs != QUANTILE_PROBS:
            raise ValueError(f"exposure_reference.json: quantile_probs {probs} != {QUANTILE_PROBS}")
        _REF_CACHE = np.asarray(payload["reference"], dtype=np.float64)
    return _REF_CACHE


def body_mask(img_u8: np.ndarray, threshold: int = BODY_THRESHOLD) -> np.ndarray:
    """Маска тела (не воздух), устойчивая к шуму: порог по сглаженному кадру."""
    if not VARIANTS[_VARIANT.get()]["noise_robust_body_mask"]:
        return img_u8 > threshold
    blurred = cv2.GaussianBlur(img_u8, BODY_BLUR, 0)
    return blurred > threshold


def body_quantiles(img_u8: np.ndarray) -> Optional[np.ndarray]:
    """Квантили пикселей тела в [0,1]; None — тела слишком мало."""
    mask = body_mask(img_u8)
    if int(mask.sum()) < 100:
        return None
    px = img_u8[mask].astype(np.float64) / 255.0
    return np.percentile(px, QUANTILE_PROBS)


def estimate_gamma(img_u8: np.ndarray, reference: Optional[np.ndarray] = None) -> float:
    """Оценка γ, приводящего квантили тела к эталону (1.0 — канонизация не нужна)."""
    ref = load_reference() if reference is None else np.asarray(reference, dtype=np.float64)
    if ref is None:
        return 1.0
    q = body_quantiles(img_u8)
    if q is None:
        return 1.0
    ok = (q > 0.02) & (q < 0.98) & (ref > 0.02) & (ref < 0.98)
    if int(ok.sum()) < 2:
        return 1.0
    gammas = np.log(ref[ok]) / np.log(q[ok])
    gamma = float(np.median(gammas))
    if not np.isfinite(gamma):
        return 1.0
    gamma = float(np.clip(gamma, *GAMMA_CLIP))
    return float(round(gamma / GAMMA_GRID) * GAMMA_GRID)


def apply_gamma(img_u8: np.ndarray, gamma: float) -> np.ndarray:
    """x -> x^γ через LUT (быстро, детерминированно)."""
    if abs(gamma - 1.0) < 1e-3:
        return img_u8
    lut = np.clip(np.round(255.0 * (np.arange(256) / 255.0) ** gamma), 0, 255).astype(np.uint8)
    return cv2.LUT(img_u8, lut)


def canonicalize_exposure(img_u8: np.ndarray,
                          reference: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """Канонизация экспозиции. Возвращает (кадр, применённая γ)."""
    gamma = estimate_gamma(img_u8, reference)
    return apply_gamma(img_u8, gamma), gamma


def canonical_frame(img_u8: np.ndarray) -> Tuple[np.ndarray, float]:
    """Канонизированный кадр независимо от текущего варианта (для инференса,
    который всегда готовит оба варианта кадра)."""
    with variant("canonical"):
        return canonicalize_exposure(img_u8)
