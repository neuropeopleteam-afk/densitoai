"""Б2: правила решающего слоя (чистые функции, без зависимостей кроме numpy).

Используются nested-харнессом (tools/decision_layer_nested.py, tools/decision_layer_eval.py).
Обе гипотезы отвергнуты 24.09.2026 (docs/NESTED_GATE_REPORT.md, часть «Б2»), в инференс не подключены.

tie_midpoint_threshold — гипотеза (а): порог переносится в середину между соседними уникальными
    уровнями референсной выборки. Решение «score >= новый порог» на самом референсе совпадает со
    строгим «score > старый порог»: блок связанных рангов, на который попал квантиль, уходит в «норму».
    Сравнение с уровнем идёт с допуском TIE_EPS: тот же уровень, прочитанный из CSV парсером pandas
    по умолчанию (без float_precision='round_trip'), отличается на 1 ulp
    (sp_pos: 0.8614457831325302 против 0.8614457831325301).

exclusive_types — гипотеза (б): одно нарушение на строку. Если у строки >= 2 флагов, остаётся тип
    с наибольшим относительным запасом (score - thr) / (1 - thr); остальные флаги сохраняются, только
    если их запас >= tau (tau = inf — всегда один тип). OR флагов строки не меняется по построению:
    хотя бы один флаг всегда остаётся, поэтому quality_class и quality_prob те же.
"""
from typing import Dict, Iterable, List, Optional
import math

import numpy as np

TIE_EPS = 1e-9


def tie_midpoint_threshold(thr: float, ref: Optional[Iterable[float]], eps: float = TIE_EPS) -> float:
    """Середина между наибольшим уровнем референса <= thr (с допуском eps) и следующим уровнем выше."""
    if ref is None:
        return float(thr)
    r = np.asarray(ref, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float(thr)
    lv = np.unique(r)
    lo = lv[lv <= thr + eps]
    if lo.size == 0:            # порог ниже всех уровней: на референсе флагуется всё, правило не нужно
        return float(thr)
    lo = float(lo[-1])
    hi = lv[lv > lo + eps]
    hi = float(hi[0]) if hi.size else 1.0
    if hi <= lo:
        return float(thr)
    return 0.5 * (lo + hi)


def relative_margin(score: float, thr: float) -> float:
    return (float(score) - float(thr)) / max(1e-12, 1.0 - float(thr))


def exclusive_types(flags: Dict[str, int], scores: Dict[str, float], thresholds: Dict[str, float],
                    tau: float = math.inf) -> Dict[str, int]:
    """Одно нарушение на строку (см. модуль). Возвращает новый словарь флагов; вход не меняется."""
    on = [c for c, f in flags.items() if int(f) == 1 and scores.get(c) is not None]
    out = {c: int(f) for c, f in flags.items()}
    if len(on) < 2:
        return out
    marg = {c: relative_margin(scores[c], thresholds[c]) for c in on}
    order: List[str] = sorted(on, key=lambda c: (-marg[c], on.index(c)))   # при равенстве — порядок критериев
    for c in order[1:]:
        if not (marg[c] >= tau):
            out[c] = 0
    return out


def exclusive_types_array(F: np.ndarray, S: np.ndarray, T: np.ndarray, tau: float = math.inf) -> np.ndarray:
    """Векторный вариант для харнесса: F, S — (n, C) флаги и скоры, T — (n, C) или (C,) пороги."""
    F = np.asarray(F).astype(int).copy()
    S = np.asarray(S, dtype=np.float64)
    T = np.broadcast_to(np.asarray(T, dtype=np.float64), S.shape)
    M = (S - T) / np.maximum(1e-12, 1.0 - T)
    multi = F.sum(1) >= 2
    if not multi.any():
        return F
    Mm = np.where(F == 1, M, -np.inf)
    # argmax берёт первый максимум — тот же тай-брейк, что и порядок критериев в exclusive_types
    top = np.argmax(Mm, axis=1)
    keep = (F == 1) & (M >= tau)
    keep[np.arange(len(F)), top] = F[np.arange(len(F)), top] == 1
    F[multi] = keep[multi].astype(int)
    return F
