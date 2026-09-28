"""2.5: сертификат действия и роли критериев — объяснение команды лаборанту, отдельное от девяти колонок.

Флаг критерия и команда «Переснять» — разные вещи (консилиум 27.09, GPT 6 Sol, решение A). Флаг остаётся в
выгрузке как есть; команда «Переснять» даётся, только если у флага есть измеримое основание. Иначе карточка,
ответ /api/analyze, results_extras.csv и SR говорят «Проверить …» и показывают основание числом.

Правила (пороги заданы до расчёта качества, без меток):
  * укладка бедра (rh_pos / lh_pos): боковой запас поля lateral_margin_mm больше HIP_WIDE_FIELD_MM —
    «поле шире обычного»; на таком поле оценка укладки менее надёжна (консилиум 27.09, Claude Opus 5.5, Т2-2: доля
    ложных тревог у норм 0.12 при запасе до 45 мм, 0.23 при 45–55 мм, 0.30 при запасе больше 55 мм). Порог 51 мм —
    верхняя терциль бокового запаса на 333 снимках бедра обучающей выборки (распределение без меток: терцили
    41.2 и 51.0 мм); в постановке 2.5 указан диапазон 51–55 мм, чувствительность к 55 мм — в отчёте.
  * посторонние предметы (sp_art): плотные участки есть, но вся их площадь вне зоны измерения — в верхних 70 %
    протяжённости кости площадь 0 мм² (признак metal_metal_band70_area_mm2 из src/geometry_features.py, тот же, по
    которому с версии 2.5.0 решает контур A sp_art, поэтому объяснение совпадает с решением модели). Ниже полосы —
    крылья подвздошных костей, зона измерения L1–L4 выше.
  * посторонние предметы, детектор не нашёл ни одного плотного участка: измеримого основания нет,
    флаг поставлен по изображению — «Проверить».
  * укладка бедра без измеренного бокового запаса — «Проверить».
Для остальных критериев (sp_pos, sp_axis, hip_roi) правило основания не задано: команда как в 2.4.1.

Роли критериев (консилиум 27.09, Claude Opus 5.5, Р4-2): укладка позвоночника, ось и область интереса бедра —
«измерение» (значение, единицы, ориентир); посторонние предметы и укладка бедра — «подсказка, решает врач».
Роль меняет только подпись в карточке; класс, флаги и пороги не меняются.

Модель, веса, пороги, quality_class и 9 колонок этот модуль не трогает (проверка — прогон 499 файлов).
"""
from typing import Any, Dict, Optional

EVIDENCE_VERSION = "2.5-ae1"
HIP_WIDE_FIELD_MM = 51.0
ART_ZONE_SHARE = 0.70  # доля протяжённости кости сверху (geometry_features.POSITION_BANDS, признак band70)
# Пункт 3 («второе мнение по оси», Kimi K3, 4.3): порог расхождения контуров по sp_axis. Выбран правилом из
# tools/p25/axis_disagreement_nested.py (наименьший порог сетки 0.30…0.80, при котором среди верных решений OOF
# флагов не больше 20 %), применённым ко всем 166 снимкам после того, как вложенная проверка это правило приняла.
AXIS_DISAGREE_T = 0.45

# Как criteria_by_region в config.yaml (копия для вызовов без конфига: results_extras.csv)
CRITERIA_BY_REGION = {"spine": ["sp_pos", "sp_axis", "sp_art"], "right_hip": ["rh_pos", "rh_roi"],
                      "left_hip": ["lh_pos", "lh_roi"]}
GROUP_OF = {"sp_pos": "sp_pos", "sp_axis": "sp_axis", "sp_art": "sp_art",
            "rh_pos": "hip_pos", "lh_pos": "hip_pos", "rh_roi": "hip_roi", "lh_roi": "hip_roi"}

# Р4-2: роль критерия в карточке. measured — измерение со шкалой; hint — подсказка, решает врач.
CRITERION_ROLE = {"sp_pos": "measured", "sp_axis": "measured", "hip_roi": "measured",
                  "sp_art": "hint", "hip_pos": "hint"}
ROLE_TEXT = {"measured": "измерение", "hint": "подсказка, решает врач"}
ROLE_NOTE = {
    "measured": "Сервис измеряет: значение, единицы и ориентир показаны ниже.",
    "hint": "Сервис подсказывает, где посмотреть; нарушение или нет — решает врач по снимку.",
}

STATUS_TEXT = {
    "confirmed": "основание измерено",
    "wide_field": "поле шире обычного",
    "below_zone": "плотный участок вне зоны L1–L4",
    "no_witness": "измеримого основания нет",
    "not_applicable": "правило основания для критерия не задано",
}
STATUS_TEXT_MEASURED = "основание — измерение критерия, значения в таблице ниже"


def role_of(crit: str) -> Optional[str]:
    return CRITERION_ROLE.get(GROUP_OF.get(crit, crit))


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _flag(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "1.0")
    n = _num(v)
    return bool(n) if n is not None else False


# --------------------------------------------------------------------------- #
# Плотные участки (sp_art): те же признаки полосы, по которым решает модель 2.5.0
# --------------------------------------------------------------------------- #
def art_zone_summary(area_mm2: Any, band_area_mm2: Any, band_n: Any = None) -> Optional[Dict[str, Any]]:
    """Сводка по зоне измерения: площадь плотных участков по всему кадру (metal_metal_area_mm2) и в верхних 70 %
    протяжённости кости (metal_metal_band70_area_mm2, признак контура A sp_art с версии 2.5.0). None — не измерено."""
    a, b = _num(area_mm2), _num(band_area_mm2)
    if a is None or b is None:
        return None
    n = _num(band_n)
    return {"area_mm2": round(a, 1), "zone_area_mm2": round(b, 1), "outside_area_mm2": round(max(a - b, 0.0), 1),
            "zone_n": int(n) if n is not None else None, "zone_from": "верхние 70 % протяжённости кости"}


# --------------------------------------------------------------------------- #
# Сертификат по критерию и по снимку
# --------------------------------------------------------------------------- #
def measures_from_debug(dbg: Dict[str, Any]) -> Dict[str, Any]:
    dbg = dbg or {}
    return {"lateral_margin_mm": dbg.get("feat_lateral_margin_mm"),
            "metal_area_mm2": dbg.get("feat_metal_metal_area_mm2"),
            "band70_area_mm2": dbg.get("feat_metal_metal_band70_area_mm2"),
            "band70_n": dbg.get("feat_metal_metal_band70_n")}


def criterion_evidence(crit: str, meas: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Основание команды для флага критерия crit. meas — {lateral_margin_mm, metal_area_mm2, band70_area_mm2,
    band70_n}. Возвращает {criterion, group, role, status, status_text, command, text, basis}.
    command: retake — «Переснять» (как в 2.4.1); check — «Проверить …»."""
    meas = meas or {}
    g = GROUP_OF.get(crit, crit)
    out: Dict[str, Any] = {"criterion": crit, "group": g, "role": CRITERION_ROLE.get(g)}
    if g == "hip_pos":
        lm = _num(meas.get("lateral_margin_mm"))
        if lm is None:
            out.update(status="no_witness", command="check",
                       text="Проверить укладку бедра по снимку: боковой запас поля не измерен, "
                            "основания для пересъёмки нет",
                       basis={"lateral_margin_mm": None, "limit_mm": HIP_WIDE_FIELD_MM})
        elif lm > HIP_WIDE_FIELD_MM:
            out.update(status="wide_field", command="check",
                       text=f"Проверить укладку по малому вертелу: поле шире обычного "
                            f"(боковой запас {lm:.0f} мм при обычном до {HIP_WIDE_FIELD_MM:.0f} мм), "
                            f"оценка менее надёжна",
                       basis={"lateral_margin_mm": round(lm, 1), "limit_mm": HIP_WIDE_FIELD_MM})
        else:
            out.update(status="confirmed", command="retake",
                       text=f"Боковой запас поля {lm:.0f} мм — в обычных пределах (до {HIP_WIDE_FIELD_MM:.0f} мм); "
                            f"команда по оценке укладки",
                       basis={"lateral_margin_mm": round(lm, 1), "limit_mm": HIP_WIDE_FIELD_MM})
    elif g == "sp_art":
        z = art_zone_summary(meas.get("metal_area_mm2"), meas.get("band70_area_mm2"), meas.get("band70_n"))
        if z is None:
            out.update(status="not_applicable", command="retake",
                       text="Положение плотных участков не измерено; команда как в версии 2.4.1", basis=None)
        elif z["area_mm2"] <= 0:
            out.update(status="no_witness", command="check",
                       text="Проверить: плотный участок измерением не найден, флаг поставлен по изображению",
                       basis=z)
        elif z["zone_area_mm2"] <= 0:
            out.update(status="below_zone", command="check",
                       text=f"Проверить: плотный участок вне зоны L1–L4, на измерение может не влиять "
                            f"(вся площадь {z['area_mm2']:.0f} мм² ниже верхних 70 % протяжённости кости, "
                            f"где крылья подвздошных костей)",
                       basis=z)
        else:
            out.update(status="confirmed", command="retake",
                       text=f"Плотные участки в зоне измерения: {z['zone_area_mm2']:.0f} мм² "
                            f"(всего по кадру {z['area_mm2']:.0f} мм²)",
                       basis=z)
    else:
        out.update(status="not_applicable", command="retake",
                   text="Правило основания для этого критерия не задано; команда как в версии 2.4.1", basis=None)
    out["status_text"] = STATUS_TEXT[out["status"]]
    if out["status"] == "not_applicable" and out["role"] == "measured":
        out["status_text"] = STATUS_TEXT_MEASURED   # sp_pos, sp_axis, hip_roi: основание — само измерение
    return out


def image_evidence(region: str, flags: Dict[str, Any], meas: Optional[Dict[str, Any]] = None,
                   is_failure: bool = False) -> Optional[Dict[str, Any]]:
    """Сертификат снимка: по каждому флагу критерия — основание; command снимка — retake, если хотя бы у одного
    флага основание есть (или правило не задано), иначе check. None — флагов нет или файл не обработан
    (отсутствие основания никогда не маскирует Failure)."""
    if is_failure:
        return None
    flagged = [c for c, v in (flags or {}).items() if _flag(v)]
    if not flagged:
        return None
    items = [criterion_evidence(c, meas) for c in flagged]
    command = "retake" if any(i["command"] == "retake" for i in items) else "check"
    checks = [i for i in items if i["command"] == "check"]
    return {"version": EVIDENCE_VERSION, "region": region, "command": command,
            "command_text": "Переснять" if command == "retake" else "Проверить",
            "items": items, "checks": [i["criterion"] for i in checks],
            "text": "; ".join(i["text"] for i in checks) if command == "check" else "",
            "note": "Флаг критерия и класс в выгрузке не меняются; меняется только команда лаборанту."}


def evidence_from_debug(dbg: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """image_evidence по debug-словарю снимка (поля <crit>_flag, feat_*)."""
    dbg = dbg or {}
    region = str(dbg.get("internal_region") or "")
    crits = ((cfg or {}).get("criteria_by_region") or CRITERIA_BY_REGION).get(region, [])
    flags = {c: dbg.get(f"{c}_flag") for c in crits}
    return image_evidence(region, flags, measures_from_debug(dbg), is_failure=bool(dbg.get("error")))


def axis_second_opinion(p_geom: Any, p_emb: Any) -> Optional[Dict[str, Any]]:
    """Пункт 3: «контуры разошлись — посмотрите ось». Вспомогательный сигнал, не меняет класс, флаг и команду.
    None — одного из контуров нет."""
    g, e = _num(p_geom), _num(p_emb)
    if g is None or e is None:
        return None
    diff = abs(g - e)
    flag = diff > AXIS_DISAGREE_T
    return {"flag": bool(flag), "diff": round(diff, 3), "threshold": AXIS_DISAGREE_T,
            "p_geom": round(g, 3), "p_emb": round(e, 3),
            "text": ("Контуры разошлись — посмотрите ось: геометрия {:.2f}, изображение {:.2f} "
                     "(расхождение {:.2f} при пороге {:.2f}). Класс и команда не меняются.").format(g, e, diff, AXIS_DISAGREE_T)
            if flag else ""}


def axis_from_debug(dbg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    dbg = dbg or {}
    if str(dbg.get("internal_region") or "") != "spine" or dbg.get("error"):
        return None
    return axis_second_opinion(dbg.get("sp_axis_p_geom"), dbg.get("sp_axis_p_emb"))


def sr_text(ev: Optional[Dict[str, Any]]) -> str:
    """Строка для SR: команда и основание по каждому флагу."""
    if not ev:
        return ""
    return ev["command_text"] + ". " + "; ".join(
        f"{i['criterion']}: {i['status_text']}" + (f" — {i['text']}" if i["status"] != "not_applicable" else "")
        for i in ev["items"])


def criterion_evidence_map(ev: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {i["criterion"]: i for i in (ev or {}).get("items", [])}


def extras_fields(ev: Optional[Dict[str, Any]], axis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Колонки results_extras.csv (не входят в 9 колонок)."""
    out = {"action_command": "", "action_evidence": "", "action_evidence_text": "", "flag_roles": "",
           "axis_contours_diverge": "" if axis is None else int(bool(axis["flag"])),
           "axis_contour_diff": "" if axis is None else axis["diff"]}
    if ev:
        out.update({"action_command": ev["command"],
                    "action_evidence": ";".join(f"{i['criterion']}:{i['status']}" for i in ev["items"]),
                    "action_evidence_text": ev["text"],
                    "flag_roles": ";".join(f"{i['criterion']}:{ROLE_TEXT.get(i['role'] or '', '')}" for i in ev["items"])})
    return out
