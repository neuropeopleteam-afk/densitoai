"""
Общие функции К3: калибровка Platt по критерию, правило порога, запас «не уверен», уровень риска.

Используются train_stacked.py (обучение калибраторов и запасов), inference.py (применение),
tools/calibration_eval.py (воспроизведение чисел). Никаких данных здесь нет — только формулы.

Правило risk_level (документировано в docs/METRICS_REPORT.md, раздел «Калибровка и зона не уверен»):
  needs_review = хотя бы один критерий региона имеет |score - threshold| < margin_by_criterion[crit]
                 (или обработка завершилась отказом / fallback без моделей);
  risk_level   = «высокий», если quality_class = 1 и not needs_review;
                 «средний», если needs_review (любой класс);
                 «низкий»,  если quality_class = 0 и not needs_review.
CSV из 9 колонок этим не затрагивается: needs_review/risk_level — только debug-CSV и API details.
"""
from __future__ import annotations

import numpy as np

RISK_HIGH, RISK_MID, RISK_LOW = 'высокий', 'средний', 'низкий'


# ------------------------------------------------------------------ Platt
def fit_platt(scores, y):
    """Логистическая регрессия на одном признаке (скоре) без существенной регуляризации: p = sigmoid(a*s + b)."""
    from sklearn.linear_model import LogisticRegression
    s = np.asarray(scores, float).reshape(-1, 1)
    yy = np.asarray(y, int)
    if len(np.unique(yy)) < 2:
        p = float(np.clip(yy.mean() if len(yy) else 0.5, 1e-3, 1 - 1e-3))
        return 0.0, float(np.log(p / (1 - p)))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(s, yy)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt(scores, a, b):
    z = a * np.asarray(scores, float) + b
    return 1.0 / (1.0 + np.exp(-z))


# ------------------------------------------------------------------ правила порога
def threshold_by_rule(rule, y, scores, f1_optimal_fn, prevalence_fn, min_pos_f1=15):
    """rule: 'f1_optimal' (текущее: F1-опт при >= min_pos_f1 позитивов, иначе prevalence),
    'prevalence' (квантиль 1 - доля позитивов), 'prevalence_x<k>' (квантиль 1 - k*доля позитивов).
    Возвращает (порог, имя метода для metrics_summary)."""
    y = np.asarray(y, int); scores = np.asarray(scores, float)
    rule = (rule or 'f1_optimal').strip()
    if rule == 'f1_optimal':
        if int(y.sum()) >= min_pos_f1:
            return float(f1_optimal_fn(y, scores)), 'f1_optimal_oof'
        return float(prevalence_fn(y, scores)), 'prevalence'
    if rule == 'prevalence':
        return float(prevalence_fn(y, scores)), 'prevalence'
    if rule.startswith('prevalence_x'):
        k = float(rule[len('prevalence_x'):])
        prev = float(y.mean())
        if prev <= 0 or prev >= 1:
            return 0.5, rule
        return float(np.quantile(scores, 1 - min(0.999, k * prev))), rule
    raise ValueError(f'неизвестное правило порога: {rule}')


# ------------------------------------------------------------------ зона «не уверен»
def is_uncertain(dist, margin):
    """Зона «не уверен»: |score - threshold| <= margin (включительно: строки ровно на пороге всегда «не уверен»)."""
    return np.asarray(dist, float) <= float(margin) + 1e-12


def margins_for_quota(margins_data, q):
    """margins_data: {crit: |score - threshold| по OOF}. Запас по критерию = наибольшее значение из наблюдаемых
    расстояний, при котором доля строк с |s - thr| <= margin не превышает q; минимум 0 (строки на пороге)."""
    out = {}
    for c, m in margins_data.items():
        m = np.sort(np.asarray(m, float))
        cand = 0.0
        for v in np.unique(m):
            if is_uncertain(m, v).mean() <= q + 1e-12:
                cand = float(v)
            else:
                break
        out[c] = cand
    return out


def select_margins(margins_data, region_criteria, max_reject=0.05, step=0.0025,
                   quota_scope="per_criterion"):
    """Подбор запасов зоны «не уверен».

    quota_scope = "per_criterion" (по умолчанию): квота max_reject применяется к каждому
    критерию отдельно. Строка попадает на просмотр, если не уверен хотя бы один критерий,
    поэтому доля строк региона выше квоты по критерию и возвращается в `row_rates`.

    quota_scope = "per_region": квота q снижается от max_reject шагом step, пока доля
    СТРОК региона в зоне не станет <= max_reject. На пяти критериях это давало запас 0.0
    четырём критериям из пяти, то есть зона срабатывала только при точном совпадении с
    порогом; режим оставлен для воспроизведения прежних цифр.

    Запас не влияет на `quality_class` и девять колонок выгрузки — только на признак
    `needs_review`. margins_data[c] — массивы одной длины внутри региона (одни и те же
    строки в одном порядке). Возвращает ({crit: margin}, {region: q})."""
    margins, quotas = {}, {}
    for region, crits in region_criteria.items():
        sub = {c: margins_data[c] for c in crits if c in margins_data}
        if not sub:
            continue
        if quota_scope == "per_criterion":
            mg = margins_for_quota(sub, max_reject)
            margins.update(mg); quotas[region] = float(max_reject)
            continue
        q = max_reject
        mg = margins_for_quota(sub, q)
        while q > 0.0005:
            mg = margins_for_quota(sub, q)
            rows = row_reject_rates(sub, {region: crits}, mg)
            if rows[region] <= max_reject + 1e-12:
                break
            q -= step
        margins.update(mg); quotas[region] = float(q)
    return margins, quotas


def row_reject_rates(margins_data, region_criteria, margins):
    res = {}
    for region, crits in region_criteria.items():
        crits = [c for c in crits if c in margins_data]
        if not crits:
            continue
        rej = np.zeros(len(margins_data[crits[0]]), bool)
        for c in crits:
            rej |= is_uncertain(margins_data[c], margins[c])
        res[region] = float(rej.mean())
    return res


# ------------------------------------------------------------------ уровень риска
def risk_level(quality_class, needs_review):
    if needs_review:
        return RISK_MID
    return RISK_HIGH if int(quality_class) == 1 else RISK_LOW
