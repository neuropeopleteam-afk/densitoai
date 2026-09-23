"""Сводная таблица SUMMARY.md по outputs/<candidate>/summary.json и outputs/<candidate>/gate_*.json (paired_gate)."""
import json, sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else 'outputs/backlog_12_15')
ORDER = [('tz_weights', '13 tz_weights'), ('noisy_or', '14 noisy_or'), ('dupweight', '12 dupweight'), ('hip_side_router', '15 hip_side_router')]


def f(x, s='+.3f'):
    return 'н/д' if x is None else format(x, s)


def gate_cell(cand, stem):
    p = OUT / cand / f'gate_{stem}.json'
    if not p.exists():
        return 'не проверялся (гейт для этого кандидата — на уровне критерия)', ''
    r = json.load(open(p))
    d, m, c = r['delta_auc'], r['delta_macro_f1'], r['conditions']
    verdict = 'ПРИНЯТ' if r['accepted'] else 'не принят'
    why = []
    if not c['ci_low_gt_0']:
        why.append(f"ДИ{int(d['ci_adjusted_level'])} ΔAUC [{d['ci_adjusted'][0]:+.3f}; {d['ci_adjusted'][1]:+.3f}] включает 0, p_boot={d['p_boot_one_sided']:.3f}")
    if not c['point_ge_minimal_effect']:
        why.append(f"ΔAUC {d['point_mean_over_repeats']:+.3f} < 0.02")
    if not c['f1_ci_low_ge_floor']:
        why.append(f"низ ДИ90 Δmacro-F1 {m['ci_f1_low']:+.3f} < -0.01")
    return verdict, '; '.join(why)


lines = ['# Идеи 12–15: сводка nested repeated GroupKFold 5 × 20 (база = боевая 2.3.2)', '',
         'База и кандидат на одних и тех же внешних фолдах; группы = компоненты (study, pixel_hash); контур A/B и правило порога — как в config.yaml 2.3.2.',
         'Старое правило: ΔAUC ≥ 0.03 в ≥ 70 % повторов (≥ 14 из 20) и средняя macro-F1 не хуже. Новое правило (идея 11, tools/paired_gate.py):',
         'нижняя граница кластерного ДИ ΔAUC > 0 при alpha/5 (семейство из 5 гипотез: 4 кандидата + GPU-эксперимент по sp_pos, Холм), ΔAUC ≥ 0.02, низ ДИ90 Δmacro-F1 ≥ −0.01.', '',
         '## Уровень критерия', '',
         '| Идея | Критерий | AUC база | AUC кандидат | ΔAUC средн. [мин; макс] | повторов ΔAUC ≥ 0.03 | Δmacro-F1 средн. | ΔAUC pooled ДИ95 (бутстрап по исследованиям) | старое правило | новое правило (paired_gate, family=5) | причина отказа (новое правило) |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
for cand, name in ORDER:
    s = json.load(open(OUT / cand / 'summary.json'))
    for c in s['criteria']:
        same = abs(c['mean_delta_auc']) < 1e-12 and abs(c['max_delta_auc']) < 1e-12 and abs(c['min_delta_auc']) < 1e-12
        v, why = gate_cell(cand, c['criterion'])
        if same:
            v, why = 'кандидат = база', ''
        lines.append(f"| {name} | {c['criterion']} | {c['auc_base_mean_over_repeats']:.3f} | {c['auc_cand_mean_over_repeats']:.3f} | "
                     f"{c['mean_delta_auc']:+.3f} [{c['min_delta_auc']:+.3f}; {c['max_delta_auc']:+.3f}] | {c['n_repeats_gain_ge_0.03']}/{c['n_repeats']} | "
                     f"{c['mean_delta_macro_f1']:+.3f} | [{c['delta_auc_pooled_ci'][0]:+.3f}; {c['delta_auc_pooled_ci'][2]:+.3f}] | "
                     f"{'принят' if c['accepted_old_rule'] else 'не принят'} | {v} | {why} |")
lines += ['', '## Уровень региона: quality_prob (бинарно «есть нарушение»; как inference.any_violation_prob + consistent_quality_prob)', '',
          '| Идея | Регион | AUC quality_prob база | AUC quality_prob кандидат | Δ средн. | Δ raw (до согласования с классом) | повторов Δ ≥ 0.03 | F1 класса база → кандидат | строк с иным классом (все повторы) | Δ pooled ДИ95 | новое правило (paired_gate) | причина отказа |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
for cand, name in ORDER:
    s = json.load(open(OUT / cand / 'summary.json'))
    for r in s['region_quality_prob']:
        v, why = gate_cell(cand, f"region_{r['region']}")
        if abs(r['mean_delta_auc_qp']) < 1e-12 and r['n_class_differs_total'] == 0:
            v, why = 'кандидат = база', ''
        lines.append(f"| {name} | {r['region']} | {r['auc_qp_base_mean']:.3f} | {r['auc_qp_cand_mean']:.3f} | {r['mean_delta_auc_qp']:+.3f} | {r['mean_delta_auc_qp_raw']:+.3f} | "
                     f"{r['n_repeats_gain_ge_0.03']}/{r['n_repeats']} | {r['f1_class_base_mean']:.3f} → {r['f1_class_cand_mean']:.3f} | {r['n_class_differs_total']} | "
                     f"[{r['delta_auc_qp_pooled_ci'][0]:+.3f}; {r['delta_auc_qp_pooled_ci'][2]:+.3f}] | {v} | {why} |")
# по сторонам для бедра (справочно)
lines += ['', '## Бедро по сторонам (справочно; порог общий, оценка приёмки — на объединённом уровне)', '',
          '| Идея | Критерий | AUC правое база → кандидат (n_pos) | AUC левое база → кандидат (n_pos) |', '|---|---|---|---|']
for cand, name in ORDER:
    s = json.load(open(OUT / cand / 'summary.json'))
    for c in s['criteria']:
        if c['region'] == 'hip' and 'auc_base_mean_right' in c:
            lines.append(f"| {name} | {c['criterion']} | {c['auc_base_mean_right']:.3f} → {c['auc_cand_mean_right']:.3f} ({c['n_pos_right']}) | "
                         f"{c['auc_base_mean_left']:.3f} → {c['auc_cand_mean_left']:.3f} ({c['n_pos_left']}) |")
lines += ['', 'Итог: ни один из четырёх кандидатов не проходит ни старое, ни новое правило приёмки. Подробности и рекомендации — REPORT.md.']
(OUT / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines))
