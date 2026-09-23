"""Сведение результатов калибровки (tools/paired_gate_calibration.py, файлы gate_calibration_<tag>.*)
в единые outputs/gate_calibration.csv/.json и markdown-таблицы для отчёта.

Вызов: python tools/paired_gate_tables.py --out-dir work/idea11_gate/outputs --tags spine hip
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ORDER = ['sp_pos', 'sp_axis', 'sp_art', 'hip_pos', 'hip_roi']
RULES = {
    'rate_old': 'старое',
    'rate_new_boot_perm': 'новое (AUC-часть)',
    'rate_new_boot_perm_f1_mean_ci60': 'новое + F1',
    'rate_ttest': 't-test по повторам',
}


def fmt(x):
    return '—' if pd.isna(x) else f'{100 * x:.0f} %'


def md_table(df, cols, headers):
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)]
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join(str(r[c]) for c in cols) + ' |')
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(Path(__file__).resolve().parents[1] / 'outputs'))
    ap.add_argument('--tags', nargs='*', default=['spine', 'hip'])
    ap.add_argument('--f1-variant', default='f1_mean_ci60', help='вариант условия по macro-F1 для итогового правила')
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    raws, tables, calib = [], [], {}
    for tag in args.tags:
        raws.append(pd.read_csv(out / f'gate_calibration_raw_{tag}.csv', keep_default_na=False, na_values=['']))
        tables.append(pd.read_csv(out / f'gate_calibration_{tag}.csv', keep_default_na=False, na_values=['']))  # 'null' — метка, не NaN
        with open(out / f'gate_calibration_{tag}.json', encoding='utf-8') as f:
            j = json.load(f)
        calib.update(j['calibration']); params = j['params']; rules = j['rules']
    raw = pd.concat(raws, ignore_index=True)
    table = pd.concat(tables, ignore_index=True)
    table['criterion'] = pd.Categorical(table['criterion'], ORDER, ordered=True)
    table = table.sort_values(['criterion', 'n_repeats', 'kind', 'scenario']).reset_index(drop=True)
    raw.to_csv(out / 'gate_calibration_raw.csv', index=False)
    table.to_csv(out / 'gate_calibration.csv', index=False)
    final_rule = f'rate_new_boot_perm_{args.f1_variant}'
    RULES['rate_new_boot_perm_f1_mean_ci60'] = 'новое (полное)'
    if final_rule not in table:
        raise SystemExit(f'нет колонки {final_rule}')

    md = []
    # 1. структура критериев
    rows = []
    for c in ORDER:
        k = calib[c]
        rows.append(dict(criterion=c, n=k['n'], n_pos=k['n_pos'], n_groups=k['n_groups'], n_pos_groups=k['n_pos_groups'],
                         auc=f"{k['auc_base']:.3f}", sigma=f"{k['sigma_split']:.3f}", sd=f"{k['target_sd_repeats']:.3f}",
                         shifts=', '.join(f"{e}: {v['c']:.3f}" for e, v in k['shift_for_effect'].items())))
    md.append('### Структура данных и калибровка модели повторов\n\n' + md_table(pd.DataFrame(rows),
              ['criterion', 'n', 'n_pos', 'n_groups', 'n_pos_groups', 'auc', 'sigma', 'sd', 'shifts'],
              ['Критерий', 'n', 'позитивов', 'групп', 'групп с позитивами', 'AUC базы (oof_stacked)', 'σ шума разбиений', 'целевой sd ΔAUC по повторам', 'сдвиг c(e) для эффекта e']))

    # 2. ложные принятия
    def rate_block(kind_filter, title, scen_label):
        sub = table[table['kind'].isin(kind_filter)].copy()
        rows = []
        for (c, sc), g in sub.groupby(['criterion', 'scenario'], observed=True, sort=False):
            r = dict(criterion=c, scenario=sc)
            for R in (10, 20):
                gg = g[g['n_repeats'] == R]
                if len(gg) == 0:
                    continue
                gg = gg.iloc[0]
                r[f'old_{R}'] = fmt(gg['rate_old']); r[f'new_auc_{R}'] = fmt(gg['rate_new_boot_perm'])
                r[f'new_full_{R}'] = fmt(gg[final_rule]); r[f'tt_{R}'] = fmt(gg['rate_ttest'])
                r[f'd_{R}'] = f"{gg['mean_delta_auc']:+.3f}"
            rows.append(r)
        df = pd.DataFrame(rows)
        cols = ['criterion', 'scenario', 'd_10', 'old_10', 'new_auc_10', 'new_full_10', 'tt_10', 'old_20', 'new_auc_20', 'new_full_20', 'tt_20']
        heads = ['Критерий', scen_label, 'ΔAUC реализ. (R=10)', 'старое R=10', 'новое AUC R=10', 'новое полное R=10', 't-test R=10',
                 'старое R=20', 'новое AUC R=20', 'новое полное R=20', 't-test R=20']
        return f'### {title}\n\n' + md_table(df, cols, heads)

    md.append(rate_block(['null'], 'Ложные принятия при истинном эффекте <= 0 (доля принятий из n_sim симуляций)', 'Сценарий нуля'))
    md.append(rate_block(['null_diluted'], 'Улучшение, переставленное внутри кластеров (не ноль: реализованный ΔAUC приведён)', 'Сценарий'))
    md.append(rate_block(['power_shift'], 'Мощность, модель «сдвиг позитивов» (кандидат = база + сигнал; оптимистичный случай)', 'Эффект'))
    md.append(rate_block(['power_binormal'], 'Мощность, бинормальная модель (другая модель с корреляцией 0.8, AUC выше на e; реалистичный случай)', 'Эффект'))

    # 3. компактная таблица «критерий × эффект -> вероятность принятия» для обоих правил (R=20, бинормальная модель)
    rows = []
    for c in ORDER:
        for kind in ('power_shift', 'power_binormal'):
            g = table[(table['criterion'] == c) & (table['kind'] == kind) & (table['n_repeats'] == 20)].sort_values('true_effect')
            r = dict(criterion=c, model='сдвиг' if kind == 'power_shift' else 'бинормальная')
            for _, x in g.iterrows():
                e = f"{x['true_effect']:.2f}"
                r[f'old_{e}'] = fmt(x['rate_old']); r[f'new_{e}'] = fmt(x[final_rule])
            rows.append(r)
    df = pd.DataFrame(rows)
    effs = ['0.02', '0.03', '0.05', '0.08']
    cols = ['criterion', 'model'] + [f'{k}_{e}' for e in effs for k in ('old', 'new')]
    heads = ['Критерий', 'Модель эффекта'] + [f'{k} +{e}' for e in effs for k in ('старое', 'новое')]
    md.append('### Критерий × эффект -> вероятность принятия (R=20; новое правило полное, с условием по macro-F1)\n\n' + md_table(df, cols, heads))

    # 4. варианты условия по F1 (эффект 0.05 сдвиг и бинормальный, R=20), чтобы обосновать выбор
    fcols = [c for c in table.columns if c.startswith('rate_new_boot_perm_f1_')] + ['rate_new_boot_perm']
    sub = table[(table['n_repeats'] == 20) & (table['scenario'].isin(['effect_shift_0.05', 'effect_binormal_0.05', 'null_binormal_equal_0.8']))]
    rows = []
    for _, x in sub.iterrows():
        r = dict(criterion=x['criterion'], scenario=x['scenario'])
        for c in fcols:
            r[c] = fmt(x[c])
        rows.append(r)
    md.append('### Влияние условия по macro-F1 на принятие (R=20)\n\n' + md_table(pd.DataFrame(rows), ['criterion', 'scenario'] + fcols,
              ['Критерий', 'Сценарий'] + [c.replace('rate_new_boot_perm_', '').replace('rate_new_boot_perm', 'без условия F1') for c in fcols]))

    # минимальный эффект с мощностью >= 0.8
    rows = []
    for c in ORDER:
        r = dict(criterion=c)
        for kind in ('power_shift', 'power_binormal'):
            for R in (10, 20):
                g = table[(table['criterion'] == c) & (table['kind'] == kind) & (table['n_repeats'] == R)].sort_values('true_effect')
                ok = g[g[final_rule] >= 0.8]
                r[f'{kind}_{R}'] = f"+{ok['true_effect'].iloc[0]:.2f}" if len(ok) else '> 0.08'
                ok_old = g[g['rate_old'] >= 0.8]
                r[f'{kind}_{R}_old'] = f"+{ok_old['true_effect'].iloc[0]:.2f}" if len(ok_old) else '> 0.08'
        rows.append(r)
    md.append('### Минимальный истинный эффект, который ловится с вероятностью >= 0.8\n\n' + md_table(pd.DataFrame(rows),
              ['criterion', 'power_shift_10', 'power_shift_20', 'power_binormal_10', 'power_binormal_20', 'power_binormal_20_old'],
              ['Критерий', 'сдвиг, R=10', 'сдвиг, R=20', 'бинормальная, R=10', 'бинормальная, R=20', 'бинормальная, R=20, старое правило']))

    (out / 'gate_calibration_tables.md').write_text('\n\n'.join(md) + '\n', encoding='utf-8')
    with open(out / 'gate_calibration.json', 'w', encoding='utf-8') as f:
        rules = dict(rules, new_boot_perm='нижняя граница кластерного ДИ90 ΔAUC > 0 и p sign-flip < 0.05 и ΔAUC >= 0.02 (AUC-часть итогового правила)',
                     **{final_rule.replace('rate_', ''): 'итоговое правило (умолчания paired_gate.py): new_boot_perm и условие по macro-F1 ' + args.f1_variant})
        params = dict(params, criteria=sorted(set(table['criterion'].astype(str))), tag=args.tags)
        json.dump(dict(params=params, rules=rules, final_rule_column=final_rule, calibration=calib,
                       table=json.loads(table.to_json(orient='records'))), f, indent=2, ensure_ascii=False)
    print('\n\n'.join(md))
    print(f'\nфайлы: {out / "gate_calibration.csv"}, {out / "gate_calibration.json"}, {out / "gate_calibration_tables.md"}')


if __name__ == '__main__':
    main()
