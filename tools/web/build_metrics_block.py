#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генерирует HTML-блок метрик для лендинга из models/metrics_summary.json (по критериям, OOF)
и таблицы «Итого» docs/METRICS_REPORT.md (бинарная задача по областям с 95 % ДИ).
Вставляет фрагмент в web/index.html между маркерами
  <!-- METRICS:BEGIN --> ... <!-- METRICS:END -->
Числа в HTML нельзя править руками: перезапустите скрипт.

Использование:
  python work/C/build_metrics_block.py [--root /path/to/densito_rebuild] [--html work/C/patch/web/index.html]
"""
import argparse
import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent.parent / "densito_rebuild"
DEFAULT_HTML = HERE / "patch" / "web" / "index.html"

# Шкала полосы AUC. Сравнение с внешними ориентирами на сайте не приводится (правило проекта, 22.09):
# показываем только положение AUC стека на шкале 0.5–1.0 и интервалы по исследованиям.
BAR_MIN, BAR_MAX = 0.5, 1.0
N_POS_SMALL = 15  # меньше — интервалы широки, читать по AUC и по паспорту выборки

CRITERIA = [  # (регион в json, ключ, человеческое название, область)
    ("spine", "sp_pos", "Укладка пациента", "позвоночник"),
    ("spine", "sp_axis", "Ось позвоночника", "позвоночник"),
    ("spine", "sp_art", "Посторонние предметы", "позвоночник"),
    ("hip", "hip_pos", "Укладка пациента", "бедро, обе стороны"),
    ("hip", "hip_roi", "Область интереса", "бедро, обе стороны"),
]
THR_METHOD = {
    "prevalence": "порог по доле позитивов",
    "f1_optimal_oof": "порог по F1 на OOF",
    "f1_optimal_oof_shared_hip_model": "порог по F1 на OOF, общая модель бедра",
}


def f3(v):
    return "—" if v is None else f"{float(v):.3f}"


def f2(v):
    return "—" if v is None else f"{float(v):.2f}"


def pct(v):
    v = min(max(float(v), BAR_MIN), BAR_MAX)
    return (v - BAR_MIN) / (BAR_MAX - BAR_MIN) * 100.0


def reliability(m):
    n_pos = int(m["n_pos"])
    stronger = "сильнее контур B (эмбеддинги)" if float(m["auc_emb"]) > float(m["auc_geom"]) else "сильнее геометрия"
    if n_pos < N_POS_SMALL:
        return "мало позитивов (" + str(n_pos) + "): интервал широкий, читать по AUC; " + stronger
    return stronger


def bar_html(auc):
    return (
        f'<div class="aucbar" role="img" aria-label="AUC {f3(auc)} на шкале 0.5–1.0">'
        f'<span class="aucbar-mark ok" style="left:{pct(auc):.1f}%"></span>'
        f'</div>'
    )


def parse_totals(md_text):
    """Таблица «Итого» METRICS_REPORT.md: область | ROC-AUC [ДИ] | F1 [ДИ] | macro-F1 [ДИ]."""
    m = re.search(r"## Итого\s*\n(.*?)(?:\n\n|\Z)", md_text, re.S)
    if not m:
        return []
    rows = []
    for line in m.group(1).splitlines():
        if not line.startswith("|") or "---" in line or "Область" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 4:
            rows.append(cells[:4])
    return rows


def build(root: Path) -> str:
    ms = json.loads((root / "models" / "metrics_summary.json").read_text(encoding="utf-8"))
    md = (root / "docs" / "METRICS_REPORT.md").read_text(encoding="utf-8")
    totals = parse_totals(md)

    out = []
    out.append('<div class="metrics-legend">'
               '<span>Полоса — положение ROC-AUC стека (OOF) на шкале 0.5–1.0; левый край — случайный классификатор</span>'
               '<span class="metrics-legend-note">интервалы F1 — бутстрап по исследованиям, 95 %</span></div>')
    out.append('<div class="table-wrap"><table class="metrics main">')
    out.append('<colgroup><col style="width:18%"><col style="width:10%"><col style="width:8%"><col style="width:8%">'
               '<col style="width:8%"><col style="width:7%"><col style="width:15%"><col style="width:11%"><col style="width:15%"></colgroup>')
    out.append('<thead><tr><th>Критерий</th><th>Область</th><th class="num">n / позит.</th>'
               '<th class="num">AUC геом.</th><th class="num">AUC эмб.</th>'
               '<th class="num">AUC стек</th><th>AUC на шкале 0.5–1.0</th>'
               '<th class="num">F1 OOF [95 % ДИ]</th><th>Надёжность</th></tr></thead><tbody>')
    aucs, small = [], []
    for region, key, title, area in CRITERIA:
        m = ms[region][key]
        auc = float(m["auc_stacked"])
        aucs.append(auc)
        if int(m["n_pos"]) < N_POS_SMALL:
            small.append(key)
        out.append(
            "<tr>"
            f'<td><div class="crit-title">{html.escape(title)}</div><div class="crit-sub">{html.escape(key)} · {html.escape(THR_METHOD.get(m.get("threshold_method", ""), str(m.get("threshold_method", ""))))}, порог {f3(m["threshold"])}</div></td>'
            f"<td>{html.escape(area)}</td>"
            f'<td class="num">{int(m["n_valid"])} / {int(m["n_pos"])}</td>'
            f'<td class="num">{f3(m["auc_geom"])}</td>'
            f'<td class="num">{f3(m["auc_emb"])}</td>'
            f'<td class="num strong">{f3(auc)}</td>'
            f'<td>{bar_html(auc)}</td>'
            f'<td class="num">{f2(m["f1_oof"])} [{f2(m["f1_ci_lo"])}; {f2(m["f1_ci_hi"])}]</td>'
            f"<td>{html.escape(reliability(m))}</td>"
            "</tr>"
        )
    out.append("</tbody></table></div>")

    # Паспорт выборки (docs/sample_passport.json): число положительных исследований по критерию, если файл есть
    pos_studies = {}
    pp = root / "docs" / "sample_passport.json"
    if pp.exists():
        try:
            pj = json.loads(pp.read_text(encoding="utf-8"))
            pos_studies = {k: int(v["pos_studies"]) for k, v in pj.get("criteria", {}).items() if v.get("pos_studies") is not None}
        except (ValueError, KeyError, TypeError):
            pos_studies = {}
    rare = [(k, n) for k, n in pos_studies.items() if k in {c[1] for c in CRITERIA} and n < 10]
    if rare:
        rare_txt = ", ".join(f"{html.escape(k)} — {n}" for k, n in rare)
        rare_sent = (f'У критериев с малым числом положительных исследований ({rare_txt}) интервалы широки: оценка описывает '
                     f'несколько конкретных исследований, а не устойчивую способность распознавать дефект '
                     f'(паспорт выборки — <code>docs/sample_passport.json</code>). ')
    else:
        small_txt = ", ".join(html.escape(k) for k in small) if small else "—"
        rare_sent = (f'У критериев с числом позитивов меньше {N_POS_SMALL} ({small_txt}) интервалы широки: оценка описывает несколько '
                     f'конкретных исследований, а не устойчивую способность распознавать дефект. ')
    out.append(
        f'<p class="metrics-summary">ROC-AUC стека по пяти критериям — от {f3(min(aucs))} до {f3(max(aucs))}. '
        + rare_sent +
        f'Слабые места показываем, а не скрываем. '
        f'Источник: <code>models/metrics_summary.json</code> (OOF, группировка по исследованию), '
        f'ДИ по F1 — бутстрап по исследованиям.</p>'
    )

    if totals:
        out.append('<h3 class="metrics-h3">Бинарная задача «есть нарушение» по областям, OOF</h3>')
        out.append('<div class="table-wrap"><table class="metrics compact"><thead><tr><th>Область</th>'
                   '<th class="num">ROC-AUC [95 % ДИ]</th><th class="num">F1 [95 % ДИ]</th>'
                   '<th class="num">Macro-F1 по типам нарушений [95 % ДИ]</th></tr></thead><tbody>')
        for area, auc, f1, mf1 in totals:
            out.append(f"<tr><td>{html.escape(area)}</td><td class=\"num strong\">{html.escape(auc)}</td>"
                       f"<td class=\"num\">{html.escape(f1)}</td><td class=\"num\">{html.escape(mf1)}</td></tr>")
        out.append("</tbody></table></div>")
        out.append('<p class="metrics-src">Источник: <code>docs/METRICS_REPORT.md</code>, таблица «Итого» '
                   '(<code>python src/eval_oof_metrics.py</code>, 95 % ДИ — бутстрап по исследованиям, 2000 повторов).</p>')
    return "\n".join(out)


def inject(html_path: Path, fragment: str) -> None:
    text = html_path.read_text(encoding="utf-8")
    begin, end = "<!-- METRICS:BEGIN -->", "<!-- METRICS:END -->"
    if begin not in text or end not in text:
        raise SystemExit(f"в {html_path} нет маркеров {begin} / {end}")
    pre, rest = text.split(begin, 1)
    _, post = rest.split(end, 1)
    html_path.write_text(pre + begin + "\n" + fragment + "\n" + end + post, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--html", default=str(DEFAULT_HTML))
    ap.add_argument("--print", action="store_true", help="только вывести фрагмент")
    a = ap.parse_args()
    frag = build(Path(a.root))
    if a.print:
        print(frag)
        return
    inject(Path(a.html), frag)
    print(f"OK: блок метрик обновлён в {a.html} ({len(frag)} символов)")


if __name__ == "__main__":
    main()
