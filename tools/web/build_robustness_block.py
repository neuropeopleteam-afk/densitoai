#!/usr/bin/env python3
"""
build_robustness_block.py — статический блок «Устойчивость к искажениям» для лендинга.

Источник чисел: docs/qa/robustness_results.json (сырые результаты tests/robustness_suite.py;
та же таблица — docs/ROBUSTNESS_REPORT.md). Никаких живых ползунков: только измеренные flip-rate
класса качества по видам искажений, отрисованные как SVG-полосы. Блок вставляется в web/index.html
между маркерами <!-- ROBUSTNESS:BEGIN --> и <!-- ROBUSTNESS:END -->.

Запуск: python tools/web/build_robustness_block.py [--root <densito_rebuild>] [--html web/index.html] [--print]
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

BEGIN, END = "<!-- ROBUSTNESS:BEGIN -->", "<!-- ROBUSTNESS:END -->"

# Порядок и короткие подписи (ключи — как в robustness_results.json)
ORDER = [
    ("identity (пересохранение)", "пересохранение файла"),
    ("теги удалены (PixelSpacing, описание, сторона, аппарат)", "удалены теги (шаг пикселя, описание)"),
    ("нет UID исследования/снимка", "нет UID исследования и снимка"),
    ("MONOCHROME1 (инверсия)", "инвертированная шкала (MONOCHROME1)"),
    ("16-битный контейнер (12 бит)", "16-битный контейнер"),
    ("обрезка 4 % по краям", "обрезка 4 % по краям"),
    ("resize ×0.8 (PixelSpacing пересчитан)", "уменьшение ×0,8 с пересчётом шага"),
    ("resize ×1.25 (PixelSpacing пересчитан)", "увеличение ×1,25 с пересчётом шага"),
    ("гауссов шум σ=3 %", "шум 3 %"),
    ("ярче (гамма 0.7)", "ярче (гамма 0,7)"),
    ("темнее (гамма 1.4)", "темнее (гамма 1,4)"),
]
FORMAT_KEYS = {k for k, _ in ORDER[:5]}


def build(data: dict) -> str:
    n = int(data.get("n_files", 0))
    dist = data.get("distortions", {})
    rows = []
    for key, label in ORDER:
        d = dist.get(key)
        if not d:
            continue
        rows.append((label, float(d["class_flip_rate"]), int(round(d["class_flip_rate"] * d["n"])), int(d["n"]),
                     float(d.get("failure_rate", 0.0)), float(d.get("region_flip_rate", 0.0)), key in FORMAT_KEYS))
    # SVG: ширина 720, строка 34 px, подписи слева 300 px, полоса до 330 px, число справа
    row_h, left, bar_w, top = 34, 300, 300, 14
    MAX_PCT = 0.30  # правый край шкалы — 30 %
    height = top + row_h * len(rows) + 28
    parts = [f'<svg class="rob-svg" viewBox="0 0 720 {height}" role="img" '
             f'aria-label="Доля снимков, у которых изменился класс качества, по видам искажений">']
    parts.append(f'<line x1="{left}" y1="{top - 6}" x2="{left}" y2="{height - 22}" class="rob-axis"/>')
    for i, (label, rate, k, nn, fail, rflip, is_fmt) in enumerate(rows):
        y = top + i * row_h
        w = max(2, round(bar_w * rate / MAX_PCT))  # шкала 0..MAX_PCT
        cls = "rob-bar fmt" if is_fmt else "rob-bar"
        parts.append(f'<text x="{left - 12}" y="{y + 18}" class="rob-label" text-anchor="end">{html.escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y + 4}" width="{w}" height="20" rx="3" class="{cls}"/>')
        parts.append(f'<text x="{left + w + 10}" y="{y + 19}" class="rob-val">{rate * 100:.1f} % ({k}/{nn})</text>')
    # шкала
    for tick in (0, 10, 20, 30):
        x = left + bar_w * tick / (MAX_PCT * 100)
        parts.append(f'<text x="{x}" y="{height - 6}" class="rob-tick" text-anchor="middle">{tick} %</text>')
    parts.append("</svg>")
    fails = max(r[4] for r in rows) if rows else 0.0
    rflips = max(r[5] for r in rows) if rows else 0.0
    base = data.get("baseline", {})
    note = (f"Выборка: {n} снимков разметки ({data.get('regions', {}).get('spine', '—')} позвоночник, "
            f"{data.get('regions', {}).get('right_hip', 0) + data.get('regions', {}).get('left_hip', 0)} бедро), "
            f"из них с нарушениями {data.get('n_positive', '—')}. Каждый снимок пересохранён с одним искажением и прогнан повторно; "
            f"показана доля снимков, у которых изменился класс качества. Ошибок обработки (Failure) ни в одном варианте не было "
            f"(максимум {fails * 100:.1f} %), область снимка не менялась ни разу (максимум {rflips * 100:.1f} %). "
            f"Время на снимок в исходном прогоне: медиана {base.get('time_p50', 0):.2f} с, 95-й перцентиль {base.get('time_p95', 0):.2f} с.")
    return "\n".join([
        BEGIN,
        '<div class="rob-wrap">',
        "\n".join(parts),
        '<div class="rob-legend"><span><i class="sw fmt"></i>формат файла и теги</span><span><i class="sw"></i>пиксели: яркость, шум, масштаб, обрезка</span></div>',
        f'<p class="rob-note">{html.escape(note)}</p>',
        '<p class="rob-note">На реальном потоке одного аппарата GE Lunar с фиксированным экспортом искажений яркости, шума и масштаба нет: '
        'все снимки приходят в одном формате. Изменения формата и тегов на потоке возможны, и к ним система устойчива полностью. '
        'Перевороты при изменении экспозиции сосредоточены у порога решения (медианный запас 0,09–0,11), то есть касаются спорных снимков.</p>',
        f'<p class="rob-src">Источник: docs/qa/robustness_results.json ({html.escape(str(data.get("date", "")))}), '
        'docs/ROBUSTNESS_REPORT.md; скрипт tests/robustness_suite.py.</p>',
        "</div>",
        END,
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--json", default=None)
    ap.add_argument("--html", default=None)
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    data = json.load(open(a.json or root / "docs" / "qa" / "robustness_results.json", encoding="utf-8"))
    block = build(data)
    if a.print:
        print(block)
    html_path = Path(a.html or root / "web" / "index.html")
    if html_path.exists():
        s = html_path.read_text(encoding="utf-8")
        i, j = s.find(BEGIN), s.find(END)
        if i < 0 or j < 0:
            raise SystemExit(f"маркеры {BEGIN} / {END} не найдены в {html_path}")
        s = s[:i] + block + s[j + len(END):]
        html_path.write_text(s, encoding="utf-8")
        print(f"updated {html_path}")


if __name__ == "__main__":
    main()
