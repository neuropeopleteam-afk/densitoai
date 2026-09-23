#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verification_report.py — HTML-отчёт самопроверки DensitoAI из verify_results.json (tools/verify.sh).

    python tools/verification_report.py --json outputs/verify/verify_results.json \
                                         --html outputs/verify/verification_report.html

Отчёт автономный (без внешних ресурсов): таблица проверок (зелёное/красное), карта доказательств
(какому пункту ТЗ отвечает каждая проверка и что она доказывает — tools/evidence_map.json), блок
«что отчёт не доказывает», окружение и версии пакетов, sha256 весов и CSV, время прогонов,
результат на данных пользователя (если был).
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1b1f23;background:#fff}
h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:28px 0 8px;border-bottom:1px solid #e1e4e8;padding-bottom:4px}
.sub{color:#586069;font-size:13px}
.badge{display:inline-block;padding:6px 14px;border-radius:6px;font-weight:600;font-size:15px;color:#fff;margin:12px 0}
.ok{background:#22863a}.fail{background:#cb2431}.warn{background:#b08800}
table{border-collapse:collapse;width:100%;font-size:13.5px} th,td{border:1px solid #e1e4e8;padding:6px 8px;text-align:left;vertical-align:top}
th{background:#f6f8fa} td.st{width:90px;font-weight:600;text-align:center;color:#fff}
td.st.ok{background:#22863a} td.st.fail{background:#cb2431} td.st.warn{background:#b08800}
code,.mono{font-family:SFMono-Regular,Consolas,Menlo,monospace;font-size:12.5px;word-break:break-all}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px} @media(max-width:800px){.grid{grid-template-columns:1fr}}
.small{font-size:12px;color:#586069}
td.tz{width:200px;font-size:12.5px} td.know{font-size:13px} .know b{color:#22863a} .miss{color:#b08800}
.notproven{background:#fff8e5;border:1px solid #f0d58c;border-radius:6px;padding:10px 14px;margin:8px 0}
.notproven h3{font-size:14px;margin:0 0 6px} .notproven p{margin:0 0 8px;font-size:13px}
"""

EVIDENCE_MAP_PATH = Path(__file__).resolve().parent / "evidence_map.json"


def load_evidence_map(path: Path = EVIDENCE_MAP_PATH) -> dict:
    """Карта доказательств: список записей {prefix, tz, requirement, knows, where}. Если файла нет —
    пустая карта (отчёт всё равно строится, раздел помечается как недоступный)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"entries": [], "not_proven": []}


def evidence_for(name: str, emap: dict):
    """Запись карты для проверки по началу имени (в имени могут быть подставленные значения, например допуск)."""
    hits = [e for e in emap.get("entries", []) if str(name).startswith(e["prefix"])]
    if not hits:
        return None
    return max(hits, key=lambda e: len(e["prefix"]))


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def kv_table(d: dict, keys=None) -> str:
    keys = keys or list(d.keys())
    rows = "".join(f"<tr><th>{esc(k)}</th><td class='mono'>{esc(d.get(k, ''))}</td></tr>" for k in keys)
    return f"<table>{rows}</table>"


def render(res: dict) -> str:
    checks = res.get("checks", [])
    n_ok = sum(1 for c in checks if c["ok"])
    status_cls = "ok" if res.get("ok") else "fail"
    status_txt = "ПРОВЕРКА ПРОЙДЕНА" if res.get("ok") else "ПРОВЕРКА НЕ ПРОЙДЕНА"
    rows = []
    for c in checks:
        if c["ok"]:
            cls, txt = "ok", "OK"
        elif c.get("level") == "warning":
            cls, txt = "warn", "ЗАМЕЧАНИЕ"
        else:
            cls, txt = "fail", "ОШИБКА"
        rows.append(f"<tr><td class='st {cls}'>{txt}</td><td>{esc(c['name'])}</td><td class='small'>{esc(c['detail'])}</td></tr>")
    checks_html = "<table><tr><th>Статус</th><th>Проверка</th><th>Подробности</th></tr>" + "".join(rows) + "</table>"

    emap = load_evidence_map()
    ev_rows, unmapped = [], []
    for c in checks:
        e = evidence_for(c["name"], emap)
        cls = "ok" if c["ok"] else ("warn" if c.get("level") == "warning" else "fail")
        txt = "OK" if c["ok"] else ("ЗАМЕЧАНИЕ" if cls == "warn" else "ОШИБКА")
        if e is None:
            unmapped.append(c["name"])
            ev_rows.append(f"<tr><td class='st {cls}'>{txt}</td><td>{esc(c['name'])}</td><td class='tz miss'>нет в карте</td>"
                           f"<td class='know miss'>Для этой проверки нет записи в tools/evidence_map.json — см. её подробности в разделе 1.</td></tr>")
            continue
        knows = esc(e.get("knows", ""))
        if not c["ok"]:
            knows = ("<b style='color:#cb2431'>Проверка не пройдена — утверждение ниже НЕ доказано.</b><br>" + knows) if cls == "fail" \
                else ("<b style='color:#b08800'>Замечание, не ошибка.</b><br>" + knows)
        ev_rows.append(f"<tr><td class='st {cls}'>{txt}</td><td>{esc(c['name'])}<div class='small mono'>{esc(e.get('where', ''))}</div></td>"
                       f"<td class='tz'>{esc(e.get('tz', ''))}<div class='small'>{esc(e.get('requirement', ''))}</div></td><td class='know'>{knows}</td></tr>")
    if emap.get("entries"):
        evidence_html = ("<p class='small'>Каждая строка — одна проверка из раздела 1, пункт ТЗ или требование заказчика, которому она отвечает, "
                         "и то, что проверяющий теперь знает, если строка зелёная. Источник сопоставления — <code>tools/evidence_map.json</code>, "
                         "полнота карты проверяется <code>tests/test_evidence_map.py</code>.</p>"
                         "<table><tr><th>Статус</th><th>Проверка</th><th>Пункт ТЗ / требование</th><th>Что вы теперь знаете</th></tr>"
                         + "".join(ev_rows) + "</table>")
        if unmapped:
            evidence_html += "<p class='small miss'>Проверок без записи в карте: " + esc(len(unmapped)) + ".</p>"
        n_fail = sum(1 for c in checks if not c["ok"] and c.get("level") != "warning")
        n_warn = sum(1 for c in checks if not c["ok"] and c.get("level") == "warning")
        summary = f"Пройдено {n_ok} из {len(checks)} проверок"
        summary += f", ошибок {n_fail}" if n_fail else ", ошибок нет"
        summary += f", замечаний {n_warn}." if n_warn else "."
        evidence_html = f"<p><b>{esc(summary)}</b></p>" + evidence_html
    else:
        evidence_html = "<p class='small miss'>Файл tools/evidence_map.json не найден рядом с отчётом — карта доказательств недоступна.</p>"
    np_blocks = "".join(f"<h3>{esc(b.get('title', ''))}</h3><p>{esc(b.get('text', ''))}</p>" for b in emap.get("not_proven", []))
    not_proven_html = (f"<div class='notproven'>{np_blocks}</div>" if np_blocks
                       else "<p class='small'>Список ограничений отчёта не задан в tools/evidence_map.json.</p>")

    env = res.get("env", {})
    env_keys = ["python", "platform", "machine", "torch", "torchvision", "numpy", "scipy", "sklearn", "pandas",
                "pydicom", "cv2", "skimage", "yaml", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "torch_threads", "TORCH_HOME"]
    timings = res.get("timings", {})
    t_html = kv_table({"Прогон 1 на фантомах, с": timings.get("run1_s", ""), "Прогон 2 на фантомах, с": timings.get("run2_s", ""),
                       "Прогон на данных пользователя, с": timings.get("data_s", ""), "Начало (UTC)": timings.get("started", "")})
    sha = res.get("sha256", {})
    pm = res.get("phantoms_manifest", {})
    weights = res.get("weights", {})
    w_rows = "".join(
        f"<tr><td class='st {'ok' if f['ok'] else 'fail'}'>{'OK' if f['ok'] else 'ОШИБКА'}</td><td class='mono'>{esc(f['file'])}</td>"
        f"<td class='mono'>{esc(f['actual'])}</td></tr>" for f in weights.get("files", []))
    w_html = ("<table><tr><th>Статус</th><th>Файл</th><th>sha256 (фактический)</th></tr>" + w_rows + "</table>") if w_rows \
        else "<p class='small'>Результат hash_weights.py отсутствует.</p>"

    ud = res.get("user_data")
    if ud:
        ud_html = kv_table({"Каталог": ud.get("dir"), "CSV": ud.get("csv"), "Строк": ud.get("rows"), "Failure": ud.get("failures"),
                            "sha256 предсказаний (без time_of_processing)": ud.get("sha256_predictions"),
                            "sha256 предсказаний без quality_prob": ud.get("sha256_predictions_without_prob"),
                            "Ожидаемый sha256 (--expected-sha)": ud.get("expected_sha") or "не задан"})
    else:
        ud_html = "<p class='small'>Прогон на данных пользователя не запрашивался (опция <code>--data</code>).</p>"

    return f"""<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8"><title>DensitoAI — отчёт самопроверки</title>
<style>{CSS}</style></head><body>
<h1>DensitoAI — отчёт самопроверки (verify.sh)</h1>
<div class="sub">Сформирован {esc(res.get('generated_at', ''))} · корень проекта <span class="mono">{esc(res.get('root', ''))}</span></div>
<div class="badge {status_cls}">{status_txt}: {n_ok} из {len(checks)} проверок</div>
<h2>1. Проверки</h2>{checks_html}
<h2>2. Карта доказательств: что вы теперь знаете</h2>{evidence_html}
<h2>3. Что этот отчёт не доказывает</h2>{not_proven_html}
<h2>4. Фантомы и контрольные суммы</h2>
<div class="grid">
{kv_table({"Файлов-фантомов": pm.get('n_files'), "Из них заведомо битых": pm.get('n_expected_failure'), "Seed генератора": pm.get('seed'),
           "Версия фантомов": pm.get('phantom_version'), "sha256 MANIFEST.json": sha.get('phantoms_manifest_json')})}
{kv_table({"sha256 предсказаний, прогон 1": sha.get('run1_predictions'), "sha256 предсказаний, прогон 2": sha.get('run2_predictions'),
           "sha256 файла expected_results.csv": sha.get('expected_results_csv'), "sha256 файла results.csv (прогон 1, с временем)": sha.get('run1_csv_file')})}
</div>
<p class="small">sha256 предсказаний считается по CSV без колонки <code>time_of_processing</code> (см. <code>tools/verify_checks.py predsha</code>).</p>
<h2>5. Веса моделей</h2>{w_html}
<h2>6. Окружение</h2>
<div class="grid">{kv_table(env, env_keys)}{t_html}</div>
<h2>7. Данные пользователя</h2>{ud_html}
<p class="small">Файлы: <span class="mono">{esc(res.get('run1', ''))}</span>, <span class="mono">{esc(res.get('run2', ''))}</span>; фантомы: <span class="mono">{esc(res.get('phantoms', ''))}</span>.</p>
</body></html>"""


def evidence_markdown(emap: dict | None = None) -> str:
    """Карта доказательств как markdown-таблица (для docs/VERIFICATION.md): python tools/verification_report.py --evidence-md."""
    emap = emap or load_evidence_map()
    cell = lambda x: str(x).replace("|", "\\|").replace("\n", " ")  # noqa: E731
    out = ["| № | Проверка verify | Пункт ТЗ / требование | Что вы теперь знаете (если строка зелёная) |", "|---|---|---|---|"]
    for i, e in enumerate(emap.get("entries", []), 1):
        mark = " (только с `--data`)" if e.get("optional") else ""
        out.append(f"| {i} | {cell(e['prefix'])}{mark} | {cell(e['tz'])} | {cell(e['knows'])} |")
    out.append("")
    out.append("Что этот отчёт не доказывает:")
    out.append("")
    for b in emap.get("not_proven", []):
        out.append(f"- **{cell(b.get('title', ''))}.** {cell(b.get('text', ''))}")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="verify_results.json (обязателен, если не задан --evidence-md)")
    ap.add_argument("--html", help="куда писать HTML-отчёт")
    ap.add_argument("--evidence-md", action="store_true", help="напечатать карту доказательств как markdown и выйти")
    a = ap.parse_args()
    if a.evidence_md:
        sys.stdout.write(evidence_markdown())
        return 0
    if not a.json or not a.html:
        ap.error("нужны --json и --html (или --evidence-md)")
    res = json.loads(Path(a.json).read_text(encoding="utf-8"))
    Path(a.html).write_text(render(res), encoding="utf-8")
    print(f"report: {a.html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
