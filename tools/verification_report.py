#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verification_report.py — HTML-отчёт самопроверки DensitoAI из verify_results.json (tools/verify.sh).

    python tools/verification_report.py --json outputs/verify/verify_results.json \
                                         --html outputs/verify/verification_report.html

Отчёт автономный (без внешних ресурсов): таблица проверок (зелёное/красное), окружение и версии
пакетов, sha256 весов и CSV, время прогонов, результат на данных пользователя (если был).
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
"""


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
<h2>2. Фантомы и контрольные суммы</h2>
<div class="grid">
{kv_table({"Файлов-фантомов": pm.get('n_files'), "Из них заведомо битых": pm.get('n_expected_failure'), "Seed генератора": pm.get('seed'),
           "Версия фантомов": pm.get('phantom_version'), "sha256 MANIFEST.json": sha.get('phantoms_manifest_json')})}
{kv_table({"sha256 предсказаний, прогон 1": sha.get('run1_predictions'), "sha256 предсказаний, прогон 2": sha.get('run2_predictions'),
           "sha256 файла expected_results.csv": sha.get('expected_results_csv'), "sha256 файла results.csv (прогон 1, с временем)": sha.get('run1_csv_file')})}
</div>
<p class="small">sha256 предсказаний считается по CSV без колонки <code>time_of_processing</code> (см. <code>tools/verify_checks.py predsha</code>).</p>
<h2>3. Веса моделей</h2>{w_html}
<h2>4. Окружение</h2>
<div class="grid">{kv_table(env, env_keys)}{t_html}</div>
<h2>5. Данные пользователя</h2>{ud_html}
<p class="small">Файлы: <span class="mono">{esc(res.get('run1', ''))}</span>, <span class="mono">{esc(res.get('run2', ''))}</span>; фантомы: <span class="mono">{esc(res.get('phantoms', ''))}</span>.</p>
</body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True)
    ap.add_argument("--html", required=True)
    a = ap.parse_args()
    res = json.loads(Path(a.json).read_text(encoding="utf-8"))
    Path(a.html).write_text(render(res), encoding="utf-8")
    print(f"report: {a.html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
