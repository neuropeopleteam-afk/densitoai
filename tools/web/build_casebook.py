#!/usr/bin/env python3
"""Офлайн-casebook: самодостаточный HTML с карточками решения на 10–12 кейсах.

Кейсы выбираются автоматически из разметки (data/geometry_features.csv) и OOF-предсказаний
стэкинга (models/oof_stacked_*.csv): по 2 примера на каждый из 5 критериев — с одиночным
нарушением по разметке и правильно предсказанные (pred_label = 1, наибольший oof_stacked),
разные исследования; плюс 1 норма позвоночника и 1 норма бедра (все критерии 0 и pred 0).
Файлы копируются под именами case01.dcm…, прогоняются тем же движком и теми же функциями,
что и API (api_server._run_job / _attach_bonus / _details), а в HTML попадают только
обезличенные строки: «кейс N», без имён файлов и UID. Картинки — base64.

ВНИМАНИЕ: результат содержит пиксели конкурсных DICOM и не предназначен для публичного
репозитория. В репозиторий кладётся только этот генератор.

Пример:
  OMP_NUM_THREADS=1 TORCH_HOME=models/torch_home python3 tools/web/build_casebook.py \
      --root . --out /path/outside/repo/casebook.html
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

CRITERIA = ["sp_pos", "sp_axis", "sp_art", "hip_pos", "hip_roi"]
CRIT_COLS = {"spine": ["sp_pos", "sp_axis", "sp_art"], "right_hip": ["rh_pos", "rh_roi"], "left_hip": ["lh_pos", "lh_roi"]}
OOF_FILES = {"sp_pos": ["oof_stacked_spine_sp_pos.csv"], "sp_axis": ["oof_stacked_spine_sp_axis.csv"],
             "sp_art": ["oof_stacked_spine_sp_art.csv"], "hip_pos": ["oof_stacked_hip_hip_pos.csv"],
             "hip_roi": ["oof_stacked_hip_hip_roi.csv"]}
LABEL_TEXT = {"sp_pos": "нарушение укладки (позвоночник)", "sp_axis": "ось позвоночника не выровнена",
              "sp_art": "посторонние предметы", "hip_pos": "нарушение укладки бедра",
              "hip_roi": "некорректная область интереса бедра", "ok": "норма"}


def _crit_of(row) -> str | None:
    """Единственный положительный критерий по разметке (None, если их 0 или больше одного)."""
    cols = CRIT_COLS.get(str(row["region"]), [])
    pos = [c for c in cols if float(row.get(c) or 0) == 1.0]
    if len(pos) != 1:
        return None
    c = pos[0]
    return {"rh_pos": "hip_pos", "lh_pos": "hip_pos", "rh_roi": "hip_roi", "lh_roi": "hip_roi"}.get(c, c)


def select_cases(root: Path, n_per: int, n_norm: int):
    import pandas as pd
    lab = pd.read_csv(root / "data" / "geometry_features.csv")
    lab["single_crit"] = lab.apply(_crit_of, axis=1)
    used_studies: set[str] = set()
    chosen = []
    for crit in CRITERIA:
        parts = [pd.read_csv(root / "models" / f) for f in OOF_FILES[crit] if (root / "models" / f).exists()]
        if not parts:
            print(f"[warn] нет OOF для {crit}", file=sys.stderr)
            continue
        oof = pd.concat(parts, ignore_index=True)
        oof = oof[(oof.y_true == 1) & (oof.pred_label == 1)].sort_values("oof_stacked", ascending=False)
        cand = oof.merge(lab[["file_path", "region", "single_crit"]], on="file_path", how="inner")
        cand = cand[cand.single_crit == crit]
        k = 0
        for _, r in cand.iterrows():
            if r.study in used_studies or k >= n_per:
                continue
            if not Path(r.file_path).exists():
                continue
            used_studies.add(r.study)
            chosen.append({"file_path": r.file_path, "label": crit, "region": r.region, "oof_stacked": float(r.oof_stacked)})
            k += 1
        if k < n_per:
            print(f"[warn] для {crit} найдено только {k} кейсов", file=sys.stderr)
    # нормы: все критерии 0 по разметке и pred_label 0 во всех OOF-файлах региона
    all_oof = pd.concat([pd.read_csv(p) for p in sorted((root / "models").glob("oof_stacked_*.csv"))], ignore_index=True)
    pred_any = all_oof.groupby("file_path").pred_label.max()
    for region_group, regions in (("spine", ["spine"]), ("hip", ["right_hip", "left_hip"])):
        k = 0
        norm = lab[(lab.region.isin(regions)) & (lab.quality_class == 0)]
        norm = norm[norm.apply(lambda r: all(float(r.get(c) or 0) == 0.0 for c in CRIT_COLS[str(r["region"])]), axis=1)]
        for _, r in norm.iterrows():
            if k >= n_norm or r.study in used_studies or pred_any.get(r.file_path, 1) != 0 or not Path(r.file_path).exists():
                continue
            used_studies.add(r.study)
            chosen.append({"file_path": r.file_path, "label": "ok", "region": r.region, "oof_stacked": None})
            k += 1
    return chosen


def run_engine(src: Path, cases, workdir: Path):
    """Прогон тем же кодом, что и API: копии под именами case01.dcm…, затем _run_job/_details."""
    out_dir = workdir / "out"
    os.environ.setdefault("DENSITO_OUTPUT_DIR", str(out_dir))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(src))
    import api_server  # noqa: E402  (импорт после установки переменных окружения)
    api_server.JOBS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="casebook_", dir=workdir))
    for i, c in enumerate(cases, 1):
        shutil.copy2(c["file_path"], tmp / f"case{i:02d}.dcm")
    job = "casebook_" + time.strftime("%Y%m%d_%H%M%S")
    job_dir = api_server.JOBS_DIR / job
    job_dir.mkdir(parents=True, exist_ok=True)
    eng, out_csv, rows, debug_rows = api_server._run_job(job, tmp, job_dir, False)
    extras_rows = list(getattr(eng, "last_extras_rows", []) or [])
    by_stem = {}
    for i, r in enumerate(rows):
        rb = api_server._attach_bonus(r, job)
        dbg = debug_rows[i] if i < len(debug_rows) else {}
        rb["details"] = api_server._details(r, dbg, eng.cfg)
        if i < len(extras_rows) and isinstance(extras_rows[i], dict):
            rb["details"]["extras"] = api_server._json_safe(extras_rows[i])
        by_stem[Path(str(r.get("path_to_study", ""))).stem] = rb
    meta = {"model_version": getattr(api_server, "PIPELINE_VERSION", "—"), "config_hash": api_server._config_hash(eng.cfg)}
    return by_stem, meta


def anonymise(rb: dict, idx: int, case: dict) -> dict:
    """Убираем всё, что может идентифицировать файл или пациента."""
    out = {k: v for k, v in rb.items() if k not in ("bonus_sr_dcm_download", "study_sr_download", "bonus_overlay_png_url", "bonus_roi_png_url")}
    out["path_to_study"] = f"кейс {idx}"
    out["study_uid"] = "—"
    out["image_uid"] = "—"
    d = dict(out.get("details") or {})
    x = dict(d.get("extras") or {})
    for k in ("file", "image_uid", "study_uid", "study_warnings"):  # кейсы — одиночные кадры, предупреждения исследования не показываем
        x.pop(k, None)
    d["extras"] = x
    out["details"] = d
    out["_label"] = LABEL_TEXT.get(case["label"], case["label"])
    out["_label_code"] = case["label"]
    return out


def _extract(text: str, begin: str, end: str) -> str:
    i, j = text.index(begin), text.index(end)
    return text[i:j + len(end)]


def build_html(index_html: Path, actions_json: Path, rows: list, meta: dict, fonts_dir: Path | None) -> str:
    src = index_html.read_text(encoding="utf-8")
    style = re.search(r"<style>([\s\S]*?)</style>", src).group(1)
    if fonts_dir and fonts_dir.exists():
        def _font(m):
            p = fonts_dir / Path(m.group(1)).name
            if p.exists():
                return "url(data:font/woff;base64," + base64.b64encode(p.read_bytes()).decode("ascii") + ")"
            return m.group(0)
        style = re.sub(r"url\((?:'|\")?(assets/fonts/[^)'\"]+)(?:'|\")?\)", _font, style)
    else:
        style = re.sub(r"url\((?:'|\")?assets/fonts/[^)]+\)", "local('sans-serif')", style)
    script = re.search(r"<script>([\s\S]*)</script>", src).group(1)
    utils = _extract(script, "// CARD-UTILS:BEGIN", "// CARD-UTILS:END")
    card = _extract(script, "// CARD:BEGIN", "// CARD:END")
    actions = json.loads(actions_json.read_text(encoding="utf-8")) if actions_json.exists() else None
    n = len(rows)
    n_viol = sum(1 for r in rows if str(r.get("quality_class")) == "1")
    data_json = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    actions_js = json.dumps(actions, ensure_ascii=False).replace("</", "<\\/") if actions else "null"
    page_css = """
.cb-wrap { max-width: 1240px; margin: 0 auto; padding: 32px 24px 80px; }
.cb-head { margin-bottom: 28px; }
.cb-head h1 { font-family: var(--font-display); font-size: 30px; margin: 0 0 8px; letter-spacing: -.01em; }
.cb-head p { color: var(--muted); max-width: 900px; font-size: 15px; margin: 6px 0; }
.cb-nav { display: flex; gap: 6px; flex-wrap: wrap; margin: 14px 0 0; }
.cb-nav a { display: inline-block; padding: 5px 11px; border-radius: 999px; border: 1px solid var(--border); font-size: 13px; font-weight: 600; text-decoration: none; color: var(--text); background: var(--surface); }
.cb-nav a.retake { border-color: var(--error); color: var(--error); }
.cb-nav a.review { border-color: var(--warning); color: var(--warning); }
.cb-nav a.ok { border-color: var(--success); color: var(--success); }
.case { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); box-shadow: var(--shadow); padding: 22px 26px; margin-bottom: 22px; }
.case-head { display: flex; gap: 14px; align-items: baseline; flex-wrap: wrap; margin-bottom: 14px; }
.case-head h2 { font-family: var(--font-display); font-size: 22px; margin: 0; }
.case-head .lbl { color: var(--muted); font-size: 14px; }
.case-head .lbl b { color: var(--text); font-weight: 600; }
.case .detail { border-top: none; padding-top: 0; margin-top: 0; }
.cb-conf { background: var(--warning-soft); color: var(--warning); border-radius: 10px; padding: 10px 14px; font-size: 13.5px; font-weight: 600; margin-top: 14px; display: inline-block; }
.lightbox { position: fixed; inset: 0; background: rgba(0,0,0,.85); display: none; align-items: center; justify-content: center; z-index: 50; cursor: zoom-out; }
.lightbox.open { display: flex; }
.lightbox img { max-width: 94vw; max-height: 94vh; image-rendering: pixelated; }
"""
    body_js = f"""
'use strict';
{utils}
{card}
const CASES = JSON.parse(document.getElementById('cases-data').textContent);
const ACTIONS = {actions_js};
if (ACTIONS && ACTIONS.actions && ACTIONS.measurement_norms) CARD_TEXT = ACTIONS;
const lightbox = document.getElementById('lightbox');
function openLightbox(src) {{ lightbox.querySelector('img').src = src; lightbox.classList.add('open'); }}
lightbox.addEventListener('click', () => lightbox.classList.remove('open'));
document.addEventListener('keydown', e => {{ if (e.key === 'Escape') lightbox.classList.remove('open'); }});
const host = document.getElementById('cases');
const nav = document.getElementById('nav');
CASES.forEach((r, i) => {{
  const dec = decision(r);
  const d = r.details || {{}};
  const regionText = SIDE[d.internal_region] ? `${{SIDE[d.internal_region]}} бедро` : (REGION_SHORT[r.anatomical_region] || r.anatomical_region || '—');
  nav.appendChild(el('a', {{ href: '#case' + (i + 1), class: dec.code, text: `${{i + 1}} · ${{dec.label}}` }}));
  const art = el('article', {{ class: 'case', id: 'case' + (i + 1) }});
  art.appendChild(el('div', {{ class: 'case-head' }}, [
    el('h2', {{ text: `Кейс ${{i + 1}}` }}),
    el('span', {{ class: 'lbl' }}, [document.createTextNode(regionText + ' · разметка: '), el('b', {{ text: r._label || '—' }})]),
  ]));
  const left = el('div'); left.appendChild(vizBlock(r, dec, openLightbox));
  const right = el('div'); right.appendChild(decisionCard(r));
  art.appendChild(el('div', {{ class: 'detail' }}, [left, right]));
  host.appendChild(art);
}});
"""
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>DensitoAI — casebook: карточки решения на {n} кейсах</title>
<style>{style}
{page_css}</style>
</head>
<body>
<div class="cb-wrap">
  <header class="cb-head">
    <h1>Casebook: карточки решения на {n} кейсах</h1>
    <p>Офлайн-подборка для врача и лаборанта: те же карточки решения, что в кабинете, на кейсах из размеченной выборки — по два примера на каждый критерий с одиночным нарушением по разметке (правильно распознанные моделью на отложенных фолдах) и две нормы. Из {n} кейсов с нарушением по решению модели: {n_viol}.</p>
    <p>Кейсы обезличены: имена файлов и идентификаторы исследований не показываются. Предупреждения уровня исследования (дубликаты, состав исследования) не приводятся, так как каждый кейс — одиночный кадр. Выбор кейсов детерминирован скриптом tools/web/build_casebook.py; версия модели {html.escape(str(meta.get('model_version')))}, конфигурация {html.escape(str(meta.get('config_hash')))[:12]}, дата сборки {time.strftime('%Y-%m-%d')}.</p>
    <div class="cb-nav" id="nav"></div>
    <div class="cb-conf">Файл содержит изображения из конкурсного набора данных и не предназначен для публикации.</div>
  </header>
  <div id="cases"></div>
</div>
<div class="lightbox" id="lightbox" role="dialog" aria-label="Увеличенное изображение"><img alt="увеличенное изображение"></div>
<script type="application/json" id="cases-data">{data_json}</script>
<script>{body_js}</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2], help="корень репозитория (data/, models/, web/)")
    ap.add_argument("--src", type=Path, default=None, help="папка src (по умолчанию root/src)")
    ap.add_argument("--index", type=Path, default=None, help="web/index.html, откуда берутся стили и код карточки")
    ap.add_argument("--actions", type=Path, default=None, help="web/assets/actions.json")
    ap.add_argument("--out", type=Path, required=True, help="куда писать casebook.html (вне публичного репозитория)")
    ap.add_argument("--workdir", type=Path, default=None, help="рабочая папка прогона (по умолчанию рядом с --out)")
    ap.add_argument("--n-per-criterion", type=int, default=2)
    ap.add_argument("--n-normals", type=int, default=1, help="норм на регион (позвоночник, бедро)")
    ap.add_argument("--no-fonts", action="store_true", help="не встраивать шрифты (меньше размер)")
    a = ap.parse_args()
    root = a.root.resolve()
    src = (a.src or root / "src").resolve()
    index_html = (a.index or root / "web" / "index.html").resolve()
    actions_json = (a.actions or root / "web" / "assets" / "actions.json").resolve()
    workdir = (a.workdir or a.out.parent / "casebook_work").resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    cases = select_cases(root, a.n_per_criterion, a.n_normals)
    print(f"выбрано кейсов: {len(cases)}")
    for i, c in enumerate(cases, 1):
        print(f"  case{i:02d}: {c['region']:9s} {c['label']:8s} oof={c['oof_stacked']}")
    by_stem, meta = run_engine(src, cases, workdir)
    rows = []
    for i, c in enumerate(cases, 1):
        rb = by_stem.get(f"case{i:02d}")
        if rb is None:
            print(f"[warn] нет строки результата для case{i:02d}", file=sys.stderr)
            continue
        rows.append(anonymise(rb, i, c))
    fonts_dir = None if a.no_fonts else index_html.parent / "assets" / "fonts"
    out_html = build_html(index_html, actions_json, rows, meta, fonts_dir)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(out_html, encoding="utf-8")
    # список выбранных файлов — только в рабочей папке (не в HTML)
    (workdir / "selected_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"casebook -> {a.out} ({a.out.stat().st_size / 1024:.0f} КБ), {len(rows)} кейсов")
    return 0


if __name__ == "__main__":
    sys.exit(main())
