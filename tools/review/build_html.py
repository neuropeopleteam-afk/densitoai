#!/usr/bin/env python
"""
Сборка офлайн-страниц для рентгенолога из ключей отбора (select_frames.py):
  out/review_gallery.html  — слепая галерея «ответ до показа модели»
  out/landmarks_form.html  — форма разметки 4 ориентиров на 40 кадрах
Изображения встраиваются как base64 PNG (рендер = inference.normalize_pixels, без оверлеев).
"""
import base64
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def service_build():
    """Версия сервиса и дата сборки набора — печатаются в шапке и попадают в выгрузку,
    чтобы ответы врача были привязаны к конкретной сборке модели."""
    import datetime, os, re as _re
    root = Path(os.environ.get("DENSITO_ROOT", HERE.parents[1]))
    ver = "неизвестна"
    cfg = root / "config.yaml"
    if cfg.exists():
        m = _re.search(r"^version:\s*['\"]?([0-9][^'\"\s]*)", cfg.read_text(encoding="utf-8"), _re.M)
        if m:
            ver = m.group(1)
    return {"service_version": ver, "built_at": datetime.date.today().isoformat()}
OUT = HERE / "out"

CRIT_QUESTIONS = {
    "spine": [("sp_pos", "Укладка некорректна?", "Укладка"),
              ("sp_axis", "Ось позвоночника не выравнена?", "Ось"),
              ("sp_art", "Посторонние предметы в кадре?", "Посторонние предметы")],
    "hip": [("hip_pos", "Укладка некорректна?", "Укладка"),
            ("hip_roi", "Область интереса некорректна?", "Область интереса")],
}
AREA_RU = {"spine": "Позвоночник", "hip": "Бедро"}

LANDMARKS = {
    "spine": [
        ("axis_top", "Верхняя точка оси — центр тела L1"),
        ("axis_bottom", "Нижняя точка оси — центр тела L4"),
        ("body_left", "Левый (на экране) край тела позвонка на середине столба L1–L4"),
        ("body_right", "Правый (на экране) край тела позвонка в той же строке"),
    ],
    "hip": [
        ("gt_apex", "Верхушка большого вертела"),
        ("lt", "Малый вертел (вершина выступа на медиальном контуре)"),
        ("head_center", "Центр головки бедренной кости"),
        ("shaft_axis", "Точка оси диафиза (середина между кортикальными краями) ниже малого вертела"),
    ],
}


def load_cache():
    z = np.load(OUT / "frames_cache.npz", allow_pickle=False)
    paths = list(z["__paths__"])
    return {p: z[hashlib.sha1(p.encode()).hexdigest()] for p in paths}


def png_b64(img):
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def b64json(obj):
    return base64.b64encode(json.dumps(obj, ensure_ascii=False).encode()).decode()


# --------------------------------------------------------------------------- #
def build_gallery(key: pd.DataFrame, imgs):
    frames = []
    for _, r in key.iterrows():
        area = r.area
        model = {}
        for c, _, short in CRIT_QUESTIONS[area]:
            v = r[f"oof_{c}"]
            model[c] = None if pd.isna(v) else int(v)
        frames.append({
            "id": r.display_id,
            "area": area,
            "area_ru": AREA_RU[area],
            "rows": int(r.img_rows), "cols": int(r.img_cols),
            "png": png_b64(imgs[r.file_path]),
            "m": b64json(model),   # предсказание модели, раскрывается после фиксации ответа
        })
    payload = json.dumps({"frames": frames, "questions": CRIT_QUESTIONS, "build": service_build()}, ensure_ascii=False)
    html = GALLERY_TEMPLATE.replace("__PAYLOAD__", payload).replace("__N__", str(len(frames)))
    (OUT / "review_gallery.html").write_text(html, encoding="utf-8")
    return len(frames)


def build_landmarks(key: pd.DataFrame, imgs):
    frames = []
    for _, r in key.iterrows():
        frames.append({
            "id": r.display_id, "area": r.area, "area_ru": AREA_RU[r.area],
            "rows": int(r.img_rows), "cols": int(r.img_cols),
            "png": png_b64(imgs[r.file_path]),
        })
    payload = json.dumps({"frames": frames, "landmarks": LANDMARKS, "build": service_build()}, ensure_ascii=False)
    html = LANDMARKS_TEMPLATE.replace("__PAYLOAD__", payload).replace("__N__", str(len(frames)))
    (OUT / "landmarks_form.html").write_text(html, encoding="utf-8")
    return len(frames)


COMMON_CSS = """
*{box-sizing:border-box}
body{font-family:system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;margin:0;background:#f3f4f6;color:#111827;font-size:15px}
header{position:sticky;top:0;z-index:10;background:#fff;border-bottom:1px solid #e5e7eb;padding:12px 20px}
h1{font-size:19px;margin:0 0 6px}
.intro{font-size:13.5px;color:#374151;max-width:1100px;line-height:1.45;margin:0 0 8px}
.intro ol{margin:4px 0 0 18px;padding:0}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.progress{flex:1;min-width:200px;height:10px;background:#e5e7eb;border-radius:5px;overflow:hidden}
.progress > div{height:100%;background:#2563eb;width:0}
button{font:inherit;padding:6px 12px;border:1px solid #9ca3af;border-radius:6px;background:#fff;cursor:pointer}
button:hover{background:#f9fafb}
button:disabled{opacity:.45;cursor:not-allowed}
button.primary{background:#2563eb;border-color:#2563eb;color:#fff}
button.primary:hover{background:#1d4ed8}
main{padding:16px 20px;display:grid;gap:16px}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px 16px;display:grid;grid-template-columns:auto 1fr;gap:18px}
.card.done{border-color:#10b981}
.card.skipped{border-color:#f59e0b}
.imgwrap{background:#000;display:inline-block;line-height:0;position:relative;border-radius:4px;overflow:hidden}
.imgwrap img,.imgwrap canvas{display:block;image-rendering:auto}
.inv img,.inv canvas{filter:invert(1)}
.tools{display:flex;gap:6px;margin-top:6px;flex-wrap:wrap}
.tools button{padding:3px 9px;font-size:13px}
.meta{font-weight:600;font-size:16px;margin-bottom:8px}
.meta span{color:#6b7280;font-weight:400;font-size:13px;margin-left:8px}
.q{margin:6px 0;padding:6px 8px;border-radius:6px;background:#f9fafb}
.q .t{margin-bottom:4px}
.q label{margin-right:14px;cursor:pointer}
textarea{width:100%;min-height:44px;font:inherit;padding:6px;border:1px solid #d1d5db;border-radius:6px}
.model{margin-top:10px;padding:10px;border-radius:8px;background:#eff6ff;border:1px solid #bfdbfe}
.model .t{font-weight:600;margin-bottom:6px}
.small{font-size:13px;color:#6b7280}
.hidden{display:none}
.tag{display:inline-block;font-size:12px;padding:2px 7px;border-radius:10px;background:#e5e7eb;margin-left:6px}
.tag.ok{background:#d1fae5;color:#065f46}
.tag.warn{background:#fef3c7;color:#92400e}
"""

GALLERY_TEMPLATE = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<title>Слепая ревизия DXA — галерея</title>
<style>""" + COMMON_CSS + """</style></head>
<body>
<header>
<h1>Слепая ревизия качества DXA-снимков (GE Lunar Prodigy)</h1>
<div class="intro">
1. Ниже __N__ кадров: позвоночник (L1–L4) и проксимальный отдел бедра. Для каждого кадра ответьте на вопросы по критериям области, при необходимости добавьте комментарий.<br>
2. Варианты ответа: «да» — нарушение есть, «нет» — нарушения нет, «не уверен» — по снимку решить нельзя.<br>
3. Нажмите «Зафиксировать ответ». После этого ответ изменить нельзя, под ним появится заключение модели — отметьте, согласны ли вы с ним.<br>
4. Ответы сохраняются в браузере автоматически; страницу можно закрывать и открывать снова (тот же браузер, тот же файл).<br>
5. Кнопки под снимком: инверсия яркости и масштаб. По умолчанию снимок показан в анатомических пропорциях (пиксель 1,05 × 0,6 мм); переключатель — в строке ниже.<br>
6. Сторона бедра видна на снимке, подсказка её не содержит. Кадры могут повторяться — это часть протокола, оценивайте каждый независимо.<br>
7. По окончании нажмите «Экспорт ответов (JSON)» и «Экспорт ответов (CSV)» и передайте оба файла. Ориентировочное время — 2–3 часа, можно делать в несколько подходов.
</div>
<div class="small" id="buildline"></div>
<div class="bar">
  <div class="progress"><div id="pbar"></div></div>
  <div id="ptext" class="small"></div>
  <label class="small"><input type="checkbox" id="onlyOpen"> только без ответа</label>
  <label class="small"><input type="checkbox" id="anat" checked> анатомические пропорции (1,05/0,6 мм)</label>
  <button id="expJson">Экспорт ответов (JSON)</button>
  <button id="expCsv">Экспорт ответов (CSV)</button>
</div>
</header>
<main id="main"></main>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
(function(){
const P = JSON.parse(document.getElementById('payload').textContent);
if (P.build && document.getElementById('buildline')) document.getElementById('buildline').textContent =
  'Набор собран ' + P.build.built_at + ' для сервиса DensitoAI ' + P.build.service_version + '. Программа не является медицинским изделием: оценивается техническое качество укладки, не диагноз.';
const KEY = 'densito_review_v1';
const ANS = {yes:'да', no:'нет', unsure:'не уверен'};
const AGREE = {yes:'да', no:'нет', partial:'частично'};
let S = {};
let STORAGE_OK = true;
try { S = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch(e) { S = {}; STORAGE_OK = false; }
// Если браузер запрещает сохранение (приватный режим, запрет данных сайтов), страница
// обязана продолжать работать: ответы держим в памяти и предупреждаем врача, что
// закрывать страницу до выгрузки нельзя. Без try/catch любой щелчок ломал бы отрисовку.
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(S)); } catch(e) { STORAGE_OK = false; storageWarn(); } };
function storageWarn() {
  let w = document.getElementById('storagewarn');
  if (!w) {
    w = document.createElement('div');
    w.id = 'storagewarn';
    w.style.cssText = 'background:#fef3c7;border:1px solid #f59e0b;color:#92400e;padding:8px 12px;margin:8px 0;border-radius:6px;font-size:13px';
    w.textContent = 'Браузер запретил сохранение на этой странице: ответы держатся только в памяти. Не закрывайте и не перезагружайте страницу, пока не нажмёте кнопки экспорта.';
    const h = document.querySelector('header'); if (h) h.appendChild(w);
  }
}
if (!STORAGE_OK) setTimeout(storageWarn, 0);
const nowIso = () => new Date().toISOString();
const dec = b => JSON.parse(decodeURIComponent(escape(atob(b))));
let zoom = {}, inv = {};
const st = id => (S[id] = S[id] || {answers:{}, comment:'', fixed_at:null, agree_model:null, comment_model:'', opened_at: nowIso()});

function render(){
  const main = document.getElementById('main'); main.innerHTML = '';
  const onlyOpen = document.getElementById('onlyOpen').checked;
  const anat = document.getElementById('anat').checked;
  P.frames.forEach((f, i) => {
    const s = st(f.id);
    if (onlyOpen && s.fixed_at) return;
    const z = zoom[f.id] || 2;
    const card = document.createElement('div'); card.className = 'card' + (s.fixed_at ? ' done' : '');
    const w = f.cols * z, h = Math.round(f.rows * z * (anat ? 1.75 : 1));
    const qs = P.questions[f.area];
    let html = `<div><div class="meta">${f.id} <span>${i+1} из ${P.frames.length}</span></div>
      <div class="imgwrap ${inv[f.id] ? 'inv' : ''}"><img src="data:image/png;base64,${f.png}" width="${w}" height="${h}" draggable="false"></div>
      <div class="tools"><button data-a="inv">Инверсия</button><button data-a="z-">−</button><span class="small">×${z}</span><button data-a="z+">+</button></div></div>`;
    html += `<div><div class="meta">Область: ${f.area_ru}</div>`;
    qs.forEach(([c, q]) => {
      html += `<div class="q"><div class="t">${q}</div>`;
      for (const k of ['yes','no','unsure']) {
        const ch = s.answers[c] === k ? 'checked' : '';
        html += `<label><input type="radio" name="${f.id}_${c}" value="${k}" ${ch} ${s.fixed_at ? 'disabled' : ''}> ${ANS[k]}</label>`;
      }
      html += `</div>`;
    });
    html += `<div class="q"><div class="t">Комментарий (необязательно)</div><textarea data-f="comment" ${s.fixed_at ? 'disabled' : ''}>${esc(s.comment)}</textarea></div>`;
    if (!s.fixed_at) {
      const ready = qs.every(([c]) => s.answers[c]);
      html += `<button class="primary" data-a="fix" ${ready ? '' : 'disabled'}>Зафиксировать ответ</button> <span class="small">${ready ? '' : 'ответьте на все вопросы'}</span>`;
    } else {
      const m = dec(f.m);
      html += `<div class="model"><div class="t">Модель (ответ зафиксирован ${fmt(s.fixed_at)}):</div>`;
      qs.forEach(([c, q, short]) => {
        const v = m[c];
        const txt = v === null ? 'нет данных' : (v === 1 ? 'нарушение' : 'норма');
        const mine = s.answers[c];
        const agree = v === null || mine === 'unsure' ? '' : ((mine === 'yes') === (v === 1) ? '<span class="tag ok">совпадает с вашим ответом</span>' : '<span class="tag warn">не совпадает с вашим ответом</span>');
        html += `<div>${short}: <b>${txt}</b> ${agree}</div>`;
      });
      html += `<div class="q" style="margin-top:8px"><div class="t">Согласны с моделью?</div>`;
      for (const k of ['yes','no','partial']) {
        html += `<label><input type="radio" name="${f.id}_agree" value="${k}" ${s.agree_model === k ? 'checked' : ''}> ${AGREE[k]}</label>`;
      }
      html += `</div><div class="q"><div class="t">Комментарий к заключению модели (необязательно)</div><textarea data-f="comment_model">${esc(s.comment_model)}</textarea></div></div>`;
    }
    html += `</div>`;
    card.innerHTML = html;
    card.addEventListener('change', e => {
      const t = e.target;
      if (t.type === 'radio') {
        const nm = t.name.slice(f.id.length + 1);
        if (nm === 'agree') { s.agree_model = t.value; s.agree_at = nowIso(); }
        else if (!s.fixed_at) s.answers[nm] = t.value;
        save(); render();
      }
    });
    card.addEventListener('input', e => {
      const t = e.target;
      if (t.tagName === 'TEXTAREA') { s[t.dataset.f] = t.value; save(); }
    });
    card.addEventListener('click', e => {
      const a = e.target.dataset && e.target.dataset.a; if (!a) return;
      if (a === 'inv') inv[f.id] = !inv[f.id];
      if (a === 'z+') zoom[f.id] = Math.min(5, z + 1);
      if (a === 'z-') zoom[f.id] = Math.max(1, z - 1);
      if (a === 'fix') {
        if (!qs.every(([c]) => s.answers[c])) return;
        if (!confirm('Зафиксировать ответ по кадру ' + f.id + '? Изменить его будет нельзя.')) return;
        s.fixed_at = nowIso();
      }
      save(); render();
    });
    main.appendChild(card);
  });
  const done = P.frames.filter(f => st(f.id).fixed_at).length;
  document.getElementById('pbar').style.width = (100 * done / P.frames.length) + '%';
  document.getElementById('ptext').textContent = `зафиксировано ${done} из ${P.frames.length}`;
}
function esc(s){ return (s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;'); }
function fmt(iso){ try { return new Date(iso).toLocaleString('ru-RU'); } catch(e) { return iso; } }
function download(name, text, mime){
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], {type: mime})); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
function rows(){
  return P.frames.map(f => { const s = st(f.id); return {
    display_id: f.id, area: f.area, answers: s.answers, comment: s.comment || '',
    fixed_at: s.fixed_at, agree_model: s.agree_model, comment_model: s.comment_model || '', opened_at: s.opened_at }; });
}
document.getElementById('expJson').onclick = () => download('review_answers.json',
  JSON.stringify({exported_at: nowIso(), build: P.build || null, n_frames: P.frames.length, answers: rows()}, null, 1), 'application/json');
document.getElementById('expCsv').onclick = () => {
  const cols = ['display_id','area','sp_pos','sp_axis','sp_art','hip_pos','hip_roi','comment','fixed_at','agree_model','comment_model'];
  const q = v => '"' + String(v === null || v === undefined ? '' : v).replace(/"/g,'""') + '"';
  const lines = [cols.join(',')];
  rows().forEach(r => lines.push(cols.map(c => q(c in r ? r[c] : (r.answers[c] || ''))).join(',')));
  download('review_answers.csv', '\\ufeff' + lines.join('\\n'), 'text/csv');
};
document.getElementById('onlyOpen').onchange = render;
document.getElementById('anat').onchange = render;
render();
})();
</script>
</body></html>
"""

LANDMARKS_TEMPLATE = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<title>Разметка ориентиров DXA</title>
<style>""" + COMMON_CSS + """
.lm{margin:4px 0;padding:5px 8px;border-radius:6px;background:#f9fafb;display:flex;gap:8px;align-items:center}
.lm.next{background:#dbeafe;font-weight:600}
.lm.set{background:#d1fae5}
.dot{width:12px;height:12px;border-radius:6px;display:inline-block;flex:none}
.imgwrap{cursor:crosshair}
</style></head>
<body>
<header>
<h1>Разметка ориентиров на DXA-снимках (микроэталон, 40 кадров)</h1>
<div class="intro">
1. На каждом кадре поставьте четыре точки в указанном порядке: щёлкните по снимку, точка ставится для выделенного ориентира.<br>
2. Позвоночник: центр тела L1, центр тела L4, левый и правый (на экране) край тела позвонка на середине столба L1–L4 — оба в одной строке.<br>
3. Бедро: верхушка большого вертела, малый вертел, центр головки бедренной кости, точка оси диафиза (середина между кортикальными краями) ниже малого вертела.<br>
4. «Отменить точку» убирает последнюю поставленную точку, «Пропустить кадр» — если ориентиры не видны или кадр непригоден (при желании укажите причину в комментарии).<br>
5. Для точности увеличьте масштаб (+) и при необходимости включите инверсию. Снимок показан в анатомических пропорциях (пиксель 1,05 × 0,6 мм). Точки сохраняются в браузере автоматически.<br>
6. По окончании нажмите «Экспорт (JSON)» и «Экспорт (CSV)» и передайте оба файла. Ориентировочное время — около часа.
</div>
<div class="small" id="buildline"></div>
<div class="bar">
  <div class="progress"><div id="pbar"></div></div>
  <div id="ptext" class="small"></div>
  <label class="small"><input type="checkbox" id="onlyOpen"> только незавершённые</label>
  <label class="small"><input type="checkbox" id="anat" checked> анатомические пропорции (1,05/0,6 мм)</label>
  <button id="expJson">Экспорт (JSON)</button>
  <button id="expCsv">Экспорт (CSV)</button>
</div>
</header>
<main id="main"></main>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
(function(){
const P = JSON.parse(document.getElementById('payload').textContent);
if (P.build && document.getElementById('buildline')) document.getElementById('buildline').textContent =
  'Набор собран ' + P.build.built_at + ' для сервиса DensitoAI ' + P.build.service_version + '. Программа не является медицинским изделием: оценивается техническое качество укладки, не диагноз.';
const KEY = 'densito_landmarks_v1';
const COLORS = ['#ef4444','#f59e0b','#22c55e','#3b82f6'];
let S = {};
let STORAGE_OK = true;
try { S = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch(e) { S = {}; STORAGE_OK = false; }
// Если браузер запрещает сохранение (приватный режим, запрет данных сайтов), страница
// обязана продолжать работать: ответы держим в памяти и предупреждаем врача, что
// закрывать страницу до выгрузки нельзя. Без try/catch любой щелчок ломал бы отрисовку.
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(S)); } catch(e) { STORAGE_OK = false; storageWarn(); } };
function storageWarn() {
  let w = document.getElementById('storagewarn');
  if (!w) {
    w = document.createElement('div');
    w.id = 'storagewarn';
    w.style.cssText = 'background:#fef3c7;border:1px solid #f59e0b;color:#92400e;padding:8px 12px;margin:8px 0;border-radius:6px;font-size:13px';
    w.textContent = 'Браузер запретил сохранение на этой странице: ответы держатся только в памяти. Не закрывайте и не перезагружайте страницу, пока не нажмёте кнопки экспорта.';
    const h = document.querySelector('header'); if (h) h.appendChild(w);
  }
}
if (!STORAGE_OK) setTimeout(storageWarn, 0);
const nowIso = () => new Date().toISOString();
let zoom = {}, inv = {};
const st = id => (S[id] = S[id] || {points:{}, order:[], skipped:false, comment:'', updated_at:null});
const IMG = {};  // кэш загруженных Image

function isDone(f){ const s = st(f.id); return s.skipped || P.landmarks[f.area].every(([k]) => s.points[k]); }

function draw(f, canvas){
  const s = st(f.id), z = zoom[f.id] || 2, lms = P.landmarks[f.area];
  const ay = document.getElementById('anat').checked ? 1.75 : 1;   // 1.05 / 0.6
  const ctx = canvas.getContext('2d');
  const im = IMG[f.id];
  const go = () => {
    canvas.width = f.cols * z; canvas.height = Math.round(f.rows * z * ay);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(im, 0, 0, canvas.width, canvas.height);
    lms.forEach(([k], j) => {
      const p = s.points[k]; if (!p) return;
      const x = (p.x + 0.5) * z, y = (p.y + 0.5) * z * ay;
      ctx.strokeStyle = COLORS[j]; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(x, y, 5, 0, 2*Math.PI); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x-9,y); ctx.lineTo(x+9,y); ctx.moveTo(x,y-9); ctx.lineTo(x,y+9); ctx.stroke();
      ctx.fillStyle = COLORS[j]; ctx.font = 'bold 12px sans-serif'; ctx.fillText(String(j+1), x+7, y-7);
    });
  };
  if (im.complete && im.naturalWidth) go(); else im.onload = go;
}

function render(){
  const main = document.getElementById('main'); main.innerHTML = '';
  const onlyOpen = document.getElementById('onlyOpen').checked;
  P.frames.forEach((f, i) => {
    const s = st(f.id);
    if (onlyOpen && isDone(f)) return;
    if (!IMG[f.id]) { const im = new Image(); im.src = 'data:image/png;base64,' + f.png; IMG[f.id] = im; }
    const z = zoom[f.id] || 2, lms = P.landmarks[f.area];
    const next = s.skipped ? -1 : lms.findIndex(([k]) => !s.points[k]);
    const card = document.createElement('div');
    card.className = 'card' + (s.skipped ? ' skipped' : (next === -1 ? ' done' : ''));
    let html = `<div><div class="meta">${f.id} <span>${i+1} из ${P.frames.length}</span></div>
      <div class="imgwrap ${inv[f.id] ? 'inv' : ''}"><canvas width="${f.cols*z}" height="${f.rows*z}"></canvas></div>
      <div class="tools"><button data-a="inv">Инверсия</button><button data-a="z-">−</button><span class="small">×${z}</span><button data-a="z+">+</button></div></div>`;
    html += `<div><div class="meta">Область: ${f.area_ru}</div>`;
    lms.forEach(([k, name], j) => {
      const p = s.points[k];
      html += `<div class="lm ${p ? 'set' : (j === next ? 'next' : '')}"><span class="dot" style="background:${COLORS[j]}"></span>
        <span>${j+1}. ${name}</span><span class="small" style="margin-left:auto">${p ? `x=${p.x.toFixed(1)}, y=${p.y.toFixed(1)}` : (j === next ? 'щёлкните по снимку' : '')}</span></div>`;
    });
    html += `<div class="tools" style="margin-top:8px">
      <button data-a="undo" ${s.order.length ? '' : 'disabled'}>Отменить точку</button>
      <button data-a="skip">${s.skipped ? 'Вернуть кадр' : 'Пропустить кадр'}</button>
      <button data-a="clear" ${s.order.length ? '' : 'disabled'}>Очистить кадр</button></div>`;
    html += `<div class="q" style="margin-top:8px"><div class="t">Комментарий (необязательно)</div><textarea data-f="comment">${esc(s.comment)}</textarea></div></div>`;
    card.innerHTML = html;
    const canvas = card.querySelector('canvas');
    draw(f, canvas);
    canvas.addEventListener('click', e => {
      if (s.skipped) return;
      const nx = lms.findIndex(([k]) => !s.points[k]); if (nx === -1) return;
      const r = canvas.getBoundingClientRect();
      // координаты в пикселях исходного изображения (учитывается фактический масштаб отображения)
      const x = (e.clientX - r.left) * f.cols / r.width - 0.5;
      const y = (e.clientY - r.top) * f.rows / r.height - 0.5;
      const k = lms[nx][0];
      s.points[k] = {x: Math.round(x*10)/10, y: Math.round(y*10)/10, zoom: z, t: nowIso()};
      s.order.push(k); s.updated_at = nowIso(); save(); render();
    });
    card.addEventListener('click', e => {
      const a = e.target.dataset && e.target.dataset.a; if (!a) return;
      if (a === 'inv') inv[f.id] = !inv[f.id];
      if (a === 'z+') zoom[f.id] = Math.min(5, z + 1);
      if (a === 'z-') zoom[f.id] = Math.max(1, z - 1);
      if (a === 'undo') { const k = s.order.pop(); if (k) delete s.points[k]; }
      if (a === 'clear') { s.points = {}; s.order = []; }
      if (a === 'skip') { s.skipped = !s.skipped; }
      s.updated_at = nowIso(); save(); render();
    });
    card.addEventListener('input', e => { if (e.target.tagName === 'TEXTAREA') { s.comment = e.target.value; save(); } });
    main.appendChild(card);
  });
  const done = P.frames.filter(isDone).length, sk = P.frames.filter(f => st(f.id).skipped).length;
  document.getElementById('pbar').style.width = (100 * done / P.frames.length) + '%';
  document.getElementById('ptext').textContent = `завершено ${done} из ${P.frames.length}` + (sk ? `, пропущено ${sk}` : '');
}
function esc(s){ return (s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;'); }
function download(name, text, mime){
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], {type: mime})); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
function rows(){
  return P.frames.map(f => { const s = st(f.id); return {display_id: f.id, area: f.area, rows: f.rows, cols: f.cols,
    skipped: s.skipped, points: s.points, comment: s.comment || '', updated_at: s.updated_at}; });
}
document.getElementById('expJson').onclick = () => download('landmarks.json',
  JSON.stringify({exported_at: nowIso(), build: P.build || null, coordinate_note: 'x, y — пиксели исходного изображения, начало (0,0) — центр левого верхнего пикселя, y растёт вниз', frames: rows()}, null, 1), 'application/json');
document.getElementById('expCsv').onclick = () => {
  const q = v => '"' + String(v === null || v === undefined ? '' : v).replace(/"/g,'""') + '"';
  const lines = ['display_id,area,landmark,x,y,skipped,comment'];
  rows().forEach(r => {
    P.landmarks[r.area].forEach(([k]) => { const p = r.points[k];
      lines.push([r.display_id, r.area, k, p ? p.x : '', p ? p.y : '', r.skipped, r.comment].map(q).join(',')); });
  });
  download('landmarks.csv', '\\ufeff' + lines.join('\\n'), 'text/csv');
};
document.getElementById('onlyOpen').onchange = render;
document.getElementById('anat').onchange = render;
render();
})();
</script>
</body></html>
"""


def main():
    imgs = load_cache()
    key = pd.read_csv(OUT / "review_key.csv")
    lm_key = pd.read_csv(OUT / "landmarks_key.csv")
    n1 = build_gallery(key, imgs)
    n2 = build_landmarks(lm_key, imgs)
    for name in ("review_gallery.html", "landmarks_form.html"):
        p = OUT / name
        print(f"{name}: {p.stat().st_size / 1024:.0f} КБ")
    print("кадров в галерее:", n1, "; в форме ориентиров:", n2)


if __name__ == "__main__":
    main()
