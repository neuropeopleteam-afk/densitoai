#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Собирает web/assets/demo_result.json и PNG-оверлеи web/assets/demo/ из реального
ответа POST /api/analyze на отобранную демо-партию.

Вход:
  tools/web/tmp/api_response_demo10.json  сырой ответ API (с base64-оверлеями)
  /tmp/demo_map.json                      роль каждого кадра в партии (что он иллюстрирует)
Выход:
  web/assets/demo_result.json             ответ без base64 и без ссылок на сервер
  web/assets/demo/<имя>.png               оверлеи, на которые ссылается json

Что делаем с ответом:
  * base64-картинки выносим в PNG-файлы, в строке остаётся bonus_overlay_png_url;
  * ссылки на файлы запроса (/api/results/...) убираем — демо работает без сервера;
  * строки, details (критерии, измерения, ROI, действие) — как есть, числа не трогаем;
  * добавляем demo: true, подпись источника и роль каждого кадра (demo_role).

Партия отобрана по фактическому выводу сервиса на обучающей выборке: пять кадров,
где сервис нашёл нарушение и тип совпал с разметкой эксперта (по одному на каждый
критерий ТЗ), четыре уверенные нормы и один честный пропуск — кадр, где эксперт
нарушение отметил, а сервис остался ниже порога. Пропуск показан намеренно: жюри
видит границу возможностей, а не только удачные случаи.
"""
import base64
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = HERE / "tmp" / "api_response_demo10.json"
MAP = Path("/tmp/demo_map.json")
OUT_JSON = ROOT / "web" / "assets" / "demo_result.json"
OUT_PNG = ROOT / "web" / "assets" / "demo"

PII_RE = re.compile(r"[A-Za-zА-Яа-яЁё]{3,}\s+[A-Za-zА-Яа-яЁё]\.?\s*[A-Za-zА-Яа-яЁё]?\.?", re.U)

ROLE_RU = {
    "поймано: sp_pos": "укладка позвоночника — эксперт отметил, сервис нашёл",
    "поймано: sp_axis": "ось позвоночника — эксперт отметил, сервис нашёл",
    "поймано: sp_art": "посторонние предметы — эксперт отметил, сервис нашёл",
    "поймано: hip_pos": "укладка бедра — эксперт отметил, сервис нашёл",
    "поймано: hip_roi": "область интереса — эксперт отметил, сервис нашёл",
    "норма, сервис согласен": "норма — эксперт и сервис согласны",
    "пропуск: эксперт нашёл, сервис нет": "пропуск: эксперт отметил область интереса, сервис остался ниже порога",
}


def main() -> int:
    d = json.loads(SRC.read_text(encoding="utf-8"))
    roles = {m["name"]: m for m in json.loads(MAP.read_text(encoding="utf-8"))} if MAP.exists() else {}
    OUT_PNG.mkdir(parents=True, exist_ok=True)
    for old in OUT_PNG.glob("*.png"):
        old.unlink()

    # Идентификаторы исследований и снимков заменяем на синтетические: в демо они не нужны,
    # а настоящий UID — это ссылка на запись пациента в архиве заказчика.
    uid_map, img_map = {}, {}
    for r in d["rows"]:
        su, iu = str(r.get("study_uid", "")), str(r.get("image_uid", ""))
        if su and su not in uid_map:
            uid_map[su] = f"demo-study-{len(uid_map) + 1:02d}"
        if iu and iu not in img_map:
            img_map[iu] = f"demo-image-{len(img_map) + 1:02d}"

    rows, n_png = [], 0
    for r in d["rows"]:
        name = Path(str(r.get("path_to_study", ""))).name
        if PII_RE.search(name.replace("_", " ").replace(".dcm", "")):
            print("ВНИМАНИЕ: имя файла похоже на ФИО:", name, file=sys.stderr)
            return 1
        out = {k: v for k, v in r.items() if not str(k).endswith("_base64") and k != "bonus_sr_dcm_download"}
        out["path_to_study"] = name
        out.pop("study_sr_download", None)      # ссылка на файл запроса, вне сервера бессмысленна
        # подмена по всей структуре строки, включая details/extras
        blob_row = json.dumps(out, ensure_ascii=False)
        for real, fake in list(uid_map.items()) + list(img_map.items()):
            blob_row = blob_row.replace(real, fake)
        out = json.loads(blob_row)
        b64 = r.get("bonus_overlay_png_base64")
        if b64:
            png = OUT_PNG / (Path(name).stem.lower() + ".png")
            png.write_bytes(base64.b64decode(b64))
            out["bonus_overlay_png_url"] = f"assets/demo/{png.name}"
            n_png += 1
        role = roles.get(name, {}).get("role", "")
        if role:
            out["demo_role"] = ROLE_RU.get(role, role)
            out["demo_miss"] = role.startswith("пропуск")
        rows.append(out)

    n_viol = sum(1 for r in rows if r.get("quality_class") == 1)
    n_miss = sum(1 for r in rows if r.get("demo_miss"))
    demo = {
        "demo": True,
        "demo_note": (
            f"Партия из {len(rows)} снимков обучающей выборки заказчика, обработана сервисом DensitoAI "
            f"{d.get('model_version', '')} без изменений: {n_viol} с нарушениями по всем пяти критериям ТЗ, "
            f"остальные — норма. Каждый кадр подписан: что отметил эксперт и что сказал сервис. "
            f"Пропусков в партии: {n_miss} — показаны намеренно, чтобы была видна граница возможностей. "
            f"Личных данных в файлах нет."
        ),
        "job_id": "demo",
        "request_id": "demo",
        "model_version": d.get("model_version"),
        "config_hash": d.get("config_hash"),
        "summary": d.get("summary"),
        "format_check": d.get("format_check"),
        "result_csv": None, "result_xlsx": None, "result_debug_csv": None,
        "result_csv_url": None, "result_xlsx_url": None, "result_debug_csv_url": None,
        "rows": rows,
    }
    blob = json.dumps(demo, ensure_ascii=False, indent=1)
    leaked = re.findall(r"1\.2\.(?:840|643)[0-9.]{10,}", blob)
    if leaked:
        print(f"ОТКАЗ: в демо остались настоящие DICOM UID ({len(leaked)} шт., первый {leaked[0][:40]})", file=sys.stderr)
        return 1
    OUT_JSON.write_text(blob, encoding="utf-8")
    print(f"OK -> {OUT_JSON} ({OUT_JSON.stat().st_size} байт, {len(rows)} строк, {n_png} оверлеев, "
          f"{n_viol} с нарушениями, {n_miss} пропуск)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
