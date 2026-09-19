#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Собирает web/assets/demo_result.json из реального ответа /api/analyze на
tests/sample_test_zip (файлы организаторов «Для теста», без ПДн).

Вход:  work/C/tmp/api_response.json  (сырой ответ POST /api/analyze?xlsx=true)
Выход: work/C/patch/web/assets/demo_result.json

Что делаем с ответом:
  * base64-картинки убираем, вместо них — bonus_overlay_png_url на PNG в assets/demo/;
  * ссылки на файлы запроса (/api/results/...) убираем — демо работает без сервера;
  * csv-текст убираем (в кабинете он не используется);
  * строки, details (критерии, измерения, ROI, действие) — как есть, числа не трогаем;
  * добавляем поле demo: true и подпись источника.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "tmp" / "api_response.json"
DST = HERE / "patch" / "web" / "assets" / "demo_result.json"

# сопоставление имени файла -> PNG оверлея в assets/demo (ascii-имена, без кириллицы в URL)
OVERLAYS = {
    "CR000000_ПОП.dcm": "assets/demo/demo_spine_overlay.png",
    "CR000000_ППОБ.dcm": "assets/demo/demo_right_hip_overlay.png",
    "CR000001_ЛПОБ.dcm": "assets/demo/demo_left_hip_overlay.png",
}
PII_RE = re.compile(r"[A-Za-zА-Яа-яЁё]{3,}\s+[A-Za-zА-Яа-яЁё]\.?\s*[A-Za-zА-Яа-яЁё]?\.?", re.U)  # «Фамилия И. О.»


def main() -> int:
    d = json.loads(SRC.read_text(encoding="utf-8"))
    rows = []
    for r in d["rows"]:
        name = Path(str(r.get("path_to_study", ""))).name
        if PII_RE.search(name.replace("_", " ").replace(".dcm", "")):
            print("ВНИМАНИЕ: имя файла похоже на ФИО:", name, file=sys.stderr)
        out = {k: v for k, v in r.items() if not str(k).endswith("_base64") and k != "bonus_sr_dcm_download"}
        out["path_to_study"] = name
        if name in OVERLAYS:
            out["bonus_overlay_png_url"] = OVERLAYS[name]
            if not (HERE / "patch" / "web" / OVERLAYS[name]).exists():
                print("нет файла оверлея:", OVERLAYS[name], file=sys.stderr)
                return 1
        rows.append(out)
    demo = {
        "demo": True,
        "demo_note": "Три тестовых файла организаторов (tests/sample_test_zip), "
                     "обработаны сервисом DensitoAI 2.1.0 без изменений. Личных данных нет.",
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
    DST.write_text(json.dumps(demo, ensure_ascii=False, indent=1), encoding="utf-8")
    print("OK ->", DST, f"({DST.stat().st_size} байт, {len(rows)} строк)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
