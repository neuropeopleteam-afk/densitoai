#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест экспорта сегментации структур (бонус ТЗ «сегментация», src/segmentation_export.py).

Только numpy / cv2 / pydicom, без torch (инференс запускается с --no-embeddings):
  1. прогон src/inference.py --seg-dir на tests/phantoms: для каждой Success-строки созданы
     <stem>_seg.dcm, _seg.png, _seg.json; для Failure-строк ничего не пишется (число файлов = 3 x Success);
  2. официальный results.csv не меняется от флага: 8 колонок (без time_of_processing) байт-в-байт
     совпадают с прогоном без --seg-dir; заголовок — ровно 9 колонок;
  3. tools/validate_seg.py -> PASS для всех SEG (структура, ссылка на исходный снимок, размер маски);
  4. маски непустые: кость > 0 px во всех SEG, число сегментов по региону (позвоночник 2, бедро 3),
     Rows x Columns = размеру исходного снимка из MANIFEST.json, ссылка = image_uid снимка;
  5. детерминизм: повторный прогон даёт байт-в-байт те же SEG и JSON (PNG — тоже);
  6. согласованность файлов: кадры SEG == маски build_masks(), PNG непрозрачен ровно там, где есть маска,
     полигоны JSON лежат внутри кадра, mm = px * (0.6, 1.05);
  7. запрещённые формулировки отсутствуют в модуле, валидаторе, JSON и текстовых полях SEG;
  8. если установлен FastAPI: в строках ответа /api/analyze есть seg_download / seg_png / seg_json_download,
     файлы скачиваются (200), у строк без сегментации ключей seg_* нет.

Запуск: python tests/test_segmentation_export.py   (код возврата 0 — ок).
"""
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2  # noqa: E402
import pydicom  # noqa: E402

import segmentation_export as se  # noqa: E402
import validate_seg  # noqa: E402
from inference import normalize_pixels  # noqa: E402

PHANTOMS = ROOT / "tests" / "phantoms"
# Запрещённые формулировки собираются из частей, чтобы сам тест проходил проверку на их отсутствие в исходниках.
BANNED = tuple("".join(parts) for parts in (("Grad", "-", "CAM"), ("автокоррекц", "ия ROI"),
                                             ("ЕР", "ИС"), ("сколи", "оз"), ("Коб", "ба")))
fails = []


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def run_inference(out_csv: Path, seg_dir=None):
    cmd = [sys.executable, str(ROOT / "src" / "inference.py"), "--input", str(PHANTOMS), "--output", str(out_csv),
           "--no-embeddings", "--debug-csv"]
    if seg_dir is not None:
        cmd += ["--seg-dir", str(seg_dir)]
    env = dict(os.environ, DENSITO_ROOT=str(ROOT))
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return p.returncode, p.stdout + p.stderr


def read_rows(path: Path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


tmp = Path(tempfile.mkdtemp(prefix="densito_seg_test_"))
manifest = json.loads((PHANTOMS / "MANIFEST.json").read_text(encoding="utf-8"))
by_image_uid = {f["image_uid"]: f for f in manifest["files"] if f.get("image_uid")}

# 1-2. прогон без и с --seg-dir
rc0, log0 = run_inference(tmp / "base" / "results.csv")
check(rc0 == 0, "инференс без --seg-dir завершился кодом 0")
rc1, log1 = run_inference(tmp / "run1" / "results.csv", tmp / "run1" / "seg")
check(rc1 == 0, "инференс с --seg-dir завершился кодом 0")
check("Segmentation export enabled" in log1, "лог сообщает о включённом экспорте сегментации")
check("segmentation_export failed" not in log1, "экспорт сегментации не падал ни на одном снимке")

rows0 = read_rows(tmp / "base" / "results.csv")
rows1 = read_rows(tmp / "run1" / "results.csv")
cols = list(rows1[0].keys()) if rows1 else []
check(cols == ["path_to_study", "study_uid", "image_uid", "anatomical_region", "quality_class",
               "violation_type", "quality_prob", "processing_status", "time_of_processing"],
      "results.csv: ровно 9 официальных колонок")
strip = lambda rs: [{k: v for k, v in r.items() if k != "time_of_processing"} for r in rs]  # noqa: E731
check(strip(rows0) == strip(rows1), "8 колонок results.csv совпадают с прогоном без --seg-dir")

debug1 = read_rows(tmp / "run1" / "results_debug.csv")


def with_debug(rows, debug):
    """Строка официального CSV + поля технического CSV (по хвосту пути файла)."""
    out = []
    for r in rows:
        d = next((d for d in debug if str(d.get("file", "")).replace("\\", "/").endswith("/" + r["path_to_study"])), {})
        out.append({**d, **r})
    return out


rows1 = with_debug(rows1, debug1)
succ = [r for r in rows1 if r.get("processing_status") == "Success"]
fail = [r for r in rows1 if r.get("processing_status") == "Failure"]
check(len(succ) == 12 and len(fail) == 3, f"фантомы: 12 Success и 3 Failure (получено {len(succ)}/{len(fail)})")
seg_files = sorted((tmp / "run1" / "seg").iterdir())
check(len([f for f in seg_files if f.name.endswith("_seg.dcm")]) == len(succ), "SEG создан для каждой Success-строки")
check(len(seg_files) == 3 * len(succ), f"в каталоге ровно 3 файла на Success-строку ({len(seg_files)})")
for r in succ:
    for k in ("bonus_seg_dcm", "bonus_seg_png", "bonus_seg_json"):
        if not (r.get(k) and Path(r[k]).is_file()):
            check(False, f"{r['path_to_study']}: нет файла {k}")
            break
for r in fail:
    check(not any(r.get(k) for k in ("bonus_seg_dcm", "bonus_seg_png", "bonus_seg_json")),
          f"Failure {r['path_to_study']}: файлов сегментации нет")

# 3. валидатор
reports = [validate_seg.validate_file(f, require_nonempty=True) for f in seg_files if f.name.endswith("_seg.dcm")]
check(all(r.passed for r in reports), "tools/validate_seg.py: PASS для всех SEG")
for r in reports:
    if not r.passed:
        print("     ", r.name, r.errors)
rc_cli = validate_seg.main([str(tmp / "run1" / "seg"), "--require-nonempty", "--log", str(tmp / "validate_seg.log")])
check(rc_cli == 0, "tools/validate_seg.py (CLI) -> код возврата 0")

# 4. маски непустые, число сегментов, размер, ссылка
for r in succ:
    ds, frames = se.read_seg_masks(r["bonus_seg_dcm"])
    fam = se.region_family(r.get("internal_region") or ("spine" if r["anatomical_region"].startswith("Поясничн") else "hip"))
    n_expected = 2 if fam == "spine" else 3
    m = by_image_uid.get(r["image_uid"], {})
    ok = (int(ds.NumberOfFrames) == n_expected and frames[0].sum() > 0
          and int(ds.Rows) == int(m.get("rows", ds.Rows)) and int(ds.Columns) == int(m.get("cols", ds.Columns))
          and ds.ReferencedSeriesSequence[0].ReferencedInstanceSequence[0].ReferencedSOPInstanceUID == r["image_uid"])
    check(ok, f"{r['path_to_study']}: {n_expected} сегм., кость {int(frames[0].sum())} px, "
              f"{ds.Rows}x{ds.Columns}, ссылка на image_uid")
    if fam == "hip":
        check(frames[1].sum() > 0 and frames[2].sum() > 0, f"{r['path_to_study']}: поле сканирования и область интереса непустые")

# 5. детерминизм
rc2, _ = run_inference(tmp / "run2" / "results.csv", tmp / "run2" / "seg")
check(rc2 == 0, "повторный прогон завершился кодом 0")
same = True
for f in seg_files:
    g = tmp / "run2" / "seg" / f.name
    if not g.is_file() or sha(f) != sha(g):
        same = False
        print("      отличается:", f.name)
check(same, "детерминизм: повторный прогон даёт байт-в-байт те же SEG, PNG и JSON")

# 6. согласованность SEG / PNG / JSON с build_masks
for r in succ[:6]:
    src = PHANTOMS / r["path_to_study"]
    ds_src = pydicom.dcmread(str(src))
    img = normalize_pixels(ds_src)
    region = r.get("internal_region") or ("spine" if img.shape[1] >= 290 else "right_hip")
    masks = se.build_masks(img, region)
    ds, frames = se.read_seg_masks(r["bonus_seg_dcm"])
    check(all(np.array_equal(frames[i], masks[k]) for i, k in enumerate(masks)),
          f"{r['path_to_study']}: кадры SEG == build_masks()")
    rgba = cv2.imread(r["bonus_seg_png"], cv2.IMREAD_UNCHANGED)
    union = np.zeros(img.shape, bool)
    for k, m in masks.items():
        union |= m
    alpha = rgba[..., 3] > 0
    # контур области интереса рисуется линией, поэтому проверяем вложенность: непрозрачные пиксели внутри объединения масок
    check(rgba.shape[:2] == img.shape and bool(alpha[~union].sum() == 0) and bool(alpha[masks["bone"]].all()),
          f"{r['path_to_study']}: PNG {rgba.shape[1]}x{rgba.shape[0]} RGBA, непрозрачен только на масках, кость закрашена")
    js = json.loads(Path(r["bonus_seg_json"]).read_text(encoding="utf-8"))
    h, w = img.shape
    inside = all(0 <= x < w and 0 <= y < h for s in js["segments"] for p in s["polygons"] for x, y in p["points_px"])
    mm_ok = all(abs(pm[0] - px[0] * 0.6) < 1e-6 and abs(pm[1] - px[1] * 1.05) < 1e-6
                for s in js["segments"] for p in s["polygons"] for px, pm in zip(p["points_px"], p["points_mm"]))
    check(inside and mm_ok and js["rows"] == h and js["cols"] == w and js["image_uid"] == r["image_uid"]
          and len(js["legend"]) == len(js["segments"]) == int(ds.NumberOfFrames),
          f"{r['path_to_study']}: JSON — полигоны в кадре, мм = px x (0.6, 1.05), легенда по сегментам")

# 7. запрещённые формулировки
texts = [(ROOT / "src" / "segmentation_export.py").read_text(encoding="utf-8"),
         (ROOT / "tools" / "validate_seg.py").read_text(encoding="utf-8")]
texts += [Path(r["bonus_seg_json"]).read_text(encoding="utf-8") for r in succ]
for r in succ:
    ds = pydicom.dcmread(r["bonus_seg_dcm"])
    texts.append(" ".join(str(getattr(s, n, "")) for s in ds.SegmentSequence for n in ("SegmentLabel", "SegmentDescription")))
    texts.append(str(ds.SeriesDescription) + " " + str(ds.ContentDescription))
check(not any(b.lower() in t.lower() for b in BANNED for t in texts), "запрещённых формулировок нет в модуле, валидаторе, JSON и SEG")
check(not any("!" in t for t in texts[2:]), "восклицательных знаков в JSON и текстовых полях SEG нет")

# 8. API (если установлен FastAPI)
try:
    from fastapi.testclient import TestClient  # noqa: E402
    api_out = tmp / "api"
    os.environ["DENSITO_OUTPUT_DIR"] = str(api_out)
    os.environ["DENSITO_NO_EMBEDDINGS"] = "1"
    import api_server  # noqa: E402
    client = TestClient(api_server.app)
    files = [("files", (f"study_01_{p.name}", p.read_bytes(), "application/dicom"))
             for p in sorted((PHANTOMS / "study_01").glob("*.dcm"))[:2]]
    files.append(("files", ("no_pixel_data.dcm", (PHANTOMS / "broken" / "no_pixel_data.dcm").read_bytes(), "application/dicom")))
    resp = client.post("/api/analyze", files=files)
    check(resp.status_code == 200, f"API /api/analyze -> {resp.status_code}")
    body = resp.json()
    for row in body.get("rows", []):
        has = all(k in row for k in ("seg_download", "seg_png", "seg_json_download"))
        if row.get("processing_status") == "Success" and row.get("region_supported", True):
            check(has, f"API {row['path_to_study']}: seg_download / seg_png / seg_json_download присутствуют")
            for k in ("seg_download", "seg_png", "seg_json_download"):
                rr = client.get(row[k])
                check(rr.status_code == 200 and len(rr.content) > 0, f"API {row['path_to_study']}: {k} скачивается (200)")
            check("_seg." in row["seg_png"] and row["seg_png"].startswith(f"/api/results/{body['job_id']}/row"),
                  f"API {row['path_to_study']}: ссылки ведут в папку запроса и именуются по номеру строки")
        else:
            check(not any(k.startswith("seg_") for k in row), f"API {row['path_to_study']}: ключей seg_* нет (Failure или область не поддерживается)")
    schema_path = ROOT / "schema" / "api_analyze_response.schema.json"
    props = json.loads(schema_path.read_text(encoding="utf-8"))["$defs"]["row_with_extras"]["properties"]
    check(all(k in props for k in ("seg_download", "seg_png", "seg_json_download")), "схема ответа API описывает seg_* как необязательные поля")
except (ImportError, RuntimeError) as e:   # в образе нет httpx: starlette.testclient поднимает RuntimeError
    print(f"INFO FastAPI/httpx недоступны ({str(e).splitlines()[0]}) — проверка API пропущена")

# web/index.html: слой «Сегментация» без внешних зависимостей
html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
check("Сегментация" in html and "seg-layer" in html and "segSrc" in html, "web/index.html: слой «Сегментация» присутствует")
check("seg_png" in html and "seg_download" in html, "web/index.html: использует seg_png и seg_download из ответа API")

print()
if fails:
    print(f"ИТОГ: {len(fails)} проблем(ы)")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ИТОГ: все проверки пройдены")
