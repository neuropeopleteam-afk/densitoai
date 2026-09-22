#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Контракт приёма входа (ТЗ п. 2.5: одна строка на каждый входной файл).

Проверяет случаи, которые не покрываются фантомами в `tools/verify.sh` (там все имена
уникальны) и не покрывались `tests/test_api_isolation.py`:

  1. два файла с одинаковым базовым именем в одном запросе;
  2. те же два файла, но клиент прислал относительные пути (загрузка папки из браузера);
  3. zip с двумя разными элементами, у которых совпадает полное имя внутри архива;
  4. zip с одинаковым базовым именем в разных подкаталогах;
  5. zip внутри zip;
  6. попытка выйти за каталог именем файла загрузки.

Запуск: python tests/test_input_contract.py
"""
import io
import os
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(tempfile.mkdtemp(prefix="densito_input_contract_"))
os.environ["DENSITO_OUTPUT_DIR"] = str(OUT)
os.environ["DENSITO_MAX_FILES"] = "40"
os.environ["DENSITO_MAX_UPLOAD_MB"] = "200"
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402
import api_server  # noqa: E402

client = TestClient(api_server.app)
fails = []


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def phantoms(n):
    """n различных исправных синтетических DICOM из tests/phantoms (разные байты)."""
    found = []
    for p in sorted((ROOT / "tests" / "phantoms").rglob("*.dcm")):
        if p.is_file() and "broken" not in p.parts:
            found.append(p)
    assert len(found) >= n, f"нужно минимум {n} исправных фантомов, найдено {len(found)}"
    return found[:n]


def analyze(files):
    r = client.post("/api/analyze", files=files)
    return r


def rows_of(r):
    if r.status_code != 200:
        return None
    return r.json().get("rows") or []


P = phantoms(4)
A, B, C, D = P[0], P[1], P[2], P[3]
assert A.read_bytes() != B.read_bytes(), "фантомы должны отличаться"

# --- 1. одинаковый basename, без путей ------------------------------------- #
r = analyze([
    ("files", ("IM1.dcm", A.read_bytes(), "application/dicom")),
    ("files", ("IM1.dcm", B.read_bytes(), "application/dicom")),
])
rows = rows_of(r)
check(rows is not None and len(rows) == 2,
      f"два файла с одинаковым именем дают две строки (получено {len(rows) if rows is not None else r.status_code})")

# --- 2. одинаковый basename, клиент прислал относительные пути -------------- #
r = analyze([
    ("files", ("study_a/IM1.dcm", A.read_bytes(), "application/dicom")),
    ("files", ("study_b/IM1.dcm", B.read_bytes(), "application/dicom")),
])
rows = rows_of(r)
check(rows is not None and len(rows) == 2,
      f"одинаковое имя в разных клиентских папках даёт две строки (получено {len(rows) if rows is not None else r.status_code})")
if rows and len(rows) == 2:
    paths = {str(x.get("path_to_study") or "") for x in rows}
    check(len(paths) == 2, f"path_to_study различаются: {sorted(paths)}")

# --- 3. zip с дублирующимися именами элементов ------------------------------ #
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as zf:
    zf.writestr("IM1.dcm", A.read_bytes())
    zf.writestr("IM1.dcm", B.read_bytes())   # zip это допускает
z_dup = buf.getvalue()
r = analyze([("files", ("dup.zip", z_dup, "application/zip"))])
rows = rows_of(r)
check(rows is not None and len(rows) == 2,
      f"zip с двумя одноимёнными элементами даёт две строки (получено {len(rows) if rows is not None else r.status_code})")

# --- 4. zip, одинаковый basename в разных подкаталогах ---------------------- #
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as zf:
    zf.writestr("a/IM1.dcm", A.read_bytes())
    zf.writestr("b/IM1.dcm", B.read_bytes())
r = analyze([("files", ("sub.zip", buf.getvalue(), "application/zip"))])
rows = rows_of(r)
check(rows is not None and len(rows) == 2,
      f"zip с одинаковым именем в разных подкаталогах даёт две строки (получено {len(rows) if rows is not None else r.status_code})")

# --- 5. zip внутри zip ------------------------------------------------------ #
inner = io.BytesIO()
with zipfile.ZipFile(inner, "w") as zf:
    zf.writestr("deep/IM9.dcm", C.read_bytes())
outer = io.BytesIO()
with zipfile.ZipFile(outer, "w") as zf:
    zf.writestr("inner.zip", inner.getvalue())
    zf.writestr("IM8.dcm", D.read_bytes())
r = analyze([("files", ("nested.zip", outer.getvalue(), "application/zip"))])
rows = rows_of(r)
check(rows is not None and len(rows) == 2,
      f"вложенный zip обрабатывается, строк 2 (получено {len(rows) if rows is not None else r.status_code})")

# --- 6. имя загрузки с обходом каталога ------------------------------------- #
r = analyze([
    ("files", ("../../etc/evil.dcm", A.read_bytes(), "application/dicom")),
])
check(r.status_code in (200, 400, 413),
      f"имя с ../ не роняет сервер (код {r.status_code})")
if r.status_code == 200:
    rows = rows_of(r)
    check(len(rows) == 1, "файл с ../ в имени даёт ровно одну строку")
    check(not Path("/etc/evil.dcm").exists(), "файл не записан за пределы каталога запроса")

# --- 7. смешанная партия: строк == файлов ---------------------------------- #
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as zf:
    zf.writestr("s1/IM1.dcm", A.read_bytes())
    zf.writestr("s1/IM2.dcm", B.read_bytes())
r = analyze([
    ("files", ("pack.zip", buf.getvalue(), "application/zip")),
    ("files", ("IM1.dcm", C.read_bytes(), "application/dicom")),
    ("files", ("IM1.dcm", D.read_bytes(), "application/dicom")),
])
rows = rows_of(r)
check(rows is not None and len(rows) == 4,
      f"смешанная партия: 2 в архиве + 2 одноимённых = 4 строки (получено {len(rows) if rows is not None else r.status_code})")

print()
if fails:
    print(f"ПРОВАЛЕНО: {len(fails)}")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL INPUT CONTRACT CHECKS PASSED")
