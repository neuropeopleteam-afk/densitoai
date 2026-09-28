#!/usr/bin/env python3
"""Имена файлов внутри zip (2.4.1, Б2): path_to_study без «?» для архивов с Windows.

Четыре типа синтетических архивов с фантомным DICOM внутри:
  1. флаг UTF-8 (бит 11) выставлен, имя кириллицей;
  2. имя в cp866, флаг не выставлен, extra field нет (старый архиватор Windows);
  3. имя в cp866 и extra field 0x7075 (Info-ZIP Unicode Path) с тем же именем в UTF-8 — так пишет
     проводник/7-Zip; на этом типе версия 2.4.0 отдавала «???»;
  4. имя латиницей.
Для каждого проверяется _fix_zip_name и сквозной прогон DensitoInference.run: path_to_study совпадает с
именем внутри архива и не содержит «?». Отдельно — образец организаторов «Для_теста.zip» (cp866 + 0x7075):
3 строки, пути кириллицей без «?» (если образца нет рядом, эта часть пропускается с сообщением).

Запуск:  python tests/test_zip_names.py     (код 0 — пройдено)
"""
import csv
import os
import struct
import sys
import tempfile
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import inference  # noqa: E402

PHANTOM = ROOT / "tests" / "phantoms" / "study_01" / "CR000000.dcm"
SAMPLE_CANDIDATES = [
    Path(os.environ["DENSITO_SAMPLE_ZIP"]) if os.environ.get("DENSITO_SAMPLE_ZIP") else None,
    ROOT / "tests" / "sample_test_zip" / "Для_теста.zip",
    Path("/home/user/workspace/projects/neuro-U1aXIfGdQRGUzlgaRDlKVA/files/densito/test_data/Для_теста.zip"),
]
FAILED = []


def check(name, cond, detail=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def make_zip(path: Path, name: str, data: bytes, kind: str) -> None:
    """Архив с одним файлом `name`. kind: utf8 | cp866 | cp866_7075 | latin.

    zipfile сам ставит флаг UTF-8 для любого не-ASCII имени, поэтому для cp866 пишем архив с
    ASCII-заглушкой той же длины в байтах и затем подменяем байты имени в локальном и центральном
    заголовках (длины полей не меняются, CRC данных не затрагивается)."""
    if kind in ("utf8", "latin"):
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(name, data)
        return
    raw = name.encode("cp866")
    placeholder = b"Z" * len(raw)
    zi = zipfile.ZipInfo(placeholder.decode("ascii"), date_time=(2026, 9, 1, 12, 0, 0))
    zi.compress_type = zipfile.ZIP_DEFLATED
    if kind == "cp866_7075":
        uni = name.encode("utf-8")
        body = struct.pack("<BL", 1, zlib.crc32(raw) & 0xFFFFFFFF) + uni
        zi.extra = struct.pack("<HH", 0x7075, len(body)) + body
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(zi, data)
    blob = path.read_bytes()
    assert blob.count(placeholder) == 2, "заглушка должна встретиться ровно в двух заголовках"
    path.write_bytes(blob.replace(placeholder, raw))


def main() -> int:
    data = PHANTOM.read_bytes()
    cases = [
        ("utf8", "Исследование Иванова/Снимок_ПОП.dcm"),
        ("cp866", "Исследование Петрова/Снимок_ППОБ.dcm"),
        ("cp866_7075", "Для теста/CR000000_ПОП.dcm"),
        ("latin", "study_A/IMG_0001.dcm"),
    ]
    work = Path(tempfile.mkdtemp(prefix="densito_zipnames_"))
    print("1. _fix_zip_name на четырёх типах архивов")
    zips = []
    for kind, name in cases:
        zp = work / f"{kind}.zip"
        make_zip(zp, name, data, kind)
        with zipfile.ZipFile(zp) as zf:
            zi = zf.infolist()[0]
            utf8_flag = bool(zi.flag_bits & 0x800)
            if kind == "utf8":
                check("utf8: флаг UTF-8 выставлен", utf8_flag)
            elif kind != "latin":
                check(f"{kind}: флаг UTF-8 не выставлен", not utf8_flag)
            if kind == "cp866_7075":
                check("cp866_7075: zipfile уже подставил имя из 0x7075",
                      zi.filename == name and zi.orig_filename != name, f"{zi.filename!r} / {zi.orig_filename!r}")
            got = inference._fix_zip_name(zi)
        check(f"{kind}: имя восстановлено", got == name, f"{got!r} != {name!r}")
        zips.append((kind, name, zp))

    print("2. сквозной прогон: path_to_study = имя внутри архива, без «?»")
    eng = inference.DensitoInference(use_embeddings=False)
    for kind, name, zp in zips:
        out = work / f"out_{kind}" / "results.csv"
        rows = eng.run(zp, out)
        with open(out, encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
        paths = [r["path_to_study"] for r in csv_rows]
        check(f"{kind}: одна строка", len(rows) == 1 and len(csv_rows) == 1, str(paths))
        check(f"{kind}: path_to_study без «?»", paths and "?" not in paths[0], str(paths))
        check(f"{kind}: path_to_study совпадает с именем в архиве", paths == [name], f"{paths} != {[name]}")

    # zip в папке (вложенный архив) — тот же путь через discover_files
    nested = work / "nested_in"
    nested.mkdir()
    (nested / "cp866_7075.zip").write_bytes((work / "cp866_7075.zip").read_bytes())
    out = work / "out_nested" / "results.csv"
    eng.run(nested, out)
    with open(out, encoding="utf-8") as f:
        p = [r["path_to_study"] for r in csv.DictReader(f)]
    check("вложенный zip cp866+0x7075: путь «архив/имя» без «?»", p == ["cp866_7075.zip/Для теста/CR000000_ПОП.dcm"], str(p))

    print("3. образец организаторов «Для_теста.zip»")
    sample = next((c for c in SAMPLE_CANDIDATES if c and c.is_file()), None)
    if sample is None:
        print("  SKIP образец организаторов не найден (задайте DENSITO_SAMPLE_ZIP)")
    else:
        out = work / "out_sample" / "results.csv"
        eng.run(sample, out)
        with open(out, encoding="utf-8") as f:
            p = sorted(r["path_to_study"] for r in csv.DictReader(f))
        check("образец: 3 строки", len(p) == 3, str(p))
        check("образец: пути без «?»", all("?" not in x for x in p), str(p))
        check("образец: пути кириллицей",
              p == ["Для теста/CR000000_ПОП.dcm", "Для теста/CR000000_ППОБ.dcm", "Для теста/CR000001_ЛПОБ.dcm"], str(p))

    print("\nИТОГ:", "OK" if not FAILED else f"FAIL ({len(FAILED)}): " + "; ".join(FAILED))
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
