#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_transfer_syntax.py — матрица transfer syntax / вариантов кодирования DICOM.

Из файлов tests/sample_test_zip/ (3 снимка организаторов; в публичном репозитории их нет — тогда фантомы; вне репозитория-релиза они
заменяются любыми DICOM того же аппарата) создаются варианты кодирования, прогоняется
src/inference.py и сравнивается с оригиналом: processing_status=Success, anatomical_region и
quality_class совпадают. Дополнительно фиксируются violation_type и |Δ quality_prob|.

    python tests/test_transfer_syntax.py                # матрица -> docs/TRANSFER_SYNTAX_MATRIX.md
    python -m pytest tests/test_transfer_syntax.py -q   # то же как тест (падает при расхождении)

Варианты, которые невозможно записать текущим набором кодеков (нет pylibjpeg-*, pyjpegls, gdcm),
попадают в таблицу со статусом «не проверено», а не «ошибка».
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileMetaDataset
from pydicom.uid import (ExplicitVRBigEndian, ExplicitVRLittleEndian, ImplicitVRLittleEndian,
                         JPEG2000Lossless, JPEGLosslessSV1, JPEGLSLossless, RLELossless)

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
SRC_DIR = Path(os.environ.get("DENSITO_TS_SOURCE", ROOT / "tests" / "sample_test_zip"))
if not SRC_DIR.exists():   # в релиз-архиве образца организаторов нет -> синтетические фантомы
    SRC_DIR = ROOT / "tests" / "phantoms" / "study_01"
MATRIX_MD = ROOT / "docs" / "TRANSFER_SYNTAX_MATRIX.md"
os.environ.setdefault("TORCH_HOME", str(ROOT / "models" / "torch_home"))
os.environ.setdefault("OMP_NUM_THREADS", "1")


# --------------------------------------------------------------------------- #
def _fresh(ds: pydicom.Dataset) -> pydicom.Dataset:
    """Копия датасета с распакованными пикселями, Explicit VR LE, готовая к модификации."""
    d = pydicom.dcmread(str(ds.filename), force=True)
    arr = d.pixel_array  # распаковать
    d.PixelData = arr.tobytes()
    d.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    d.is_little_endian, d.is_implicit_VR = True, False
    return d


def v_implicit_le(d):
    d.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    d.is_implicit_VR = True
    return d


def v_explicit_le(d):
    return d


def v_explicit_be(d):
    """Big Endian: для 8-битных пикселей с VR=OW байты меняются местами попарно (PS3.5 8.1.1)."""
    d.file_meta.TransferSyntaxUID = ExplicitVRBigEndian
    if int(d.BitsAllocated) == 8:
        raw = np.frombuffer(d.PixelData, dtype="u1")
        if len(raw) % 2:
            raw = np.append(raw, 0)
        d.PixelData = raw.reshape(-1, 2)[:, ::-1].tobytes()
        d["PixelData"].VR = "OW"
    else:
        d.PixelData = d.pixel_array.astype(">u2").tobytes()
    d._big_endian = True
    return d


def _compress(uid):
    def f(d):
        d.compress(uid, generate_instance_uid=False)   # SOPInstanceUID оставляем как в оригинале
        return d
    return f


def v_16bit(d):
    arr = d.pixel_array.astype(np.uint16) * 16  # 8 -> 12 эффективных бит
    d.PixelData = arr.tobytes()
    d.BitsAllocated, d.BitsStored, d.HighBit = 16, 12, 11
    return d


def v_16bit_rescale(d):
    arr = d.pixel_array.astype(np.uint16) * 4
    d.PixelData = arr.tobytes()
    d.BitsAllocated, d.BitsStored, d.HighBit = 16, 16, 15
    d.RescaleSlope, d.RescaleIntercept = "0.5", "-100"
    return d


def v_mono1(d):
    arr = d.pixel_array
    d.PixelData = (int(arr.max()) - arr.astype(np.int32)).astype(arr.dtype).tobytes()
    d.PhotometricInterpretation = "MONOCHROME1"
    return d


def v_with_pixel_spacing(d):
    d.PixelSpacing = ["1.05", "0.6"]
    return d


def v_without_pixel_spacing(d):
    for kw in ("PixelSpacing", "ImagerPixelSpacing"):
        if kw in d:
            del d[kw]
    return d


def v_raw_no_meta(d):
    """Без преамбулы и file meta (сырой поток Implicit VR LE, как пишут некоторые архивы)."""
    d.preamble = None
    d.file_meta = FileMetaDataset()
    d.is_implicit_VR, d.is_little_endian = True, True
    d._raw_no_meta = True
    return d


def v_no_uids(d):
    del d.StudyInstanceUID
    del d.SOPInstanceUID
    return d


VARIANTS = [
    ("implicit_le", "Implicit VR Little Endian (1.2.840.10008.1.2) — как в оригинале", v_implicit_le),
    ("explicit_le", "Explicit VR Little Endian (1.2.840.10008.1.2.1)", v_explicit_le),
    ("explicit_be", "Explicit VR Big Endian (1.2.840.10008.1.2.2, retired)", v_explicit_be),
    ("rle", "RLE Lossless (1.2.840.10008.1.2.5)", _compress(RLELossless)),
    ("jpeg2000_lossless", "JPEG 2000 Lossless (1.2.840.10008.1.2.4.90)", _compress(JPEG2000Lossless)),
    ("jpegls_lossless", "JPEG-LS Lossless (1.2.840.10008.1.2.4.80)", _compress(JPEGLSLossless)),
    ("jpeg_lossless_sv1", "JPEG Lossless SV1 (1.2.840.10008.1.2.4.70)", _compress(JPEGLosslessSV1)),
    ("bits16", "16 бит (BitsAllocated 16 / BitsStored 12), Explicit VR LE", v_16bit),
    ("bits16_rescale", "16 бит + RescaleSlope 0.5 / RescaleIntercept −100", v_16bit_rescale),
    ("monochrome1", "MONOCHROME1 (инвертированные пиксели)", v_mono1),
    ("pixel_spacing", "С тегом PixelSpacing 1.05/0.6 (в оригинале тега нет)", v_with_pixel_spacing),
    ("no_pixel_spacing", "Без PixelSpacing и ImagerPixelSpacing", v_without_pixel_spacing),
    ("raw_no_meta", "Без преамбулы и file meta header (raw Implicit VR LE)", v_raw_no_meta),
    ("no_uids", "Без StudyInstanceUID и SOPInstanceUID (ожидается hash-UID, Success)", v_no_uids),
]


def _save(d, path: Path) -> None:
    if getattr(d, "_raw_no_meta", False):
        d.save_as(str(path), enforce_file_format=False, little_endian=True, implicit_vr=True)
    elif getattr(d, "_big_endian", False):
        from pydicom.filewriter import dcmwrite  # noqa: WPS433
        dcmwrite(str(path), d, implicit_vr=False, little_endian=False, force_encoding=True, enforce_file_format=False)
    else:
        d.save_as(str(path), enforce_file_format=True)


def build_variants(sources: list[Path], workdir: Path) -> tuple[list[dict], dict]:
    """Возвращает (список созданных файлов, статусы записи по вариантам)."""
    files, write_status = [], {}
    for src in sources:
        orig = pydicom.dcmread(str(src), force=True)
        d0 = workdir / "orig" / src.name
        d0.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, d0)
        files.append({"variant": "orig", "src": src.name, "path": d0})
        for key, _desc, fn in VARIANTS:
            out = workdir / key / src.name
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                d = fn(_fresh(orig))
                _save(d, out)
            except Exception as e:  # noqa: BLE001
                write_status[key] = f"не проверено: {type(e).__name__}: {str(e)[:110]}"
                if out.exists():
                    out.unlink()
                continue
            # контроль записи: файл читается pydicom (само чтение пикселей проверяет инференс)
            try:
                chk = pydicom.dcmread(str(out), force=True)
                _ = chk.pixel_array
                write_status.setdefault(key, "записан")
            except Exception as e:  # noqa: BLE001
                write_status.setdefault(key, f"записан; pydicom.pixel_array: {type(e).__name__}")
            files.append({"variant": key, "src": src.name, "path": out})
    return files, write_status


def run_inference(indir: Path, out_csv: Path) -> float:
    t0 = time.perf_counter()
    cmd = [sys.executable, str(ROOT / "src" / "inference.py"), "--input", str(indir), "--output", str(out_csv), "--debug-csv"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if r.returncode != 0 or not out_csv.exists():
        raise RuntimeError(f"inference failed rc={r.returncode}\n{r.stderr[-2000:]}")
    return time.perf_counter() - t0


def compare(files: list[dict], rows: list[dict], debug_rows: list[dict]) -> list[dict]:
    by_path = {r["path_to_study"].replace("\\", "/"): r for r in rows}
    # debug-CSV: колонка file — абсолютный путь; ключ = «<variant>/<имя файла>»
    dbg_by_path = {"/".join(Path(r["file"]).parts[-2:]): r for r in debug_rows} if debug_rows else {}
    orig = {f["src"]: by_path[f"orig/{f['src']}"] for f in files if f["variant"] == "orig"}
    out = []
    for f in files:
        if f["variant"] == "orig":
            continue
        r = by_path.get(f"{f['variant']}/{f['src']}")
        o = orig[f["src"]]
        if r is None:
            out.append({**f, "ok": False, "status": "нет строки"})
            continue
        dp = abs(float(r["quality_prob"]) - float(o["quality_prob"]))
        ok = (r["processing_status"] == "Success" and r["anatomical_region"] == o["anatomical_region"]
              and r["quality_class"] == o["quality_class"])
        dbg = dbg_by_path.get(f"{f['variant']}/{f['src']}", {})
        dbg_o = dbg_by_path.get(f"orig/{f['src']}", {})
        out.append({"variant": f["variant"], "src": f["src"], "ok": ok, "status": r["processing_status"],
                    "region_same": r["anatomical_region"] == o["anatomical_region"],
                    "class_same": r["quality_class"] == o["quality_class"],
                    "violation_same": r["violation_type"] == o["violation_type"],
                    "uid_same": r["study_uid"] == o["study_uid"] and r["image_uid"] == o["image_uid"],
                    "dprob": dp, "internal_region_same": dbg.get("internal_region", "") == dbg_o.get("internal_region", "")
                    if dbg and dbg_o else None})
    return out


def write_matrix(results: list[dict], write_status: dict, sources: list[Path], t_inf: float, out_md: Path) -> None:
    env = f"pydicom {pydicom.__version__}, numpy {np.__version__}, Python {sys.version.split()[0]}"
    plugins = []
    for mod in ("pylibjpeg", "libjpeg", "openjpeg", "jpeg_ls", "gdcm"):
        try:
            m = __import__(mod)
            plugins.append(f"{mod} {getattr(m, '__version__', '?')}")
        except Exception:  # noqa: BLE001
            plugins.append(f"{mod} —")
    lines = ["# Матрица transfer syntax и вариантов кодирования DICOM", "",
             f"Сгенерировано `tests/test_transfer_syntax.py` {time.strftime('%Y-%m-%d %H:%M')}; окружение: {env}; "
             f"кодеки: {', '.join(plugins)}.", "",
             f"Источник: {len(sources)} файла из `{SRC_DIR.relative_to(ROOT) if SRC_DIR.is_relative_to(ROOT) else SRC_DIR}` ({', '.join(s.name for s in sources)}). "
             "Критерий «OK»: processing_status = Success, anatomical_region и quality_class совпадают с оригиналом "
             "(Implicit VR LE, 8 бит, MONOCHROME2, без PixelSpacing). Δprob — максимальное по файлам |Δ quality_prob|.", "",
             f"Инференс всех вариантов одним прогоном: {t_inf:.1f} с.", "",
             "| Вариант | Описание | Записан | Success | Регион | Класс | Тип нарушения | UID | max Δprob | Итог |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    n_ok = n_total = 0
    for key, desc, _ in VARIANTS:
        rs = [r for r in results if r["variant"] == key]
        ws = write_status.get(key, "не создан")
        if not rs:
            lines.append(f"| `{key}` | {desc} | {ws} | — | — | — | — | — | — | не проверено |")
            continue
        n_total += 1
        succ = sum(1 for r in rs if r["status"] == "Success")
        reg = sum(1 for r in rs if r["region_same"])
        cls = sum(1 for r in rs if r["class_same"])
        viol = sum(1 for r in rs if r["violation_same"])
        uid = sum(1 for r in rs if r["uid_same"])
        dp = max(r["dprob"] for r in rs)
        ok = all(r["ok"] for r in rs)
        n_ok += ok
        n = len(rs)
        lines.append(f"| `{key}` | {desc} | {ws} | {succ}/{n} | {reg}/{n} | {cls}/{n} | {viol}/{n} | {uid}/{n} | {dp:.1e} | {'OK' if ok else 'РАСХОЖДЕНИЕ'} |")
    lines += ["", f"Итог: {n_ok} из {n_total} проверенных вариантов без расхождений; "
              f"{len(VARIANTS) - n_total} вариантов не удалось записать текущими кодеками (статус «не проверено»).", "",
              "Примечания.", "",
              "- Для `no_uids` UID ожидаемо не совпадают: инференс подставляет `hash-<sha>` от пути; проверяется только Success/регион/класс.",
              "- Варианты с потерями (JPEG Baseline) не включены: изменение пикселей — не вопрос совместимости формата.",
              "- Кодеки для сжатых синтаксисов (pylibjpeg, pylibjpeg-libjpeg, pylibjpeg-openjpeg) нужны только для чтения сжатых DICOM; "
              "экспорт GE Lunar Prodigy — несжатый Implicit VR LE. В `requirements.txt` они вынесены отдельным блоком."]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_matrix(write_md: bool = True) -> tuple[list[dict], dict]:
    sources = sorted(p for p in SRC_DIR.rglob("*.dcm"))
    assert sources, f"нет DICOM в {SRC_DIR}"
    sources = sources[:3]
    work = Path(tempfile.mkdtemp(prefix="densito_ts_"))
    try:
        files, write_status = build_variants(sources, work / "in")
        out_csv = work / "out" / "results.csv"
        out_csv.parent.mkdir(parents=True)
        t_inf = run_inference(work / "in", out_csv)
        with open(out_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        dbg_path = out_csv.with_name("results_debug.csv")
        debug_rows = []
        if dbg_path.exists():
            with open(dbg_path, newline="", encoding="utf-8") as f:
                debug_rows = list(csv.DictReader(f))
        results = compare(files, rows, debug_rows)
        if write_md:
            write_matrix(results, write_status, sources, t_inf, MATRIX_MD)
            (MATRIX_MD.with_suffix(".json")).write_text(
                json.dumps({"results": results, "write_status": write_status}, ensure_ascii=False, indent=1, default=str),
                encoding="utf-8")
        return results, write_status
    finally:
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- #
def test_transfer_syntax_matrix():
    results, _ = run_matrix(write_md=True)
    bad = [f"{r['variant']}/{r['src']}" for r in results if not r["ok"]]
    assert not bad, f"расхождения с оригиналом: {bad}"


if __name__ == "__main__":
    res, ws = run_matrix(write_md=True)
    n_bad = sum(1 for r in res if not r["ok"])
    print(open(MATRIX_MD, encoding="utf-8").read())
    sys.exit(1 if n_bad else 0)
