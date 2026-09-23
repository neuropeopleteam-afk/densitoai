#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stress_set.py — стресс-набор устойчивости входа (проверка 18 в tools/verify.sh).

Из синтетических фантомов tests/phantoms/ (и только из них — кадры пациентов не используются)
во временном каталоге собирается пакет «норма + битые и нестандартные входы»: копия всех фантомов
с исходными относительными путями плюс подкаталог stress/ с случаями (обрезанный файл, нулевой файл,
не-DICOM с расширением .dcm, DICOM без PixelData / без Rows и Columns / без PixelSpacing,
MONOCHROME1, 16 бит со знаком и отрицательными значениями, 8 бит Explicit VR LE, RGB, кадр 8×8,
огромный кадр, Explicit VR Big Endian, Deflated Explicit VR LE, RLE Lossless, JPEG Baseline,
BitsStored 12 при BitsAllocated 16, чужая модальность CT, постоянный кадр, zip с битым файлом внутри,
не-zip с расширением .zip, вложенность каталогов глубиной 5, кириллица и пробелы в именах).

Пакет обрабатывается одним прогоном src/inference.py, затем проверяется:
  1. процесс завершился кодом 0, CSV есть, заголовок — 9 колонок в порядке ТЗ, в журнале нет Traceback;
  2. на каждый входной файл (в том числе внутри zip) — ровно одна строка, лишних строк нет;
  3. у каждого случая processing_status равен ожидаемому (tests/stress/expected_stress.csv);
  4. строка Failure оформлена как в src/inference.py: quality_class 0, violation_type пустой,
     quality_prob = output.fallback_quality_prob (0.5), в debug CSV записана причина (error);
  5. для случаев с правилом same_class (кодирование без потерь) регион, класс и тип нарушения
     совпадают со строкой исходного фантома в эталонном прогоне, |Δ quality_prob| ≤ --tol;
  6. строки исходных фантомов в смешанном пакете побитово равны строкам одиночного прогона
     (все колонки, кроме time_of_processing);
  7. время прогона ≤ --budget секунд (по умолчанию 300).

    python tools/stress_set.py --phantoms tests/phantoms --baseline outputs/verify/run1/results.csv \
        --out outputs/verify/stress_check.json --workdir outputs/verify/stress [--md docs/STRESS_SET.md]

Без --baseline эталонный прогон на фантомах выполняется здесь же. Ожидания — tests/stress/expected_stress.csv;
`--write-expected` перезаписывает его из встроенного списка случаев (только разработчику).
Код возврата: 0 — все проверки пройдены, 1 — есть расхождения (JSON пишется в любом случае).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pydicom
from pydicom.dataset import FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.uid import (DeflatedExplicitVRLittleEndian, ExplicitVRBigEndian, ExplicitVRLittleEndian,
                         JPEGBaseline8Bit, RLELossless)

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
COLUMNS = ["path_to_study", "study_uid", "image_uid", "anatomical_region", "quality_class",
           "violation_type", "quality_prob", "processing_status", "time_of_processing"]
TIME_COL = "time_of_processing"
PATH_COL = "path_to_study"
SAME_COLS = ("anatomical_region", "quality_class", "violation_type")
EXPECTED_CSV = ROOT / "tests" / "stress" / "expected_stress.csv"
EXPECTED_FIELDS = ["case", "file", "source", "expected_status", "rule", "description"]
DEFAULT_HUGE = "4000x3000"   # rows x cols; 2500x1500 — запасной размер для слабых машин

# Исходные фантомы (относительно tests/phantoms). Для правила same_class берутся и фантомы с нарушением,
# чтобы проверить сохранение класса 1 и строки нарушения, а не только нормы.
SRC_SPINE_OK = "study_01/CR000000.dcm"
SRC_RHIP_OK = "study_01/CR000001.dcm"
SRC_LHIP_OK = "study_01/CR000002.dcm"
SRC_SPINE_ART = "study_03/CR000000.dcm"     # «Присутствуют посторонние предметы»
SRC_SPINE_POS = "study_04/CR000000.dcm"     # «Некорректная укладка»
SRC_HIP_2 = "study_02/CR000001.dcm"
SRC_HIP_3 = "study_03/CR000002.dcm"


# --------------------------------------------------------------------------- #
# Вспомогательные функции
# --------------------------------------------------------------------------- #
def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def _norm(p: str) -> str:
    return p.replace("\\", "/").lstrip("./")


def row_bytes(r: dict, drop: tuple[str, ...] = (TIME_COL,)) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([r.get(c, "") for c in COLUMNS if c not in drop])
    return buf.getvalue().encode("utf-8")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fallback_prob() -> str:
    """output.fallback_quality_prob из config.yaml в том виде, в каком его печатает CSV (0.5)."""
    try:
        import yaml
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
        val = float((cfg.get("output") or {}).get("fallback_quality_prob", 0.5))
    except Exception:  # noqa: BLE001
        val = 0.5
    return repr(val) if val != int(val) else f"{val:.1f}"


def jpeg_decoder_available() -> tuple[bool, str]:
    """Есть ли у pydicom декодер JPEG Baseline (Pillow или pylibjpeg-libjpeg)."""
    try:
        from pydicom.pixels import get_decoder
        dec = get_decoder(JPEGBaseline8Bit)
        avail = [n for n in dec.available_plugins] if hasattr(dec, "available_plugins") else []
        return bool(dec.is_available), ",".join(avail)
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}"


def pil_available() -> bool:
    try:
        import PIL  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Построение случаев
# --------------------------------------------------------------------------- #
def _fresh(src: Path) -> pydicom.Dataset:
    """Копия фантома с распакованными пикселями, Explicit VR LE, готовая к изменению."""
    d = pydicom.dcmread(str(src), force=True)
    arr = d.pixel_array
    d.PixelData = arr.tobytes()
    d.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    return d


def _save(d: pydicom.Dataset, out: Path, big_endian: bool = False) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if big_endian:
        from pydicom.filewriter import dcmwrite
        dcmwrite(str(out), d, implicit_vr=False, little_endian=False, force_encoding=True,
                 enforce_file_format=False)
    else:
        d.save_as(str(out), enforce_file_format=True)


def c_truncated_half(src: Path, out: Path) -> None:
    data = src.read_bytes()
    out.write_bytes(data[: len(data) // 2])


def c_zero_bytes(src: Path, out: Path) -> None:
    out.write_bytes(b"")


def c_not_dicom(src: Path, out: Path) -> None:
    rng = np.random.default_rng(20260923)
    body = b"This is not a DICOM file. \xd0\xad\xd1\x82\xd0\xbe \xd0\xbd\xd0\xb5 DICOM.\n" * 20
    out.write_bytes(body + rng.integers(0, 256, 4096, dtype=np.uint8).tobytes())


def c_no_pixel_data(src: Path, out: Path) -> None:
    d = _fresh(src)
    del d.PixelData
    _save(d, out)


def c_no_rows_cols(src: Path, out: Path) -> None:
    d = _fresh(src)
    del d.Rows
    del d.Columns
    _save(d, out)


def c_no_pixel_spacing(src: Path, out: Path) -> None:
    d = _fresh(src)
    for kw in ("PixelSpacing", "ImagerPixelSpacing"):
        if kw in d:
            del d[kw]
    _save(d, out)


def c_monochrome1(src: Path, out: Path) -> None:
    d = _fresh(src)
    arr = d.pixel_array
    d.PixelData = (int(arr.max()) - arr.astype(np.int32)).astype(arr.dtype).tobytes()
    d.PhotometricInterpretation = "MONOCHROME1"
    _save(d, out)


def c_signed16_negative(src: Path, out: Path) -> None:
    """16 бит со знаком: значения 0..255 -> (v*16 - 2048), диапазон [-2048; 2032]."""
    d = _fresh(src)
    arr = d.pixel_array.astype(np.int32) * 16 - 2048
    d.PixelData = arr.astype(np.int16).tobytes()
    d.BitsAllocated, d.BitsStored, d.HighBit, d.PixelRepresentation = 16, 16, 15, 1
    _save(d, out)


def c_bits8_explicit(src: Path, out: Path) -> None:
    d = _fresh(src)     # 8 бит, Explicit VR LE (фантомы — Implicit VR LE)
    _save(d, out)


def c_rgb(src: Path, out: Path) -> None:
    d = _fresh(src)
    arr = d.pixel_array
    d.PixelData = np.stack([arr, arr, arr], axis=-1).tobytes()
    d.SamplesPerPixel, d.PlanarConfiguration, d.PhotometricInterpretation = 3, 0, "RGB"
    _save(d, out)


def c_tiny_8x8(src: Path, out: Path) -> None:
    d = _fresh(src)
    arr = d.pixel_array[:8, :8].copy()
    arr[::2, ::2] = 200
    d.PixelData = arr.tobytes()
    d.Rows, d.Columns = 8, 8
    _save(d, out)


def make_huge(rows: int, cols: int) -> Callable[[Path, Path], None]:
    def f(src: Path, out: Path) -> None:
        d = _fresh(src)
        arr = d.pixel_array
        ry = np.linspace(0, arr.shape[0] - 1, rows).round().astype(int)
        rx = np.linspace(0, arr.shape[1] - 1, cols).round().astype(int)
        big = arr[ry][:, rx]        # увеличение повтором отсчётов, без интерполяции
        d.PixelData = np.ascontiguousarray(big).tobytes()
        d.Rows, d.Columns = rows, cols
        _save(d, out)
    return f


def c_explicit_be(src: Path, out: Path) -> None:
    """Explicit VR Big Endian: 8-битные пиксели с VR=OW пишутся с перестановкой байтов попарно (PS3.5 8.1.1)."""
    d = _fresh(src)
    d.file_meta.TransferSyntaxUID = ExplicitVRBigEndian
    raw = np.frombuffer(d.PixelData, dtype="u1")
    if len(raw) % 2:
        raw = np.append(raw, 0)
    d.PixelData = raw.reshape(-1, 2)[:, ::-1].tobytes()
    d["PixelData"].VR = "OW"
    _save(d, out, big_endian=True)


def c_deflated_le(src: Path, out: Path) -> None:
    d = _fresh(src)
    d.file_meta.TransferSyntaxUID = DeflatedExplicitVRLittleEndian
    _save(d, out)


def c_rle(src: Path, out: Path) -> None:
    d = _fresh(src)
    d.compress(RLELossless, generate_instance_uid=False)
    _save(d, out)


def c_jpeg_baseline(src: Path, out: Path) -> None:
    """JPEG Baseline (1.2.840.10008.1.2.4.50). Кодировщика JPEG у pydicom нет; поток JPEG даёт Pillow.
    Если Pillow нет — в кадр кладётся заведомо не-JPEG поток: ожидается Failure с причиной."""
    d = _fresh(src)
    arr = d.pixel_array
    if pil_available():
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=95)
        frame = buf.getvalue()
    else:
        frame = b"\xff\xd8NOT-A-JPEG" + arr.tobytes()[:1024]
    d.PixelData = encapsulate([frame])
    d["PixelData"].VR = "OB"
    d.file_meta.TransferSyntaxUID = JPEGBaseline8Bit
    d.LossyImageCompression = "01"
    _save(d, out)


def c_bits12_of_16(src: Path, out: Path) -> None:
    d = _fresh(src)
    arr = d.pixel_array.astype(np.uint16) * 16
    d.PixelData = arr.tobytes()
    d.BitsAllocated, d.BitsStored, d.HighBit = 16, 12, 11
    _save(d, out)


def c_modality_ct(src: Path, out: Path) -> None:
    d = _fresh(src)
    d.Modality = "CT"
    d.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    d.file_meta.MediaStorageSOPClassUID = d.SOPClassUID
    _save(d, out)


def c_constant(src: Path, out: Path) -> None:
    d = _fresh(src)
    arr = np.full(d.pixel_array.shape, 137, dtype=np.uint8)
    d.PixelData = arr.tobytes()
    _save(d, out)


def c_corrupt_zip(src: Path, out: Path) -> None:
    rng = np.random.default_rng(20260924)
    out.write_bytes(b"PK\x03\x04" + rng.integers(0, 256, 8192, dtype=np.uint8).tobytes())


def c_copy(src: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out)


# Случай: (id, файл внутри stress/<id>/, исходный фантом, ожидаемый статус, правило, описание).
# Правила: failure_row — строка Failure по правилам inference.py (class 0, violation пустой, prob 0.5, причина);
#          same_class — Success, регион/класс/нарушение равны исходному фантому, |Δprob| ≤ tol;
#          success — Success (содержимое кадра изменено, класс не сравнивается).
def case_table(huge: tuple[int, int], jpeg_ok: Optional[bool] = None) -> list[dict]:
    """jpeg_ok=None — определить по окружению; True — ожидание для образа (Pillow есть в requirements.txt)."""
    if jpeg_ok is None:
        jpeg_ok, _ = jpeg_decoder_available()
    hr, hc = huge
    rows = [
        dict(case="truncated_half", file="CR000000.dcm", source=SRC_SPINE_OK, expected_status="Failure",
             rule="failure_row", description="Файл обрезан до половины байтов (заголовок цел, пиксели неполные)",
             build=c_truncated_half),
        dict(case="zero_bytes", file="CR000000.dcm", source=SRC_SPINE_OK, expected_status="Failure",
             rule="failure_row", description="Файл нулевой длины с расширением .dcm", build=c_zero_bytes),
        dict(case="not_dicom_ext", file="CR000000.dcm", source=SRC_SPINE_OK, expected_status="Failure",
             rule="failure_row", description="Текст и случайные байты с расширением .dcm", build=c_not_dicom),
        dict(case="no_pixel_data", file="CR000000.dcm", source=SRC_SPINE_OK, expected_status="Failure",
             rule="failure_row", description="DICOM без PixelData (только заголовок)", build=c_no_pixel_data),
        dict(case="no_rows_cols", file="CR000000.dcm", source=SRC_RHIP_OK, expected_status="Failure",
             rule="failure_row", description="DICOM без Rows и Columns при наличии PixelData", build=c_no_rows_cols),
        dict(case="no_pixel_spacing", file="CR000000.dcm", source=SRC_SPINE_ART, expected_status="Success",
             rule="same_class", description="Без PixelSpacing и ImagerPixelSpacing (паспортная константа аппарата)",
             build=c_no_pixel_spacing),
        dict(case="monochrome1", file="CR000000.dcm", source=SRC_SPINE_POS, expected_status="Success",
             rule="same_class", description="PhotometricInterpretation MONOCHROME1 (инвертированные пиксели)",
             build=c_monochrome1),
        dict(case="signed16_negative", file="CR000000.dcm", source=SRC_SPINE_ART, expected_status="Success",
             rule="same_class", description="16 бит со знаком (PixelRepresentation 1), значения от −2048",
             build=c_signed16_negative),
        dict(case="bits8_explicit_le", file="CR000000.dcm", source=SRC_LHIP_OK, expected_status="Success",
             rule="same_class", description="8 бит, Explicit VR Little Endian", build=c_bits8_explicit),
        dict(case="rgb", file="CR000000.dcm", source=SRC_SPINE_ART, expected_status="Success",
             rule="same_class", description="RGB, SamplesPerPixel 3, три одинаковых канала", build=c_rgb),
        dict(case="tiny_8x8", file="CR000000.dcm", source=SRC_SPINE_OK, expected_status="Failure",
             rule="failure_row", description="Кадр 8×8 (меньше validation.min_rows/min_cols = 64)", build=c_tiny_8x8),
        dict(case="huge_frame", file="CR000000.dcm", source=SRC_SPINE_OK, expected_status="Success",
             rule="success", description=f"Кадр {hr}×{hc} (в пределах validation.max 4096), 8 бит",
             build=make_huge(hr, hc)),
        dict(case="explicit_be", file="CR000000.dcm", source=SRC_HIP_2, expected_status="Success",
             rule="same_class", description="Explicit VR Big Endian (retired)", build=c_explicit_be),
        dict(case="deflated_le", file="CR000000.dcm", source=SRC_SPINE_ART, expected_status="Success",
             rule="same_class", description="Deflated Explicit VR Little Endian", build=c_deflated_le),
        dict(case="rle_lossless", file="CR000000.dcm", source=SRC_SPINE_POS, expected_status="Success",
             rule="same_class", description="RLE Lossless (кодировщик pydicom)", build=c_rle),
        dict(case="jpeg_baseline", file="CR000000.dcm", source=SRC_SPINE_OK,
             expected_status="Success" if jpeg_ok else "Failure",
             rule="success" if jpeg_ok else "failure_row",
             description="JPEG Baseline: Success, если у pydicom есть декодер (Pillow); иначе Failure с причиной",
             build=c_jpeg_baseline),
        dict(case="bits12_of_16", file="CR000000.dcm", source=SRC_HIP_3, expected_status="Success",
             rule="same_class", description="BitsAllocated 16 / BitsStored 12 / HighBit 11", build=c_bits12_of_16),
        dict(case="modality_ct", file="CR000000.dcm", source=SRC_SPINE_ART, expected_status="Success",
             rule="same_class", description="Чужая модальность (Modality CT, SOP Class CT Image Storage)",
             build=c_modality_ct),
        dict(case="constant_frame", file="CR000000.dcm", source=SRC_SPINE_OK, expected_status="Failure",
             rule="failure_row", description="Все пиксели одинаковые (постоянный кадр)", build=c_constant),
        dict(case="zip_with_broken", file="Выгрузка (битый внутри).zip/norm/CR000000.dcm", source=SRC_RHIP_OK,
             expected_status="Success", rule="same_class",
             description="zip: исправный фантом рядом с обрезанным файлом", build=None),
        dict(case="zip_with_broken", file="Выгрузка (битый внутри).zip/broken/CR000001.dcm", source=SRC_SPINE_OK,
             expected_status="Failure", rule="failure_row",
             description="zip: обрезанный файл внутри архива", build=None),
        dict(case="corrupt_zip", file="архив.zip", source=SRC_SPINE_OK, expected_status="Failure",
             rule="failure_row", description="Случайные байты с расширением .zip (не архив)", build=c_corrupt_zip),
        dict(case="nested_depth5", file="a/b/c/d/e/CR000000.dcm", source=SRC_SPINE_POS, expected_status="Success",
             rule="same_class", description="Вложенность каталогов глубиной 5", build=c_copy),
        dict(case="cyrillic_spaces", file="Исследование № 7 (копия)/снимок бедра  левый.dcm", source=SRC_LHIP_OK,
             expected_status="Success", rule="same_class", description="Кириллица, пробелы и скобки в именах",
             build=c_copy),
    ]
    return rows


def build_zip_with_broken(phantoms: Path, case_dir: Path) -> None:
    stage = case_dir / "_stage"
    (stage / "norm").mkdir(parents=True, exist_ok=True)
    (stage / "broken").mkdir(parents=True, exist_ok=True)
    shutil.copy2(phantoms / SRC_RHIP_OK, stage / "norm" / "CR000000.dcm")
    c_truncated_half(phantoms / SRC_SPINE_OK, stage / "broken" / "CR000001.dcm")
    zpath = case_dir / "Выгрузка (битый внутри).zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(stage).as_posix())
    shutil.rmtree(stage)


def build_input(phantoms: Path, dst: Path, cases: list[dict]) -> dict:
    """Копия фантомов (те же относительные пути) + stress/<case>/... Возвращает статусы записи."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for p in sorted(phantoms.rglob("*.dcm")):
        rel = p.relative_to(phantoms)
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst / rel)
    status: dict = {}
    done_zip = False
    for c in cases:
        case_dir = dst / "stress" / c["case"]
        case_dir.mkdir(parents=True, exist_ok=True)
        if c["case"] == "zip_with_broken":
            if not done_zip:
                build_zip_with_broken(phantoms, case_dir)
                done_zip = True
            status[c["case"]] = "записан"
            continue
        out = case_dir / c["file"]
        try:
            c["build"](phantoms / c["source"], out)
            status[c["case"]] = "записан"
        except Exception as e:  # noqa: BLE001
            status[c["case"]] = f"не записан: {type(e).__name__}: {str(e)[:120]}"
    return status


# --------------------------------------------------------------------------- #
# Прогон и проверки
# --------------------------------------------------------------------------- #
def run_inference(inp: Path, out_csv: Path, python: str) -> tuple[int, float, str]:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "2")
    env["PYTHONHASHSEED"] = "0"
    env["DENSITO_ROOT"] = str(ROOT)
    cmd = [python, str(ROOT / "src" / "inference.py"), "--input", str(inp), "--output", str(out_csv), "--debug-csv"]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    log = (p.stdout or "") + (p.stderr or "")
    (out_csv.parent / "stdout.log").write_text(log, encoding="utf-8")
    return p.returncode, round(time.perf_counter() - t0, 1), log


def find_rows(rows: list[dict], rel: str) -> list[dict]:
    rel_n = _norm(rel)
    return [r for r in rows if _norm(r[PATH_COL]).endswith(rel_n)]


def evaluate(cases: list[dict], write_status: dict, base_rows: list[dict], header: list[str],
             rows: list[dict], dbg_rows: list[dict], rc: int, log: str, elapsed: float,
             tol: float, budget: float, n_expected_files: int) -> dict:
    problems: list[str] = []
    fb = fallback_prob()
    base_by_path = {_norm(r[PATH_COL]): r for r in base_rows}

    # 1. процесс, заголовок, журнал
    if rc != 0:
        problems.append(f"inference.py завершился кодом {rc}")
    if header != COLUMNS:
        problems.append(f"заголовок CSV: {header}")
    if "Traceback" in log:
        problems.append("в журнале прогона есть Traceback")

    # 2. одна строка на файл, лишних нет
    if len(rows) != n_expected_files:
        problems.append(f"строк {len(rows)}, ожидалось {n_expected_files} (файлов во входе, включая zip)")

    dbg_by_path: dict[str, dict] = {}
    for d in dbg_rows:
        dbg_by_path[_norm(d.get("file", ""))] = d

    results: list[dict] = []
    max_dprob = 0.0
    for c in cases:
        rel = f"stress/{c['case']}/{c['file']}"
        res = {"case": c["case"], "file": c["file"], "expected_status": c["expected_status"], "rule": c["rule"],
               "written": write_status.get(c["case"], ""), "description": c["description"],
               "status": "", "region": "", "quality_class": "", "violation_type": "", "quality_prob": "",
               "reason": "", "dprob": None, "ok": False, "problems": []}
        found = find_rows(rows, rel)
        if not write_status.get(c["case"], "").startswith("записан"):
            res["problems"].append(f"случай не создан: {write_status.get(c['case'])}")
        elif len(found) != 1:
            res["problems"].append(f"строк для файла {len(found)}, ожидалась 1")
        else:
            r = found[0]
            res.update(status=r["processing_status"], region=r["anatomical_region"], quality_class=r["quality_class"],
                       violation_type=r["violation_type"], quality_prob=r["quality_prob"])
            fnorm = _norm(c["file"])
            if ".zip/" in fnorm:  # элемент архива: в debug CSV путь во временном каталоге распаковки
                inner = fnorm.split(".zip/", 1)[1]
                dbg = next((d for k, d in dbg_by_path.items() if k.endswith("/" + inner) and f"/{c['case']}/" not in k), None)
            else:
                dbg = next((d for k, d in dbg_by_path.items() if k.endswith(fnorm.split("/")[-1])
                            and f"/{c['case']}/" in k), None)
            if dbg is not None:
                res["reason"] = (dbg.get("error") or "")[:160]
            if r["processing_status"] != c["expected_status"]:
                res["problems"].append(f"статус {r['processing_status']}, ожидался {c['expected_status']}")
            if c["rule"] == "failure_row" or r["processing_status"] == "Failure":
                if r["quality_class"] != "0":
                    res["problems"].append(f"Failure с quality_class={r['quality_class']}")
                if r["violation_type"] != "":
                    res["problems"].append(f"Failure с violation_type={r['violation_type']!r}")
                if r["quality_prob"] != fb:
                    res["problems"].append(f"Failure с quality_prob={r['quality_prob']} (ожидалось {fb})")
                if not r["study_uid"] or not r["image_uid"]:
                    res["problems"].append("Failure с пустым study_uid/image_uid")
                if dbg is not None and not (dbg.get("error") or "").strip():
                    res["problems"].append("в debug CSV нет причины (error пуст)")
            if c["rule"] == "same_class" and r["processing_status"] == "Success":
                b = base_by_path.get(_norm(c["source"]))
                if b is None:
                    res["problems"].append(f"нет строки источника {c['source']} в эталоне")
                else:
                    for col in SAME_COLS:
                        if r[col] != b[col]:
                            res["problems"].append(f"{col}: {r[col]!r} != {b[col]!r} (источник {c['source']})")
                    try:
                        dp = abs(float(r["quality_prob"]) - float(b["quality_prob"]))
                        res["dprob"] = dp
                        max_dprob = max(max_dprob, dp)
                        if dp > tol:
                            res["problems"].append(f"|Δ quality_prob| = {dp:.3g} > {tol:g}")
                    except ValueError:
                        res["problems"].append("quality_prob не число")
                    if r["study_uid"] != b["study_uid"] or r["image_uid"] != b["image_uid"]:
                        res["problems"].append("UID отличаются от источника")
            if r["processing_status"] not in ("Success", "Failure"):
                res["problems"].append(f"статус вне словаря: {r['processing_status']}")
        res["ok"] = not res["problems"]
        results.append(res)
        problems += [f"{c['case']}: {p}" for p in res["problems"]]

    # 6. строки исходных фантомов побитово равны одиночному прогону
    n_bit, bit_bad = 0, []
    for b in base_rows:
        rel = _norm(b[PATH_COL])
        found = [r for r in rows if _norm(r[PATH_COL]) == rel]
        if len(found) != 1:
            bit_bad.append(f"{rel}: строк {len(found)}")
            continue
        n_bit += 1
        if row_bytes(found[0]) != row_bytes(b):
            bit_bad.append(f"{rel}: {row_bytes(found[0]).decode().strip()} != {row_bytes(b).decode().strip()}")
    if bit_bad:
        problems.append("норма в смешанном пакете не совпала с одиночным прогоном: " + "; ".join(bit_bad[:4]))

    # 7. время
    if elapsed > budget:
        problems.append(f"время прогона {elapsed} с > бюджета {budget:g} с")

    n_fail_exp = sum(1 for c in cases if c["expected_status"] == "Failure")
    n_succ_exp = len(cases) - n_fail_exp
    return {
        "ok": not problems, "problems": problems, "n_cases": len(cases), "n_expected_failure": n_fail_exp,
        "n_expected_success": n_succ_exp, "n_cases_ok": sum(1 for r in results if r["ok"]),
        "n_rows": len(rows), "n_expected_files": n_expected_files, "rc": rc, "elapsed_s": elapsed,
        "budget_s": budget, "tol": tol, "max_dprob_same_class": max_dprob, "fallback_quality_prob": fb,
        "n_baseline_rows_bitwise": n_bit, "baseline_bitwise_ok": not bit_bad,
        "baseline_sha256_no_time": sha256_bytes(b"".join(row_bytes(r) for r in base_rows)),
        "mixed_baseline_part_sha256_no_time": sha256_bytes(b"".join(
            row_bytes(next(r for r in rows if _norm(r[PATH_COL]) == _norm(b[PATH_COL]))) for b in base_rows
            if any(_norm(r[PATH_COL]) == _norm(b[PATH_COL]) for r in rows))),
        "cases": results,
    }


def count_input_files(inp: Path) -> int:
    """Число строк, которое должен дать пакет: файлы вне архивов + элементы исправных zip; битый zip — 1 строка."""
    n = 0
    for p in inp.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(p) as zf:
                    n += sum(1 for zi in zf.infolist() if not zi.is_dir())
            except zipfile.BadZipFile:
                n += 1
        else:
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Таблица для docs/STRESS_SET.md и expected_stress.csv
# --------------------------------------------------------------------------- #
def write_expected(cases: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(EXPECTED_FIELDS)
        for c in cases:
            w.writerow([c[k] for k in EXPECTED_FIELDS])


def apply_expected(cases: list[dict], exp_csv: Path) -> list[str]:
    """Ожидания из tests/stress/expected_stress.csv переопределяют встроенные (кроме jpeg_baseline,
    зависящего от кодека). Возвращает список расхождений между файлом и встроенным списком."""
    if not exp_csv.exists():
        return [f"нет {exp_csv}"]
    _, erows = read_rows(exp_csv)
    by_key = {(r["case"], r["file"]): r for r in erows}
    notes = []
    for c in cases:
        e = by_key.get((c["case"], c["file"]))
        if e is None:
            notes.append(f"{c['case']}/{c['file']}: нет в expected_stress.csv")
            continue
        if c["case"] == "jpeg_baseline":
            continue
        if e["expected_status"] != c["expected_status"] or e["rule"] != c["rule"]:
            notes.append(f"{c['case']}: файл ожиданий {e['expected_status']}/{e['rule']}, "
                         f"встроено {c['expected_status']}/{c['rule']}")
            c["expected_status"], c["rule"] = e["expected_status"], e["rule"]
    for k in by_key:
        if not any((c["case"], c["file"]) == k for c in cases):
            notes.append(f"{k[0]}/{k[1]}: есть в expected_stress.csv, но не строится")
    return notes


def markdown_table(res: dict, env: dict) -> str:
    lines = [
        "# Стресс-набор устойчивости входа",
        "",
        f"Сгенерировано `tools/stress_set.py` {time.strftime('%Y-%m-%d %H:%M')}; окружение: pydicom {env['pydicom']}, "
        f"numpy {env['numpy']}, Python {env['python']}; декодер JPEG Baseline: {env['jpeg_decoder']}.",
        "",
        "Источник всех случаев — синтетические фантомы `tests/phantoms/` (кадры пациентов не используются). "
        "Пакет = копия 15 фантомов (с исходными путями) + подкаталог `stress/` со случаями ниже; обрабатывается одним "
        "прогоном `src/inference.py`. Правила: `failure_row` — строка Failure по `src/inference.py` "
        f"(quality_class 0, violation_type пустой, quality_prob {res['fallback_quality_prob']}, причина в debug CSV); "
        "`same_class` — Success, регион, класс и тип нарушения равны строке исходного фантома в одиночном прогоне, "
        f"|Δ quality_prob| ≤ {res['tol']:g}; `success` — Success (кадр изменён, класс не сравнивается).",
        "",
        f"Прогон пакета: {res['n_rows']} строк на {res['n_expected_files']} файлов за {res['elapsed_s']} с "
        f"(бюджет {res['budget_s']:g} с), код возврата {res['rc']}. Случаев {res['n_cases']}, ожидается Failure "
        f"{res['n_expected_failure']}, Success {res['n_expected_success']}; пройдено {res['n_cases_ok']}/{res['n_cases']}. "
        f"Строки 15 фантомов в смешанном пакете {'побитово равны' if res['baseline_bitwise_ok'] else 'НЕ равны'} "
        f"одиночному прогону (без time_of_processing), sha256 {res['baseline_sha256_no_time'][:16]}…; "
        f"max |Δ quality_prob| по правилу same_class {res['max_dprob_same_class']:.1e}.",
        "",
        "| Случай | Описание | Источник | Ожидание | Правило | Статус | Регион | Класс | Тип нарушения | prob | Причина (debug) | Итог |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in res["cases"]:
        reason = c["reason"].replace("|", "\\|")[:90]
        lines.append(f"| `{c['case']}` | {c['description']} | `{c.get('source', '')}` | {c['expected_status']} | "
                     f"`{c['rule']}` | {c['status'] or '—'} | {c['region'] or '—'} | {c['quality_class'] or '—'} | "
                     f"{c['violation_type'] or '—'} | {c['quality_prob'] or '—'} | {reason or '—'} | "
                     f"{'OK' if c['ok'] else 'FAIL: ' + '; '.join(c['problems'])[:120]} |")
    lines += [
        "",
        "Примечания.",
        "",
        "- Причина сбоя пишется в `results_debug.csv` (колонка `error`), в основной CSV — только `Failure`; "
        "`study_uid`/`image_uid` строки Failure берутся из тегов, если заголовок читается, иначе — `hash-<sha>` "
        "от содержимого файла и папки.",
        "- Файлы фантомов не содержат PixelSpacing, поэтому случай `no_pixel_spacing` дополнительно удаляет и "
        "ImagerPixelSpacing; сервис подставляет паспортный размер пикселя 1,05 × 0,6 мм.",
        "- `jpeg_baseline`: кодировщика JPEG у pydicom нет, поток JPEG даёт Pillow (входит в requirements.txt). "
        "Ожидание зависит от наличия декодера в окружении и вычисляется при запуске; в `expected_stress.csv` "
        "записано ожидание для образа (Pillow есть → Success).",
        "- Огромный кадр строится повтором отсчётов фантома без интерполяции; его класс не сравнивается "
        "с исходным (геометрия в пикселях меняется), проверяется только контролируемый Success.",
        "- Набор не доказывает качество на реальных снимках и не покрывает сжатия с потерями иных кодеков "
        "(JPEG 2000, JPEG-LS: см. `docs/TRANSFER_SYNTAX_MATRIX.md`).",
    ]
    return "\n".join(lines) + "\n"


def env_info() -> dict:
    ok, plugins = jpeg_decoder_available()
    return {"python": sys.version.split()[0], "platform": platform.platform(), "pydicom": pydicom.__version__,
            "numpy": np.__version__, "pillow": pil_available(),
            "jpeg_decoder": (f"есть ({plugins})" if ok else "нет")}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phantoms", default=str(ROOT / "tests" / "phantoms"))
    ap.add_argument("--baseline", default="", help="CSV одиночного прогона на фантомах (иначе выполняется здесь)")
    ap.add_argument("--out", required=True, help="JSON результата")
    ap.add_argument("--workdir", default="", help="рабочий каталог (по умолчанию временный)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--tol", type=float, default=1e-3, help="допуск |Δ quality_prob| для правила same_class")
    ap.add_argument("--budget", type=float, default=300.0, help="бюджет времени прогона, с")
    ap.add_argument("--huge", default=DEFAULT_HUGE, help="размер огромного кадра rows x cols (по умолчанию 4000x3000)")
    ap.add_argument("--expected", default=str(EXPECTED_CSV), help="CSV ожиданий (tests/stress/expected_stress.csv)")
    ap.add_argument("--write-expected", action="store_true", help="перезаписать CSV ожиданий встроенным списком")
    ap.add_argument("--md", default="", help="записать таблицу случаев в Markdown (docs/STRESS_SET.md)")
    ap.add_argument("--keep", action="store_true", help="не удалять рабочий каталог")
    a = ap.parse_args()

    hr, hc = (int(x) for x in a.huge.lower().split("x"))
    cases = case_table((hr, hc))
    if a.write_expected:
        write_expected(case_table((hr, hc), jpeg_ok=True), Path(a.expected))
        print(f"expected_stress.csv записан: {a.expected} ({len(cases)} случаев)")
        return 0
    exp_notes = apply_expected(cases, Path(a.expected))

    phantoms = Path(a.phantoms).resolve()
    work = Path(a.workdir).resolve() if a.workdir else Path(tempfile.mkdtemp(prefix="densito_stress_"))
    work.mkdir(parents=True, exist_ok=True)
    inp = work / "input"
    write_status = build_input(phantoms, inp, cases)
    n_files = count_input_files(inp)

    base_csv = Path(a.baseline).resolve() if a.baseline else work / "baseline" / "results.csv"
    result: dict = {"phantoms": str(phantoms), "workdir": str(work), "huge": f"{hr}x{hc}", "env": env_info(),
                    "expected_csv": a.expected, "expected_notes": exp_notes, "write_status": write_status}
    if not a.baseline:
        rc0, t0, _ = run_inference(phantoms, base_csv, a.python)
        result["baseline_run"] = {"rc": rc0, "elapsed_s": t0}
    if not base_csv.exists():
        result.update(ok=False, problems=[f"нет эталонного CSV {base_csv}"])
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1
    _, base_rows = read_rows(base_csv)
    result["baseline_csv"] = str(base_csv)

    out_csv = work / "mixed" / "results.csv"
    rc, elapsed, log = run_inference(inp, out_csv, a.python)
    header, rows = read_rows(out_csv) if out_csv.exists() else ([], [])
    dbg_csv = out_csv.with_name("results_debug.csv")
    dbg_rows = read_rows(dbg_csv)[1] if dbg_csv.exists() else []
    for c in cases:
        c.pop("build", None)
    res = evaluate(cases, write_status, base_rows, header, rows, dbg_rows, rc, log, elapsed,
                   a.tol, a.budget, n_files)
    for r_case, c in zip(res["cases"], cases):
        r_case["source"] = c["source"]
    result.update(res)
    result["mixed_csv"] = str(out_csv)
    if exp_notes and any("нет в expected" in n or "не строится" in n for n in exp_notes):
        result["ok"] = False
        result["problems"] = result["problems"] + [f"ожидания: {n}" for n in exp_notes]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if a.md:
        Path(a.md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.md).write_text(markdown_table(result, result["env"]), encoding="utf-8")

    for c in result["cases"]:
        mark = "OK  " if c["ok"] else "FAIL"
        print(f"[{mark}] {c['case']:20s} ожидание {c['expected_status']:7s} статус {c['status'] or '—':7s} "
              f"{c['region'][:12]:12s} класс {c['quality_class'] or '—'} {c['reason'][:60]}")
    print(f"норма в пакете побитово: {'да' if result['baseline_bitwise_ok'] else 'нет'} "
          f"({result['n_baseline_rows_bitwise']} строк); строк {result['n_rows']}/{result['n_expected_files']}; "
          f"время {elapsed} с; код {rc}")
    if result["problems"]:
        print("ПРОБЛЕМЫ:\n  " + "\n  ".join(result["problems"]))
    print("STRESS:", "OK" if result["ok"] else "FAILED", f"({result['n_cases_ok']}/{result['n_cases']} случаев)")
    if not a.keep and not a.workdir:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
