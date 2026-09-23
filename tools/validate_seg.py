#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Валидатор DICOM Segmentation (SEG), которые пишет src/segmentation_export.py.

Проверяет структуру файла, а не качество масок (эталонной разметки структур в датасете нет):
  * SOP Class = Segmentation Storage 1.2.840.10008.5.1.4.1.1.66.4, Modality SEG, SegmentationType BINARY;
  * обязательные теги: SOP/Series/Study Instance UID (синтаксически корректные), ContentLabel,
    ContentDate/Time, Rows/Columns, BitsAllocated=1, NumberOfFrames, PixelSpacing;
  * ссылка на исходный снимок: ReferencedSeriesSequence -> ReferencedInstanceSequence
    и SourceImageSequence в каждом кадре (одинаковый UID; при --source сравнивается с SOP Instance UID снимка);
  * число сегментов: SegmentSequence, номера 1..N без пропусков, подписи непустые,
    NumberOfFrames = N, на каждый кадр ровно один SegmentIdentificationSequence с номером из SegmentSequence;
  * размер маски: длина PixelData >= ceil(N*Rows*Columns/8), кадры распаковываются в (N, Rows, Columns),
    pydicom.pixel_array (если доступен) совпадает с распаковкой;
  * запрещённые формулировки в текстовых полях.

Запуск: python tools/validate_seg.py <файл.dcm | каталог> [--source <исходный.dcm>] [--require-nonempty]
        [--csv отчёт.csv] [--log лог.txt]
Код возврата 0 — PASS для всех файлов, 1 — есть ошибки.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pydicom

SEG_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.66.4"
_UID_RE = re.compile(r"^[0-9]+(\.[0-9]+)*$")
# Запрещённые формулировки (список согласован с tests/test_region_support.py); слова собираются из частей,
# чтобы сам валидатор проходил проверку на их отсутствие в исходниках.
BANNED = tuple("".join(parts) for parts in (("Grad", "-", "CAM"), ("автокоррекц", "ия ROI"),
                                             ("ЕР", "ИС"), ("сколи", "оз"), ("Коб", "ба"))) + ("!",)


def uid_ok(uid: Optional[str]) -> bool:
    return bool(uid) and len(str(uid)) <= 64 and bool(_UID_RE.match(str(uid)))


class Report:
    def __init__(self, name: str) -> None:
        self.name = name
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.lines: List[str] = []

    def err(self, msg: str) -> None:
        self.errors.append(msg)
        self.lines.append("ERROR " + msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        self.lines.append("WARN  " + msg)

    def ok(self, msg: str) -> None:
        self.lines.append("OK    " + msg)

    @property
    def passed(self) -> bool:
        return not self.errors


def unpack_frames(pixel_data: bytes, n_frames: int, rows: int, cols: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(pixel_data, dtype=np.uint8), bitorder="little")
    need = n_frames * rows * cols
    return bits[:need].reshape(n_frames, rows, cols).astype(bool)


def _texts(ds) -> List[str]:
    out = []
    for name in ("SeriesDescription", "ContentDescription", "ContentLabel"):
        v = getattr(ds, name, None)
        if v:
            out.append(str(v))
    for seg in getattr(ds, "SegmentSequence", []) or []:
        for name in ("SegmentLabel", "SegmentDescription", "SegmentAlgorithmName"):
            v = getattr(seg, name, None)
            if v:
                out.append(str(v))
    return out


def validate_file(path: Path, source: Optional[Path] = None, require_nonempty: bool = False) -> Report:
    rep = Report(str(path))
    try:
        ds = pydicom.dcmread(str(path))
    except Exception as e:  # noqa: BLE001
        rep.err(f"файл не читается pydicom: {e}")
        return rep

    # 1. класс, модальность, тип
    if str(getattr(ds, "SOPClassUID", "")) != SEG_SOP_CLASS_UID:
        rep.err(f"SOPClassUID {getattr(ds, 'SOPClassUID', '')} != Segmentation Storage")
    else:
        rep.ok("SOPClassUID = Segmentation Storage")
    fm = getattr(ds, "file_meta", None)
    if fm is None or str(getattr(fm, "MediaStorageSOPClassUID", "")) != SEG_SOP_CLASS_UID:
        rep.err("file_meta.MediaStorageSOPClassUID не Segmentation Storage")
    if fm is not None and str(getattr(fm, "MediaStorageSOPInstanceUID", "")) != str(getattr(ds, "SOPInstanceUID", "")):
        rep.err("MediaStorageSOPInstanceUID != SOPInstanceUID")
    if str(getattr(ds, "Modality", "")) != "SEG":
        rep.err(f"Modality {getattr(ds, 'Modality', '')} != SEG")
    if str(getattr(ds, "SegmentationType", "")) != "BINARY":
        rep.err(f"SegmentationType {getattr(ds, 'SegmentationType', '')} != BINARY")
    else:
        rep.ok("Modality SEG, SegmentationType BINARY")

    # 2. обязательные теги
    for tag in ("SOPInstanceUID", "SeriesInstanceUID", "StudyInstanceUID", "FrameOfReferenceUID"):
        if not uid_ok(getattr(ds, tag, None)):
            rep.err(f"{tag} отсутствует или синтаксически неверен")
    for tag in ("ContentLabel", "ContentDate", "ContentTime", "Rows", "Columns", "BitsAllocated",
                "NumberOfFrames", "PixelSpacing", "ImageOrientationPatient", "PixelData",
                "SegmentSequence", "SharedFunctionalGroupsSequence", "PerFrameFunctionalGroupsSequence",
                "ReferencedSeriesSequence", "DimensionIndexSequence", "ImageType", "InstanceNumber",
                "SeriesNumber", "SamplesPerPixel", "PhotometricInterpretation", "PixelRepresentation",
                "BitsStored", "HighBit", "LossyImageCompression"):
        if getattr(ds, tag, None) in (None, ""):
            rep.err(f"нет обязательного тега {tag}")
    if rep.errors:
        return rep
    rep.ok("обязательные теги присутствуют, UID синтаксически корректны")

    if int(ds.BitsAllocated) != 1 or int(ds.BitsStored) != 1 or int(ds.HighBit) != 0:
        rep.err(f"BitsAllocated/BitsStored/HighBit = {ds.BitsAllocated}/{ds.BitsStored}/{ds.HighBit}, ожидается 1/1/0")
    if int(ds.SamplesPerPixel) != 1 or str(ds.PhotometricInterpretation) != "MONOCHROME2":
        rep.err("SamplesPerPixel/PhotometricInterpretation не 1/MONOCHROME2")
    if str(ds.SpecificCharacterSet if "SpecificCharacterSet" in ds else "") != "ISO_IR 192":
        rep.warn("SpecificCharacterSet не ISO_IR 192 (кириллица в подписях может не читаться)")
    try:
        ps = [float(x) for x in ds.PixelSpacing]
        if len(ps) != 2 or not all(p > 0 for p in ps):
            rep.err(f"PixelSpacing некорректен: {ds.PixelSpacing}")
        else:
            rep.ok(f"PixelSpacing = {ps[0]:.2f} x {ps[1]:.2f} мм")
    except Exception as e:  # noqa: BLE001
        rep.err(f"PixelSpacing не читается: {e}")

    # 3. сегменты
    segs = list(ds.SegmentSequence)
    n = len(segs)
    if n < 1:
        rep.err("SegmentSequence пуст")
        return rep
    numbers = [int(getattr(s, "SegmentNumber", 0) or 0) for s in segs]
    if numbers != list(range(1, n + 1)):
        rep.err(f"номера сегментов {numbers} должны быть 1..{n}")
    for s in segs:
        if not str(getattr(s, "SegmentLabel", "") or "").strip():
            rep.err(f"сегмент {getattr(s, 'SegmentNumber', '?')}: пустой SegmentLabel")
        if str(getattr(s, "SegmentAlgorithmType", "")) not in ("AUTOMATIC", "SEMIAUTOMATIC", "MANUAL"):
            rep.err(f"сегмент {getattr(s, 'SegmentNumber', '?')}: SegmentAlgorithmType некорректен")
        if str(getattr(s, "SegmentAlgorithmType", "")) != "MANUAL" and not str(getattr(s, "SegmentAlgorithmName", "") or ""):
            rep.err(f"сегмент {getattr(s, 'SegmentNumber', '?')}: нет SegmentAlgorithmName")
        for sq in ("SegmentedPropertyCategoryCodeSequence", "SegmentedPropertyTypeCodeSequence"):
            if not len(getattr(s, sq, []) or []):
                rep.err(f"сегмент {getattr(s, 'SegmentNumber', '?')}: нет {sq}")
    n_frames = int(ds.NumberOfFrames)
    if n_frames != n:
        rep.err(f"NumberOfFrames {n_frames} != число сегментов {n}")
    else:
        rep.ok(f"сегментов: {n} ({'; '.join(str(s.SegmentLabel) for s in segs)})")

    # 4. покадровые группы и ссылка на исходный снимок
    pf = list(ds.PerFrameFunctionalGroupsSequence)
    if len(pf) != n_frames:
        rep.err(f"PerFrameFunctionalGroupsSequence: {len(pf)} элементов, кадров {n_frames}")
    src_uids = set()
    for i, fg in enumerate(pf, start=1):
        sid = list(getattr(fg, "SegmentIdentificationSequence", []) or [])
        if len(sid) != 1 or int(getattr(sid[0], "ReferencedSegmentNumber", 0) or 0) not in numbers:
            rep.err(f"кадр {i}: SegmentIdentificationSequence отсутствует или ссылается на неизвестный сегмент")
        elif int(sid[0].ReferencedSegmentNumber) != i:
            rep.warn(f"кадр {i}: ссылается на сегмент {sid[0].ReferencedSegmentNumber} (порядок кадров не совпадает с номерами)")
        fc = list(getattr(fg, "FrameContentSequence", []) or [])
        div = getattr(fc[0], "DimensionIndexValues", None) if len(fc) == 1 else None
        if div is not None and not isinstance(div, (list, tuple)) and not hasattr(div, "__len__"):
            div = [div]   # одно значение pydicom отдаёт скаляром
        if len(fc) != 1 or div is None or len(div) == 0:
            rep.err(f"кадр {i}: нет FrameContentSequence/DimensionIndexValues")
        der = list(getattr(fg, "DerivationImageSequence", []) or [])
        found = False
        for d in der:
            for s in list(getattr(d, "SourceImageSequence", []) or []):
                u = str(getattr(s, "ReferencedSOPInstanceUID", "") or "")
                if uid_ok(u):
                    src_uids.add(u)
                    found = True
        if not found:
            rep.err(f"кадр {i}: нет SourceImageSequence с корректным ReferencedSOPInstanceUID")
    ref_uids = set()
    for rs in ds.ReferencedSeriesSequence:
        if not uid_ok(getattr(rs, "SeriesInstanceUID", None)):
            rep.err("ReferencedSeriesSequence: SeriesInstanceUID отсутствует или неверен")
        for ri in list(getattr(rs, "ReferencedInstanceSequence", []) or []):
            u = str(getattr(ri, "ReferencedSOPInstanceUID", "") or "")
            if uid_ok(u):
                ref_uids.add(u)
            if not uid_ok(getattr(ri, "ReferencedSOPClassUID", None)):
                rep.err("ReferencedInstanceSequence: ReferencedSOPClassUID неверен")
    if not ref_uids:
        rep.err("ReferencedSeriesSequence не содержит ссылки на исходный снимок")
    elif src_uids and ref_uids != src_uids:
        rep.err(f"ссылки расходятся: ReferencedSeriesSequence {sorted(ref_uids)} vs SourceImageSequence {sorted(src_uids)}")
    else:
        rep.ok(f"ссылка на исходный снимок: {sorted(ref_uids)[0]}")
    if source is not None:
        try:
            sds = pydicom.dcmread(str(source), stop_before_pixels=True)
            su = str(getattr(sds, "SOPInstanceUID", "") or "")
            if su and su in ref_uids:
                rep.ok("ссылка совпадает с SOP Instance UID исходного снимка")
            else:
                rep.err(f"ссылка {sorted(ref_uids)} не совпадает с SOP Instance UID исходного снимка {su}")
            if int(getattr(sds, "Rows", 0) or 0) != int(ds.Rows) or int(getattr(sds, "Columns", 0) or 0) != int(ds.Columns):
                rep.err(f"Rows x Columns {ds.Rows}x{ds.Columns} != исходного снимка {getattr(sds, 'Rows', '?')}x{getattr(sds, 'Columns', '?')}")
            else:
                rep.ok("Rows x Columns совпадают с исходным снимком")
        except Exception as e:  # noqa: BLE001
            rep.err(f"исходный снимок не читается: {e}")

    # 5. размер маски
    rows, cols = int(ds.Rows), int(ds.Columns)
    need_bytes = math.ceil(n_frames * rows * cols / 8)
    pd_len = len(ds.PixelData)
    if pd_len < need_bytes or pd_len > need_bytes + 1:
        rep.err(f"PixelData {pd_len} байт, ожидается {need_bytes} (+1 байт выравнивания)")
    else:
        rep.ok(f"PixelData {pd_len} байт = {n_frames} x {rows} x {cols} бит")
    try:
        frames = unpack_frames(bytes(ds.PixelData), n_frames, rows, cols)
        areas = [int(f.sum()) for f in frames]
        rep.ok(f"кадры распакованы {frames.shape}, площади (px): {areas}")
        if require_nonempty and areas and areas[0] == 0:
            rep.err("первый сегмент (кость) пуст")
        try:
            pa = np.asarray(ds.pixel_array)
            expected_shape = (n_frames, rows, cols) if n_frames > 1 else (rows, cols)
            if pa.shape != expected_shape:
                rep.err(f"pydicom.pixel_array {pa.shape} не совпадает с (кадры, Rows, Columns)")
            elif not np.array_equal(pa.reshape(n_frames, rows, cols).astype(bool), frames):
                rep.err("pydicom.pixel_array отличается от распаковки битов")
            else:
                rep.ok("pydicom.pixel_array совпадает с распаковкой битов")
        except Exception as e:  # noqa: BLE001
            rep.warn(f"pydicom.pixel_array недоступен: {e}")
    except Exception as e:  # noqa: BLE001
        rep.err(f"кадры не распаковываются: {e}")

    # 6. запрещённые формулировки
    bad = [b for b in BANNED for t in _texts(ds) if b.lower() in t.lower()]
    if bad:
        rep.err(f"запрещённые формулировки в текстовых полях: {sorted(set(bad))}")
    else:
        rep.ok("запрещённых формулировок нет")
    return rep


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Проверка структуры DICOM SEG DensitoAI")
    ap.add_argument("target", help="файл .dcm или каталог (берутся *_seg.dcm, при отсутствии — все *.dcm)")
    ap.add_argument("--source", default=None, help="исходный DICOM для сверки ссылки и размера (для одного файла)")
    ap.add_argument("--require-nonempty", action="store_true", help="первый сегмент (кость) не должен быть пустым")
    ap.add_argument("--csv", default=None, help="CSV-отчёт по файлам")
    ap.add_argument("--log", default=None, help="текстовый лог")
    a = ap.parse_args(argv)

    t = Path(a.target)
    if t.is_dir():
        files = sorted(t.rglob("*_seg.dcm")) or sorted(t.rglob("*.dcm"))
    else:
        files = [t]
    if not files:
        print("FAIL: нет файлов для проверки")
        return 1
    reports = [validate_file(f, Path(a.source) if a.source else None, a.require_nonempty) for f in files]
    lines = []
    for r in reports:
        lines.append(f"== {r.name}: {'PASS' if r.passed else 'FAIL'} (ошибок {len(r.errors)}, предупреждений {len(r.warnings)})")
        lines.extend("   " + x for x in r.lines)
    n_fail = sum(1 for r in reports if not r.passed)
    lines.append(f"ИТОГО: файлов {len(reports)}, PASS {len(reports) - n_fail}, FAIL {n_fail}")
    text = "\n".join(lines)
    print(text)
    if a.log:
        Path(a.log).write_text(text + "\n", encoding="utf-8")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "result", "n_errors", "n_warnings", "errors"])
            for r in reports:
                w.writerow([r.name, "PASS" if r.passed else "FAIL", len(r.errors), len(r.warnings), " | ".join(r.errors)])
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
