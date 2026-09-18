#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проверочный скрипт / smoke-тест инференса DensitoAI.

Что проверяет:
  1. Пайплайн прогоняется на реальных DICOM (образец организаторов "Для теста" +
     8 файлов из обучающей выборки, если она доступна) и печатает итоговый CSV.
  2. «Инференс никогда не падает»: в папку подмешиваются заведомо плохие входы —
     битый DICOM, пустой файл, не-DICOM с расширением .dcm, DICOM без PixelData,
     DICOM с константным (пустым) изображением, DICOM 8x8 px, MONOCHROME1,
     16-битный DICOM. Для каждого должна быть строка (Failure или Success), а
     процесс — завершиться кодом 0.
  3. Формат CSV: точные имена колонок, только разрешённые строки регионов/нарушений,
     quality_class ∈ {0,1}, quality_prob ∈ [0,1], processing_status ∈ {Success, Failure}.
  4. Воспроизводимость: два прогона дают идентичные предсказания.

Запуск:  python tests/test_inference_format.py
Код возврата 0 — всё ок, 1 — есть проблемы (печатаются).
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from inference import validate_output_csv, load_config  # noqa: E402

SAMPLE_DIR = ROOT / "tests" / "sample_test_zip"
LABELS = ROOT / "data" / "labels_full.csv"


def make_dicom(path: Path, arr: np.ndarray, photometric="MONOCHROME2", bits=8, with_pixels=True):
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = fm
    ds.is_little_endian, ds.is_implicit_VR = True, False
    ds.SOPClassUID = fm.MediaStorageSOPClassUID
    ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.Modality = "OT"
    ds.PhotometricInterpretation = photometric
    ds.SamplesPerPixel = 1
    ds.Rows, ds.Columns = arr.shape
    ds.BitsAllocated = bits
    ds.BitsStored = bits
    ds.HighBit = bits - 1
    ds.PixelRepresentation = 0
    if with_pixels:
        ds.PixelData = arr.astype(np.uint8 if bits == 8 else np.uint16).tobytes()
    ds.save_as(str(path), write_like_original=False)


def build_test_dir(tmp: Path) -> dict:
    """Собирает входную папку: реальные файлы + мусор. Возвращает ожидания по файлам."""
    expect = {}  # filename -> expected processing_status ("Success"/"Failure")
    # 1) образец организаторов
    if SAMPLE_DIR.exists():
        for f in SAMPLE_DIR.rglob("*.dcm"):
            shutil.copy(f, tmp / f.name)
            expect[f.name] = "Success"
    # 2) 8 файлов из обучающей выборки (по регионам), если доступны
    if LABELS.exists():
        df = pd.read_csv(LABELS)
        picked = []
        for region in ("spine", "right_hip", "left_hip"):
            sub = df[df["region"] == region]
            picked += list(sub["file_path"].head(3)) if region == "spine" else list(sub["file_path"].head(2))
        # плюс один спорный позитив по оси и один по артефактам, если есть
        for crit in ("sp_axis", "sp_art"):
            pos = df[df[crit] == 1]
            if len(pos):
                picked.append(pos["file_path"].iloc[0])
        study_dir = tmp / "train_samples"
        for i, fp in enumerate(picked):
            fp = Path(fp)
            if fp.exists():
                dst = study_dir / fp.parent.parent.parent.name / f"{fp.stem}_{i}.dcm"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(fp, dst)
                expect[dst.name] = "Success"
    # 3) плохие входы
    bad = tmp / "bad_inputs"
    bad.mkdir()
    (bad / "corrupt_truncated.dcm").write_bytes(b"\x00" * 128 + b"DICM" + os.urandom(300))
    expect["corrupt_truncated.dcm"] = "Failure"
    (bad / "empty_file.dcm").write_bytes(b"")
    expect["empty_file.dcm"] = "Failure"
    (bad / "not_a_dicom.dcm").write_text("hello, this is a text file pretending to be DICOM")
    expect["not_a_dicom.dcm"] = "Failure"
    make_dicom(bad / "no_pixels.dcm", np.zeros((300, 300), np.uint8), with_pixels=False)
    expect["no_pixels.dcm"] = "Failure"
    make_dicom(bad / "blank_constant.dcm", np.full((300, 300), 77, np.uint8))
    expect["blank_constant.dcm"] = "Failure"
    make_dicom(bad / "tiny_8x8.dcm", (np.random.rand(8, 8) * 255).astype(np.uint8))
    expect["tiny_8x8.dcm"] = "Failure"
    # синтетические, но валидные: должны обработаться (Success), хоть и "нестандартные"
    rng = np.random.default_rng(0)
    synth = (rng.random((320, 300)) * 60).astype(np.uint8)
    synth[40:280, 130:170] = 220  # "кость" — вертикальная полоса
    make_dicom(bad / "synthetic_mono1.dcm", 255 - synth, photometric="MONOCHROME1")
    expect["synthetic_mono1.dcm"] = "Success"
    make_dicom(bad / "synthetic_16bit_hip.dcm", (synth[:, :280].astype(np.uint16) * 200), bits=16)
    expect["synthetic_16bit_hip.dcm"] = "Success"
    # не-DICOM без расширения .dcm — должен быть проигнорирован (строки нет)
    (bad / "readme.txt").write_text("ignore me")
    return expect


def main() -> int:
    problems = []
    tmp = Path(tempfile.mkdtemp(prefix="densito_test_"))
    try:
        expect = build_test_dir(tmp)
        out_csv = tmp / "out" / "results.csv"
        cmd = [sys.executable, str(SRC / "inference.py"), "--input", str(tmp), "--output", str(out_csv),
               "--debug-csv", "--xlsx"]
        print("RUN:", " ".join(cmd))
        r1 = subprocess.run(cmd, capture_output=True, text=True)
        print(r1.stdout[-3000:])
        if r1.returncode != 0:
            problems.append(f"exit code {r1.returncode}; stderr tail: {r1.stderr[-1500:]}")
        if not out_csv.exists():
            problems.append("output CSV missing")
            raise SystemExit

        # --- формат
        fmt = validate_output_csv(out_csv, load_config())
        problems += [f"FORMAT: {p}" for p in fmt]

        df = pd.read_csv(out_csv, keep_default_na=False)
        print("\n===== RESULT CSV (как его увидит жюри) =====")
        with pd.option_context("display.max_columns", 20, "display.width", 250, "display.max_colwidth", 45):
            print(df.to_string(index=False))
        print("=============================================\n")

        # --- ожидания по статусам
        names = {Path(p).name: s for p, s in zip(df["path_to_study"], df["processing_status"])}
        for fname, exp in expect.items():
            if fname not in names:
                problems.append(f"no row for {fname}")
            elif names[fname] != exp:
                problems.append(f"{fname}: expected {exp}, got {names[fname]}")
        if "readme.txt" in names:
            problems.append("non-DICOM readme.txt produced a row")
        # --- регион на образце организаторов
        for fname, region in (("CR000000_ПОП.dcm", "Поясничный отдел позвоночника"),
                              ("CR000000_ППОБ.dcm", "Проксимальный отдел бедра"),
                              ("CR000001_ЛПОБ.dcm", "Проксимальный отдел бедра")):
            row = df[df["path_to_study"].str.endswith(fname)]
            if len(row) and row["anatomical_region"].iloc[0] != region:
                problems.append(f"{fname}: region {row['anatomical_region'].iloc[0]!r} != {region!r}")
        # --- кол-во строк = число DICOM-кандидатов
        if len(df) != len(expect):
            problems.append(f"rows {len(df)} != expected {len(expect)}")
        # --- Failure-строки консервативны
        fail = df[df["processing_status"] == "Failure"]
        if not (fail["quality_class"].astype(str) == "0").all() or not (fail["violation_type"] == "").all():
            problems.append("Failure rows must have quality_class=0 and empty violation_type")
        # --- xlsx
        if not out_csv.with_suffix(".xlsx").exists():
            problems.append("xlsx not written")

        # --- воспроизводимость
        out2 = tmp / "out" / "results_rerun.csv"
        r2 = subprocess.run(cmd[:cmd.index("--output") + 1] + [str(out2)], capture_output=True, text=True)
        if r2.returncode == 0 and out2.exists():
            d2 = pd.read_csv(out2, keep_default_na=False)
            cols = ["path_to_study", "anatomical_region", "quality_class", "violation_type", "quality_prob",
                    "processing_status"]
            if not df[cols].equals(d2[cols]):
                problems.append("re-run gave different predictions (non-deterministic)")
        else:
            problems.append("re-run failed")

        # --- время
        print(f"time_of_processing: mean={df['time_of_processing'].astype(float).mean():.3f}s "
              f"max={df['time_of_processing'].astype(float).max():.3f}s")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
