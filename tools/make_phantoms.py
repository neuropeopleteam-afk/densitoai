#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_phantoms.py — генератор синтетических DICOM-фантомов для самопроверки DensitoAI.

Пиксели полностью синтетические (эллипс «тела» с шумом, «кость» — яркие фигуры), теги и
размеры кадра повторяют экспорт GE Lunar Prodigy (Modality CR, MONOCHROME2, 300 px позвоночник,
280/248 px бедро, 8/16 бит). Персональных данных нет: PatientName/PatientID — PHANTOM.
Генерация детерминирована (seed): один и тот же вызов даёт байт-в-байт одинаковые файлы.

Выход (по умолчанию tests/phantoms/):
    study_01..study_04/CR00000{0,1,2}.dcm   — 4 исследования × (позвоночник, правое бедро, левое бедро)
    broken/truncated_pixels.dcm              — заголовок цел, PixelData обрезан
    broken/no_pixel_data.dcm                 — DICOM без PixelData
    broken/not_a_dicom.dcm                   — текстовый файл с расширением .dcm
    MANIFEST.json                            — sha256, UID, ожидаемый processing_status по каждому файлу

Запуск из корня проекта:  python tools/make_phantoms.py [--out tests/phantoms] [--seed 20260919]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import (ExplicitVRLittleEndian, ImplicitVRLittleEndian,
                         generate_uid)
from scipy import ndimage

PHANTOM_VERSION = "1.1"
UID_ROOT = "2.25."          # UUID-производный корень (ISO/IEC 9834-8), не требует регистрации
CR_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.1"

# --------------------------------------------------------------------------- #
# Растеризация без внешних библиотек рисования (детерминированно на любой платформе)
# --------------------------------------------------------------------------- #
def _grid(h: int, w: int):
    yy, xx = np.mgrid[0:h, 0:w]
    return yy.astype(np.float64), xx.astype(np.float64)


def ellipse(h, w, cy, cx, ry, rx, angle_deg=0.0):
    yy, xx = _grid(h, w)
    a = np.deg2rad(angle_deg)
    y, x = yy - cy, xx - cx
    xr = x * np.cos(a) + y * np.sin(a)
    yr = -x * np.sin(a) + y * np.cos(a)
    return (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0


def rot_rect(h, w, cy, cx, half_h, half_w, angle_deg=0.0):
    yy, xx = _grid(h, w)
    a = np.deg2rad(angle_deg)
    y, x = yy - cy, xx - cx
    xr = x * np.cos(a) + y * np.sin(a)
    yr = -x * np.sin(a) + y * np.cos(a)
    return (np.abs(xr) <= half_w) & (np.abs(yr) <= half_h)


def thick_line(h, w, p0, p1, thickness):
    """Отрезок p0->p1 (y, x) толщиной thickness px."""
    yy, xx = _grid(h, w)
    (y0, x0), (y1, x1) = p0, p1
    dy, dx = y1 - y0, x1 - x0
    L2 = dy * dy + dx * dx
    t = np.clip(((yy - y0) * dy + (xx - x0) * dx) / L2, 0, 1)
    py, px = y0 + t * dy, x0 + t * dx
    return np.hypot(yy - py, xx - px) <= thickness / 2.0


# --------------------------------------------------------------------------- #
# Фантомы
# --------------------------------------------------------------------------- #
def spine_phantom(rng: np.random.RandomState, variant: str, h: int = 317, w: int = 300):
    """Позвоночник AP: тёмный «столб» мягких тканей, столбик позвонков, рёбра сверху, таз снизу."""
    img = np.zeros((h, w), np.float64)
    tilt = 0.0
    cx = w / 2 + rng.uniform(-6, 6)
    if variant == "axis_tilt":
        tilt = rng.choice([-1, 1]) * rng.uniform(9, 13)
    if variant == "shifted":
        # укладка со смещением: пациент лежит не по центру стола, поэтому вместе со
        # позвоночником смещается и силуэт тела (иначе крыло таза обрезается краем
        # силуэта и измеряемая ось получает паразитный наклон — артефакт фантома)
        cx = w / 2 + rng.choice([-1, 1]) * rng.uniform(30, 42)
    body = ellipse(h, w, h / 2, cx, h * 0.75, w * 0.42)
    img[body] = 55 + 25 * rng.rand(int(body.sum()))
    img = ndimage.gaussian_filter(img, 2.0)

    # столбик из 6 позвонков с межпозвонковыми промежутками
    n_vert, vh, gap = 6, 30, 10
    top = 60
    a = np.deg2rad(tilt)
    for i in range(n_vert):
        cy = top + i * (vh + gap) + vh / 2
        # смещение по x из-за наклона оси относительно центра кадра
        dx = (cy - h / 2) * np.tan(a)
        m = rot_rect(h, w, cy, cx + dx, vh / 2, 20, tilt)
        img[m] = 175 + 45 * rng.rand(int(m.sum()))
        # поперечные отростки
        m2 = rot_rect(h, w, cy, cx + dx, 5, 38, tilt)
        img[m2 & ~m] = np.maximum(img[m2 & ~m], 140)
    # рёбра — дуги вверху
    for side in (-1, 1):
        for k in range(2):
            ring = ellipse(h, w, 40 + 14 * k, cx + side * 55, 22, 60) & ~ellipse(h, w, 40 + 14 * k, cx + side * 55, 18, 56)
            ring &= (_grid(h, w)[0] > 18)
            img[ring] = np.maximum(img[ring], 165)
    # таз — яркие крылья внизу
    for side in (-1, 1):
        pel = ellipse(h, w, h - 20, cx + side * 85, 45, 60)
        img[pel] = np.maximum(img[pel], 150 + 30 * rng.rand(int(pel.sum())))

    if variant == "metal":
        # металлический предмет: квадрат максимальной яркости рядом со столбом
        m = rot_rect(h, w, h * 0.45, cx + 70, 14, 10, 20)
        img[m] = 255
    img = ndimage.gaussian_filter(img, 0.8)
    img += rng.normal(0, 3.0, img.shape)
    img[~body] = np.clip(img[~body] * 0.15, 0, 12)
    return np.clip(img, 0, 252), {"variant": variant, "axis_tilt_deg": float(tilt), "center_x": float(cx)}


def hip_phantom(rng: np.random.RandomState, side: str, variant: str, h: int = 291, w: int = 280):
    """Проксимальный отдел бедра AP. Для правого бедра диафиз слева кадра, таз — справа (как на
    аппарате); для левого — зеркально."""
    img = np.zeros((h, w), np.float64)
    body = ellipse(h, w, h * 0.55, w / 2, h * 0.9, w * 0.62)
    img[body] = 60 + 25 * rng.rand(int(body.sum()))
    img = ndimage.gaussian_filter(img, 2.0)

    # базовая геометрия для правого бедра (диафиз слева), затем зеркалим при необходимости
    shift_x = 0.0
    if variant == "cropped_field":
        # обрезка поля: FOV сдвинут латерально, диафиз частично уходит за край кадра,
        # но таз остаётся со своей (медиальной) стороны — иначе сторону нельзя определить
        # ни детектором, ни человеком, и фантом проверял бы не то
        shift_x = -rng.uniform(38, 52)
    head_c = (95.0, 135.0 + shift_x)         # головка бедра
    neck_end = (125.0, 95.0 + shift_x)       # шейка -> большой вертел
    shaft_top = (140.0, 85.0 + shift_x)
    shaft_bot = (h + 10.0, 60.0 + shift_x)
    bone = np.zeros((h, w), bool)
    bone |= ellipse(h, w, head_c[0], head_c[1], 24, 24)
    bone |= thick_line(h, w, head_c, neck_end, 26)
    bone |= thick_line(h, w, shaft_top, shaft_bot, 30)
    bone |= ellipse(h, w, 125, 82 + shift_x, 22, 16)   # большой вертел
    # тазовые кости: вертлужная впадина и крыло
    pelvis = ellipse(h, w, 60, 190 + shift_x, 55, 70) & ~ellipse(h, w, 100, 145 + shift_x, 40, 40)
    pelvis |= ellipse(h, w, 150, 200 + shift_x, 50, 35) & ~ellipse(h, w, 150, 200 + shift_x, 30, 18)
    img[pelvis] = np.maximum(img[pelvis], 150 + 35 * rng.rand(int(pelvis.sum())))
    img[bone] = 180 + 45 * rng.rand(int(bone.sum()))
    # костномозговой канал — темнее внутри диафиза
    canal = thick_line(h, w, (shaft_top[0] + 40, shaft_top[1] - 6), shaft_bot, 10)
    img[canal] = 150
    if variant == "metal":
        m = ellipse(h, w, 200, 200 + shift_x, 9, 9)
        img[m] = 255
    img = ndimage.gaussian_filter(img, 0.8)
    img += rng.normal(0, 3.0, img.shape)
    img[~body] = np.clip(img[~body] * 0.15, 0, 12)
    img = np.clip(img, 0, 252)
    if side == "left_hip":
        img = img[:, ::-1]
    return np.ascontiguousarray(img), {"variant": variant, "side": side, "shift_x": float(shift_x)}


# --------------------------------------------------------------------------- #
# DICOM
# --------------------------------------------------------------------------- #
def det_uid(*parts: str) -> str:
    """Детерминированный UID: 2.25.<int(sha256)> (≤ 64 символа)."""
    hsh = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return UID_ROOT + str(int(hsh[:30], 16))


def build_dataset(img: np.ndarray, *, study_uid: str, series_uid: str, sop_uid: str,
                  instance_number: int, bits: int, explicit_vr: bool, pixel_spacing: bool,
                  study_idx: int) -> FileDataset:
    h, w = img.shape
    meta = FileMetaDataset()
    meta.FileMetaInformationVersion = b"\x00\x01"
    meta.MediaStorageSOPClassUID = CR_IMAGE_STORAGE
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian if explicit_vr else ImplicitVRLittleEndian
    meta.ImplementationClassUID = det_uid("densitoai-phantoms", PHANTOM_VERSION)
    meta.ImplementationVersionName = "DENSITO_PHANTOM"

    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = not explicit_vr
    ds.SpecificCharacterSet = "ISO_IR 192"
    ds.SOPClassUID = CR_IMAGE_STORAGE
    ds.SOPInstanceUID = sop_uid
    ds.StudyDate = f"2026010{study_idx}"
    ds.StudyTime = "090000"
    ds.AccessionNumber = f"PHANTOM{study_idx:03d}"
    ds.Modality = "CR"
    ds.Manufacturer = "GE Healthcare"
    ds.InstitutionName = "SYNTHETIC PHANTOM"
    ds.ReferringPhysicianName = "PHANTOM"
    ds.StudyDescription = "DXA Обследование"
    ds.SeriesDescription = "Изображения DXA"
    ds.ManufacturerModelName = "Lunar Prodigy Advance"
    ds.PatientName = "PHANTOM^SYNTHETIC"
    ds.PatientID = f"PHANTOM-{study_idx:02d}"
    ds.PatientBirthDate = ""
    ds.PatientSex = "O"
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "Synthetic phantom, no real patient"
    ds.BodyPartExamined = ""
    ds.SoftwareVersions = "18.41.005"
    ds.ViewPosition = ""
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.StudyID = f"PHANTOM{study_idx:02d}"
    ds.SeriesNumber = "2"
    ds.InstanceNumber = str(instance_number)
    ds.PatientOrientation = ["L", "F"]
    ds.Laterality = ""
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = int(h), int(w)
    ds.BitsAllocated = bits
    ds.BitsStored = bits
    ds.HighBit = bits - 1
    ds.PixelRepresentation = 0
    if pixel_spacing:
        ds.PixelSpacing = ["1.05", "0.6"]
    if bits == 8:
        ds.PixelData = np.round(img).astype(np.uint8).tobytes()
    else:
        # 16 бит: 0..4095 (12 бит эффективных, как у многих экспортов), BitsStored оставляем 16
        ds.PixelData = np.round(img / 252.0 * 4095.0).astype(np.uint16).tobytes()
    return ds


def sha256_file(p: Path) -> str:
    hsh = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hsh.update(chunk)
    return hsh.hexdigest()


# Сценарии исследований: (bits, explicit_vr, pixel_spacing, hip_width, варианты по файлам)
STUDIES = [
    dict(bits=8, explicit_vr=False, pixel_spacing=False, hip_w=280,
         spine="normal", right_hip="normal", left_hip="normal"),
    dict(bits=8, explicit_vr=False, pixel_spacing=True, hip_w=280,
         spine="axis_tilt", right_hip="cropped_field", left_hip="normal"),
    dict(bits=8, explicit_vr=True, pixel_spacing=True, hip_w=248,
         spine="metal", right_hip="normal", left_hip="metal"),
    dict(bits=16, explicit_vr=True, pixel_spacing=True, hip_w=280,
         spine="shifted", right_hip="normal", left_hip="cropped_field"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="tests/phantoms")
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--png", action="store_true", help="дополнительно сохранить PNG-превью (для отладки)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"phantom_version": PHANTOM_VERSION, "seed": args.seed, "generator": "tools/make_phantoms.py",
                "pydicom": pydicom.__version__, "numpy": np.__version__, "files": []}

    for si, spec in enumerate(STUDIES, start=1):
        rng = np.random.RandomState(args.seed * 100 + si)
        study_uid = det_uid("study", str(args.seed), str(si))
        series_uid = det_uid("series", str(args.seed), str(si))
        sdir = out / f"study_{si:02d}"
        sdir.mkdir(exist_ok=True)
        plan = [("spine", spec["spine"]), ("right_hip", spec["right_hip"]), ("left_hip", spec["left_hip"])]
        for ii, (region, variant) in enumerate(plan):
            if region == "spine":
                img, info = spine_phantom(rng, variant)
            else:
                img, info = hip_phantom(rng, region, variant, w=spec["hip_w"])
            sop_uid = det_uid("sop", str(args.seed), str(si), region)
            ds = build_dataset(img, study_uid=study_uid, series_uid=series_uid, sop_uid=sop_uid,
                               instance_number=ii + 1, bits=spec["bits"], explicit_vr=spec["explicit_vr"],
                               pixel_spacing=spec["pixel_spacing"], study_idx=si)
            fpath = sdir / f"CR{ii:06d}.dcm"
            ds.save_as(str(fpath), enforce_file_format=True)
            if args.png:
                try:
                    import cv2  # noqa: WPS433
                    cv2.imwrite(str(fpath.with_suffix(".png")), np.round(img).astype(np.uint8))
                except Exception:  # noqa: BLE001
                    pass
            manifest["files"].append({
                "path": fpath.relative_to(out).as_posix(), "kind": "phantom", "region": region,
                "variant": variant, "rows": int(img.shape[0]), "cols": int(img.shape[1]),
                "bits": spec["bits"], "transfer_syntax": str(ds.file_meta.TransferSyntaxUID),
                "pixel_spacing_tag": spec["pixel_spacing"], "study_uid": study_uid, "image_uid": sop_uid,
                "expected_status": "Success", "geometry": info,
            })

    # --- заведомо битые файлы -------------------------------------------------
    bdir = out / "broken"
    bdir.mkdir(exist_ok=True)
    rng = np.random.RandomState(args.seed * 100 + 99)
    img, _ = spine_phantom(rng, "normal")
    b_study = det_uid("study", str(args.seed), "broken")
    b_series = det_uid("series", str(args.seed), "broken")

    # 1) обрезанный PixelData: пишем корректный файл и отсекаем последние 60 % байт
    sop1 = det_uid("sop", str(args.seed), "broken", "truncated")
    ds = build_dataset(img, study_uid=b_study, series_uid=b_series, sop_uid=sop1, instance_number=1,
                       bits=8, explicit_vr=True, pixel_spacing=False, study_idx=9)
    p1 = bdir / "truncated_pixels.dcm"
    ds.save_as(str(p1), enforce_file_format=True)
    raw = p1.read_bytes()
    cut = len(raw) - int(len(ds.PixelData) * 0.6)
    p1.write_bytes(raw[:cut])
    manifest["files"].append({"path": p1.relative_to(out).as_posix(), "kind": "broken",
                              "reason": "PixelData truncated (60 % of pixel bytes removed)",
                              "study_uid": b_study, "image_uid": sop1, "expected_status": "Failure"})

    # 2) DICOM без PixelData
    sop2 = det_uid("sop", str(args.seed), "broken", "nopixels")
    ds = build_dataset(img, study_uid=b_study, series_uid=b_series, sop_uid=sop2, instance_number=2,
                       bits=8, explicit_vr=True, pixel_spacing=False, study_idx=9)
    del ds.PixelData
    p2 = bdir / "no_pixel_data.dcm"
    ds.save_as(str(p2), enforce_file_format=True)
    manifest["files"].append({"path": p2.relative_to(out).as_posix(), "kind": "broken",
                              "reason": "no PixelData element", "study_uid": b_study, "image_uid": sop2,
                              "expected_status": "Failure"})

    # 3) не DICOM: текст с расширением .dcm (нет преамбулы DICM)
    p3 = bdir / "not_a_dicom.dcm"
    p3.write_bytes(("Это не DICOM-файл. Строка-заглушка для проверки обработки ошибок DensitoAI.\n" * 20).encode("utf-8"))
    manifest["files"].append({"path": p3.relative_to(out).as_posix(), "kind": "non_dicom",
                              "reason": "plain text with .dcm extension", "expected_status": "Failure"})

    for f in manifest["files"]:
        p = out / f["path"]
        f["sha256"] = sha256_file(p)
        f["size_bytes"] = p.stat().st_size
    manifest["n_files"] = len(manifest["files"])
    manifest["n_expected_failure"] = sum(1 for f in manifest["files"] if f["expected_status"] == "Failure")
    (out / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"phantoms: {manifest['n_files']} files -> {out} (expected Failure: {manifest['n_expected_failure']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
