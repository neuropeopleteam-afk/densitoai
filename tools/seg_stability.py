#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Самопроверка устойчивости масок сегментации (src/segmentation_export.py).

Эталонной разметки структур в датасете нет, поэтому качество масок относительно эталона здесь
не измеряется. Вместо этого считается Dice между маской исходного снимка и маской того же снимка после
преобразований, которые не должны менять структуры:
  (а) сдвиг яркости хранимых пикселей на -10 % и +10 % (x0.9 / x1.1 с округлением и обрезкой по разрядности);
  (б) горизонтальный переворот кадра, маска переворачивается обратно;
  (в) пересохранение с другим PhotometricInterpretation (MONOCHROME2 <-> MONOCHROME1, пиксели инвертируются
      так, чтобы изображение осталось тем же).
Каждое преобразование проходит весь путь чтения инференса (inference.normalize_pixels), затем build_masks.

Запуск: python tools/seg_stability.py --out outputs/seg_stability.json [--phantoms tests/phantoms]
        [--dataset "<каталог Исследования>" --n-studies 10]
Печатает медиану и минимум Dice по структурам и наборам; None — если обе маски пустые (Dice не определён).
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pydicom  # noqa: E402

import segmentation_export as se  # noqa: E402
from inference import DEFAULT_CONFIG, normalize_pixels  # noqa: E402

SPINE_MIN_COLS = int(DEFAULT_CONFIG["regions"]["spine_min_cols"])


def guess_region(img_u8: np.ndarray) -> str:
    """Правило инференса по ширине кадра (сторона бедра на маски не влияет)."""
    return "spine" if img_u8.shape[1] >= SPINE_MIN_COLS else "right_hip"


def roundtrip(ds) -> Any:
    """Пересохранение датасета в память и чтение обратно — как если бы файл переписали."""
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    buf.seek(0)
    return pydicom.dcmread(buf)


def variant_brightness(ds, factor: float):
    """Яркость хранимых пикселей x factor с округлением и обрезкой по BitsStored (как при иной экспозиции
    или пересохранении с другой яркостью). Именно хранимые значения, а не RescaleSlope: линейный slope
    полностью снимается перцентильным окном инференса и дал бы Dice = 1 без всякой проверки."""
    d = copy.deepcopy(ds)
    arr = d.pixel_array
    bits = int(getattr(d, "BitsStored", 8) or 8)
    top = (1 << bits) - 1 if int(getattr(d, "PixelRepresentation", 0) or 0) == 0 else (1 << (bits - 1)) - 1
    new = np.clip(np.rint(arr.astype(np.float64) * factor), 0, top).astype(arr.dtype)
    d.PixelData = np.ascontiguousarray(new).tobytes()
    if len(d.PixelData) % 2:
        d.PixelData += b"\x00"
    return roundtrip(d)


def variant_flip(ds):
    d = copy.deepcopy(ds)
    arr = np.ascontiguousarray(d.pixel_array[..., ::-1]) if d.pixel_array.ndim == 2 else np.ascontiguousarray(d.pixel_array[:, :, ::-1])
    d.PixelData = arr.tobytes()
    return roundtrip(d)


def variant_photometric(ds):
    d = copy.deepcopy(ds)
    arr = d.pixel_array
    if arr.ndim != 2:
        arr = arr[0] if arr.ndim == 3 and arr.shape[-1] not in (3, 4) else arr
    bits = int(getattr(d, "BitsStored", 8) or 8)
    if int(getattr(d, "PixelRepresentation", 0) or 0) == 1:
        return None   # знаковые пиксели: инверсия неоднозначна, вариант пропускаем
    top = (1 << bits) - 1
    inv = (top - arr.astype(np.int64)).astype(arr.dtype)
    cur = str(getattr(d, "PhotometricInterpretation", "MONOCHROME2"))
    d.PhotometricInterpretation = "MONOCHROME1" if cur == "MONOCHROME2" else "MONOCHROME2"
    d.PixelData = np.ascontiguousarray(inv).tobytes()
    if d.PixelData and len(d.PixelData) % 2:
        d.PixelData += b"\x00"
    return roundtrip(d)


def masks_of(ds) -> Dict[str, np.ndarray]:
    img = normalize_pixels(ds)
    return se.build_masks(img, guess_region(img))


def evaluate_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        ds = pydicom.dcmread(str(path))
        base = masks_of(ds)
    except Exception as e:  # noqa: BLE001
        return {"file": str(path), "error": str(e)}
    out: Dict[str, Any] = {"file": str(path), "rows": int(ds.Rows), "cols": int(ds.Columns),
                           "region": guess_region(normalize_pixels(ds)),
                           "areas_px": {k: int(v.sum()) for k, v in base.items()}, "dice": {}}
    variants = {}
    try:
        variants["brightness_-10"] = masks_of(variant_brightness(ds, 0.9))
        variants["brightness_+10"] = masks_of(variant_brightness(ds, 1.1))
    except Exception as e:  # noqa: BLE001
        out["error_brightness"] = str(e)
    try:
        fl = masks_of(variant_flip(ds))
        variants["flip"] = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in fl.items()}
    except Exception as e:  # noqa: BLE001
        out["error_flip"] = str(e)
    try:
        ph = variant_photometric(ds)
        if ph is not None:
            variants["photometric"] = masks_of(ph)
    except Exception as e:  # noqa: BLE001
        out["error_photometric"] = str(e)
    for name, m in variants.items():
        out["dice"][name] = {k: se.dice(base[k], m[k]) for k in base}
    return out


def collect(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Медиана и минимум Dice по (преобразование, структура) для набора."""
    agg: Dict[str, Dict[str, List[float]]] = {}
    for r in results:
        for var, per in (r.get("dice") or {}).items():
            for k, v in per.items():
                if v is not None:
                    agg.setdefault(var, {}).setdefault(k, []).append(float(v))
    summary: Dict[str, Any] = {}
    for var, per in agg.items():
        summary[var] = {k: {"n": len(v), "median": round(float(np.median(v)), 4), "min": round(float(np.min(v)), 4),
                            "mean": round(float(np.mean(v)), 4)} for k, v in per.items()}
    return summary


def dataset_files(root: Path, n_studies: int) -> List[Path]:
    studies = sorted(p for p in root.iterdir() if p.is_dir())[:n_studies]
    files: List[Path] = []
    for s in studies:
        files.extend(sorted(p for p in s.rglob("*") if p.is_file() and p.suffix.lower() in (".dcm", ".dicom", "")))
    return files


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Устойчивость масок сегментации к преобразованиям снимка (Dice)")
    ap.add_argument("--phantoms", default=str(ROOT / "tests" / "phantoms"))
    ap.add_argument("--dataset", default=None, help="каталог с исследованиями (подкаталог = исследование)")
    ap.add_argument("--n-studies", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "seg_stability.json"))
    a = ap.parse_args(argv)

    sets: Dict[str, List[Path]] = {}
    ph = Path(a.phantoms)
    if ph.exists():
        sets["phantoms"] = sorted(p for p in ph.rglob("*.dcm") if "broken" not in p.parts)
    if a.dataset:
        sets["dataset_first_studies"] = dataset_files(Path(a.dataset), a.n_studies)

    report: Dict[str, Any] = {
        "what": "Dice между маской исходного снимка и маской после преобразования (сдвиг яркости на -10/+10 %, "
                "переворот и обратный переворот, смена PhotometricInterpretation). Это проверка устойчивости "
                "алгоритма, а не сравнение с эталоном: эталонной разметки структур в датасете нет.",
        "structures": {"spine": [s["key"] for s in se.STRUCTURES["spine"]], "hip": [s["key"] for s in se.STRUCTURES["hip"]]},
        "sets": {},
    }
    for name, files in sets.items():
        results = [evaluate_file(f) for f in files]
        ok = [r for r in results if r and "error" not in r]
        report["sets"][name] = {
            "n_files": len(files), "n_evaluated": len(ok), "n_errors": len(results) - len(ok),
            "n_spine": sum(1 for r in ok if r["region"] == "spine"), "n_hip": sum(1 for r in ok if r["region"] != "spine"),
            "summary": collect(ok),
            "files": results,
        }
        print(f"[{name}] файлов {len(files)}, оценено {len(ok)}")
        for var, per in report["sets"][name]["summary"].items():
            print(f"  {var:16s} " + "  ".join(f"{k}: med {v['median']:.4f} min {v['min']:.4f} (n={v['n']})" for k, v in per.items()))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
