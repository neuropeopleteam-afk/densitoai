#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
side_detector_audit.py — независимая проверка детектора стороны бедра.

Референс — теги DICOM (`Laterality` / `ImageLaterality`) и текстовые теги (SeriesDescription,
ProtocolName, StudyDescription, BodyPartExamined) там, где сторона в них написана. Для каждого
кадра бедра считаются: скор детектора, его решение, референс, признаки обрезки поля
(scan_length_mm, edge_distance_mm) — чтобы увидеть, ошибается ли детектор именно на обрезанных.

  python tools/side_detector_audit.py --input DIR --out CSV [--summary JSON]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pydicom  # noqa: E402

from geometry_features import read_dicom_normalized  # noqa: E402
import hip_features as hf  # noqa: E402

SIDE_TAGS = ("Laterality", "ImageLaterality")
TEXT_TAGS = ("SeriesDescription", "ProtocolName", "StudyDescription", "BodyPartExamined",
             "SeriesNumber", "AcquisitionDeviceProcessingDescription", "ViewPosition")
RIGHT_WORDS = ("ППОБ", "RIGHT HIP", "R HIP", "RT HIP", "ПРАВ")
LEFT_WORDS = ("ЛПОБ", "LEFT HIP", "L HIP", "LT HIP", "ЛЕВ")


def tag_side(ds) -> tuple[str, str]:
    """('right'|'left'|'', источник) по тегам."""
    for t in SIDE_TAGS:
        v = str(getattr(ds, t, "") or "").strip().upper()
        if v.startswith("R"):
            return "right", t
        if v.startswith("L"):
            return "left", t
    text = " ".join(str(getattr(ds, t, "") or "") for t in TEXT_TAGS).upper()
    if any(w in text for w in RIGHT_WORDS):
        return "right", "text"
    if any(w in text for w in LEFT_WORDS):
        return "left", "text"
    return "", ""


def name_side(path: Path) -> str:
    up = path.stem.upper()
    if "ППОБ" in up:
        return "right"
    if "ЛПОБ" in up:
        return "left"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", default="")
    ap.add_argument("--spine-min-cols", type=int, default=300)
    a = ap.parse_args()

    files = sorted(p for p in Path(a.input).rglob("*") if p.is_file() and p.suffix.lower() in (".dcm", ""))
    rows: list[dict] = []
    for p in files:
        try:
            ds_head = pydicom.dcmread(str(p), force=True, stop_before_pixels=True)
            cols = int(getattr(ds_head, "Columns", 0) or 0)
            if cols == 0 or cols >= a.spine_min_cols:
                continue  # позвоночник или не кадр
            img_u8, ds = read_dicom_normalized(str(p))
            mask = hf.segment_bone_hip(img_u8)
            score = float(hf.hip_side_score(img_u8, mask))
            det = "right" if score >= 0 else "left"
            ref, ref_src = tag_side(ds)
            nm = name_side(p)
            f = hf.hip_features_canonical(mask if det == "right" else np.ascontiguousarray(mask[:, ::-1]))
            rows.append({
                "path": str(p),
                "rows": int(img_u8.shape[0]), "cols": int(img_u8.shape[1]),
                "side_detected": det, "side_score": round(score, 4),
                "side_ref": ref, "side_ref_source": ref_src, "side_from_name": nm,
                "scan_length_mm": round(float(f.get("scan_length_mm", 0) or 0), 2),
                "edge_distance_mm": round(float(f.get("edge_distance_mm", 0) or 0), 2),
                "bone_area_ratio": round(float(f.get("bone_area_ratio", 0) or 0), 4),
                "study_uid": str(getattr(ds, "StudyInstanceUID", "") or ""),
            })
        except Exception as e:  # noqa: BLE001
            rows.append({"path": str(p), "error": f"{type(e).__name__}: {e}"})

    cols_out = ["path", "rows", "cols", "side_detected", "side_score", "side_ref", "side_ref_source",
                "side_from_name", "scan_length_mm", "edge_distance_mm", "bone_area_ratio",
                "study_uid", "error"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols_out)
        wr.writeheader()
        for r in rows:
            wr.writerow({c: r.get(c, "") for c in cols_out})

    ok = [r for r in rows if r.get("side_ref")]
    agree = [r for r in ok if r["side_ref"] == r["side_detected"]]
    by_study: dict[str, list] = {}
    for r in rows:
        if r.get("study_uid"):
            by_study.setdefault(r["study_uid"], []).append(r)
    # баланс внутри исследования: при чётном числе кадров должно быть 50/50
    bal_bad = []
    for s, rs in by_study.items():
        n = len(rs)
        if n % 2 == 0 and n >= 2:
            nr = sum(1 for r in rs if r["side_detected"] == "right")
            if nr != n // 2:
                bal_bad.append({"study": s, "n": n, "right": nr})
    cropped = [r for r in rows if r.get("edge_distance_mm") == 0.0 and r.get("side_ref")]
    cropped_bad = [r for r in cropped if r["side_ref"] != r["side_detected"]]
    summary = {
        "n_hip_frames": len([r for r in rows if not r.get("error")]),
        "n_errors": len([r for r in rows if r.get("error")]),
        "n_with_reference": len(ok),
        "n_agree": len(agree),
        "accuracy_vs_reference": round(len(agree) / len(ok), 4) if ok else None,
        "reference_sources": {s: sum(1 for r in ok if r["side_ref_source"] == s)
                              for s in sorted({r["side_ref_source"] for r in ok})},
        "n_studies": len(by_study),
        "studies_unbalanced": bal_bad[:20],
        "n_studies_unbalanced": len(bal_bad),
        "n_cropped_with_ref": len(cropped),
        "n_cropped_wrong": len(cropped_bad),
        "examples_wrong": [{"path": r["path"], "ref": r["side_ref"], "det": r["side_detected"],
                            "score": r["side_score"], "edge_mm": r["edge_distance_mm"],
                            "scan_len_mm": r["scan_length_mm"]}
                           for r in ok if r["side_ref"] != r["side_detected"]][:20],
    }
    if a.summary:
        Path(a.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
