"""Строит эталон экспозиции (models/exposure_reference.json) по обучающей выборке
и печатает распределение оценённой γ — проверка, что на обучении канонизация ≈ тождество.

Запуск: python tools/build_exposure_reference.py [--write]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geometry_features import read_dicom_normalized  # noqa: E402
import preprocess  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="записать models/exposure_reference.json")
    a = ap.parse_args()

    df = pd.read_csv(ROOT / "data" / "labels_full.csv")
    df = df[df["region"].isin(["spine", "right_hip", "left_hip"])].reset_index(drop=True)

    rows, skipped = [], 0
    for path in df["file_path"]:
        try:
            img, _ = read_dicom_normalized(path)
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        q = preprocess.body_quantiles(img)
        if q is None:
            skipped += 1
            continue
        rows.append(q)

    Q = np.stack(rows)
    ref = np.median(Q, axis=0)
    print(f"кадров {len(Q)} (пропущено {skipped})")
    print("квантили тела (probs {}):".format(preprocess.QUANTILE_PROBS))
    for i, p in enumerate(preprocess.QUANTILE_PROBS):
        print(f"  p{p:<5.1f} медиана {ref[i]:.4f}  разброс [{Q[:, i].min():.4f}; {Q[:, i].max():.4f}]")

    gammas = np.array([preprocess.estimate_gamma_from_quantiles(q, ref)
                       if hasattr(preprocess, "estimate_gamma_from_quantiles")
                       else _gamma(q, ref) for q in Q])
    print(f"\nγ на обучении: медиана {np.median(gammas):.4f}, "
          f"|γ−1| медиана {np.median(np.abs(gammas - 1)):.4f}, p95 {np.percentile(np.abs(gammas - 1), 95):.4f}, "
          f"диапазон [{gammas.min():.3f}; {gammas.max():.3f}]")
    clipped = int(((gammas <= preprocess.GAMMA_CLIP[0] + 1e-9) | (gammas >= preprocess.GAMMA_CLIP[1] - 1e-9)).sum())
    print(f"упёрлись в ограничитель {preprocess.GAMMA_CLIP}: {clipped}/{len(gammas)}")

    if a.write:
        payload = {
            "quantile_probs": list(preprocess.QUANTILE_PROBS),
            "reference": [float(x) for x in ref],
            "n_frames": int(len(Q)),
            "source": "data/labels_full.csv, медиана квантилей пикселей тела после окна 1–99 %",
            "body_threshold": preprocess.BODY_THRESHOLD,
            "body_blur": list(preprocess.BODY_BLUR),
            "gamma_clip": list(preprocess.GAMMA_CLIP),
        }
        out = ROOT / "models" / "exposure_reference.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"\nзаписан {out}")


def _gamma(q, ref):
    ok = (q > 0.02) & (q < 0.98) & (ref > 0.02) & (ref < 0.98)
    if int(ok.sum()) < 2:
        return 1.0
    g = float(np.median(np.log(ref[ok]) / np.log(q[ok])))
    return float(np.clip(g, *preprocess.GAMMA_CLIP)) if np.isfinite(g) else 1.0


if __name__ == "__main__":
    main()
