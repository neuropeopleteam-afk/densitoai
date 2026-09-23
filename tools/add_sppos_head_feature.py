#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_sppos_head_feature.py — колонка `synth_pos_logit` (H2, 2.4.0) в обучающих признаках контура A.

Считает логит головы «дефект укладки» (models/head_densito_synth.pth, src/sppos_head.py) по каноническим
эмбеддингам densito обучающего набора (data/embeddings_densito_canonical.npy, порядок строк =
data/labels_for_embeddings.csv) и записывает его в data/geometry_features_canonical.csv (строки позвоночника;
для бедра — пусто: признак определён только для sp_pos). Это ровно тот же расчёт, что делает инференс
(src/inference.py: DensitoInference._add_embedding_features) — из уже посчитанного эмбеддинга контура B.

    python tools/add_sppos_head_feature.py                 # записать колонку (torch)
    python tools/add_sppos_head_feature.py --backend numpy  # то же без torch (песочница)
    python tools/add_sppos_head_feature.py --check work/gpu_sppos/outputs/scores_synth_head_v3.csv \\
           --out-json verify/head_score_check.json          # сверка sigmoid(логит) с p_defect GPU-прогона

Сверка: скор GPU-прогона считался на эмбеддингах, посчитанных на GPU; CPU-эмбеддинги из data/ отличаются от
них в 4-м знаке (work/gpu_sppos/REPORT.md: max_abs_diff 0.0039), поэтому в сверке печатаются максимум и доля
строк с |Δp| <= 1e-3, а обучение стека идёт на CPU-логите — том, который воспроизводится в продакшене.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.environ.get("DENSITO_DATA_DIR", ROOT / "data"))
MODELS_DIR = Path(os.environ.get("DENSITO_MODELS_DIR", ROOT / "models"))
sys.path.insert(0, str(ROOT / "src"))

import sppos_head as sh  # noqa: E402

EMB_FILE = "embeddings_densito_canonical.npy"
GEOM_FILE = "geometry_features_canonical.csv"


def compute_logits(backend: str, E: np.ndarray, head_file: Path) -> np.ndarray:
    if backend == "torch":
        head = sh.SynthPosHead(head_file)
        return head.logit_many(E)
    w = sh.SynthPosHead.load_weights_numpy(head_file)
    return sh.logit_numpy(w, E)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["torch", "numpy"], default="torch")
    ap.add_argument("--head", default=str(MODELS_DIR / sh.HEAD_FILE))
    ap.add_argument("--check", default=None, help="CSV GPU-прогона (row_id, p_defect, ...) для сверки")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--dry-run", action="store_true", help="не писать geometry_features_canonical.csv")
    args = ap.parse_args()

    labels = pd.read_csv(DATA_DIR / "labels_for_embeddings.csv")
    E = np.load(DATA_DIR / EMB_FILE)
    assert len(labels) == len(E), f"{EMB_FILE}: {len(E)} строк, labels_for_embeddings.csv: {len(labels)}"
    logit = compute_logits(args.backend, E, Path(args.head))
    spine = (labels["region"] == "spine").values
    print(f"backend={args.backend}; строк {len(E)}, позвоночник {int(spine.sum())}; "
          f"логит: min {logit.min():.4f} max {logit.max():.4f} медиана {np.median(logit):.4f}")

    report = {"backend": args.backend, "n_rows": int(len(E)), "n_spine": int(spine.sum()),
              "logit_min": float(logit.min()), "logit_max": float(logit.max()),
              "head_file": str(Path(args.head)), "head_sha256": _sha256(Path(args.head))}

    if args.check:
        sc = pd.read_csv(args.check)
        assert len(sc) == len(labels), "строки сверочного CSV не совпадают по числу"
        key_a = labels["file_path"].astype(str).str.split("Исследования/").str[-1]
        key_b = sc["file_path"].astype(str).str.split("Исследования/").str[-1]
        assert (key_a.values == key_b.values).all(), "порядок строк сверочного CSV не совпадает с labels_for_embeddings.csv"
        p = sh.logit_to_prob(logit)
        d = np.abs(p - sc["p_defect"].values.astype(float))
        gpu_logit = np.log(np.clip(sc["p_defect"].values, 1e-6, 1 - 1e-6) / (1 - np.clip(sc["p_defect"].values, 1e-6, 1 - 1e-6)))
        dl = np.abs(logit - gpu_logit)
        from scipy.stats import spearmanr
        rho = float(spearmanr(logit[spine], gpu_logit[spine]).correlation)
        report["check"] = {
            "csv": str(args.check), "tol": args.tol,
            "max_abs_diff_p": float(d.max()), "median_abs_diff_p": float(np.median(d)),
            "n_within_tol": int((d <= args.tol).sum()), "share_within_tol": float((d <= args.tol).mean()),
            "n_within_tol_spine": int((d[spine] <= args.tol).sum()),
            "max_abs_diff_logit": float(dl.max()), "median_abs_diff_logit": float(np.median(dl)),
            "spearman_logit_spine": rho,
            "note": "скор CSV считался на GPU-эмбеддингах; здесь — CPU-эмбеддинги data/embeddings_densito_canonical.npy "
                    "(разница эмбеддингов до 0.0039 по компоненте, work/gpu_sppos/REPORT.md)",
        }
        print(f"сверка с {args.check}: max|Δp| {d.max():.5f}, медиана {np.median(d):.5f}, "
              f"в допуске {args.tol}: {(d <= args.tol).sum()}/{len(d)} (позвоночник {(d[spine] <= args.tol).sum()}/{spine.sum()}); "
              f"max|Δлогит| {dl.max():.4f}; Спирмен по позвоночнику {rho:.5f}")

    if not args.dry_run:
        gpath = DATA_DIR / GEOM_FILE
        # читаем как строки: существующие ячейки (их читают hip_pos/hip_roi и остальные критерии) должны
        # остаться байт в байт прежними — меняется только добавляемая колонка
        g = pd.read_csv(gpath, dtype=str, keep_default_na=False)
        m = dict(zip(labels["file_path"].astype(str), logit))
        vals = np.array([m.get(str(fp), np.nan) for fp in g["file_path"]], dtype=float)
        vals[(g["region"] != "spine").values] = np.nan
        n_missing = int(np.isnan(vals[(g["region"] == "spine").values]).sum())
        assert n_missing == 0, f"{n_missing} строк позвоночника без эмбеддинга"
        g[sh.FEATURE_NAME] = ["" if not np.isfinite(v) else f"{v:.10g}" for v in vals]
        g.to_csv(gpath, index=False)
        report["written"] = {"file": str(gpath), "n_spine_filled": int(np.isfinite(vals).sum())}
        print(f"записано {gpath}: колонка {sh.FEATURE_NAME}, заполнено {int(np.isfinite(vals).sum())} строк позвоночника")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def _sha256(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
