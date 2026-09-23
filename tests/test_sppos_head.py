#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тест головы укладки H2 (2.4.0): признак контура A `synth_pos_logit` критерия sp_pos (src/sppos_head.py).

Проверяется:
  1. загрузка models/head_densito_synth.pth (torch и путь без torch читают одни и те же тензоры);
  2. детерминизм: два независимых расчёта совпадают бит в бит;
  3. диапазон логита на обучающих эмбеддингах (data/embeddings_densito_canonical.npy) и отсутствие NaN;
  4. сверка на 5 кадрах датасета (tests/sppos_head_reference.csv): логит совпадает с обучающей колонкой
     data/geometry_features_canonical.csv (|Δ| < 1e-6) и sigmoid(логит) — со скором GPU-прогона
     work/gpu_sppos/outputs/scores_synth_head_v3.csv (|Δp| <= 3e-3: скор GPU считался на GPU-эмбеддингах,
     CPU-эмбеддинги отличаются в 4-м знаке);
  5. отсутствие файла головы — понятная ошибка (HeadMissingError / FileNotFoundError), а не ноль;
  6. при наличии torch и фантомов (tests/phantoms): инференс кладёт признак в debug-строку (feat_synth_pos_logit),
     значение конечно и повторяется при втором прогоне.

Запуск: python tests/test_sppos_head.py   (или pytest -q tests/test_sppos_head.py)
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
MODELS_DIR = Path(os.environ.get("DENSITO_MODELS_DIR", ROOT / "models"))
DATA_DIR = Path(os.environ.get("DENSITO_DATA_DIR", ROOT / "data"))
sys.path.insert(0, str(ROOT / "src"))

import sppos_head as sh  # noqa: E402

HEAD = MODELS_DIR / sh.HEAD_FILE
REF = Path(__file__).resolve().parent / "sppos_head_reference.csv"

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:  # noqa: BLE001
    HAS_TORCH = False


def _emb():
    return np.load(DATA_DIR / "embeddings_densito_canonical.npy")


def _ref_rows():
    with open(REF, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_head_file_present_and_hashed():
    assert HEAD.exists(), f"нет {HEAD}"
    sha = (ROOT / "models" / "WEIGHTS_SHA256.txt")
    if sha.exists():
        assert "head_densito_synth.pth" in sha.read_text(encoding="utf-8"), "голова не внесена в WEIGHTS_SHA256.txt"
    manifest = ROOT / "models" / "models_manifest.json"
    if manifest.exists():
        import json
        m = json.load(open(manifest, encoding="utf-8"))
        cols = m.get("model_spine_sp_pos_geom.pkl", {}).get("feature_cols") or []
        assert sh.FEATURE_NAME in cols, f"model_spine_sp_pos_geom.pkl без {sh.FEATURE_NAME}: {cols}"


def test_numpy_weights_shapes():
    w = sh.SynthPosHead.load_weights_numpy(HEAD)
    for k, shape in sh.STATE_SHAPES.items():
        assert w[k].shape == shape, (k, w[k].shape)
        assert np.all(np.isfinite(w[k])), k


def test_determinism_and_range():
    E = _emb()
    w = sh.SynthPosHead.load_weights_numpy(HEAD)
    l1 = sh.logit_numpy(w, E)
    l2 = sh.logit_numpy(sh.SynthPosHead.load_weights_numpy(HEAD), E)
    assert np.array_equal(l1, l2), "numpy-путь не детерминирован"
    assert np.all(np.isfinite(l1)) and l1.shape == (len(E),)
    assert -5.0 < l1.min() and l1.max() < 5.0, (l1.min(), l1.max())   # на 499 кадрах: −1.65 … 2.74
    assert l1.std() > 0.1, "логит вырожден"
    if HAS_TORCH:
        head = sh.SynthPosHead(HEAD)
        t1 = head.logit_many(E)
        t2 = sh.SynthPosHead(HEAD).logit_many(E)
        assert np.array_equal(t1, t2), "torch-путь не детерминирован"
        assert np.abs(t1 - l1).max() < 1e-4, f"torch и numpy расходятся: {np.abs(t1 - l1).max()}"
        one = head.logit(E[0])
        assert abs(one - t1[0]) < 1e-6


def test_reference_frames():
    E = _emb()
    w = sh.SynthPosHead.load_weights_numpy(HEAD)
    head = sh.SynthPosHead(HEAD) if HAS_TORCH else None
    rows = _ref_rows()
    assert len(rows) == 5
    for r in rows:
        i = int(r["row_index_labels_for_embeddings"])
        lg = float(sh.logit_numpy(w, E[i])[0]) if head is None else head.logit(E[i])
        assert abs(lg - float(r["synth_pos_logit"])) < 1e-4, (r["file_rel"], lg, r["synth_pos_logit"])
        p = float(sh.logit_to_prob(lg))
        assert abs(p - float(r["p_defect_gpu"])) <= 3e-3, (r["file_rel"], p, r["p_defect_gpu"])
    # обучающая колонка совпадает с расчётом по всем строкам позвоночника
    try:
        import pandas as pd
        g = pd.read_csv(DATA_DIR / "geometry_features_canonical.csv")
        lab = pd.read_csv(DATA_DIR / "labels_for_embeddings.csv")
        m = dict(zip(lab["file_path"].astype(str), sh.logit_numpy(w, E)))
        sp = g[g["region"] == "spine"]
        d = np.abs(sp["synth_pos_logit"].values - np.array([m[str(f)] for f in sp["file_path"]]))
        assert d.max() < 1e-6, d.max()
    except FileNotFoundError:
        pass


def test_missing_file_is_explicit_error():
    bad = HEAD.with_name("no_such_head.pth")
    try:
        sh.SynthPosHead.load_weights_numpy(bad)
    except FileNotFoundError as e:
        assert "sp_pos" in str(e) and sh.HEAD_FILE in str(e)
    else:
        raise AssertionError("отсутствие файла головы не привело к ошибке")
    if HAS_TORCH:
        try:
            sh.SynthPosHead(bad)
        except FileNotFoundError as e:
            assert isinstance(e, sh.HeadMissingError)
        else:
            raise AssertionError("отсутствие файла головы не привело к ошибке (torch)")


def test_inference_debug_feature_on_phantoms():
    if not HAS_TORCH:
        return
    phantoms = ROOT / "tests" / "phantoms"
    if not phantoms.exists():
        return
    import inference as inf  # noqa: E402
    engine = inf.DensitoInference(cfg=inf.load_config(), models_dir=MODELS_DIR, use_embeddings=True)
    assert engine.sppos_head is not None, "голова не загружена при старте инференса"
    checked = 0
    for study in sorted(phantoms.glob("study_*")):
        for dcm in sorted(study.rglob("*.dcm"))[:2]:
            row, debug = engine.process_file(dcm, root=phantoms)
            if row.get("processing_status") != "Success" or debug.get("internal_region") != "spine":
                continue
            v = debug.get("feat_synth_pos_logit")
            assert v is not None and np.isfinite(float(v)), (dcm, v)
            _, debug2 = engine.process_file(dcm, root=phantoms)
            assert float(debug2["feat_synth_pos_logit"]) == float(v), "признак не повторяется при втором прогоне"
            checked += 1
            if checked >= 2:
                return
    assert checked >= 1, "ни одного кадра позвоночника среди фантомов"


if __name__ == "__main__":
    t0 = time.time()
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}  ({time.time() - t0:.1f} с)")
    print(f"все проверки пройдены за {time.time() - t0:.1f} с")
