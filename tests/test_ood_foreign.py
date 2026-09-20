#!/usr/bin/env python
"""
Проверка поведения на входе, которого в обучающей выборке быть не может.

Сценарий технической экспертизы: в сервис подают корректный DICOM, но снимок —
не денситометрия (здесь синтезируется картина, похожая на обзорный снимок грудной
клетки: два светлых поля на тёмном фоне плюс полосы рёбер).

Что обязано выполняться:
  * запрос не падает: строка получает Success, а не необработанное исключение;
  * контракт девяти колонок ТЗ не нарушается (в нём нет значения «не знаю»,
    поэтому область и класс заполняются как обычно);
  * бонусный слой честно помечает вход нетипичным: ood_flag = True, в ood_reason
    указана причина. Именно этот флаг кабинет показывает оператору.

Файл генерируется на ходу с фиксированным seed, в репозитории не хранится.
"""
import copy
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tests" / "phantoms" / "study_01"
COLUMNS_TZ = ["path_to_study", "study_uid", "image_uid", "anatomical_region", "quality_class",
              "violation_type", "quality_prob", "processing_status", "time_of_processing"]

fails = []


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def make_foreign(dst: Path) -> Path:
    """DICOM с валидными тегами, но пиксели — не денситометрия."""
    ds = pydicom.dcmread(str(sorted(SRC.glob("*.dcm"))[0]))
    h, w = ds.pixel_array.shape
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:h, 0:w]
    img = np.full((h, w), 300.0)
    for cx in (w * 0.32, w * 0.68):          # два «лёгочных поля»
        img += 2600 * np.exp(-(((x - cx) / (w * 0.17)) ** 2 + ((y - h * 0.45) / (h * 0.26)) ** 2))
    img += 500 * np.sin(y / 9.0) * (img > 700)   # «рёбра»
    img += rng.normal(0, 60, img.shape)
    out = copy.deepcopy(ds)
    out.PixelData = np.clip(img, 0, 4095).astype(ds.pixel_array.dtype).tobytes()
    path = dst / "FOREIGN_not_dxa.dcm"
    out.save_as(str(path), enforce_file_format=False)
    return path


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        inp = td / "in"
        inp.mkdir()
        make_foreign(inp)
        csv = td / "out.csv"
        env = dict(os.environ, TORCH_HOME=str(ROOT / "models" / "torch_home"), OMP_NUM_THREADS="2")
        r = subprocess.run([sys.executable, str(ROOT / "src" / "inference.py"),
                            "--input", str(inp), "--output", str(csv), "--extras"],
                           capture_output=True, text=True, env=env, timeout=1800)
        check(r.returncode == 0, f"прогон завершился без ошибки (код {r.returncode})")
        if not csv.exists():
            check(False, "csv с результатом создан")
            return 1
        df = pd.read_csv(csv)
        check(list(df.columns) == COLUMNS_TZ, "девять колонок ТЗ и их порядок не изменились")
        check(len(df) == 1, f"одна строка на один файл (получено {len(df)})")
        check(df["processing_status"].iloc[0] == "Success",
              f"нетипичный вход не роняет обработку (статус {df['processing_status'].iloc[0]})")
        check(df["quality_class"].iloc[0] in (0, 1), "класс качества заполнен значением из ТЗ")

        ex = csv.with_name(csv.stem + "_extras.csv")
        check(ex.exists(), "бонусный слой extras записан")
        if ex.exists():
            e = pd.read_csv(ex)
            check(bool(e["ood_flag"].iloc[0]), "вход помечен как нетипичный (ood_flag)")
            check(isinstance(e["ood_reason"].iloc[0], str) and e["ood_reason"].iloc[0].strip() != "",
                  f"указана причина: {str(e['ood_reason'].iloc[0])[:70]}")
    print("\nИТОГ:", "все проверки пройдены" if not fails else f"{len(fails)} провалов: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
