"""Сверка: признаки, которые считает сервис (src/inference.py: read_and_validate -> extract_geometry, вариант
baseline, та же функция foreign_object_features), совпадают с пересчитанной выгрузкой data/geometry_features.csv
на всех 166 кадрах позвоночника. Запуск: OMP_NUM_THREADS=1 python tools/spart_position/check_service_parity.py"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import inference  # noqa: E402

OLD = "/home/user/workspace/dataset_extracted/Датасет/НД_для_обучения/Исследования/"
NEW = os.environ.get("DENSITO_DICOM_ROOT", "/home/user/workspace/work/dataset/Исследования/")
cfg = inference.load_config()
g = pd.read_csv(ROOT / "data" / "geometry_features.csv")
g = g[g.region == "spine"].reset_index(drop=True)
cols = [c for c in g.columns if c.startswith("metal_metal_")]
maxdiff = {c: 0.0 for c in cols}
for _, r in g.iterrows():
    info = inference.read_and_validate(Path(NEW + r.file_path[len(OLD):]), cfg)
    f = inference.extract_geometry(info, "spine", cfg, variant="baseline")
    for c in cols:
        maxdiff[c] = max(maxdiff[c], abs(float(f[c[len("metal_"):]] if c[len("metal_"):] in f else f[c]) - float(r[c])))
print(f"кадров {len(g)}; колонок {len(cols)}; максимум |сервис - выгрузка| по колонкам:")
for c, v in maxdiff.items():
    print(f"  {c}: {v:.3g}")
print("ИТОГ:", "совпадает" if max(maxdiff.values()) < 1e-6 else "РАСХОЖДЕНИЕ")
