"""Пересчёт data/geometry_features.csv тем же кодом (src/extract_all_features.py -> geometry_features.extract_all_features),
с переназначением корня путей DICOM: в labels_full.csv записаны пути исходной машины
(/home/user/workspace/dataset_extracted/Датасет/НД_для_обучения/Исследования/...), файлы лежат в
DENSITO_DICOM_ROOT. Строки file_path в CSV не меняются (ключи слияния с эмбеддингами и хэшами).

Запуск: OMP_NUM_THREADS=1 nice python tools/spart_position/run_extract.py --variant baseline --out <csv>
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import pydicom  # noqa: E402

OLD = "/home/user/workspace/dataset_extracted/Датасет/НД_для_обучения/Исследования/"
NEW = os.environ.get("DENSITO_DICOM_ROOT", "/home/user/workspace/work/dataset/Исследования/")
_orig = pydicom.dcmread


def _remap(p, *a, **k):
    if isinstance(p, (str, Path)) and str(p).startswith(OLD):
        p = NEW + str(p)[len(OLD):]
    return _orig(p, *a, **k)


pydicom.dcmread = _remap
import extract_all_features  # noqa: E402
import geometry_features  # noqa: E402
geometry_features.pydicom.dcmread = _remap


def _labels_from_csv():
    """build_dataset.load_labels() читает разметка.xlsx, которого в песочнице нет. Метки по исследованию
    восстанавливаются из data/labels_full.csv (они оттуда и пришли); совпадение колонок *_c со старой
    выгрузкой проверяется в tools/spart_position/check_regen.py."""
    import pandas as pd
    lf = pd.read_csv(ROOT / "data" / "labels_full.csv")
    cols = ["sp_pos", "sp_axis", "sp_art", "rh_pos", "rh_roi", "lh_pos", "lh_roi"]
    xl = lf.groupby(lf["study"].astype(str))[cols].first()
    return xl


import build_dataset  # noqa: E402
build_dataset.load_labels = _labels_from_csv

if __name__ == "__main__":
    extract_all_features.main()
