"""tools/make_pixel_hashes.py с переназначением корня путей DICOM (как run_extract.py). Выход — outputs/pixel_hashes.csv."""
import os
import runpy
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
sys.argv = [str(ROOT / "tools" / "make_pixel_hashes.py")] + sys.argv[1:]
runpy.run_path(str(ROOT / "tools" / "make_pixel_hashes.py"), run_name="__main__")
