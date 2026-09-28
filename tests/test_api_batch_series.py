#!/usr/bin/env python3
"""2.4.1: POST /api/batch пишет архив дополнительных серий рядом с CSV (ТЗ п. 2.7); DENSITO_SERIES_ZIP=0 отключает."""
from __future__ import annotations

import os
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_TMP_OUT = Path(tempfile.mkdtemp(prefix="densito_batchz_"))
os.environ["DENSITO_OUTPUT_DIR"] = str(_TMP_OUT)
os.environ.pop("DENSITO_ADMIN_KEY", None)
os.environ.pop("DENSITO_SERIES_ZIP", None)
os.environ.setdefault("OMP_NUM_THREADS", "1")

from fastapi.testclient import TestClient  # noqa: E402
import api_server as A  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def main() -> int:
    c = TestClient(A.app)
    src = ROOT / "tests" / "phantoms" / "study_01"
    r = c.post("/api/batch", json={"input_dir": str(src), "output_csv": "b1.csv"})
    check("POST /api/batch: 200", r.status_code == 200, r.text[:300])
    j = r.json() if r.status_code == 200 else {}
    zp = j.get("additional_series_zip")
    check("в ответе путь архива серий", bool(zp) and Path(zp).is_file(), str(zp))
    if zp and Path(zp).is_file():
        names = zipfile.ZipFile(zp).namelist()
        check("в архиве есть DICOM и индекс", any(n.endswith(".dcm") for n in names) and "series_index.csv" in names)
        check("архив лежит рядом с CSV", Path(zp).parent == Path(j["output_csv"]).parent)
    os.environ["DENSITO_SERIES_ZIP"] = "0"
    r2 = c.post("/api/batch", json={"input_dir": str(src), "output_csv": "b2.csv"})
    check("DENSITO_SERIES_ZIP=0: архива нет", r2.status_code == 200 and r2.json().get("additional_series_zip") is None)
    os.environ.pop("DENSITO_SERIES_ZIP", None)
    print("\nИТОГ:", "OK" if not FAILED else f"FAIL ({len(FAILED)}): " + "; ".join(FAILED))
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
