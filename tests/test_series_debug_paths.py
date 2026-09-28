#!/usr/bin/env python3
"""2.4.1: служебный results_debug.csv пакетного режима не ссылается на удалённый временный каталог серий.
Пути SC/SEG ведут внутрь additional_series.zip и существуют в архиве; PNG/JSON сегментации (в архив не входят) пустые."""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILED = []


def check(name, cond, detail=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def main() -> int:
    out = Path(tempfile.mkdtemp(prefix="densito_sdp_")) / "results.csv"
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    env.pop("DENSITO_SERIES_ZIP", None)
    r = subprocess.run([sys.executable, str(ROOT / "src" / "inference.py"), "--input", str(ROOT / "tests" / "phantoms"),
                        "--output", str(out), "--debug-csv"], env=env, capture_output=True, text=True, timeout=900)
    check("пакет на фантомах: код 0", r.returncode == 0, r.stderr[-500:])
    dbg = out.with_name("results_debug.csv")
    z = out.with_name("additional_series.zip")
    check("debug CSV и архив записаны", dbg.is_file() and z.is_file())
    if not (dbg.is_file() and z.is_file()):
        return 1
    names = set(zipfile.ZipFile(z).namelist())
    rows = list(csv.DictReader(open(dbg, encoding="utf-8")))
    text = dbg.read_text(encoding="utf-8")
    check("в debug CSV нет путей во временный каталог densito_series_", "densito_series_" not in text)
    linked = [row[k] for row in rows for k in ("bonus_overlay_dcm", "bonus_seg_dcm") if row.get(k)]
    check("есть ссылки SC/SEG на архив", len(linked) > 0, str(len(linked)))
    check("каждая ссылка SC/SEG ведёт на файл внутри архива",
          all(v.startswith("additional_series.zip/") and v.split("/", 1)[1] in names for v in linked))
    check("PNG/JSON сегментации (в архив не входят) пустые",
          all(not row.get(k) for row in rows for k in ("bonus_seg_png", "bonus_seg_json")))
    print("\nИТОГ:", "OK" if not FAILED else f"FAIL ({len(FAILED)}): " + "; ".join(FAILED))
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
