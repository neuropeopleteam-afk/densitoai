#!/usr/bin/env python3
"""Архив дополнительных серий в пакетном режиме (2.4.1, ТЗ п. 2.7).

Пакетный запуск по умолчанию кладёт рядом с results.csv архив additional_series.zip: SR на исследование,
наложение результата (DICOM Secondary Capture) и сегментация (DICOM SEG) на каждый снимок Success, индекс
series_index.csv. Проверяется на фантомах tests/phantoms (12 снимков Success в 4 исследованиях + 3 битых файла):
  * по умолчанию архив есть, DICOM в нём 12 SC + 12 SEG + SR на каждое исследование из results.csv, каждый
    читается pydicom, SOP Class соответствует виду, имена внутри — study_NNN_<uid>/image_NNN_<область>_...;
  * имена внутри архива не содержат ФИО и исходных путей (вход лежит в папке с ФИО);
  * DENSITO_SERIES_ZIP=0 — архива нет, results.csv совпадает побайтно во всех колонках, кроме time_of_processing;
  * явный --series-zip PATH пишет архив туда (и при DENSITO_SERIES_ZIP=0);
  * ошибка упаковки (путь архива занят каталогом) не ломает results.csv и код возврата (0);
  * временные каталоги серий удаляются.

Запуск:  python tests/test_series_zip.py     (код 0 — пройдено; ~1–2 мин на 2 vCPU)
"""
import csv
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PH = ROOT / "tests" / "phantoms"
FAILED = []
SOP = {"sc": "1.2.840.10008.5.1.4.1.1.7", "seg": "1.2.840.10008.5.1.4.1.1.66.4"}
SR_SOPS = {"1.2.840.10008.5.1.4.1.1.88.11", "1.2.840.10008.5.1.4.1.1.88.22", "1.2.840.10008.5.1.4.1.1.88.33"}


def check(name, cond, detail=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def run(inp: Path, out_csv: Path, extra=(), env_extra=None):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("DENSITO_SERIES_ZIP", None)
    env.update(env_extra or {})
    t = time.perf_counter()
    p = subprocess.run([sys.executable, str(ROOT / "src" / "inference.py"), "--input", str(inp), "--output", str(out_csv),
                        *extra], capture_output=True, text=True, env=env, timeout=1800)
    return p.returncode, time.perf_counter() - t, p


def rows_of(p: Path):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def without_time(rows):
    i = rows[0].index("time_of_processing")
    return [r[:i] + r[i + 1:] for r in rows]


def t_sum(rows):
    i = rows[0].index("time_of_processing")
    return sum(float(r[i]) for r in rows[1:])


def stage_dirs() -> set:
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "densito_series_*")))


def main() -> int:
    import pydicom
    tmp = Path(tempfile.mkdtemp(prefix="densito_seriesz_"))
    inp = tmp / "Иванова Мария Петровна"
    shutil.copytree(PH, inp, ignore=shutil.ignore_patterns("*.csv", "*.json"))
    stages0 = stage_dirs()

    print("1. по умолчанию: additional_series.zip рядом с results.csv")
    a = tmp / "a" / "results.csv"
    rc, wall_a, p = run(inp, a)
    check("код возврата 0", rc == 0, p.stderr[-300:])
    za = a.parent / "additional_series.zip"
    check("архив additional_series.zip создан", za.is_file())
    check("права архива как у results.csv (не 0600 от mkstemp)", za.is_file() and (za.stat().st_mode & 0o777) == (a.stat().st_mode & 0o777),
          oct(za.stat().st_mode & 0o777) if za.is_file() else "")
    ra = rows_of(a)
    n_success = sum(1 for r in ra[1:] if r[ra[0].index("processing_status")] == "Success")
    n_studies = len({r[1] for r in ra[1:]})
    check("фантомы: 12 Success, 6 исследований в results.csv", n_success == 12 and n_studies == 6, f"{n_success} {n_studies}")
    if za.is_file():
        zf = zipfile.ZipFile(za)
        names = zf.namelist()
        dcm = [n for n in names if n.endswith(".dcm")]
        check(f"DICOM в архиве: {2 * n_success + n_studies} (SC {n_success} + SEG {n_success} + SR {n_studies})",
              len(dcm) == 2 * n_success + n_studies, str(len(dcm)))
        kinds = {"sc": 0, "seg": 0, "sr": 0}
        bad = []
        for n in dcm:
            try:
                ds = pydicom.dcmread(io.BytesIO(zf.read(n)))
            except Exception as e:  # noqa: BLE001
                bad.append(f"{n}: {e}")
                continue
            sop = str(ds.SOPClassUID)
            k = "sc" if n.endswith("_overlay_SC.dcm") else "seg" if n.endswith("_SEG.dcm") else "sr"
            kinds[k] += 1
            if (k in SOP and sop != SOP[k]) or (k == "sr" and sop not in SR_SOPS):
                bad.append(f"{n}: SOP {sop}")
        check("каждый DICOM читается pydicom, SOP Class по виду", not bad, str(bad[:3]))
        check("по видам: SC 12, SEG 12, SR 6", kinds == {"sc": 12, "seg": 12, "sr": 6}, str(kinds))
        pat = re.compile(r"^study_\d{3}_[0-9A-Za-z.\-]+/(study_SR|image_\d{3}_[a-z_]+_(overlay_SC|SEG))\.dcm$")
        check("имена study_NNN_<uid>/image_NNN_<область>_...", all(pat.match(n) for n in dcm),
              str([n for n in dcm if not pat.match(n)][:3]))
        check("в именах нет ФИО и исходных путей", not [n for n in names if "Иванова" in n or "CR0000" in n or "phantom" in n.lower()])
        idx = list(csv.DictReader(io.StringIO(zf.read("series_index.csv").decode("utf-8"))))
        check("series_index.csv: строка на каждый DICOM", sorted(r["zip_path"] for r in idx) == sorted(dcm))
        ok_rows = all(ra[int(r["results_csv_row"])][2] == r["image_uid"] for r in idx if r["results_csv_row"])
        check("series_index.csv: results_csv_row указывает на строку results.csv", ok_rows)

    print("2. DENSITO_SERIES_ZIP=0: архива нет, results.csv тот же")
    b = tmp / "b" / "results.csv"
    rc, wall_b, p = run(inp, b, env_extra={"DENSITO_SERIES_ZIP": "0"})
    check("код возврата 0", rc == 0, p.stderr[-300:])
    check("архива нет", not (b.parent / "additional_series.zip").exists())
    rb = rows_of(b)
    check("results.csv совпадает во всех колонках, кроме time_of_processing", without_time(ra) == without_time(rb))
    check("заголовок и число строк те же", ra[0] == rb[0] and len(ra) == len(rb))
    print(f"  время: с архивом {wall_a:.1f} с (сумма time_of_processing {t_sum(ra):.2f} с), "
          f"без архива {wall_b:.1f} с ({t_sum(rb):.2f} с), {len(ra) - 1} файлов")

    print("3. явный --series-zip PATH (сильнее DENSITO_SERIES_ZIP=0)")
    c = tmp / "c" / "results.csv"
    custom = tmp / "c_out" / "серии.zip"
    rc, _, p = run(inp, c, ["--series-zip", str(custom)], {"DENSITO_SERIES_ZIP": "0"})
    check("код возврата 0", rc == 0, p.stderr[-300:])
    check("архив по заданному пути", custom.is_file() and len([n for n in zipfile.ZipFile(custom).namelist() if n.endswith(".dcm")]) == 30)
    check("рядом с CSV архива нет", not (c.parent / "additional_series.zip").exists())
    check("results.csv тот же", without_time(rows_of(c)) == without_time(ra))

    print("4. ошибка упаковки не ломает results.csv и код возврата")
    d = tmp / "d" / "results.csv"
    (d.parent / "additional_series.zip").mkdir(parents=True)   # путь архива занят каталогом
    rc, _, p = run(inp, d)
    check("код возврата 0", rc == 0, f"rc={rc} {p.stderr[-300:]}")
    check("results.csv записан и тот же", d.is_file() and without_time(rows_of(d)) == without_time(ra))
    log = (d.with_suffix(".log")).read_text(encoding="utf-8", errors="replace") if d.with_suffix(".log").is_file() else ""
    check("в журнале предупреждение об архиве", "additional series zip not written" in log)
    check("временных файлов архива не осталось", not list(d.parent.glob(".series_*")) and not list(d.parent.glob("*.tmp*")) and not list((d.parent / "additional_series.zip").iterdir()))
    check("временные каталоги densito_series_* удалены", not (stage_dirs() - stages0), str(stage_dirs() - stages0))

    print("5. каталог результатов внутри входной папки: повторный запуск не читает свой архив")
    e1 = inp / "out" / "results.csv"
    rc1, _, p1 = run(inp, e1)
    rc2, _, p2 = run(inp, e1.with_name("results_rerun.csv"))
    check("оба запуска — код 0", rc1 == 0 and rc2 == 0, (p1.stderr + p2.stderr)[-300:])
    re1, re2 = rows_of(e1), rows_of(e1.with_name("results_rerun.csv"))
    check("архив первого запуска создан во входной папке", (e1.parent / "additional_series.zip").is_file())
    check("повторный запуск: те же строки (архив серий не стал входом)", without_time(re1) == without_time(re2)
          and len(re2) == len(ra), f"{len(re1)} {len(re2)}")

    print("6. --sc-dir без других бонус-флагов пишет Secondary Capture (до 2.4.1 не писал ничего)")
    f = tmp / "f" / "results.csv"
    sc = tmp / "f_sc"
    rc, _, p = run(inp / "study_01", f, ["--sc-dir", str(sc)], {"DENSITO_SERIES_ZIP": "0"})
    n_sc = len(list(sc.glob("*_overlay.dcm"))) if sc.is_dir() else 0
    check("код 0, SC на каждый снимок Success (3)", rc == 0 and n_sc == 3, f"rc={rc} n_sc={n_sc}")

    print("\nИТОГ:", "OK" if not FAILED else f"FAIL ({len(FAILED)}): " + "; ".join(FAILED))
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
