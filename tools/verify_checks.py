#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_checks.py — проверки результатов verify.sh (вызывается из tools/verify.sh, можно и вручную).

Подкоманды:
  checks   --run1 CSV --run2 CSV --phantoms DIR --out JSON [--expected CSV] [--tol 1e-3]
           [--data-csv CSV --expected-sha SHA] [--data-dir DIR] [--timings JSON]
  predsha  CSV            — sha256 колонок предсказаний (без time_of_processing), печать в stdout
  update-expected --run1 CSV --phantoms DIR — записать tests/phantoms/expected_results.csv из run1

Канонический вид CSV предсказаний для sha256: строки CSV без колонки time_of_processing,
разделитель запятая, кавычки по правилам csv.QUOTE_MINIMAL, перевод строки "\n", кодировка UTF-8,
порядок строк — как в файле. Так же считает `tools/verify_checks.py predsha results.csv`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))

COLUMNS = ["path_to_study", "study_uid", "image_uid", "anatomical_region", "quality_class",
           "violation_type", "quality_prob", "processing_status", "time_of_processing"]
TIME_COL = "time_of_processing"
PRED_COLS = [c for c in COLUMNS if c != TIME_COL]


# --------------------------------------------------------------------------- #
def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def canonical_predictions(path: Path, drop_cols=(TIME_COL,)) -> bytes:
    header, rows = read_csv(path)
    keep = [c for c in header if c not in drop_cols]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(keep)
    for r in rows:
        w.writerow([r[c] for c in keep])
    return buf.getvalue().encode("utf-8")


def pred_sha(path: Path, drop_cols=(TIME_COL,)) -> str:
    return hashlib.sha256(canonical_predictions(path, drop_cols)).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_path(p: str) -> str:
    return p.replace("\\", "/").lstrip("./")


def _match_row(rows: list[dict], rel_path: str):
    rel = _norm_path(rel_path)
    for r in rows:
        if _norm_path(r["path_to_study"]).endswith(rel):
            return r
    return None


def env_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "machine": platform.machine(), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
            "TORCH_HOME": os.environ.get("TORCH_HOME", "")}
    for mod in ("torch", "torchvision", "numpy", "scipy", "sklearn", "pandas", "pydicom", "cv2", "skimage", "yaml"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            info[mod] = f"not importable: {type(e).__name__}"
    try:
        import torch  # noqa: WPS433
        info["torch_threads"] = torch.get_num_threads()
    except Exception:  # noqa: BLE001
        pass
    return info


# --------------------------------------------------------------------------- #
def run_checks(a) -> int:
    checks: list[dict] = []

    def add(name, ok, detail="", level="error"):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail), "level": level})

    phantoms = Path(a.phantoms)
    manifest = json.loads((phantoms / "MANIFEST.json").read_text(encoding="utf-8"))
    run1, run2 = Path(a.run1), Path(a.run2)

    # 1. схема CSV
    header, rows = read_csv(run1)
    add("Схема CSV: 9 колонок в порядке ТЗ", header == COLUMNS, f"получено: {header}")

    # 2. число строк = число входных файлов
    add("Число строк = число входных файлов", len(rows) == manifest["n_files"],
        f"строк {len(rows)}, файлов по MANIFEST {manifest['n_files']}")

    # 3. UID совпадают с тегами DICOM (для корректных фантомов)
    import pydicom  # noqa: WPS433
    bad = []
    for f in manifest["files"]:
        if f["kind"] != "phantom":
            continue
        row = _match_row(rows, f["path"])
        if row is None:
            bad.append(f"{f['path']}: строки нет")
            continue
        ds = pydicom.dcmread(str(phantoms / f["path"]), stop_before_pixels=True, force=True)
        if row["study_uid"] != str(ds.StudyInstanceUID) or row["image_uid"] != str(ds.SOPInstanceUID):
            bad.append(f"{f['path']}: {row['study_uid']}/{row['image_uid']} != тегов")
        if row["study_uid"] != f["study_uid"] or row["image_uid"] != f["image_uid"]:
            bad.append(f"{f['path']}: не совпадает с MANIFEST")
    n_ph = sum(1 for f in manifest["files"] if f["kind"] == "phantom")
    add("study_uid / image_uid совпадают с тегами DICOM", not bad, f"проверено {n_ph} файлов; " + ("; ".join(bad) or "расхождений нет"))

    # 4. Failure ровно у битых
    exp_fail = {_norm_path(f["path"]) for f in manifest["files"] if f["expected_status"] == "Failure"}
    got_fail = {_norm_path(r["path_to_study"]) for r in rows if r["processing_status"] == "Failure"}
    got_fail_rel = {p for p in got_fail}
    ok = all(any(g.endswith(e) for g in got_fail_rel) for e in exp_fail) and len(got_fail_rel) == len(exp_fail)
    add("processing_status=Failure ровно у битых файлов", ok,
        f"ожидалось {sorted(exp_fail)}, получено {sorted(got_fail_rel)}")

    # 5. значения полей
    problems = []
    for i, r in enumerate(rows, 2):
        if r["quality_class"] not in ("0", "1"):
            problems.append(f"строка {i}: quality_class={r['quality_class']}")
        try:
            p = float(r["quality_prob"])
            if not 0.0 <= p <= 1.0:
                problems.append(f"строка {i}: quality_prob={p}")
        except ValueError:
            problems.append(f"строка {i}: quality_prob не число")
        if r["processing_status"] not in ("Success", "Failure"):
            problems.append(f"строка {i}: status={r['processing_status']}")
        if r["quality_class"] == "1" and not r["violation_type"]:
            problems.append(f"строка {i}: class=1 без violation_type")
        if r["quality_class"] == "0" and r["violation_type"]:
            problems.append(f"строка {i}: class=0 с violation_type")
        if r["processing_status"] == "Failure" and r["quality_class"] != "0":
            problems.append(f"строка {i}: Failure с class=1")
    add("Значения полей: class∈{0,1}, prob∈[0,1], status, согласованность class/violation", not problems,
        "; ".join(problems) or f"{len(rows)} строк без замечаний")

    # 6. детерминизм между двумя прогонами
    c1, c2 = canonical_predictions(run1), canonical_predictions(run2)
    add("Детерминизм: два прогона совпадают бит в бит (все колонки, кроме time_of_processing)", c1 == c2,
        f"sha256 run1 {hashlib.sha256(c1).hexdigest()[:16]}…, run2 {hashlib.sha256(c2).hexdigest()[:16]}…")

    # 7. веса моделей
    weights_json = Path(a.weights_json) if a.weights_json else None
    if weights_json and weights_json.exists():
        wj = json.loads(weights_json.read_text(encoding="utf-8"))
        n_ok = sum(1 for f in wj["files"] if f["ok"])
        add("sha256 весов моделей = models/WEIGHTS_SHA256.txt", wj["ok"],
            f"{n_ok}/{len(wj['files'])} файлов совпали; нет на диске: {wj['missing_on_disk']}; "
            f"нет в списке: {wj['missing_in_manifest']}; models_manifest.json -> отсутствуют: {wj['manifest_json_missing_files']}")
    else:
        add("sha256 весов моделей = models/WEIGHTS_SHA256.txt", False, "результат hash_weights.py --check не найден")

    # 8. сравнение с эталоном
    expected = Path(a.expected) if a.expected else None
    if expected and expected.exists():
        eh, erows = read_csv(expected)
        by_path = {_norm_path(r["path_to_study"]): r for r in rows}
        diffs, max_dp, exact = [], 0.0, True
        for er in erows:
            r = by_path.get(_norm_path(er["path_to_study"]))
            if r is None:
                diffs.append(f"{er['path_to_study']}: строки нет")
                continue
            for c in ("study_uid", "image_uid", "anatomical_region", "quality_class", "violation_type", "processing_status"):
                if r[c] != er[c]:
                    diffs.append(f"{er['path_to_study']}: {c} {r[c]!r} != {er[c]!r}")
            dp = abs(float(r["quality_prob"]) - float(er["quality_prob"]))
            max_dp = max(max_dp, dp)
            if r["quality_prob"] != er["quality_prob"]:
                exact = False
            if dp > a.tol:
                diffs.append(f"{er['path_to_study']}: quality_prob {r['quality_prob']} vs {er['quality_prob']}")
        if len(erows) != len(rows):
            diffs.append(f"строк {len(rows)} vs эталон {len(erows)}")
        add(f"Совпадение с эталоном tests/phantoms/expected_results.csv (классы точно, quality_prob ±{a.tol:g})",
            not diffs, "; ".join(diffs) or f"{len(erows)} строк совпали; max |Δquality_prob| = {max_dp:.2e}; бит в бит: {'да' if exact else 'нет'}")
        add("Эталон совпадает бит в бит (sha256 предсказаний)", exact,
            f"run1 {pred_sha(run1)}; expected {hashlib.sha256(canonical_predictions(expected, ())).hexdigest()}",
            level="warning")
    else:
        add("Совпадение с эталоном tests/phantoms/expected_results.csv", False, "файл эталона не найден")

    # 9. данные пользователя (опционально)
    data_result = None
    if a.data_csv:
        dcsv = Path(a.data_csv)
        if dcsv.exists():
            dh, drows = read_csv(dcsv)
            sha_full = pred_sha(dcsv)
            sha_cls = pred_sha(dcsv, (TIME_COL, "quality_prob"))
            n_fail = sum(1 for r in drows if r["processing_status"] == "Failure")
            data_result = {"dir": a.data_dir, "csv": str(dcsv), "rows": len(drows), "failures": n_fail,
                           "sha256_predictions": sha_full, "sha256_predictions_without_prob": sha_cls,
                           "expected_sha": a.expected_sha or ""}
            add("Данные пользователя: CSV получен, схема верна", dh == COLUMNS,
                f"{len(drows)} строк, Failure {n_fail}, sha256 предсказаний {sha_full}")
            if a.expected_sha:
                ok = a.expected_sha.lower().strip() == sha_full or a.expected_sha.lower().strip() == sha_cls
                add("Данные пользователя: sha256 предсказаний = --expected-sha", ok,
                    f"получено {sha_full} (без quality_prob: {sha_cls}), ожидалось {a.expected_sha}")
        else:
            add("Данные пользователя: инференс завершился", False, f"нет файла {dcsv}")

    timings = {}
    if a.timings and Path(a.timings).exists():
        timings = json.loads(Path(a.timings).read_text(encoding="utf-8"))

    hard_fail = [c for c in checks if not c["ok"] and c["level"] == "error"]
    result = {
        "ok": not hard_fail,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "root": str(ROOT), "phantoms": str(phantoms), "run1": str(run1), "run2": str(run2),
        "checks": checks, "env": env_info(), "timings": timings,
        "phantoms_manifest": {"n_files": manifest["n_files"], "n_expected_failure": manifest["n_expected_failure"],
                              "seed": manifest["seed"], "phantom_version": manifest["phantom_version"]},
        "sha256": {
            "run1_predictions": pred_sha(run1), "run2_predictions": pred_sha(run2),
            "expected_results_csv": sha256_file(expected) if expected and expected.exists() else "",
            "phantoms_manifest_json": sha256_file(phantoms / "MANIFEST.json"),
            "run1_csv_file": sha256_file(run1),
        },
        "weights": json.loads(weights_json.read_text(encoding="utf-8")) if weights_json and weights_json.exists() else {},
        "user_data": data_result,
    }
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for c in checks:
        mark = "OK  " if c["ok"] else ("WARN" if c["level"] == "warning" else "FAIL")
        print(f"[{mark}] {c['name']}")
        if not c["ok"]:
            print("       " + c["detail"][:600])
    print("VERIFY:", "OK" if result["ok"] else "FAILED", f"({len(checks) - len(hard_fail)}/{len(checks)} checks passed)")
    return 0 if result["ok"] else 1


def update_expected(a) -> int:
    header, rows = read_csv(Path(a.run1))
    out = Path(a.phantoms) / "expected_results.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(PRED_COLS)
        for r in rows:
            w.writerow([r[c] for c in PRED_COLS])
    print(f"expected_results.csv written: {out} ({len(rows)} rows); sha256 {pred_sha(Path(a.run1))}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("checks")
    c.add_argument("--run1", required=True)
    c.add_argument("--run2", required=True)
    c.add_argument("--phantoms", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--expected", default=None)
    c.add_argument("--tol", type=float, default=1e-3)
    c.add_argument("--weights-json", default=None)
    c.add_argument("--data-csv", default=None)
    c.add_argument("--data-dir", default=None)
    c.add_argument("--expected-sha", default=None)
    c.add_argument("--timings", default=None)
    p = sub.add_parser("predsha")
    p.add_argument("csv")
    p.add_argument("--without-prob", action="store_true")
    u = sub.add_parser("update-expected")
    u.add_argument("--run1", required=True)
    u.add_argument("--phantoms", required=True)
    a = ap.parse_args()
    if a.cmd == "checks":
        return run_checks(a)
    if a.cmd == "predsha":
        print(pred_sha(Path(a.csv), (TIME_COL, "quality_prob") if a.without_prob else (TIME_COL,)))
        return 0
    return update_expected(a)


if __name__ == "__main__":
    sys.exit(main())
