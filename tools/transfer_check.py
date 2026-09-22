#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transfer_check.py — проверка инвариантности предсказаний к форме подачи данных.

Закрытый набор организаторов приходит без суффиксов `_ПОП/_ППОБ`, в другом порядке и, возможно,
одним архивом. Этот скрипт доказывает, что результат от этого не зависит: он делает копию входа
со случайными именами файлов и каталогов, прогоняет инференс и сверяет предсказания с эталонным
прогоном по ключу (study_uid, image_uid) — совпадать должны anatomical_region, quality_class,
violation_type, quality_prob, processing_status.

  python tools/transfer_check.py --input tests/phantoms --baseline out/run1/results.csv \
      --out out/transfer_check.json [--modes rename,zip] [--workdir /tmp/tc] [--keep]

Код возврата 0 — инвариантность подтверждена, 1 — есть расхождения.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
CMP_COLS = ["anatomical_region", "quality_class", "violation_type", "processing_status"]
PROB_COL = "quality_prob"
SEED = 20260922


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=","))


def make_renamed_copy(src: Path, dst: Path, seed: int = SEED) -> dict:
    """Копия входа: каталоги s000.., файлы f0000 с исходным расширением, порядок перемешан."""
    rng = random.Random(seed)
    files = sorted(p for p in src.rglob("*") if p.is_file())
    order = list(range(len(files)))
    rng.shuffle(order)
    dst.mkdir(parents=True, exist_ok=True)
    dirs = sorted({p.parent.relative_to(src).as_posix() for p in files})
    dir_map = {d: f"s{i:03d}" for i, d in enumerate(sorted(dirs, key=lambda x: rng.random()))}
    mapping = {}
    for new_i, old_i in enumerate(order):
        p = files[old_i]
        rel_dir = p.parent.relative_to(src).as_posix()
        sub = dir_map[rel_dir]
        out_dir = dst / sub if sub != "." else dst
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = p.suffix if p.suffix.lower() not in (".dcm",) else ".dcm"
        new_name = f"f{new_i:04d}{suffix}"
        shutil.copy2(p, out_dir / new_name)
        mapping[(out_dir / new_name).relative_to(dst).as_posix()] = p.relative_to(src).as_posix()
    return mapping


def run_inference(inp: Path, out_csv: Path, python: str) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "2")
    env["PYTHONHASHSEED"] = "0"
    env["DENSITO_ROOT"] = str(ROOT)
    cmd = [python, str(ROOT / "src" / "inference.py"), "--input", str(inp), "--output", str(out_csv)]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write((p.stdout or "")[-2000:] + (p.stderr or "")[-2000:])
    return p.returncode


def compare(base: list[dict], var: list[dict], tol: float) -> dict:
    """Сверка по (study_uid, image_uid); строки со сбоем (пустые UID) сверяются по количеству."""
    def key(r):
        return (r.get("study_uid", ""), r.get("image_uid", ""))

    b_ok = {key(r): r for r in base if key(r) != ("", "")}
    v_ok = {key(r): r for r in var if key(r) != ("", "")}
    b_fail = [r for r in base if r.get("processing_status") != "Success"]
    v_fail = [r for r in var if r.get("processing_status") != "Success"]

    problems: list[str] = []
    if len(base) != len(var):
        problems.append(f"строк {len(base)} против {len(var)}")
    missing = sorted(set(b_ok) - set(v_ok))
    extra = sorted(set(v_ok) - set(b_ok))
    for k in missing[:5]:
        problems.append(f"нет строки с UID {k[1][:24]}…")
    for k in extra[:5]:
        problems.append(f"лишняя строка с UID {k[1][:24]}…")
    if len(b_fail) != len(v_fail):
        problems.append(f"строк со сбоем {len(b_fail)} против {len(v_fail)}")

    diff_cols: dict[str, int] = {}
    max_dprob = 0.0
    n_cmp = 0
    for k in sorted(set(b_ok) & set(v_ok)):
        rb, rv = b_ok[k], v_ok[k]
        n_cmp += 1
        for c in CMP_COLS:
            if (rb.get(c) or "") != (rv.get(c) or ""):
                diff_cols[c] = diff_cols.get(c, 0) + 1
                if len(problems) < 12:
                    problems.append(f"{c}: «{rb.get(c)}» против «{rv.get(c)}» (UID {k[1][-12:]})")
        try:
            d = abs(float(rb.get(PROB_COL, "0") or 0) - float(rv.get(PROB_COL, "0") or 0))
            max_dprob = max(max_dprob, d)
            if d > tol:
                diff_cols[PROB_COL] = diff_cols.get(PROB_COL, 0) + 1
                if len(problems) < 12:
                    problems.append(f"quality_prob Δ={d:.4f} > {tol:g} (UID {k[1][-12:]})")
        except ValueError:
            problems.append(f"quality_prob не число (UID {k[1][-12:]})")
    return {"n_compared": n_cmp, "n_baseline": len(base), "n_variant": len(var),
            "max_abs_dprob": round(max_dprob, 6), "diff_by_column": diff_cols,
            "ok": not problems, "problems": problems}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="каталог с исходными файлами")
    ap.add_argument("--baseline", default="", help="CSV эталонного прогона (если нет — прогоним сами)")
    ap.add_argument("--out", required=True, help="куда писать JSON результата")
    ap.add_argument("--modes", default="rename,zip", help="rename, zip или оба через запятую")
    ap.add_argument("--workdir", default="", help="рабочий каталог (по умолчанию временный)")
    ap.add_argument("--tol", type=float, default=0.0, help="допуск по quality_prob (по умолчанию 0 — бит в бит)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--keep", action="store_true", help="не удалять рабочий каталог")
    a = ap.parse_args()

    src = Path(a.input).resolve()
    work = Path(a.workdir).resolve() if a.workdir else Path(tempfile.mkdtemp(prefix="densito_tc_"))
    work.mkdir(parents=True, exist_ok=True)
    result: dict = {"input": str(src), "modes": {}, "tol": a.tol}

    base_csv = Path(a.baseline).resolve() if a.baseline else work / "baseline" / "results.csv"
    if not a.baseline:
        if run_inference(src, base_csv, a.python) != 0 and not base_csv.exists():
            result["error"] = "эталонный прогон не создал CSV"
            Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return 1
    base = read_rows(base_csv)

    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    renamed = work / "renamed"
    if renamed.exists():
        shutil.rmtree(renamed)
    mapping = make_renamed_copy(src, renamed)
    (work / "mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=1), encoding="utf-8")

    for mode in modes:
        if mode == "rename":
            inp = renamed
        elif mode == "zip":
            zpath = work / "renamed_bundle.zip"
            with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for p in sorted(renamed.rglob("*")):
                    if p.is_file():
                        zf.write(p, p.relative_to(renamed).as_posix())
            inp = zpath
        else:
            result["modes"][mode] = {"ok": False, "problems": [f"неизвестный режим {mode}"]}
            continue
        out_csv = work / f"var_{mode}" / "results.csv"
        rc = run_inference(inp, out_csv, a.python)
        if not out_csv.exists():
            result["modes"][mode] = {"ok": False, "problems": [f"прогон не создал CSV (код {rc})"]}
            continue
        result["modes"][mode] = compare(base, read_rows(out_csv), a.tol)

    result["ok"] = all(v.get("ok") for v in result["modes"].values()) and bool(result["modes"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: (v if k != "modes" else {m: {"ok": d.get("ok"), "n": d.get("n_compared"),
                                                      "diff": d.get("diff_by_column")}
                                                 for m, d in v.items()})
                      for k, v in result.items()}, ensure_ascii=False, indent=2))
    if not a.keep and not a.workdir:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
