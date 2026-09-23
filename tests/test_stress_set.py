#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_stress_set.py — стресс-набор устойчивости входа на фантомах (проверка 18 в tools/verify.sh).

Вызывает tools/stress_set.py: из tests/phantoms собирается пакет «15 фантомов + stress/<случай>/…»
(обрезанный и нулевой файл, не-DICOM, без PixelData, без Rows/Columns, без PixelSpacing, MONOCHROME1,
16 бит со знаком, 8 бит, RGB, кадр 8×8, огромный кадр, Explicit VR BE, Deflated, RLE, JPEG Baseline,
BitsStored 12, модальность CT, постоянный кадр, zip с битым файлом, вложенность 5, кириллица и пробелы
в именах). Проверяется: инференс завершился кодом 0 без Traceback, ровно одна строка на файл,
9 колонок, статус каждого случая равен ожидаемому (tests/stress/expected_stress.csv), строки Failure
оформлены по правилам сервиса (class 0, violation_type пустой, quality_prob 0.5), корректные нестандартные
кодировки дают тот же класс, что исходный фантом, а 15 строк фантомов в смешанном пакете побитово равны
одиночному прогону.

    python tests/test_stress_set.py                      # как скрипт: код 0 — ок, 1 — расхождения
    python -m pytest tests/test_stress_set.py -q         # как тест

Тест пропускается, если недоступен torch (инференс контура B требует его; вне образа
densitoai его обычно нет). Переменные окружения:
    DENSITO_ROOT           корень репозитория (по умолчанию — родитель каталога tests/)
    DENSITO_STRESS_HUGE    размер огромного кадра, по умолчанию 4000x3000 (для слабых машин — 2500x1500)
    DENSITO_STRESS_BUDGET  лимит времени пакета в секундах, по умолчанию 300
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
PHANTOMS = ROOT / "tests" / "phantoms"
TOOL = ROOT / "tools" / "stress_set.py"
EXPECTED = ROOT / "tests" / "stress" / "expected_stress.csv"
HUGE = os.environ.get("DENSITO_STRESS_HUGE", "4000x3000")
BUDGET = os.environ.get("DENSITO_STRESS_BUDGET", "300")
os.environ.setdefault("OMP_NUM_THREADS", "2")


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def run_stress(workdir: Path) -> dict:
    out_json = workdir / "stress_check.json"
    cmd = [sys.executable, str(TOOL), "--phantoms", str(PHANTOMS), "--out", str(out_json),
           "--workdir", str(workdir / "ss"), "--python", sys.executable, "--expected", str(EXPECTED),
           "--huge", HUGE, "--budget", BUDGET]
    env = dict(os.environ)
    env["DENSITO_ROOT"] = str(ROOT)
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if not out_json.exists():
        raise AssertionError(f"stress_set не создал JSON (код {p.returncode}):\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return json.loads(out_json.read_text(encoding="utf-8"))


def check(res: dict) -> list[str]:
    problems: list[str] = list(res.get("problems", []))
    cases = res.get("cases", [])
    if not cases:
        problems.append("в JSON нет случаев")
    if res.get("n_cases", 0) < 20:
        problems.append(f"случаев {res.get('n_cases')} < 20")
    if res.get("n_expected_failure", 0) < 8:
        problems.append(f"случаев с ожиданием Failure {res.get('n_expected_failure')} < 8")
    if not res.get("baseline_bitwise_ok"):
        problems.append("строки фантомов в смешанном пакете не равны одиночному прогону побитово")
    for c in cases:
        if not c.get("ok"):
            problems.append(f"{c['case']}/{c['file']}: ожидалось {c['expected_status']}, получено "
                            f"{c.get('status') or '—'}: {'; '.join(c.get('problems', []))}")
    if not res.get("ok"):
        problems.append("stress_set: ok=false")
    # убрать дубли, сохранив порядок
    seen: set[str] = set()
    return [p for p in problems if not (p in seen or seen.add(p))]


def test_stress_set() -> None:
    if not torch_available():
        try:
            import pytest
            pytest.skip("torch недоступен — тест выполняется только внутри образа densitoai")
        except ImportError:
            return
    with tempfile.TemporaryDirectory(prefix="densito_stress_") as td:
        res = run_stress(Path(td))
        problems = check(res)
        assert not problems, "\n".join(problems)


def main() -> int:
    if not torch_available():
        print("SKIP: torch недоступен — стресс-набор выполняется только внутри образа densitoai")
        return 0
    with tempfile.TemporaryDirectory(prefix="densito_stress_") as td:
        res = run_stress(Path(td))
        problems = check(res)
        for c in res.get("cases", []):
            mark = "ок" if c.get("ok") else "РАСХОЖДЕНИЕ"
            print(f"  {mark:12s} {c['case']:20s} {c['expected_status']:8s} -> {c.get('status') or '—':8s} "
                  f"{'; '.join(c.get('problems', []))}")
        print(f"случаев {res.get('n_cases_ok')}/{res.get('n_cases')} по ожиданию; строк {res.get('n_rows')}/"
              f"{res.get('n_expected_files')}; побитово с одиночным прогоном: "
              f"{'да' if res.get('baseline_bitwise_ok') else 'нет'}; время {res.get('elapsed_s')} с")
        if problems:
            print("test_stress_set: РАСХОЖДЕНИЯ")
            for p in problems:
                print("  -", p)
            return 1
        print("test_stress_set: OK")
        return 0


if __name__ == "__main__":
    sys.exit(main())
