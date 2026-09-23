#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_transfer_twin.py — прогон-двойник на фантомах: результат инференса не зависит от формы
подачи данных (имена файлов и каталогов, вложенность, регистр расширений, порядок, zip).

Вызывает tools/transfer_check.py на tests/phantoms в режимах rename, zip, shuffle, mixed
с побитовой сверкой нормализованного CSV (--bitwise) и сверкой DICOM SR по исследованиям (--sr).

    python tests/test_transfer_twin.py                   # как скрипт: код 0 — ок, 1 — расхождения
    python -m pytest tests/test_transfer_twin.py -q      # как тест

Тест пропускается, если недоступен torch (инференс контура B требует его; вне образа
densitoai его обычно нет) — по аналогии с другими тестами репозитория, которые выполняются
только внутри образа. Переменные окружения:
    DENSITO_ROOT          корень репозитория (по умолчанию — родитель каталога tests/)
    DENSITO_TWIN_MODES    режимы через запятую (по умолчанию rename,zip,shuffle,mixed)
    DENSITO_SR_STRICT=1   требовать побайтового совпадения SR без сортировки ссылок Evidence
                          (по умолчанию порядок ссылок Evidence допускается разным — он
                          повторяет порядок обхода файлов, см. REPORT идеи 1)
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
TOOL = ROOT / "tools" / "transfer_check.py"
MODES = os.environ.get("DENSITO_TWIN_MODES", "rename,zip,shuffle,mixed")
os.environ.setdefault("OMP_NUM_THREADS", "2")


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def run_twin(workdir: Path) -> dict:
    out_json = workdir / "transfer_twin.json"
    cmd = [sys.executable, str(TOOL), "--input", str(PHANTOMS), "--out", str(out_json),
           "--modes", MODES, "--bitwise", "--sr", "--workdir", str(workdir / "tc"), "--keep"]
    if os.environ.get("DENSITO_SR_STRICT", "") != "1":
        cmd.append("--sr-lenient")
    env = dict(os.environ)
    env["DENSITO_ROOT"] = str(ROOT)
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if not out_json.exists():
        raise AssertionError(f"transfer_check не создал JSON (код {p.returncode}):\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return json.loads(out_json.read_text(encoding="utf-8"))


def check(res: dict) -> list[str]:
    problems: list[str] = []
    if not res.get("modes"):
        return ["ни один режим не выполнен"]
    for mode, d in res["modes"].items():
        if d.get("n_rows_baseline") != d.get("n_rows_twin"):
            problems.append(f"{mode}: строк {d.get('n_rows_baseline')} против {d.get('n_rows_twin')}")
        if d.get("n_matched") != d.get("n_rows_baseline"):
            problems.append(f"{mode}: совпало {d.get('n_matched')} из {d.get('n_rows_baseline')}")
        bit = d.get("bitwise") or {}
        if not bit.get("ok"):
            problems.append(f"{mode}: нормализованный CSV не совпал побитово "
                            f"({bit.get('sha256_baseline_no_time_no_path', '')[:12]} против "
                            f"{bit.get('sha256_twin_no_time_no_path', '')[:12]})")
        sr = d.get("sr") or {}
        if sr and not sr.get("skipped"):
            if sr.get("n_sr_baseline", 0) == 0:
                problems.append(f"{mode}: SR не записаны")
            if not sr.get("ok"):
                problems.append(f"{mode}: SR расходятся: {sr.get('problems', [])[:3]}")
        for p in d.get("discrepancies", [])[:3]:
            problems.append(f"{mode}: {p}")
    return problems


def test_transfer_twin():
    if not torch_available():
        try:
            import pytest
            pytest.skip("torch недоступен — тест выполняется только внутри образа densitoai")
        except ImportError:
            return
    assert PHANTOMS.exists(), f"нет {PHANTOMS}"
    with tempfile.TemporaryDirectory(prefix="densito_twin_") as tmp:
        res = run_twin(Path(tmp))
        problems = check(res)
        assert not problems, "\n".join(problems)


def main() -> int:
    if not torch_available():
        print("SKIP: torch недоступен — прогон-двойник выполняется только внутри образа densitoai")
        return 0
    if not PHANTOMS.exists():
        print(f"SKIP: нет {PHANTOMS}")
        return 0
    with tempfile.TemporaryDirectory(prefix="densito_twin_") as tmp:
        res = run_twin(Path(tmp))
        problems = check(res)
        for mode, d in res["modes"].items():
            bit = d.get("bitwise") or {}
            sr = d.get("sr") or {}
            print(f"{mode:8s} строк {d.get('n_rows_twin')}/{d.get('n_rows_baseline')} совпало {d.get('n_matched')} "
                  f"bitwise={bit.get('ok')} sr={sr.get('ok', '—')} "
                  f"sha256={bit.get('sha256_twin_no_time_no_path', '')[:16]}")
        if problems:
            print("ПРОБЛЕМЫ:\n  " + "\n  ".join(problems))
            return 1
        print("OK: прогон-двойник совпал во всех режимах")
        return 0


if __name__ == "__main__":
    sys.exit(main())
