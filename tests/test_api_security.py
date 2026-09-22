#!/usr/bin/env python3
"""Негативные тесты доступа и устойчивости входа (ТЗ §2.9 «изоляция запросов», §2.4 «устойчивость»).

Проверяется то, что легко объявить словами и трудно доказать:
  1. результаты запроса не отдаются без кода доступа (job_token), даже если код запроса известен;
  2. список всех запросов сервиса закрыт (без админского ключа — 403);
  3. запрос без файла кода доступа (старый, созданный до введения токенов) закрыт;
  4. файл кода доступа нельзя скачать как файл результата, обход каталога не работает;
  5. ссылки в ответе подписаны кодом доступа (включая вложенные — SR исследования, бонус-файлы);
  6. zip-архив с огромным содержимым («zip-бомба») отклоняется до распаковки,
     как по заявленному размеру и степени сжатия, так и по фактически прочитанным байтам.

Запуск:  python tests/test_api_security.py      (код 0 — все проверки пройдены)
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# API должен собираться на временном каталоге: реальные результаты не трогаем
_TMP_OUT = Path(tempfile.mkdtemp(prefix="densito_sec_out_"))
os.environ["DENSITO_OUTPUT_DIR"] = str(_TMP_OUT)
os.environ.pop("DENSITO_ADMIN_KEY", None)

import inference  # noqa: E402
import api_server as A  # noqa: E402
from fastapi import HTTPException  # noqa: E402

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("  OK   " if cond else "  FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(f"{name}: {detail}")


def expect_http(name: str, code: int, fn, *a, **kw) -> None:
    try:
        fn(*a, **kw)
    except HTTPException as e:
        check(name, e.status_code == code, f"ожидали {code}, получили {e.status_code}: {e.detail}")
    except Exception as e:  # noqa: BLE001
        check(name, False, f"ожидали HTTP {code}, получили {type(e).__name__}: {e}")
    else:
        check(name, False, f"ожидали HTTP {code}, ошибки не было")


def make_job(job: str, with_token: bool = True) -> str:
    d = A.JOBS_DIR / job
    (d / "sr").mkdir(parents=True, exist_ok=True)
    (d / "results.csv").write_text("path_to_study\nx.dcm\n", encoding="utf-8")
    (d / "summary.json").write_text(json.dumps({
        "job_id": job,
        "result_csv_url": f"/api/results/{job}/results.csv",
        "study_sr": {"1.2.3": f"/api/results/{job}/1.2.3_SR.dcm"},
        "rows": [{"bonus_overlay_png_url": f"/api/results/{job}/x_overlay.png"}],
    }, ensure_ascii=False), encoding="utf-8")
    return A._issue_job_token(d) if with_token else ""


def zip_with(entries: list[tuple[str, bytes]], path: Path, level: int = zipfile.ZIP_DEFLATED) -> Path:
    with zipfile.ZipFile(path, "w", level) as zf:
        for name, data in entries:
            zf.writestr(name, data)
    return path


def main() -> int:
    print("== 1. доступ к результатам только по коду доступа ==")
    job = "20260101_120000_abcdef"
    tok = make_job(job)
    check("код доступа выдан и не пуст", bool(tok) and len(tok) >= 20, tok)
    A._check_job_access(job, tok)                                   # не должно бросать
    check("верный код доступа пускает", True)
    expect_http("без кода доступа — 403", 403, A._check_job_access, job, "")
    expect_http("чужой код доступа — 403", 403, A._check_job_access, job, tok[:-1] + ("A" if tok[-1] != "A" else "B"))
    expect_http("несуществующий запрос — 404", 404, A._check_job_access, "20260101_120000_000000", tok)
    expect_http("некорректный код запроса — 404", 404, A._check_job_access, "../../etc", tok)

    print("== 2. запрос без кода доступа (старый) закрыт ==")
    old = "20251231_235959_aaaaaa"
    make_job(old, with_token=False)
    expect_http("старый запрос без токена — 404", 404, A._check_job_access, old, "")

    print("== 3. список всех запросов закрыт ==")
    expect_http("список запросов без админского ключа — 403", 403, A.jobs_list, 50, None, None)
    A.ADMIN_KEY = "secret-admin-key"
    try:
        out = A.jobs_list(50, None, "secret-admin-key")
        check("с админским ключом список отдаётся", isinstance(out, dict) and "jobs" in out, str(out)[:120])
        expect_http("с неверным админским ключом — 403", 403, A.jobs_list, 50, None, "wrong")
        A._check_job_access(job, "secret-admin-key")
        check("админский ключ открывает карточку запроса", True)
    finally:
        A.ADMIN_KEY = ""

    print("== 4. файл кода доступа и обход каталога ==")
    expect_http("файл кода доступа не отдаётся", 404, A._safe_job_file, job, A.JOB_TOKEN_FILE)
    expect_http("обход каталога не работает", 404, A._safe_job_file, job, "../../etc/passwd")
    expect_http("абсолютный путь не работает", 404, A._safe_job_file, job, "/etc/passwd")

    print("== 5. ссылки в ответе подписаны кодом доступа ==")
    card = json.loads((A.JOBS_DIR / job / "summary.json").read_text(encoding="utf-8"))
    signed = A._with_token(card, job, tok)
    urls = [signed["result_csv_url"], signed["study_sr"]["1.2.3"], signed["rows"][0]["bonus_overlay_png_url"]]
    check("все ссылки на файлы запроса получили ?t=", all(u.endswith("?t=" + tok) for u in urls), str(urls))
    check("посторонние ссылки не подписываются",
          A._with_token({"u": "/api/health", "v": "assets/x.png"}, job, tok) == {"u": "/api/health", "v": "assets/x.png"})
    check("код доступа не попадает в сохранённую карточку", "job_token" not in card, str(list(card)))

    print("== 6. пределы распаковки архива ==")
    tmp = Path(tempfile.mkdtemp(prefix="densito_sec_zip_"))
    dest = tmp / "out"
    dest.mkdir()
    bomb = zip_with([("big.bin", b"\0" * (4 << 20))], tmp / "bomb.zip")
    ratio = (4 << 20) / max(1, bomb.stat().st_size)
    try:
        inference.safe_extract_zip(bomb, dest)
        check("архив с высокой степенью сжатия отклонён", False, f"распаковался, степень сжатия ≈ {ratio:.0f}×")
    except ValueError as e:
        check("архив с высокой степенью сжатия отклонён", "небезопасн" in str(e) or "предел" in str(e), str(e)[:160])

    many = zip_with([(f"f{i}.dcm", b"x" * 16) for i in range(5)], tmp / "many.zip")
    keep_members, keep_mb = inference.MAX_ZIP_MEMBERS, inference.MAX_ZIP_UNPACKED_MB
    try:
        inference.MAX_ZIP_MEMBERS = 3
        try:
            inference.safe_extract_zip(many, dest)
            check("архив с числом файлов сверх предела отклонён", False, "распаковался")
        except ValueError as e:
            check("архив с числом файлов сверх предела отклонён", "предел" in str(e), str(e)[:160])
        inference.MAX_ZIP_MEMBERS = keep_members
        inference.MAX_ZIP_UNPACKED_MB = 0.00001          # ~10 байт
        try:
            inference.safe_extract_zip(many, dest)
            check("архив сверх предела распакованного объёма отклонён", False, "распаковался")
        except ValueError as e:
            check("архив сверх предела распакованного объёма отклонён", "объём" in str(e), str(e)[:160])
    finally:
        inference.MAX_ZIP_MEMBERS, inference.MAX_ZIP_UNPACKED_MB = keep_members, keep_mb

    # честный архив по-прежнему распаковывается
    good = zip_with([("a/IM1.dcm", b"y" * 32), ("a/IM1.dcm ", b"z" * 32)], tmp / "good.zip")
    got = inference.safe_extract_zip(good, dest)
    check("нормальный архив распаковывается", len(got) == 2, f"получено {len(got)}")

    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(_TMP_OUT, ignore_errors=True)

    if FAILED:
        print(f"\nНЕ ПРОЙДЕНО: {len(FAILED)}")
        for f in FAILED:
            print("  -", f)
        return 1
    print("\nALL API SECURITY CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
