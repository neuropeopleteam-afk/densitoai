#!/usr/bin/env python3
"""Тесты паспорта производительности (tools/perf_passport.py). Запуск: python tests/test_perf_passport.py (без pytest).

Проверяется (секунды, без долгого замера):
  1. режим (а) на CSV поставки (dataset/regress_2_3_2.csv или DENSITO_REGRESS_CSV): статистики time_of_processing
     совпадают с сохранёнными в docs/perf_passport.json (раздел csv) до 1e-6; два вызова дают одинаковый результат;
     сумма и n совпадают с прямым чтением колонки; по областям n суммируется в общее n;
  2. режим (а) через CLI во временный каталог: JSON и MD пишутся, прежний раздел measure не теряется, MD содержит
     место «Сервер, тихие условия: ЗАПОЛНИТ ОРКЕСТРАТОР», разделы «Как воспроизвести» и «Что это не значит»;
  3. аргументы режима (б): нечётное N, N < 2, без --data, несуществующая папка -> код 2, ничего не пишется;
  4. документ docs/PERFORMANCE.md согласован с docs/perf_passport.json (медиана и p95 CSV присутствуют в тексте),
     запрещённые слова отсутствуют; раздел «Сервер, тихие условия» либо заглушка, либо заполнен из CSV тихого
     прогона (`--quiet-csv`): медиана/p95 в JSON совпадают с распределением CSV, числа присутствуют в MD;
  6. режим --quiet-csv во временный каталог: раздел 2 заполняется, ручные числа (--quiet-extra) сохраняются между
     запусками, неизвестный ключ --quiet-extra -> код 2; если CSV тихого прогона недоступен — пропуск.
  5. выбор файлов для (б) детерминирован и даёт N/2 + N/2 из разных исследований (если датасет доступен).
Если CSV поставки недоступен, тесты 1 и 5 пропускаются с сообщением.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "tools"))

import pandas as pd  # noqa: E402

import perf_passport as pp  # noqa: E402

CSV = Path(os.environ.get("DENSITO_REGRESS_CSV", str(pp.DEFAULT_CSV)))
QUIET_CSV = Path(os.environ.get("DENSITO_REGRESS_CSV_QUIET", str(pp.QUIET_CSV_DEFAULT)))
DATA = Path(os.environ.get("DENSITO_DATA_DIR", "/home/user/workspace/densito/dataset/Исследования"))
JSON_PATH = ROOT / "docs" / "perf_passport.json"
MD_PATH = ROOT / "docs" / "PERFORMANCE.md"
FORBIDDEN = ["Grad" + "-CAM", "автокоррекц" + "ия ROI", "ЕР" + "ИС", "сколи" + "оз", "угол К" + "обба"]


def test_csv_reproduced():
    saved = json.loads(JSON_PATH.read_text(encoding="utf-8"))["csv"]
    a = pp.analyze_csv(CSV)
    b = pp.analyze_csv(CSV)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), "два вызова analyze_csv различаются"
    for key in ("n_rows", "n_success", "n_failure", "n_studies"):
        assert a[key] == saved[key], (key, a[key], saved[key])
    for blk in ("all_rows", "success_rows"):
        for k, v in saved[blk].items():
            assert abs(float(a[blk][k]) - float(v)) <= 1e-6, (blk, k, a[blk][k], v)
    assert set(a["by_region"]) == set(saved["by_region"]), "набор областей"
    for region, d in saved["by_region"].items():
        for k, v in d.items():
            assert abs(float(a["by_region"][region][k]) - float(v)) <= 1e-6, (region, k)
    for q, d in saved["by_run_quarter"].items():
        for k, v in d.items():
            assert abs(float(a["by_run_quarter"][q][k]) - float(v)) <= 1e-6, (q, k)
    # прямое чтение колонки
    df = pd.read_csv(CSV)
    t = pd.to_numeric(df["time_of_processing"], errors="coerce")
    assert a["all_rows"]["n"] == len(df)
    assert abs(a["all_rows"]["sum_s"] - float(t.sum())) <= 1e-3
    assert abs(a["all_rows"]["median_s"] - float(t.median())) <= 1e-4  # округление до 4 знаков
    assert abs(a["all_rows"]["p95_s"] - float(t.quantile(0.95))) <= 1e-4
    assert sum(d["n"] for d in a["by_region"].values()) == a["success_rows"]["n"]
    assert a["all_rows"]["min_s"] <= a["all_rows"]["median_s"] <= a["all_rows"]["p90_s"] <= a["all_rows"]["p95_s"] <= a["all_rows"]["max_s"]
    assert "чужая нагрузка" in a["conditions"] and "4 vCPU" in a["conditions"], "условия боевого прогона должны быть указаны"


def test_cli_mode_a_tmp():
    with tempfile.TemporaryDirectory() as td:
        out_json, out_md = Path(td) / "p.json", Path(td) / "P.md"
        # прежний раздел measure должен сохраниться
        out_json.write_text(json.dumps({"measure": {"marker": 1}}), encoding="utf-8")
        args = ["--csv", str(CSV), "--out-json", str(out_json), "--out-md", str(out_md)]
        if not CSV.exists():
            args = ["--csv", "", "--out-json", str(out_json), "--out-md", str(out_md)]
        rc = pp.main(args)
        assert rc == 0, rc
        p = json.loads(out_json.read_text(encoding="utf-8"))
        assert p["measure"] == {"marker": 1}, "раздел measure потерян при запуске режима (а)"
        assert p["server_quiet"]["status"] == pp.PLACEHOLDER
        md = out_md.read_text(encoding="utf-8")
        assert "Сервер, тихие условия: " + pp.PLACEHOLDER in md
        assert "## 5. Как воспроизвести" in md and "## 6. Что это не значит" in md
        assert "Пропускная способность" in md and "внешними ориентирами" in md
        if CSV.exists():
            assert "csv" in p and p["csv"]["n_rows"] == p["csv"]["n_success"] + p["csv"]["n_failure"]
        # --fresh стирает прежние разделы
        rc = pp.main(args + ["--fresh"])
        assert rc == 0
        p2 = json.loads(out_json.read_text(encoding="utf-8"))
        assert "measure" not in p2


def test_measure_args_rejected():
    with tempfile.TemporaryDirectory() as td:
        out_json, out_md = Path(td) / "p.json", Path(td) / "P.md"
        base = ["--out-json", str(out_json), "--out-md", str(out_md), "--csv", str(CSV)]
        bad = [
            ["--measure", "3", "--data", td],                 # нечётное
            ["--measure", "1", "--data", td],                 # меньше 2
            ["--measure", "12"],                              # без --data
            ["--measure", "12", "--data", str(Path(td) / "нет_такой_папки")],
        ]
        for extra in bad:
            rc = pp.main(base + extra)
            assert rc == 2, (extra, rc)
            assert not out_json.exists() and not out_md.exists(), ("при ошибке аргументов файлы не пишутся", extra)
        # validate_args на корректных аргументах — None (сам замер не запускаем)
        ns = pp.build_parser().parse_args(base + ["--measure", "12", "--data", td])
        assert pp.validate_args(ns) is None
        # через интерпретатор: код возврата 2 и сообщение в stderr
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "perf_passport.py"), "--measure", "5", "--data", td]
                           + base, capture_output=True, text=True, env={**os.environ, "DENSITO_ROOT": str(ROOT)})
        assert r.returncode == 2 and "чётным" in r.stderr, (r.returncode, r.stderr[-300:])


def test_docs_consistent():
    p = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    md = MD_PATH.read_text(encoding="utf-8")
    for w in FORBIDDEN:
        assert w.lower() not in md.lower(), w
    assert pp.render_md(p) == md, "docs/PERFORMANCE.md не совпадает с генерацией из docs/perf_passport.json"
    sq = p["server_quiet"]
    if sq.get("csv"):
        assert "Сервер, тихие условия: " + pp.PLACEHOLDER not in md and pp.PLACEHOLDER not in md
        qa = sq["csv"]["all_rows"]
        assert sq["median_s"] == qa["median_s"] and sq["p95_s"] == qa["p95_s"] and sq["sum_s"] == qa["sum_s"]
        for k in ("median_s", "p95_s", "max_s"):
            assert pp._f(qa[k]) in md, ("тихий сервер", k, pp._f(qa[k]))
        assert "## 2. Боевой сервер, тихие условия" in md
        for region in (pp.REGION_SPINE, pp.REGION_HIP):
            assert region in sq["csv"]["by_region"], region
        ex = sq.get("extra") or {}
        if ex.get("docker_image_gb"):
            assert sq["docker_image_mb"] == round(float(ex["docker_image_gb"]) * 1000)
        if ex.get("batch_processing_wall_s"):
            assert sq["batch_wall_499_s"] == ex["batch_processing_wall_s"]
        if QUIET_CSV.exists() and sq["csv"]["source_name"] == QUIET_CSV.name:
            a = pp.analyze_csv(QUIET_CSV, sq["csv"]["conditions"])
            for blk in ("all_rows", "success_rows"):
                for k, v in sq["csv"][blk].items():
                    assert abs(float(a[blk][k]) - float(v)) <= 1e-6, ("тихий сервер", blk, k)
            for region, d in sq["csv"]["by_region"].items():
                for k, v in d.items():
                    assert abs(float(a["by_region"][region][k]) - float(v)) <= 1e-6, ("тихий сервер", region, k)
    else:
        assert "Сервер, тихие условия: " + pp.PLACEHOLDER in md
    c = p.get("csv")
    if c:
        for k in ("median_s", "p95_s", "max_s"):
            assert pp._f(c["all_rows"][k]) in md, (k, pp._f(c["all_rows"][k]))
        for region in (pp.REGION_SPINE, pp.REGION_HIP):
            assert region in c["by_region"], region
    m = p.get("measure")
    if m:
        assert "песочница" in m["label"] or "ориентировочно" in m["label"] or "vCPU" in m["label"]
        ip = m["inprocess"]
        assert ip["n_files"] == len(m["files"]) and ip["n_failure"] == 0
        assert ip["rss_mb"]["peak_end"] >= ip["rss_mb"]["after_backbones"] >= ip["rss_mb"]["after_models"]
        assert ip["cold_start"]["cold_start_total_s"] > 0
        studies = [f["study"] for f in m["files"]]
        assert len(set(studies)) == len(studies), "файлы замера должны быть из разных исследований"
        assert sum(1 for f in m["files"] if f["region"] == pp.REGION_SPINE) == len(m["files"]) // 2
        # накладные пакета = стена - сумма time_of_processing
        b = m["subprocess"]["batch_one_call"]
        assert abs(b["overhead_s"] - (b["wall_s"] - b["time_of_processing_sum_s"])) <= 1e-2
        assert m["environment"]["cpu_count"] and m["environment"]["python"]
    assert p["server_quiet"]["median_s"] is None or isinstance(p["server_quiet"]["median_s"], (int, float))


def test_quiet_csv_tmp():
    with tempfile.TemporaryDirectory() as td:
        out_json, out_md = Path(td) / "p.json", Path(td) / "P.md"
        base = ["--csv", str(CSV) if CSV.exists() else "", "--out-json", str(out_json), "--out-md", str(out_md)]
        rc = pp.main(base + ["--quiet-csv", str(QUIET_CSV), "--quiet-extra", '{"docker_image_gb": 2.5, "api_batch_files": 4}'])
        assert rc == 0, rc
        p = json.loads(out_json.read_text(encoding="utf-8"))
        sq = p["server_quiet"]
        assert sq["csv"]["source_name"] == QUIET_CSV.name and sq["median_s"] == sq["csv"]["all_rows"]["median_s"]
        assert sq["docker_image_mb"] == 2500 and sq["extra"]["api_batch_files"] == 4
        md = out_md.read_text(encoding="utf-8")
        assert pp.PLACEHOLDER not in md and "## 2. Боевой сервер, тихие условия" in md
        assert pp._f(sq["csv"]["all_rows"]["median_s"]) in md and "2,5" in md
        # повторный запуск без --quiet-*: раздел и ручные числа сохраняются, новые ручные числа дописываются
        rc = pp.main(base + ["--quiet-extra", '{"container_wall_s": 100}'])
        assert rc == 0
        p2 = json.loads(out_json.read_text(encoding="utf-8"))
        assert p2["server_quiet"]["csv"] == sq["csv"]
        assert p2["server_quiet"]["extra"]["docker_image_gb"] == 2.5 and p2["server_quiet"]["extra"]["container_wall_s"] == 100
        # неизвестный ключ и несуществующий CSV -> код 2, файлы не перезаписаны
        before = out_json.read_text(encoding="utf-8")
        assert pp.main(base + ["--quiet-extra", '{"neizvestno": 1}']) == 2
        assert pp.main(base + ["--quiet-csv", str(Path(td) / "нет.csv")]) == 2
        assert out_json.read_text(encoding="utf-8") == before


def test_select_files_deterministic():
    a = pp.select_files(DATA, 12, CSV if CSV.exists() else None)
    b = pp.select_files(DATA, 12, CSV if CSV.exists() else None)
    assert a == b
    assert len(a) == 12 and len({f["study"] for f in a}) == 12
    assert sum(1 for f in a if f["region"] == pp.REGION_SPINE) == 6
    assert sum(1 for f in a if f["region"] == pp.REGION_HIP) == 6
    for f in a:
        assert Path(f["path"]).is_file()


if __name__ == "__main__":
    n_ok = 0
    skipped = []
    tests = [test_cli_mode_a_tmp, test_measure_args_rejected]
    if JSON_PATH.exists():
        tests.append(test_docs_consistent)
    else:  # в образе нет docs/: сверка паспорта с документами выполняется из дерева исходников
        skipped.append(f"test_docs_consistent: нет {JSON_PATH}")
    if not CSV.exists():
        skipped.append(f"test_csv_reproduced: нет {CSV}")
    elif not JSON_PATH.exists():
        skipped.append(f"test_csv_reproduced: нет {JSON_PATH}")
    else:
        tests.insert(0, test_csv_reproduced)
    if QUIET_CSV.exists():
        tests.append(test_quiet_csv_tmp)
    else:
        skipped.append(f"test_quiet_csv_tmp: нет {QUIET_CSV}")
    if DATA.is_dir():
        tests.append(test_select_files_deterministic)
    else:
        skipped.append(f"test_select_files_deterministic: нет {DATA}")
    for t in tests:
        t()
        n_ok += 1
        print(f"OK  {t.__name__}")
    for s in skipped:
        print(f"SKIP {s}")
    print(f"ALL CHECKS PASSED ({n_ok} тестов, {len(skipped)} пропущено)")
