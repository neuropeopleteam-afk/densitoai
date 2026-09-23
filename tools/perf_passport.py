#!/usr/bin/env python3
"""Паспорт производительности сервиса DensitoAI: секунды на снимок и память, честно и воспроизводимо.

Два источника чисел, оба записываются в docs/perf_passport.json и docs/PERFORMANCE.md:

  (а) Готовый CSV поставки (9 колонок) с боевого прогона — распределение `time_of_processing`
      по файлам: медиана, среднее, p90, p95, максимум, сумма; то же по областям; медианы по
      четвертям порядка прогона (видно, как менялась чужая нагрузка). Ничего не считается заново,
      это разбор колонки, которую сервис записал сам. По умолчанию — `dataset/regress_2_3_2.csv`
      (боевая 2.3.2, 499 файлов; сервер 4 vCPU, Docker, во время прогона была чужая нагрузка).

  (б) Локальный замер (`--measure N --data <папка с исследованиями>`): N файлов датасета
      (половина — позвоночник, половина — бедро, из разных исследований, выбор детерминированный),
      прогон `src/inference.py` тремя способами:
        1. в процессе: холодный старт (импорт, конфиг, модели, прогрев бэкбонов) отдельно от файлов;
           на каждый файл — `time_of_processing` сервиса (то, что попадает в CSV: чтение DICOM →
           область → признаки → эмбеддинги → модели, без записи CSV) и его разложение: чтение DICOM,
           геометрия (контур A), проходы бэкбонов (контур B), остальное; запись CSV/XLSX — отдельно;
        2. пакетный прогон тех же файлов одним вызовом `python src/inference.py` (стена, пиковый RSS);
        3. одиночные вызовы — отдельный процесс на каждый файл (стена, пиковый RSS каждого).
      Фиксируются: пиковый RSS (`resource.getrusage`), потоки (OMP_NUM_THREADS, torch), версии
      python/torch/numpy/sklearn, число vCPU, средняя нагрузка машины (loadavg) до и после.

Что это не значит: числа зависят от машины и от соседней нагрузки; пропускная способность не
обещается; сравнений с внешними ориентирами нет. Числа поставки (метрики, пороги, модели,
config.yaml, 9 колонок CSV) скрипт не трогает и не читает для расчёта.

Запуск:
    python tools/perf_passport.py                       # (а) по CSV по умолчанию -> docs/perf_passport.json, docs/PERFORMANCE.md
    python tools/perf_passport.py --csv out/results.csv # (а) по своему CSV
    OMP_NUM_THREADS=2 nice -n 5 python tools/perf_passport.py --measure 12 --data /path/Исследования
                                                        # (а) + (б); (б) ~1–2 мин на 2 vCPU
    python tools/perf_passport.py --out-json X --out-md Y   # писать в другое место (тест)
Раздел (б) в JSON сохраняется между запусками режима (а): повторный запуск без --measure
переписывает только раздел CSV; `--fresh` очищает всё.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
DEFAULT_CSV = Path(os.environ.get("DENSITO_REGRESS_CSV", "/home/user/workspace/densito/dataset/regress_2_3_2.csv"))
DEFAULT_JSON = ROOT / "docs" / "perf_passport.json"
DEFAULT_MD = ROOT / "docs" / "PERFORMANCE.md"
COLUMNS = ["path_to_study", "study_uid", "image_uid", "anatomical_region", "quality_class",
           "violation_type", "quality_prob", "processing_status", "time_of_processing"]
REGION_SPINE = "Поясничный отдел позвоночника"
REGION_HIP = "Проксимальный отдел бедра"
CSV_CONDITIONS_DEFAULT = ("боевой сервер 2.3.2: 4 vCPU, Docker, регрессия на 499 файлах; во время прогона на сервере "
                          "была чужая нагрузка, поэтому числа завышены и неровные по ходу прогона")
PLACEHOLDER = "ЗАПОЛНИТ ОРКЕСТРАТОР"
QUIET_CSV_DEFAULT = Path(os.environ.get("DENSITO_REGRESS_CSV_QUIET", "/home/user/workspace/densito/dataset/regress_2_4_0.csv"))
QUIET_CONDITIONS_DEFAULT = ("боевой сервер, тихие условия (loadavg около 0,5 перед запуском): 4 vCPU Intel Xeon E5-2680 v2 2,80 GHz, "
                            "5,9 ГБ RAM, Docker, образ densitoai:2.4.0, один процесс src/inference.py, --network none, "
                            "OMP_NUM_THREADS=2, nice -n 10; регрессия 2.4.0 на 499 файлах 23.09.2026")
# Ключи ручных чисел тихого сервера (--quiet-extra): подпись → (ключ JSON, единицы)
QUIET_EXTRA_FIELDS = [
    ("batch_processing_wall_s", "Суммарное время обработки 499 файлов одним процессом, с"),
    ("container_wall_s", "Стена контейнера пакета 499 файлов (с загрузкой моделей), с"),
    ("api_batch_files", "Живой API: файлов в партии через /api/analyze"),
    ("api_batch_model_s", "Живой API: время модели на партию, с"),
    ("api_batch_wall_s", "Живой API: стена партии с бонусными выходами (SC, SR, SEG, ROI), с"),
    ("api_container_idle_rss_mib", "RSS контейнера API в простое, МиБ"),
    ("docker_image_gb", "Размер образа Docker (`docker image ls`), ГБ"),
]


# --------------------------------------------------------------------------- #
# (а) распределение time_of_processing из готового CSV
# --------------------------------------------------------------------------- #
def _dist(t: pd.Series) -> Dict[str, Any]:
    t = pd.to_numeric(t, errors="coerce").dropna().astype(float)
    if len(t) == 0:
        return {"n": 0}
    return {
        "n": int(len(t)),
        "median_s": round(float(t.median()), 4),
        "mean_s": round(float(t.mean()), 4),
        "p90_s": round(float(t.quantile(0.90)), 4),
        "p95_s": round(float(t.quantile(0.95)), 4),
        "min_s": round(float(t.min()), 4),
        "max_s": round(float(t.max()), 4),
        "sum_s": round(float(t.sum()), 4),
    }


def analyze_csv(csv_path: Path, conditions: str = CSV_CONDITIONS_DEFAULT) -> Dict[str, Any]:
    """Распределение time_of_processing по строкам CSV поставки (только чтение колонки, без пересчёта)."""
    df = pd.read_csv(csv_path)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV без колонок поставки {missing}: {csv_path}")
    df["time_of_processing"] = pd.to_numeric(df["time_of_processing"], errors="coerce")
    ok = df[df["processing_status"] == "Success"]
    out: Dict[str, Any] = {
        "source": str(csv_path),
        "source_name": csv_path.name,
        "conditions": conditions,
        "n_rows": int(len(df)),
        "n_success": int(len(ok)),
        "n_failure": int(len(df) - len(ok)),
        "n_studies": int(df["study_uid"].nunique()),
        "all_rows": _dist(df["time_of_processing"]),
        "success_rows": _dist(ok["time_of_processing"]),
        "by_region": {},
        "by_run_quarter": {},
        "note": ("time_of_processing — секунды на файл, которые сервис записал сам: чтение DICOM → область → "
                 "признаки → эмбеддинги → модели; без загрузки моделей и без записи CSV. Failure-строки имеют "
                 "время до сбоя."),
    }
    for region, g in ok.groupby("anatomical_region", sort=True):
        out["by_region"][str(region)] = _dist(g["time_of_processing"])
    # четверти порядка прогона: чужая нагрузка на сервере видна как разница медиан между четвертями
    if len(ok) >= 8:
        q = pd.qcut(np.arange(len(ok)), 4, labels=False)
        for k in range(4):
            g = ok[q == k]
            row = {"n": int(len(g)), "median_all_s": round(float(g["time_of_processing"].median()), 4)}
            for region, gg in g.groupby("anatomical_region"):
                row[f"median_{'spine' if region == REGION_SPINE else 'hip'}_s"] = round(float(gg["time_of_processing"].median()), 4)
            out["by_run_quarter"][f"q{k + 1}"] = row
    return out


# --------------------------------------------------------------------------- #
# (б) локальный замер
# --------------------------------------------------------------------------- #
def _rss_mb_self() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)  # Linux: КБ -> МБ


def _loadavg() -> List[float]:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except OSError:
        return []


def _env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "nice": os.nice(0),
        "in_docker": Path("/.dockerenv").exists(),
    }
    try:
        import torch  # noqa: F401
        info["torch"] = torch.__version__
        info["torch_num_threads_default"] = torch.get_num_threads()
    except Exception as e:  # noqa: BLE001
        info["torch"] = f"недоступен: {e}"
    for mod in ("numpy", "sklearn", "pydicom"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            info[mod] = None
    try:
        info["cpu_model"] = next((ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
                                  if ln.lower().startswith("model name")), None)
    except Exception:  # noqa: BLE001
        info["cpu_model"] = None
    try:
        mem = Path("/proc/meminfo").read_text().splitlines()[0].split()
        info["mem_total_gb"] = round(int(mem[1]) / 1024 / 1024, 1)
    except Exception:  # noqa: BLE001
        info["mem_total_gb"] = None
    return info


def _read_cols(path: Path) -> Optional[int]:
    try:
        import pydicom
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        return int(getattr(ds, "Columns", 0) or 0) or None
    except Exception:  # noqa: BLE001
        return None


def select_files(data_dir: Path, n: int, csv_path: Optional[Path], spine_min_cols: int = 300) -> List[Dict[str, Any]]:
    """N файлов: ceil(N/2) позвоночник + floor(N/2) бедро, по одному из исследования, детерминированно
    (исследования — в лексикографическом порядке, внутри — первый подходящий файл).
    Область: из CSV поставки, если файл там есть; иначе по ширине кадра (>= spine_min_cols — позвоночник)."""
    region_by_rel: Dict[str, str] = {}
    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path)
        for p, r in zip(df["path_to_study"], df["anatomical_region"]):
            region_by_rel[str(p).replace("\\", "/")] = str(r)
    studies = sorted(d for d in data_dir.iterdir() if d.is_dir())
    want = {"spine": (n + 1) // 2, "hip": n // 2}
    got: Dict[str, List[Dict[str, Any]]] = {"spine": [], "hip": []}
    for kind in ("spine", "hip"):
        for st in studies:
            if len(got[kind]) >= want[kind]:
                break
            if any(x["study"] == st.name for x in got["spine"] + got["hip"]):
                continue  # исследование уже занято другой областью — берём разные исследования
            for f in sorted(p for p in st.rglob("*") if p.is_file()):
                rel = f.relative_to(data_dir).as_posix()
                region = region_by_rel.get(f"{data_dir.name}/{rel}") or region_by_rel.get(rel)
                if region is None:
                    cols = _read_cols(f)
                    if cols is None:
                        continue
                    region = REGION_SPINE if cols >= spine_min_cols else REGION_HIP
                k = "spine" if region == REGION_SPINE else "hip"
                if k == kind:
                    got[kind].append({"path": str(f), "rel": rel, "study": st.name, "region": region})
                    break
    files = got["spine"] + got["hip"]
    if len(files) < n:
        raise SystemExit(f"нашлось только {len(files)} файлов из {n} в {data_dir}")
    return files


def measure_inprocess(files: List[Dict[str, Any]], data_dir: Path, xlsx: bool = True, passes: int = 3) -> Dict[str, Any]:
    """Холодный старт и разложение времени на файл внутри одного процесса (тот же код, что в сервисе).
    Файлы проходятся `passes` раз подряд; статистики — по лучшему проходу (наименьшая медиана): на машине
    с соседней нагрузкой это ближе к цене самого снимка, остальные проходы показаны как разброс."""
    sys.path.insert(0, str(ROOT / "src"))
    load: Dict[str, float] = {}
    rss0 = _rss_mb_self()
    t = time.perf_counter()
    import inference  # noqa: E402  — импорт torch/sklearn/pydicom входит в холодный старт
    load["import_s"] = round(time.perf_counter() - t, 3)
    t = time.perf_counter()
    cfg = inference.load_config()
    engine = inference.DensitoInference(cfg=cfg)
    load["models_s"] = round(time.perf_counter() - t, 3)
    rss_models = _rss_mb_self()
    t = time.perf_counter()
    sources = sorted({str(mb.meta.get("emb_source", inference.EmbeddingExtractor.DEFAULT_SOURCE))
                      for mb in list(engine.registry.emb.values()) + list(engine.registry.any_emb.values())})
    for s in sources:
        engine.embedder._init(s)
    load["backbones_warmup_s"] = round(time.perf_counter() - t, 3)
    load["backbones"] = sources
    load["cold_start_total_s"] = round(load["import_s"] + load["models_s"] + load["backbones_warmup_s"], 3)
    rss_ready = _rss_mb_self()
    try:
        import torch
        torch_threads = torch.get_num_threads()
    except Exception:  # noqa: BLE001
        torch_threads = None

    # разложение времени: оборачиваем функции, которые process_file вызывает по имени модуля / через embedder
    acc: Dict[str, float] = {"read_s": 0.0, "geom_s": 0.0, "emb_s": 0.0}
    orig_read, orig_geom = inference.read_and_validate, inference.extract_geometry
    orig_extract = engine.embedder.extract

    def timed_read(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig_read(*a, **k)
        finally:
            acc["read_s"] += time.perf_counter() - t0

    def timed_geom(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig_geom(*a, **k)
        finally:
            acc["geom_s"] += time.perf_counter() - t0

    def timed_extract(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig_extract(*a, **k)
        finally:
            acc["emb_s"] += time.perf_counter() - t0

    inference.read_and_validate, inference.extract_geometry = timed_read, timed_geom
    engine.embedder.extract = timed_extract
    all_passes: List[List[Dict[str, Any]]] = []
    rows: List[Dict[str, Any]] = []
    try:
        for ps in range(max(1, passes)):
            cur: List[Dict[str, Any]] = []
            for i, f in enumerate(files):
                for k in acc:
                    acc[k] = 0.0
                row, dbg = engine.process_file(Path(f["path"]), data_dir)
                if ps == 0:
                    rows.append(row)
                tot = float(row["time_of_processing"])
                cur.append({
                    "pass": ps + 1, "idx": i, "rel": f["rel"], "region": row["anatomical_region"],
                    "status": row["processing_status"],
                    "time_of_processing_s": tot,
                    "read_dicom_s": round(acc["read_s"], 4),
                    "geometry_s": round(acc["geom_s"], 4),
                    "embeddings_s": round(acc["emb_s"], 4),
                    "other_s": round(max(0.0, tot - acc["read_s"] - acc["geom_s"] - acc["emb_s"]), 4),
                    "first_file": ps == 0 and i == 0,
                })
            all_passes.append(cur)
    finally:
        inference.read_and_validate, inference.extract_geometry = orig_read, orig_geom
        engine.embedder.extract = orig_extract
    # запись CSV (+XLSX) — вне time_of_processing, как в сервисе
    tmp = Path(tempfile.mkdtemp(prefix="densito_perf_"))
    try:
        t = time.perf_counter()
        inference.write_results(rows, tmp / "results.csv", cfg, xlsx=False)
        write_csv_s = round(time.perf_counter() - t, 4)
        write_xlsx_s = None
        if xlsx:
            t = time.perf_counter()
            inference.write_results(rows, tmp / "results_x.csv", cfg, xlsx=True)
            write_xlsx_s = round(time.perf_counter() - t - write_csv_s, 4)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    rss_end = _rss_mb_self()

    def agg(items: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
        return _dist(pd.Series([x[key] for x in items]))

    pass_stats = [{"pass": k + 1, **agg(pf, "time_of_processing_s")} for k, pf in enumerate(all_passes)]
    best = min(range(len(all_passes)), key=lambda k: pass_stats[k]["median_s"])
    per_file = all_passes[best]
    first_pass = all_passes[0]

    by_region: Dict[str, Any] = {}
    for region in sorted({p["region"] for p in per_file}):
        items = [p for p in per_file if p["region"] == region]
        by_region[region] = {
            "n": len(items),
            "time_of_processing": agg(items, "time_of_processing_s"),
            "read_dicom_median_s": agg(items, "read_dicom_s").get("median_s"),
            "geometry_median_s": agg(items, "geometry_s").get("median_s"),
            "embeddings_median_s": agg(items, "embeddings_s").get("median_s"),
            "other_median_s": agg(items, "other_s").get("median_s"),
        }
    return {
        "cold_start": load,
        "rss_mb": {"after_import_baseline": rss0, "after_models": rss_models, "after_backbones": rss_ready,
                   "peak_end": rss_end},
        "torch_num_threads_effective": torch_threads,
        "n_files": len(per_file),
        "passes": len(all_passes),
        "best_pass": best + 1,
        "pass_stats": pass_stats,
        "n_failure": sum(1 for p in per_file if p["status"] != "Success"),
        "all": agg(per_file, "time_of_processing_s"),
        "first_pass_all": agg(first_pass, "time_of_processing_s"),
        "first_pass_excluding_first_file": agg(first_pass[1:], "time_of_processing_s") if len(first_pass) > 1 else None,
        "first_file_first_pass_s": first_pass[0]["time_of_processing_s"],
        "by_region": by_region,
        "breakdown_sum_s": {k: round(sum(p[k] for p in per_file), 4)
                            for k in ("read_dicom_s", "geometry_s", "embeddings_s", "other_s", "time_of_processing_s")},
        "write_csv_s": write_csv_s,
        "write_xlsx_extra_s": write_xlsx_s,
        "per_file": per_file,
    }


_CHILD = r"""
import json, os, resource, sys, time
sys.path.insert(0, os.path.join(os.environ["DENSITO_ROOT"], "src"))
t0 = time.perf_counter()
import inference
t_import = time.perf_counter() - t0
rc = inference.main(sys.argv[2:])
wall = time.perf_counter() - t0
ru = resource.getrusage(resource.RUSAGE_SELF)
json.dump({"rc": rc, "wall_s": round(wall, 3), "import_s": round(t_import, 3), "maxrss_mb": round(ru.ru_maxrss / 1024, 1),
           "user_s": round(ru.ru_utime, 2), "sys_s": round(ru.ru_stime, 2)}, open(sys.argv[1], "w"))
"""


def _run_child(args: List[str], env: Dict[str, str]) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        out = fh.name
    try:
        t = time.perf_counter()
        p = subprocess.run([sys.executable, "-c", _CHILD, out] + args, env=env, capture_output=True, text=True, timeout=1800)
        wall_outer = round(time.perf_counter() - t, 3)
        res = json.loads(Path(out).read_text()) if Path(out).exists() and Path(out).stat().st_size else {"rc": p.returncode}
        res["wall_outer_s"] = wall_outer
        if p.returncode != 0:
            res["stderr_tail"] = p.stderr[-800:]
        return res
    finally:
        Path(out).unlink(missing_ok=True)


def measure_subprocess(files: List[Dict[str, Any]], data_dir: Path) -> Dict[str, Any]:
    """Пакет одним вызовом и одиночные вызовы по одному файлу (старт процесса на файл)."""
    env = dict(os.environ)
    env["DENSITO_ROOT"] = str(ROOT)
    env.setdefault("PYTHONPATH", str(ROOT / "src"))
    work = Path(tempfile.mkdtemp(prefix="densito_perf_pkg_"))
    try:
        batch_in = work / "input"
        for f in files:  # копии в структуре исследований, чтобы один вызов увидел все N файлов
            dst = batch_in / f["rel"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f["path"], dst)
        batch = _run_child(["--input", str(batch_in), "--output", str(work / "batch.csv")], env)
        batch_rows = pd.read_csv(work / "batch.csv") if (work / "batch.csv").exists() else pd.DataFrame()
        batch["n_rows"] = int(len(batch_rows))
        if len(batch_rows):
            batch["time_of_processing_sum_s"] = round(float(batch_rows["time_of_processing"].sum()), 4)
            batch["overhead_s"] = round(batch["wall_s"] - batch["time_of_processing_sum_s"], 3)  # старт + модели + запись
            batch["wall_per_file_s"] = round(batch["wall_s"] / len(batch_rows), 3)
        singles: List[Dict[str, Any]] = []
        for i, f in enumerate(files):
            r = _run_child(["--input", f["path"], "--output", str(work / f"single_{i}.csv")], env)
            r["rel"], r["region"] = f["rel"], f["region"]
            csv_i = work / f"single_{i}.csv"
            if csv_i.exists():
                d = pd.read_csv(csv_i)
                r["time_of_processing_s"] = round(float(d["time_of_processing"].iloc[0]), 4) if len(d) else None
            singles.append(r)
        walls = pd.Series([s["wall_s"] for s in singles if "wall_s" in s])
        return {
            "batch_one_call": batch,
            "single_calls": {
                "n": len(singles),
                "wall": _dist(walls),
                "maxrss_mb_max": max((s.get("maxrss_mb", 0) for s in singles), default=None),
                "time_of_processing": _dist(pd.Series([s.get("time_of_processing_s") for s in singles])),
                "per_file": singles,
            },
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_measure(n: int, data_dir: Path, csv_path: Optional[Path], label: str, passes: int = 3) -> Dict[str, Any]:
    files = select_files(data_dir, n, csv_path)
    load_before = _loadavg()
    t_all = time.perf_counter()
    inproc = measure_inprocess(files, data_dir, passes=passes)   # до _env_info: torch импортируется внутри замера холодного старта
    sub = measure_subprocess(files, data_dir)
    # импорт в чистом процессе (пакетный вызов): в этом процессе numpy/pandas уже были загружены
    inproc["cold_start"]["import_clean_process_s"] = sub["batch_one_call"].get("import_s")
    env_info = _env_info()
    return {
        "label": label,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": str(data_dir),
        "files": [{"rel": f["rel"], "study": f["study"], "region": f["region"]} for f in files],
        "environment": env_info,
        "loadavg_before": load_before,
        "loadavg_after": _loadavg(),
        "inprocess": inproc,
        "subprocess": sub,
        "total_measure_wall_s": round(time.perf_counter() - t_all, 1),
    }


# --------------------------------------------------------------------------- #
# Документ
# --------------------------------------------------------------------------- #
def _f(x: Any, nd: int = 2) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        if 0 < abs(x) < 0.5 * 10 ** (-nd):
            return f"< 0,{'0' * (nd - 1)}1"
        return f"{x:.{nd}f}".replace(".", ",")
    return str(x)


def _dist_row(name: str, d: Dict[str, Any], cond: str) -> str:
    if not d or d.get("n", 0) == 0:
        return f"| {name} | 0 | — | — | — | — | — | — | {cond} |"
    return (f"| {name} | {d['n']} | {_f(d['median_s'])} | {_f(d['mean_s'])} | {_f(d['p90_s'])} | {_f(d['p95_s'])} "
            f"| {_f(d['max_s'])} | {_f(d['sum_s'], 1)} | {cond} |")


DIST_HEADER = ("| Набор | n | медиана, с | среднее, с | p90, с | p95, с | максимум, с | всего, с | Условия |\n"
               "|---|---|---|---|---|---|---|---|---|")


def render_md(p: Dict[str, Any]) -> str:
    L: List[str] = []
    L.append("# Паспорт производительности DensitoAI 2.4.0")
    L.append("")
    L.append("Секунды на снимок и память сервиса — что измерено, в каких условиях и как повторить. Все числа")
    L.append("ниже получены скриптом `tools/perf_passport.py` и лежат в `docs/perf_passport.json`; документ")
    L.append("сгенерирован из него. Числа поставки (метрики, пороги, модели, `config.yaml`, 9 колонок CSV) не менялись.")
    L.append("")
    L.append("Что измеряется. `time_of_processing` — секунды на файл, которые сервис записывает в CSV сам")
    L.append("(`src/inference.py: process_file`): чтение DICOM → область → признаки контура A → эмбеддинги")
    L.append("контура B → модели → строка. В него не входят загрузка моделей и прогрев бэкбонов (один раз на")
    L.append("процесс) и запись CSV/XLSX (один раз на пакет). Стена пакета = холодный старт + сумма")
    L.append("`time_of_processing` + запись.")
    L.append("")
    # --- (а)
    c = p.get("csv")
    L.append("## 1. Боевой прогон 2.3.2 по CSV поставки (источник а)")
    L.append("")
    if c:
        L.append(f"Источник: `{c['source_name']}` — {c['n_rows']} строк, {c['n_studies']} исследований, "
                 f"Success {c['n_success']}, Failure {c['n_failure']}. Условия: {c['conditions']}.")
        L.append("")
        L.append(DIST_HEADER)
        cond = "сервер, чужая нагрузка"
        L.append(_dist_row("Все файлы", c["all_rows"], cond))
        for region, d in c["by_region"].items():
            L.append(_dist_row(region, d, cond))
        L.append("")
        if c.get("by_run_quarter"):
            L.append("Медиана по четвертям порядка прогона (нагрузка на сервере менялась по ходу регрессии):")
            L.append("")
            L.append("| Четверть прогона | n | медиана всех, с | медиана позвоночник, с | медиана бедро, с |")
            L.append("|---|---|---|---|---|")
            for k, r in c["by_run_quarter"].items():
                L.append(f"| {k} | {r['n']} | {_f(r['median_all_s'])} | {_f(r.get('median_spine_s'))} | {_f(r.get('median_hip_s'))} |")
            L.append("")
        L.append("Разброс между четвертями в несколько раз при одинаковом коде и одинаковых файлах — это след")
        L.append("чужой нагрузки, а не свойство сервиса. Числа этого раздела — верхняя, «грязная» оценка; их")
        L.append("нельзя читать как скорость сервиса в спокойных условиях.")
    else:
        L.append("Раздел не заполнен: CSV поставки не найден при генерации.")
    L.append("")
    sq = p.get("server_quiet") or {}
    qc = sq.get("csv")
    if qc:
        L.append("## 2. Боевой сервер, тихие условия (регрессия 2.4.0 по CSV)")
        L.append("")
        L.append(f"Источник: `{qc['source_name']}` — {qc['n_rows']} строк, {qc['n_studies']} исследований, "
                 f"Success {qc['n_success']}, Failure {qc['n_failure']}. Условия: {qc['conditions']}.")
        L.append("")
        L.append(DIST_HEADER)
        cond = "сервер, тихие условия"
        L.append(_dist_row("Все файлы", qc["all_rows"], cond))
        for region, d in qc["by_region"].items():
            L.append(_dist_row(region, d, cond))
        L.append("")
        if qc.get("by_run_quarter"):
            L.append("Медиана по четвертям порядка прогона (ровный ход — признак отсутствия чужой нагрузки):")
            L.append("")
            L.append("| Четверть прогона | n | медиана всех, с | медиана позвоночник, с | медиана бедро, с |")
            L.append("|---|---|---|---|---|")
            for k, r in qc["by_run_quarter"].items():
                L.append(f"| {k} | {r['n']} | {_f(r['median_all_s'])} | {_f(r.get('median_spine_s'))} | {_f(r.get('median_hip_s'))} |")
            L.append("")
        ex = sq.get("extra") or {}
        if ex:
            L.append("Замеры вне CSV (сняты на том же сервере в тех же условиях, вручную, `--quiet-extra`):")
            L.append("")
            L.append("| Показатель | Значение | Условия |")
            L.append("|---|---|---|")
            for key, title in QUIET_EXTRA_FIELDS:
                if key in ex and ex[key] is not None:
                    v = ex[key]
                    if isinstance(v, bool) or not isinstance(v, (int, float)):
                        val = str(v)
                    elif float(v) == int(v):
                        val = str(int(v))
                    else:
                        val = f"{float(v):.2f}".rstrip("0").rstrip(".").replace(".", ",")
                    if key == "container_wall_s" and ex.get("container_wall_note"):
                        val += f" ({ex['container_wall_note']})"
                    if key == "api_batch_model_s" and ex.get("api_batch_files"):
                        val += f" ({_f(float(v) / float(ex['api_batch_files']), 2)} с на файл)"
                    cnd = "контейнер densitoai-api-container, те же условия" if key.startswith("api_") else (
                        "образ densitoai:2.4.0" if key == "docker_image_gb" else "то же")
                    L.append(f"| {title} | {val} | {cnd} |")
            L.append("")
            if ex.get("note"):
                L.append(str(ex["note"]))
                L.append("")
        L.append("Числа этого раздела — рабочая оценка скорости сервиса на боевом сервере без соседней нагрузки. "
                 "Пиковая память контейнера во время пакета отдельно не измерялась; пиковый RSS процесса — раздел 3.")
        L.append("")
    else:
        L.append("## 2. Сервер, тихие условия: " + PLACEHOLDER)
        L.append("")
        L.append("| Показатель | Значение | Условия |")
        L.append("|---|---|---|")
        L.append(f"| Медиана / p95 `time_of_processing`, с | {PLACEHOLDER} | боевой сервер 4 vCPU, Docker, без чужой нагрузки, 499 файлов |")
        L.append(f"| Стена пакета 499 файлов, с | {PLACEHOLDER} | то же |")
        L.append(f"| Пиковый RSS контейнера, МБ | {PLACEHOLDER} | то же |")
        L.append(f"| Размер образа Docker (`docker image ls`), МБ | {PLACEHOLDER} | образ 2.4.0 |")
        L.append("")
        L.append("Числа этого раздела снимаются на боевом сервере после окончания регрессии и сборки; до заполнения")
        L.append("считать неизвестными, не подставлять локальные. Заполнение: `python tools/perf_passport.py --quiet-csv <CSV регрессии в тихих условиях>`.")
        L.append("")
    # --- (б)
    m = p.get("measure")
    if m and "inprocess" not in m:
        m = None  # неполный или чужой раздел — не рисуем
    L.append("## 3. Локальный замер (источник б)")
    L.append("")
    if m:
        e = m["environment"]
        ip = m["inprocess"]
        sp = m["subprocess"]
        L.append(f"Условия: {m['label']}. Машина: {e.get('cpu_count')} vCPU"
                 f"{' (' + str(e.get('cpu_model')) + ')' if e.get('cpu_model') else ''}, "
                 f"{_f(e.get('mem_total_gb'), 1)} ГБ; Python {e.get('python')}, torch {e.get('torch')}, numpy {e.get('numpy')}, "
                 f"scikit-learn {e.get('sklearn')}, pydicom {e.get('pydicom')}; OMP_NUM_THREADS={e.get('OMP_NUM_THREADS')}, "
                 f"потоков torch фактически {ip.get('torch_num_threads_effective')} "
                 f"(`inference.py` ставит `min(8, cpu_count)`), nice {e.get('nice')}, Docker: {'да' if e.get('in_docker') else 'нет'}. "
                 f"Средняя нагрузка машины (loadavg 1/5/15 мин) до замера {m['loadavg_before']}, после {m['loadavg_after']}. "
                 f"Дата {m['timestamp']}.")
        L.append("")
        L.append(f"Файлы: {ip['n_files']} из `{Path(m['data_dir']).name}/` — "
                 f"{sum(1 for f in m['files'] if f['region'] == REGION_SPINE)} позвоночник, "
                 f"{sum(1 for f in m['files'] if f['region'] == REGION_HIP)} бедро, каждый из своего исследования "
                 f"(список — в JSON, `measure.files`). Failure: {ip['n_failure']}.")
        L.append("")
        L.append("### 3.1. Холодный старт и память (один процесс)")
        L.append("")
        cs, rss = ip["cold_start"], ip["rss_mb"]
        L.append("| Этап | Время, с | RSS после этапа, МБ |")
        L.append("|---|---|---|")
        L.append(f"| Импорт (torch, sklearn, pydicom, код сервиса) | {_f(cs.get('import_clean_process_s'))} в чистом процессе; "
                 f"{_f(cs['import_s'])} здесь (numpy/pandas уже загружены) | {_f(rss['after_import_baseline'], 0)} (до импорта, с numpy/pandas) |")
        L.append(f"| Загрузка моделей (`DensitoInference`) | {_f(cs['models_s'])} | {_f(rss['after_models'], 0)} |")
        L.append(f"| Прогрев бэкбонов ({', '.join(cs['backbones'])}) | {_f(cs['backbones_warmup_s'])} | {_f(rss['after_backbones'], 0)} |")
        L.append(f"| Холодный старт всего (в процессе) | {_f(cs['cold_start_total_s'])} | — |")
        L.append(f"| Пиковый RSS процесса после {ip['n_files']} файлов и записи CSV/XLSX | — | {_f(rss['peak_end'], 0)} |")
        L.append("")
        L.append("### 3.2. Время на файл в процессе (то, что попадает в `time_of_processing`)")
        L.append("")
        L.append(f"Файлы пройдены {ip.get('passes', 1)} раза подряд в одном процессе; таблица — по лучшему проходу "
                 f"(№ {ip.get('best_pass', 1)}, наименьшая медиана), остальные проходы — ниже как разброс от соседней нагрузки.")
        L.append("")
        L.append(DIST_HEADER)
        cond = m["label"]
        L.append(_dist_row("Все файлы (лучший проход)", ip["all"], cond))
        for region, d in ip["by_region"].items():
            L.append(_dist_row(region, d["time_of_processing"], cond))
        if ip.get("first_pass_all") and ip.get("best_pass", 1) != 1:
            L.append(_dist_row("Первый проход, все файлы", ip["first_pass_all"], cond + "; первый файл после прогрева"))
        L.append("")
        if ip.get("pass_stats"):
            L.append("| Проход | медиана, с | p95, с | максимум, с |")
            L.append("|---|---|---|---|")
            for st in ip["pass_stats"]:
                L.append(f"| {st['pass']} | {_f(st['median_s'])} | {_f(st['p95_s'])} | {_f(st['max_s'])} |")
            L.append("")
            L.append(f"Первый файл первого прохода (сразу после прогрева бэкбонов): {_f(ip.get('first_file_first_pass_s'))} с.")
            L.append("")
        L.append("Разложение медианного файла по этапам (медианы по файлам области, с):")
        L.append("")
        L.append("| Область | чтение DICOM | геометрия (контур A) | бэкбоны (контур B) | остальное (модели, стэкинг, строка) |")
        L.append("|---|---|---|---|---|")
        for region, d in ip["by_region"].items():
            L.append(f"| {region} | {_f(d['read_dicom_median_s'], 3)} | {_f(d['geometry_median_s'], 3)} "
                     f"| {_f(d['embeddings_median_s'], 3)} | {_f(d['other_median_s'], 3)} |")
        bs = ip["breakdown_sum_s"]
        L.append("")
        L.append(f"Сумма по {ip['n_files']} файлам лучшего прохода: `time_of_processing` {_f(bs['time_of_processing_s'])} с, из них чтение "
                 f"{_f(bs['read_dicom_s'])}, геометрия {_f(bs['geometry_s'])}, бэкбоны {_f(bs['embeddings_s'])}, остальное "
                 f"{_f(bs['other_s'])}. Запись CSV {ip['n_files']} строк: {_f(ip['write_csv_s'], 3)} с; дополнительно XLSX: "
                 f"{_f(ip['write_xlsx_extra_s'], 3)} с (вне `time_of_processing`).")
        L.append("")
        L.append("### 3.3. Пакет одним вызовом и одиночные вызовы (отдельный процесс на файл)")
        L.append("")
        b, s = sp["batch_one_call"], sp["single_calls"]
        L.append("| Режим | Стена, с | из них `time_of_processing`, с | накладные (старт, модели, запись), с | стена на файл, с | пиковый RSS, МБ |")
        L.append("|---|---|---|---|---|---|")
        L.append(f"| `python src/inference.py --input <папка {b.get('n_rows')} файлов>` | {_f(b.get('wall_s'))} | "
                 f"{_f(b.get('time_of_processing_sum_s'))} | {_f(b.get('overhead_s'))} | {_f(b.get('wall_per_file_s'), 3)} | {_f(b.get('maxrss_mb'), 0)} |")
        w = s["wall"]
        L.append(f"| один файл на процесс, {s['n']} вызовов: медиана | {_f(w.get('median_s'))} | {_f(s['time_of_processing'].get('median_s'))} | "
                 f"{_f((w.get('median_s') or 0) - (s['time_of_processing'].get('median_s') or 0))} | {_f(w.get('median_s'))} | {_f(s.get('maxrss_mb_max'), 0)} (макс.) |")
        L.append(f"| один файл на процесс: максимум | {_f(w.get('max_s'))} | {_f(s['time_of_processing'].get('max_s'))} | — | — | — |")
        L.append(f"| один файл на процесс: сумма {s['n']} вызовов | {_f(w.get('sum_s'), 1)} | {_f(s['time_of_processing'].get('sum_s'))} | — | — | — |")
        L.append("")
        L.append("Вывод из таблицы: при старте процесса на каждый файл почти всё время уходит на холодный старт, а не на")
        L.append("снимок; пакетный режим (и API, где модели загружены один раз) стоимость старта платит однократно.")
        L.append("")
        L.append(f"Весь замер занял {_f(m['total_measure_wall_s'], 0)} с.")
    else:
        L.append("Раздел не заполнен: запуск с `--measure N --data <папка>` ещё не выполнялся на этой машине.")
    L.append("")
    # --- сопоставление
    L.append("## 4. Сопоставление источников")
    L.append("")
    if c and m:
        ip = m["inprocess"]
        ratio = (c["all_rows"]["median_s"] / ip["all"]["median_s"]) if ip["all"].get("median_s") else None
        L.append(f"Медиана `time_of_processing` на боевом сервере под чужой нагрузкой ({_f(c['all_rows']['median_s'])} с) и в "
                 f"локальном замере ({_f(ip['all']['median_s'])} с) различаются в {_f(ratio, 0)} раз при одном и том же коде пути "
                 "(2.3.2 и 2.4.0 различаются одним признаком `sp_pos` поверх уже посчитанного эмбеддинга, см. CHANGELOG). "
                 "Вероятная причина — условия, а не код: занятые ядра сервера во время прогона и переподписка потоков "
                 "(`inference.py` ставит `torch.set_num_threads(min(8, cpu_count))`; при лимите CPU контейнера ниже cpu_count "
                 "потоки конкурируют за ядра).")
        L.append("")
        if qc:
            ratio_q = (c["all_rows"]["median_s"] / qc["all_rows"]["median_s"]) if qc["all_rows"].get("median_s") else None
            L.append(f"Тихий сервер (раздел 2) подтверждает: тот же сервер без чужой нагрузки даёт медиану "
                     f"{_f(qc['all_rows']['median_s'])} с и p95 {_f(qc['all_rows']['p95_s'])} с — в {_f(ratio_q, 1)} раза меньше, чем под "
                     f"нагрузкой, и всё ещё в {_f(qc['all_rows']['median_s'] / ip['all']['median_s'], 0)} раз больше локального замера "
                     "(старее процессор сервера, OMP_NUM_THREADS=2 в контейнере, nice 10). Медианы по четвертям тихого прогона ровные.")
            L.append("")
            L.append(f"В документах корректно писать так: «на боевом сервере в тихих условиях — медиана "
                     f"{_f(qc['all_rows']['median_s'])} с на снимок, p95 {_f(qc['all_rows']['p95_s'])} с ({qc['all_rows']['n']} файлов за "
                     + (f"{_f(float(ex_q['batch_processing_wall_s']), 0)} с одним процессом, сумма `time_of_processing` {_f(qc['all_rows']['sum_s'], 1)} с; "
                        if (ex_q := (sq.get('extra') or {})).get('batch_processing_wall_s') else
                        f"{_f(qc['all_rows']['sum_s'], 0)} с суммарного `time_of_processing` одним процессом; ")
                     + f"позвоночник "
                     f"{_f(qc['by_region'].get(REGION_SPINE, {}).get('median_s'))} с, бедро {_f(qc['by_region'].get(REGION_HIP, {}).get('median_s'))} с); "
                     f"под чужой нагрузкой — медиана {_f(c['all_rows']['median_s'])} с, p95 {_f(c['all_rows']['p95_s'])} с (выгрузка 2.3.2); "
                     f"локально ({m['label']}) — порядка {_f(ip['all']['median_s'])} с». Одно число без условий не писать.")
        else:
            L.append(f"До заполнения раздела 2 в документах корректно писать так: «локально ({m['label']}) — порядка "
                     f"{_f(ip['all']['median_s'])} с на снимок (медиана замера {ip['n_files']} файлов; позвоночник "
                     f"{_f(ip['by_region'].get(REGION_SPINE, {}).get('time_of_processing', {}).get('median_s'))} с, бедро "
                     f"{_f(ip['by_region'].get(REGION_HIP, {}).get('time_of_processing', {}).get('median_s'))} с), на боевом сервере под "
                     f"чужой нагрузкой — медиана {_f(c['all_rows']['median_s'])} с, p95 {_f(c['all_rows']['p95_s'])} с (499 файлов)». "
                     "Одно число без условий не писать.")
    else:
        L.append("Заполняется, когда есть оба источника.")
    L.append("")
    L.append("## 5. Как воспроизвести")
    L.append("")
    L.append("```bash")
    L.append("# (а) распределение time_of_processing по CSV поставки (секунды)")
    L.append("python tools/perf_passport.py --csv dataset/regress_2_3_2.csv")
    L.append("# раздел 2: CSV регрессии на тихом сервере; ручные числа (стена, API, RSS, образ) — JSON-строкой или файлом")
    L.append("python tools/perf_passport.py --quiet-csv dataset/regress_2_4_0.csv \\")
    L.append("    --quiet-extra '{\"container_wall_s\": 384, \"docker_image_gb\": 2.57}'   # прежние ручные числа сохраняются")
    L.append("# (б) локальный замер на 12 файлах (6 позвоночник + 6 бедро, разные исследования; 3 прохода), ~1–2 мин на 2 vCPU")
    L.append("OMP_NUM_THREADS=2 nice -n 5 python tools/perf_passport.py --measure 12 --data /path/Исследования \\")
    L.append("    --label \"своя машина: N vCPU, что ещё работало\"")
    L.append("# в образе: docker run --rm -v /path/Исследования:/data:ro <образ> python tools/perf_passport.py --measure 12 --data /data")
    L.append("# тест: воспроизводимость (а) и проверка аргументов, без долгого замера")
    L.append("python tests/test_perf_passport.py")
    L.append("```")
    L.append("")
    L.append("Выбор файлов детерминированный (исследования по алфавиту, первый файл нужной области, по одному на")
    L.append("исследование), поэтому два замера на разных машинах сравнивают одни и те же снимки. Числа режима (а)")
    L.append("воспроизводятся побайтово; числа режима (б) — с разбросом машины.")
    L.append("")
    L.append("## 6. Что это не значит")
    L.append("")
    L.append("- Числа зависят от машины, числа потоков, соседней нагрузки и версии библиотек; на другой машине они другие.")
    L.append("- Пропускная способность (снимков в час) не обещается: сервис обрабатывает файлы последовательно в одном")
    L.append("  процессе, а требование ТЗ — не более 3 минут на исследование — единственная граница, с которой сравниваются числа.")
    L.append("- Сравнений с внешними ориентирами и другими инструментами нет и не подразумевается.")
    L.append("- Локальный замер сделан на 12 файлах, а не на 499; это оценка порядка величины, а не распределение потока.")
    if (p.get("server_quiet") or {}).get("extra", {}) and (p["server_quiet"]["extra"] or {}).get("api_batch_wall_s"):
        L.append("- Время API с бонусными выходами (визуализация, SR, SEG) снято один раз на одной партии (раздел 2); это порядок")
        L.append("  величины, а не распределение; подробнее о режиме API — `README.md`.")
    else:
        L.append("- Время API с бонусными выходами (визуализация, SR) здесь не измеряется; оно больше и описано отдельно в `README.md`.")
    L.append("- Пиковый RSS измерен как `ru_maxrss` процесса Python; память контейнера в целом (с ОС и кэшами) больше.")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Паспорт производительности DensitoAI: time_of_processing из CSV и локальный замер")
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help=f"CSV поставки (9 колонок) для режима (а); по умолчанию {DEFAULT_CSV}")
    ap.add_argument("--csv-conditions", default=CSV_CONDITIONS_DEFAULT, help="Текст условий боевого прогона для CSV")
    ap.add_argument("--measure", type=int, default=None, metavar="N", help="Режим (б): локальный замер на N файлах (чётное, >= 2)")
    ap.add_argument("--data", default=None, help="Папка с исследованиями для --measure (подпапки = исследования)")
    ap.add_argument("--label", default=None, help="Подпись условий локального замера (машина, нагрузка)")
    ap.add_argument("--passes", type=int, default=3, help="Сколько раз пройти файлы в одном процессе (режим б; по умолчанию 3)")
    ap.add_argument("--out-json", default=str(DEFAULT_JSON))
    ap.add_argument("--out-md", default=str(DEFAULT_MD))
    ap.add_argument("--fresh", action="store_true", help="Не сохранять прежние разделы из существующего JSON")
    ap.add_argument("--quiet-csv", default=None, metavar="CSV",
                    help=f"CSV регрессии на боевом сервере в тихих условиях → раздел 2 (например {QUIET_CSV_DEFAULT})")
    ap.add_argument("--quiet-conditions", default=QUIET_CONDITIONS_DEFAULT, help="Текст условий тихого прогона")
    ap.add_argument("--quiet-extra", default=None, metavar="JSON",
                    help="Ручные числа тихого сервера: JSON-строка или путь к JSON-файлу с ключами "
                         + ", ".join(k for k, _ in QUIET_EXTRA_FIELDS) + ", note, container_wall_note")
    return ap


def _load_quiet_extra(spec: Optional[str]) -> Optional[Dict[str, Any]]:
    """--quiet-extra: JSON-строка или путь к файлу; допускаются только известные ключи."""
    if not spec:
        return None
    text = Path(spec).read_text(encoding="utf-8") if Path(spec).is_file() else spec
    data = json.loads(text)
    if not isinstance(data, dict):
        raise SystemExit("--quiet-extra: ожидается JSON-объект")
    allowed = {k for k, _ in QUIET_EXTRA_FIELDS} | {"note", "container_wall_note", "measured_on"}
    bad = sorted(set(data) - allowed)
    if bad:
        raise SystemExit(f"--quiet-extra: неизвестные ключи {bad}; допустимы {sorted(allowed)}")
    return data


def build_server_quiet(prev: Optional[Dict[str, Any]], quiet_csv: Optional[Path], conditions: str,
                       extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Раздел server_quiet: распределение по CSV тихого прогона + ручные числа; без CSV — заглушка."""
    prev = prev or {}
    csv_part = analyze_csv(quiet_csv, conditions) if quiet_csv else prev.get("csv")
    merged_extra = dict(prev.get("extra") or {})
    if extra:
        merged_extra.update(extra)
    if not csv_part:
        return {"status": PLACEHOLDER, "median_s": None, "p95_s": None, "batch_wall_499_s": None,
                "container_peak_rss_mb": None, "docker_image_mb": None, "extra": merged_extra or None,
                "note": "числа боевого сервера в тихих условиях и размер образа вносит оркестратор после замера на сервере"}
    a = csv_part["all_rows"]
    return {
        "status": f"заполнено по {csv_part['source_name']}",
        "median_s": a["median_s"], "p95_s": a["p95_s"], "sum_s": a["sum_s"],
        "batch_wall_499_s": merged_extra.get("batch_processing_wall_s"),
        "container_peak_rss_mb": None,
        "docker_image_mb": (round(float(merged_extra["docker_image_gb"]) * 1000) if merged_extra.get("docker_image_gb") else None),
        "csv": csv_part,
        "extra": merged_extra or None,
        "note": "распределение — из CSV тихого прогона (analyze_csv); ручные числа — из --quiet-extra",
    }


def validate_args(args: argparse.Namespace) -> Optional[str]:
    """Проверка аргументов без запуска; возвращает текст ошибки или None."""
    if args.measure is not None:
        if args.measure < 2:
            return "--measure N: N должно быть не меньше 2"
        if args.measure % 2:
            return "--measure N: N должно быть чётным (половина позвоночник, половина бедро)"
        if args.passes < 1:
            return "--passes: не меньше 1"
        if not args.data:
            return "--measure требует --data <папка с исследованиями>"
        if not Path(args.data).is_dir():
            return f"--data: папка не найдена: {args.data}"
    if args.csv and not Path(args.csv).exists() and args.measure is None:
        return f"--csv: файл не найден: {args.csv}"
    if args.quiet_csv and not Path(args.quiet_csv).exists():
        return f"--quiet-csv: файл не найден: {args.quiet_csv}"
    if args.quiet_extra:
        try:
            _load_quiet_extra(args.quiet_extra)
        except (SystemExit, ValueError, OSError) as e:
            return f"--quiet-extra: {e}"
    return None


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    err = validate_args(args)
    if err:
        print(err, file=sys.stderr)
        return 2
    out_json, out_md = Path(args.out_json), Path(args.out_md)
    passport: Dict[str, Any] = {}
    if out_json.exists() and not args.fresh:
        try:
            passport = json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            passport = {}
    passport["version"] = _read_version()
    passport["generated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    passport["tool"] = "tools/perf_passport.py"
    if args.csv and Path(args.csv).exists():
        passport["csv"] = analyze_csv(Path(args.csv), args.csv_conditions)
        print(f"(а) {args.csv}: n={passport['csv']['n_rows']}, медиана {passport['csv']['all_rows']['median_s']} с, "
              f"p95 {passport['csv']['all_rows']['p95_s']} с, всего {passport['csv']['all_rows']['sum_s']} с")
    elif args.csv:
        print(f"(а) пропущено: нет файла {args.csv}", file=sys.stderr)
    if args.measure:
        label = args.label or f"локальная машина, {os.cpu_count()} vCPU, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}"
        passport["measure"] = run_measure(args.measure, Path(args.data), Path(args.csv) if args.csv else None, label,
                                          passes=args.passes)
        ip = passport["measure"]["inprocess"]
        print(f"(б) {ip['n_files']} файлов: холодный старт {ip['cold_start']['cold_start_total_s']} с, "
              f"медиана на файл {ip['all']['median_s']} с, пиковый RSS {ip['rss_mb']['peak_end']} МБ")
    passport["server_quiet"] = build_server_quiet(passport.get("server_quiet"),
                                                  Path(args.quiet_csv) if args.quiet_csv else None,
                                                  args.quiet_conditions, _load_quiet_extra(args.quiet_extra))
    if passport["server_quiet"].get("csv"):
        qa = passport["server_quiet"]["csv"]["all_rows"]
        print(f"(тихий сервер) {passport['server_quiet']['csv']['source_name']}: n={qa['n']}, медиана {qa['median_s']} с, "
              f"p95 {qa['p95_s']} с, всего {qa['sum_s']} с")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(passport, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_md(passport), encoding="utf-8")
    print(f"-> {out_json}\n-> {out_md}")
    return 0


def _read_version() -> str:
    try:
        import yaml
        return str(yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")).get("version", "?"))
    except Exception:  # noqa: BLE001
        return "?"


if __name__ == "__main__":
    sys.exit(main())
