#!/usr/bin/env python3
"""Стресс-тест устойчивости DensitoAI к искажениям входа (шаг 2 плана).

Берёт стратифицированную выборку снимков разметки, для каждого искажения пересохраняет DICOM,
прогоняет через тот же движок инференса, что и API, и сравнивает с исходным результатом:
  * flip-rate      — доля снимков, у которых изменился класс качества (quality_class);
  * region flip    — доля снимков, у которых изменилась область;
  * |Δ prob| p50/p95 — сдвиг вероятности нарушения;
  * failure-rate   — доля Failure;
  * time p50/p95   — время на снимок.
Отдельно: упаковка (вложенный zip, битый zip, одинаковые имена в разных папках).

Запуск:  python tests/robustness_suite.py [--n 60] [--out docs/ROBUSTNESS_REPORT.md]
"""
from __future__ import annotations

import argparse, io, json, logging, os, shutil, sys, tempfile, time, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydicom.uid import ExplicitVRLittleEndian

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inference import DensitoInference, load_config  # noqa: E402

logging.basicConfig(level=logging.WARNING)
RNG = np.random.default_rng(20260918)


# ----------------------------------------------------------------------------- искажения
def _base(ds):
    """Копия датасета с распакованными пикселями и ExplicitVRLittleEndian."""
    ds = pydicom.dcmread(str(ds), force=True) if not isinstance(ds, pydicom.Dataset) else ds
    arr = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr
        ds.PhotometricInterpretation = "MONOCHROME2"
    return ds, arr


def _write(ds, arr, out: Path, bits=None, photometric="MONOCHROME2"):
    bits = bits or int(ds.BitsAllocated)
    if bits == 8:
        px = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        px = np.clip(arr, 0, 65535).astype(np.uint16)
    ds.PixelData = px.tobytes()
    ds.Rows, ds.Columns = px.shape
    ds.BitsAllocated = bits; ds.BitsStored = bits; ds.HighBit = bits - 1
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    for t in ("RescaleSlope", "RescaleIntercept", "WindowCenter", "WindowWidth"):
        if t in ds:
            del ds[t]
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.is_little_endian, ds.is_implicit_VR = True, False
    ds.save_as(str(out), write_like_original=False)


def d_identity(src, out):
    ds, arr = _base(src); _write(ds, arr, out)


def d_tags_stripped(src, out):
    ds, arr = _base(src)
    for t in ("PixelSpacing", "ImagerPixelSpacing", "StudyDescription", "SeriesDescription",
              "BodyPartExamined", "Laterality", "ImageLaterality", "ViewPosition", "Manufacturer",
              "ManufacturerModelName", "PatientName", "PatientID", "InstitutionName", "StationName",
              "ProtocolName", "OperatorsName"):
        if t in ds:
            del ds[t]
    _write(ds, arr, out)


def d_no_uids(src, out):
    ds, arr = _base(src)
    for t in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"):
        if t in ds:
            del ds[t]
    _write(ds, arr, out)


def d_mono1(src, out):
    ds, arr = _base(src)
    mx = (2 ** int(ds.BitsStored) - 1)
    _write(ds, mx - arr, out, photometric="MONOCHROME1")


def d_bits16(src, out):
    ds, arr = _base(src)
    lo, hi = arr.min(), arr.max()
    a16 = (arr - lo) / max(hi - lo, 1) * 4095.0     # 12 значащих бит в 16-битном контейнере
    _write(ds, a16, out, bits=16)


def d_noise(src, out, sigma_frac=0.03):
    ds, arr = _base(src)
    mx = (2 ** int(ds.BitsStored) - 1)
    _write(ds, arr + RNG.normal(0, sigma_frac * mx, arr.shape), out)


def d_bright_up(src, out):
    ds, arr = _base(src)
    mx = (2 ** int(ds.BitsStored) - 1)
    _write(ds, (arr / mx) ** 0.7 * mx, out)


def d_bright_down(src, out):
    ds, arr = _base(src)
    mx = (2 ** int(ds.BitsStored) - 1)
    _write(ds, (arr / mx) ** 1.4 * mx, out)


def _resize(arr, f):
    from PIL import Image
    im = Image.fromarray(arr.astype(np.float32))
    h, w = arr.shape
    return np.asarray(im.resize((max(8, round(w * f)), max(8, round(h * f))), Image.BILINEAR), dtype=np.float32)


def d_resize_080(src, out):
    ds, arr = _base(src)
    a = _resize(arr, 0.8)
    if "PixelSpacing" in ds:
        ds.PixelSpacing = [float(ds.PixelSpacing[0]) / 0.8, float(ds.PixelSpacing[1]) / 0.8]
    _write(ds, a, out)


def d_resize_125(src, out):
    ds, arr = _base(src)
    a = _resize(arr, 1.25)
    if "PixelSpacing" in ds:
        ds.PixelSpacing = [float(ds.PixelSpacing[0]) / 1.25, float(ds.PixelSpacing[1]) / 1.25]
    _write(ds, a, out)


def d_crop_border(src, out):
    """Обрезка 4 % по каждому краю (как при ином поле экспорта)."""
    ds, arr = _base(src)
    h, w = arr.shape
    dh, dw = int(h * 0.04), int(w * 0.04)
    _write(ds, arr[dh:h - dh, dw:w - dw], out)


DISTORTIONS = {
    "identity (пересохранение)": d_identity,
    "теги удалены (PixelSpacing, описание, сторона, аппарат)": d_tags_stripped,
    "нет UID исследования/снимка": d_no_uids,
    "MONOCHROME1 (инверсия)": d_mono1,
    "16-битный контейнер (12 бит)": d_bits16,
    "гауссов шум σ=3 %": d_noise,
    "ярче (гамма 0.7)": d_bright_up,
    "темнее (гамма 1.4)": d_bright_down,
    "resize ×0.8 (PixelSpacing пересчитан)": d_resize_080,
    "resize ×1.25 (PixelSpacing пересчитан)": d_resize_125,
    "обрезка 4 % по краям": d_crop_border,
}


# ----------------------------------------------------------------------------- выборка
def pick_sample(n: int) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "labels_for_embeddings.csv")
    df = df[df["file_path"].map(lambda p: Path(p).exists())]
    parts = []
    for region, grp in df.groupby("region"):
        k = max(1, round(n * len(grp) / len(df)))
        pos = grp[grp["quality_class"] == 1]
        neg = grp[grp["quality_class"] != 1]
        kp = min(len(pos), max(1, k // 2))
        parts.append(pos.sample(kp, random_state=1))
        parts.append(neg.sample(min(len(neg), k - kp), random_state=1))
    return pd.concat(parts).reset_index(drop=True)


def run_rows(engine: DensitoInference, files, root: Path):
    rows = []
    for f in files:
        row, _ = engine.process_file(Path(f), root)
        rows.append(row)
    return pd.DataFrame(rows)


def q(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


# ----------------------------------------------------------------------------- упаковка
def packaging_checks(engine: DensitoInference, sample_files, tmp: Path):
    """Вложенный zip, битый zip, одинаковые имена в разных папках, zip внутри папки."""
    res = {}
    src = [Path(p) for p in sample_files[:3]]
    # 1) одинаковые имена в разных папках
    d = tmp / "same_names"; (d / "A").mkdir(parents=True); (d / "B").mkdir()
    for p in src[:2]:
        shutil.copy(p, d / "A" / "CR000000.dcm") if p is src[0] else shutil.copy(p, d / "B" / "CR000000.dcm")
    out = tmp / "same.csv"
    rows = engine.run(d, out)
    res["одинаковые имена в разных папках"] = (len(rows) == 2 and len({r["path_to_study"] for r in rows}) == 2
                                                and len({r["image_uid"] for r in rows}) == 2)
    # 2) вложенный zip
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        for p in src:
            z.write(p, f"inner/{p.name}")
    outer = tmp / "nested.zip"
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("level1/inner.zip", inner.getvalue())
        z.write(src[0], "level1/direct.dcm")
    rows = engine.run(outer, tmp / "nested.csv")
    res["вложенный zip (zip в zip + файл рядом)"] = (len(rows) >= 1 and
                                                    all(r["processing_status"] == "Success" for r in rows),
                                                    f"{len(rows)} строк")
    # 3) битый zip
    broken = tmp / "broken.zip"
    broken.write_bytes(outer.read_bytes()[: outer.stat().st_size // 2])
    try:
        rows = engine.run(broken, tmp / "broken.csv")
        res["битый zip (обрезан наполовину)"] = (True, f"без исключения, {len(rows)} строк")
    except ValueError as e:
        res["битый zip (обрезан наполовину)"] = (True, f"понятная ошибка → HTTP 400: {str(e)[:60]}…")
    except Exception as e:  # noqa: BLE001
        res["битый zip (обрезан наполовину)"] = (False, f"необработанное исключение {type(e).__name__}: {e}")
    # 4) файлы без расширения и с чужим расширением
    d = tmp / "ext"; d.mkdir()
    shutil.copy(src[0], d / "IMG0001")
    shutil.copy(src[1], d / "scan.img")
    shutil.copy(src[2], d / "study.DCM")
    rows = engine.run(d, tmp / "ext.csv")
    res["без расширения / .img / .DCM"] = (len(rows) == 3, f"{len(rows)} строк из 3 файлов")
    return res


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default=str(ROOT / "docs" / "ROBUSTNESS_REPORT.md"))
    ap.add_argument("--json", default=str(ROOT / "docs" / "qa" / "robustness_results.json"))
    a = ap.parse_args()

    sample = pick_sample(a.n)
    files = list(sample["file_path"])
    print(f"выборка: {len(files)} файлов; по областям: {sample['region'].value_counts().to_dict()}; "
          f"с нарушениями: {int((sample['quality_class'] == 1).sum())}", flush=True)

    engine = DensitoInference(cfg=load_config())
    engine.run(Path(files[0]).parent, Path(tempfile.mkdtemp()) / "warm.csv", limit=1)  # прогрев

    tmp = Path(tempfile.mkdtemp(prefix="densito_robust_"))
    t0 = time.perf_counter()
    base = run_rows(engine, files, None)
    base_time = time.perf_counter() - t0
    print(f"baseline: {len(base)} строк за {base_time:.1f} с, failures={int((base.processing_status != 'Success').sum())}", flush=True)

    results = {}
    for name, fn in DISTORTIONS.items():
        d = tmp / f"d{len(results):02d}"; d.mkdir()
        outs = []
        n_write_fail = 0
        for i, f in enumerate(files):
            o = d / f"{i:03d}.dcm"
            try:
                fn(Path(f), o); outs.append(o)
            except Exception as e:  # noqa: BLE001
                n_write_fail += 1
                print(f"  [warn] не удалось создать искажение '{name}' для {f}: {e}", flush=True)
                outs.append(None)
        rows = []
        for f, o in zip(files, outs):
            if o is None:
                rows.append({"processing_status": "SKIP"}); continue
            row, _ = engine.process_file(o, d)
            rows.append(row)
        df = pd.DataFrame(rows)
        ok = (df["processing_status"] == "Success").values & (base["processing_status"] == "Success").values
        cls_flip = (df.loc[ok, "quality_class"].astype(int).values != base.loc[ok, "quality_class"].astype(int).values)
        reg_flip = (df.loc[ok, "anatomical_region"].values != base.loc[ok, "anatomical_region"].values)
        dprob = np.abs(df.loc[ok, "quality_prob"].astype(float).values - base.loc[ok, "quality_prob"].astype(float).values)
        t = df.loc[df["processing_status"] == "Success", "time_of_processing"].astype(float).values
        results[name] = {
            "n": int(len(df)), "n_ok": int(ok.sum()),
            "failure_rate": float((df["processing_status"] != "Success").mean()),
            "class_flip_rate": float(cls_flip.mean()) if ok.sum() else float("nan"),
            "region_flip_rate": float(reg_flip.mean()) if ok.sum() else float("nan"),
            "dprob_p50": q(dprob, 50), "dprob_p95": q(dprob, 95),
            "time_p50": q(t, 50), "time_p95": q(t, 95),
        }
        r = results[name]
        print(f"{name:55s} flip={r['class_flip_rate']:.3f} region_flip={r['region_flip_rate']:.3f} "
              f"fail={r['failure_rate']:.3f} dP50={r['dprob_p50']:.3f} dP95={r['dprob_p95']:.3f} "
              f"t50={r['time_p50']:.2f} t95={r['time_p95']:.2f}", flush=True)

    bt = base.loc[base["processing_status"] == "Success", "time_of_processing"].astype(float).values
    pack = packaging_checks(engine, files, tmp / "pack")
    for k, v in pack.items():
        print(f"упаковка: {k}: {'OK' if (v[0] if isinstance(v, tuple) else v) else 'FAIL'}"
              + (f" ({v[1]})" if isinstance(v, tuple) else ""), flush=True)

    payload = {"n_files": len(files), "regions": sample["region"].value_counts().to_dict(),
               "n_positive": int((sample["quality_class"] == 1).sum()),
               "baseline": {"failure_rate": float((base.processing_status != "Success").mean()),
                            "time_p50": q(bt, 50), "time_p95": q(bt, 95), "wall_total_s": base_time},
               "distortions": results,
               "packaging": {k: {"ok": bool(v[0] if isinstance(v, tuple) else v),
                                 "note": (v[1] if isinstance(v, tuple) else "")} for k, v in pack.items()},
               "cpu": os.cpu_count(), "date": time.strftime("%Y-%m-%d")}
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    write_report(payload, Path(a.out))
    shutil.rmtree(tmp, ignore_errors=True)
    print("REPORT", a.out)


def write_report(p, out: Path):
    L = []
    L.append("# Отчёт об устойчивости DensitoAI к искажениям входа\n")
    L.append(f"Дата: {p['date']}. Выборка: {p['n_files']} снимков разметки "
             f"({', '.join(f'{k}: {v}' for k, v in p['regions'].items())}), из них с нарушениями: {p['n_positive']}. "
             f"Движок: тот же `src/inference.py`, что и API; CPU: {p['cpu']} ядра, без GPU.\n")
    L.append("Метод: каждый снимок пересохраняется с одним искажением и прогоняется повторно; сравнивается с исходным "
             "результатом того же снимка. **Flip-rate** — доля снимков, у которых изменился класс качества "
             "(0/1). **Region flip** — изменилась ли определённая область. **|Δp|** — сдвиг вероятности нарушения. "
             "**Failure** — доля строк со статусом Failure (по ТЗ битый вход не должен ронять пакет).\n")
    b = p["baseline"]
    L.append(f"Исходный прогон: failure {b['failure_rate']*100:.1f} %, время на снимок p50 {b['time_p50']:.2f} с, "
             f"p95 {b['time_p95']:.2f} с (всего {b['wall_total_s']:.0f} с на {p['n_files']} файлов).\n")
    L.append("## Искажения пикселей и тегов\n")
    L.append("| Искажение | Flip-rate класса | Region flip | \\|Δp\\| p50 | \\|Δp\\| p95 | Failure | t p50, с | t p95, с |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, r in p["distortions"].items():
        L.append(f"| {k} | {r['class_flip_rate']*100:.1f} % ({round(r['class_flip_rate']*r['n_ok'])}/{r['n_ok']}) | "
                 f"{r['region_flip_rate']*100:.1f} % | {r['dprob_p50']:.3f} | {r['dprob_p95']:.3f} | "
                 f"{r['failure_rate']*100:.1f} % | {r['time_p50']:.2f} | {r['time_p95']:.2f} |")
    L.append("")
    L.append("## Упаковка и имена файлов\n")
    L.append("| Проверка | Результат | Примечание |")
    L.append("|---|---|---|")
    for k, v in p["packaging"].items():
        L.append(f"| {k} | {'OK' if v['ok'] else 'FAIL'} | {v['note']} |")
    L.append("")
    L.append("## Как читать\n")
    L.append("- `identity` показывает шум самого пайплайна при пересохранении DICOM (ожидаемо 0 %).")
    L.append("- Изменения яркости/шума/битности проходят через перцентильную нормализацию 1–99 % — flip-rate здесь "
             "показывает, насколько контуры A (геометрия) и B (эмбеддинг) чувствительны к экспозиции.")
    L.append("- Resize с пересчитанным PixelSpacing имитирует другой экспорт того же снимка: измерения в мм должны "
             "сохраниться, а правило области по ширине (300/280/248 px) перестаёт работать — область определяет резервный "
             "классификатор по эмбеддингу, поэтому Region flip тут — главный показатель.")
    L.append("- Удаление тегов проверяет, что система не зависит от PixelSpacing (подставляется паспортное значение "
             "аппарата) и от текстовых описаний.")
    L.append("- Вход, которого в выборке быть не может (валидный DICOM, но снимок не денситометрический), "
             "проверяется отдельно: `python tests/test_ood_foreign.py` — строка обязана получить Success при "
             "сохранённом контракте колонок, а бонусный слой обязан поднять `ood_flag` с причиной.")
    L.append("- Скрипт: `python tests/robustness_suite.py --n 60`; сырые числа — `docs/qa/robustness_results.json`.")
    out.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
