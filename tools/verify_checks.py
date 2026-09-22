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
def preproc_consistency_check() -> tuple:
    """(имя, ok, детали): config.yaml preprocessing.variant_by_criterion == meta в .pkl и metrics_summary.json."""
    name = "К11: вариант предобработки в config == варианту обучения моделей"
    root = Path(__file__).resolve().parents[1]
    try:
        import pickle

        import yaml

        cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")) or {}
        sec = cfg.get("preprocessing") or {}
        enabled = bool(sec.get("enabled", False))
        by_crit = (sec.get("variant_by_criterion") or {}) if enabled else {}

        def want(crit, kind):
            entry = by_crit.get(crit) or {}
            return str(entry.get(kind) or "baseline")

        models_dir = root / "models"
        problems, checked = [], 0
        pkl_map = {"sp_pos": "model_spine_sp_pos", "sp_axis": "model_spine_sp_axis",
                   "sp_art": "model_spine_sp_art", "hip_pos": "model_hip_pos", "hip_roi": "model_hip_roi"}
        for crit, base in pkl_map.items():
            for kind, suffix, meta_key in (("geom", "_geom.pkl", "preproc_geom"),
                                           ("emb", "_emb_pca.pkl", "preproc_emb")):
                p = models_dir / f"{base}{suffix}"
                if not p.exists():
                    continue
                with open(p, "rb") as fh:
                    obj = pickle.load(fh)
                # train_final_models кладёт meta в верхний уровень dict (там же, где emb_source)
                meta = obj if isinstance(obj, dict) else {}
                got = str(meta.get(meta_key, meta.get("meta", {}).get(meta_key, "baseline")
                                   if isinstance(meta.get("meta"), dict) else "baseline"))
                checked += 1
                if got != want(crit, kind):
                    problems.append(f"{p.name}: обучена на '{got}', config требует '{want(crit, kind)}'")

        ms = models_dir / "metrics_summary.json"
        if ms.exists():
            summary = json.loads(ms.read_text(encoding="utf-8"))
            for region in ("spine", "hip"):
                for crit, payload in (summary.get(region) or {}).items():
                    if not isinstance(payload, dict) or "preproc" not in payload:
                        continue
                    pp = payload["preproc"] or {}
                    for kind in ("geom", "emb"):
                        checked += 1
                        got = str(pp.get(kind, "baseline"))
                        if got != want(crit, kind):
                            problems.append(f"metrics_summary {region}/{crit}.{kind}: обучено на '{got}', "
                                            f"config требует '{want(crit, kind)}'")
        detail = (f"enabled={enabled}, сверено {checked} записей" if not problems
                  else "; ".join(problems[:6]))
        return name, not problems, detail
    except Exception as e:  # noqa: BLE001
        return name, False, f"проверка не выполнена: {e}"


def emb_source_consistency_check() -> tuple:
    """(имя, ok, детали): config.yaml embeddings.source_by_criterion == meta в .pkl и metrics_summary.json.

    К13 сделал источник эмбеддингов контура B настраиваемым по критерию. Если конфиг
    расходится с тем, на чём обучена модель, сервис посчитает эмбеддинг одним бэкбоном,
    а логрегрессию применит от другого — предсказания тихо испортятся, без ошибки.
    """
    name = "К13: источник эмбеддингов в config == источнику обучения моделей"
    root = Path(__file__).resolve().parents[1]
    try:
        import pickle

        import yaml

        cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")) or {}
        by_crit = ((cfg.get("embeddings") or {}).get("source_by_criterion") or {})

        def want(crit):
            # конфиг — источник истины; при отсутствии записи — imagenet, как в train_stacked
            return str(by_crit.get(crit) or ("densito" if not by_crit and crit == "sp_pos" else "imagenet"))

        models_dir = root / "models"
        problems, checked = [], 0
        pkl_map = {"sp_pos": "model_spine_sp_pos", "sp_axis": "model_spine_sp_axis",
                   "sp_art": "model_spine_sp_art", "hip_pos": "model_hip_pos", "hip_roi": "model_hip_roi"}
        for crit, base in pkl_map.items():
            p = models_dir / f"{base}_emb_pca.pkl"
            if not p.exists():
                continue
            with open(p, "rb") as fh:
                obj = pickle.load(fh)
            meta = obj if isinstance(obj, dict) else {}
            got = str(meta.get("emb_source", "imagenet"))
            checked += 1
            if got != want(crit):
                problems.append(f"{p.name}: обучена на '{got}', config требует '{want(crit)}'")
            # файл весов бэкбона должен быть в образе (imagenet берётся из torch_home)
            if got != "imagenet":
                wfile = models_dir / f"backbone_{got}.pth"
                checked += 1
                if not wfile.exists():
                    problems.append(f"нет файла весов {wfile.name} для источника '{got}' ({crit})")

        ms = models_dir / "metrics_summary.json"
        if ms.exists():
            summary = json.loads(ms.read_text(encoding="utf-8"))
            for region in ("spine", "hip"):
                for crit, payload in (summary.get(region) or {}).items():
                    if not isinstance(payload, dict) or "emb_source" not in payload:
                        continue
                    checked += 1
                    got = str(payload.get("emb_source") or "imagenet")
                    if got != want(crit):
                        problems.append(f"metrics_summary {region}/{crit}: обучено на '{got}', "
                                        f"config требует '{want(crit)}'")
        detail = (f"сверено {checked} записей: " + ", ".join(f"{c}={want(c)}" for c in pkl_map)
                  if not problems else "; ".join(problems[:6]))
        return name, not problems, detail
    except Exception as e:  # noqa: BLE001
        return name, False, f"проверка не выполнена: {e}"


def official_dictionary_check() -> tuple:
    """(имя, ok, детали): строки выгрузки совпадают со словарём организаторов.

    Словарь лежит в schema/official_dictionary.json и списан с файла разъяснений
    организаторов (ответы на вопросы 1, 6, 8, 15). Проверяется ровно то, от чего
    зависит их скрипт подсчёта метрик: значения violation_type, значения
    anatomical_region, разделитель нескольких нарушений, имя колонки вероятности
    и паспортный размер пикселя. Расхождение здесь стоит дороже любой модели:
    правильный класс с чужой строкой в их подсчёте — это промах.
    """
    name = "Строки выгрузки == словарю организаторов (violation_type, регионы, разделитель)"
    root = Path(__file__).resolve().parents[1]
    try:
        import yaml

        d = json.loads((root / "schema" / "official_dictionary.json").read_text(encoding="utf-8"))
        cfg_text = (root / "config.yaml").read_text(encoding="utf-8")
        cfg = yaml.safe_load(cfg_text)
        problems = []

        # 1. каждая строка нарушения из словаря присутствует в конфиге
        for region, items in d["violation_type"].items():
            for s in items:
                if s not in cfg_text:
                    problems.append(f"нет строки нарушения «{s}» ({region})")

        # 2. в конфиге нет строк нарушений, которых нет в словаре
        allowed = {s for items in d["violation_type"].values() for s in items}
        declared = set()
        for section in ("violations", "violation_type", "criteria"):
            node = cfg.get(section)
            if isinstance(node, dict):
                for v in node.values():
                    if isinstance(v, str):
                        declared.add(v)
                    elif isinstance(v, dict):
                        for vv in v.values():
                            if isinstance(vv, str):
                                declared.add(vv)
        for s in sorted(declared):
            looks_like_violation = s in allowed or s.startswith(("Некорректн", "Не выравнена", "Присутствуют"))
            if looks_like_violation and s not in allowed:
                problems.append(f"строка «{s}» отсутствует в словаре организаторов")

        # 3. регионы
        for s in d["anatomical_region"]:
            if s not in cfg_text:
                problems.append(f"нет строки региона «{s}»")

        # 4. разделитель и имя колонки вероятности
        inf = (root / "src" / "inference.py").read_text(encoding="utf-8")
        sep = d["violation_separator"]
        if f"'{sep}'.join" not in inf and f'"{sep}".join' not in inf:
            problems.append(f"в inference.py не найдено объединение нарушений через «{sep}»")
        col = d["probability_column"]
        for f in ("schema/results_row.schema.json", "schema/api_analyze_response.schema.json"):
            if col not in (root / f).read_text(encoding="utf-8"):
                problems.append(f"колонка {col} отсутствует в {f}")

        # 5. паспортный размер пикселя
        px = d["pixel_spacing_mm"]
        for val in (px["y_row"], px["x_col"]):
            if str(val) not in cfg_text:
                problems.append(f"размер пикселя {val} мм отсутствует в config.yaml")

        detail = (f"сверено: {sum(len(v) for v in d['violation_type'].values())} строк нарушений, "
                  f"{len(d['anatomical_region'])} региона, разделитель «{sep}», колонка {col}, "
                  f"пиксель {px['y_row']}×{px['x_col']} мм" if not problems else "; ".join(problems[:6]))
        return name, not problems, detail
    except Exception as e:  # noqa: BLE001
        return name, False, f"проверка не выполнена: {e}"


def transfer_invariance_check(json_path: Path) -> tuple[str, bool, str]:
    """Инвариантность предсказаний к именам файлов, порядку и упаковке (tools/transfer_check.py).
    Закрытый набор приходит без суффиксов в именах и, возможно, одним архивом."""
    name = "Инвариантность к именам файлов, порядку и упаковке (переименованная копия и zip)"
    if not json_path or not Path(json_path).exists():
        return name, False, "нет transfer_check.json (проверка не выполнялась)"
    try:
        d = json.loads(Path(json_path).read_text(encoding="utf-8"))
        modes = d.get("modes", {})
        if not modes:
            return name, False, "в transfer_check.json нет режимов"
        parts, ok_all = [], True
        for mode, res in sorted(modes.items()):
            ok = bool(res.get("ok"))
            ok_all = ok_all and ok
            det = (f"{mode}: сверено {res.get('n_compared')} строк, max|Δprob| {res.get('max_abs_dprob')}"
                   if ok else f"{mode}: {'; '.join(res.get('problems', [])[:3])}")
            parts.append(det)
        return name, ok_all, "; ".join(parts)
    except Exception as e:  # noqa: BLE001
        return name, False, f"проверка не выполнена: {e}"


def phantom_ground_truth_check(phantoms: Path, debug_csv: Path) -> tuple[str, bool, str]:
    """Сверка измеренных признаков с истинной геометрией фантомов из MANIFEST.json:
    наклон оси, сдвиг центра, обрезка поля, область (позвоночник/бедро).
    Металл на синтетических фантомах не проверяется: их «кость» сама по себе яркая,
    порог по интенсивности на таких картинках даёт ложное срабатывание (реальные данные
    оцениваются метриками sp_art, а не фантомами)."""
    name = "Истинная геометрия MANIFEST.json: наклон, сдвиг, обрезка поля, область"
    try:
        man = json.loads((Path(phantoms) / "MANIFEST.json").read_text(encoding="utf-8"))
        truth = {f["path"]: f for f in man["files"] if f.get("kind") == "phantom"}
        if not Path(debug_csv).exists():
            return name, False, f"нет debug CSV {debug_csv}"
        with open(debug_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        problems: list[str] = []
        notes: list[str] = []
        n = 0

        def num(r, k):
            v = (r.get(k) or "").strip()
            try:
                return float(v)
            except ValueError:
                return None

        for r in rows:
            fp = (r.get("file") or "").replace("\\\\", "/")
            key = next((k for k in truth if fp.endswith(k)), None)
            if key is None:
                continue
            t = truth[key]
            g = t.get("geometry", {})
            variant = g.get("variant", "")
            region_true = t.get("region", "")
            region_meas = (r.get("internal_region") or "").strip()
            n += 1
            # область: позвоночник против бедра — строго; сторона бедра — отдельно
            group_true = "spine" if region_true == "spine" else "hip"
            group_meas = "spine" if region_meas == "spine" else "hip"
            if group_true != group_meas:
                problems.append(f"{key}: область {region_true} против {region_meas}")
            elif group_true == "hip" and region_true != region_meas:
                # сторона проверяется строго, включая обрезанное поле: с версии фантомов 1.1
                # и правила crop_aware (src/hip_features.py) детектор даёт 8/8 верных сторон
                problems.append(f"{key}: сторона {region_true} против {region_meas}"
                                + (" (обрезанное поле)" if variant == "cropped_field" else ""))
            if region_true == "spine":
                ang = num(r, "feat_axis_angle_deg")
                off = num(r, "feat_center_offset_ratio")
                tilt = float(g.get("axis_tilt_deg", 0.0) or 0.0)
                if ang is None:
                    problems.append(f"{key}: нет feat_axis_angle_deg")
                elif abs(tilt) >= 5.0:
                    if ang * tilt <= 0:
                        problems.append(f"{key}: знак наклона {ang:.1f} против истины {tilt:.1f}")
                    elif abs(abs(ang) - abs(tilt)) > 6.0:
                        problems.append(f"{key}: наклон {ang:.1f} против истины {tilt:.1f} (>6°)")
                elif variant == "shifted" and abs(ang) > 3.0:
                    # до версии фантомов 1.1 здесь было известное отклонение 5.7°: генератор
                    # сдвигал позвоночник, но не силуэт тела. Теперь проверяется строго.
                    problems.append(f"{key}: боковой сдвиг даёт наклон оси {ang:.1f}° при истине 0°")
                elif abs(ang) > 3.0:
                    problems.append(f"{key}: наклон {ang:.1f} при истине {tilt:.1f}")
                if off is not None:
                    if variant == "shifted" and abs(off) < 0.05:
                        problems.append(f"{key}: сдвиг центра {off:.3f} < 0.05 при истине «shifted»")
                    if variant == "normal" and abs(off) > 0.03:
                        problems.append(f"{key}: сдвиг центра {off:.3f} > 0.03 при истине «normal»")
            else:
                if variant == "cropped_field":
                    # Вариант «обрезанное поле» с версии фантомов 1.1 моделирует латерально
                    # сдвинутое поле сканирования: силуэт тела упирается в край кадра, кость
                    # подходит к краю поля (в среднем 3–5 мм), но таз остаётся со своей стороны —
                    # иначе сторону нельзя определить в принципе (см. METRICS_REPORT, п.8).
                    edge = num(r, "feat_edge_distance_mm")
                    if edge is None or edge > 6.0:
                        problems.append(f"{key}: обрезанное поле, край {edge} мм > 6.0")
        if n == 0:
            return name, False, "ни один фантом не сопоставлен с MANIFEST.json"
        det = f"сверено {n} фантомов по истинной геометрии"
        if notes:
            det += "; известное отклонение — " + "; ".join(notes)
        return name, not problems, det if not problems else "; ".join(problems[:6])
    except Exception as e:  # noqa: BLE001
        return name, False, f"проверка не выполнена: {e}"

def sc_series_check(phantoms) -> tuple:
    """Бонус ТЗ 2.6 «серия с визуализацией»: DICOM Secondary Capture с оверлеем должен быть
    производным изображением со ссылкой на исходный снимок, с предупреждающей надписью в
    пикселях и с детерминированными UID (повторный прогон не создаёт новую серию в PACS)."""
    name = "Серия с визуализацией (DICOM SC): производное изображение, ссылка на источник, детерминированные UID"
    try:
        from geometry_features import read_dicom_normalized, extract_all_features
        from visualize_report import render_overlay, overlay_to_dicom_sc, DISCLAIMER_TEXT
        src = sorted(Path(phantoms).glob("study_01/CR000000.dcm"))
        if not src:
            src = sorted(Path(phantoms).rglob("CR*.dcm"))[:1]
        if not src:
            return name, False, "нет фантомов для проверки"
        fp = str(src[0])
        img_u8, ref = read_dicom_normalized(fp)
        feats = extract_all_features(fp, "spine")
        ov = render_overlay(img_u8, "spine", feats, None, 1, "Нарушение оси позвоночника (отклонение более 5 градусов)")
        d1 = overlay_to_dicom_sc(ov, ref, model_version="test", config_hash="deadbeef")
        d2 = overlay_to_dicom_sc(ov, ref, model_version="test", config_hash="deadbeef")
        it = list(getattr(d1, "ImageType", []))
        seq = getattr(d1, "SourceImageSequence", None)
        problems = []
        if it[:2] != ["DERIVED", "SECONDARY"]:
            problems.append(f"ImageType={it}")
        if getattr(d1, "BurnedInAnnotation", "") != "YES":
            problems.append("нет BurnedInAnnotation=YES")
        if not DISCLAIMER_TEXT.strip():
            problems.append("пустая предупреждающая надпись")
        if ov.shape[0] <= img_u8.shape[0]:
            problems.append("нет панели с подписями под снимком")
        if not seq or str(seq[0].ReferencedSOPInstanceUID) != str(getattr(ref, "SOPInstanceUID", "")):
            problems.append("нет ссылки на исходный SOP Instance UID")
        if d1.SOPInstanceUID != d2.SOPInstanceUID or d1.SeriesInstanceUID != d2.SeriesInstanceUID:
            problems.append("UID не детерминированы между прогонами")
        if (getattr(d1, "SamplesPerPixel", 0), getattr(d1, "BitsAllocated", 0)) != (3, 8):
            problems.append("не RGB 8 бит")
        det = (f"SC {d1.Rows}x{d1.Columns}, ImageType={it}, ссылка на {str(getattr(ref, 'SOPInstanceUID', ''))[:24]}…, "
               f"UID повторяем, надпись в пикселях есть")
        return name, not problems, det if not problems else "; ".join(problems)
    except Exception as e:  # noqa: BLE001
        return name, False, f"проверка не выполнена: {e}"


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

    # 7b. К11: вариант предобработки в config.yaml == тому, с которым обучались модели.
    # Без этой проверки расхождение конфига и pkl тихо даёт неправильные предсказания:
    # признаки считались бы на одном варианте кадра, а модель ждала бы другого.
    add(*preproc_consistency_check())

    # 7b2. К13: источник эмбеддингов в config.yaml == тому, на котором обучена логрегрессия контура B,
    # и файл весов этого бэкбона лежит в models/ (иначе контур B тихо отключится в образе).
    add(*emb_source_consistency_check())

    # 7c. Словарь организаторов: их скрипт сверяет строки буквально, поэтому расхождение
    # в одном символе обнуляет macro-F1 по типам нарушений при верном классе.
    add(*official_dictionary_check())

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

    # 10. одна версия и один config_hash во всех файлах поставки
    ok_ver, det_ver = version_consistency()
    add("Версия и config_hash одинаковы во всех файлах поставки", ok_ver, det_ver)

    # 11. изоляция запросов и пределы распаковки архива (негативные тесты)
    add("Безопасность API: изоляция запросов по коду, закрытый список, пределы архива", *api_security_check())

    # 15. инвариантность к форме подачи (переименование, порядок, zip)
    add(*transfer_invariance_check(Path(a.transfer) if getattr(a, "transfer", "") else None))

    # 16. истинная геометрия фантомов из MANIFEST.json
    dbg = Path(a.debug_csv) if getattr(a, "debug_csv", "") else Path(a.run1).parent / "results_debug.csv"
    add(*phantom_ground_truth_check(phantoms, dbg))

    # 17. бонус ТЗ 2.6: серия с визуализацией как DICOM SC
    add(*sc_series_check(phantoms))

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


# места, где версия и config_hash продублированы текстом. Ключ — файл, значение — список
# (регулярное выражение с одной группой, что за значение). Файл, которого нет, пропускается.


def version_consistency():
    """Версия и config_hash из config.yaml против тех же значений, вписанных в документы,
    сборочные скрипты и веб. Список мест — один на всю поставку, он же используется при сборке
    релиза (tools/check_version_consistency.py). Расхождение версий между образом и инструкцией
    проверки ломает саму проверку поставки, поэтому это ошибка, а не замечание."""
    import sys as _sys

    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        return False, "нет config.yaml"
    _sys.path.insert(0, str(ROOT / "src"))
    _sys.path.insert(0, str(ROOT / "tools"))
    try:
        from inference import load_config, config_hash  # noqa: E402

        cfg = load_config()
        version, chash = str(cfg.get("version", "")).strip(), config_hash(cfg)
    except Exception as e:  # noqa: BLE001
        return False, f"не удалось прочитать config.yaml: {e}"
    try:
        import check_version_consistency as cvc  # noqa: E402
    except Exception as e:  # noqa: BLE001
        return False, f"не найден список мест с версией: {e}"

    bad = (cvc.scan(cvc.VERSION_PLACES, "v", version, missing_ok=True)
           + cvc.scan(cvc.HASH_PLACES, "h", chash, missing_ok=True))
    n = sum(1 for rel, _ in cvc.VERSION_PLACES + cvc.HASH_PLACES if (ROOT / rel).exists())
    detail = ("; ".join(bad) if bad else
              f"версия {version}, config_hash {chash} — совпадают в {n} файлах, доступных в образе")
    return not bad, detail


def api_security_check():
    """Негативные тесты доступа: результаты одного запроса не видны другому, список всех
    запросов закрыт, «zip-бомба» отклоняется. Запускается отдельным процессом, потому что
    подменяет DENSITO_OUTPUT_DIR и импортирует api_server со своими настройками."""
    import subprocess as _sp
    import sys as _sys
    test = ROOT / "tests" / "test_api_security.py"
    if not test.exists():
        return False, f"нет файла {test}"
    env = dict(os.environ)
    env.pop("DENSITO_ADMIN_KEY", None)
    try:
        r = _sp.run([_sys.executable, str(test)], capture_output=True, text=True, timeout=300,
                    cwd=str(ROOT), env=env)
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    out = (r.stdout or "") + (r.stderr or "")
    n_ok = out.count("  OK   ")
    if r.returncode == 0 and "ALL API SECURITY CHECKS PASSED" in out:
        return True, f"пройдено {n_ok} негативных проверок (доступ по коду запроса, закрытый список, пределы архива)"
    bad = [ln.strip() for ln in out.splitlines() if ln.startswith("  FAIL")]
    return False, "; ".join(bad)[:400] or out[-400:]


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
    c.add_argument("--transfer", default="", help="JSON от tools/transfer_check.py")
    c.add_argument("--debug-csv", dest="debug_csv", default="", help="debug CSV прогона 1")
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
