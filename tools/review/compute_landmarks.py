#!/usr/bin/env python
"""
Микроэталон ориентиров: экспорт врача (landmarks.json или landmarks.csv из landmarks_form.html)
против того, что реально выдаёт существующая геометрия проекта (см. landmark_geometry.py).

Метрики (мм; PixelSpacing из тега DICOM, иначе 1.05 x 0.6 мм — константа аппарата из config.yaml):
  Позвоночник
    axis_top, axis_bottom  -> горизонтальное расстояние от точки врача до линии оси кода в той же
                              строке (|dx|*sx). Код не находит L1/L4, поэтому сравнение «точка–линия».
    axis (две точки)       -> |угол оси врача - угол оси кода| в градусах (обе в мм).
    body_left/body_right   -> |dx|*sx до крайнего пикселя МАСКИ кости в той же строке.
  Бедро
    gt_apex     -> евклидово расстояние до точки максимального латерального выступа над диафизом
                   (то, из чего код берёт greater_troch_offset_mm). Это не «верхушка», сопоставление приближённое.
    lt          -> евклидово расстояние до пика медиального выступа (lesser_troch_prominence_mm);
                   у кода пик найден не на всех кадрах — метрика по покрытым.
    head_center -> в коде НЕТ соответствующей точки. Метрика не считается, только фиксируется.
    shaft_axis  -> горизонтальное расстояние от точки врача до оси диафиза кода в той же строке.
  Для каждого сопоставления: MRE (среднее и медиана, мм), PCK@10 мм, бутстрап-ДИ 95 % по кадрам (2000 ресемплов).

Использование:
  python compute_landmarks.py --export landmarks.json [--key out/landmarks_key.csv] [--out landmarks_metrics.md]
  python compute_landmarks.py --synthetic     # проверка на синтетическом экспорте
"""
import argparse
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import json
import os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(os.environ.get("DENSITO_ROOT", HERE.parents[1])) / "src"))
import landmark_geometry as lg  # noqa: E402
from inference import normalize_pixels  # noqa: E402

PCK_MM = 10.0
N_BOOT = 2000
SEED = 20260919
DEFAULT_SPACING = (1.05, 0.6)  # (row, col) мм


def load_export(path: Path) -> dict:
    """-> {display_id: {'points': {name: (x, y)}, 'skipped': bool}}"""
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        frames = data["frames"] if isinstance(data, dict) else data
        out = {}
        for f in frames:
            pts = {k: (float(v["x"]), float(v["y"])) for k, v in (f.get("points") or {}).items()}
            out[f["display_id"]] = {"points": pts, "skipped": bool(f.get("skipped"))}
        return out
    df = pd.read_csv(path)
    out = {}
    for did, d in df.groupby("display_id"):
        pts = {r.landmark: (float(r.x), float(r.y)) for r in d.itertuples() if pd.notna(r.x) and pd.notna(r.y)}
        out[did] = {"points": pts, "skipped": bool(d.skipped.astype(str).str.lower().eq("true").any())}
    return out


def read_image(fp: str):
    ds = pydicom.dcmread(fp, force=True)
    img = normalize_pixels(ds)
    sp = None
    for tag in ("PixelSpacing", "ImagerPixelSpacing"):
        v = getattr(ds, tag, None)
        if v is not None:
            try:
                sp = (float(v[0]), float(v[1]))
                break
            except Exception:  # noqa: BLE001
                pass
    return img, (sp or DEFAULT_SPACING), sp is not None


def boot_ci(vals, fn, rng, n_boot=N_BOOT):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    stats = np.array([fn(vals[i]) for i in idx])
    return tuple(np.nanpercentile(stats, [2.5, 97.5]))


def evaluate(export: dict, key: pd.DataFrame):
    rows = []      # per (frame, landmark) errors
    angles = []    # spine axis angle differences
    coverage = {}
    n_tag = 0
    for r in key.itertuples():
        e = export.get(r.display_id)
        if e is None or e["skipped"] or not e["points"]:
            continue
        img, (sy, sx), has_tag = read_image(r.file_path)
        n_tag += has_tag
        P = e["points"]
        if r.area == "spine":
            g = lg.spine_code_geometry(img)
            if not g["ok"]:
                continue
            for k in ("axis_top", "axis_bottom"):
                if k in P:
                    x, y = P[k]
                    rows.append((r.display_id, "spine", k, abs(x - lg.spine_axis_x_at(g, y)) * sx, "dx_to_axis_line"))
            if "axis_top" in P and "axis_bottom" in P:
                (x1, y1), (x2, y2) = P["axis_top"], P["axis_bottom"]
                if abs(y2 - y1) > 1:
                    ang_doc = np.degrees(np.arctan2((x2 - x1) * sx, (y2 - y1) * sy))
                    angles.append((r.display_id, ang_doc, g["signed_angle_deg"], abs(ang_doc - g["signed_angle_deg"]),
                                   abs(abs(ang_doc) - g["axis_angle_deg"])))
            for k, side in (("body_left", 0), ("body_right", 1)):
                if k in P:
                    x, y = P[k]
                    edges = lg.spine_mask_edges_at(g, y)
                    if edges[side] is not None:
                        rows.append((r.display_id, "spine", k, abs(x - edges[side]) * sx, "dx_to_mask_edge"))
        else:
            g = lg.hip_code_geometry(img)
            if not g["ok"]:
                continue
            if "shaft_axis" in P:
                x, y = P["shaft_axis"]
                rows.append((r.display_id, "hip", "shaft_axis", abs(x - lg.hip_shaft_x_at(g, y)) * sx, "dx_to_shaft_line"))
            for k, cp in (("gt_apex", g["gt_point"]), ("lt", g["lt_point"])):
                coverage.setdefault(k, [0, 0])
                if k in P:
                    coverage[k][1] += 1
                    if cp is not None:
                        coverage[k][0] += 1
                        x, y = P[k]
                        d = np.hypot((x - cp[0]) * sx, (y - cp[1]) * sy)
                        rows.append((r.display_id, "hip", k, d, "euclid_to_code_point"))
            if "head_center" in P:
                coverage.setdefault("head_center", [0, 0])
                coverage["head_center"][1] += 1
    df = pd.DataFrame(rows, columns=["display_id", "area", "landmark", "err_mm", "kind"])
    ang = pd.DataFrame(angles, columns=["display_id", "angle_doc_deg", "angle_code_signed_deg", "abs_diff_signed_deg",
                                        "abs_diff_unsigned_deg"])
    return df, ang, coverage, n_tag


NAMES = {
    "axis_top": "Позвоночник: верхняя точка оси (L1) — расстояние до линии оси кода",
    "axis_bottom": "Позвоночник: нижняя точка оси (L4) — расстояние до линии оси кода",
    "body_left": "Позвоночник: левый край тела — до края маски кости в строке",
    "body_right": "Позвоночник: правый край тела — до края маски кости в строке",
    "gt_apex": "Бедро: верхушка большого вертела — до точки макс. латерального выступа кода (приближённо)",
    "lt": "Бедро: малый вертел — до пика медиального выступа кода",
    "shaft_axis": "Бедро: точка оси диафиза — расстояние до оси диафиза кода",
}


def report(df, ang, coverage, n_frames_used, n_tag, n_frames_key, out_md: Path):
    rng = np.random.default_rng(SEED)
    lines = ["# Микроэталон ориентиров: врач против существующей геометрии", "",
             f"Кадров в ключе: {n_frames_key}; с точками врача и успешной геометрией: {n_frames_used}; "
             f"PixelSpacing из тега: {n_tag} кадров, остальные — константа 1,05 × 0,6 мм.", "",
             f"Метрика: MRE — средняя ошибка (мм), PCK@{PCK_MM:.0f} — доля ошибок ≤ {PCK_MM:.0f} мм; "
             f"ДИ 95 % — бутстрап по кадрам, {N_BOOT} ресемплов.", "",
             "| Ориентир | Способ сопоставления | n | MRE, мм [ДИ] | медиана, мм | PCK@10 мм [ДИ] |",
             "|---|---|---|---|---|---|"]
    order = ["axis_top", "axis_bottom", "body_left", "body_right", "gt_apex", "lt", "shaft_axis"]
    summary = {}
    for k in order:
        d = df[df.landmark == k]
        if len(d) == 0:
            lines.append(f"| {NAMES[k]} | — | 0 | нет данных | — | — |")
            continue
        v = d.err_mm.to_numpy()
        mre, ci_m = v.mean(), boot_ci(v, np.mean, rng)
        pck, ci_p = (v <= PCK_MM).mean(), boot_ci(v, lambda a: (a <= PCK_MM).mean(), rng)
        lines.append(f"| {NAMES[k]} | {d.kind.iloc[0]} | {len(v)} | {mre:.1f} [{ci_m[0]:.1f}; {ci_m[1]:.1f}] | "
                     f"{np.median(v):.1f} | {pck:.2f} [{ci_p[0]:.2f}; {ci_p[1]:.2f}] |")
        summary[k] = {"n": int(len(v)), "mre_mm": float(mre), "mre_ci": [float(c) for c in ci_m],
                      "median_mm": float(np.median(v)), "pck10": float(pck), "pck10_ci": [float(c) for c in ci_p]}
    hc = coverage.get("head_center", [0, 0])[1]
    lines.append(f"| Бедро: центр головки бедренной кости | нет соответствующей точки в коде | {hc} | не считается | — | — |")
    lines += ["", "## Угол оси позвоночника (две точки врача против линейной аппроксимации центроидов кода)", ""]
    if len(ang):
        v = ang.abs_diff_signed_deg.to_numpy()
        ci = boot_ci(v, np.mean, rng)
        lines.append(f"n = {len(v)}; средняя |разность знаковых углов| = {v.mean():.2f}° [{ci[0]:.2f}; {ci[1]:.2f}], "
                     f"медиана {np.median(v):.2f}°; доля кадров с разностью ≤ 2° = {(v <= 2).mean():.2f}, ≤ 5° = {(v <= 5).mean():.2f}.")
        summary["axis_angle"] = {"n": int(len(v)), "mae_deg": float(v.mean()), "mae_ci": [float(c) for c in ci],
                                 "share_le2": float((v <= 2).mean()), "share_le5": float((v <= 5).mean())}
    else:
        lines.append("нет кадров с обеими точками оси.")
    lines += ["", "## Покрытие точек кода (бедро)", ""]
    for k, (have, tot) in coverage.items():
        if k == "head_center":
            lines.append(f"- head_center: точек врача {tot}; в коде центр головки не вычисляется — метрика не считается.")
        else:
            lines.append(f"- {k}: точек врача {tot}, у кода точка найдена на {have} кадрах; метрика по {have}.")
    lines += ["", "## Оговорки", "",
              "- Код проекта агрегирует геометрию в скалярные признаки и не выдаёт анатомические ориентиры как таковые; "
              "здесь восстановлены промежуточные точки/линии тех же вычислений (scripts/landmark_geometry.py).",
              "- Для L1/L4 сравнение «точка–линия» по горизонтали: код не разделяет позвонки.",
              "- «Край тела» сравнивается с краем маски Otsu, куда попадают поперечные отростки и рёбра; ожидаемо большая ошибка — это свойство маски, а не разметки.",
              "- gt_apex сопоставляется с точкой максимального латерального выступа — она ниже верхушки; PCK по нему — нижняя оценка."]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def make_synthetic(key: pd.DataFrame, rng, noise_mm=4.0) -> dict:
    """Синтетический экспорт: точки кода + шум (где точки есть), иначе случайные точки в кадре."""
    export = {}
    for r in key.itertuples():
        img, (sy, sx), _ = read_image(r.file_path)
        h, w = img.shape
        pts = {}
        if r.area == "spine":
            g = lg.spine_code_geometry(img)
            y1, y2 = 0.2 * h, 0.8 * h
            if g["ok"]:
                pts["axis_top"] = (lg.spine_axis_x_at(g, y1) + rng.normal(0, noise_mm / sx), y1)
                pts["axis_bottom"] = (lg.spine_axis_x_at(g, y2) + rng.normal(0, noise_mm / sx), y2)
                ym = 0.5 * h
                l, rr = lg.spine_mask_edges_at(g, ym)
                if l is not None:
                    pts["body_left"] = (l + rng.normal(0, noise_mm / sx), ym)
                    pts["body_right"] = (rr + rng.normal(0, noise_mm / sx), ym)
            for k in ("axis_top", "axis_bottom", "body_left", "body_right"):
                pts.setdefault(k, (rng.uniform(0, w - 1), rng.uniform(0, h - 1)))
        else:
            g = lg.hip_code_geometry(img)
            if g["ok"]:
                yb = min(h - 5, g["shaft_rows"][1] - 5)
                pts["shaft_axis"] = (lg.hip_shaft_x_at(g, yb) + rng.normal(0, noise_mm / sx), yb)
                if g["gt_point"]:
                    pts["gt_apex"] = (g["gt_point"][0] + rng.normal(0, noise_mm / sx), g["gt_point"][1] + rng.normal(0, noise_mm / sy))
                if g["lt_point"]:
                    pts["lt"] = (g["lt_point"][0] + rng.normal(0, noise_mm / sx), g["lt_point"][1] + rng.normal(0, noise_mm / sy))
            for k in ("gt_apex", "lt", "head_center", "shaft_axis"):
                pts.setdefault(k, (rng.uniform(0, w - 1), rng.uniform(0, h - 1)))
        skipped = rng.random() < 0.05
        export[r.display_id] = {"points": {k: (round(v[0], 1), round(v[1], 1)) for k, v in pts.items()}, "skipped": skipped}
    return export


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, help="landmarks.json или landmarks.csv от врача")
    ap.add_argument("--key", type=Path, default=HERE.parent / "out" / "landmarks_key.csv")
    ap.add_argument("--out", type=Path, default=None, help="markdown-отчёт")
    ap.add_argument("--synthetic", action="store_true", help="сгенерировать синтетический экспорт и проверить скрипт")
    args = ap.parse_args()
    key = pd.read_csv(args.key)
    rng = np.random.default_rng(SEED)
    if args.synthetic:
        export = make_synthetic(key, rng)
        synth_path = HERE.parent / "tmp" / "synthetic_landmarks.json"
        synth_path.parent.mkdir(exist_ok=True)
        synth_path.write_text(json.dumps({"frames": [
            {"display_id": k, "skipped": v["skipped"], "points": {n: {"x": p[0], "y": p[1]} for n, p in v["points"].items()}}
            for k, v in export.items()]}, ensure_ascii=False, indent=1), encoding="utf-8")
        print("синтетический экспорт:", synth_path)
        out_md = args.out or HERE.parent / "tmp" / "synthetic_landmarks_metrics.md"
    else:
        if not args.export:
            ap.error("--export обязателен (или --synthetic)")
        export = load_export(args.export)
        out_md = args.out or args.export.with_name("landmarks_metrics.md")
    df, ang, cov, n_tag = evaluate(export, key)
    n_used = df.display_id.nunique()
    summary = report(df, ang, cov, n_used, n_tag, len(key), out_md)
    df.to_csv(out_md.with_suffix(".per_point.csv"), index=False)
    print(out_md.read_text(encoding="utf-8"))
    print("per-point CSV:", out_md.with_suffix(".per_point.csv"))


if __name__ == "__main__":
    main()
