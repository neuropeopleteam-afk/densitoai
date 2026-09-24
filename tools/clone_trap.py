#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""clone_trap.py — «ловушка клонов»: сколько даёт случайное разбиение по файлам при побайтных клонах кадров.

В выборке 499 файлов, но уникальных кадров (хэш пикселей) 252: экспорт PACS кладёт в исследование копии одного
снимка. Если разбить файлы случайно, копия кадра из теста почти всегда есть в обучении, и модель «узнаёт» кадр
вместо того, чтобы оценивать качество. Скрипт показывает это на одной и той же модели и одних и тех же признаках:

  * признаки — числовые колонки data/geometry_features.csv, строго без колонок меток и идентификаторов
    (DROP_EXACT, суффикс _c, всё, что содержит class/violation/applicable); список проверяется: ни одна
    оставшаяся колонка не совпадает с мишенью;
  * мишени — y_true пяти критериев из models/oof_stacked_*.csv (метка стороны детектора, как в поставке);
  * модель — RandomForestClassifier(300 деревьев, n_jobs=1), одна и та же в обоих разбиениях;
  * разбиение A — StratifiedKFold(5, shuffle) по файлам; разбиение B — StratifiedGroupKFold(5, shuffle) по группам
    «исследование + хэш пикселей» (связные компоненты: файлы одного исследования или с одинаковым хэшем);
  * 10 сидов (сид задаёт и разбиение, и лес); метрика — ROC-AUC OOF на файлах.

Модели и пороги сервиса не используются и не меняются: это демонстрация протокола валидации.

  python tools/clone_trap.py            # docs/clone_trap.json, docs/clone_trap.md, docs/clone_trap.png
  python tools/clone_trap.py --seeds 2  # быстрый прогон
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
CRITERIA = [("spine", "sp_pos"), ("spine", "sp_axis"), ("spine", "sp_art"), ("hip", "hip_pos"), ("hip", "hip_roi")]
DROP_EXACT = {"sp_pos", "sp_axis", "sp_art", "rh_pos", "rh_roi", "lh_pos", "lh_roi", "quality_class",
              "instance_number", "rows", "cols", "applicable"}


def feature_columns(g: pd.DataFrame) -> list[str]:
    num = g.select_dtypes("number").columns
    keep = []
    for c in num:
        lc = c.lower()
        if c in DROP_EXACT or lc.endswith("_c") or "class" in lc or "violation" in lc or "applicable" in lc:
            continue
        keep.append(c)
    return keep


def groups_study_hash(study: pd.Series, h: pd.Series) -> np.ndarray:
    """Связные компоненты графа «файл — исследование — хэш»."""
    parent: dict = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for s, hh in zip(study, h):
        a, b = find(("s", s)), find(("h", hh))
        if a != b:
            parent[a] = b
    roots = [find(("s", s)) for s in study]
    _, inv = np.unique([str(r) for r in roots], return_inverse=True)
    return inv


def oof_auc(X, y, splits, seed) -> tuple[float, np.ndarray]:
    oof = np.zeros(len(y))
    for tr, te in splits:
        m = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof)), oof


def run(seeds: int) -> dict:
    geom = pd.read_csv(ROOT / "data" / "geometry_features.csv", low_memory=False)
    hashes = pd.read_csv(ROOT / "docs" / "k5" / "pixel_hashes.csv")[["file_path", "pixel_hash"]]
    cols = feature_columns(geom)
    out = {"model": "RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=seed)",
           "features": {"n": len(cols), "source": "data/geometry_features.csv, числовые колонки без меток",
                        "dropped_rule": sorted(DROP_EXACT) + ["*_c", "*class*", "*violation*", "*applicable*"]},
           "split_A": "StratifiedKFold(5, shuffle=True, random_state=seed) по файлам",
           "split_B": "StratifiedGroupKFold(5, shuffle=True, random_state=seed), группы = исследование + хэш пикселей",
           "seeds": list(range(seeds)), "criteria": {}}
    for region, crit in CRITERIA:
        o = pd.read_csv(ROOT / "models" / f"oof_stacked_{region}_{crit}.csv")[["file_path", "study", "y_true"]]
        d = o.merge(geom[["file_path"] + cols], on="file_path", how="left", validate="one_to_one")
        d = d.merge(hashes, on="file_path", how="left")
        d["pixel_hash"] = d["pixel_hash"].fillna(d["file_path"])
        y = d["y_true"].astype(int).to_numpy()
        X = d[cols].astype(float).fillna(-999.0).to_numpy()
        # страховка от утечки: ни один признак не повторяет мишень
        for j, c in enumerate(cols):
            assert not np.array_equal(np.nan_to_num(X[:, j]), y.astype(float)), f"признак {c} совпадает с мишенью"
        grp = groups_study_hash(d["study"].astype(str), d["pixel_hash"])
        res = {"n": int(len(y)), "n_pos": int(y.sum()), "n_groups": int(len(np.unique(grp))),
               "n_unique_frames": int(d["pixel_hash"].nunique()), "A_random_files": [], "B_grouped": [],
               "A_test_rows_with_clone_in_train": []}
        for s in range(seeds):
            spA = list(StratifiedKFold(5, shuffle=True, random_state=s).split(X, y))
            leak = 0
            for tr, te in spA:
                tr_h = set(d["pixel_hash"].iloc[tr])
                leak += int(d["pixel_hash"].iloc[te].isin(tr_h).sum())
            res["A_test_rows_with_clone_in_train"].append(leak / len(y))
            res["A_random_files"].append(oof_auc(X, y, spA, s)[0])
            spB = list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, groups=grp))
            for tr, te in spB:
                assert not set(grp[tr]) & set(grp[te])
            res["B_grouped"].append(oof_auc(X, y, spB, s)[0])
        a, b = np.array(res["A_random_files"]), np.array(res["B_grouped"])
        res["summary"] = {"auc_A_mean": float(a.mean()), "auc_A_sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                          "auc_B_mean": float(b.mean()), "auc_B_sd": float(b.std(ddof=1)) if len(b) > 1 else 0.0,
                          "delta_mean": float((a - b).mean()), "delta_min": float((a - b).min()),
                          "delta_max": float((a - b).max()),
                          "share_test_rows_with_clone_in_train_A": float(np.mean(res["A_test_rows_with_clone_in_train"]))}
        out["criteria"][crit] = res
        print(f"{crit}: A {a.mean():.3f}±{res['summary']['auc_A_sd']:.3f}  B {b.mean():.3f}±{res['summary']['auc_B_sd']:.3f}"
              f"  Δ {res['summary']['delta_mean']:+.3f}  клонов теста в обучении (A) "
              f"{res['summary']['share_test_rows_with_clone_in_train_A']:.0%}", flush=True)
    return out


def to_markdown(res: dict) -> str:
    L = ["# Ловушка клонов: случайное разбиение по файлам против группового (`tools/clone_trap.py`)", "",
         f"Модель: {res['model']}; признаков {res['features']['n']} ({res['features']['source']}). "
         f"A — {res['split_A']}; B — {res['split_B']}. Сидов {len(res['seeds'])}; ROC-AUC OOF по файлам, среднее ± sd.", "",
         "| Критерий | n / позитивов | Групп | Уникальных кадров | AUC A (файлы) | AUC B (группы) | Δ A − B (мин–макс) | Строк теста с клоном в обучении (A) |",
         "|---|---|---|---|---|---|---|---|"]
    for c, r in res["criteria"].items():
        s = r["summary"]
        L.append(f"| `{c}` | {r['n']} / {r['n_pos']} | {r['n_groups']} | {r['n_unique_frames']} | "
                 f"{s['auc_A_mean']:.3f} ± {s['auc_A_sd']:.3f} | {s['auc_B_mean']:.3f} ± {s['auc_B_sd']:.3f} | "
                 f"{s['delta_mean']:+.3f} ({s['delta_min']:+.3f}…{s['delta_max']:+.3f}) | "
                 f"{s['share_test_rows_with_clone_in_train_A']:.0%} |")
    L += ["", "Модели и пороги сервиса здесь не участвуют; это проверка протокола валидации на одной модели.", ""]
    return "\n".join(L)


def plot(res: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    crits = list(res["criteria"])
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=150)
    for i, c in enumerate(crits):
        r = res["criteria"][c]
        for off, key, col, lab in ((-0.15, "A_random_files", "#b5452c", "A: случайно по файлам"),
                                   (0.15, "B_grouped", "#2c6fb5", "B: по исследованию + хэшу")):
            v = np.array(r[key])
            ax.scatter(np.full(len(v), i + off) + np.linspace(-0.05, 0.05, len(v)), v, s=12, color=col, alpha=0.7,
                       label=lab if i == 0 else None)
            ax.hlines(v.mean(), i + off - 0.1, i + off + 0.1, color=col, lw=2)
    ax.axhline(0.5, color="#888", lw=0.8, ls=":")
    ax.set_xticks(range(len(crits)), crits)
    ax.set_ylabel("ROC-AUC OOF")
    ax.set_ylim(0.3, 1.02)
    ax.set_title(f"Одна модель, одни признаки, {len(res['seeds'])} сидов: цена клонов кадров", fontsize=10)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out-json", type=Path, default=ROOT / "docs" / "clone_trap.json")
    ap.add_argument("--out-md", type=Path, default=ROOT / "docs" / "clone_trap.md")
    ap.add_argument("--out-png", type=Path, default=ROOT / "docs" / "clone_trap.png")
    ap.add_argument("--plot-only", action="store_true",
                    help="только нарисовать PNG по готовому --out-json (в venv проекта нет matplotlib: python3 tools/clone_trap.py --plot-only)")
    a = ap.parse_args(argv)
    if a.plot_only:
        plot(json.loads(a.out_json.read_text(encoding="utf-8")), a.out_png)
        print(f"OK: {a.out_png}")
        return 0
    t0 = time.time()
    res = run(a.seeds)
    print(f"время {time.time() - t0:.0f} с")
    a.out_json.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    a.out_md.write_text(to_markdown(res), encoding="utf-8")
    print(to_markdown(res))
    try:
        plot(res, a.out_png)
    except ModuleNotFoundError:
        print("matplotlib не установлен: PNG — командой python3 tools/clone_trap.py --plot-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
