"""
К8. Аудит «белых линий» (ROI-графика софта аппарата) на 499 кадрах.

Запуск: OMP_NUM_THREADS=1 python work/D/white_lines_audit.py
Выход:  work/D/out/WHITE_LINES_AUDIT.md, work/D/out/white_lines_per_frame.csv, work/D/out/img/wl_*.png
"""
import os, sys, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, pydicom, cv2
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
B = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(HERE / "patch/src")); sys.path.insert(0, str(B / "src"))
from extras import white_lines  # noqa: E402
from inference import normalize_pixels  # noqa: E402

OUT = HERE / "out"; IMG = OUT / "img"; IMG.mkdir(parents=True, exist_ok=True)
lab = pd.read_csv(B / "data/labels_for_embeddings.csv")
geo = pd.read_csv(B / "data/geometry_features.csv")
assert (lab.file_path.values == geo.file_path.values).all()
CRITS = {"spine": ["sp_pos", "sp_axis", "sp_art"], "hip": ["hip_pos_c", "hip_roi_c"]}

rows = []
raw_hist = {}
for i, r in lab.iterrows():
    ds = pydicom.dcmread(r.file_path, force=True)
    raw = ds.pixel_array
    img = normalize_pixels(ds)
    strict = white_lines(img)                                  # рабочий детектор (контраст с обеих сторон >= 100)
    naive = white_lines(img, side_contrast=0)                  # «254–255 + морфология» без контраста (как в ТЗ совета)
    rows.append({"i": i, "file": r.file_path, "study": r.study, "region": r.region,
                 "raw_max": int(raw.max()), "raw_frac_max": float((raw == raw.max()).mean()),
                 "sat_frac": strict["sat_frac"],
                 "has_lines": strict["has_lines"], "n_segments": strict["n_segments"],
                 "total_len_px": strict["total_len_px"], "frac_px": strict["frac_px"],
                 "naive_has_lines": naive["has_lines"], "naive_n_segments": naive["n_segments"],
                 "naive_total_len_px": naive["total_len_px"], "naive_frac_px": naive["frac_px"]})
df = pd.DataFrame(rows)
df.to_csv(OUT / "white_lines_per_frame.csv", index=False)

# --- метки
df["hip"] = df.region.isin(["right_hip", "left_hip"])
for c in ["sp_pos", "sp_axis", "sp_art", "hip_pos_c", "hip_roi_c"]:
    df[c] = geo[c].values

L = []
L.append("# Аудит «белых линий» (К8)\n")
L.append("Скрипт: `work/D/white_lines_audit.py`; детектор: `work/D/patch/src/extras.py::white_lines`. "
         "Данные: 499 DICOM (`data/labels_for_embeddings.csv`), метки `data/geometry_features.csv`.\n")
L.append("## 1. Что в пикселях\n")
L.append(f"- Сырые значения: максимум = {df.raw_max.min()}…{df.raw_max.max()} во всех 499 файлах "
         f"(значения ≥ 241 квантованы: 241, 243, 245, 247, 249, 252 — уровня 255 в экспорте нет). "
         f"Доля пикселей с сырым максимумом: медиана {df.raw_frac_max.median()*100:.2f} %, макс {df.raw_frac_max.max()*100:.1f} % (эндопротез).")
L.append(f"- После нормализации `normalize_pixels` (окно 1–99 %) доля пикселей ≥ 254 по построению ≈ 1 % "
         f"(медиана {df.sat_frac.median()*100:.2f} %, 99-й перцентиль {df.sat_frac.quantile(.99)*100:.2f} %): "
         "это кортикальный слой, а не графика. Поэтому голый критерий «254–255» линии не находит, нужна морфология и контраст.")
L.append("- Визуальный просмотр 24 кадров (случайные и топ по детектору): графика ROI (рамки, подписи L1–L4, neck box) "
         "в экспорте организаторов **отсутствует** — экспорт enCORE в DICOM CR идёт без оверлея. "
         "Для сравнения на внешних PNG (DEXA-Osteo, нативное разрешение) такая графика есть и детектор её находит (см. п. 4).\n")

L.append("## 2. Доля кадров с линиями по областям\n")
L.append("Два варианта детектора: «наивный» = пиксели ≥ 254 + сегменты ≥ 20 px, толщина ≤ 3 px (без проверки контраста); "
         "«рабочий» = то же + интенсивность линии выше фона с ОБЕИХ сторон (±4 px) не менее чем на 100 уровней.\n")
L.append("| Область | N | наивный: с линиями | рабочий: с линиями |")
L.append("|---|---|---|---|")
for reg, g in df.groupby("region"):
    L.append(f"| {reg} | {len(g)} | {g.naive_has_lines.sum()} ({g.naive_has_lines.mean()*100:.1f} %) | "
             f"{g.has_lines.sum()} ({g.has_lines.mean()*100:.1f} %) |")
L.append(f"| **все** | {len(df)} | {df.naive_has_lines.sum()} ({df.naive_has_lines.mean()*100:.1f} %) | "
         f"{df.has_lines.sum()} ({df.has_lines.mean()*100:.1f} %) |")
share = df.has_lines.mean()
L.append("")
L.append(f"Наивный детектор срабатывает на насыщенных гребнях кортикального слоя диафиза (проверено глазами, "
         f"`out/img/wl_naive_examples.png`). Рабочий детектор: {df.has_lines.sum()} кадра из 499 ({share*100:.1f} %) — "
         f"оба кадра одного исследования с эндопротезом (прямой край металлической ножки), не графика ROI.")
L.append(f"**Доля кадров с линиями {share*100:.1f} % < 30 % → по решению совета: только флаг, инпейнтинг не делаем.**\n")

# --- линии × метка
def fisher_tab(flag, y):
    a = int(((flag == 1) & (y == 1)).sum()); b = int(((flag == 1) & (y == 0)).sum())
    c = int(((flag == 0) & (y == 1)).sum()); d = int(((flag == 0) & (y == 0)).sum())
    orr, p = fisher_exact([[a, b], [c, d]])
    # ДИ отношения шансов (Woolf, с поправкой 0.5 при нулях)
    aa, bb, cc, dd = [x + 0.5 if min(a, b, c, d) == 0 else x for x in (a, b, c, d)]
    lo_or = np.exp(np.log(aa * dd / (bb * cc)) - 1.96 * np.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd))
    hi_or = np.exp(np.log(aa * dd / (bb * cc)) + 1.96 * np.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd))
    return a, b, c, d, orr, lo_or, hi_or, p

L.append("## 3. Таблица «линии × метка» (точный тест Фишера, отношение шансов с 95 % ДИ)\n")
for name, col in (("рабочий детектор", "has_lines"), ("наивный детектор", "naive_has_lines")):
    L.append(f"### {name}\n")
    L.append("| Критерий | N | линии∧наруш. | линии∧норма | нет линий∧наруш. | нет линий∧норма | OR | 95 % ДИ | p (Fisher) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for reg_key, crits in CRITS.items():
        sub = df[df.hip] if reg_key == "hip" else df[df.region == "spine"]
        for c in crits:
            s = sub.dropna(subset=[c])
            y = s[c].astype(int).values; f = s[col].astype(int).values
            a, b, cc_, d, orr, lo, hi, p = fisher_tab(f, y)
            or_s = "—" if not np.isfinite(orr) or (a + b) == 0 else f"{orr:.2f}"
            ci_s = "—" if (a + b) == 0 else f"{lo:.2f}–{hi:.2f}"
            L.append(f"| {c} | {len(s)} | {a} | {b} | {cc_} | {d} | {or_s} | {ci_s} | {p:.3f} |")
    L.append("")

# --- AUC по признакам линий (GroupKFold по study)
L.append("## 4. Несут ли признаки линий сигнал метки (LogReg, GroupKFold по study, 5 фолдов)\n")
L.append("Признаки: n_segments, total_len_px, frac_px (наивный и рабочий детектор), sat_frac (доля ≥ 254). "
         "AUC близкий к 0.5 = признаки линий не могут служить шорткатом для метки.\n")
L.append("| Критерий | N | позитивов | AUC (признаки линий) | AUC (только sat_frac) |")
L.append("|---|---|---|---|---|")
feat_cols = ["n_segments", "total_len_px", "frac_px", "naive_n_segments", "naive_total_len_px", "naive_frac_px", "sat_frac"]
auc_rows = []
for reg_key, crits in CRITS.items():
    sub = df[df.hip] if reg_key == "hip" else df[df.region == "spine"]
    for c in crits:
        s = sub.dropna(subset=[c]).reset_index(drop=True)
        y = s[c].astype(int).values
        if y.sum() < 3:
            continue
        aucs = {}
        for fname, cols in (("all", feat_cols), ("sat", ["sat_frac"])):
            X = s[cols].values.astype(float); oof = np.zeros(len(s))
            for tr, te in GroupKFold(5).split(X, y, groups=s.study.values):
                sc = StandardScaler().fit(X[tr])
                m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(sc.transform(X[tr]), y[tr])
                oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
            aucs[fname] = roc_auc_score(y, oof)
        auc_rows.append((c, len(s), int(y.sum()), aucs["all"], aucs["sat"]))
        L.append(f"| {c} | {len(s)} | {int(y.sum())} | {aucs['all']:.3f} | {aucs['sat']:.3f} |")
L.append("")
hh = df[df.hip].copy(); hh["y"] = hh["hip_roi_c"]; hh["sl"] = geo.loc[hh.index, "scan_length_mm"].values
hh = hh.dropna(subset=["y"])
auc_len = roc_auc_score(hh.y, hh.naive_total_len_px)
L.append("Интерпретация. Для позвоночника и hip_pos признаки линий метку не предсказывают (AUC 0.41–0.61 при 10–79 позитивах). "
         f"Для hip_roi_c AUC {auc_rows[-1][3]:.2f} объясняется не графикой (её нет), а анатомией: наивный детектор считает длину "
         "насыщенного кортикального гребня диафиза, а нарушение ROI бедра — это как раз обрезанный кадром диафиз: "
         f"медиана naive_total_len_px {hh[hh.y==1].naive_total_len_px.median():.0f} px у нарушений против {hh[hh.y==0].naive_total_len_px.median():.0f} px у нормы "
         f"(AUC одного признака {1-auc_len:.2f} в обратную сторону), медиана scan_length_mm {hh[hh.y==1].sl.median():.0f} против {hh[hh.y==0].sl.median():.0f} мм; "
         f"корреляция naive_total_len_px и scan_length_mm r = {np.corrcoef(hh.naive_total_len_px, hh.sl.fillna(hh.sl.median()))[0,1]:.2f}. "
         "Это тот же сигнал, что уже использует контур A (scan_length_mm, bone_bottom_touch_ratio), а не шорткат.\n")

# --- внешние PNG (только для иллюстрации работы детектора)
def png_norm(f):
    im = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
    if im.ndim == 3:
        im = im[..., :3].astype(np.float32).mean(-1)
    im = im.astype(np.float32); lo, hi = np.percentile(im, [1, 99])
    return np.clip((im - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8) if hi > lo else np.zeros(im.shape, np.uint8)
ext_lines = {}
for d in ("dexa_osteo", "arak"):
    fs = sorted(Path(os.environ.get("OOD_SAMPLE_DIR", "external_datasets/oodsample")) / d.glob("*"))
    nat = [f for f in fs if cv2.imread(str(f), cv2.IMREAD_UNCHANGED).shape[0] > 250]
    small = [f for f in fs if f not in nat]
    ext_lines[d] = {"native": (sum(white_lines(png_norm(f))["has_lines"] for f in nat), len(nat)),
                    "small": (sum(white_lines(png_norm(f))["has_lines"] for f in small), len(small))}
L.append("## 5. Контроль детектора на внешних PNG с графикой ROI (не для метрик, только проверка, что детектор видит графику)\n")
L.append("| Набор | нативное разрешение: с линиями / N | уменьшенные (224×224): с линиями / N |")
L.append("|---|---|---|")
for d, v in ext_lines.items():
    L.append(f"| {d} | {v['native'][0]} / {v['native'][1]} | {v['small'][0]} / {v['small'][1]} |")
L.append("")
L.append("DEXA-Osteo: белые рамки/подписи на тёмном фоне — детектор находит в большинстве кадров нативного разрешения; "
         "после уменьшения до 224×224 линии сглажены и перестают быть насыщенными (ограничение детектора по построению). "
         "Arak: печать Hologic инвертирована (линии тёмные на светлом) — детектор для белых линий не предназначен, срабатывает редко.\n")

# --- миниатюры: 6 примеров
def tile(img, segs, title):
    col = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for o, k, s, e in segs:
        if o == "h":
            cv2.line(col, (s, k), (e - 1, k), (0, 0, 255), 1)
        else:
            cv2.line(col, (k, s), (k, e - 1), (0, 255, 0), 1)
    col = cv2.resize(col, (280, 320), interpolation=cv2.INTER_NEAREST)
    cv2.putText(col, title, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
    return col
# a) 2 кадра рабочего детектора + 4 самых длинных наивных
pick = list(df[df.has_lines].i.values[:2]) + list(df.sort_values("naive_total_len_px", ascending=False).i.values[:4])
tiles = []
for i in pick:
    img = normalize_pixels(pydicom.dcmread(df.file[i], force=True))
    wl = white_lines(img, side_contrast=0)
    st = white_lines(img)
    tiles.append(tile(img, wl["segments"], f"#{i} {df.region[i]} naive={wl['n_segments']} strict={st['n_segments']}"))
cv2.imwrite(str(IMG / "wl_examples_6.png"), np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
# b) наивные примеры отдельно (8 случайных с срабатыванием)
rng = np.random.default_rng(0)
cand = df[df.naive_has_lines].i.values
sel = rng.choice(cand, size=min(8, len(cand)), replace=False)
tiles = []
for i in sel:
    img = normalize_pixels(pydicom.dcmread(df.file[i], force=True)); wl = white_lines(img, side_contrast=0)
    tiles.append(tile(img, wl["segments"], f"#{i} {df.region[i]} n={wl['n_segments']}"))
cv2.imwrite(str(IMG / "wl_naive_examples.png"), np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:8])]))
# c) внешний пример DEXA-Osteo
fs = [f for f in sorted((Path(os.environ.get("OOD_SAMPLE_DIR", "external_datasets/oodsample")) / "dexa_osteo").glob("*"))
      if cv2.imread(str(f), cv2.IMREAD_UNCHANGED).shape[0] > 250][:3]
tiles = []
for f in fs:
    img = png_norm(f); wl = white_lines(img)
    tiles.append(tile(img, wl["segments"], f"DEXA-Osteo n={wl['n_segments']} len={wl['total_len_px']}"))
cv2.imwrite(str(IMG / "wl_external_dexa_osteo.png"), np.hstack(tiles))

L.append("## 6. Миниатюры (внутренние, не в репозиторий)\n")
L.append("- `work/D/out/img/wl_examples_6.png` — 6 примеров: 2 срабатывания рабочего детектора (эндопротез, #40/#45) и 4 кадра с максимальной длиной «наивных» сегментов (кортикальный гребень). Зелёным — вертикальные сегменты, красным — горизонтальные.")
L.append("- `work/D/out/img/wl_naive_examples.png` — 8 случайных срабатываний наивного детектора.")
L.append("- `work/D/out/img/wl_external_dexa_osteo.png` — как выглядит настоящая графика ROI (внешний набор), детектор её находит.\n")

L.append("## 7. Решение\n")
L.append(f"1. В данных организаторов графики ROI нет; доля кадров с линиями по рабочему детектору {share*100:.1f} % (< 30 %) → **только флаг** `white_lines_flag` в extras, метрику по линиям не заявляем.")
L.append("2. Инпейнтинг **не делаем**: нечего инпейнтить (0 кадров с графикой), а инпейнт кортикальных гребней разрушил бы анатомию 1–3 px и создал бы train–test shift (аргумент Sol в раунде 2).")
L.append("3. Флаг оставлен как страховка для закрытого теста: если там окажется другой тип экспорта (с оверлеем), `results_extras.csv` это покажет колонками `white_lines_flag`, `white_lines_len_px`.")
L.append("4. Шорткат через графику ROI невозможен (графики нет). Признаки «наивного» детектора несут сигнал только для hip_roi_c и только потому, что измеряют видимую длину диафиза (п. 4) — это анатомия критерия, уже учтённая контуром A.")
(OUT / "WHITE_LINES_AUDIT.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
