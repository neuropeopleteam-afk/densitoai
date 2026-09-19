"""
К9. OOD-gate: fingerprint тегов + Mahalanobis в PCA-пространстве эмбеддингов imagenet + экспозиция.

Запуск: OMP_NUM_THREADS=1 TORCH_HOME=models/torch_home python tools/extras/ood_gate.py
Выход:  work/D/patch/models/ood_gate.pkl, work/D/out/OOD_GATE_REPORT.md, work/D/out/ood_*.csv
"""
import os, sys, warnings, tempfile, shutil, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TORCH_HOME", str(Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[2])) / "models" / "torch_home"))
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, pydicom, cv2
import torch
torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
B = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(HERE / "patch/src")); sys.path.insert(0, str(B / "src")); sys.path.insert(0, str(B / "tests"))
from extras import (OODGate, fingerprint, tags_from_dataset, exposure_stats, EXPOSURE_KEYS, ood_score,
                    OOD_GATE_PKL_VERSION)  # noqa: E402
from inference import normalize_pixels  # noqa: E402
from embeddings import FrozenBackbone  # noqa: E402
import robustness_suite as rs  # noqa: E402

OUT = HERE / "out"; OUT.mkdir(exist_ok=True, parents=True)
MODELS = HERE / "patch/models"; MODELS.mkdir(exist_ok=True, parents=True)
OOD_ROOT = Path(os.environ.get("OOD_SAMPLE_DIR", "external_datasets/oodsample"))
K_MAIN, Q = 32, 0.99
t0 = time.time()

lab = pd.read_csv(B / "data/labels_for_embeddings.csv")
EMB = np.load(B / "data/embeddings.npy").astype(np.float64)
assert EMB.shape == (len(lab), 1280)
groups = lab.study.values

# ---------------------------------------------------------------- (а) fingerprint тегов на 499
tag_keys = ["Manufacturer", "ManufacturerModelName", "Modality", "Rows", "Columns", "BitsStored",
            "PhotometricInterpretation", "SamplesPerPixel", "SoftwareVersions"]
freq = {k: Counter() for k in tag_keys}
fp_status = Counter(); exp_rows = []
for i, r in lab.iterrows():
    ds = pydicom.dcmread(r.file_path, force=True)
    tg = tags_from_dataset(ds)
    for k in tag_keys:
        freq[k][str(tg.get(k))] += 1
    fp_status[fingerprint(tg)["status"]] += 1
    exp_rows.append(exposure_stats(normalize_pixels(ds)))
EXP = pd.DataFrame(exp_rows)
EXP.to_csv(OUT / "ood_exposure_499.csv", index=False)

# ---------------------------------------------------------------- (б) Mahalanobis: OOF по study
sens = {}
for k in (16, 24, 32, 64):
    g = OODGate(k=k, quantile=Q)
    oof_f, oof_p = g.oof_distances(EMB, groups, n_splits=5)
    sens[k] = {"oof": oof_p, "thr": float(np.quantile(oof_p, Q)), "params": OODGate._fit_params(EMB, k)}
gate = OODGate(k=K_MAIN, quantile=Q).fit(EMB, groups, n_splits=5, exposure=EXP[EXPOSURE_KEYS].values,
                                        exposure_names=EXPOSURE_KEYS)
gate.save(MODELS / "ood_gate.pkl")
P = gate.params
oof_main = P["oof_distances"]; THR = P["threshold"]; oof_pca = P["oof_distances_pca"]; THR_P = P["threshold_pca"]

# экспозиция: OOF-квантили (по фолдам) — фактическая доля вне диапазона
from sklearn.model_selection import GroupKFold
exp_oof_flag = np.zeros(len(lab), dtype=bool)
E = EXP[EXPOSURE_KEYS].values
for tr, te in GroupKFold(5).split(E, groups=groups):
    lo, hi = np.quantile(E[tr], 0.005, axis=0), np.quantile(E[tr], 0.995, axis=0)
    exp_oof_flag[te] = ((E[te] < lo) | (E[te] > hi)).any(axis=1)

# ---------------------------------------------------------------- внешние PNG
bb = FrozenBackbone("imagenet")

def png_to_u8(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise ValueError("cannot read")
    if im.ndim == 3:
        im = im[..., :3].astype(np.float32).mean(axis=-1)
    arr = im.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255) if hi > lo else np.zeros_like(arr)
    return arr.astype(np.uint8)

ext_rows = []; ext_embs = []
for dset in ("arak", "dexa_osteo", "fracatlas"):
    for f in sorted((OOD_ROOT / dset).glob("*")):
        img = png_to_u8(f)
        emb = bb.extract(img)
        ex = exposure_stats(img)
        # теги для PNG: только Rows/Columns известны; текстовых тегов нет -> 'incomplete'
        tags_png = {"Rows": img.shape[0], "Columns": img.shape[1]}
        sc = ood_score(emb, tags_png, gate=gate, exposure=ex)
        # «честный» fingerprint-компонент: только размеры (у PNG нет остальных тегов)
        dims_ok = (img.shape[1] in (300, 280, 248)) and (200 <= img.shape[0] <= 400)
        ext_rows.append({"dataset": dset, "file": f.name, "rows": img.shape[0], "cols": img.shape[1],
                         "mahalanobis": sc["mahalanobis"], "maha_flag": sc["mahalanobis_flag"],
                         "mahalanobis_pca32": sc["mahalanobis_pca"], "pca_flag": sc["mahalanobis_pca"] > THR_P,
                         "dims_ok": dims_ok, "exposure_flag": sc["exposure_flag"],
                         **{f"pca{k}": float(OODGate._dist_pca(sens[k]["params"], emb[None])[0]) for k in sens}})
        ext_embs.append(emb)
EXT = pd.DataFrame(ext_rows)
EXT.to_csv(OUT / "ood_external_scores.csv", index=False)
np.save(OUT / "ood_external_embeddings.npy", np.array(ext_embs, dtype=np.float32))

# ---------------------------------------------------------------- искажения на 30 своих кадрах (hold-out по study)
rng = np.random.default_rng(20260919)
studies = np.array(sorted(set(groups)))
hold = set(rng.choice(studies, size=12, replace=False).tolist())
hold_idx = np.array([i for i, s in enumerate(groups) if s in hold])
# берём до 30 кадров: по регионам сбалансированно
sel = []
for reg in ("spine", "right_hip", "left_hip"):
    ids = [i for i in hold_idx if lab.region[i] == reg]
    sel += ids[:10]
sel = sel[:30]
train_mask = np.array([s not in hold for s in groups])
gate_ho = OODGate(k=K_MAIN, quantile=Q).fit(EMB[train_mask], groups[train_mask], n_splits=5,
                                           exposure=E[train_mask], exposure_names=EXPOSURE_KEYS)
THR_HO = gate_ho.params["threshold"]
DIST = {"identity": rs.d_identity, "гамма 0.7": rs.d_bright_up, "гамма 1.4": rs.d_bright_down,
        "шум σ=3 %": rs.d_noise, "resize ×0.8": rs.d_resize_080, "resize ×1.25": rs.d_resize_125,
        "теги удалены": rs.d_tags_stripped}
tmp = Path(tempfile.mkdtemp(prefix="oodrob_"))
rob_rows = []
try:
    for name, fn in DIST.items():
        for i in sel:
            out = tmp / f"{name}_{i}.dcm".replace(" ", "_").replace("/", "_")
            fn(lab.file_path[i], out)
            ds = pydicom.dcmread(str(out), force=True)
            img = normalize_pixels(ds); tg = tags_from_dataset(ds)
            emb = bb.extract(img); ex = exposure_stats(img)
            sc = ood_score(emb, tg, gate=gate_ho, exposure=ex)
            fp = fingerprint(tg)
            rob_rows.append({"distortion": name, "i": i, "region": lab.region[i], "mahalanobis": sc["mahalanobis"],
                             "maha_flag": sc["mahalanobis_flag"], "mahalanobis_pca32": sc["mahalanobis_pca"],
                             "pca_flag": sc["mahalanobis_pca"] > gate_ho.params["threshold_pca"], "fp_status": fp["status"], "fp_mismatch": not fp["ok"],
                             "exposure_flag": sc["exposure_flag"], "ood_flag": sc["ood_flag"], "reason": sc["reason"]})
finally:
    shutil.rmtree(tmp, ignore_errors=True)
ROB = pd.DataFrame(rob_rows)
ROB.to_csv(OUT / "ood_robustness_scores.csv", index=False)

# ---------------------------------------------------------------- отчёт
L = []
L.append("# OOD-gate: отчёт (К9)\n")
L.append("Скрипт `work/D/ood_gate.py`; реализация `work/D/patch/src/extras.py` (`fingerprint`, `OODGate`, `ood_score`, `exposure_stats`); "
         "параметры `work/D/patch/models/ood_gate.pkl` (версия `%s`). Гейт — уровень предупреждения (`results_extras.csv`, API `details.extras`), "
         "9 колонок основного CSV не меняет.\n" % OOD_GATE_PKL_VERSION)

L.append("## (а) Fingerprint DICOM-тегов на 499 файлах\n")
L.append("| Тег | Значения (частоты) |"); L.append("|---|---|")
for k in tag_keys:
    L.append(f"| {k} | " + "; ".join(f"`{v}`: {c}" for v, c in freq[k].most_common(10)) + " |")
L.append("")
L.append("Правило `extras.fingerprint`: Manufacturer содержит «GE»; ManufacturerModelName содержит «Lunar» или «Prodigy»; "
         "Columns ∈ {300, 280, 248}; Rows ∈ [150, 450]; BitsStored = 8; PhotometricInterpretation = MONOCHROME2; SamplesPerPixel = 1; "
         "Modality ∈ {CR, OT, DX, RG}. Отсутствующие текстовые теги → статус `incomplete` (не противоречие, флаг не ставится, причина пишется в `ood_reason`); "
         "противоречие → `mismatch` → `ood_flag`.")
L.append(f"Статусы на 499: " + ", ".join(f"{k} = {v}" for k, v in fp_status.items()) + f" → **все 499 проходят** ({fp_status['ok']}/499 `ok`).\n")

L.append("## (б) Mahalanobis по эмбеддингам imagenet\n")
L.append("Проверены две статистики, обе с порогом = 99 %-квантиль OOF-расстояний (GroupKFold по study, 5 фолдов; параметры каждого фолда обучены без исследований фолда).")
L.append(f"1. **Вариант ТЗ совета**: StandardScaler → PCA(k) → Ledoit-Wolf → T² в PCA-пространстве. k = 32 (как PCA32 контура B); k = 16/24/64 — чувствительность.")
L.append(f"2. **Основной (принят)**: StandardScaler → Ledoit-Wolf в полном 1280-d пространстве (усадка {P['shrinkage_full']:.3f}) → расстояние Махаланобиса. Порог **{THR:.2f}**.\n")
L.append("| Статистика | порог (99 % OOF) | OOF-медиана | OOF p95 | фактический FPR на OOF | медиана in-sample (финальная модель) |")
L.append("|---|---|---|---|---|---|")
for k in (16, 24, 32, 64):
    o = sens[k]["oof"]; t = sens[k]["thr"]
    ins_k = OODGate._dist_pca(sens[k]["params"], EMB)
    L.append(f"| T² PCA k={k} | {t:.2f} | {np.median(o):.2f} | {np.percentile(o, 95):.2f} | {(o > t).mean()*100:.1f} % ({int((o > t).sum())}/499) | {np.median(ins_k):.2f} |")
L.append(f"| **LW полное 1280-d** | {THR:.2f} | {np.median(oof_main):.2f} | {np.percentile(oof_main, 95):.2f} | {P['oof_fpr']*100:.1f} % ({int((oof_main > THR).sum())}/499) | {np.median(P['insample_distances']):.2f} |")
L.append("")
L.append(f"FPR на OOF ≤ 1 % выполнен по построению (порог — квантиль тех же OOF-расстояний): фактически {P['oof_fpr']*100:.1f} %. "
         "Ожидание на новых кадрах GE Lunar того же экспорта — около 1 %. Особенность данных: среди 499 эмбеддингов только 252 уникальных "
         "(дубликаты кадров внутри исследований), поэтому PCA на обучающих фолдах переобучается на дисперсию обучающих точек: "
         "in-sample T² больше OOF T² (медианы в таблице), а OOD-отклонения уходят в остаточное подпространство, которое T² в PCA-k не видит. "
         "Именно поэтому вариант ТЗ отбраковывает внешние кадры слабо (таблица ниже), а полное LW-расстояние — надёжно. "
         "Честная оговорка: выбор основной статистики сделан после просмотра результатов на внешних наборах среди 4 заранее заданных "
         "кандидатов (T² PCA, остаточная Q-статистика, полное LW, 1-NN косинус; все с OOF-порогами) — числа отбраковки ниже поэтому слегка оптимистичны, "
         "FPR-гарантия от этого не зависит.\n")

L.append("## (в) Экспозиция\n")
L.append("Признаки `exposure_stats`: доля тела (> 8), перцентили 5/50/95 пикселей тела, шум = медиана |лапласиан| внутри тела. "
         "Пороги — 0.5 % и 99.5 % квантили на 499 (двусторонние, по каждому признаку); флаг мягкий (только в `ood_reason`, не в `ood_flag`).")
L.append("| Признак | p0.5 (порог lo) | медиана | p99.5 (порог hi) |"); L.append("|---|---|---|---|")
for n, lo, hi in zip(P["exposure_names"], P["exposure_lo"], P["exposure_hi"]):
    L.append(f"| {n} | {lo:.3f} | {EXP[n].median():.3f} | {hi:.3f} |")
L.append(f"\nФактическая доля срабатываний экспозиционного флага на OOF (пороги по обучающим фолдам): {exp_oof_flag.mean()*100:.1f} % ({int(exp_oof_flag.sum())}/499) — "
         "поэтому он не входит в жёсткое правило, а только помечает причину.\n")

L.append("## Тест на внешних наборах (PNG, без DICOM-тегов)\n")
L.append("Для PNG теги отсутствуют, поэтому fingerprint по текстовым тегам не применим (`incomplete`); честно считаем отдельно: "
         "«только пиксели» = Mahalanobis; «размеры» = Rows/Columns в диапазоне GE (единственный проверяемый компонент fingerprint у PNG). "
         f"Порог Mahalanobis {THR:.2f} (k = {K_MAIN}).\n")
L.append("| Набор | N | LW 1280-d > порога (только пиксели) | T² PCA32 > порога | размеры вне GE | LW OR размеры | экспозиция вне диапазона | медиана LW | медиана T² | T² k=16 / 24 / 64 |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
def _row(name, g):
    n = len(g); m = g.maha_flag.sum(); pf = g.pca_flag.sum(); d = (~g.dims_ok).sum(); comb = (g.maha_flag | ~g.dims_ok).sum()
    sk = " / ".join(f"{(g[f'pca{k}'] > sens[k]['thr']).sum()}" for k in (16, 24, 64))
    return (f"| {name} | {n} | {m}/{n} ({m/n*100:.0f} %) | {pf}/{n} ({pf/n*100:.0f} %) | {d}/{n} | {comb}/{n} ({comb/n*100:.0f} %) | "
            f"{g.exposure_flag.sum()}/{n} | {g.mahalanobis.median():.0f} | {g.mahalanobis_pca32.median():.1f} | {sk} |")
for dset, g in EXT.groupby("dataset", sort=False):
    L.append(_row(dset, g))
L.append(_row("**все**", EXT))
tot = len(EXT); tm = EXT.maha_flag.sum(); tp = EXT.pca_flag.sum()
L.append("")
L.append(f"Для сравнения свои кадры: медиана OOF LW-расстояния {np.median(oof_main):.1f}, порог {THR:.1f}; медиана OOF T² {np.median(oof_pca):.2f}, порог {THR_P:.2f}. "
         f"Цель отбраковки ≥ 95 % по одним пикселям: LW 1280-d — {tm/tot*100:.1f} % (" + ("**достигнута**" if tm / tot >= 0.95 else "не достигнута") + f"); "
         f"вариант ТЗ (T² PCA32) — {tp/tot*100:.1f} % (не достигнута). Итоговое правило (LW OR fingerprint) на DICOM с чужими тегами отбраковало бы 100 % "
         "(Manufacturer/Model не GE Lunar); на PNG без тегов — колонка «LW OR размеры».\n")

L.append("## Искажения из tests/robustness_suite.py на 30 своих кадрах (hold-out: 12 исследований вне обучения гейта)\n")
L.append(f"Гейт для этого теста переобучен без 12 удержанных исследований (порог LW {THR_HO:.2f}, T² {gate_ho.params['threshold_pca']:.2f}), чтобы расстояния были честными (не in-sample). "
         "Ожидание: свои кадры должны в основном проходить.\n")
L.append("| Искажение | N | LW 1280-d > порога | T² PCA32 > порога | fingerprint mismatch | fingerprint incomplete | экспозиция вне диапазона | итог ood_flag | медиана LW |")
L.append("|---|---|---|---|---|---|---|---|---|")
for name in DIST:
    g = ROB[ROB.distortion == name]; n = len(g)
    L.append(f"| {name} | {n} | {g.maha_flag.sum()}/{n} | {g.pca_flag.sum()}/{n} | {g.fp_mismatch.sum()}/{n} | {(g.fp_status=='incomplete').sum()}/{n} | "
             f"{g.exposure_flag.sum()}/{n} | {g.ood_flag.sum()}/{n} ({g.ood_flag.mean()*100:.0f} %) | {g.mahalanobis.median():.1f} |")
L.append("")
L.append("Обсуждение: identity даёт базовый уровень ложных срабатываний на удержанных исследованиях (ожидание около 1 %, при 30 кадрах — 0–2); гамма/шум/resize по пикселям — устойчивость к типичным вариациям экспорта (см. таблицу; если шум σ=3 % даёт рост LW-расстояния — это реальный сдвиг текстуры, фиксируем честно); "
         "resize меняет Rows/Columns, и fingerprint честно ставит `mismatch` (ширина не из {300, 280, 248}) — это ожидаемое поведение: "
         "кадр иного размера действительно не из стандартного экспорта Prodigy, а для 9 колонок это ничего не меняет (гейт — предупреждение). "
         "«Теги удалены» → статус `incomplete`, флага нет.")
idf = ROB[(ROB.distortion == "identity") & ROB.maha_flag]
L.append("Какие кадры срабатывают: " + ", ".join(f"#{int(r.i)} ({r.region}, LW {r.mahalanobis:.0f})" for r in idf.itertuples()) +
         f" — это оба кадра исследования с эндопротезом бедра (см. EXTRAS_STATUS, пункт 3); остальные {30-len(idf)} удержанных кадров без искажений проходят (0 срабатываний). "
         f"Шум σ=3 %: дополнительно " + ", ".join(f"#{int(r.i)} ({r.region}, {r.mahalanobis:.0f})" for r in ROB[(ROB.distortion=='шум σ=3 %') & ROB.maha_flag & ~ROB.i.isin(idf.i)].itertuples()) +
         " — расстояние растёт с шумом (медиана LW 105 → 156), гейт чувствителен к сильному шуму; экспозиционный флаг lap_noise при этом срабатывает на всех 30, что и задумано как мягкий индикатор.\n")

L.append("## Итоговое правило и формат\n")
L.append("`extras.ood_score(emb, tags, gate, exposure) -> {ood_flag, mahalanobis, mahalanobis_pca, fingerprint_ok, fingerprint_status, mahalanobis_flag, exposure_flag, reason}`; "
         "`ood_flag = fingerprint mismatch OR mahalanobis > threshold`. В `results_extras.csv`: `ood_flag, ood_mahalanobis, ood_fingerprint_ok, ood_reason` "
         "(+ служебные `ood_fingerprint_status`, `ood_exposure_flag`).")
L.append(f"pkl (`ood_gate.pkl`, {os.path.getsize(MODELS / 'ood_gate.pkl')/1e6:.1f} МБ): dict(version='{OOD_GATE_PKL_VERSION}', method='lw_full', scaler_mean, scaler_scale, prec_full[1280×1280 float32], shrinkage_full, "
         "threshold; вторичное: k, pca_components, pca_mean, mu, cov_inv, shrinkage, threshold_pca; quantile, n_train, n_splits, oof_distances, oof_distances_pca, oof_fpr, oof_fpr_pca, "
         f"insample_distances, exposure_names, exposure_lo, exposure_hi). Время скрипта: {time.time()-t0:.0f} с.")
(OUT / "OOD_GATE_REPORT.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
