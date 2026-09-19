#!/usr/bin/env python
"""
Согласие рентгенолога с разметкой и с моделью по экспорту слепой галереи.

Вход:  экспорт врача (review_answers.json или review_answers.csv из review_gallery.html) + out/review_key.csv.
Выход: markdown-таблица (+ JSON со всеми числами).

Что считается (по каждому критерию: sp_pos, sp_axis, sp_art, hip_pos, hip_roi):
  * врач vs метка разметки (train-метки, geometry_features.csv): доля согласия, Cohen kappa, бутстрап-ДИ 95 %
    (ресемплинг по кадрам, 2000 повторов), чувствительность/специфичность врача относительно метки;
  * врач vs предсказание модели (OOF pred_label, models/oof_stacked_*.csv): то же;
  * раздельно: все уникальные кадры, «случайные» (30), «расхождения» (20); повторы в этих блоках не участвуют;
  * intra-reader: согласие врача с самим собой по 5 повторам (доля совпавших ответов, pooled kappa);
  * распределение ответа «согласны с моделью?» (да/нет/частично) по подмножествам.
Ответ «не уверен» исключается из kappa (учитывается отдельно как доля).

Использование:
  python compute_agreement.py --export review_answers.json [--key out/review_key.csv] [--out agreement.md]
  python compute_agreement.py --synthetic
"""
import argparse
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CRITS = ["sp_pos", "sp_axis", "sp_art", "hip_pos", "hip_roi"]
LABEL_COL = {"sp_pos": "sp_pos", "sp_axis": "sp_axis", "sp_art": "sp_art", "hip_pos": "hip_pos_c", "hip_roi": "hip_roi_c"}
CRIT_RU = {"sp_pos": "Позвоночник: укладка", "sp_axis": "Позвоночник: ось", "sp_art": "Позвоночник: посторонние предметы",
           "hip_pos": "Бедро: укладка", "hip_roi": "Бедро: область интереса"}
ANS = {"yes": 1.0, "no": 0.0, "unsure": np.nan, "": np.nan}
N_BOOT = 2000
SEED = 20260919


def load_export(path: Path) -> pd.DataFrame:
    """-> DataFrame: display_id, <crit> (1/0/NaN), unsure_<crit> (bool), agree_model, fixed"""
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data["answers"] if isinstance(data, dict) else data
        recs = []
        for r in rows:
            rec = {"display_id": r["display_id"], "agree_model": r.get("agree_model"), "fixed": bool(r.get("fixed_at"))}
            for c in CRITS:
                a = (r.get("answers") or {}).get(c, "")
                rec[c] = ANS.get(a, np.nan)
                rec[f"unsure_{c}"] = a == "unsure"
            recs.append(rec)
        return pd.DataFrame(recs)
    df = pd.read_csv(path, dtype=str).fillna("")
    out = pd.DataFrame({"display_id": df.display_id, "agree_model": df.agree_model.replace("", None),
                        "fixed": df.fixed_at != ""})
    for c in CRITS:
        out[c] = df[c].map(lambda a: ANS.get(a, np.nan))
        out[f"unsure_{c}"] = df[c] == "unsure"
    return out


def cohen_kappa(a, b):
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    n = len(a)
    if n == 0:
        return np.nan
    po = (a == b).mean()
    pe = sum(((a == k).mean()) * ((b == k).mean()) for k in (0, 1))
    if pe >= 1.0:   # оба ряда константны — kappa не определена
        return np.nan
    return (po - pe) / (1 - pe)


def boot_ci(a, b, fn, rng, n_boot=N_BOOT):
    a = np.asarray(a); b = np.asarray(b)
    n = len(a)
    if n < 2:
        return (np.nan, np.nan)
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = np.array([fn(a[i], b[i]) for i in idx], dtype=float)
    return tuple(np.nanpercentile(stats, [2.5, 97.5]))


def pair_stats(doc, ref, rng):
    """doc, ref — 0/1 без NaN."""
    doc = np.asarray(doc, dtype=int); ref = np.asarray(ref, dtype=int)
    n = len(doc)
    if n == 0:
        return {"n": 0}
    agr = float((doc == ref).mean())
    k = cohen_kappa(doc, ref)
    k_ci = boot_ci(doc, ref, cohen_kappa, rng)
    a_ci = boot_ci(doc, ref, lambda x, y: (x == y).mean(), rng)
    pos = ref == 1
    sens = float((doc[pos] == 1).mean()) if pos.any() else np.nan
    spec = float((doc[~pos] == 0).mean()) if (~pos).any() else np.nan
    return {"n": int(n), "n_ref_pos": int(pos.sum()), "n_doc_pos": int(doc.sum()), "agreement": agr,
            "agreement_ci": [float(a_ci[0]), float(a_ci[1])], "kappa": float(k), "kappa_ci": [float(k_ci[0]), float(k_ci[1])],
            "sensitivity_vs_ref": sens, "specificity_vs_ref": spec}


def fmt(s):
    if s.get("n", 0) == 0:
        return "| — | — | — | — | — |"
    def f(x):
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"
    return (f"| {s['n']} ({s['n_ref_pos']} поз.) | {f(s['agreement'])} [{f(s['agreement_ci'][0])}; {f(s['agreement_ci'][1])}] "
            f"| {f(s['kappa'])} [{f(s['kappa_ci'][0])}; {f(s['kappa_ci'][1])}] | {f(s['sensitivity_vs_ref'])} | {f(s['specificity_vs_ref'])} |")


def analyse(exp: pd.DataFrame, key: pd.DataFrame):
    rng = np.random.default_rng(SEED)
    m = key.merge(exp, on="display_id", how="left", suffixes=("", "_doc"))
    m = m[m.fixed.fillna(False).astype(bool)]
    uniq = m[~m.is_repeat.astype(bool)]
    subsets = {"все уникальные": uniq, "случайные": uniq[~uniq.is_discrepancy.astype(bool)],
               "расхождения": uniq[uniq.is_discrepancy.astype(bool)]}
    res = {"n_fixed": int(len(m)), "n_shown": int(len(key)), "subsets": {}, "unsure": {}, "intra": {}, "agree_model": {}}
    for sname, d in subsets.items():
        res["subsets"][sname] = {}
        for c in CRITS:
            doc = d[f"{c}_doc"] if f"{c}_doc" in d else d[c]  # колонка врача
            lab = d[LABEL_COL[c]]
            mod = d[f"oof_{c}"]
            ok_lab = doc.notna() & lab.notna()
            ok_mod = doc.notna() & mod.notna()
            res["subsets"][sname][c] = {
                "vs_label": pair_stats(doc[ok_lab], lab[ok_lab], rng),
                "vs_model": pair_stats(doc[ok_mod], mod[ok_mod], rng),
                "n_unsure": int(d[f"unsure_{c}"].fillna(False).sum()),
                "n_applicable": int(lab.notna().sum()),
            }
        # согласие с моделью (ответ врача после показа)
        am = d.agree_model.dropna()
        res["agree_model"][sname] = {k: int((am == k).sum()) for k in ("yes", "no", "partial")}
        res["agree_model"][sname]["n"] = int(len(am))
    # intra-reader по повторам
    rep = m[m.is_repeat.astype(bool)]
    pairs_doc, pairs_rep, per_crit = [], [], {}
    for r in rep.itertuples():
        orig = m[m.display_id == r.repeat_of_id]
        if len(orig) == 0:
            continue
        o = orig.iloc[0]
        for c in CRITS:
            dc = f"{c}_doc" if f"{c}_doc" in m else c
            a, b = getattr(r, dc), o[dc]
            if pd.isna(a) or pd.isna(b):
                continue
            pairs_doc.append(int(a)); pairs_rep.append(int(b))
            per_crit.setdefault(c, []).append(int(a == b))
    if pairs_doc:
        res["intra"] = {"n_pairs_answers": len(pairs_doc), "n_repeat_frames": int(len(rep)),
                        "agreement": float(np.mean(np.array(pairs_doc) == np.array(pairs_rep))),
                        "kappa_pooled": float(cohen_kappa(pairs_doc, pairs_rep)),
                        "kappa_ci": [float(x) for x in boot_ci(pairs_doc, pairs_rep, cohen_kappa, rng)],
                        "per_crit_agreement": {c: float(np.mean(v)) for c, v in per_crit.items()},
                        "per_crit_n": {c: len(v) for c, v in per_crit.items()}}
    return res


def to_markdown(res):
    L = ["# Слепая ревизия: согласие врача с разметкой и с моделью", "",
         f"Зафиксировано ответов: {res['n_fixed']} из {res['n_shown']} показов. "
         f"Kappa — Cohen; ДИ 95 % — бутстрап по кадрам, {N_BOOT} ресемплов. «Не уверен» исключён из kappa (число указано отдельно). "
         "Чувствительность/специфичность — ответ врача относительно референса (метка или модель).", ""]
    for sname, sub in res["subsets"].items():
        L += [f"## Подмножество: {sname}", "",
              "| Критерий | Референс | n (поз. референса) | Доля согласия [ДИ] | Kappa [ДИ] | Чувст. | Специф. |",
              "|---|---|---|---|---|---|---|"]
        for c in CRITS:
            s = sub[c]
            L.append(f"| {CRIT_RU[c]} | метка разметки {fmt(s['vs_label'])}")
            L.append(f"| {CRIT_RU[c]} | модель (OOF) {fmt(s['vs_model'])}")
        L.append("")
        L.append("«Не уверен»: " + ", ".join(f"{CRIT_RU[c]} — {sub[c]['n_unsure']} из {sub[c]['n_applicable']}" for c in CRITS) + ".")
        am = res["agree_model"][sname]
        L.append(f"Ответ «согласны с моделью?» (n = {am['n']}): да {am['yes']}, нет {am['no']}, частично {am['partial']}.")
        L.append("")
    L += ["## Intra-reader (повторы под другими id)", ""]
    it = res["intra"]
    if it:
        L.append(f"Повторных кадров: {it['n_repeat_frames']}; пар ответов по критериям: {it['n_pairs_answers']}; "
                 f"доля совпадений {it['agreement']:.2f}; pooled kappa {it['kappa_pooled']:.2f} "
                 f"[{it['kappa_ci'][0]:.2f}; {it['kappa_ci'][1]:.2f}].")
        L.append("По критериям: " + ", ".join(f"{CRIT_RU[c]} {v:.2f} (n={it['per_crit_n'][c]})" for c, v in it["per_crit_agreement"].items()) + ".")
    else:
        L.append("Нет пар «оригинал–повтор» с зафиксированными ответами.")
    L += ["", "Примечание: метки train по итогам сессии не меняются; расхождения врач–метка идут в лист адьюдикации "
          "(out/adjudication_template.csv) только для отчёта."]
    return "\n".join(L) + "\n"


def fill_adjudication(exp: pd.DataFrame, key: pd.DataFrame, out_csv: Path):
    """Заполняет лист адьюдикации строками, где ответ врача расходится с меткой или с моделью."""
    m = key.merge(exp, on="display_id", how="inner", suffixes=("", "_doc"))
    rows = []
    for r in m.itertuples():
        for c in CRITS:
            lab = getattr(r, LABEL_COL[c]); mod = getattr(r, f"oof_{c}")
            doc = getattr(r, f"{c}_doc") if hasattr(r, f"{c}_doc") else getattr(r, c)
            if pd.isna(lab):
                continue
            doc_s = "не уверен" if getattr(r, f"unsure_{c}") else ("" if pd.isna(doc) else ("да" if doc == 1 else "нет"))
            if doc_s == "" or (not pd.isna(doc) and doc == lab and (pd.isna(mod) or mod == lab)):
                continue
            rows.append({"id": r.display_id, "критерий": c, "метка разметки": "да" if lab == 1 else "нет",
                         "ответ врача": doc_s, "модель": "" if pd.isna(mod) else ("да" if mod == 1 else "нет"),
                         "итог адьюдикации": "", "комментарий": "",
                         "повтор": "да" if r.is_repeat else "", "подмножество": "расхождения" if r.is_discrepancy else "случайные"})
    pd.DataFrame(rows, columns=["id", "критерий", "метка разметки", "ответ врача", "модель", "итог адьюдикации",
                                "комментарий", "повтор", "подмножество"]).to_csv(out_csv, index=False, encoding="utf-8-sig")
    return len(rows)


def make_synthetic(key: pd.DataFrame, rng, p_agree=0.8, p_unsure=0.05):
    rows = []
    for r in key.itertuples():
        answers = {}
        for c in CRITS:
            lab = getattr(r, LABEL_COL[c])
            if pd.isna(lab):
                continue
            u = rng.random()
            if u < p_unsure:
                answers[c] = "unsure"
            else:
                agree = rng.random() < p_agree
                v = int(lab) if agree else 1 - int(lab)
                answers[c] = "yes" if v == 1 else "no"
        fixed = rng.random() < 0.97
        rows.append({"display_id": r.display_id, "area": r.area, "answers": answers if fixed else {},
                     "comment": "", "fixed_at": "2026-09-25T10:00:00Z" if fixed else None,
                     "agree_model": rng.choice(["yes", "no", "partial"], p=[0.6, 0.2, 0.2]) if fixed else None,
                     "comment_model": ""})
    return {"exported_at": "2026-09-25T13:00:00Z", "n_frames": len(key), "answers": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path)
    ap.add_argument("--key", type=Path, default=HERE.parent / "out" / "review_key.csv")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--adjudication", type=Path, default=None, help="куда записать заполненный лист адьюдикации")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    key = pd.read_csv(args.key)
    rng = np.random.default_rng(SEED)
    if args.synthetic:
        synth = make_synthetic(key, rng)
        p = HERE.parent / "tmp" / "synthetic_review_answers.json"
        p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps(synth, ensure_ascii=False, indent=1), encoding="utf-8")
        args.export = p
        out_md = args.out or HERE.parent / "tmp" / "synthetic_agreement.md"
        adj = args.adjudication or HERE.parent / "tmp" / "synthetic_adjudication.csv"
        print("синтетический экспорт:", p)
    else:
        if not args.export:
            ap.error("--export обязателен (или --synthetic)")
        out_md = args.out or args.export.with_name("agreement.md")
        adj = args.adjudication or args.export.with_name("adjudication_filled.csv")
    exp = load_export(args.export)
    res = analyse(exp, key)
    out_md.write_text(to_markdown(res), encoding="utf-8")
    out_md.with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    n_adj = fill_adjudication(exp, key, adj)
    print(out_md.read_text(encoding="utf-8"))
    print(f"JSON: {out_md.with_suffix('.json')}; лист адьюдикации ({n_adj} строк): {adj}")


if __name__ == "__main__":
    main()
