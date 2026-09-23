"""Сводка outputs/summary.json: синтетическая валидация, диагностический AUC скора головы, nested ΔAUC, гейт идеи 11, метаморфика."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

W = Path(__file__).resolve().parents[1]
OUT = W / "outputs"
lab = pd.read_csv("/home/user/workspace/densito/src/densito_rebuild/data/labels_for_embeddings.csv")
m = (lab.region == "spine").values
y = lab.loc[m, "sp_pos"].values.astype(int)
S = {"runs": {}, "nested": {}, "paired_gate_idea11": {}, "metamorphic": {}}
for run in ["finetune_v3", "head_v3", "head_v3_seed1", "head_v3_seed2", "head_v3_seed3", "finetune_v2", "finetune_v1"]:
    d = OUT / run
    if not (d / "cv_metrics.json").exists():
        continue
    cv = json.load(open(d / "cv_metrics.json"))
    sc = pd.read_csv(d / "scores_synth.csv")
    rec = {"mode": cv.get("mode"), "synthetic_cv_summary": {k: v for k, v in cv["summary"].items() if not k.startswith("train_")}}
    for c in ["p_defect", "p_defect_oof", "pred_oy", "pred_oy_oof", "pred_bottom", "pred_bottom_oof", "pred_ty_oof"]:
        if c in sc:
            rec[f"diag_auc_sp_pos_{c}"] = float(roc_auc_score(y, sc.loc[m, c].values))
    rec["p_defect_spine_mean"] = float(sc.loc[m, "p_defect"].mean())
    if (d / "metamorphic.json").exists():
        mm = json.load(open(d / "metamorphic.json"))
        keep = lambda dd: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in dd.items()
                           if not k.endswith("_grid") and not k.endswith("_by_grid")}
        S["metamorphic"][run] = {k: keep(v) for k, v in mm.items() if isinstance(v, dict)}
    S["runs"][run] = rec
for nd in sorted(OUT.glob("nested_*")):
    if (nd / "summary.json").exists():
        S["nested"][nd.name] = json.load(open(nd / "summary.json"))
for g in sorted(OUT.glob("paired_gate_*.json")):
    r = json.load(open(g))
    S["paired_gate_idea11"][g.name] = [{k: it.get(k) for k in ("input", "accepted", "old_rule", "delta_auc", "signflip_permutation", "delta_macro_f1")}
                                       for it in r] if isinstance(r, list) else r
json.dump(S, open(OUT / "summary.json", "w"), indent=1, ensure_ascii=False, default=float)
print("ok", OUT / "summary.json")
