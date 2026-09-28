#!/usr/bin/env python3
"""40 кадров слепой проверки: найдено нарушений с верной причиной, ложные тревоги, флаги sp_art —
для страницы поставки (web/review/index.html, 2.4.0 OOF) и страницы с OOF эксперимента
(outputs/spart_position/review_page_new.html, пишет regression_review.py)."""
import json
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
SIDE = {"rh_pos": "hip_pos", "lh_pos": "hip_pos", "rh_roi": "hip_roi", "lh_roi": "hip_roi"}
man = json.loads((ROOT / "tools" / "review" / "kit_light_manifest.json").read_text(encoding="utf-8"))
order = {s["file"]: s["idx"] for s in man["shows_full"] if not s["is_repeat"]}
for name, p in (("2.4.0 (поставка)", "web/review/index.html"), ("2.5.0 (эксперимент)", "outputs/spart_position/review_page_new.html")):
    html = (ROOT / p).read_text(encoding="utf-8")
    s = json.loads(re.search(r"^const DATA = (.+);$", html, re.M).group(1))["service"]
    viol = found_bin = found_reason = norm = fa = 0
    art_ok, art_bad = [], []
    for f in man["frames"]:
        lab = [c for c, v in f["labels"].items() if v == 1]
        sv = s[f["file"]]
        fl = [SIDE.get(c["code"], c["code"]) for c in sv["criteria"] if c["flag"]]
        if lab:
            viol += 1; found_bin += int(sv["quality_class"]); found_reason += int(any(c in lab for c in fl))
        else:
            norm += 1; fa += int(sv["quality_class"])
        if "sp_art" in fl:
            (art_ok if f["labels"].get("sp_art") == 1 else art_bad).append(order[f["file"]])
    print(f"{name}: нарушение найдено {found_bin}/{viol}, из них с верной причиной {found_reason}/{viol}; "
          f"ложных тревог {fa}/{norm}; флаг sp_art верный на показах {sorted(art_ok)}, ложный на показах {sorted(art_bad)}")
