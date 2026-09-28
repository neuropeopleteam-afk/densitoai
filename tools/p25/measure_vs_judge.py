#!/usr/bin/env python3
"""2.5, пункт 2 («сервис — измеритель, врач — судья», консилиум 27.09, Claude Opus 5.5, Р4-2): пересчёт по сырым
ответам врачей на 40 кадрах слепой проверки.

Правило задано до расчёта (текст Р4-2): измеряемые критерии — sp_pos, sp_axis, hip_roi; подсказки — sp_art,
hip_pos. Пара «врач + сервис» говорит «нарушение», если так сказал врач ИЛИ сервис поставил флаг по измеряемому
критерию; флаг по подсказке пару не меняет (решает врач). Для сравнения — «врач один», «сервис один»,
наивное «или» (любой флаг сервиса). Счёт как в tools/review/compute_light_agreement.py: первый показ кадра,
«затрудняюсь» исключено, повторная отправка сессии и второй проход с того же компьютера исключены.
Точный двусторонний тест Макнемара (исправления пары против новых ошибок относительно разметки) — чтобы честно
показать, что на 40 кадрах различия статистически не значимы.

Флаги сервиса — OOF текущих моделей (models/oof_stacked_*.csv; для 2.5.0 — новый sp_art). --version-page
сравнивает с другой страницей проверки (например, поставка 2.4.x: web/review/index.html).
Запуск: python tools/p25/measure_vs_judge.py [--page <страница с DATA.service>] [--out <json>]
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

MEASURED = ("sp_pos", "sp_axis", "hip_roi")
HINTS = ("sp_art", "hip_pos")


def binom_two_sided(k, n):
    """Точный двусторонний тест Макнемара: min(1, 2·P(X <= min(k, n-k))), X ~ Bin(n, 0.5)."""
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n)


def stats(pred, truth):
    tp = sum(p and t for p, t in zip(pred, truth)); fp = sum(p and not t for p, t in zip(pred, truth))
    fn = sum((not p) and t for p, t in zip(pred, truth)); tn = sum((not p) and (not t) for p, t in zip(pred, truth))
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": tp + fp + fn + tn, "sens": round(sens, 3),
            "spec": round(spec, 3), "youden": round(sens + spec - 1, 3), "acc": round((tp + tn) / (tp + fp + fn + tn), 3)}


def flags_from_page(page):
    svc = json.loads(re.search(r"^const DATA = (.+);$", Path(page).read_text(encoding="utf-8"), re.M).group(1))["service"]
    return {f: {re.sub(r"^(rh|lh)_", "hip_", c["code"]) for c in s["criteria"] if c.get("flag")} for f, s in svc.items()}


def evaluate(frames, flags, first):
    files = sorted(frames, key=lambda f: frames[f]["show"])
    truth = {f: any(frames[f]["labels"].values()) for f in files}
    out = {"service_alone": stats([bool(flags[f]) for f in files], [truth[f] for f in files]), "doctors": {}}
    for name, ans in first.items():
        use = [f for f in files if ans.get(f) in ("ok", "bad")]
        doc = {f: ans[f] == "bad" for f in use}
        rules = {"doctor_alone": {f: doc[f] for f in use},
                 "naive_or": {f: doc[f] or bool(flags[f]) for f in use},
                 "measure_judge": {f: doc[f] or bool(flags[f] & set(MEASURED)) for f in use}}
        res = {k: stats([v[f] for f in use], [truth[f] for f in use]) for k, v in rules.items()}
        fixed = [frames[f]["show"] for f in use if rules["measure_judge"][f] == truth[f] and doc[f] != truth[f]]
        broke = [frames[f]["show"] for f in use if rules["measure_judge"][f] != truth[f] and doc[f] == truth[f]]
        res["mcnemar_pair_vs_doctor"] = {"fixed_shows": fixed, "new_errors_shows": broke,
                                         "p_exact_two_sided": round(binom_two_sided(len(fixed), len(fixed) + len(broke)), 3)}
        res["n_unsure"] = len(files) - len(use)
        out["doctors"][name] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", type=Path, default=None, help="взять флаги сервиса со страницы проверки вместо OOF")
    ap.add_argument("--out", type=Path, default=C.ROOT / "docs" / "p25" / "measure_vs_judge.json")
    a = ap.parse_args()
    frames = C.load_frames40()
    if a.page:
        flags = flags_from_page(a.page)
        src = str(a.page)
    else:
        flags = {f: {c for c, v in d["flags"].items() if v} for f, d in frames.items()}
        src = "OOF models/oof_stacked_*.csv"
    res = {"rule": "пара = врач «нарушение» ИЛИ флаг сервиса по измеряемому критерию (sp_pos, sp_axis, hip_roi)",
           "measured": MEASURED, "hints": HINTS, "service_flags_from": src, "n_frames": len(frames)}
    res.update(evaluate(frames, flags, C.load_first_answers()))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("флаги сервиса:", src)
    print("сервис один:", res["service_alone"])
    for n, r in res["doctors"].items():
        print(n, "| один: чувств.", r["doctor_alone"]["sens"], "спец.", r["doctor_alone"]["spec"], "Юден",
              r["doctor_alone"]["youden"], "| наивное «или»:", r["naive_or"]["youden"], "| измеритель+судья:",
              r["measure_judge"]["sens"], r["measure_judge"]["spec"], r["measure_judge"]["youden"], "| точность",
              r["doctor_alone"]["acc"], "->", r["measure_judge"]["acc"], "| Макнемар", r["mcnemar_pair_vs_doctor"])
    print("->", a.out)


if __name__ == "__main__":
    main()
