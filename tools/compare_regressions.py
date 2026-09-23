#!/usr/bin/env python3
"""Сравнение двух выгрузок регрессии (официальный CSV поставки) по image_uid.

Печатает: сколько строк различаются по колонкам anatomical_region / quality_class / violation_type /
quality_prob / processing_status (время не сравнивается), по каким областям, какие строки сменили класс
или состав типов нарушения, какие типы нарушения затронуты, итог по классам.

Запуск: python tools/compare_regressions.py <кандидат.csv> <эталон.csv> [--allow-types "Некорректная укладка"]
Код возврата 1, если изменения вышли за разрешённые типы нарушения (--allow-types) или различаются строки
других областей (--region).
"""
import argparse
import collections
import csv
import sys

COLS = ("anatomical_region", "quality_class", "violation_type", "quality_prob", "processing_status")


def load(path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by = {r["image_uid"]: r for r in rows}
    if len(by) != len(rows):
        print(f"ВНИМАНИЕ: повторяющиеся image_uid в {path}: {len(rows) - len(by)}", file=sys.stderr)
    return by


def types(s):
    return {t for t in (s or "").split(";") if t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cand")
    ap.add_argument("ref")
    ap.add_argument("--allow-types", default=None,
                    help="типы нарушения, которые могут появляться/исчезать (через ;); остальные — стоп")
    ap.add_argument("--region", default=None, help="единственная область, строки которой могут меняться")
    a = ap.parse_args()
    cand, ref = load(a.cand), load(a.ref)
    if set(cand) != set(ref):
        print(f"СТОП: наборы image_uid различаются: только в кандидате {len(set(cand) - set(ref))}, "
              f"только в эталоне {len(set(ref) - set(cand))}")
        return 1
    allow = types(a.allow_types) if a.allow_types else None
    diff, cls, vt, regions, touched, max_dp, bad = [], [], [], collections.Counter(), collections.Counter(), 0.0, 0
    for k, rc in cand.items():
        rr = ref[k]
        if all(rc[c] == rr[c] for c in COLS):
            continue
        diff.append(k)
        regions[rc["anatomical_region"]] += 1
        max_dp = max(max_dp, abs(float(rc["quality_prob"]) - float(rr["quality_prob"])))
        changed = types(rc["violation_type"]) ^ types(rr["violation_type"])
        for t in changed:
            touched[t] += 1
        if allow is not None and (changed - allow):
            bad += 1
        if a.region and rc["anatomical_region"] != a.region:
            bad += 1
        if rc["quality_class"] != rr["quality_class"]:
            cls.append((k, rr, rc))
        elif rc["violation_type"] != rr["violation_type"]:
            vt.append((k, rr, rc))
    n = len(cand)
    print(f"строк {n}, различаются {len(diff)}, по областям {dict(regions)}, макс |Δquality_prob| {max_dp:.4f}")
    print(f"отказов: кандидат {sum(r['processing_status'] != 'Success' for r in cand.values())}, "
          f"эталон {sum(r['processing_status'] != 'Success' for r in ref.values())}")
    print(f"смена класса: {len(cls)}")
    for k, rr, rc in cls:
        print(f"  …{k[-12:]}: {rr['quality_class']} → {rc['quality_class']} | {rr['violation_type'] or 'норма'} → "
              f"{rc['violation_type'] or 'норма'} | {rr['quality_prob']} → {rc['quality_prob']}")
    print(f"смена состава типов при том же классе: {len(vt)}")
    for k, rr, rc in vt:
        print(f"  …{k[-12:]}: {rr['violation_type']} → {rc['violation_type']} | {rr['quality_prob']} → {rc['quality_prob']}")
    print(f"затронутые типы нарушения: {dict(touched)}")
    cc, cr = collections.Counter(r["quality_class"] for r in cand.values()), \
        collections.Counter(r["quality_class"] for r in ref.values())
    print(f"классы: эталон норма {cr['0']} / нарушение {cr['1']} → кандидат {cc['0']} / {cc['1']}")
    if bad:
        print(f"СТОП: {bad} изменений вне разрешённых типов/области")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
