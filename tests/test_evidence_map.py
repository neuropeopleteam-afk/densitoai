#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_evidence_map.py — карта доказательств (tools/evidence_map.json) полна и однозначна.

Проверяет без запуска инференса:
  1. каждая проверка, объявленная в tools/verify_checks.py (add("...") и name = "..."), имеет ровно одну
     запись в карте (сопоставление по началу имени — поле prefix);
  2. каждая запись карты соответствует хотя бы одной проверке из verify_checks.py (нет «мёртвых» записей);
  3. у каждой записи заполнены id, prefix, tz, requirement, knows, where; id и prefix уникальны;
  4. блок not_proven не пуст и содержит обязательные ограничения (закрытые данные, другой аппарат,
     клиническая корректность);
  5. tools/verification_report.py находит запись для каждой проверки той же функцией, что и отчёт;
  6. если рядом есть outputs/verify/verify_results.json — каждая проверка из него тоже находится в карте.

Запуск: python tests/test_evidence_map.py   (код 0 — всё пройдено; печатает число проверок)
Корень проекта — родитель каталога tests/ или переменная DENSITO_ROOT.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(os.environ.get("DENSITO_ROOT") or Path(__file__).resolve().parents[1])
CHECKS_PY = ROOT / "tools" / "verify_checks.py"
MAP_JSON = ROOT / "tools" / "evidence_map.json"
REPORT_PY = ROOT / "tools" / "verification_report.py"
RESULTS_JSON = ROOT / "outputs" / "verify" / "verify_results.json"

REQUIRED_FIELDS = ("id", "prefix", "tz", "requirement", "knows", "where")
REQUIRED_NOT_PROVEN = ("закрыт", "аппарат", "клинич")  # подстроки в title/text блока not_proven

# add("Имя", ...) / add(f"Имя {x}", ...) / name = "Имя" / name = f"Имя"
_RE_ADD = re.compile(r'\badd\(\s*(f?)"([^"\n]+)"')
_RE_NAME = re.compile(r'^\s*name\s*=\s*(f?)"([^"\n]+)"', re.M)

_failed = 0
_total = 0


def check(cond: bool, msg: str) -> None:
    global _failed, _total
    _total += 1
    if not cond:
        _failed += 1
        print(f"FAIL: {msg}")


def static_prefix(name: str, is_fstring: bool) -> str:
    """Для f-строки — часть до первой подстановки (по ней и сопоставляем); обычная строка берётся целиком."""
    return name.split("{", 1)[0].rstrip() if is_fstring else name


def declared_check_names(src: str) -> list[str]:
    names = _RE_ADD.findall(src) + _RE_NAME.findall(src)
    seen, out = set(), []
    for flag, n in names:
        p = static_prefix(n, flag == "f")
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def main() -> int:
    check(CHECKS_PY.is_file(), f"нет {CHECKS_PY}")
    check(MAP_JSON.is_file(), f"нет {MAP_JSON}")
    if _failed:
        return 1
    emap = json.loads(MAP_JSON.read_text(encoding="utf-8"))
    entries = emap.get("entries", [])
    check(len(entries) > 0, "карта пуста")

    # 3. поля, уникальность
    ids = [e.get("id") for e in entries]
    prefixes = [e.get("prefix") for e in entries]
    check(len(set(ids)) == len(ids), f"повторяющиеся id: {sorted({i for i in ids if ids.count(i) > 1})}")
    check(len(set(prefixes)) == len(prefixes), "повторяющиеся prefix")
    for e in entries:
        for f in REQUIRED_FIELDS:
            check(bool(str(e.get(f, "")).strip()), f"запись {e.get('id')}: пустое поле {f}")
        check("!" not in str(e.get("knows", "")), f"запись {e.get('id')}: восклицательный знак в тексте")
    # один prefix не должен быть началом другого — иначе сопоставление неоднозначно
    for a in prefixes:
        for b in prefixes:
            if a != b:
                check(not b.startswith(a), f"prefix «{a}» является началом «{b}» — сопоставление неоднозначно")

    # 1. каждая проверка verify_checks.py → ровно одна запись
    names = declared_check_names(CHECKS_PY.read_text(encoding="utf-8"))
    check(len(names) >= 17, f"в verify_checks.py найдено только {len(names)} имён проверок (ожидалось не меньше 17)")
    used = set()
    for n in names:
        hits = [e for e in entries if n.startswith(e["prefix"]) or e["prefix"].startswith(n)]
        check(len(hits) == 1, f"проверка «{n}»: записей в карте {len(hits)} (нужна ровно одна)")
        for e in hits:
            used.add(e["id"])

    # 2. каждая запись → хотя бы одна проверка
    for e in entries:
        check(e["id"] in used, f"запись карты «{e['id']}» ({e['prefix']}) не соответствует ни одной проверке verify_checks.py")

    # 4. блок not_proven
    npv = emap.get("not_proven", [])
    check(len(npv) >= 3, f"not_proven: {len(npv)} блоков, нужно не меньше 3")
    blob = " ".join((b.get("title", "") + " " + b.get("text", "")).lower() for b in npv)
    for key in REQUIRED_NOT_PROVEN:
        check(key in blob, f"not_proven: нет ограничения со словом «{key}»")

    # 5. функция отчёта находит те же записи
    if REPORT_PY.is_file():
        sys.path.insert(0, str(REPORT_PY.parent))
        try:
            import verification_report as vr  # type: ignore
            rmap = vr.load_evidence_map()
            for n in names:
                check(vr.evidence_for(n, rmap) is not None, f"verification_report.evidence_for не нашла «{n}»")
            fake = {"ok": True, "generated_at": "", "root": "", "checks": [{"name": n, "ok": True, "detail": "", "level": "error"} for n in names],
                    "env": {}, "timings": {}, "phantoms_manifest": {}, "sha256": {}, "weights": {}}
            html_out = vr.render(fake)
            check("Карта доказательств" in html_out and "не доказывает" in html_out, "в HTML нет разделов карты доказательств")
            check("нет в карте" not in html_out, "в HTML есть проверки без записи в карте")
        except Exception as ex:  # noqa: BLE001
            check(False, f"verification_report.py не импортируется: {ex}")
    else:
        print(f"пропуск: нет {REPORT_PY}")

    # 6. фактический verify_results.json, если есть
    if RESULTS_JSON.is_file():
        res = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        for c in res.get("checks", []):
            hits = [e for e in entries if str(c.get("name", "")).startswith(e["prefix"])]
            check(len(hits) == 1, f"verify_results.json: «{c.get('name')}» → записей {len(hits)}")
    else:
        print(f"пропуск: нет {RESULTS_JSON}")

    print(f"test_evidence_map: {_total - _failed}/{_total} проверок пройдено; имён проверок в verify_checks.py: {len(names)}, записей в карте: {len(entries)}")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
