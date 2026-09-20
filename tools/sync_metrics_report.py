"""Переносит сгенерированные таблицы из docs/metrics_oof_full.md в docs/METRICS_REPORT.md.

Зачем: METRICS_REPORT.md — руками написанный отчёт по ТЗ §8.4, но таблицы метрик в нём —
копия генератора `src/eval_oof_metrics.py`. Копировать руками = расхождение чисел между
документами. Скрипт заменяет блок от «## Поясничный отдел позвоночника — по типам нарушений»
до конца таблицы «## Итого» включительно, ничего вокруг не трогая.

Запуск: python tools/sync_metrics_report.py [--check]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "metrics_oof_full.md"
DST = ROOT / "docs" / "METRICS_REPORT.md"
START = "## Поясничный отдел позвоночника — по типам нарушений"
END_SECTION = "## Итого"


def extract_block(text: str) -> str:
    i = text.index(START)
    j = text.index(END_SECTION, i)
    # конец блока «Итого» — последняя строка таблицы (строка, начинающаяся с '|')
    # берём заголовок «Итого» и только его таблицу (пояснения генератора не нужны)
    tail = text[j:].splitlines()
    out = [tail[0]]
    for line in tail[1:]:
        if line.startswith("#"):
            break
        if out and out[-1].startswith("|") and not line.startswith("|"):
            break
        out.append(line)
    while out and not out[-1].strip():
        out.pop()
    return text[i:j] + "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="только проверить совпадение (код 1 при расхождении)")
    a = ap.parse_args()

    block = extract_block(SRC.read_text())
    dst = DST.read_text()
    i = dst.index(START)
    j = dst.index(END_SECTION, i)
    tail = dst[j:].splitlines()
    k = 1
    while k < len(tail) and not tail[k].startswith("#"):
        k += 1
    end_offset = j + len("\n".join(tail[:k]))
    current = dst[i:end_offset].rstrip() + "\n"

    if current.strip() == block.strip():
        print("OK: таблицы METRICS_REPORT.md совпадают с metrics_oof_full.md")
        return
    if a.check:
        print("РАСХОЖДЕНИЕ: METRICS_REPORT.md не совпадает с metrics_oof_full.md "
              "(запустите python tools/sync_metrics_report.py)", file=sys.stderr)
        sys.exit(1)
    DST.write_text(dst[:i] + block + dst[end_offset:])
    print(f"обновлён {DST.relative_to(ROOT)}: таблицы перенесены из {SRC.relative_to(ROOT)}")

    # версия в шапке отчёта — из config.yaml
    cfg = (ROOT / "config.yaml").read_text()
    m = re.search(r'^version:\s*"([^"]+)"', cfg, re.M)
    if m:
        txt = DST.read_text()
        txt2 = re.sub(r"Версия системы: \*\*[^*]+\*\*", f"Версия системы: **{m.group(1)}**", txt, count=1)
        if txt2 != txt:
            DST.write_text(txt2)
            print(f"версия в шапке -> {m.group(1)}")


if __name__ == "__main__":
    main()
