#!/usr/bin/env python3
"""Комментарий к преобразованию кадра перед backbone (2.4.1, src/embeddings.py).

Комментарий обещал «паддинг до фиксированного прямоугольника с учётом анизотропии пикселя», а код делает простое
растяжение transforms.Resize((320, 192)) без паддинга. Код не меняется (на этом преобразовании обучены модели),
исправлен комментарий. Тест проверяет по исходному тексту src/embeddings.py (без загрузки весов):
  * конвейер — ToPILImage, Resize((320, 192)), ToTensor, Normalize; паддинга (Pad) нет;
  * комментарии блока не обещают паддинг и учёт анизотропии, а описывают растяжение 320×192.

Запуск:  python tests/test_embeddings_transform.py     (код 0 — пройдено)
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILED = []


def check(name, cond, detail=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def main() -> int:
    src = (ROOT / "src" / "embeddings.py").read_text(encoding="utf-8")
    m = re.search(r"self\.tf = transforms\.Compose\(\[(.*?)\]\)", src, flags=re.S)
    check("блок transforms.Compose найден", m is not None)
    if not m:
        return 1
    block = m.group(1)
    code = [ln.split("#", 1)[0].strip() for ln in block.splitlines() if ln.split("#", 1)[0].strip()]
    comments = " ".join(ln.split("#", 1)[1].strip() for ln in block.splitlines() if "#" in ln).lower()
    steps = [re.match(r"transforms\.(\w+)", c).group(1) for c in code if c.startswith("transforms.")]
    check("конвейер: ToPILImage, Resize, ToTensor, Normalize", steps == ["ToPILImage", "Resize", "ToTensor", "Normalize"], str(steps))
    check("Resize((320, 192)) — растяжение до 320×192", any(c.startswith("transforms.Resize((320, 192))") for c in code))
    check("паддинга в коде нет", "Pad" not in " ".join(code))
    check("комментарий не обещает паддинг", not re.search(r"паддинг до|padding to|с учётом анизотропии", comments), comments[:160])
    check("комментарий описывает растяжение и отсутствие учёта анизотропии",
          "растяжение" in comments and "анизотропия пикселя отдельно не учитывается" in comments, comments[:160])

    print("\nИТОГ:", "OK" if not FAILED else f"FAIL ({len(FAILED)}): " + "; ".join(FAILED))
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
