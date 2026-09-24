#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверка согласованности версии и хэша конфигурации по всей поставке.

Зачем. Версия сервиса указана примерно в двадцати местах: конфиг, код, образ, compose,
инструкция техгруппе, README, карточка модели, подвал лендинга, релизные архивы. Один
рассогласованный номер стоит нам вопроса «какая из версий верная», поэтому проверка встроена
в сборку релиза: `tools/make_release.sh` падает, если хоть одно место расходится.

Эталон — `config.yaml → version` и `config_hash`, посчитанный из того же конфига той же формулой,
что в `src/inference.py: config_hash`.

Запуск:
    python tools/check_version_consistency.py            # проверка, код 0/1
    python tools/check_version_consistency.py --manifest dist/VERSION_MANIFEST.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # корень поставки: и в репозитории, и в образе (/app)

# (файл, [регулярные выражения с группой v]) — где обязана стоять эталонная версия.
VERSION_PLACES = [
    ("config.yaml", [r'^version:\s*"(?P<v>[\d.]+)"']),
    ("src/inference.py", [r'^__version__\s*=\s*"(?P<v>[\d.]+)"', r'^\s*"version":\s*"(?P<v>[\d.]+)"']),
    ("docker-compose.yml", [r'image:\s*densitoai:(?P<v>[\d.]+)']),
    ("Dockerfile", [r'densitoai:(?P<v>[\d.]+)']),
    ("build_and_run.sh", [r'densitoai:(?P<v>[\d.]+)']),
    ("tools/make_release.sh", [r'VERSION=\$\{1:-\$\{VERSION:-(?P<v>[\d.]+)\}\}']),
    ("tools/offline_check.sh", [r'densitoai:(?P<v>[\d.]+)']),
    ("tools/verify.sh", [r'densitoai:(?P<v>[\d.]+)']),
    ("README.md", [r'densitoai:(?P<v>[\d.]+)', r'densitoai-(?P<v>[\d.]+)-(?:src|image)',
                   r'Версия \*\*(?P<v>[\d.]+)\*\*', r'make_release\.sh (?P<v>[\d.]+)']),
    ("docs/VERIFICATION.md", [r'densitoai:(?P<v>[\d.]+)', r'densitoai-(?P<v>[\d.]+)-image',
                              r'^# Проверка поставки DensitoAI (?P<v>[\d.]+)']),
    ("docs/METRICS_REPORT.md", [r'Версия системы: \*\*(?P<v>[\d.]+)\*\*']),
    ("docs/DZM_CONFORMANCE.md", [r'Версия решения (?P<v>[\d.]+)']),
    ("docs/EVIDENCE.md", [r'Версия (?P<v>[\d.]+), config_hash']),
    ("models/MODEL_CARD.md", [r'version\):\s*\*\*(?P<v>[\d.]+)\*\*']),
    ("web/index.html", [r'версия сервиса (?P<v>[\d.]+)']),
    ("web/docs.html", [r'версия сервиса (?P<v>[\d.]+)']),
    ("web/review/index.html", [r'"service_version":"(?P<v>[\d.]+)"']),
    ("tools/review/kit_light_manifest.json", [r'"service_version": "(?P<v>[\d.]+)"']),
]

# Где обязан стоять эталонный config_hash (12 знаков).
HASH_PLACES = [
    ("docs/EVIDENCE.md", [r'config_hash (?P<h>[0-9a-f]{12})']),
    ("docs/DZM_CONFORMANCE.md", [r'config_hash (?P<h>[0-9a-f]{12})']),
    ("models/MODEL_CARD.md", [r'config_hash:\s*\*\*(?P<h>[0-9a-f]{12})\*\*']),
    ("web/index.html", [r'хэш конфигурации (?P<h>[0-9a-f]{12})']),
    ("web/docs.html", [r'хэш конфигурации (?P<h>[0-9a-f]{12})']),
]


def canonical_version() -> str:
    """Версия из config.yaml. Без тяжёлых зависимостей: проверка должна работать и на хосте."""
    text = (ROOT / "config.yaml").read_text(encoding="utf-8")
    m = re.search(r'^version:\s*"([\d.]+)"', text, flags=re.MULTILINE)
    return m.group(1) if m else ""


def hash_from_local() -> str:
    """config_hash той же формулой, что в продакшене (нужны зависимости пайплайна)."""
    import importlib.util

    if importlib.util.find_spec("yaml") is None:
        # без PyYAML load_config падает с ConfigError (fail-closed); здесь — понятный текст заранее
        raise RuntimeError("нет PyYAML: хэш рабочего дерева был бы посчитан от значений по умолчанию")
    sys.path.insert(0, str(ROOT / "src"))
    from inference import load_config, config_hash as _ch  # noqa: E402

    return _ch(load_config())


def hash_from_image(image: str) -> str:
    """config_hash, который сообщает сам поставляемый образ: сверяем документы с образом,
    а не с текущим рабочим деревом."""
    import subprocess

    code = ("import sys; sys.path.insert(0, '/app/src');"
            " from inference import load_config, config_hash;"
            " print(config_hash(load_config()))")
    out = subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "python",
                          image, "-c", code], capture_output=True, text=True, timeout=180)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[:300])
    return out.stdout.strip().splitlines()[-1].strip()


def scan(places, group: str, expected: str, missing_ok: bool = False) -> list:
    """Расхождения по списку мест. `missing_ok=True` — для запуска внутри образа, куда попадает
    только часть поставки (README, docs/, Dockerfile, compose в образ не копируются)."""
    problems = []
    for rel, patterns in places:
        p = ROOT / rel
        if not p.exists():
            if not missing_ok:
                problems.append(f"{rel}: файла нет")
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        found = 0
        for pat in patterns:
            for m in re.finditer(pat, text, flags=re.MULTILINE):
                found += 1
                got = m.group(group)
                if got != expected:
                    line = text[:m.start()].count("\n") + 1
                    problems.append(f"{rel}:{line}: {got} вместо {expected}")
        if found == 0:
            problems.append(f"{rel}: не найдено ни одного места с версией/хэшем "
                            f"(шаблоны: {'; '.join(patterns)})")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="Согласованность версии и config_hash по поставке")
    ap.add_argument("--expect-version", help="дополнительно сверить с этой версией (аргумент релиза)")
    ap.add_argument("--manifest", help="записать манифест версии в этот файл")
    ap.add_argument("--image", help="взять config_hash из этого образа (docker run --network none)")
    ap.add_argument("--require-hash", action="store_true",
                    help="падать, если config_hash посчитать не удалось")
    args = ap.parse_args()

    version = canonical_version()
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        print(f"НЕ ПРОШЛО: config.yaml → version = {version!r}")
        return 1

    chash, hash_src = "", ""
    for src_name, fn in (("образ " + str(args.image), (lambda: hash_from_image(args.image)) if args.image else None),
                         ("рабочее дерево", hash_from_local)):
        if fn is None:
            continue
        try:
            chash, hash_src = fn(), src_name
            break
        except Exception as e:  # noqa: BLE001
            print(f"config_hash не получен из {src_name}: {str(e)[:140]}")

    problems = scan(VERSION_PLACES, "v", version)
    if chash:
        problems += scan(HASH_PLACES, "h", chash)
    elif args.require_hash:
        problems.append("config_hash посчитать не удалось, а он требуется (--require-hash)")
    else:
        print("ВНИМАНИЕ: config_hash не проверен (нет зависимостей пайплайна и не указан --image)")
    if args.expect_version and args.expect_version != version:
        problems.append(f"аргумент сборки {args.expect_version} != config.yaml {version}")

    print(f"Эталон: версия {version}, config_hash {chash or 'не проверен'}"
          + (f" (источник: {hash_src})" if hash_src else ""))
    if problems:
        print(f"НЕ ПРОШЛО, расхождений {len(problems)}:")
        for x in problems:
            print("  -", x)
        return 1
    print(f"Согласовано: {len(VERSION_PLACES)} файлов с версией, {len(HASH_PLACES)} с хэшем конфигурации")

    if args.manifest:
        w = ROOT / "models" / "WEIGHTS_SHA256.txt"
        mf = [
            "DensitoAI — манифест версии поставки",
            f"версия сервиса:      {version}",
            f"config_hash:         {chash}",
            f"образ:               densitoai:{version}",
            f"файлов весов:        {len(w.read_text(encoding='utf-8').splitlines()) if w.exists() else 0}",
            f"sha256 списка весов: {hashlib.sha256(w.read_bytes()).hexdigest() if w.exists() else '-'}",
            "проверено:           config.yaml, src/inference.py, Dockerfile, docker-compose.yml,",
            "                     README.md, docs/VERIFICATION.md, docs/METRICS_REPORT.md,",
            "                     docs/DZM_CONFORMANCE.md, docs/EVIDENCE.md, models/MODEL_CARD.md,",
            "                     web/index.html, web/docs.html, web/review/index.html,",
            "                     tools/review/kit_light_manifest.json, tools/*.sh",
            "проверка получателем: docker run --rm --network none densitoai:"
            f"{version} verify",
        ]
        out = Path(args.manifest)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(mf) + "\n", encoding="utf-8")
        print(f"манифест записан: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
