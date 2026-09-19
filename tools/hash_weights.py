#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hash_weights.py — контрольные суммы весов моделей DensitoAI.

    python tools/hash_weights.py            # записать models/WEIGHTS_SHA256.txt
    python tools/hash_weights.py --check    # сверить файлы с models/WEIGHTS_SHA256.txt (код 0/1)
    sha256sum -c models/WEIGHTS_SHA256.txt  # то же самое стандартной утилитой (из корня проекта)

Файлы: models/*.pkl, models/backbone_densito.pth, models/torch_home/hub/checkpoints/*.pth,
models/models_manifest.json, config.yaml. Формат строк совместим с `sha256sum -c`.
Дополнительно проверяется, что каждый ключ models_manifest.json существует как файл.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parents[1]))
OUT_NAME = "models/WEIGHTS_SHA256.txt"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tracked_files(root: Path) -> list[Path]:
    files = sorted((root / "models").glob("*.pkl"))
    for extra in ("models/backbone_densito.pth", "models/models_manifest.json", "config.yaml"):
        if (root / extra).exists():
            files.append(root / extra)
    files += sorted((root / "models/torch_home/hub/checkpoints").glob("*.pth"))
    return files


def compute(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in tracked_files(root)}


def read_manifest_txt(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sha, _, name = line.partition("  ")
        out[name.strip()] = sha.strip()
    return out


def manifest_keys_exist(root: Path) -> list[str]:
    mp = root / "models/models_manifest.json"
    if not mp.exists():
        return ["models/models_manifest.json missing"]
    data = json.loads(mp.read_text(encoding="utf-8"))
    return [f"models/{k}" for k in data if not (root / "models" / k).exists()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--json", default=None, help="записать результат сверки в JSON")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    actual = compute(root)
    target = root / OUT_NAME

    if not args.check:
        lines = [f"{sha}  {name}" for name, sha in actual.items()]
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"written {target} ({len(lines)} files)")
        missing = manifest_keys_exist(root)
        if missing:
            print("WARNING: models_manifest.json refers to missing files:", missing)
            return 1
        return 0

    if not target.exists():
        print(f"ERROR: {target} not found; run without --check first")
        return 1
    expected = read_manifest_txt(target)
    result = {"ok": True, "files": [], "missing_in_manifest": [], "missing_on_disk": [],
              "manifest_json_missing_files": manifest_keys_exist(root)}
    for name, sha in expected.items():
        if name not in actual:
            result["missing_on_disk"].append(name)
            result["ok"] = False
            continue
        good = actual[name] == sha
        result["files"].append({"file": name, "expected": sha, "actual": actual[name], "ok": good})
        result["ok"] &= good
    for name in actual:
        if name not in expected:
            result["missing_in_manifest"].append(name)
            result["ok"] = False
    if result["manifest_json_missing_files"]:
        result["ok"] = False
    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for f in result["files"]:
        print(("OK   " if f["ok"] else "FAIL ") + f["file"])
    for n in result["missing_on_disk"]:
        print("MISSING ON DISK   " + n)
    for n in result["missing_in_manifest"]:
        print("NOT IN MANIFEST   " + n)
    for n in result["manifest_json_missing_files"]:
        print("models_manifest.json -> missing file  " + n)
    print("weights check:", "OK" if result["ok"] else "FAILED")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
