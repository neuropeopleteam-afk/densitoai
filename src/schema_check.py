#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проверка объектов по JSON Schema (draft 2020-12, подмножество) без внешних зависимостей.

Если в окружении есть пакет `jsonschema`, используется он (полная реализация); иначе —
встроенный мини-валидатор, покрывающий ключевые слова, которые используются в
`schema/results_row.schema.json` и `schema/api_analyze_response.schema.json`:
type, enum, const, minimum, maximum, exclusiveMinimum, minLength, maxLength, pattern,
required, properties, additionalProperties, items, minItems, allOf, anyOf, oneOf, not, if/then/else,
$ref только вида "#/$defs/<name>".

Схема — источник истины формата: строки регионов/нарушений, диапазон quality_prob,
статусы Success/Failure. Использование:

    from schema_check import load_schema, validate_instance, coerce_csv_row
    problems = validate_instance(coerce_csv_row(row_dict), load_schema("results_row"))
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

_SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("DENSITO_ROOT", _SRC_DIR.parent)).resolve()
SCHEMA_DIR = Path(os.environ.get("DENSITO_SCHEMA_DIR", PROJECT_ROOT / "schema")).resolve()

try:  # полная реализация, если установлена
    import jsonschema as _jsonschema  # type: ignore
except Exception:  # noqa: BLE001
    _jsonschema = None

BACKEND = "jsonschema" if _jsonschema is not None else "builtin"


def schema_path(name: str) -> Path:
    name = name if name.endswith(".json") else f"{name}.schema.json"
    return SCHEMA_DIR / name


def load_schema(name: str) -> Dict[str, Any]:
    with open(schema_path(name), "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Приведение строки CSV к типам схемы (CSV хранит всё как строки)
# --------------------------------------------------------------------------- #
def coerce_csv_row(row: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(row)
    for k in ("quality_class",):
        v = row.get(k)
        if isinstance(v, str) and re.fullmatch(r"-?[0-9]+", v.strip() or "x"):
            out[k] = int(v)
    for k in ("quality_prob", "time_of_processing"):
        v = row.get(k)
        if isinstance(v, str):
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


# --------------------------------------------------------------------------- #
# Мини-валидатор
# --------------------------------------------------------------------------- #
def _type_ok(value: Any, t: str) -> bool:
    if t == "object":
        return isinstance(value, dict)
    if t == "array":
        return isinstance(value, list)
    if t == "string":
        return isinstance(value, str)
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool) or (isinstance(value, float) and value.is_integer())
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "boolean":
        return isinstance(value, bool)
    if t == "null":
        return value is None
    return True


def _validate(value: Any, schema: Any, root: Dict[str, Any], path: str, errors: List[str]) -> None:
    if schema is True or schema == {}:
        return
    if schema is False:
        errors.append(f"{path}: schema false")
        return
    if "$ref" in schema:
        ref = schema["$ref"]
        if ref.startswith("#/"):
            node: Any = root
            for part in ref[2:].split("/"):
                node = node[part]
            _validate(value, node, root, path, errors)
        return
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_type_ok(value, x) for x in types):
            errors.append(f"{path}: expected type {t}, got {type(value).__name__} ({value!r})")
            return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: {value!r} != const {schema['const']!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} > maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: {value} <= exclusiveMinimum {schema['exclusiveMinimum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: length {len(value)} < minLength {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: length {len(value)} > maxLength {schema['maxLength']}")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            errors.append(f"{path}: {value!r} does not match pattern")
    if isinstance(value, dict):
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"{path}: missing required property '{req}'")
        props = schema.get("properties", {})
        for k, sub in props.items():
            if k in value:
                _validate(value[k], sub, root, f"{path}.{k}", errors)
        ap = schema.get("additionalProperties", True)
        if ap is False:
            extra = [k for k in value if k not in props]
            if extra:
                errors.append(f"{path}: additional properties not allowed: {extra}")
        elif isinstance(ap, dict):
            for k in value:
                if k not in props:
                    _validate(value[k], ap, root, f"{path}.{k}", errors)
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: {len(value)} items < minItems {schema['minItems']}")
        if "items" in schema:
            for i, it in enumerate(value):
                _validate(it, schema["items"], root, f"{path}[{i}]", errors)
    for sub in schema.get("allOf", []):
        _validate(value, sub, root, path, errors)
    if "anyOf" in schema:
        if not any(not _collect(value, sub, root, path) for sub in schema["anyOf"]):
            errors.append(f"{path}: does not match anyOf")
    if "oneOf" in schema:
        n = sum(1 for sub in schema["oneOf"] if not _collect(value, sub, root, path))
        if n != 1:
            errors.append(f"{path}: matches {n} of oneOf (need exactly 1)")
    if "not" in schema and not _collect(value, schema["not"], root, path):
        errors.append(f"{path}: matches forbidden schema (not)")
    if "if" in schema:
        if not _collect(value, schema["if"], root, path):
            if "then" in schema:
                _validate(value, schema["then"], root, path, errors)
        elif "else" in schema:
            _validate(value, schema["else"], root, path, errors)


def _collect(value: Any, schema: Any, root: Dict[str, Any], path: str) -> List[str]:
    errs: List[str] = []
    _validate(value, schema, root, path, errs)
    return errs


def validate_instance(instance: Any, schema: Dict[str, Any], prefix: str = "$") -> List[str]:
    """Список нарушений схемы (пустой — валидно). Использует jsonschema, если есть."""
    if _jsonschema is not None:
        try:
            validator_cls = _jsonschema.validators.validator_for(schema)
            validator = validator_cls(schema)
            return [f"{prefix}{'.' + '.'.join(str(p) for p in e.absolute_path) if e.absolute_path else ''}: {e.message}"
                    for e in sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))]
        except Exception as e:  # noqa: BLE001 — при сбое библиотеки падаем на встроенный
            pass
    return _collect(instance, schema, schema, prefix)


def validate_results_rows(rows: List[Dict[str, Any]], schema: Optional[Dict[str, Any]] = None,
                          first_line: int = 2) -> List[str]:
    """Строки results.csv (как читает csv.DictReader) -> список проблем с номерами строк."""
    schema = schema or load_schema("results_row")
    problems: List[str] = []
    for i, r in enumerate(rows, first_line):
        for e in validate_instance(coerce_csv_row(r), schema, prefix=f"line {i}"):
            problems.append("schema: " + e)
    return problems


if __name__ == "__main__":
    import csv
    import sys
    if len(sys.argv) < 2:
        print("usage: schema_check.py results.csv [schema_name]")
        sys.exit(2)
    with open(sys.argv[1], "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    probs = validate_results_rows(rows, load_schema(sys.argv[2] if len(sys.argv) > 2 else "results_row"))
    print(f"backend={BACKEND} rows={len(rows)} problems={len(probs)}")
    print("\n".join(probs[:50]))
    sys.exit(1 if probs else 0)
