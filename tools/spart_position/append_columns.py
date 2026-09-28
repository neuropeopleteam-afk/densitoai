"""Дописывает к data/geometry_features.csv (текст как есть, побайтно для старых колонок) новые колонки
позиционного признака sp_art из пересчитанной выгрузки. Старые колонки не переписываются, чтобы не
менять представление чисел и значения, посчитанные прежними версиями кода (например, hip_side_score).
Запуск: python tools/spart_position/append_columns.py <базовый.csv> <пересчитанный.csv> <выход.csv>"""
import csv
import sys

base, regen, out = sys.argv[1:4]
A = list(csv.reader(open(base, encoding="utf-8")))
B = list(csv.reader(open(regen, encoding="utf-8")))
ia = {c: i for i, c in enumerate(A[0])}
ib = {c: i for i, c in enumerate(B[0])}
new_cols = [c for c in B[0] if c not in ia and c.startswith("metal_metal_")]
assert len(A) == len(B)
fp_a, fp_b = ia["file_path"], ib["file_path"]
with open(out, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(A[0] + new_cols)
    for ra, rb in zip(A[1:], B[1:]):
        assert ra[fp_a] == rb[fp_b], "порядок строк"
        w.writerow(ra + [rb[ib[c]] for c in new_cols])
print(f"дописано колонок: {len(new_cols)}")
