"""Сверка пересчитанной выгрузки с поставленной: все старые колонки должны совпасть (NaN == NaN)."""
import sys
import numpy as np
import pandas as pd
old = pd.read_csv(sys.argv[1]); new = pd.read_csv(sys.argv[2])
assert len(old) == len(new) and (old["file_path"].values == new["file_path"].values).all(), "порядок строк"
bad = []
for c in old.columns:
    if c not in new.columns:
        bad.append((c, "нет колонки")); continue
    a, b = old[c], new[c]
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        eq = np.isclose(a.values.astype(float), b.values.astype(float), rtol=0, atol=1e-9, equal_nan=True)
    else:
        eq = (a.astype(str).values == b.astype(str).values)
    if not eq.all():
        bad.append((c, int((~eq).sum())))
added = [c for c in new.columns if c not in old.columns]
print(f"строк {len(new)}; старых колонок {len(old.columns)}, расхождений: {len(bad)} {bad[:10]}")
print(f"новых колонок {len(added)}: {added}")
