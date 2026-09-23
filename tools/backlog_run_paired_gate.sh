#!/bin/bash
# Прогон калиброванного гейта (идея 11, tools/paired_gate.py) по всем четырём кандидатам.
# Семейство гипотез: 4 кандидата бэклога + 1 параллельный GPU-эксперимент по sp_pos = 5 (--n-hypotheses 5, Холм:
# для одного входа за вызов это alpha/5 = 0.01 односторонний, т.е. нижняя граница 98 % ДИ ΔAUC > 0).
# Каждый файл (критерий или уровень региона) проверяется отдельно; внутри кандидата критерии не корректируются
# дополнительно (оговорено в docs/NESTED_GATE_REPORT.md, часть 3).
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 DENSITO_ROOT="$PWD"
GATE=tools/paired_gate.py
OUTROOT=${OUTROOT:-outputs/backlog_12_15}
FAM=${FAMILY_M:-5}
run() {  # run <candidate> <file-stem>
  python "$GATE" $OUTROOT/$1/$2_scores.csv --n-hypotheses $FAM --family holm --n-boot 4000 --n-perm 2000 \
      --out $OUTROOT/$1/gate_$2.json > $OUTROOT/$1/gate_$2.txt 2>&1
  echo "--- $1 / $2"; cat $OUTROOT/$1/gate_$2.txt
}
# 13: только критерии, где кандидат отличается от базы
for f in sp_axis sp_art hip_roi; do run tz_weights $f; done
# 14: уровень quality_prob региона
for f in region_spine region_hip; do run noisy_or $f; done
# 12: все критерии
for f in sp_pos sp_axis sp_art hip_pos hip_roi; do run dupweight $f; done
# 15: объединённый уровень бедра
for f in hip_pos hip_roi; do run hip_side_router $f; done
echo GATEDONE > $OUTROOT/GATEDONE
