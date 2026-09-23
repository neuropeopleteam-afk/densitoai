#!/usr/bin/env bash
# =============================================================================
# verify.sh — самопроверка DensitoAI без сети (хост с python или контейнер).
#
#   bash tools/verify.sh                                  # фантомы: схема, строки, UID, Failure,
#                                                         # детерминизм, sha256 весов, эталон,
#                                                         # прогон-двойник, стресс-набор
#   bash tools/verify.sh --data /path/to/dicoms           # + прогон на данных пользователя,
#                                                         #   печать sha256 предсказаний
#   bash tools/verify.sh --data DIR --expected-sha <sha>  # + сверка sha256 с заданным
#   bash tools/verify.sh --update-expected                # перезаписать эталон (только разработчику)
#   docker run --rm --network none densitoai:2.4.0 verify # то же внутри образа
#
# Переменные: VERIFY_OUT (каталог результатов; по умолчанию $DENSITO_OUTPUT_DIR/verify или outputs/verify),
#             PYTHON (интерпретатор), OMP_NUM_THREADS (по умолчанию 2), TORCH_HOME (models/torch_home),
#             VERIFY_SKIP_TRANSFER=1 / VERIFY_SKIP_STRESS=1 (пропустить прогон-двойник / стресс-набор),
#             VERIFY_STRESS_HUGE (размер огромного кадра стресс-набора, по умолчанию 4000x3000).
# Результат: <VERIFY_OUT>/verify_results.json, verification_report.html, run1/, run2/, stress/, data/; код 0/1.
# Совместимость: POSIX sh + bash; используются только printf/test/case, без массивов.
# =============================================================================
set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=${DENSITO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
cd "$ROOT"

PYTHON=${PYTHON:-python3}
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$OMP_NUM_THREADS}
export TORCH_HOME=${TORCH_HOME:-$ROOT/models/torch_home}
export DENSITO_ROOT="$ROOT"
export PYTHONHASHSEED=0

DATA_DIR=""; EXPECTED_SHA=""; UPDATE_EXPECTED=0
while [ $# -gt 0 ]; do
  case "$1" in
    --data)          DATA_DIR=$2; shift 2 ;;
    --expected-sha)  EXPECTED_SHA=$2; shift 2 ;;
    --update-expected) UPDATE_EXPECTED=1; shift ;;
    --out)           VERIFY_OUT=$2; shift 2 ;;
    -h|--help)       sed -n '2,18p' "$0"; exit 0 ;;
    *) printf 'verify.sh: неизвестный аргумент %s\n' "$1" >&2; exit 2 ;;
  esac
done

OUT=${VERIFY_OUT:-${DENSITO_OUTPUT_DIR:-$ROOT/outputs}/verify}
mkdir -p "$OUT/run1" "$OUT/run2"
PHANTOMS="$ROOT/tests/phantoms"
[ -f "$PHANTOMS/MANIFEST.json" ] || { printf 'verify.sh: нет %s/MANIFEST.json (запустите python tools/make_phantoms.py)\n' "$PHANTOMS" >&2; exit 1; }

now_s() { date +%s; }
T0=$(now_s)
printf '== DensitoAI verify == root=%s out=%s OMP_NUM_THREADS=%s\n' "$ROOT" "$OUT" "$OMP_NUM_THREADS"

# --- 1. два прогона на фантомах ---------------------------------------------
printf -- '-- прогон 1 на фантомах\n'
T1=$(now_s)
"$PYTHON" "$ROOT/src/inference.py" --input "$PHANTOMS" --output "$OUT/run1/results.csv" --debug-csv >"$OUT/run1/stdout.log" 2>&1 || RC1=$?
T2=$(now_s)
printf -- '-- прогон 2 на фантомах\n'
"$PYTHON" "$ROOT/src/inference.py" --input "$PHANTOMS" --output "$OUT/run2/results.csv" >"$OUT/run2/stdout.log" 2>&1 || RC2=$?
T3=$(now_s)
RC1=${RC1:-0}; RC2=${RC2:-0}
[ -f "$OUT/run1/results.csv" ] || { printf 'verify.sh: прогон 1 не создал CSV (код %s); см. %s/run1/stdout.log\n' "$RC1" "$OUT" >&2; tail -20 "$OUT/run1/stdout.log" >&2; exit 1; }
[ -f "$OUT/run2/results.csv" ] || { printf 'verify.sh: прогон 2 не создал CSV (код %s)\n' "$RC2" >&2; exit 1; }

if [ "$UPDATE_EXPECTED" = 1 ]; then
  "$PYTHON" "$ROOT/tools/verify_checks.py" update-expected --run1 "$OUT/run1/results.csv" --phantoms "$PHANTOMS"
fi

# --- 1б. инвариантность к форме подачи (имена, порядок, zip) -----------------
TRANSFER_JSON=""
if [ "${VERIFY_SKIP_TRANSFER:-0}" = "1" ]; then
  printf -- '-- проверка инвариантности к именам/порядку/zip пропущена (VERIFY_SKIP_TRANSFER=1)\n'
else
  printf -- '-- проверка инвариантности: переименование, zip, перемешивание, смешанный вход (побитово)\n'
  TRANSFER_JSON="$OUT/transfer_check.json"
  "$PYTHON" "$ROOT/tools/transfer_check.py" --input "$PHANTOMS" --baseline "$OUT/run1/results.csv" --out "$TRANSFER_JSON" --modes rename,zip,shuffle,mixed --bitwise --workdir "$OUT/transfer" --python "$PYTHON" >"$OUT/transfer_check.log" 2>&1 || printf 'verify.sh: transfer_check завершился с ошибкой, см. %s/transfer_check.log\n' "$OUT" >&2
  rm -rf "$OUT/transfer/renamed" "$OUT/transfer/renamed_bundle.zip" "$OUT/transfer/shuffle" "$OUT/transfer/mixed" "$OUT/transfer/var_rename" "$OUT/transfer/var_zip" "$OUT/transfer/var_shuffle" "$OUT/transfer/var_mixed" 2>/dev/null || true
fi

# --- 1в. стресс-набор: битые и нестандартные входы из фантомов + смешанный пакет ---
STRESS_JSON=""
if [ "${VERIFY_SKIP_STRESS:-0}" = "1" ]; then
  printf -- '-- стресс-набор устойчивости входа пропущен (VERIFY_SKIP_STRESS=1)\n'
else
  printf -- '-- стресс-набор: битые и нестандартные входы, смешанный пакет норма+битые (побитово)\n'
  STRESS_JSON="$OUT/stress_check.json"
  "$PYTHON" "$ROOT/tools/stress_set.py" --phantoms "$PHANTOMS" --baseline "$OUT/run1/results.csv" --out "$STRESS_JSON" --workdir "$OUT/stress" --python "$PYTHON" --huge "${VERIFY_STRESS_HUGE:-4000x3000}" >"$OUT/stress_check.log" 2>&1 || printf 'verify.sh: stress_set завершился с ошибкой, см. %s/stress_check.log\n' "$OUT" >&2
  rm -rf "$OUT/stress/input" 2>/dev/null || true
fi

# --- 2. sha256 весов ----------------------------------------------------------
printf -- '-- sha256 весов моделей\n'
"$PYTHON" "$ROOT/tools/hash_weights.py" --check --root "$ROOT" --json "$OUT/weights_check.json" >"$OUT/weights_check.log" 2>&1 || true

# --- 3. данные пользователя (опционально) ------------------------------------
DATA_ARGS=""
T4=$T3; T5=$T3
if [ -n "$DATA_DIR" ]; then
  printf -- '-- прогон на данных пользователя: %s\n' "$DATA_DIR"
  mkdir -p "$OUT/data"
  T4=$(now_s)
  "$PYTHON" "$ROOT/src/inference.py" --input "$DATA_DIR" --output "$OUT/data/results.csv" --debug-csv --xlsx >"$OUT/data/stdout.log" 2>&1 || printf 'verify.sh: инференс на данных завершился с ошибкой, см. %s/data/stdout.log\n' "$OUT" >&2
  T5=$(now_s)
  DATA_ARGS="--data-csv $OUT/data/results.csv --data-dir $DATA_DIR"
  [ -n "$EXPECTED_SHA" ] && DATA_ARGS="$DATA_ARGS --expected-sha $EXPECTED_SHA"
fi

# --- 4. проверки + отчёт -----------------------------------------------------
printf '{"run1_s": %s, "run2_s": %s, "data_s": %s, "started": "%s"}\n' \
  "$((T2 - T1))" "$((T3 - T2))" "$((T5 - T4))" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$OUT/timings.json"

RC=0
# shellcheck disable=SC2086
"$PYTHON" "$ROOT/tools/verify_checks.py" checks \
  --run1 "$OUT/run1/results.csv" --run2 "$OUT/run2/results.csv" --phantoms "$PHANTOMS" \
  --expected "$PHANTOMS/expected_results.csv" --weights-json "$OUT/weights_check.json" \
  --timings "$OUT/timings.json" --out "$OUT/verify_results.json" \
  --transfer "$TRANSFER_JSON" --debug-csv "$OUT/run1/results_debug.csv" --stress "$STRESS_JSON" $DATA_ARGS || RC=$?

"$PYTHON" "$ROOT/tools/verification_report.py" --json "$OUT/verify_results.json" --html "$OUT/verification_report.html" || RC=1
printf 'Отчёт: %s/verification_report.html (JSON: %s/verify_results.json), всего %s с, код возврата %s\n' \
  "$OUT" "$OUT" "$(( $(now_s) - T0 ))" "$RC"
exit "$RC"
