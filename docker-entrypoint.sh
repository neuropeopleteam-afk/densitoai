#!/usr/bin/env bash
# Точка входа контейнера DensitoAI.
#   batch [args...]  — пакетная обработка: /data/input -> /data/output/results.csv (по умолчанию)
#   api   [args...]  — HTTP API на порту 8000
#   verify [args...] — самопроверка без сети: фантомы, детерминизм, sha256 весов, отчёт
#                      (/data/output/verify/verification_report.html); код 0/1.
#                      verify --data /data/input --expected-sha <sha>  — сверка на данных пользователя
#   test             — самопроверка формата/устойчивости
#   любая другая команда выполняется как есть (например: bash)
set -euo pipefail
cmd="${1:-batch}"; shift || true
case "$cmd" in
  batch)
    exec python /app/src/inference.py \
      --input  "${DENSITO_INPUT:-/data/input}" \
      --output "${DENSITO_OUTPUT:-/data/output/results.csv}" \
      --xlsx --debug-csv "$@" ;;
  api)
    cd /app/src && exec python /app/src/api_server.py --host 0.0.0.0 --port "${DENSITO_PORT:-8000}" "$@" ;;
  verify)
    exec bash /app/tools/verify.sh "$@" ;;
  test)
    # два набора: формат и фантомы — и поведение на входе, которого в выборке не было
    python /app/tests/test_inference_format.py || exit 1
    exec python /app/tests/test_ood_foreign.py ;;
  *)
    exec "$cmd" "$@" ;;
esac
