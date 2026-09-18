#!/usr/bin/env bash
# Точка входа контейнера DensitoAI.
#   batch [args...]  — пакетная обработка: /data/input -> /data/output/results.csv (по умолчанию)
#   api   [args...]  — HTTP API на порту 8000
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
  test)
    exec python /app/tests/test_inference_format.py ;;
  *)
    exec "$cmd" "$@" ;;
esac
