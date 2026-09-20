#!/usr/bin/env bash
# offline_check.sh — контрольный запуск готового образа без сети.
#   bash tools/offline_check.sh [image] [output_dir]
# Запускает `verify` внутри контейнера с --network none, лимитами 2 CPU / 3 ГБ и
# сохраняет отчёт в <output_dir>/verify/verification_report.html. Код возврата = код verify.
set -eu
IMAGE=${1:-${IMAGE:-densitoai:2.3.0}}
OUT=${2:-${OUT:-./outputs}}
CPUS=${CPUS:-2}; MEM=${MEM:-3g}
mkdir -p "$OUT"
OUT_ABS=$(cd "$OUT" && pwd)
command -v docker >/dev/null 2>&1 || { echo "docker не найден" >&2; exit 2; }
echo ">>> offline verify: image=$IMAGE cpus=$CPUS mem=$MEM out=$OUT_ABS/verify"
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "образ $IMAGE не найден локально (docker load -i dist/densitoai-*-image.tar.gz)" >&2; exit 2; }
set +e
docker run --rm --network none --cpus="$CPUS" --memory="$MEM" \
  --user "$(id -u):$(id -g)" \
  -v "$OUT_ABS:/data/output" \
  "$IMAGE" verify
RC=$?
set -e
echo ">>> код возврата verify: $RC; отчёт: $OUT_ABS/verify/verification_report.html"
exit $RC
