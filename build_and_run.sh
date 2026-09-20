#!/usr/bin/env bash
# =============================================================================
# DensitoAI — сборка образа и запуск (Linux, Docker >= 20.10)
#
#   ./build_and_run.sh build                         — только собрать образ
#   ./build_and_run.sh run  <input_dir> <output_dir> [доп. аргументы inference.py]
#                                                    — пакетная обработка папки/архива
#   ./build_and_run.sh api  [port] [input_dir] [output_dir]
#                                                    — поднять HTTP API (по умолчанию :8000)
#   ./build_and_run.sh test                          — самопроверка внутри контейнера
#   ./build_and_run.sh shell                         — bash внутри контейнера
#
# Пример:
#   ./build_and_run.sh run /mnt/dxa_test ./outputs
#   -> ./outputs/results.csv, results.xlsx, results_debug.csv, results.log
#
# Переменные окружения: IMAGE (имя образа), NO_BUILD=1 (не пересобирать),
#                        CPUS (лимит CPU, по умолчанию min(nproc, 8)), MEM (лимит памяти, 8g)
# =============================================================================
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

IMAGE="${IMAGE:-densitoai:2.3.0}"
# по умолчанию — все ядра хоста, но не больше 8 (docker падает с ошибкой, если
# запросить --cpus больше, чем реально доступно на машине; nproc всегда <= фактического)
_NPROC="$(nproc 2>/dev/null || echo 4)"
CPUS="${CPUS:-$(( _NPROC < 8 ? _NPROC : 8 ))}"
MEM="${MEM:-8g}"

need_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker не найден. Установите Docker Engine: https://docs.docker.com/engine/install/" >&2
    echo "Без Docker см. README.md, раздел «Запуск без контейнера»." >&2
    exit 2
  fi
}

build() {
  need_docker
  echo ">>> Сборка образа ${IMAGE} ..."
  DOCKER_BUILDKIT=1 docker build --pull -t "${IMAGE}" .
  echo ">>> Готово: ${IMAGE}"
}

abs() { readlink -f "$1"; }

run_batch() {
  local in="${1:?укажите входную папку или zip-архив}"
  local out="${2:?укажите папку для результатов}"
  shift 2
  mkdir -p "${out}"
  local in_abs; in_abs="$(abs "${in}")"
  local out_abs; out_abs="$(abs "${out}")"
  local mount_in="/data/input"
  local extra_env=()
  if [[ -f "${in_abs}" ]]; then                     # одиночный zip/dcm -> монтируем файл
    mount_in="/data/input/$(basename "${in_abs}")"
    extra_env=(-e "DENSITO_INPUT=${mount_in}")
  fi
  echo ">>> Вход:  ${in_abs}"
  echo ">>> Выход: ${out_abs}/results.csv (+ .xlsx, _debug.csv, .log)"
  docker run --rm \
    --cpus="${CPUS}" --memory="${MEM}" \
    --user "$(id -u):$(id -g)" \
    -v "${in_abs}:${mount_in}:ro" \
    -v "${out_abs}:/data/output" \
    "${extra_env[@]}" \
    "${IMAGE}" batch "$@"
  echo ">>> Результат: ${out_abs}/results.csv"
}

run_api() {
  local port="${1:-8000}"
  local in="${2:-./tests/sample_test_zip}"
  local out="${3:-./outputs}"
  mkdir -p "${out}"
  echo ">>> API: http://localhost:${port}/docs  (Ctrl+C для остановки)"
  docker run --rm -it \
    --cpus="${CPUS}" --memory="${MEM}" \
    --user "$(id -u):$(id -g)" \
    -p "${port}:8000" \
    -v "$(abs "${in}"):/data/input:ro" \
    -v "$(abs "${out}"):/data/output" \
    "${IMAGE}" api
}

cmd="${1:-help}"; shift || true
case "${cmd}" in
  build)  build ;;
  run)    [[ "${NO_BUILD:-0}" == "1" ]] || build; run_batch "$@" ;;
  api)    [[ "${NO_BUILD:-0}" == "1" ]] || build; run_api "$@" ;;
  test)   [[ "${NO_BUILD:-0}" == "1" ]] || build; need_docker; docker run --rm "${IMAGE}" test ;;
  shell)  need_docker; docker run --rm -it -v "$(pwd)/outputs:/data/output" "${IMAGE}" bash ;;
  *)      sed -n '2,20p' "$0"; exit 1 ;;
esac
