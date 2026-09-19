#!/usr/bin/env bash
# make_release.sh — релиз-архив исходников и (по запросу) образа с контрольными суммами.
#   bash tools/make_release.sh [version]            # dist/densitoai-<version>-src.tar.gz + SHA256SUMS
#   WITH_IMAGE=1 bash tools/make_release.sh 2.1.0   # + docker save densitoai:2.1.0 | gzip -> dist/...-image.tar.gz
# В архив исходников НЕ входят: outputs/, data/ (кроме geometry_features.csv), gpu/, external_datasets,
# *.pyc, .git, dist/. Конкурсные DICOM в архив не попадают (в репозитории их нет, см. docs/LICENSES_AND_DATA_AUDIT.md);
# tests/sample_test_zip/ (образец организаторов «Для теста») включается только при WITH_SAMPLE=1.
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT"
VERSION=${1:-${VERSION:-2.1.0}}
IMAGE=${IMAGE:-densitoai:$VERSION}
DIST="$ROOT/dist"; mkdir -p "$DIST"
NAME="densitoai-$VERSION"
SRC_TGZ="$DIST/$NAME-src.tar.gz"

if command -v sha256sum >/dev/null 2>&1; then SHA=sha256sum; else SHA="shasum -a 256"; fi

# --- исходники --------------------------------------------------------------
EXCLUDES="--exclude=./outputs --exclude=./dist --exclude=./gpu --exclude=./.git --exclude=./external_datasets \
  --exclude=*.pyc --exclude=__pycache__ --exclude=.pytest_cache --exclude=./data/*.npy --exclude=./data/labels*.csv --exclude=./data/sample_test_zip \
  --exclude=./web/node_modules --exclude=./.env"
[ "${WITH_SAMPLE:-0}" = 1 ] || EXCLUDES="$EXCLUDES --exclude=./tests/sample_test_zip"
if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 && [ "${USE_GIT:-1}" = 1 ]; then
  echo ">>> git archive HEAD -> $SRC_TGZ"
  git -C "$ROOT" archive --format=tar --prefix="$NAME/" HEAD | gzip -n >"$SRC_TGZ"
  git -C "$ROOT" rev-parse HEAD >"$DIST/$NAME-GIT_COMMIT.txt"
else
  echo ">>> tar исходников -> $SRC_TGZ"
  # shellcheck disable=SC2086
  tar --sort=name --mtime='2026-01-01 00:00Z' --owner=0 --group=0 --numeric-owner \
      $EXCLUDES --transform "s,^\./,$NAME/," -cf - . | gzip -n >"$SRC_TGZ"
fi

# --- образ (только по запросу; выполняет оркестратор на машине с docker) ------
if [ "${WITH_IMAGE:-0}" = 1 ]; then
  command -v docker >/dev/null 2>&1 || { echo "docker не найден" >&2; exit 2; }
  echo ">>> docker save $IMAGE | gzip -> $DIST/$NAME-image.tar.gz"
  docker save "$IMAGE" | gzip -n >"$DIST/$NAME-image.tar.gz"
  docker image inspect --format '{{.Id}} {{join .RepoDigests ","}}' "$IMAGE" >"$DIST/$NAME-image-id.txt"
fi

# --- контрольные суммы ---------------------------------------------------------
cd "$DIST"
# shellcheck disable=SC2046
$SHA $(ls "$NAME"-* | grep -v SHA256SUMS) >SHA256SUMS
cp "$ROOT/models/WEIGHTS_SHA256.txt" "$DIST/$NAME-WEIGHTS_SHA256.txt" 2>/dev/null || true
echo ">>> dist/:"; ls -la "$DIST"; echo ">>> SHA256SUMS:"; cat SHA256SUMS
echo "Проверка получателем: cd dist && sha256sum -c SHA256SUMS"
