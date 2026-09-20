# syntax=docker/dockerfile:1.6
# =============================================================================
# DensitoAI — контроль качества DXA-снимков (ЛЦТ 2026)
# Полностью автономный CPU-образ: все зависимости и веса внутри, внешних
# сервисов и загрузок во время работы нет. Версии базового образа и пакетов
# зафиксированы (ТЗ п.3.2).
#
# Сборка:   docker build --platform linux/amd64 -t densitoai:2.2.0 .
# Проверка: docker run --rm --network none densitoai:2.2.0 verify   (tools/offline_check.sh)
# =============================================================================
# Базовый образ закреплён по digest (multi-arch index python:3.12.8-slim-bookworm,
# получен 2026-09-19 запросом к registry-1.docker.io; для linux/amd64 внутри индекса —
# sha256:8859bd6ca943079262c27e38b7119cdacede77c463139a15651dd340087a6cc9).
# Проверить: docker buildx imagetools inspect python:3.12.8-slim-bookworm
ARG BASE_DIGEST=sha256:2199a62885a12290dc9c5be3ca0681d367576ab7bf037da120e564723292a2f0
FROM --platform=linux/amd64 python:3.12.8-slim-bookworm@${BASE_DIGEST} AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # веса EfficientNet-B0 лежат в образе -> torchvision ничего не скачивает
    TORCH_HOME=/app/models/torch_home \
    DENSITO_ROOT=/app \
    DENSITO_MODELS_DIR=/app/models \
    DENSITO_CONFIG=/app/config.yaml \
    DENSITO_OUTPUT_DIR=/data/output \
    # потоки BLAS/OpenMP: 2 по умолчанию (стенд 2 vCPU / 3 ГБ); переопределяется
    # `docker run -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 ...`
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    PYTHONHASHSEED=0

# libgomp1 — OpenMP для torch/sklearn; libglib2.0-0 — OpenCV headless
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- зависимости (отдельный слой для кэширования) --------------------------
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade "pip==25.0.1" \
 && python -m pip install -r /app/requirements.txt

# --- код, конфиг, модели ----------------------------------------------------
COPY config.yaml /app/config.yaml
COPY src/ /app/src/
COPY models/ /app/models/
# data/geometry_features.csv нужен только для медиан импутации (маленький файл)
COPY data/geometry_features.csv /app/data/geometry_features.csv
# tests/: test_inference_format.py, test_transfer_syntax.py, синтетические фантомы + эталон
# (tests/phantoms/); tests/sample_test_zip/ копируется, если есть в контексте сборки
# (в релиз-архив образец организаторов по умолчанию не входит, см. tools/make_release.sh)
COPY tests/ /app/tests/
COPY tools/ /app/tools/
# JSON Schema результата (inference.validate_output_csv, tests/test_schema.py)
COPY schema/ /app/schema/
# Веб-интерфейс (лендинг + кабинет врача + режим лаборанта, один HTML без CDN).
# Кабинет обязан быть в образе: решение разворачивает техгруппа заказчика без интернета,
# и UI не должен зависеть от нашего демо-стенда (api_server отдаёт его с корня).
COPY web/ /app/web/
RUN chmod +x /app/tools/*.sh /app/tools/*.py

# --- непривилегированный пользователь, точки монтирования -------------------
RUN useradd -m -u 1000 densito \
 && mkdir -p /data/input /data/output \
 && chown -R densito:densito /app /data
USER densito

VOLUME ["/data/input", "/data/output"]
EXPOSE 8000

# Проверка при сборке: импорт модулей, sha256 весов, полная самопроверка на фантомах
# (два прогона, детерминизм, эталон). Результат сборки не зависит от сети.
RUN python -c "import sys; sys.path.insert(0,'/app/src'); import inference, api_server; print('imports OK')" \
 && python /app/tools/hash_weights.py --check --root /app \
 && VERIFY_OUT=/tmp/verify_build bash /app/tools/verify.sh \
 && rm -rf /tmp/verify_build

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status==200 else 1)" || exit 0

# По умолчанию — пакетная обработка смонтированной папки.
# Команды entrypoint: batch | api | verify [--data DIR --expected-sha SHA] | test | bash
COPY --chown=densito:densito docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["batch"]
