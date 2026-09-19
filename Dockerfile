# syntax=docker/dockerfile:1.6
# =============================================================================
# DensitoAI — контроль качества DXA-снимков (ЛЦТ 2026)
# Полностью автономный CPU-образ: все зависимости и веса внутри, внешних
# сервисов и загрузок во время работы нет. Версии базового образа и пакетов
# зафиксированы (ТЗ п.3.2).
# =============================================================================
FROM python:3.12.8-slim-bookworm AS base

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
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

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
# JSON Schema результата (проверка в inference.validate_output_csv) и утилиты (validate_sr, make_model_card)
COPY schema/ /app/schema/
COPY tools/ /app/tools/
# data/geometry_features.csv нужен только для медиан импутации (маленький файл)
COPY data/geometry_features.csv /app/data/geometry_features.csv
COPY tests/test_inference_format.py /app/tests/test_inference_format.py
COPY tests/test_schema.py /app/tests/test_schema.py
COPY tests/sample_test_zip/ /app/tests/sample_test_zip/

# --- непривилегированный пользователь, точки монтирования -------------------
RUN useradd -m -u 1000 densito \
 && mkdir -p /data/input /data/output \
 && chown -R densito:densito /app /data
USER densito

VOLUME ["/data/input", "/data/output"]
EXPOSE 8000

# Проверка при сборке: импорт всех модулей и прогон на образце организаторов
RUN python -c "import sys; sys.path.insert(0,'/app/src'); import inference, api_server; print('imports OK')" \
 && python /app/src/inference.py --input /app/tests/sample_test_zip --output /tmp/selftest.csv --no-embeddings \
 && rm -f /tmp/selftest.csv /tmp/selftest.log

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status==200 else 1)" || exit 0

# По умолчанию — пакетная обработка смонтированной папки.
# Переопределяется аргументами `docker run ... <image> [args inference.py]`
# или командой `api` (см. entrypoint).
COPY --chown=densito:densito docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["batch"]
