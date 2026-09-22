# Проверка поставки DensitoAI 2.3.0 (инструкция для технической группы заказчика)

> **Назначение и ограничения.** Программа не является медицинским изделием и не предназначена
> для диагностики, профилактики, лечения или мониторинга заболеваний. Сервис оценивает
> техническое качество укладки и области интереса исследования и не формирует медицинское
> заключение. Решение по исследованию принимает врач. Область применения: денситометрия (DXA)
> поясничного отдела позвоночника и проксимального отдела бедра, оборудование GE Lunar Prodigy;
> вне этой области применения результат не определён. Источник формулировки —
> `config.yaml → intended_use`, та же строка отдаётся в `/api/health` и показана в веб-интерфейсе.

Проверка выполняется на машине с Docker (Linux x86-64, 2 CPU и 3 ГБ памяти достаточно), без доступа в интернет.
Пик памяти процесса обработки в наших замерах — около 0,6 ГБ; самопроверка на 15 фантомах занимает
около 15–20 с на 2 vCPU (в стенде разработки: 12 с при одном потоке).

## Пять команд

1. Проверить контрольные суммы и загрузить образ.

        cd dist && sha256sum -c SHA256SUMS && cd ..
        docker load -i dist/densitoai-2.3.0-image.tar.gz

2. Самопроверка образа без сети (фантомные DICOM внутри образа, два прогона, сравнение с эталоном, sha256 весов).

        mkdir -p outputs
        docker run --rm --network none --cpus=2 --memory=3g --user "$(id -u):$(id -g)" \
          -v "$PWD/outputs:/data/output" densitoai:2.3.0 verify
        # то же одной строкой: bash tools/offline_check.sh densitoai:2.3.0 ./outputs

   Код возврата 0 — все проверки пройдены; 1 — есть расхождение. Отчёт: `outputs/verify/verification_report.html`
   (таблица проверок зелёным/красным, версии пакетов, sha256 весов, время), машинно — `outputs/verify/verify_results.json`.

3. Пакетная обработка собственных DICOM (папка монтируется только на чтение; сеть не нужна).

        docker run --rm --network none --cpus=2 --memory=3g --user "$(id -u):$(id -g)" \
          -v /path/to/dicoms:/data/input:ro -v "$PWD/outputs:/data/output" densitoai:2.3.0 batch

   Результат: `outputs/results.csv` (9 столбцов: study_uid, image_uid, anatomical_region, quality_class, quality_prob,
   violation_list, processing_status, time_of_processing, error_message), `outputs/results.xlsx`, журнал `outputs/inference.log`.

4. Воспроизводимость на собственных данных: тот же прогон через `verify --data` дважды даёт одинаковый CSV
   (без столбца time_of_processing) и печатает его sha256. Повторный запуск с `--expected-sha <sha>` завершится
   кодом 0 только при точном совпадении предсказаний.

        docker run --rm --network none -v /path/to/dicoms:/data/input:ro -v "$PWD/outputs:/data/output" \
          densitoai:2.3.0 verify --data /data/input
        docker run --rm --network none -v /path/to/dicoms:/data/input:ro -v "$PWD/outputs:/data/output" \
          densitoai:2.3.0 verify --data /data/input --expected-sha <sha из предыдущего вывода>

5. Проверить, что веса модели в образе — те, что заявлены в поставке (`models/WEIGHTS_SHA256.txt`).

        docker run --rm --network none densitoai:2.3.0 bash -c "cd /app && sha256sum -c models/WEIGHTS_SHA256.txt"
        docker run --rm --network none densitoai:2.3.0 bash -c "cd /app && python tools/hash_weights.py --check"

## Что именно проверяет `verify`

| Проверка | Критерий |
|---|---|
| Схема CSV | ровно 9 столбцов в фиксированном порядке |
| Полнота | число строк = число входных файлов (15 фантомов, в т.ч. 3 заведомо битых) |
| Идентификаторы | study_uid / image_uid равны тегам StudyInstanceUID / SOPInstanceUID |
| Обработка ошибок | Failure ровно для 3 битых файлов (обрезанные пиксели, без PixelData, не DICOM), остальные Success |
| Детерминизм | два прогона побитово совпадают во всех столбцах, кроме time_of_processing |
| Эталон | классы совпадают с `tests/phantoms/expected_results.csv`, quality_prob в пределах ±0,001 |
| Веса | sha256 `models/*.pkl`, `backbone_densito.pth`, `models_manifest.json`, `config.yaml`, EfficientNet-B0 совпадают с `models/WEIGHTS_SHA256.txt` |
| Данные пользователя (`--data`) | строки = файлы, детерминизм, sha256 предсказаний (+ сверка с `--expected-sha`) |

Эталонные значения поставки: sha256 предсказаний на фантомах
`31273378c11027ea275805cdd98f8622cb92c44f95c93f0b94ca5d5c150b4cdb`; sha256 предсказаний на образце организаторов
(`tests/sample_test_zip`, 3 файла) `df0d9a94731d2589924170f06a6267caf0250507adfd60ce20da5de6aac93bec`.
Эти значения получены в стенде разработки (CPU x86-64, OMP_NUM_THREADS 1 и 2 дают одинаковый результат); на другом
процессоре возможны отличия quality_prob в последних знаках — тогда проверка эталона проходит по допуску ±0,001,
а побитовое сравнение фиксируется как предупреждение в отчёте.

## Поддерживаемые форматы DICOM

Измерено на 3 файлах образца (`docs/TRANSFER_SYNTAX_MATRIX.md`): Implicit/Explicit VR LE, Explicit VR BE, RLE Lossless,
JPEG 2000 Lossless (при установленных кодеках), 8 и 16 бит, MONOCHROME1/2, с PixelSpacing и без, без UID (генерируются
детерминированно), сырой поток без преамбулы — результат идентичен оригиналу. JPEG Lossless и JPEG-LS не проверены
(нет кодировщика в стенде).

### Инвариантность к форме подачи данных и сверка с истиной фантомов

`verify` (и `docker run --rm --network none densitoai:2.3.0 verify`) дополнительно доказывает две вещи.

1. **Независимость от имён файлов, порядка и упаковки.** `tools/transfer_check.py` делает копию
   фантомов со случайными именами файлов и каталогов (`f0000.dcm`, `s000/`), перемешивает порядок,
   отдельно упаковывает эту копию в zip и прогоняет инференс. Предсказания сверяются с эталонным
   прогоном по ключу (`study_uid`, `image_uid`): `anatomical_region`, `quality_class`, `violation_type`,
   `processing_status` должны совпасть, `quality_prob` — бит в бит. Это прямая проверка того, что
   отсутствие суффиксов `_ПОП/_ППОБ` в закрытом наборе ничего не меняет. Результат — файл
   `transfer_check.json` и строка проверки в `verify_results.json`. Резервные идентификаторы для
   файлов без `StudyInstanceUID`/`SOPInstanceUID` считаются от sha256 содержимого (а не от пути),
   поэтому переименование не меняет и их. Пропустить проверку: `VERIFY_SKIP_TRANSFER=1`.
2. **Сверка с истинной геометрией.** `tests/phantoms/MANIFEST.json` хранит истинные параметры каждого
   фантома (наклон оси, положение центра, вариант дефекта, область). Проверка сравнивает с ними
   измеренные признаки: знак и величину наклона (допуск 6°), сдвиг центра, обрезку поля (край ≤ 1 мм),
   область «позвоночник/бедро» и сторону бедра. Металл на синтетических фантомах не проверяется:
   их «кость» сама яркая, порог по интенсивности на таких картинках срабатывает ложно. Два известных
   отклонения (паразитный наклон при боковом сдвиге, сторона при обрезанном поле) печатаются в отчёте
   как примечания и описаны в «Ограничениях методологии» отчёта по метрикам.
