# Приём снимков по DICOM (C-STORE) — `src/dicom_receiver.py`

Дополнительный функционал. Снимки попадают в DensitoAI напрямую с денситометра или из PACS /
рабочей станции по протоколу DICOM, без ручной загрузки через кабинет. Приёмник — Storage SCP на
`pynetdicom==3.0.4`; анализ выполняет тот же API-маршрут `/api/analyze`, что и кабинет, поэтому
результаты (CSV, XLSX, технический CSV, DICOM SR, PNG) лежат там же и в том же формате.

Не медицинское изделие; результат — подсказка контроля качества укладки, а не диагноз.

## 1. Параметры подключения

| Параметр | Значение по умолчанию | Переменная окружения |
|---|---|---|
| AE Title приёмника (Called AE) | `DENSITOAI` | `DENSITO_RECEIVER_AET` |
| Порт | `11112` (проброшен в `docker-compose.yml`, `DICOM_PORT`) | `DENSITO_RECEIVER_PORT` |
| IP | адрес сервера с контейнером | `DENSITO_RECEIVER_BIND` (интерфейс, `0.0.0.0`) |
| Разрешённые AE Title отправителей | любые | `DENSITO_RECEIVER_ALLOWED_AET` (через запятую) |
| Папка приёма | `/data/output/inbox` | `DENSITO_INBOX` |
| Адрес API для анализа | `http://127.0.0.1:8000` | `DENSITO_API_URL` (`inproc` — без сети, в процессе приёмника) |
| Таймаут тишины | 5 с | `DENSITO_RECEIVER_IDLE_S` |
| Пауза после закрытия ассоциации | 1 с | `DENSITO_RECEIVER_RELEASE_S` |
| Хранить принятые DICOM после анализа | да (автоудаление не выполняется) | `DENSITO_INBOX_KEEP=0` — удалять после успешного анализа |

В настройках DICOM денситометра или в PACS создаётся узел назначения (Storage destination):
AE Title `DENSITOAI`, IP сервера, порт `11112`. Максимальный размер PDU не ограничен.
Перед отправкой снимков полезно проверить связь кнопкой C-ECHO (Verify) на аппарате.

### Поддерживаемые SOP-классы

| SOP Class | UID |
|---|---|
| Verification (C-ECHO) | `1.2.840.10008.1.1` |
| Computed Radiography Image Storage — так экспортирует GE Lunar Prodigy (все 499 файлов выборки) | `1.2.840.10008.5.1.4.1.1.1` |
| Digital X-Ray Image Storage — For Presentation | `1.2.840.10008.5.1.4.1.1.1.1` |
| Digital X-Ray Image Storage — For Processing | `1.2.840.10008.5.1.4.1.1.1.1.1` |
| Secondary Capture Image Storage | `1.2.840.10008.5.1.4.1.1.7` |

Другие SOP-классы (CT, MR, SR и т.д.) на уровне ассоциации не согласуются: PACS получит отказ по
контексту, а не ошибку хранения.

### Transfer syntax

По умолчанию предлагаются те, что пайплайн читает без дополнительных кодеков
(`docs/TRANSFER_SYNTAX_MATRIX.md`): Implicit VR Little Endian `1.2.840.10008.1.2`,
Explicit VR Little Endian `1.2.840.10008.1.2.1`, Explicit VR Big Endian `1.2.840.10008.1.2.2`,
RLE Lossless `1.2.840.10008.1.2.5`. Экспорт GE Lunar — несжатый Implicit VR LE. Если PACS хранит
снимки в JPEG 2000 / JPEG Lossless, он перекодирует их в несжатый вид сам (стандартное поведение
при отсутствии общего сжатого контекста); при установленных кодеках `pylibjpeg` в образе можно
включить все transfer syntax: `DENSITO_RECEIVER_ALL_TS=1`.

## 2. Что происходит с файлом

1. C-STORE принимается, объект пишется атомарно (временный файл → переименование) в
   `<inbox>/<StudyInstanceUID>/<SOPInstanceUID>.dcm`. Повторная отправка того же объекта
   перезаписывает файл. Объект без `StudyInstanceUID` или `SOPInstanceUID` отклоняется статусом
   `0xA900`, ошибка записи — `0xA700`.
2. Исследование считается принятым и уходит на анализ, когда ассоциация закрыта и после последнего
   снимка прошла 1 с, либо — если аппарат держит ассоциацию открытой — после 5 с тишины.
   Все снимки исследования из этой партии передаются одним запросом `POST /api/analyze?xlsx=true`
   (multipart, имена `<study_uid>/<sop_uid>.dcm`, поэтому `path_to_study` в CSV — тот же путь).
   Снимки, пришедшие позже, образуют отдельный запрос.
3. Результат: `/data/output/jobs/<job_id>/` — `results.csv`, `results.xlsx`, `results_debug.csv`,
   `results_extras.csv`, `summary.json`, `sr/<study_uid>_SR.dcm`, PNG-оверлеи — ровно как для
   загрузки через кабинет. Код запроса и код доступа записываются в `<inbox>/<study_uid>/job.json`
   (права 0600) со ссылками: `/api/jobs/<job_id>?t=<job_token>` (карточка),
   `/api/results/<job_id>/results.csv?t=<job_token>`, `/#app/job/<job_id>/<job_token>` (кабинет).
   Список всех запросов, включая принятые по DICOM, — `GET /api/jobs` с админским ключом
   (`DENSITO_ADMIN_KEY`).
4. После анализа принятые DICOM по умолчанию остаются в `<inbox>/<study_uid>/` (`DENSITO_INBOX_KEEP=1`,
   автоудаление загрузок не выполняется); удаление после успешного анализа включается только явно —
   `DENSITO_INBOX_KEEP=0` или `--no-keep`. При сбое анализа (API не поднялся, ошибка 5xx) файлы в
   любом режиме остаются, событие пишется в журнал, и при следующем старте приёмник
   отправляет их повторно. Пока API грузит модели, приёмник ждёт и повторяет запрос
   (36 попыток по 5 с).
5. Журнал `<inbox>/receiver_log.csv` (разделитель `;`): `time; event; calling_aet; study_uid;
   sop_uid; sop_class_uid; transfer_syntax; size_bytes; job_id; status`. События: `start`, `echo`,
   `store`, `reject`, `analyze`, `analyze_failed`, `resume`, `stop`. Персональных данных в журнале
   нет — только UID, AE Title, размеры и коды запросов. Технический лог — `<inbox>/receiver.log`.

## 3. Запуск

```bash
# отдельный сервис рядом с API (общий том ./outputs, порт 11112 наружу)
docker compose up -d densito-api densito-receiver

# один контейнер: API и приёмник вместе (healthcheck по /api/health не меняется)
docker run -d -p 8000:8000 -p 11112:11112 -v $(pwd)/outputs:/data/output densitoai:2.4.0 api+receiver

# только приёмник, API на другом хосте
docker run -d -p 11112:11112 -e DENSITO_API_URL=http://densito-api:8000 \
  -v $(pwd)/outputs:/data/output densitoai:2.4.0 receiver

# без контейнера
DENSITO_INBOX=./outputs/inbox DENSITO_API_URL=http://127.0.0.1:8000 python src/dicom_receiver.py
```

Аргументы командной строки дублируют переменные: `--port --aet --bind --inbox --api-url --idle
--release --keep / --no-keep --all-ts -v`. Остановка — SIGTERM/SIGINT: сервер закрывается, уже принятые
исследования дожидаются анализа.

## 4. Проверка

```bash
# связь (C-ECHO)
python -m pynetdicom echoscu <ip> 11112 -aec DENSITOAI
# отправка исследования (два фантома из tests/phantoms, без персональных данных)
python -m pynetdicom storescu <ip> 11112 tests/phantoms/study_01 -r -aec DENSITOAI -aet LUNAR01
# через несколько секунд
cat outputs/inbox/receiver_log.csv          # строки store и analyze с job_id
ls outputs/jobs/<job_id>/                   # results.csv, results.xlsx, summary.json, sr/
```

Автотест: `python tests/test_dicom_receiver.py` — SCP на свободном порту, C-ECHO, C-STORE двух
фантомов, inbox, журнал, отказы (нет StudyInstanceUID, чужой SOP-класс, JPEG 2000), таймаут
тишины, повторная отправка после сбоя, HTTP-путь через тестовый API на uvicorn и, если в
окружении есть torch, через настоящий `api_server` (иначе SKIP).

## 5. Ограничения

- Без TLS и без аутентификации отправителя, кроме фильтра по AE Title. Только для внутренней сети
  отделения; порт 11112 не должен быть доступен извне.
- Реализованы только C-ECHO и C-STORE (роль SCP). C-FIND, C-MOVE, Storage Commitment, MPPS —
  нет: приёмник не запрашивает снимки из PACS сам, результаты в PACS не отправляет
  (SR и SC лежат в папке запроса и передаются в PACS отдельно, если нужно).
- Один запрос анализа — одна партия снимков исследования; параллельные ассоциации принимаются,
  анализ выполняется последовательно (как и в API).
- Принятые объекты содержат кадры пациентов; папка inbox — на томе `/data/output`, доступ к ней
  регулируется на уровне сервера. По умолчанию файлы хранятся (`DENSITO_INBOX_KEEP=1`); удаление после
  успешного анализа включается только `DENSITO_INBOX_KEEP=0`.
- Не медицинское изделие; сервис не заменяет оценку специалиста.
