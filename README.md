# DensitoAI — автоматический контроль качества DXA-снимков

Сервис для ЛЦТ 2026 (задача Департамента здравоохранения Москвы): по DICOM-изображениям
рентгеновской денситометрии (DXA) определяет анатомическую область (поясничный отдел
позвоночника / проксимальный отдел бедра), оценивает, есть ли нарушения качества укладки и
области интереса, и формирует табличный отчёт строго в формате ТЗ.

Работает **полностью локально** (CPU, без внешних сервисов), обрабатывает каждый файл
независимо и **никогда не прерывает пакет** из-за одного плохого файла.

Версия **2.5.0**. Демо: [densito.ru](https://densito.ru) — сайт закрыт общим демо-паролем: логин `demo`, пароль `Hakaton`
(он же показан на странице входа). Публичный репозиторий без пароля:
`git clone https://github.com/neuropeopleteam-afk/densitoai.git`; релиз 2.5.0 — образ Docker и исходники — в
[Releases](https://github.com/neuropeopleteam-afk/densitoai/releases) (файлы лежат в ветке `release-2.5.0`, Git LFS).
Презентация для жюри — `docs/presentation/DensitoAI_LCT2026_prezentatsiya.pdf`.

## Быстрый старт

Готовый образ из релиза 2.5.0, Linux или WSL, нужен только Docker. Команды выполняются в пустой папке.

```bash
# 1. Скачать образ (659 МБ) и контрольные суммы, проверить и загрузить
curl -LO https://github.com/neuropeopleteam-afk/densitoai/raw/release-2.5.0/densitoai-2.5.0-image.tar.gz
curl -LO https://github.com/neuropeopleteam-afk/densitoai/raw/release-2.5.0/SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing        # densitoai-2.5.0-image.tar.gz: OK
docker load -i densitoai-2.5.0-image.tar.gz      # Loaded image: densitoai:2.5.0

# 2. Самопроверка без сети: 18 из 18
docker run --rm --network none densitoai:2.5.0 verify

# 3. Пакетная обработка: папка или zip со снимками -> outputs/results.csv (9 колонок ТЗ)
mkdir -p outputs
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v /path/to/dicom:/data/input:ro -v "$PWD/outputs:/data/output" densitoai:2.5.0 batch

# 4. Веб-кабинет http://localhost:8000/ и Swagger http://localhost:8000/docs (остановка — Ctrl+C)
docker run --rm -p 127.0.0.1:8000:8000 --user "$(id -u):$(id -g)" -e DENSITO_REGISTRY_OPEN=1 \
  -v "$PWD/outputs:/data/output" densitoai:2.5.0 api
```

- `--user` нужен, чтобы контейнер мог писать в `outputs` (иначе на Linux — `PermissionError`, §13 п. 7).
- Сборка из исходников вместо готового образа: `git clone https://github.com/neuropeopleteam-afk/densitoai.git && cd densitoai && ./build_and_run.sh build`
  (5–10 минут, нужен интернет), дальше `NO_BUILD=1 ./build_and_run.sh run /path/to/dicom ./outputs` — без `NO_BUILD=1`
  команды `run`, `api` и `test` каждый раз пересобирают образ.
- Windows 11 (Docker Desktop, PowerShell) — §4, «Запуск на Windows».
- Образец организаторов «Для теста»: снимок `CR000000_ПОП.dcm` получает «Присутствуют посторонние предметы»
  (quality_prob 0.952) — концы рёбер в верхних углах попадают в зону измерения. Это известное ограничение (§10), не сбой.

> **Назначение и ограничения.** Программа не является медицинским изделием и не предназначена
> для диагностики, профилактики, лечения или мониторинга заболеваний. Сервис оценивает
> техническое качество укладки и области интереса исследования и не формирует медицинское
> заключение. Решение по исследованию принимает врач. Область применения: денситометрия (DXA)
> поясничного отдела позвоночника и проксимального отдела бедра, оборудование GE Lunar Prodigy;
> вне этой области применения результат не определён. Источник формулировки —
> `config.yaml → intended_use`, та же строка отдаётся в `/api/health` и показана в веб-интерфейсе.
([код на GitHub](https://github.com/neuropeopleteam-afk/densitoai)).
Ключевые документы: `docs/METRICS_REPORT.md` (метрики ТЗ §8.4 с 95 % ДИ),
`docs/ENGINEERING_REPORT.md`, `docs/qa/` (ответы организаторов и их учёт).

---

## Содержание

0. [Быстрый старт](#быстрый-старт)
1. [Назначение, возможности, ограничения](#1-назначение-возможности-ограничения)
2. [Структура проекта](#2-структура-проекта)
3. [Системные требования и зависимости](#3-системные-требования-и-зависимости)
4. [Сборка и запуск контейнера](#4-сборка-и-запуск-контейнера)
5. [Запуск без контейнера](#5-запуск-без-контейнера)
6. [Описание API](#6-описание-api)
7. [Формат входных и выходных данных](#7-формат-входных-и-выходных-данных)
8. [Модель, предобработка, постобработка](#8-модель-предобработка-постобработка)
9. [Известные ошибки и их обработка](#9-известные-ошибки-и-их-обработка)
10. [Качество и честные ограничения по критериям](#10-качество-и-честные-ограничения-по-критериям)
11. [Процедура обучения / дообучения](#11-процедура-обучения--дообучения)
12. [Руководство пользователя (кратко)](#12-руководство-пользователя-кратко)
13. [Руководство по развёртыванию](#13-руководство-по-развёртыванию)
14. [Соответствие ТЗ и статус](#14-соответствие-тз-и-статус)

---

## 1. Назначение, возможности, ограничения

**Назначение.** Автоматизированная первичная оценка качества DXA-исследований для
снижения нагрузки на врачей-рентгенологов: выявление снимков, которые нужно переснять или
перепроверить.

**Что делает.**
- Принимает папку, zip-архив или одиночный файл DICOM; рекурсивно находит все DICOM (по
  расширению `.dcm/.dicom/.dic/.ima` или по сигнатуре `DICM`), в том числе внутри вложенных
  архивов.
- Для каждого изображения определяет область: «Поясничный отдел позвоночника» или
  «Проксимальный отдел бедра» (сторона бедра определяется внутренне для выбора модели, в
  отчёт не выводится — по ТЗ).
- Оценивает нарушения по закрытому перечню:
  - позвоночник: «Некорректная укладка», «Не выравнена ось позвоночника»,
    «Присутствуют посторонние предметы»;
  - бедро: «Некорректная укладка», «Некорректная область интереса».
- Возвращает `quality_class` (0/1), список нарушений через `;`, оценку риска нарушения
  `quality_prob` ∈ [0, 1] (ранговая оценка, согласованная с классом, а не калиброванная вероятность — §7,
  `docs/CALIBRATION.md`), статус обработки и время обработки каждого файла.
- Пишет CSV (основной артефакт) + XLSX + отладочный CSV с промежуточными признаками и
  лог-файл.
- Предоставляет HTTP API для пакетной обработки (ТЗ п.3.2) и CLI.

**Что НЕ делает / ограничения.**
- Не ставит диагноз и не оценивает МПК/T-score — только качество снимка.
- Бонусы (§14): `additional_series.zip` (DICOM SR на исследование, наложение, SEG — ТЗ п. 2.7) пишется рядом с
  CSV по умолчанию (отключить — `DENSITO_SERIES_ZIP=0`); отдельные PNG-визуализации, SR на снимок и предложение
  по полю сканирования — по флагам (`--visualize-dir`, `--sr-dir`, `--roi-autocorrect-dir`). 9 колонок
  `results.csv` бонусы не меняют.
- Обучена на 499 изображениях (100 исследований) одного аппарата (GE Lunar, размеры кадра
  300×317 / 280×291 / 248×291 px). На снимках других аппаратов и разрешений качество не
  гарантируется — такие входы отмечаются в отладочном CSV как «нестандартные».
- Для ряда критериев обучающих позитивов очень мало (4–17), метрики по ним статистически
  ненадёжны — см. раздел 10. **Это важно понимать при интерпретации результатов.**
- Не гарантирует пропускную способность: время на снимок зависит от машины и соседней нагрузки (на боевом сервере в
  тихих условиях медиана 0,54 с, под чужой нагрузкой 1,84 с (прогон версии 2.3.2), локально 0,07 с — `docs/PERFORMANCE.md`); граница ТЗ —
  3 мин на исследование.

---

## 2. Структура проекта

```
densitoai/
├── README.md                    ← этот файл
├── config.yaml                  ← единый конфиг: строки формата, пороги, веса, fallback-правила
├── requirements.txt             ← зафиксированные версии (pip freeze)
├── Dockerfile                   ← python:3.12.8-slim-bookworm, CPU
├── docker-entrypoint.sh         ← режимы контейнера: batch | api | verify | test | receiver | api+receiver
├── docker-compose.yml           ← сервисы densito-verify, densito-batch, densito-api, densito-receiver (лимиты 2 CPU / 3 ГБ)
├── build_and_run.sh             ← сборка + запуск (Linux)
├── .dockerignore
├── src/
│   ├── inference.py             ← ГЛАВНЫЙ пайплайн: CLI, чтение DICOM, регион, признаки,
│   │                               модели, стэкинг, запись CSV/XLSX, валидатор формата
│   ├── api_server.py            ← FastAPI: /api/health, /api/analyze, /api/batch, /api/results
│   ├── geometry_features.py     ← Контур A: сегментация кости, ось, металл, ROI (мм)
│   ├── embeddings.py            ← Контур B: замороженный EfficientNet-B0 → 1280-d
│   ├── build_dataset.py         ← сборка labels_full.csv из разметки .xlsx и DICOM
│   ├── extract_all_features.py  ← признаки контура A для всего трейна → data/geometry_features.csv
│   ├── train_stacked.py         ← обучение: GroupKFold(5) по исследованию, LR, ранг-стэкинг, пороги, ДИ
│   ├── train_final_models.py    ← финальные модели по критериям + any-модели (geom/emb)
│   ├── eval_oof_metrics.py      ← отчёт по метрикам ТЗ §8.4 с 95 % ДИ → docs/metrics_oof_full.md
│   ├── visualize_report.py      ← [бонус] прозрачный оверлей: измерения + оценки моделей по критериям
│   ├── dicom_sr.py              ← [бонус] DICOM Structured Report: на снимок (--sr-dir) и один на исследование (--sr-study)
│   ├── schema_check.py          ← проверка строк CSV / ответа API по JSON Schema (schema/)
│   ├── extras.py                ← предупреждения вне 9 колонок: белые линии, OOD-gate (models/ood_gate.pkl), эндопротез, когерентность
│   ├── auto_roi.py              ← [бонус] предложение исправленного ROI бедра (диагностический PNG)
│   ├── segmentation_export.py   ← [бонус] сегментация структур: DICOM SEG + PNG-маска + JSON контуров (--seg-dir)
│   ├── dicom_receiver.py        ← приёмник DICOM (C-STORE/C-ECHO, pynetdicom) → /api/analyze
│   └── hip_features.py, hip_eval.py, train_multilabel.py ← исследовательские скрипты (бедро, старая CNN)
├── models/
│   ├── MODEL_CONTRACT.md        ← формат .pkl, который ожидает инференс
│   ├── model_<region>_<crit>_geom.pkl / _emb_pca.pkl   ← обученные модели по критериям
│   ├── model_<region>_any_geom.pkl / _any_emb_pca.pkl  ← any-модели «есть нарушение» (бедро — единая, под обоими именами сторон)
│   ├── model_region_emb.pkl     ← контентный детектор региона (для нестандартной ширины кадра)
│   ├── models_manifest.json     ← перечень моделей, версии, дата обучения
│   ├── oof_stacked_<region>_<crit>.csv                 ← OOF-скоры (референс для ранга)
│   ├── metrics_summary.json     ← метрики, пороги, ДИ по каждому критерию
│   └── torch_home/              ← веса EfficientNet-B0 (offline, 21 МБ)
├── data/
│   ├── labels_full.csv          ← 499 строк: файл, регион (сторона бедра — анатомический детектор), метки 7 критериев; см. data/README.md
│   ├── geometry_features.csv    ← признаки контура A на трейне (медианы для импутации)
│   └── embeddings.npy           ← эмбеддинги трейна (не входит в образ)
├── schema/                      ← JSON Schema строки результата, ответа /api/analyze и сводки по партии
├── tools/
│   ├── verify.sh, verify_checks.py, verification_report.py ← самопроверка без сети (фантомы, детерминизм, sha256, эталон, словарь организаторов)
│   ├── make_phantoms.py, hash_weights.py, pii_scan.py, make_release.sh, offline_check.sh
│   ├── validate_sr.py           ← валидатор DICOM SR исследования
│   ├── make_model_card.py       ← генерирует models/MODEL_CARD.md из metrics_summary.json
│   ├── nested_gate.py, pixel_hash.py ← nested repeated GroupKFold для выбора весов стэкинга (К2)
│   ├── paired_gate.py           ← калиброванный гейт приёмки: парный кластерный бутстрап по исследованиям, sign-flip тест, поправка Холма (идея 11)
│   ├── calibration_report.py    ← отчёт о калибровке quality_prob и скоров критериев (ECE/Brier/наклон, SVG)
│   ├── uncertain_zone.py        ← кривая «покрытие → полнота» зоны «не уверен» по δ на OOF (ДИ по исследованиям, SVG, проекция на боевой прогон)
│   ├── baseline_table.py        ← таблица бейзлайнов на тех же OOF-строках → docs/BASELINES.md, docs/baselines.json
│   ├── stress_set.py            ← стресс-набор битых и нестандартных входов (проверка 18 verify) → docs/STRESS_SET.md
│   ├── measurement_check.py     ← поверка измерительного контура A на фантомах: поворот/сдвиг в мм → docs/MEASUREMENT_CHECK.md
│   ├── perf_passport.py         ← паспорт производительности: секунды на снимок и память с условиями → docs/PERFORMANCE.md
│   ├── compare_regressions.py   ← сравнение двух выгрузок регрессии по image_uid (класс, типы, quality_prob, статус)
│   ├── department_summary.py    ← сводка по партии для отделения (JSON/Markdown/CSV, ДИ Уилсона 95 %)
│   ├── extras/                  ← аудиты К8/К9 (белые линии, OOD-gate, эндопротез, когерентность) → docs/EXTRAS_STATUS.md
│   └── review/                  ← инструмент слепой ревизии для рентгенолога (галерея, ориентиры, kappa/PCK)
├── web/
│   └── index.html               ← [бонус] веб-интерфейс (загрузка DICOM, таблица, оверлеи, SR, CSV)
├── tests/
│   ├── test_inference_format.py ← smoke-тест: реальные + битые входы, формат, детерминизм
│   ├── test_api_isolation.py    ← 88 проверок API (изоляция jobs и бонус-файлов, код доступа, traversal, лимиты, битый zip, отказ по области)
│   ├── test_schema.py           ← JSON Schema результата и ответа API (37 проверок)
│   ├── test_transfer_syntax.py  ← матрица transfer syntax / битности / MONOCHROME1 → docs/TRANSFER_SYNTAX_MATRIX.md
│   ├── test_calibration_report.py ← детерминизм, ECE идеального и константного предиктора, сверка с metrics_oof_full.json
│   ├── test_uncertain_zone.py   ← монотонность по δ, воспроизведение текущей точки (13.9 % / 9.4 %), детерминизм, валидность JSON/SVG
│   ├── test_perf_passport.py    ← паспорт производительности: CSV-статистика, согласованность MD и JSON, режим тихого сервера
│   ├── test_dicom_receiver.py   ← SCP на свободном порту, C-ECHO/C-STORE фантомов, inbox, журнал, HTTP-путь
│   ├── test_baseline_table.py, test_stress_set.py, test_department_summary.py, test_measurement_check.py ← бейзлайны, стресс-набор, сводка по партии, поверка измерений
│   ├── stress/                  ← ожидания стресс-набора (expected_stress.csv)
│   ├── phantoms/                ← 15 синтетических DICOM-фантомов + MANIFEST.json + expected_results.csv (эталон verify)
│   └── sample_test_zip/         ← сюда можно положить образец организаторов «Для теста.zip» (в репозитории его нет; без него тесты берут фантомы)
├── docs/
│   ├── EXPERT_TESTING_GUIDE.md  ← инструкция эксперту-тестировщику (1 страница)
│   ├── ENGINEERING_REPORT.md    ← инженерный отчёт о сдаче (файлы, статус Docker, расхождения с ТЗ)
│   ├── METRICS_REPORT.md        ← метрики ТЗ §8.4 с 95 % ДИ, скорость, надёжность
│   ├── EVIDENCE.md              ← данные, лицензии, протокол валидации, отвергнутые гипотезы, ограничения
│   ├── DZM_CONFORMANCE.md       ← соответствие ТЗ и желательным пунктам: сделано / частично / не делаем
│   ├── VERIFICATION.md          ← инструкция технической группе: офлайн-проверка образа за 5 команд
│   ├── CALIBRATION.md           ← можно ли читать quality_prob и p_cal как вероятность (OOF, ДИ по исследованиям)
│   ├── calibration/             ← calibration.json + reliability_*.svg (агрегаты, без кадров)
│   ├── UNCERTAIN_ZONE.md        ← что даёт зона «не уверен» и что дали бы другие ширины полосы: кривая «покрытие → полнота» по δ
│   ├── uncertain_zone.json, uncertain_zone/ ← точки кривой с ДИ и coverage_recall_*.svg (агрегаты, без кадров)
│   ├── DICOM_RECEIVER.md        ← подключение аппарата/PACS, SOP-классы, transfer syntax, ограничения
│   ├── BASELINES.md, baselines.json ← бейзлайны на тех же OOF-строках (tools/baseline_table.py)
│   ├── STRESS_SET.md            ← таблица случаев стресс-набора (tools/stress_set.py --md)
│   ├── MEASUREMENT_CHECK.md, measurement_check.json ← акт поверки измерительного контура на фантомах (tools/measurement_check.py)
│   ├── PERFORMANCE.md, perf_passport.json ← паспорт производительности: секунды на снимок и память, условия (tools/perf_passport.py)
│   ├── NESTED_GATE_REPORT.md    ← nested CV вентильного стэкинга (К2): полный отчёт по повторам
│   ├── EXTRAS_STATUS.md, OOD_GATE_REPORT.md, WHITE_LINES_AUDIT.md ← аудит белых линий, OOD-gate, эндопротез, когерентность (К8/К9)
│   ├── TRANSFER_SYNTAX_MATRIX.md, LICENSES_AND_DATA_AUDIT.md ← измеренная матрица форматов; лицензии и аудит ПДн
│   ├── qa/                      ← транскрипт и разбор Q&A с организаторами, учёт в решении
│   │   ├── JURY_QA.md, DEMO_SCRIPT.md ← вопросы жюри с ответами и источниками; сценарии питча (3 мин), демонстрации сайта (5–7 мин) и ролика
│   ├── presentation/DensitoAI_LCT2026_prezentatsiya.pdf ← презентация для жюри, 23 слайда по шаблону организаторов
│   ├── LETTER_TO_ORGANIZERS.md  ← сопроводительное письмо к сдаче
│   ├── FINAL_PLAN.md, review_fable5.md ← проектные решения
│   └── *.png                    ← примеры визуализаций признаков
└── outputs/                     ← результаты запусков (не входит в образ)
```

---

## 3. Системные требования и зависимости

| | Минимальная конфигурация | Рекомендуемая |
|---|---|---|
| CPU | 2 ядра x86-64 | 8 ядер |
| RAM | 4 ГБ | 8–16 ГБ |
| Диск | 3 ГБ (образ 2,58 ГБ, архив релиза 659 МБ) | 5 ГБ |
| GPU | **не требуется** | не используется (CPU-сборка torch) |
| ОС | Linux x86-64 с Docker ≥ 20.10 (или Python 3.12 без Docker) | Ubuntu 22.04/24.04 |

Производительность (измерено, CPU sandbox, 1 поток на файл): чтение + геометрия ≈ 5–15 мс на
файл; с эмбеддингами EfficientNet-B0 ≈ 0.1 с на файл; инициализация backbone ≈ 3 с один раз.
Исследование из 3–5 снимков обрабатывается за секунды: на боевом сервере медиана 0,54 с на файл, худшее
исследование 8,1 с (требование ТЗ п.2.7 — ≤ 3 мин; `docs/PERFORMANCE.md`).

Единый benchmark (`tests/robustness_suite.py`, 81 снимок разметки, 2 vCPU, без GPU, оба бэкбона):
время на снимок **p50 0.04 с, p95 0.07 с** без бонус-файлов; на боевом контейнере с визуализациями,
SR и авто-ROI ≈ 2 с/файл, первый запрос после старта контейнера ≈ 18 с (загрузка моделей).
Устойчивость к искажениям входа (теги, MONOCHROME1, 16 бит, UID, шум, гамма, resize, вложенные и
битые zip, одинаковые имена) — `docs/ROBUSTNESS_REPORT.md`.
Паспорт производительности — `docs/PERFORMANCE.md` (`python tools/perf_passport.py`, числа в `docs/perf_passport.json`):
на боевом сервере (4 vCPU, Docker, образ `densitoai:2.5.0`, один процесс `src/inference.py`, OMP_NUM_THREADS=2) в тихих
условиях — регрессия 2.4.0 на 499 файлах: медиана `time_of_processing` 0,54 с, p95 1,76 с, максимум 4,84 с
(позвоночник 0,82 с, бедро 0,40 с), 499 файлов за 368 с одним процессом (стена контейнера с загрузкой моделей 6 мин 24 с);
тот же сервер под чужой нагрузкой (CSV прогона версии 2.3.2, не 2.4.0): медиана 1,84 с, p95 6,11 с, максимум 12,68 с; локально (2 vCPU,
параллельная нагрузка, ориентировочно) медиана 0,07 с (позвоночник 0,10 с, бедро 0,05 с), холодный старт 3,3 с, пиковый RSS
процесса около 0,55 ГБ; RSS контейнера API в простое 536 МиБ, образ Docker 2,58 ГБ. Числа зависят от машины и соседней
нагрузки; пропускная способность не обещается; сравнений с внешними ориентирами нет.
На финальном стенде (44 vCPU, 256 ГБ, 2×H200) GPU не задействуется — это осознанно:
модель лёгкая, а CPU-сборка исключает проблемы с драйверами.

Зависимости (полный список с версиями — `requirements.txt`): Python 3.12, torch 2.14.0+cpu,
torchvision 0.29.0+cpu, scikit-learn 1.9.1, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5,
pydicom 3.0.2, opencv-python-headless 5.0.0.93, scikit-image 0.26.0, PyYAML, openpyxl,
fastapi 0.141.1, uvicorn 0.53.0. **Версию scikit-learn менять нельзя** без пересохранения
`.pkl`.

---

## 4. Сборка и запуск контейнера

Требуется Docker (BuildKit). Всё выполняется из корня проекта. Если образ уже загружен из релиза (`docker load`,
«Быстрый старт»), запускайте с `NO_BUILD=1`: без него `run`, `api` и `test` сначала пересобирают образ
(`docker build --pull`, нужен интернет).

```bash
# 1) Собрать образ densitoai:2.5.0 (внутри сборки запускается самопроверка на образце)
./build_and_run.sh build

# 2) Пакетная обработка: входная папка (или zip) → выходная папка (с готовым образом — NO_BUILD=1 ./build_and_run.sh run …)
./build_and_run.sh run /path/to/dicom_folder ./outputs
#    результат: ./outputs/results.csv  (+ results.xlsx, results_debug.csv, results.log)
#    с 2.4.1 рядом — additional_series.zip (дополнительные серии, ТЗ п. 2.7; см. ниже)

# 3) HTTP API на порту 8000 (Swagger: http://localhost:8000/docs)
./build_and_run.sh api 8000 /path/to/dicom_folder ./outputs

# 4) Самопроверка внутри контейнера
./build_and_run.sh test
```

Эквивалент «руками» (`--user` — чтобы контейнер мог писать в папку вывода, §13 п. 7):

```bash
docker build -t densitoai:2.5.0 .
mkdir -p outputs
docker run --rm --user "$(id -u):$(id -g)" -v /path/to/input:/data/input:ro -v "$PWD/outputs:/data/output" densitoai:2.5.0 batch
docker run --rm -p 8000:8000 --user "$(id -u):$(id -g)" -v "$PWD/outputs:/data/output" densitoai:2.5.0 api
# после этого http://localhost:8000/ — рабочий кабинет (загрузка снимков, карточки решений,
# история запросов), http://localhost:8000/docs — Swagger. Интерфейс лежит внутри образа,
# ничего не тянет из интернета: ни одного внешнего src/href, шрифты и изображения локальные.
# С 2.4.1 Swagger /docs тоже работает без интернета (файлы swagger-ui лежат в образе, web/assets/swagger/),
# а страницы /docs.html и /index.html отдаёт и локальный контейнер, не только сайт.
# любые аргументы inference.py можно передать после batch:
docker run --rm --user "$(id -u):$(id -g)" -v ...:/data/input:ro -v ...:/data/output densitoai:2.5.0 batch --no-embeddings --limit 50
```

**Журнал и «Проверить сервис на своих снимках» при запуске на своём компьютере (2.4.1).** Команда `api` выше
запускает кабинет без учётных записей: вкладка «Журнал исследований» и страница `/expert/` в этом случае закрыты и
отвечают 503 «Журнал исследований не настроен». Для проверки на своём компьютере включите открытый демо-режим:

```bash
docker run --rm -p 8000:8000 --user "$(id -u):$(id -g)" -e DENSITO_REGISTRY_OPEN=1 -v "$PWD/outputs:/data/output" densitoai:2.5.0 api
```

Открытый режим (`DENSITO_REGISTRY_OPEN` = `1`, `true` или `yes`, `src/registry.py: open_mode`) — только для локальной
проверки: вход не нужен, доступ сервис не ограничивает, полные ФИО не показываются, роль администратора недоступна.
В отделении открытый режим не включают — техгруппа создаёт учётные записи командой `python src/registry.py add-user`
(§6, «Журнал исследований отделения»).

**Время в журнале (2.4.1).** В образе задан московский пояс (`ENV TZ=MSK-3`): даты и время в журнале исследований и в
названиях экспертных проверок — московские. Другой пояс задаётся при запуске POSIX-строкой `-e TZ=<POSIX-строка>`,
например для Новосибирска `-e TZ=NOVT-7`. На 9 колонок `results.csv` пояс не влияет (`time_of_processing` — секунды).

**Zip-архив дополнительных серий (ТЗ п. 2.7, 2.4.1).** В пакетном режиме рядом с `results.csv` пишется
`additional_series.zip`: DICOM SR на исследование, наложение как DICOM Secondary Capture и DICOM SEG. Путь задаётся
флагом `--series-zip PATH`, отключить — `DENSITO_SERIES_ZIP=0`. В API — `GET /api/results/{job_id}/additional_series.zip`
(с кодом доступа, как остальные файлы запроса) и поле `additional_series_zip_url` в ответе `/api/analyze`; в кабинете —
кнопка «Скачать дополнительные серии (zip)». Архив добавляет около 33 мс на снимок (парный замер на 120 снимках, 3 повтора, 2 vCPU); регрессия 2.4.1 на сервере
(4 vCPU, 499 снимков, 100 исследований): худшее исследование — 8,1 с при допуске ТЗ 3 минуты (`docs/PERFORMANCE.md`). Архив не меняет 9 колонок `results.csv`; ошибка упаковки
пишется в журнал и не меняет код возврата. Имена внутри архива — `study_NNN_<StudyInstanceUID>/image_NNN_<область>_overlay_SC.dcm`,
`..._SEG.dcm`, `study_SR.dcm` и индекс `series_index.csv`, без ФИО и исходных путей. `POST /api/batch` пишет архив
`<имя CSV>_additional_series.zip` рядом с CSV и возвращает путь в поле `additional_series_zip`.
`/redoc` отключён (грузился с внешнего CDN); схема — `/openapi.json`, интерфейс — `/docs`.

Через docker compose:

```bash
INPUT_DIR=/path/to/input OUTPUT_DIR=./outputs docker compose run --rm densito-batch
docker compose up densito-api
```

Приём снимков по DICOM (C-STORE) без ручной загрузки — команда `receiver` и режим `api+receiver`
(подробно: `docs/DICOM_RECEIVER.md`):

```bash
docker compose up -d densito-api densito-receiver          # API :8000 + приёмник DICOM :11112 (AE Title DENSITOAI)
docker run -d -p 8000:8000 -p 11112:11112 --user "$(id -u):$(id -g)" -v "$PWD/outputs:/data/output" densitoai:2.5.0 api+receiver
python -m pynetdicom echoscu  <ip> 11112 -aec DENSITOAI                      # проверка связи
python -m pynetdicom storescu <ip> 11112 tests/phantoms/study_01 -r -aec DENSITOAI
```

Принятые снимки складываются в `/data/output/inbox/<study_uid>/<sop_uid>.dcm` и после закрытия
ассоциации (или 5 с тишины) уходят в `/api/analyze` тем же путём, что и загрузка через кабинет:
результат — `/data/output/jobs/<job_id>/` (CSV, XLSX, SR, PNG), коды запроса и доступа — в
`inbox/<study_uid>/job.json`, журнал приёма без персональных данных — `inbox/receiver_log.csv`.
Принятые DICOM после анализа остаются в inbox (автоудаление не выполняется; `DENSITO_INBOX_KEEP=0`
включает удаление после успешного анализа).
Переменные: `DENSITO_RECEIVER_PORT` (11112), `DENSITO_RECEIVER_AET` (DENSITOAI), `DENSITO_INBOX`,
`DENSITO_INBOX_KEEP` (1), `DENSITO_API_URL` (`http://127.0.0.1:8000`), `DENSITO_RECEIVER_ALLOWED_AET`.
Без TLS — только внутренняя сеть отделения.

Переменные окружения контейнера: `DENSITO_INPUT` (по умолчанию `/data/input`),
`DENSITO_OUTPUT` (`/data/output/results.csv`), `DENSITO_PORT` (8000), `OMP_NUM_THREADS` (2),
`DENSITO_NO_EMBEDDINGS=1` (отключить контур B); с 2.4.1 — `DENSITO_REGISTRY_OPEN=1` (открытый демо-режим журнала, только
локально), `TZ` (по умолчанию `MSK-3`), `DENSITO_SERIES_ZIP=0` (не писать `additional_series.zip`).

### Запуск на Windows (2.4.1)

Проверено владельцем на Windows 11 с Docker Desktop (`PLAN.md`, 27.09). Команды — в PowerShell.

1. Docker Desktop с движком WSL 2 (в Windows включены компоненты WSL и VirtualMachinePlatform).
2. Образ из релиза: сверьте контрольную сумму `Get-FileHash .\<файл>.tar.gz -Algorithm SHA256` со значением в
   `SHA256SUMS`, затем загрузите образ: `docker load -i .\<файл>.tar.gz`.
3. Самопроверка без сети: `docker run --rm --network none densitoai:2.5.0 verify`.
4. Пакетный режим: `docker run --rm --network none -v ${PWD}\input:/data/input:ro -v ${PWD}\outputs:/data/output densitoai:2.5.0 batch`.
   Zip-архив кладите во входную папку целиком и **не распаковывайте** его через «Извлечь всё» в Проводнике или
   `Expand-Archive` в PowerShell: они портят кириллические имена в кодировке cp866 из архивов, созданных в Windows,
   и тогда сервис честно сообщает, что DICOM не найдено (CSV только с шапкой). Сервис распаковывает zip сам.
5. Кабинет: команда `api` из раздела выше (для журнала на своём компьютере — с `-e DENSITO_REGISTRY_OPEN=1`),
   затем `http://localhost:8000/` в браузере. Остановка контейнера — Ctrl+C в окне PowerShell.

> **Статус проверки сборки.** Образ `densitoai:2.5.0` собран командой выше на сервере linux/amd64
> (Docker 29, базовый образ закреплён по digest); сборка намеренно падает, если внутренняя самопроверка
> `verify` не даёт 18/18 или не совпадают sha256 весов. Размер образа 2,58 ГБ; сборка с кэшем слоёв
> занимает 5–6 мин, первая (скачивание torch ≈ 200 МБ) — 10–15 мин. На собранном образе выполнены
> 8 наборов тестов и регрессия на 499 файлах (`docs/VERIFICATION.md`, `CHANGELOG.md`). Если сборка
> падает на этапе pip — наиболее вероятная причина недоступность индексов pip; см. §13.

---

### Проверка и воспроизводимость

Образ содержит средства самопроверки без сети. Полная инструкция для технической группы — `docs/VERIFICATION.md`.

```bash
docker build --platform linux/amd64 -t densitoai:2.5.0 .        # базовый образ закреплён по digest
bash tools/offline_check.sh densitoai:2.5.0 ./outputs            # = docker run --rm --network none ... verify
# отчёт: outputs/verify/verification_report.html, код возврата 0/1
```

Что проверяет `verify` (`tools/verify.sh`, работает и на хосте: `bash tools/verify.sh`):
15 синтетических DICOM-фантомов (`tests/phantoms/`, генератор `tools/make_phantoms.py`, sha256 в `MANIFEST.json`) —
схема CSV (9 столбцов), строки = файлы, study_uid/image_uid = теги, Failure ровно для 3 битых файлов,
детерминизм двух прогонов (все столбцы кроме time_of_processing), совпадение с эталоном
`tests/phantoms/expected_results.csv`, sha256 весов по `models/WEIGHTS_SHA256.txt` (`tools/hash_weights.py --check`),
совпадение варианта предобработки в `config.yaml` с тем, на котором обучены модели, совпадение источника
эмбеддингов контура B в `config.yaml` с `emb_source` в обученных моделях (и наличие самого файла весов
бэкбона в образе), и **сверка строк выгрузки со словарём организаторов** (`schema/official_dictionary.json`:
значения `violation_type` и `anatomical_region`, разделитель `;`, имя колонки `quality_prob`, паспортный
размер пикселя 1,05 × 0,6 мм), согласованность версии и `config_hash` во всех файлах поставки, безопасность API,
истинная геометрия фантомов по `MANIFEST.json`, серия с визуализацией, **прогон-двойник** и **стресс-набор** битых и нестандартных входов — итого 18 пунктов
(таблица — `docs/VERIFICATION.md`).
Прогон-двойник (`tools/transfer_check.py`) делает копии входа с переименованными файлами и каталогами, случайной
вложенностью, регистром расширений, кириллицей в именах и упаковкой в один или несколько zip, прогоняет инференс
и сверяет с эталоном по ключу (`study_uid`, `image_uid`): регион, класс, нарушения, `quality_prob` и статус должны
совпасть, а нормализованный CSV (без `time_of_processing` и `path_to_study`, строки отсортированы) — побитово по
sha256. На 10 исследованиях (46 снимков) все режимы дали одинаковый CSV; тест — `tests/test_transfer_twin.py`
(внутри образа). Файлы без `StudyInstanceUID` получают `study_uid` от содержимого папки, поэтому их нельзя
раскладывать по разным папкам.
Стресс-набор (`tools/stress_set.py`, проверка 18) собирает из тех же фантомов 24 случая битых и нестандартных
входов: обрезанный до половины и нулевой файл, не-DICOM с расширением `.dcm`, DICOM без PixelData, без Rows/Columns,
без PixelSpacing, MONOCHROME1, 16 бит со знаком, 8 бит, RGB, кадр 8×8 и 4000×3000, Explicit VR Big Endian, Deflated,
RLE Lossless, JPEG Baseline, BitsStored 12/16, чужая модальность (CT — отказ по области применения), постоянный кадр, zip с исправным и битым файлом,
битый zip, вложенность 5, кириллица и пробелы в именах. Все они обрабатываются одним пакетом вместе с 15 фантомами:
битый вход даёт ровно одну строку `Failure` (class 0, `violation_type` пустой, `quality_prob` 0.499999, причина — в
`results_debug.csv`), корректные кодировки дают тот же класс и регион, что исходный фантом, а 15 строк нормы в
смешанном пакете совпадают с одиночным прогоном бит в бит. Таблица случаев — `docs/STRESS_SET.md`, ожидания —
`tests/stress/expected_stress.csv`, тест — `tests/test_stress_set.py`. Пропустить: `VERIFY_SKIP_STRESS=1`; на слабой
машине уменьшить огромный кадр: `VERIFY_STRESS_HUGE=2500x1500`.
Поверка измерительного контура (`tools/measurement_check.py`, вне `verify`) проверяет, что геометрические функции
контура A измеряют то, что заявляют, независимо от разметки экспертов: 13 базовых фантомов (6 из `tests/phantoms/`,
остальные строятся генераторами `tools/make_phantoms.py`) поворачиваются на −15…+15° и сдвигаются на −30…+30 мм в
миллиметровых координатах (PixelSpacing 1.05×0.6 мм, масштаб не меняется), после чего измеренное сравнивается с
заданным. В заявленном диапазоне |угол| ≤ 8°, |сдвиг| ≤ 20 мм: угол оси позвоночника — MAE 0.35°, максимум 1.05°,
наклон 0.907; смещение центра позвоночника — MAE 0.09 мм, максимум 0.26 мм; угол диафиза бедра — MAE 0.04°,
максимум 0.09°; расстояние кости до края кадра — MAE 0.27 мм, максимум 0.52 мм; длина диафиза при сдвиге вниз —
MAE 0.31 мм, максимум 1.55 мм; сторона бедра определена верно в 245/245 кадрах. За пределами диапазона угол оси
занижается (−1.0° при 10°, −2.3° при 15°), при наклоне только столбика позвонков без таза и рёбер угол нелинеен —
это записано как граница применимости. Акт — `docs/MEASUREMENT_CHECK.md`, числа — `docs/measurement_check.json`,
тест — `python tests/test_measurement_check.py` (~30 с, два прогона на детерминизм). Это поверка на фантомах, не замена
клинической валидации.
Сравнение регрессий (`tools/compare_regressions.py`): две выгрузки регрессии (официальный CSV поставки) сравниваются по
`image_uid` — область, класс, состав типов нарушения, `quality_prob`, статус (время не сравнивается); печатаются строки,
сменившие класс или типы, затронутые типы и итог по классам. Запуск: `python tools/compare_regressions.py <кандидат.csv>
<эталон.csv> --allow-types "Некорректная укладка" --region "Поясничный отдел позвоночника"`; код возврата 1, если изменения
вышли за разрешённые типы или затронули строки других областей. Так сверялись выгрузки 2.4.0 и 2.3.2 (`docs/EVIDENCE.md`,
раздел H2).
Проверка на собственных данных: `verify --data /data/input [--expected-sha <sha>]`.

Ресурсы: стенд 2 CPU / 3 ГБ (`docker-compose.yml`: `mem_limit: 3g`, `cpus: 2`; пик памяти инференса около 0,6 ГБ).
Потоки BLAS задаются переменными `OMP_NUM_THREADS`/`MKL_NUM_THREADS` (по умолчанию 2, переопределяются `-e`).

Релиз: `bash tools/make_release.sh 2.5.0` создаёт `dist/densitoai-2.5.0-src.tar.gz` (без outputs/, data-выгрузок и конкурсных
DICOM), `WITH_IMAGE=1` добавляет `dist/densitoai-2.5.0-image.tar.gz` (`docker save | gzip`); контрольные суммы — `dist/SHA256SUMS`.
Матрица форматов DICOM (измерено): `docs/TRANSFER_SYNTAX_MATRIX.md`; лицензии и аудит данных: `docs/LICENSES_AND_DATA_AUDIT.md`.

---

## 5. Запуск без контейнера

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # --extra-index-url для torch уже внутри файла

# пакетная обработка
python src/inference.py --input /path/to/dicoms --output outputs/results.csv --xlsx --debug-csv

# проверка формата уже существующего CSV
python src/inference.py --output outputs/results.csv --validate-only

# API
cd src && python api_server.py --port 8000

# smoke-тест (реальные + заведомо битые файлы, формат, детерминизм)
python tests/test_inference_format.py
```

Аргументы `inference.py`:

| Аргумент | Смысл |
|---|---|
| `-i/--input` | папка, zip-архив или одиночный DICOM |
| `-o/--output` | путь CSV (если указать `.xlsx` — CSV пишется рядом, XLSX по указанному пути) |
| `--xlsx` | дополнительно записать XLSX |
| `--debug-csv` | записать `<output>_debug.csv` с признаками, скорами, порогами, методом, ошибками |
| `--models-dir`, `--config` | переопределить пути к моделям / конфигу |
| `--no-embeddings` | отключить контур B (только геометрия; быстрее, но хуже) |
| `--path-mode relative\|absolute\|name` | как писать `path_to_study` (по умолчанию — относительно входной папки) |
| `--limit N` | обработать первые N файлов (отладка) |
| `--log-file`, `-v` | лог (по умолчанию `<output>.log`), подробный вывод |
| `--validate-only` | только проверить формат CSV и выйти |
| `--visualize-dir DIR` | [бонус] каталог для PNG-оверлеев с геометрическими признаками качества (см. `src/visualize_report.py`) |
| `--sr-dir DIR` | [бонус] каталог для DICOM Structured Report `.dcm` на каждый снимок (см. `src/dicom_sr.py`) |
| `--extras` | дополнительный файл `<output>_extras.csv` (уровень предупреждений, 9 колонок не меняются): `white_lines_flag`, `ood_flag` + расстояние Махаланобиса и fingerprint тегов GE Lunar, `endoprosthesis_suspected`, `study_warnings` (дубликаты кадров, повтор областей, нет области). Статусы и числа — `docs/EXTRAS_STATUS.md`; в API — `details.extras`, `result_extras_csv_url` |
| `--sr-study [--sr-study-dir DIR]` | [бонус] один DICOM Comprehensive SR на исследование, включая норму и Failure, с sha256 файла и пикселей каждого снимка; по умолчанию `<каталог CSV>/sr/<study_uid>_SR.dcm`; проверка — `python tools/validate_sr.py <dir> --csv results.csv` |
| `--roi-autocorrect-dir DIR` | [бонус] каталог для диагностических PNG с предложением коррекции ROI для бедра (см. `src/auto_roi.py`), создаётся только при найденном нарушении |
| `--seg-dir DIR` | [бонус] каталог для экспорта сегментации структур: на каждый Success-снимок `<stem>_seg.dcm` (DICOM Segmentation со ссылкой на исходный SOPInstanceUID), `<stem>_seg.png` (полупрозрачная маска) и `<stem>_seg.json` (легенда, площади, контуры в px и мм); см. `src/segmentation_export.py` |
| `--series-zip PATH` | (2.4.1) путь zip-архива дополнительных серий (ТЗ п. 2.7: DICOM SR на исследование, наложение SC, SEG); в пакетном режиме архив по умолчанию пишется рядом с CSV как `additional_series.zip`, `DENSITO_SERIES_ZIP=0` отключает |

все бонус-флаги по умолчанию отключены и не влияют на основной CSV; исключение с 2.4.1 — `additional_series.zip` в пакетном
режиме (пишется по умолчанию, `DENSITO_SERIES_ZIP=0` отключает), на основной CSV он тоже не влияет.

Переменные окружения: `DENSITO_ROOT`, `DENSITO_MODELS_DIR`, `DENSITO_CONFIG`,
`DENSITO_OUTPUT_DIR` (для API), `TORCH_HOME` (веса backbone).

---

## 6. Описание API

Сервер: FastAPI/uvicorn, `src/api_server.py`, порт 8000, документация OpenAPI — `/docs` (с 2.4.1 открывается без
интернета: swagger-ui лежит в образе, `web/assets/swagger/`).
Все вызовы локальные, внешних запросов сервис не делает.

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/api/health` | статус, версия, число загруженных моделей, активные пороги, включены ли fallback-правила |
| POST | `/api/analyze` | `multipart/form-data`, поле `files` (1..N файлов `.dcm` и/или `.zip`), query `xlsx=true/false`. Ответ JSON: `job_id`, **`job_token`** (код доступа к результатам этого запроса), `summary`, `format_check`, `rows` (строки официального формата), `csv` (текст CSV), имена и ссылки файлов результата — ссылки уже подписаны кодом доступа |
| POST | `/api/batch` | JSON `{"input_dir": "/data/input", "output_csv": "/data/output/results.csv", "xlsx": true, "limit": null}` — обработать папку/архив, уже доступный внутри контейнера (смонтированный том). Ответ: `summary`, `format_check`, пути к файлам, `additional_series_zip` (2.4.1) |
| GET | `/api/jobs/{job_id}` | карточка запроса: сводка, строки без картинок, ссылки на файлы. **Нужен код доступа**: `?t=<job_token>` или заголовок `X-Job-Token` |
| GET | `/api/results/{job_id}/{name}` | файл результата этого запроса (CSV, XLSX, технический CSV, `summary.json`, PNG-оверлей, DICOM SR). **Нужен код доступа** |
| GET | `/api/results/{job_id}/additional_series.zip` | (2.4.1) zip-архив дополнительных серий запроса по ТЗ п. 2.7: DICOM SR на исследование, наложение SC, SEG; ссылка — поле `additional_series_zip_url` в ответе `/api/analyze`. **Нужен код доступа** |
| GET | `/api/jobs` | список всех запросов сервиса — **закрыт**: отдаётся только по админскому ключу (`DENSITO_ADMIN_KEY`, заголовок `X-Admin-Key`). Кабинет ведёт свою историю в браузере, поэтому один пользователь не видит запросы другого |
| GET | `/api/results/{name}` | совместимость: файл результата `/api/batch` из корня папки результатов по имени |
| POST / GET | `/api/results/{job_id}/decisions` | решение специалиста по предложению коррекции области интереса бедра (JSON `{image_uid, decision: "подтверждено" / "отклонено" / "своя", roi_box_px при "своя", specialist, comment}`; код задачи через `?t=` или заголовок `X-Job-Token`); GET — список решений задачи |
| GET | `/api/results/{job_id}/decisions.csv` | выгрузка решений (`;`, BOM) |
| GET | `/api/results/{job_id}/decisions_sr/{study_uid}` | DICOM SR «Решение специалиста по области интереса» (`<study_uid>_SR_decisions.dcm`; основной SR исследования не меняется) |
| GET | `/api/results/{job_id}/summary` | сводка по партии без персональных данных (JSON, схема `schema/department_summary.schema.json`): всего исследований и файлов, доли с нарушением по области, типу нарушения, аппарату (хэш StationName/серийного номера) и дате исследования (день, ISO-неделя), доля Failure с причинами, доля зоны «не уверен» по критериям, список исследований для пересмотра (только UID). Рядом с каждой долей — доверительный интервал Уилсона 95 % и пометка `low_data` при n < 20. Query `top` (размер списка для пересмотра, 1..100), `min_n` (порог «мало данных», 1..1000). **Нужен код доступа** |
| GET | `/api/results/{job_id}/summary.md`, `/summary.csv` | та же сводка в Markdown (для печати и рассылки) и CSV (`;`, BOM; колонки `section;group;subgroup;metric;k;n;rate;ci_low;ci_high;low_data`). **Нужен код доступа** |
| POST | `/api/review` | приём JSON слепой ревизии рентгенолога со страницы `web/review/`, не более 2 МБ; файл сохраняется в `/data/output/review/`, ответ `{"ok": true, "saved": <имя файла>}`. Ничего не отдаёт и на инференс не влияет |
| GET | `/api/registry/status` | включён ли журнал исследований (есть ли учётные записи); без персональных данных |
| POST | `/api/registry/login` | вход в журнал: JSON `{login, password}` → `session` (заголовок `X-Registry-Session` для остальных вызовов, срок 12 ч) |
| GET | `/api/registry/studies` | поиск исследований: `q` (фамилия / ID пациента / № направления / имя файла), `date_from`, `date_to` (ДД.ММ.ГГГГ), `region` (spine/hip), `result` (violation/ok/uncertain/fail/unsupported), `status`, `violation`, `has_comments`, `limit`, `offset`. ФИО и ID в ответе маскированы |
| GET | `/api/registry/studies/{study_uid}` | карточка исследования: снимки со ссылками на карточку задачи, статус и его история, комментарии; `?reveal=1` — полные ФИО, ID, дата рождения, № направления (просмотр записывается в журнал доступа) |
| POST | `/api/registry/studies/{study_uid}/comments`, `/status` | комментарий врача или лаборанта (`{text, image_uid?}`, до 2000 символов) и статус (`new`, `retake`, `retaken`, `accepted`, `disputed`; лаборант — только `retaken`, `disputed`) |
| GET | `/api/registry/studies.csv`, `/api/registry/audit` | выгрузка найденного (ФИО маскированы); журнал доступа — только администратору |

Помимо HTTP, снимки можно передавать по DICOM: приёмник `src/dicom_receiver.py` (Storage SCP,
C-ECHO/C-STORE; SOP-классы CR, DX for Presentation/Processing, Secondary Capture; несжатые transfer
syntax и RLE) вызывает тот же `POST /api/analyze`, поэтому ответ, файлы запроса и код доступа
не отличаются от веб-загрузки. См. `docs/DICOM_RECEIVER.md`.
Примеры:

```bash
curl http://localhost:8000/api/health
curl -F "files=@study.zip" "http://localhost:8000/api/analyze?xlsx=true"   # в ответе job_id и job_token
curl -O "http://localhost:8000/api/results/<job_id>/results.csv?t=<job_token>"
curl -O "http://localhost:8000/api/results/<job_id>/additional_series.zip?t=<job_token>"   # дополнительные серии (2.4.1)
curl "http://localhost:8000/api/results/<job_id>/summary?t=<job_token>"          # сводка по партии (JSON)
curl -O "http://localhost:8000/api/results/<job_id>/summary.md?t=<job_token>"    # то же в Markdown
curl -X POST -H 'Content-Type: application/json' \
     -d '{"input_dir":"/data/input","output_csv":"/data/output/results.csv","xlsx":true}' \
     http://localhost:8000/api/batch
curl -O http://localhost:8000/api/results/results.csv
```

Коды ошибок: 400 — нет файлов; 404 — путь/файл не найден; 413 — превышен лимит загрузки
(`DENSITO_MAX_UPLOAD_MB`, по умолчанию 512 МБ — `src/api_server.py`); 500 — внутренняя ошибка (в теле — тип и
текст). Ошибки отдельных DICOM **не** дают 500 — они попадают в строки со статусом `Failure`.

---

**Предложение коррекции области интереса бедра (дополнительный функционал, требует подтверждения специалистом).**
В каждой строке ответа `/api/analyze` есть поле `roi_suggestion`: для бедра — предложенная область в пикселях и мм,
дефицит поля сканирования в мм, причина и `source: "предложение системы"`; для позвоночника, отказа по области и
Failure — `null`. Само предложение ничего не меняет: решение принимает специалист через
`POST /api/results/{job_id}/decisions`; решения хранятся в `decisions.json` задачи, выгружаются `decisions.csv` и
отдельным DICOM SR. В кабинете для строк бедра показывается рамка предложения поверх снимка и кнопки «Подтвердить»,
«Отклонить», «Своя» (рамку можно растянуть мышью), поле «Специалист»; в таблице — статус решения, кнопки
«Выгрузить решения (CSV)» и «SR решений». Схемы: `schema/roi_decision.schema.json`, поля в
`schema/api_analyze_response.schema.json`. Решение не влияет на 9 официальных полей; ROI аппарата не меняется.

**Сводка по партии для заведующего отделением и старшего лаборанта (дополнительный функционал).**
`GET /api/results/{job_id}/summary` (JSON), `/summary.md`, `/summary.csv` агрегируют `results.csv` и технический CSV
задачи без персональных данных: всего исследований и файлов, доля файлов и исследований с нарушением по области
и типу нарушения, разрез по аппарату (производитель, модель и хэш StationName или серийного номера — исходные
значения и оператор не выводятся) и по дате исследования (день и ISO-неделя), доля и причины Failure, доля зоны
«не уверен» (`needs_review` технического CSV, по критериям), список исследований для пересмотра по наибольшей
`quality_prob` (только UID). Рядом с каждой долей — доверительный интервал Уилсона 95 %; при n < 20 стоит пометка
«мало данных». Знаменатель долей нарушений — успешно обработанные файлы. Аппарат и дата берутся из тегов DICOM при
загрузке (`device_tags.csv` в каталоге задачи; в обезличенных выгрузках, где `StationName`/`StudyDate` заменены
заглушкой, аппарат помечается «не указан», а разрез по датам пуст). Те же файлы строит CLI
`tools/department_summary.py --results results.csv [--debug results_debug.csv] [--dicom-root <папка DICOM>]
--out-dir <папка>` — для одного или нескольких `results.csv`. Схема — `schema/department_summary.schema.json`, тест —
`tests/test_department_summary.py`. Сводка ничего не меняет в 9 официальных полях и в файлах задачи.

**Журнал исследований отделения (дополнительный функционал, `src/registry.py`).** Общий для отделения список
исследований, загруженных через `/api/analyze` (веб или DICOM-приёмник): поиск по фамилии, ID пациента, номеру
направления и имени файла (регистр и «ё» не важны), фильтры по дате исследования, области, результату, статусу и
наличию комментариев; статусы «нужна пересъёмка», «переснято», «принято», «спорно»; комментарии врача и лаборанта к
исследованию или отдельному снимку. Хранение — SQLite в томе выходов (`/data/output/registry.sqlite3`), внутри контура
учреждения; снимки не копируются. Персональные данные: в списке ФИО сокращены до инициалов, ID — до последних цифр;
полные значения открываются кнопкой в карточке, и каждый просмотр пишется в журнал доступа (кто, когда, какое
исследование). Журнал выключен (маршруты отвечают 503 «Журнал исследований не настроен»), пока техгруппа не создала учётные записи:
`docker exec -it densitoai-api-container python /app/src/registry.py --output-dir /data/output add-user <логин> --name "Петрова А. В." --role doctor|lab|admin`.
Пароли — PBKDF2-SHA256 (200 000 итераций), файл `registry_users.json` с правами 0600; сессия — подписанный HMAC
токен, 5 неудачных попыток входа — блокировка на 60 с. Открытый режим без учётных записей — `DENSITO_REGISTRY_OPEN=1`: вход
не нужен, автор комментария — роль из переключателя «Врач / Лаборант» и необязательное имя; доступ в этом режиме
обязан ограничивать внешний контур (на демо-стенде `/api/registry/` закрыт basic auth сайта в nginx). Журнал не влияет на 9 официальных полей и на пакетный
режим. Тест — `tests/test_registry.py`.

**Экспертная проверка сервиса в отделении (дополнительный функционал, `src/expert_review.py`, страница `/expert/`).**
Заведующий формирует проверку — случайную выборку снимков из журнала исследований (примерно половина с нарушением по
решению сервиса, половина в норме; фильтры по области и дате). Врач оценивает каждый снимок по критериям области
(«норма / нарушение / не могу оценить»), видя только чистый кадр без отметок и без решения сервиса. Отчёт по каждому
критерию и по задаче «есть нарушение»: совпадение, чувствительность и специфичность сервиса относительно врача с 95 %
интервалами Уилсона, каппа Коэна, список расхождений; выгрузка ответов в CSV. Ответы хранятся в базе журнала и видны в
карточке исследования; модель по ним не дообучается. Маршруты: `GET/POST /api/expert/sets`, `GET /api/expert/sets/{id}`,
`GET /api/expert/sets/{id}/frame/{pos}.png`, `POST /api/expert/sets/{id}/answers`, `GET /api/expert/sets/{id}/report`,
`GET /api/expert/sets/{id}/answers.csv`, `GET /api/expert/study/{study_uid}`. Чистый кадр сохраняется при загрузке
(`bonus/rowNNNN_frame.png`). Тест — `tests/test_expert_review.py`.

**Второй тур экспертной проверки (2.5.0, `/expert/`).** После «Завершить» можно открыть второй тур: сервис показывает своё
решение только по критериям, где ответ эксперта с ним разошёлся. Эксперт оставляет или меняет ответ. Ответы первого тура
не меняются; чувствительность, специфичность и каппа считаются только по ним. Второй тур не слепой и описывает, как
эксперт использует подсказку сервиса; в `answers.csv` — 4 колонки в конце, в отчёте — сколько ответов изменено, в какую
сторону и совпадение до и после. Тест — `tests/test_expert_second_round.py`.

**Команда лаборанту с основанием (2.5.0, `src/action_evidence.py`).** «Переснять» сервис пишет только когда у нарушения
есть измеримое основание. Если основания нет, команда — «Проверить …» с причиной: например, «Проверить укладку по малому
вертелу: поле шире обычного, оценка менее надёжна» с боковым запасом в мм (порог 51 мм — верхняя терциль по 333 снимкам
бедра, задан до расчёта без меток). Класс в `results.csv` при этом не меняется. У каждого критерия в карточке указана
роль: «измерение» (укладка и ось позвоночника, область интереса бедра) или «подсказка, решает врач» (посторонние предметы,
укладка бедра). Если два независимых контура оценки оси позвоночника разошлись больше чем на 0.45, карточка пишет
«Контуры разошлись — посмотрите ось»; во вложенной проверке это ловит 55 % ошибок оси при нагрузке 24 % снимков
позвоночника (в худшем повторе 25.3 %), сигнал слабый (AUC 0.56). Поля: `details.action_evidence`,
`criteria[].role`, `results_extras.csv` (`action_command`, `action_evidence`, `action_evidence_text`), SR. Эффект ролей
на 40 кадрах слепой проверки не значим (у врача 1 индекс Юдена 0.424 → 0.584, p = 0.375; у врачей 2 и 3 эффекта нет,
`docs/REVIEW_DOCTORS.md`). Тест — `tests/test_action_evidence.py`.

## 7. Формат входных и выходных данных

### Вход
- Папка любой вложенности, zip-архив или одиночный файл. Внутри — DICOM-файлы DXA
  (Modality OT/CR/DX; тестировалось на GE Lunar: uint8 MONOCHROME2, 300×317 px спина,
  280×291 / 248×291 px бедро). Поддерживаются MONOCHROME1 (инверсия), 16-бит, RGB (→ gray),
  многокадровые (первый кадр), Rescale Slope/Intercept.
- Регион определяется по содержимому, а не по имени файла (организаторы подтвердили, что
  суффиксов `_ПОП/_ППОБ/_ЛПОБ` в закрытом тесте не будет): теги DICOM (`BodyPartExamined`,
  `SeriesDescription`, `Laterality`, если заполнены) → ширина кадра (300 px — позвоночник, 280/248 px — бедро; правило
  организаторов: ≥ 290 — позвоночник) → для **нестандартной ширины** (не 300/280/248 —
  другой аппарат, ресемплинг) регион уточняется моделью по содержимому
  (`models/model_region_emb.pkl`: эмбеддинг → логистическая регрессия, OOF-точность 1.00
  на 499 снимках; источник решения пишется в `results_debug.csv: region_src`) → для
  бедра сторона по анатомическому правилу `hip_features.detect_hip_side` (таз всегда
  медиальнее диафиза; старая плотностная эвристика ошибалась на ~24 % снимков и оставлена
  только как резерв). Сторона используется **только внутренне** (выбор модели/порога,
  зеркалирование правого бедра для единой модели) — в отчёт не попадает.

### Выход — CSV `results.csv` (UTF-8, разделитель `,`, `\n`), плюс `results.xlsx`

| Колонка | Тип | Значение |
|---|---|---|
| `path_to_study` | str | путь к файлу относительно входной папки (настраивается `--path-mode`) |
| `study_uid` | str | `StudyInstanceUID`; если тега нет — `hash-<sha1 папки>` |
| `image_uid` | str | `SOPInstanceUID`; если нет — `hash-<sha1 пути>` |
| `anatomical_region` | str | **точно** `Поясничный отдел позвоночника` или `Проксимальный отдел бедра` |
| `quality_class` | int | `0` — качественное, `1` — есть нарушение (= `violation_type` непустой) |
| `violation_type` | str | нарушения из закрытого перечня через `;` (без пробела), пусто если нет |
| `quality_prob` | float | оценка риска нарушения ∈ [0, 1] (для ROC-AUC): ранговый скор стэкинга, согласованный с классом, не калиброванная вероятность — калиброванная величина по критерию лежит в `<crit>_p_cal` (debug-CSV, details API) |
| `processing_status` | str | `Success` / `Failure` (строки по ТЗ п.2.5) |
| `time_of_processing` | float | секунды на файл (чтение → признаки → модели), без учёта записи |

Строка **Failure** (файл не прочитан/не валиден): `quality_class=0`, `violation_type=""`,
`quality_prob=0.499999`, регион — по имени файла/заголовку DICOM, иначе «Проксимальный отдел
бедра». Число (а не NaN) — чтобы CSV всегда парсился как числовой и не ломал подсчёт ROC-AUC у
проверяющих; значение строго меньше 0.5, поэтому для всех строк выполняется «quality_class = 1 ⇔
quality_prob ≥ 0.5» (24.09.2026; до этого было 0.5, что нарушало инвариант).

Дополнительно: `results_debug.csv` (регион и источник его определения, размеры, все
геометрические признаки, `p_geom/p_emb/score/threshold/flag/method` по каждому критерию,
текст ошибки для Failure) и `results.log`. С 2.4.1 в пакетном режиме рядом пишется `additional_series.zip` —
дополнительные серии по ТЗ п. 2.7 (DICOM SR на исследование, наложение SC, SEG; §4).

Формат проверяется автоматически после каждой записи (`validate_output_csv`) и командой
`--validate-only`.

---

## 8. Модель, предобработка, постобработка

### Предобработка
1. `pydicom.dcmread(force=True)`; проверка наличия `PixelData`, размера (64–4096 px),
   неконстантности изображения.
2. Нормализация в uint8: MONOCHROME1 → инверсия; Rescale; RGB → gray; перцентильное окно
   1–99 %.
3. `PixelSpacing`/`ImagerPixelSpacing` из тегов, иначе константа аппарата 1.05 × 0.6 мм
   (в отладочном CSV ставится предупреждение).
4. Определение региона и (для бедра) стороны (см. §7).

### Двухконтурная архитектура (по одному набору моделей на критерий)

**Контур A — геометрия (интерпретируемая).** `geometry_features.py`: сегментация кости
(Otsu + морфология) → признаки:
- позвоночник: угол оси (линейная аппроксимация центров строк маски), кривизна, смещение
  центра относительно кадра, доля ширины кости; детектор металла (яркие компактные области с
  большим перепадом интенсивности) → площадь в мм²;
- бедро: угол диафиза, расстояние ROI до края кадра (мм), доля площади кости.
Признаки → `StandardScaler → LogisticRegression` на критерий.
Для `sp_pos` (2.4.0, H2) в контур A входит третий признак `synth_pos_logit` — логит головы
`models/head_densito_synth.pth` (`src/sppos_head.py`), обученной на синтетических смещениях кадра
без меток; вход головы — тот же канонический эмбеддинг `densito`, что считает контур B, поэтому время
на снимок не растёт. Признак принят по действующему правилу nested (20/20); калиброванный гейт при
6 положительных исследованиях не имеет мощности (ДИ упирается в 0) — `docs/NESTED_GATE_REPORT.md`, часть 5.
Отсутствие файла головы — явная ошибка при старте, а не молчаливый ноль (`config.yaml: features`).
Для `sp_art` (2.5.0) контур A считает плотные участки только в зоне измерения — верхних 70 % протяжённости кости
(`metal_metal_band70_area_log`, `metal_metal_band70_max_gap`, `config.yaml -> geometry_cols.sp_art`): прежний признак
по всему кадру в основном видел крылья подвздошных костей; основание — ТЗ, рис. 3 (предмет вне зоны измерения L1–L4 врач
нарушением не считает). AUC контура A 0.560 → 0.804, стека 0.823 → 0.899 — `docs/NESTED_GATE_REPORT.md`, часть 8.

**Контур B — визуальные эмбеддинги.** Замороженный EfficientNet-B0 (ImageNet), вход
320×192, выход 1280-d → `StandardScaler → PCA(32) → LogisticRegression(L2)` на критерий.
Сеть не дообучается (мало данных); веса лежат в образе (`models/torch_home`).

**Стэкинг.** Скор каждого контура переводится в перцентильный ранг относительно его
OOF-распределения на трейне (`models/oof_stacked_*.csv`), затем
`score = 0.5·rank_geom + 0.5·rank_emb`. Порог по критерию — из `metrics_summary.json`
(правило К3, выбранное в nested: `prevalence` для `sp_pos`, `hip_pos`, `hip_roi` и `prevalence_x1.4` для `sp_axis`,
`sp_art` — поле `threshold_rule` в `metrics_summary.json`; F1-оптимальный порог по OOF не используется с 2.2.0, так как
нестабилен при 10–35 позитивах — `docs/METRICS_REPORT.md`, раздел «Правило порога»). Приоритет источников порога: `config.yaml` → ключ `threshold` внутри `.pkl` →
`metrics_summary.json` (в т.ч. единый блок `hip`) → 0.5.

### Постобработка
- `violation_type` = объединение критериев, где `score ≥ threshold`, в официальных
  формулировках; `quality_class = 1` тогда и только тогда, когда список непустой.
- `quality_prob` = смесь (0.5/0.5, `config.yaml: stacking.any_blend_weight_model`)
  отдельной any-модели «есть нарушение» (геометрия + эмбеддинги) и `max` скоров по
  критериям региона (`max` | `noisy_or`), затем **согласование с классом**
  (`consistent_quality_prob`): `quality_class = 1 ⇔ quality_prob ≥ 0.5`, порядок внутри
  класса сохраняется (монотонное преобразование). Это устранило 79/499 противоречивых строк
  версии 2.0.x и подняло OOF ROC-AUC бинарной задачи (в сборке 2.1.0 с F1-оптимальными порогами — до 0.783 / 0.773,
позвоночник / бедро; в тот же день с nested-выбранными правилами порога К3 — 0.764 / 0.732). В поставке 2.5.0 те же
величины равны **0.846 / 0.760** (`models/metrics_oof_full.json`, `docs/METRICS_REPORT.md`, таблица «Итого»; см. §10).
- **Fallback без моделей.** Если для критерия нет `.pkl` (или он не читается) — работает
  физическое правило-сигмоида по одному признаку (`config.yaml: fallback_rules`), центры
  откалиброваны как prevalence-квантили на трейне, для ROI бедра — порог ТЗ «≥ 2 см от
  края». В отладочном CSV это видно как `method = fallback_rule`, в `/api/health` —
  `fallback_rules_active: true`. Правила заметно слабее моделей.

Все строки формата, пороги, веса, правила — в `config.yaml`; код содержит встроенные
значения по умолчанию на случай отсутствия конфига.

---

## 9. Известные ошибки и их обработка

Принцип: **необработанных исключений нет** (ТЗ п.2.7). Каждый файл обрабатывается в своём
`try/except`; любая ошибка → строка `Failure` + запись в лог с типом и текстом, батч
продолжается. Проверено smoke-тестом (`tests/test_inference_format.py`)
и стресс-набором `tools/stress_set.py` (24 случая, проверка 18 `verify`, таблица — `docs/STRESS_SET.md`) на:

| Ситуация | Поведение |
|---|---|
| Файл не DICOM / пустой / обрезанный | `Failure` («DICOM has no PixelData» / ошибка парсинга) |
| DICOM без `PixelData` | `Failure` |
| Константное (пустое) изображение | `Failure` («constant (blank) image») |
| Размер вне 64–4096 px | `Failure` («image size out of range») |
| Нестандартная ширина кадра (не 300/280/248 px: 512×512, 1024×1400, 350×500…) | CLI: `Success`, регион — моделью по содержимому (`region_src = content_emb`); кадр вне 200–400 × 120–600 px помечается в debug (`region_supported = 0`, `region_support_scope = geometry`). API и кабинет: отказ «вне области применения» |
| Явное противоречие в тегах: чужой `Manufacturer`, модальность CT/MR и т. п., боковая проекция, другая область | `Failure` и в CLI, и в API, причина — в debug (`region_support_reason`) |
| Пустой (0 байт) файл, загруженный через API | HTTP 400 «Файл … пустой», запрос не обрабатывается (загрузите файлы заново без пустого); в CLI пустой файл — строка `Failure` |
| Пустая папка / пустой zip / во входе нет ни одного DICOM | CSV только с заголовком, сообщение в stderr, код выхода 2 |
| Файл конфигурации отсутствует или не читается | Отказ до инференса, код 2; встроенные значения — только при `DENSITO_ALLOW_DEFAULT_CONFIG=1` |
| Пустой multipart-запрос к `/api/analyze` | HTTP 422 с описанием ошибки |
| Файл результата чужого запроса (код запроса известен, кода доступа нет) | HTTP 403; список всех запросов — HTTP 403 без админского ключа. Проверено `tests/test_api_security.py` |
| Загрузка больше `DENSITO_MAX_UPLOAD_MB` (512 МБ) | HTTP 413; предел проверяется во время чтения, файл не загружается в память целиком |
| «Zip-бомба»: архив разжимается больше чем в 200 раз, либо больше 4 ГБ, либо больше 20 000 файлов | HTTP 400 с понятным текстом, распаковка прерывается (пределы: `DENSITO_MAX_ZIP_RATIO`, `DENSITO_MAX_ZIP_UNPACKED_MB`, `DENSITO_MAX_ZIP_MEMBERS`) |
| Нет `StudyInstanceUID` / `SOPInstanceUID` | `Success`, UID заменён на `hash-…` |
| Нет `PixelSpacing` | `Success`, константа аппарата, предупреждение в debug |
| Невалидные UID (частое у анонимизированных файлов) | предупреждение pydicom подавлено, обработка штатная |
| MONOCHROME1 / 16-бит / RGB / многокадровый | нормализуется, `Success` |
| DICOM без `Rows`/`Columns` при наличии `PixelData` | `Failure` (`Missing required element … Columns`) |
| Битый zip (не архив) / битый файл внутри исправного zip | `Failure` одной строкой на архив / на файл; соседние файлы архива обрабатываются |
| Explicit VR BE, Deflated LE, RLE Lossless, BitsStored 12/16, вложенность 5, кириллица в именах | `Success`, класс равен исходному фантому (стресс-набор) |
| JPEG Baseline | `Success` при декодере Pillow (в образе есть); без декодера — `Failure` с причиной в debug |
| Доля кости вне 1–90 % | `Success`, флаг «non-standard image» в debug |
| `.pkl` отсутствует или битый | WARNING, fallback-правило по критерию |
| Backbone (torch) не инициализируется | WARNING, контур B отключён, работает контур A |
| Вложенный zip не распаковывается | WARNING, архив пропущен |
| Не найдено ни одного DICOM | CSV только с заголовком, сообщение в stderr, код возврата 2 (как в строке «Пустая папка» выше) |
| Не-DICOM файлы (txt, png, csv…) | игнорируются, строк не создают |
| Снимок без явного противоречия в тегах (теги модальности и производителя пустые), но не похожий на денситометрию | `Success`, девять колонок ТЗ заполнены как обычно, и вход помечен нетипичным в бонусном слое: `ood_flag = true`, `ood_reason` с причиной. Проверено `tests/test_ood_foreign.py` на синтезированной картине грудной клетки: расстояние Махаланобиса 701.9 против порога 186.7 плюс экспозиция вне диапазона |

Код возврата CLI: 0 — CSV записан и в нём есть хотя бы одна строка (даже если все строки Failure); 2 — во входе
нет ни одного DICOM, ошибка конфигурации или фатальная ошибка (входной путь не существует, нет прав на запись
результата); 1 — только в режиме `--validate-only`, если формат CSV не прошёл проверку.

Чужой аппарат: отказ (`Failure`) гарантирован, когда тег производителя или модели **заполнен** и явно не GE Lunar
Prodigy. Пустой тег сам по себе не отвергается — тогда решает проверка геометрии кадра (запись в debug CSV; API и
кабинет по ней выдают отказ).

Почему нетипичный снимок всё равно получает область и класс: девять колонок ТЗ (п. 2.5) не содержат
значения «не знаю», а `anatomical_region` ограничен закрытым перечнем строк. Поэтому контракт заполняется
всегда, а сомнение выносится в бонусный слой (`ood_flag`, `ood_reason`, `needs_review`, `risk_level`) и
показывается оператору в кабинете. Так формат остаётся машинно-читаемым для заказчика, а человек видит
предупреждение.

---

## 10. Качество и честные ограничения по критериям

Данные: 499 изображений из 100 исследований (166 позвоночник, 333 бедро), один аппарат
(GE Lunar Prodigy), разметка одного эксперта. Валидация — OOF с группировкой по исследованию
(утечек между снимками одного пациента нет), 95 % ДИ — bootstrap по исследованиям.
Полные таблицы в формате ТЗ §8.4 (чувствительность, специфичность, сбалансированная точность,
F1, ROC-AUC, PR-AUC, macro-F1, ДИ) — **`docs/METRICS_REPORT.md`**; воспроизведение —
`python src/eval_oof_metrics.py`.

**Бинарная задача «есть нарушение» (quality_class / quality_prob), OOF:**

| Область | n / n_pos | Sens | Spec | Bal.Acc | F1 [95 % ДИ] | ROC-AUC [95 % ДИ] | PR-AUC |
|---|---|---|---|---|---|---|---|
| Позвоночник | 166 / 60 | 0.82 | 0.77 | 0.80 | 0.74 [0.57; 0.85] | 0.85 [0.73; 0.94] | 0.74 |
| Бедро | 329 / 92 | 0.52 | 0.81 | 0.66 | 0.52 [0.31; 0.68] | 0.76 [0.63; 0.86] | 0.64 |

**Как читать `quality_prob` и скоры критериев (`docs/CALIBRATION.md`).** `quality_prob` — грубая, а не точная
вероятность: на OOF ECE (10 бинов) 0.14 для позвоночника и 0.15 для бедра, Brier 0.166 / 0.198 против 0.231 / 0.201 у
константы «доля позитивов»; наклон логистической рекалибровки 0.91 / 0.58 — значения в зоне класса 1 завышены.
`quality_prob >= 0.5` означает «сервис поставил класс 1»: при 0.5–0.8 нарушение подтверждается у 44 % кадров
позвоночника, при >= 0.8 — у 75 %; для бедра при 0.5–0.9 — у 36 %, при >= 0.9 — у 96 %. Внутри класса это ранг:
порядок кадров осмыслен (ROC-AUC 0.846 / 0.760), разница «0.72 против 0.78» — нет. Ранговые скоры критериев
вероятностями не являются (ECE 0.26–0.45, Brier хуже константы у всех пяти); величина `<crit>_p_cal` (Platt,
debug-CSV и API details) при cross-fit проверке лучше константы по Brier у всех пяти критериев, но читается как
вероятность только для `sp_art` и `hip_pos` (наклон 0.91 и 0.77, 17 и 24 положительных исследования); для `sp_pos`, `hip_roi`,
`sp_axis` (6, 6, 10 положительных исследований) — лишь как «низкая / средняя / высокая». Порог по правилу prevalence
стоит там, где калиброванная вероятность нарушения по критерию 0.22–0.34: флаг критерия означает «шанс нарушения
от четверти–трети и выше». Воспроизведение: `python tools/calibration_report.py --bins`, тест —
`python tests/test_calibration_report.py`.

**Что даёт зона «не уверен» (`docs/UNCERTAIN_ZONE.md`).** Текущее правило (запасы по критериям под квоту 5 %)
помечает 13.9 % строк позвоночника и 9.4 % строк бедра на OOF; полнота среди уверенных строк 0.81 [0.61; 0.97] и
0.47 [0.24; 0.69]. В зону попадают в основном строки со сработавшим флагом (18 из 23 и 20 из 31), пропусков в ней
мало (3 из 11 и 6 из 44) — зона понижает уверенность в найденных нарушениях, а не поднимает полноту. Единая полоса
по скорам критериев вместо пяти запасов не даёт прироста полноты среди уверенных за пределами шума ни при одной
ширине до 20 % строк в зоне (парные ДИ разности содержат 0); полнота с учётом зоны растёт почти линейно с числом
строк в зоне (бедро: +22 строки за +0.065). Кандидаты с приростом за пределами шума — условие
«any-модель расходится с критериями»: для бедра (+1 строка в зону на OOF, полнота среди уверенных 0.47 → 0.64,
парный ДИ [+0.05; +0.32]) и в 2.5.0 для позвоночника (δ = 0.23: полнота среди уверенных 0.81 → 0.89, парный ДИ
[+0.01; +0.19], нижняя граница близка к 0), но пойманные пропуски — не более чем из 6 исследований в каждой области; правило не внедряется, решение отложено до
большего числа положительных исследований. Воспроизведение: `python tools/uncertain_zone.py --markdown`, тест —
`python tests/test_uncertain_zone.py`.

**По типам нарушений, OOF (стек геометрия + эмбеддинги):**

| Область | Тип нарушения | n_pos / n | Sens | Spec | F1 [95 % ДИ] | ROC-AUC [95 % ДИ] | Надёжность |
|---|---|---|---|---|---|---|---|
| Позвоночник | Присутствуют посторонние предметы (`sp_art`) | 35 / 166 | 0.80 | 0.84 | 0.67 [0.39; 0.83] | 0.90 [0.80; 0.96] | **приемлемая** — контур B (эмбеддинги) и с 2.5.0 контур A по плотным участкам в зоне измерения (решение владельца проекта, калиброванный гейт не пройден) |
| Позвоночник | Не выравнена ось позвоночника (`sp_axis`) | 17 / 166 | 0.77 | 0.91 | 0.60 [0.29; 0.82] | 0.89 [0.75; 0.98] | **приемлемая** — работают оба контура: геометрия (угол оси) и контур B на бэкбоне `densito_inv` (К13); позитивов мало (17) |
| Позвоночник | Некорректная укладка (`sp_pos`) | 10 / 166 | 0.70 | 0.93 | 0.50 [0.12; 1.00] | 0.89 [0.72; 1.00] | **низкая** — 10 позитивов, порог prevalence; контур A с признаком головы укладки `synth_pos_logit` (H2, 2.4.0), контур B — наш бэкбон `densito`; предобработка `canonical` (К11) |
| Бедро | Некорректная область интереса (`hip_roi`) | 16 / 329 | 0.56 | 0.97 | 0.51 [0.00; 0.91] | 0.92 [0.73; 1.00] | **хорошая** — единая модель обеих сторон + физические признаки (длина скана, диафиз ниже вертела) |
| Бедро | Некорректная укладка (`hip_pos`) | 79 / 329 | 0.44 | 0.81 | 0.43 [0.22; 0.60] | 0.73 [0.61; 0.82] | умеренная — консервативный порог, признаки ротации ловятся частично; предобработка `canonical` (К11) |

Macro-F1 по типам нарушений: позвоночник 0.59 [0.44; 0.74], бедро 0.47 [0.20; 0.68] (`models/metrics_oof_full.json`).

**Бейзлайны на тех же OOF-строках (`docs/BASELINES.md`, `python tools/baseline_table.py`).** Чтобы было видно, что
даёт архитектура, а не сама выборка, рядом со стеком посчитаны на тех же кадрах и том же разбиении: «всегда норма»,
«всегда нарушение», случайный классификатор по доле позитивов, каждый контур отдельно и логрегрессия на всех
геометрических признаках региона без отбора (GroupKFold(5) по исследованию). ROC-AUC тривиальных бейзлайнов — 0.5;
F1 «всегда нарушение» — нижняя планка 2p/(1+p): sp_pos 0.11, sp_axis 0.19, sp_art 0.35, hip_pos 0.39, hip_roi 0.09
(F1 стека при тех же правилах порога — 0.50 / 0.60 / 0.67 / 0.43 / 0.51). Логрегрессия на всех признаках без отбора даёт
AUC 0.84 / 0.51 / 0.48 / 0.51 / 0.37 против 0.893 / 0.889 / 0.899 / 0.725 / 0.917 у стека — цена отбора признаков при
10–35 позитивах. Стек не везде выше сильнейшего контура по AUC (sp_pos — контур A 0.915; у sp_art в 2.5.0 стек 0.899 уже выше контура B 0.897): вес
0.5/0.5 зафиксирован заранее ради устойчивости, а у sp_pos контур A при пороге по доле позитивов помечает 10 клонов
одного отрицательного исследования и даёт F1 = 0. Таблица описывает различия внутри одной выборки и не доказывает
перенос на другие аппараты и разметчиков.

Предобработка (20.09, К11): вариант выбран по критерию внутри nested (`tools/preproc_gate.py`,
`docs/PREPROC_GATE_REPORT.md`): критерии укладки `sp_pos` и `hip_pos` — на канонизированной экспозиции
(экспозиция для них — шум), `sp_axis` / `sp_art` / `hip_roi` — без неё, как в 2.1.0 (абсолютная плотность для них —
сигнал). Критерии на `baseline` воспроизводят цифры 2.1.0 бит в бит.

Источник эмбеддингов контура B (20.09, К13): выбран по критерию внутри nested (`tools/emb_gate.py`,
`docs/EMB_GATE_REPORT.md`). Принято одно изменение: `sp_axis` перешёл с ImageNet на наш бэкбон
`densito_inv` (EfficientNet-B0, предобучен на инвариантность эмбеддинга к гамме и шуму) — nested ΔAUC
стэка +0.111, прирост ≥0.03 в 19 из 20 повторов, Δmacro-F1 +0.113. `sp_pos` остался на `densito`,
`sp_art` / `hip_pos` / `hip_roi` — на ImageNet (все кандидаты там не прошли правило приёмки).

Правило приёмки изменений (23.09, идея 11): к правилу «ΔAUC ≥ 0.03 в ≥ 70 % повторов без потери macro-F1» добавлен
калиброванный гейт `tools/paired_gate.py` — парный кластерный бутстрап ΔAUC по исследованиям (ДИ 90 %), перестановочный
тест по знакам групп и поправка Холма на число кандидатов в партии. Симуляция на реальной структуре данных
(`tools/paired_gate_calibration.py`, `docs/gate_calibration/`): при равных AUC старое правило ложно принимает 1–23 %
кандидатов, парный t-test по повторам — 15–39 %, новый гейт — 1–8 % при номинальных 5 %. Ограничение: у `sp_pos`
(позитивы в 6 исследованиях) мощность гейта ограничена дискретностью перестановочного распределения — больше повторов не
помогает, нужны новые положительные исследования. Подробно — `docs/NESTED_GATE_REPORT.md`, часть 4.
Кандидат H2 (голова укладки, `sp_pos`; 23.09, идея 24) принят решением владельца проекта по действующему правилу (ΔAUC +0.119, 20/20
повторов); калиброванный гейт его не принял (ДИ90 [−0.004; +0.218], p 0.12, sign-flip p 0.008) — не потому, что эффект
сомнителен по знаку, а потому, что при 6 положительных исследованиях у гейта нет мощности; это указано явно
(`docs/NESTED_GATE_REPORT.md`, часть 5; `models/nested_gate_decisions.json`).
Признак `sp_art` в зоне измерения (27.09, 2.5.0) включён так же — решением владельца проекта: действующее правило пройдено
(ΔAUC +0.066 ± 0.035, ≥ 0.03 в 19 из 20 повторов, macro-F1 не упала ни в одном), калиброванный гейт — нет
(ДИ95 [−0.029; +0.176], p_boot 0.072, sign-flip 0.092). Здесь 17 положительных исследований и мощность гейта 0.87, поэтому
отказ — не нехватка мощности, а неоднородность прироста по исследованиям. На 40 кадрах слепой проверки врачей эффекта нет
(совпадение с разметкой 27 → 28 из 40); на синтетических фантомах позвоночника без предмета сервис ставит флаг предметов на
всех трёх (эталон `tests/phantoms/expected_results.csv` обновлён) — `docs/NESTED_GATE_REPORT.md`, часть 8.
На образце организаторов «Для теста» снимок `CR000000_ПОП.dcm` в 2.5.0 получает «Присутствуют посторонние предметы»
(quality_prob 0.952, команда «Переснять»; в 2.4.x — норма 0.325, `sp_art` 0.596 при пороге 0.602): в зону попадают концы рёбер
в верхних углах кадра, 303 мм². Ограничение зоны по ширине (5–40 мм от края позвоночника) проверено вложенной проверкой 28.09
по предрегистрации и не принято: все варианты хуже `band70` (лучший — 40 мм, ΔAUC −0.004, macro-F1 −0.072), у 10 из 32 кадров
с предметом по разметке все плотные участки дальше 40 мм от позвоночника (`docs/NESTED_GATE_REPORT.md`, часть 8; `docs/spart_lateral/`).

Пороги (19.09, К3): правило порога по критерию выбрано nested-сравнением трёх предзаданных правил по минимальному regret F1
(`config.yaml: thresholds_rule`, раздел «Калибровка и зона не уверен» в `docs/METRICS_REPORT.md`): `sp_pos` prevalence 0.861,
`sp_axis` prevalence×1.4 0.777, `sp_art` prevalence×1.4 0.624 (в 2.4.x — 0.602), `hip_pos` prevalence 0.658, `hip_roi` prevalence 0.918.
Значения порогов `sp_pos` и `hip_pos` сдвинулись (0.791 → 0.822, 0.643 → 0.658) только потому, что правило
prevalence пересчитано на новом стэке (К11); само правило не менялось. В 2.4.0 порог `sp_pos` пересчитан тем же
правилом на стэке с признаком H2: 0.822 → 0.861 (10 строк OOF стоят ровно на пороге, поэтому флагов 18 при 10 позитивах). До К3
пороги `sp_art`/`hip_pos`/`hip_roi` были F1-оптимумом на тех же OOF-предсказаниях (0.557 / 0.709 / 0.891), что давало
оптимистичные OOF-цифры (F1 0.60 / 0.51 / 0.64; бинарная F1 0.65 / 0.67); nested показал, что prevalence-правила устойчивее
(sd порога 0.03–0.04 против 0.09–0.13) и дают больший F1 на внешних фолдах (см. таблицу ниже).

**Две колонки: OOF (как выше) и nested — «ожидание на закрытом тесте».** В OOF-цифрах порог подобран на тех же
OOF-предсказаниях, поэтому они оптимистичны. Nested repeated GroupKFold (5 внешних фолдов × 10 повторов, 3 внутренних;
группы — исследование + хэш пикселей; вес стэкинга и порог выбираются только на внутренних фолдах; `tools/nested_gate.py`,
полный отчёт `docs/NESTED_GATE_REPORT.md`; правила порога — `tools/calibration_eval.py --stage nested`) даёт оценку без подгонки — среднее по 10 повторам (для AUC — мин–макс); F1 nested — для текущего правила порога:

| Критерий | ROC-AUC OOF | ROC-AUC nested | F1 OOF | F1 nested |
|---|---|---|---|---|
| `sp_pos` | 0.893 | 0.878 (0.84–0.92; H2, финальная голова 0.873) | 0.50 | 0.61 (было 0.50 без H2) |
| `sp_axis` | 0.889 | 0.860 (0.83–0.88) | 0.60 | 0.43 (0.748 и 0.22 были на ImageNet до К13) |
| `sp_art` | 0.899 | 0.884 (0.85–0.92; 20 повторов, `docs/NESTED_GATE_REPORT.md`, часть 8) | 0.67 | 0.64 (база 2.4.x в том же протоколе 0.53) |
| `hip_pos` | 0.725 | 0.709 (0.69–0.74) | 0.43 | 0.47 (было 0.45) |
| `hip_roi` | 0.917 | 0.873 (0.79–0.91) | 0.51 | 0.35 (было 0.34) |

Столбец «ROC-AUC nested» — поле `nested_auc_production` (= `nested_auc_mean`) в `models/metrics_summary.json`: оценка
именно поставленной конфигурации. До 24.09 у `sp_art`, `hip_pos`, `hip_roi` в `nested_auc_mean` стояла отвергнутая
альтернатива (вес вентиля К2: 0.809 / 0.681 / 0.852); теперь она только в `nested_auc_alternative`. Пять «повторов»
внешнего цикла `train_stacked.py` дают одну и ту же раскладку фолдов и независимыми не являются; настоящие повторы
разбиения — nested (`GroupKFold(shuffle=True)`, 10 сидов) и `docs/metrics_split_spread.json` (10 разбиений: AUC
0.860 / 0.851 / 0.864 / 0.712 / 0.899, F1 `sp_axis` 0.439 ± 0.063, `hip_roi` 0.352 ± 0.155).

**Оговорка по F1 `hip_roi`.** F1 0.514 опирается на 16 положительных кадров, из которых 5 (одно исследование) получили
метку через детектор стороны: разметка задана по исследованию для правого и левого бедра, а какой файл какая сторона —
решает детектор. При старой плотностной привязке стороны у тех же OOF-скоров F1 0.267 (11 положительных, AUC 0.880)
— `docs/metrics_side_label.json`, `python tools/side_label_sensitivity.py`. Публикуем оба числа; сторону этих 5 кадров
должен подтвердить рентгенолог.

**Метрики в схеме организаторов (`docs/ORGANIZER_METRICS.md`).** Разметка организаторов задана по исследованию
(лист «Калибровка»: позвоночник / правое / левое бедро), оценка — по файлам. `tools/organizer_metrics.py` переносит
метку исследования на каждый файл по стороне детектора и считает всё в одной схеме (OOF, 495 размеченных файлов из
100 исследований, 249 уникальных кадров; 95 % ДИ — кластерный бутстрап по исследованиям, 2000 повторов):

| Показатель | По файлам | По уникальным кадрам |
|---|---|---|
| Бинарная ROC-AUC по `quality_prob`: позвоночник / бедро / все файлы одним пулом | 0.846 / 0.760 / 0.796 [0.70; 0.88] | 0.855 / 0.745 / 0.794 |
| Бинарная F1: позвоночник / бедро / общий пул | 0.737 / 0.516 / 0.608 [0.49; 0.71] | 0.704 / 0.488 / 0.586 |
| Macro-F1 по областям (позвоночник / бедро; среднее) | 0.590 / 0.473; 0.532 | 0.645 / 0.491; 0.568 |
| Macro-F1 по 5 критериям | 0.544 [0.40; 0.67] | 0.583 |
| Macro-F1 по 4 строкам словаря («Некорректная укладка» общая) | 0.557 [0.40; 0.67] | 0.549 |
| Уровень визита (100 визитов, 48 с нарушением): чувствительность / специфичность | 0.792 [0.67; 0.90] / 0.615 [0.48; 0.75] | — |

Три трактовки macro-F1 расходятся до 0.04; какую выберут организаторы, неизвестно, поэтому показываем все.
По уникальным кадрам бинарные F1 ниже, чем по файлам (ROC-AUC общего пула почти та же: 0.794 против 0.796): в счёте по файлам кадр с копиями весит больше, а копии
одного кадра всегда получают одинаковый ответ. Чем опасно случайное разбиение по файлам при таких клонах — `docs/clone_trap.md` (одна модель, те же
признаки: ROC-AUC 0.90–0.99 по файлам против 0.52–0.88 по группам «исследование + хэш»).

### Проверьте нас сами: одна команда

```bash
# OOF: те же числа, что в docs/METRICS_REPORT.md и таблице выше (сверка с models/metrics_oof_full.json)
python tools/organizer_metrics.py --oof --markup <путь>/разметка.xlsx
# ваш прогон сервиса: results.csv (9 колонок) против разметки организаторов; сторона — по детектору (по умолчанию)
python tools/organizer_metrics.py --results results.csv --markup <путь>/разметка.xlsx --dicom-root <папка исследований>
```

Выход — markdown и JSON (`--out-md`, `--out-json`): бинарные F1 и ROC-AUC по областям и общим пулом, F1 по типам,
три трактовки macro-F1, по файлам и по уникальным кадрам, метрики уровня визита, блоки связанных рангов на пороге.
Режим `--results` на обучающих 499 файлах даёт in-sample числа (модель видела эти кадры) — скрипт об этом
предупреждает; честная оценка — только `--oof` или закрытый тест. Тест: `python tests/test_organizer_metrics.py`.

ROC-AUC OOF в таблице — `auc_stacked` из `models/metrics_summary.json`; `docs/METRICS_REPORT.md` пересчитывает тот же прогон `src/eval_oof_metrics.py` по `models/metrics_oof_full.json`, расхождения возможны только в третьем знаке (для `sp_art` в 2.5.0 оба источника дают 0.899).

Вывод: ранжирование (ROC-AUC) переносится почти без потерь, а F1 редких критериев (`sp_axis`, `hip_roi`: 16–17 позитивов)
остаётся низким при любом правиле порога — это главный источник
риска на закрытом тесте. Официальный CSV использует фиксированные пороги из `metrics_summary.json`; строки с малым запасом до
порога помечаются `needs_review` / `risk_level` в debug-CSV и API (доля «не уверен» на OOF — 13.9 % строк позвоночника и 9.4 % строк бедра, см. выше).

Почему так и что это значит:
- **Редкие классы (`sp_pos`: 10 позитивов, `hip_roi`: 16).** ДИ по построению широкие;
  пороги для `sp_pos` взяты как prevalence-квантиль (не подбирались по F1), чтобы не
  переобучиться на шуме. По `sp_pos` на закрытом тесте следует ожидать результат заметно слабее
  OOF-цифр 2.4.0: признак H2 проверен только nested на тех же 10 позитивах из 6 исследований, ДИ F1
  доходит до 1.0, калиброванный гейт мощности не имеет.
- **Укладка бедра (`hip_pos`).** Позитивов достаточно (79), но экспертная разметка опирается
  на ротацию бедра (видимость малого вертела), которую угол диафиза и глобальный эмбеддинг
  ловят частично. Объединение правого и левого бедра в одну модель (с зеркалированием) дало
  заметный прирост против раздельных моделей 2.0.x (AUC 0.58/0.48 → 0.70).
- **Ось позвоночника (`sp_axis`).** Геометрия работает (AUC 0.84 отдельно), но измеренный угол
  систематически меньше экспертной оценки (медиана позитивов 2.9° при пороге ТЗ 5°) —
  эксперт, вероятно, оценивает не только глобальный наклон. Именно это и подтвердил К13:
  контур B на ImageNet был около случайного (AUC 0.49), а на бэкбоне `densito_inv` даёт 0.80,
  то есть в кадре есть признак наклона помимо измеренного угла.
- **Посторонние предметы (`sp_art`).** До 2.5.0 геометрический признак был слаб (AUC 0.56): он суммировал
  плотные участки по всему кадру, в основном крылья подвздошных костей. С 2.5.0 он считает только зону
  измерения (AUC 0.80), основной вклад по-прежнему дают эмбеддинги (AUC 0.90), стек — 0.90. В верхнюю полосу
  попадают и плотные участки анатомии у верхних углов кадра; на 40 кадрах врачей эффекта нет.
- **Домен.** Один аппарат, одна клиника, один разметчик: перенос на другие денситометры не
  проверен; для нестандартной геометрии кадра предусмотрен контентный детектор региона.
- **Оценка организаторов.** Финальные метрики (ТЗ п.8.4) считаются на закрытом тесте;
  наши OOF-оценки — честный ориентир без утечек, но не гарантия.

---

## 11. Процедура обучения / дообучения

Скрипты обучения используют **константы путей в начале файла** (не CLI-аргументы) —
перед запуском на другой машине поправьте `ROOT`/`STUDIES_DIR`/`LABELS_XLSX` в
`build_dataset.py`, `DATA_DIR`/`OUT_DIR` в `train_stacked.py` и пути в блоках
`if __name__ == '__main__'` у `extract_all_features.py` / `embeddings.py`.

```bash
# 0) окружение — см. §5
# 1) датасет: папка исследований + файл разметки .xlsx  ->  data/labels_full.csv
python src/build_dataset.py
# 2) признаки контура A  ->  data/geometry_features.csv
python src/extract_all_features.py
# 3) эмбеддинги контура B  ->  data/embeddings.npy (+ data/labels_for_embeddings.csv)
python src/embeddings.py
# 3b) признак H2 для sp_pos (логит головы укладки из канонического эмбеддинга densito)
#     -> колонка synth_pos_logit в data/geometry_features_canonical.csv
python tools/add_sppos_head_feature.py --check
# 4) обучение + OOF-валидация GroupKFold(5) по исследованию + пороги + bootstrap-ДИ
python src/train_stacked.py
#    -> models/oof_stacked_*.csv, models/metrics_summary.json
# 4b) финальные модели на 100 % данных -> models/model_<region>_<crit>_{geom,emb_pca}.pkl
#     (формат models/MODEL_CONTRACT.md), метрики OOF и контрольные суммы весов
python src/train_final_models.py
python src/eval_oof_metrics.py
python tools/hash_weights.py --root .
# 5) проверить, что инференс подхватил модели (method = stacked_rank_avg) и формат в порядке
python src/inference.py -i tests/phantoms -o outputs/check.csv --debug-csv -v
python tests/test_inference_format.py
```

> Порядок `train_stacked.py` → `train_final_models.py` → `eval_oof_metrics.py` — тот же, которым собраны модели 2.5.0
> (`docs/NESTED_GATE_REPORT.md`, `docs/PREPROC_GATE_REPORT.md`). Таблицы в `data/` — производные от обучающего набора
> организаторов (`data/README.md`); сами снимки в репозиторий не входят.

Ключевые решения (см. `docs/FINAL_PLAN.md`): без поворотных аугментаций на реальных метках
(портят критерий «ось»); синтетические позитивы для оси (поворот корректных снимков на
4–12°) и для металла (copy-paste) — опция; для бедра возможна единая модель на две стороны
с зеркалированием правого (`mirror_right`). Дообучение на новых данных: дополнить
`labels_full.csv` (колонки `file_path, region, sp_pos, sp_axis, sp_art, rh_pos, rh_roi,
lh_pos, lh_roi`) и повторить шаги 2–5. Случайные зёрна фиксированы; инференс детерминирован
(проверяется тестом).

---

## 12. Руководство пользователя (кратко)

1. Сложите DICOM-файлы (или архив) в папку, например `/data/dxa_batch`.
2. Запустите `./build_and_run.sh run /data/dxa_batch ./outputs` (или загрузите файлы через
   `/api/analyze`, или откройте `http://localhost:8000/docs`).
3. Откройте `outputs/results.xlsx`. Строки с `quality_class = 1` — снимки, требующие
   внимания; в `violation_type` — что именно не так; `quality_prob` — оценка риска нарушения (чем выше,
   тем вероятнее нарушение; ранговая оценка, а не калиброванная вероятность). Строки `Failure` — файлы, которые не удалось прочитать
   (причина — в `results.log` и `results_debug.csv`, колонка `error`).
4. Для разбора спорных случаев смотрите `results_debug.csv`: измеренный угол оси
   (`feat_axis_angle_deg`), площадь плотных участков по кадру и в зоне измерения (`feat_metal_metal_area_mm2`,
   `feat_metal_metal_band70_area_mm2` — по ней с 2.5.0 решает контур A `sp_art`), расстояние ROI до
   края (`feat_edge_distance_mm`), а также `<crit>_score` и `<crit>_threshold`.
5. Для снимков бедра в кабинете показывается предложение коррекции области интереса (рамка на снимке):
   подтвердите, отклоните или задайте свою область — решение сохраняется в задаче, выгружается в CSV и DICOM SR
   и не влияет на официальную таблицу.
6. Если в задаче больше одного исследования, в кабинете под очередью появляется блок «Сводка по партии»:
   доли снимков с нарушением по области, типу нарушения и аппарату с доверительными интервалами, зона «не уверен»,
   причины Failure и список исследований для пересмотра (по UID, нажатие показывает исследование в очереди).
   Кнопки «Сводка (Markdown)» и «Сводка (CSV)» скачивают то же в файл. Персональных данных в сводке нет.
   Кнопка «Скачать дополнительные серии (zip)» (2.4.1) скачивает `additional_series.zip` запроса: DICOM SR на
   исследование, наложение SC, SEG (ТЗ п. 2.7).
7. Вкладка «Журнал исследований» (если техгруппа завела учётные записи; на своём компьютере — открытый демо-режим
   `-e DENSITO_REGISTRY_OPEN=1`, §4): поиск по фамилии, ID, номеру направления
   или дате, статусы «нужна пересъёмка» / «переснято» / «принято» / «спорно» и комментарии врача и лаборанта к
   исследованию. ФИО в списке сокращены; полные данные — по кнопке, просмотр фиксируется.
8. Инструкция для эксперта-тестировщика — `docs/EXPERT_TESTING_GUIDE.md`.

---

## 13. Руководство по развёртыванию

1. Сервер Linux x86-64, Docker ≥ 20.10 с BuildKit, доступ к PyPI и
   `download.pytorch.org` **только на этапе сборки** (в работе сеть не нужна).
2. `git clone https://github.com/neuropeopleteam-afk/densitoai.git && cd densitoai && ./build_and_run.sh build`
   или готовый образ из релиза — «Быстрый старт» в начале README.
   Офлайн-стенд: соберите образ на машине с интернетом и перенесите
   `docker save densitoai:2.5.0 | gzip > densitoai.tar.gz` → `docker load`.
3. Пакетный режим: `./build_and_run.sh run <input> <output>`; сервисный режим:
   `docker compose up -d densito-api` (порт 8000, `restart: unless-stopped`, healthcheck на
   `/api/health`).
3а. Приём снимков с денситометра/PACS по DICOM: `docker compose up -d densito-api densito-receiver`
   (порт 11112, AE Title `DENSITOAI`, общий том `/data/output`) или один контейнер
   `densitoai:2.5.0 api+receiver` с `-p 11112:11112`. На аппарате/PACS создаётся узел Storage:
   AE Title `DENSITOAI`, IP сервера, порт 11112. Порт 11112 не должен быть доступен извне сети
   отделения (TLS нет). Принятые файлы остаются в `/data/output/inbox` (удаление — только при
   `DENSITO_INBOX_KEEP=0`). Подробности и проверка: `docs/DICOM_RECEIVER.md`.
4. Ресурсы: `build_and_run.sh` автоматически берёт `min(nproc, 8)` ядер / 8 ГБ памяти
   по умолчанию (переопределяется через `CPUS`/`MEM`) — без этого на хосте с
   меньшим числом ядер, чем 8, `docker run --cpus=8` завершается с ошибкой (выявлено
   при реальном тесте на 4-ядерном сервере). `docker-compose.yml` — `DOCKER_CPUS`/`DOCKER_MEM`
   (по умолчанию 4/8g; `deploy.resources.limits` действует только в Swarm-режиме, но
   оставлено для документации/совместимости). Параллелизм внутри torch — `OMP_NUM_THREADS`.
5. Обновление моделей без пересборки: смонтировать папку с новыми `.pkl` и
   `metrics_summary.json` в `/app/models` (`-v /new/models:/app/models:ro`) — веса
   backbone при этом должны остаться в `models/torch_home`.
6. Логи: `/data/output/results.log` (batch), `/data/output/api_server.log` (API), stdout.
   `NNPACK`-предупреждения torch (безвредный CPU-fallback на части облачных CPU) отключены
   в коде (`torch.backends.nnpack.set_flags(False)` в `src/embeddings.py`), логи чистые.
7. Контейнер работает от непривилегированного пользователя `densito` (uid 1000).
   **Важно для тома `/data/output`**: без совпадающего uid запись в смонтированную
   папку завершится `PermissionError` (выявлено при реальном тесте на production-сервере).
   `build_and_run.sh` уже запускает batch/api с `--user "$(id -u):$(id -g)"`, а `docker-compose.yml` —
   с `user: "${UID:-1000}:${GID:-1000}"`, так что результаты сразу принадлежат вызывающему
   пользователю. При ручном `docker run` без этих обёртки добавьте флаг вручную или
   выдайте права на хост-директорию вывода командой `chmod 777`/`chown 1000:1000`.

---

## 14. Соответствие ТЗ и статус

Проверено по «4.-DepZdrav.pdf» (ТЗ) и разъяснениям организаторов (`docs/qa/`):

| Требование ТЗ | Статус |
|---|---|
| п.2.5 формат таблицы (`path_to_study, study_uid, image_uid, anatomical_region, quality_class, violation_type, quality_prob, processing_status, time_of_processing`) | выполнено; проверяется `tests/test_inference_format.py` и `format_check` в ответе API |
| п.2.7 ≤ 3 мин/исследование, без необработанных исключений, batch → csv/xlsx | выполнено: на боевом сервере в тихих условиях медиана 0,54 с/файл, p95 1,76 с (499 файлов за 368 с одним процессом), под чужой нагрузкой медиана 1,84 с (прогон версии 2.3.2, `docs/PERFORMANCE.md`, раздел 1), 499/499 успешно, 0 исключений |
| п.2.7 zip-архив с дополнительными сериями (при наличии функционала) | сделано в 2.4.1: в пакетном режиме рядом с `results.csv` — `additional_series.zip` (DICOM SR на исследование, наложение SC, SEG; `--series-zip PATH`, `DENSITO_SERIES_ZIP=0`) — `src/inference.py`; API `GET /api/results/{job_id}/additional_series.zip`, поле `additional_series_zip_url` — `src/api_server.py`; тест `tests/test_series_zip.py` |
| п.2.7 время на исследование | по регрессии 2.4.0 (100 исследований, 499 снимков, сервер 4 vCPU) сумма `time_of_processing` по исследованию: медиана 0,66 с, p95 2,12 с, максимум 5,14 с (29 снимков); расчёт — `docs/PERFORMANCE.md`, раздел 2 (2.4.1); в 2.4.1 с архивом серий по умолчанию — медиана 1,05 с, p95 3,34 с, максимум 8,14 с |
| п.3 контейнер, полностью локально, пиновка версий, скрипт сборки/запуска Linux | выполнено: `Dockerfile`, `build_and_run.sh`, `docker-compose.yml`, `requirements.txt` с точными версиями; образ собран и прогнан на сервере (самопроверка внутри сборки) |
| п.3.1 минимальная и рекомендуемая конфигурация (ОЗУ, процессор, GPU, диск) | выполнено: §3, таблица «Минимальная / Рекомендуемая» (минимум 2 ядра, 4 ГБ ОЗУ, 3 ГБ диска, GPU не требуется; рекомендуемая 8 ядер, 8–16 ГБ, 5 ГБ); измеренные память и размер образа — `docs/PERFORMANCE.md` (строка добавлена в 2.4.1) |
| п.3.2 API пакетной обработки | выполнено: `/api/batch`, `/api/analyze`, `/api/health`, `/api/jobs/{job_id}`, `/api/jobs` (по админскому ключу), `/api/results/{job_id}/{name}`, `/api/results/{name}`, `/api/review`, Swagger `/docs` |
| п.5 README (назначение/ограничения, структура, требования, сборка, API, форматы, модель/пре-/постобработка, ошибки) + руководства пользователя, развёртывания, обучения | выполнено: §1–§13 этого файла, `docs/EXPERT_TESTING_GUIDE.md` |
| п.8.4 метрики по областям и типам с 95 % ДИ | выполнено: `docs/METRICS_REPORT.md`, `src/eval_oof_metrics.py` |
| Публичный репозиторий (Q&A организаторов) | выполнено: `https://github.com/neuropeopleteam-afk/densitoai` |
| Материалы защиты | `docs/qa/JURY_QA.md` (вопросы жюри, ответы с источниками, раздел «Слабые места»), `docs/qa/DEMO_SCRIPT.md` (питч, демонстрация, ролик), презентация для жюри — PDF в репозитории `docs/presentation/DensitoAI_LCT2026_prezentatsiya.pdf` и на стенде https://densito.ru/lct/DensitoAI_LCT2026_prezentatsiya.pdf (23 слайда по шаблону организаторов) |

**Бонусы п.2.6 (все реализованы, опциональны, не влияют на основной CSV даже при внутренней ошибке):**

- [x] **Визуализация** — `src/visualize_report.py`. Оверлей на каждый снимок: контур кости,
      ось/диафиз, границы ROI, детекции металла и панель по каждому критерию с измеренной
      величиной (угол, мм) и оценкой модели «скор / порог → НАРУШЕНИЕ / норма». Это честный
      аналог карты внимания для архитектуры геометрия + эмбеддинги: свёрточных активаций, из
      которых строят карты внимания, на этапе классификации нет (контур A — геометрия, контур B —
      замороженный backbone + логистическая регрессия). Флаг `--visualize-dir`; в API —
      `bonus_overlay_png_base64`.
- [x] **Серия с визуализацией в DICOM** — `src/visualize_report.py: overlay_to_dicom_sc`.
      Тот же оверлей как Secondary Capture рядом с исходной серией: `ImageType = DERIVED/SECONDARY/OTHER`,
      `SourceImageSequence` со ссылкой на исходный снимок, `BurnedInAnnotation = YES`, предупреждение
      «не медицинское изделие, не диагноз» нанесено в пикселях, SOP/Series UID детерминированы от
      исходного SOP и sha256 пикселей — повторный прогон не создаёт дубль серии в PACS.
      Флаг `--sc-dir`; в API — каталог `sc/` в папке запроса. Проверяется 17-й проверкой `verify`.
- [x] **DICOM SR** — `src/dicom_sr.py`. Comprehensive SR Storage с вердиктом, `quality_prob`,
      списком нарушений и измерениями (UCUM, кодировка `99DENSITO`, `ISO_IR 192`).
      По умолчанию пишется ОДИН SR на исследование (`--sr-study` / `--sr-study-dir`, в API включено);
      SR на каждый снимок — только с `--sr-per-image` (в API `DENSITO_SR_PER_IMAGE=1`), иначе на одно
      исследование получались два набора SR. Флаг `--sr-dir`; в API — `bonus_sr_dcm_download`.
      Валидирован раунд-трипом pydicom и `tools/validate_sr.py`.
- [x] **Предложение коррекции области интереса с подтверждением специалистом** (бедро; дополнительный
      функционал по разъяснению организатора) — `src/auto_roi.py` измеряет дефицит поля сканирования в мм и
      предлагает область; API отдаёт `roi_suggestion` в строке `/api/analyze`, решение специалиста
      («подтверждено» / «отклонено» / «своя») принимается `POST /api/results/{job_id}/decisions`, хранится в
      `decisions.json`, выгружается `decisions.csv` и DICOM SR решений (`dicom_sr.build_decision_sr`). В кабинете —
      рамка на снимке и кнопки «Подтвердить», «Отклонить», «Своя». Сам ROI аппарата сервис не меняет и не может:
      он измеряет дефицит и предлагает область; решение не влияет на 9 официальных полей. Диагностический PNG
      прежнего флага `--roi-autocorrect-dir` и поле `bonus_roi_png_base64` сохранены как исторические имена.
- [x] **Сегментация структур** (дополнительный функционал по разъяснению организатора 23.09) —
      `src/segmentation_export.py`. Для каждого успешно обработанного снимка: DICOM Segmentation
      (SOP Class 1.2.840.10008.5.1.4.1.1.66.4; позвоночник — сегменты «кость» и «посторонние предметы
      (металл)», бедро — «кость», «поле сканирования», «область интереса»; ссылка на исходный
      SOPInstanceUID, детерминированные UID), PNG-маска для наложения на снимок и JSON с легендой,
      площадями и полигонами контуров в пикселях и миллиметрах. Флаг `--seg-dir`; в API — ссылки
      `seg_download`, `seg_png`, `seg_json_download`; в кабинете — слой «Сегментация». Проверка файла:
      `python tools/validate_seg.py <файл или каталог> [--source <исходный .dcm>]`; устойчивость масок к
      яркости, перевороту и PhotometricInterpretation — `tools/seg_stability.py` → `outputs/seg_stability.json`.
      Маски — те же эвристики, что используются для измерений, а не обученная модель сегментации;
      эталонной разметки структур нет, поэтому качество масок относительно эталона пока не измерено.
- [x] **Сводка по партии для отделения** (дополнительный функционал) — `tools/department_summary.py`,
      `GET /api/results/{job_id}/summary` (+ `.md`, `.csv`), блок «Сводка по партии» в кабинете при нескольких
      исследованиях в задаче. Агрегаты без персональных данных: по области, типу нарушения, аппарату (хэш),
      дате исследования; доля Failure с причинами; зона «не уверен»; UID для пересмотра. У каждой доли — ДИ
      Уилсона 95 % и пометка «мало данных» при n < 20. Схема `schema/department_summary.schema.json`; тест
      `tests/test_department_summary.py` (79 проверок).
- [x] **Приём снимков по DICOM** (дополнительный функционал) — `src/dicom_receiver.py`: Storage SCP
      (C-ECHO/C-STORE, AE Title `DENSITOAI`, порт 11112; CR, DX, Secondary Capture; несжатые transfer syntax и RLE),
      принятое исследование уходит в тот же `/api/analyze`; команды `receiver` и `api+receiver`, сервис
      `densito-receiver`. Тест `tests/test_dicom_receiver.py`; описание — `docs/DICOM_RECEIVER.md`.
- [x] **Веб-интерфейс** — `web/index.html`: загрузка DICOM перетаскиванием, таблица результатов,
      оверлей и рекомендация по полю сканирования по клику, скачивание CSV, DICOM SR и серии
      визуализации. Развёрнут на
      [densito.ru](https://densito.ru) (логин `demo`, пароль `Hakaton`).

**Известные резервы качества (не блокируют сдачу):**

- Специализированные признаки ротации бедра (видимость малого вертела, форма шейки) —
  главный резерв по `hip_pos`.
- Перевес стэкинга по критерию (геометрия для `sp_axis`, эмбеддинги для `sp_art`) —
  задаётся в `config.yaml: stacking`, по умолчанию 0.5/0.5.
- `path_to_study` — относительный путь к файлу от входной папки (`--path-mode` для
  альтернатив), как в образце ответа организаторов.
- Домен: один аппарат; на других денситометрах пороги геометрии могут требовать
  перекалибровки (процедура — §11).

Подробный инженерный отчёт: `docs/ENGINEERING_REPORT.md`. История изменений: `CHANGELOG.md`.
