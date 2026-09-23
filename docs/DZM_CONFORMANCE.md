# Соответствие требованиям ТЗ и ориентирам заказчика (ДЗМ / НПКЦ ДиТ)

Версия решения 2.3.2 (config.yaml → version), config_hash 5ac94710a6cd. Статусы строго трёх видов:

> **Назначение и ограничения.** Программа не является медицинским изделием и не предназначена
> для диагностики, профилактики, лечения или мониторинга заболеваний. Сервис оценивает
> техническое качество укладки и области интереса исследования и не формирует медицинское
> заключение. Решение по исследованию принимает врач. Область применения: денситометрия (DXA)
> поясничного отдела позвоночника и проксимального отдела бедра, оборудование GE Lunar Prodigy;
> вне этой области применения результат не определён. Источник формулировки —
> `config.yaml → intended_use`, та же строка отдаётся в `/api/health` и показана в веб-интерфейсе.
**сделано** — есть в коде и проверено тестом или прогоном; **частично** — реализовано не полностью, ограничение указано;
**не делаем** — осознанно не реализовано, причина указана. Для каждого пункта — файл и место в коде. Пункты ТЗ взяты из
`README.md` §14 и `docs/qa/QA_COMPLIANCE_MATRIX.md`; нумерация «п. X» — по документу ТЗ «4.-DepZdrav.pdf».

## 1. Обязательные требования ТЗ

> Ответ организатора в чате хакатона (23.09.2026): обязательно — определить анатомическую область, отнести изображение
> к корректным или некорректным и определить тип нарушения по критериям из присланной разметки, результат — в выходной
> таблице. Всё остальное, включая коррекцию разметки и сегментацию, — дополнительный функционал. Строки ниже с пометкой
> «п. 2.5» закрывают обязательный минимум; сегментация и коррекция области интереса с подтверждением специалистом
> делаются как бонусы (раздел 2, PLAN.md идеи 22–23).

| Пункт | Требование | Статус | Где реализовано / чем проверено |
|---|---|---|---|
| п. 2.5 | Таблица результата: 9 колонок `path_to_study, study_uid, image_uid, anatomical_region, quality_class, violation_type, quality_prob, processing_status, time_of_processing`, порядок фиксирован | сделано | `config.yaml → output.columns`; `src/inference.py: write_results`, `validate_output_csv`; `tests/test_inference_format.py` (ALL CHECKS PASSED); JSON Schema `schema/results_row.schema.json` + `tests/test_schema.py` |
| п. 2.5 | Строка на каждое изображение, дубликаты не схлопывать | сделано | `src/inference.py: run` — одна строка на файл; QA_COMPLIANCE_MATRIX №2 |
| п. 2.5 | `anatomical_region` — русские строки без стороны; `violation_type` — закрытый словарь через `;`; пусто при `quality_class = 0` | сделано | `config.yaml → regions, violations, violation_separator`; проверка согласованности класс/нарушение — `validate_output_csv` и правила `allOf` в `schema/results_row.schema.json` |
| п. 2.5 | `quality_prob` в [0; 1], согласована с классом | сделано | `config.yaml → stacking.consistent_quality_prob`; `src/inference.py` (class 1 → prob ≥ 0.5); OOF ROC-AUC итоговой prob: позвоночник 0.773, бедро 0.760 (после К3, К11 и К13) (`docs/METRICS_REPORT.md`) |
| п. 2.7 | ≤ 3 мин на исследование, битый вход не роняет пакет | сделано | строка `Failure` с `quality_prob 0.5` (`src/inference.py: process_file`, ветка исключения); `tests/test_inference_format.py` (bad_inputs: 7 сценариев); время на файл p50 0.04 с, p95 0.07 с (`docs/ROBUSTNESS_REPORT.md`, identity) |
| п. 2.7 | Пакетный режим → CSV и XLSX | сделано | `src/inference.py --xlsx`; `api_server.py: /api/batch`, `/api/analyze?xlsx=true` |
| п. 3 | Контейнер, полностью локально, пиновка версий, скрипты сборки/запуска Linux | сделано | `Dockerfile` (CPU-only, без сети при инференсе), `build_and_run.sh`, `docker-compose.yml`, `requirements.txt` (все версии `==`); в патче E в образ добавлены `schema/`, `tools/`, `tests/test_schema.py` |
| п. 3.2 | API пакетной обработки | сделано | `src/api_server.py`: `/api/health`, `/api/analyze`, `/api/batch`, `/api/results/{job}/{file}`, `/api/jobs/{job}` (по коду доступа), `/api/jobs` (только по админскому ключу), `/api/review`, Swagger `/docs`; `tests/test_api_isolation.py` — 88 проверок пройдены на образе 2.3.2 (`PLAN.md`, `CHANGELOG.md` 2.3.2) |
| п. 5 | README: назначение и ограничения, структура, сборка, API, форматы, модель, ошибки | сделано | `README.md` §1–§14; карточка модели `models/MODEL_CARD.md` (генерируется `tools/make_model_card.py`) |
| п. 8.1 | Подход: обоснованность методики, честность валидации | частично | OOF GroupKFold по исследованию с бутстрап-ДИ (`src/train_stacked.py`, `src/eval_oof_metrics.py`, `docs/METRICS_REPORT.md`). Вложенная (nested) repeated GroupKFold оценка выбора порогов, весов стекинга, предобработки и источника эмбеддингов **сделана**: 5 внешних фолдов x 10–20 повторов, 3 внутренних, группы = исследование + хэш пикселей (`tools/nested_gate.py`, `docs/NESTED_GATE_REPORT.md`, `docs/EMB_GATE_REPORT.md`, nested-числа в `models/metrics_summary.json` и `models/MODEL_CARD.md`). **Не сделано:** один разметчик, межэкспертного согласия нет (`docs/EVIDENCE.md`) |
| п. 8.2 | Техническая реализация: изоляция запросов, лимиты, защита входа | сделано | `src/api_server.py`: каталог на запрос `JOB_RE`, `_safe_job_file` (без выхода за каталог), лимиты `MAX_FILES_PER_REQUEST`/`MAX_UPLOAD_MB`, защита Zip Slip (`src/inference.py: safe_extract_zip`, «Zip entry skipped»); `tests/test_api_isolation.py` |
| п. 8.3 | Соответствие ТЗ: корректность разметки анатомических структур | частично | Область — по содержимому (теги → ширина → `models/model_region_emb.pkl`), сторона бедра — `src/hip_features.py: detect_hip_side`. **Нет** валидированной локализации структур (Dice/IoU/keypoints) — оверлей `src/visualize_report.py` показывает измеренную геометрию, метрика локализации не заявляется |
| п. 8.4 | Эффективность: метрики по областям и типам с 95 % ДИ | сделано | `docs/METRICS_REPORT.md`, `models/metrics_summary.json`, `models/metrics_oof_full.json`; сводка с ДИ и n_pos — `models/MODEL_CARD.md` §5 |
| п. 8.4 | Калибровка вероятности (Brier/ECE), неопределённость отдельного предсказания | частично | Калибровка Platt по критерию обучена cross-fit и лежит в `models/calibration.pkl`; Brier/ECE до и после — `docs/METRICS_REPORT.md`, раздел «Калибровка». Значение `p_cal` отдаётся в debug-CSV и `details` API, **в девять колонок выгрузки не входит**. `quality_prob` в выгрузке остаётся согласованным с классом ранговым скором, не калиброванной вероятностью (`config.yaml → stacking`) — так и заявляется. Неопределённость отдельного предсказания: зона «не уверен» по запасу до порога, `needs_review` / `risk_level` (`config.yaml → uncertainty`, `tools/uncertainty_margins.py`) |
| п. 8.5 | Презентация и защита | частично | Вне кода; артефакты — `docs/LETTER_TO_ORGANIZERS.md`, `docs/EXPERT_TESTING_GUIDE.md`; презентация — задача К10, не в этом каталоге |

## 2. Бонусы ТЗ п. 2.6 и «желательные» пункты

| Пункт | Пожелание | Статус | Где реализовано |
|---|---|---|---|
| п. 2.6 | Визуализация зон / дополнительная серия | сделано | `src/visualize_report.py` (оверлей PNG: контур кости, ось, ROI, металл, панель критериев + предупреждение «не медицинское изделие, не диагноз» в пикселях); `--visualize-dir`; API `bonus_overlay_png_base64`. **Дополнительная серия как DICOM Secondary Capture**: `overlay_to_dicom_sc`, `save_overlay_sc`, флаг `--sc-dir`, в API — каталог `sc/` запроса; ImageType `DERIVED/SECONDARY/OTHER`, `SourceImageSequence` на исходный снимок, `BurnedInAnnotation=YES`, UID детерминированы от исходного SOP и sha256 пикселей (повторный прогон не создаёт дубль серии). Проверка 17 в `verify`. Это не карта внимания нейросети — так и описано в README §14 |
| п. 2.6 | DICOM SR на снимок | сделано, по умолчанию выключен | `src/dicom_sr.py: build_sr` (Comprehensive SR, `99DENSITO`), `--sr-dir` + `--sr-per-image`, API `bonus_sr_dcm_download` (переменная `DENSITO_SR_PER_IMAGE=1`). По умолчанию на исследование пишется ОДИН SR (строка ниже): раньше выдавались оба набора сразу, что противоречит методике НПКЦ ДиТ |
| п. 2.6 / НПКЦ ДиТ | **Один DICOM SR на исследование, включая норму** | сделано (патч E) | `src/dicom_sr.py: build_study_sr, save_study_sr, study_header_from_ds`; `src/inference.py: --sr-study`, `write_study_sr`, `_origin_hashes`; API: `study_sr` в ответе `/api/analyze`, `study_sr_download` в строке, файл по `/api/results/{job}/<study_uid>_SR.dcm`; валидатор `tools/validate_sr.py` (PASS на образце организаторов, 2 SR / 2 исследования, и на 3 исследованиях датасета + 1 битый файл, 4 SR / 4 группы); `tests/test_schema.py`. При наличии в исследовании только одной из двух областей (позвоночник или бедро) в корень SR добавляется TEXT `STUDY-COMPLETENESS` «Примечание о полноте исследования (представленные области)» с нейтральным текстом; полнота не является критерием оценки качества снимка и не влияет на CSV. То же — поле `study_completeness` в ответах `/api/analyze` и `/api/batch`. UID SR полных исследований не меняются. Тест: `tests/test_study_completeness.py` |
| п. 2.6 | Рекомендация по полю сканирования бедра (пункт ТЗ о коррекции области интереса) | частично | `src/auto_roi.py`: измерение дефицита поля сканирования в мм и рекомендация технологу; сам ROI аппарата не меняется — корректнее называть «рекомендация по полю сканирования» |
| желат. | Оценка белой разметки денситометра | не делаем | Разметка есть не на всех снимках, эталона нет (QA_COMPLIANCE_MATRIX №18). Аудит «линии × метка» (PLAN К8) проведён — `docs/WHITE_LINES_AUDIT.md`: в экспорте организаторов графики ROI нет, линии найдены на 2 кадрах из 499 (0,4 %, оба — край эндопротеза), ассоциации с метками нет; поэтому только экспериментальный флаг `white_lines` в `src/extras.py` (`--extras`, `results_extras.csv`), оценку разметки не делаем и метрику не заявляем |
| желат. | Защита от посторонних изображений | частично | Не-DICOM/битые файлы → `Failure`; проверка размеров кадра (`config.yaml → validation`). С 2.3.2 в API и вебе — отказ по неподдерживаемой области (`src/region_support.py`: предплечье, «всё тело», боковая проекция, чужая модальность, кадр вне геометрии 200–400 × 120–600 px → `Failure` без вердикта; `tests/test_region_support.py`, 44 проверки). OOD-детектор чужого аппарата (PLAN К9) реализован как предупреждение: отпечаток DICOM-тегов + расстояние Махаланобиса по эмбеддингам (`src/extras.py: fingerprint, OODGate, ood_score`, `models/ood_gate.pkl`, `docs/OOD_GATE_REPORT.md`; FPR 1,0 % на OOF, 130/130 внешних кадров отбракованы); флаг `ood_flag` идёт в `results_extras.csv` и `details.extras` API, в кабинете — пометка «нестандартный источник снимка». Девять колонок основного CSV он не меняет. Пакетный путь CSV отказа по области не делает — так и заявляется |
| желат. | Устойчивость к другим аппаратам | частично | Геометрия зависит только от pixel spacing (константы `hip_features.py`, `geometry_features.py`); измерения устойчивости — `docs/ROBUSTNESS_REPORT.md` (формат-инварианты 0 % переворотов; шум σ = 3 % — 22.2 %, гамма 0.7/1.4 — 24.7 %/28.4 %). На чужом приборе не проверялось |
| желат. | Веб-интерфейс | сделано | `web/index.html`, `src/api_server.py` |
| желат. | Инструкция проверки для экспертов | сделано | `docs/EXPERT_TESTING_GUIDE.md`, README §14, `tests/test_inference_format.py`, `tests/test_api_isolation.py`, `tests/test_schema.py` |
| желат. | Машиночитаемая схема результата и ответа API | сделано (патч E) | `schema/results_row.schema.json`, `schema/api_analyze_response.schema.json`; проверка встроена в `inference.validate_output_csv` через `src/schema_check.py` (jsonschema при наличии, иначе встроенный валидатор; в песочнике — builtin) |
| желат. | Карточка модели | сделано (патч E) | `tools/make_model_card.py` → `models/MODEL_CARD.md` из `metrics_summary.json`, `models_manifest.json`, `config.yaml`, `requirements.txt` |

## 3. Внутренняя имитация классификатора дефектов НПКЦ ДиТ

**Рамка.** Ниже — наша интерпретация публичной методологии НПКЦ ДиТ (матрица зрелости ИИ-сервисов, 3 кв. 2024:
https://mosmed.ai/media/Матрица_зрелости_ИИ-сервисов_3_кв_2024.pdf ; базовые функциональные требования 20.05.2026:
https://mosmed.ai/documents/349/БФТ_полный_20.05.2026.pdf ). Это не официальная проверка и не заявление о прохождении
классификатора; коды дефектов и формулировки — из конспекта консилиума (`docs/council/`), они могут отличаться от
действующей редакции методологии. Ориентиры заказчика по конспекту: AUC > 0.81; техническая ось — 100 − доля дефектов,
приемлемо ≤ 10 % дефектных исследований. Наши OOF ROC-AUC бинарной задачи (0.773 позвоночник, 0.760 бедро,
`docs/METRICS_REPORT.md`) с этими ориентирами не сопоставляются: выборка, разметка и протокол подсчёта у заказчика
другие, и ниже речь только о технических дефектах оформления результата, а не о качестве классификации.

| Код | Дефект (наша интерпретация) | Как избегаем | Статус | Где |
|---|---|---|---|---|
| В1 | Нет дополнительной серии | Оверлей PNG на снимок и та же визуализация как DICOM Secondary Capture (`ImageType DERIVED/SECONDARY/OTHER`, `SourceImageSequence` на исходный снимок, `BurnedInAnnotation=YES`, детерминированные UID) | сделано (2.3.1) | `src/visualize_report.py: overlay_to_dicom_sc, save_overlay_sc`; `src/inference.py --sc-dir`; в API — каталог `sc/` запроса; проверка 17 `verify` (`tools/verify_checks.py: sc_series_check`) |
| В2 | Нет SR на исследование | Режим `--sr-study` пишет SR для каждого исследования, в т.ч. с нормой («Нарушений не выявлено» по каждому снимку) | сделано | `src/inference.py: write_study_sr`; проверка «SR для каждого исследования CSV» — `tools/validate_sr.py` |
| В3 | Более одного SR на исследование | Один файл `sr/<study_uid>_SR.dcm`; детерминированные Series/SOP Instance UID от study_uid + версия + config_hash — повторный прогон не создаёт «второй» SR | сделано | `src/dicom_sr.py: deterministic_uid, build_study_sr`; проверка «ровно один SR на исследование» — `tools/validate_sr.py`; детерминизм — `tests/test_schema.py` |
| В4 | Нет наименования сервиса | TEXT «Наименование ИИ-сервиса» = `DensitoAI` в корне SR (HAS OBS CONTEXT), Manufacturer/SoftwareVersions | сделано | `src/dicom_sr.py: SERVICE_NAME` |
| В5 | Нет версии | TEXT «Версия модели» = config.yaml version, «Хэш конфигурации» = config_hash | сделано | `src/dicom_sr.py: build_study_sr`; `src/inference.py: config_hash` |
| Г1 | Обрезка кадра | Исходные изображения не изменяются; SR ссылается на оригинальные SOPInstanceUID | сделано | `src/dicom_sr.py: _image_item` (ReferencedSOPSequence) |
| Г2 | Несовпадение контраста | К оригиналу не применяется постобработка; оверлей — отдельный PNG | сделано | `src/visualize_report.py` (отдельный файл) |
| Г3 | Обработаны не все кадры | Строка на каждый файл, Failure тоже включается; в SR — счётчики «Число снимков», «не обработанных (Failure)» | сделано | `src/inference.py: run`; `src/dicom_sr.py` (NUM N-IMAGES / N-FAILURE) |
| Г4 | Пришла одна область из двух | Мягкое примечание в SR исследования (TEXT `STUDY-COMPLETENESS`) и поле `study_completeness` в API, только если хотя бы один снимок обработан; на Failure-исследованиях примечания нет; в обучающей выборке так устроено 22 исследования из 100 (21 — только позвоночник, 1 — только бедро) | сделано | `src/dicom_sr.py: study_completeness`; `tests/test_study_completeness.py` |
| Г5 | Изменён оригинал | В SR по каждому снимку SHA-256 файла и SHA-256 массива пикселей оригинала — проверка неизменности | сделано | `src/inference.py: _origin_hashes` → `results_debug.csv: sha256_file, sha256_pixels`; SR TEXT «SHA-256 …» |
| Д1 | Разметка вне органа | Метрика локализации не считается; оверлей строится по маске кости | частично | `src/visualize_report.py`; валидированной локализации нет (см. п. 8.3) |
| Д2 | Неверная область/проекция | Область по содержимому (теги → ширина → `model_region_emb.pkl`, OOF-точность 1.00 на 499 снимках по README §Вход); в SR — TEXT «Анатомическая область» | сделано | `src/inference.py: classify_region`; QA_COMPLIANCE_MATRIX №6–7 |

Ограничения этой имитации: `dciodvfy` (dicom3tools) в песочнике не установлен — внешняя проверка IOD не запускалась,
`tools/validate_sr.py` вызывает его автоматически при наличии в PATH. Исходные UID организаторов содержат компоненты с
ведущим нулём (формально вне PS3.5 §9.1); SR ссылается на них как есть, чтобы ссылки совпадали с PACS — валидатор
выдаёт предупреждение, не ошибку. Для файлов без читаемого StudyInstanceUID (битый DICOM) study_uid = `hash-…`
и SR формируется отдельной группой без ссылок IMAGE (только TEXT), что валидатор помечает предупреждением.
