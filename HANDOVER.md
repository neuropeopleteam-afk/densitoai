# HANDOVER — DensitoAI (ЛЦТ 2026, задача 4)

Документ для быстрого входа в проект: человека, новой модели-ассистента или нового чата.
Прочитай его целиком (5 минут), потом — `README.md` §1, §8, §10, §14 и `docs/METRICS_REPORT.md`.
Секреты (пароли, ключи) здесь **не хранятся** — см. приватное дополнение
`/opt/neuropeople/HANDOVER_PRIVATE.md` на сервере или файл «Доступы и ключи» у владельца.

Обновлено: 2026-09-18, версия продукта 2.1.0.

---

## 0. Одним абзацем

DensitoAI — локальный сервис (Docker, только CPU) автоматической оценки качества DXA-снимков
(денситометрия) для Департамента здравоохранения Москвы. Вход — DICOM (папка/zip/файлы), выход
— CSV/XLSX строго по формату ТЗ п.2.5 (9 полей), плюс REST API, веб-интерфейс, PNG-визуализация,
DICOM SR и предложение исправленного ROI бедра. Архитектура двухконтурная: интерпретируемая
геометрия кости (контур A) + эмбеддинги EfficientNet-B0 (контур B), ранговый стэкинг по каждому
из 5 критериев, отдельная any-модель для калиброванной `quality_prob`. Всё обучено на 499
снимках / 100 исследованиях, метрики — только OOF с группировкой по исследованию и бутстрап-ДИ.

## 1. Контекст конкурса и что важно владельцу

- Конкурс: «Лидеры цифровой трансформации 2026», задача 4 «Сервис ИИ по оценке качества
  исследований плотности костей человека». Постановщик — ДЗМ. ТЗ, разъяснения организаторов
  (письменные + транскрипт Q&A) и их разбор — `docs/qa/`.
- Финальная проверка — экспертами на **закрытом** наборе без участия команды; метрики
  считает организатор (F1 и ROC-AUC приоритетны, желательны 95 % ДИ). Критерии оценки — ТЗ §8.1–8.5.
- Владелец проекта общается голосовым вводом (текст может быть искажён), ждёт коротких ответов
  по-русски, без восклицательных знаков; ценит автономность («делай до конца, не спрашивай»),
  приоритет — сначала работающий и протестированный продукт, потом документы, презентация в самом
  конце. Цель — победа.
- Название команды — DensitoAI. ФИО участников и контакты **пока не известны**; в презентации
  стоят заглушки.

## 2. Где что лежит

| Что | Где |
|---|---|
| Публичный репозиторий (bare) | `https://neuropeople.pro/git/densitoai.git` (gitweb: `https://neuropeople.pro/git/browse?p=densitoai.git;a=summary`) |
| Рабочая копия репозитория на сервере | `/tmp/densito_wc` (может пропасть после перезагрузки — переклонировать из `/opt/git/densitoai.git`) |
| Каталог сборки (источник истины для кода) | `/opt/neuropeople/densito_rebuild_test/densito_rebuild/` |
| Docker-образ / контейнер | `densitoai:2.1.0` / `densitoai-api-container` (порт 127.0.0.1:8026 → 8000) |
| Выходы API | `/opt/neuropeople/densito_api_outputs` (смонтировано в `/data/output`) |
| Веб-интерфейс (статика) | `/opt/neuropeople/densitoai-web/index.html` (копия `web/index.html`) |
| Nginx | `/etc/nginx/sites-available/noema-manus` — `/` (basic auth) → статика, `/api/` → :8026 без auth, `/docs` и `/openapi.json` → Swagger, `/git/` → git-http-backend + gitweb |
| Живое демо | `https://neuropeople.pro` (basic auth demo-пользователь), Swagger `https://neuropeople.pro/docs` |
| Датасет | у владельца (`Dataset.zip` → `Датасет/НД_для_обучения/Исследования/<study>/…/*.dcm` + xlsx разметки); на сервере в каталоге сборки `data/` лежат производные (labels_full.csv, признаки, эмбеддинги) |
| Презентация | `docs/DensitoAI_LCT2026_presentation.pptx` (14 слайдов на шаблоне организаторов; сборщик — вне репозитория, python-pptx) |
| Метрики | `docs/METRICS_REPORT.md`, `docs/metrics_oof_full.md`, `models/metrics_oof_full.json`, `models/metrics_summary.json` |
| Инструкция эксперту/рентгенологу | `docs/EXPERT_TESTING_GUIDE.md`, `docs/RADIOLOGIST_TEST_BRIEF.md` |
| Анализ улучшений и фишек | `docs/IMPROVEMENT_ANALYSIS.md` |
| Письмо организаторам | `docs/LETTER_TO_ORGANIZERS.md` |

## 3. Как работает продукт (карта кода)

```
src/inference.py          # ядро: чтение DICOM → регион → признаки → модели → строка CSV; CLI batch
src/api_server.py         # FastAPI: /api/health, /api/analyze (upload), /api/batch (папка), /api/results/{name}
src/geometry_features.py  # контур A, позвоночник: сегментация, ось, центр, металл/плотные объекты
src/hip_features.py       # контур A, бедро: диафиз, угол, границы ROI, сторона
src/embeddings.py, embeddings_hip_canonical.py  # контур B: EfficientNet-B0 (torch, CPU) → 1280-d
src/train_stacked.py      # обучение по критериям, OOF 5×GroupKFold, пороги, стэкинг
src/train_final_models.py # финальные модели на всех данных → models/*.pkl + manifest
src/eval_oof_metrics.py   # отчёт ТЗ §8.4 с бутстрап-ДИ → docs/metrics_oof_full.md
src/visualize_report.py   # PNG-оверлей (ось, контур, ROI, предметы, панель скор/порог)
src/dicom_sr.py           # DICOM Structured Report
src/auto_roi.py           # предложение исправленного ROI бедра
web/index.html            # веб-интерфейс (чистый HTML/JS, ходит в /api/analyze)
config.yaml               # ВСЕ пороги, веса, строки регионов/нарушений; версия
models/                   # pkl-модели, OOF-файлы, manifest с контрольными суммами
tests/test_inference_format.py  # автотест формата (запускается при сборке образа)
docs/qa/                  # разъяснения организаторов и матрица соответствия
```

Ключевые решения (почему так):
- **Регион** определяется: DICOM-теги → ширина кадра (300 = позвоночник, 280/248 = бедро,
  подтверждено организаторами) → при нестандартной ширине контентная модель
  `model_region_emb.pkl`; имя файла — только подсказка при нестандартном размере.
- **Бедро** правое/левое — одна объединённая модель с зеркалированием (мало данных на сторону);
  сторона в выгрузку не выводится (по ТЗ).
- **Правило решения**: `quality_class = 1`, если хотя бы один критерий ≥ порога; `violation_type`
  — все сработавшие через `;`; `quality_prob` = 0.5·any-модель + 0.5·max по критериям, затем
  монотонно согласована с классом (класс 1 ⇔ prob ≥ 0.5). Альтернативы проверены — выигрыша нет.
- **Таргеты**: ключ соединения — папка исследования; столбец «Итог» не используется как таргет;
  правило «ROI не помечается при некорректной укладке» учтено.
- **Сбой файла** → строка `Failure`, `quality_class = 0`, `quality_prob = 0.5`, остальные
  файлы обрабатываются.

## 4. Текущие метрики (OOF, v2.1.0) — не завышать

| Область | n / pos | Sens | Spec | F1 | ROC-AUC | Macro-F1 по типам |
|---|---|---|---|---|---|---|
| Позвоночник | 166 / 60 | 0.80 | 0.65 | 0.66 [0.50; 0.79] | 0.775 [0.64; 0.89] | 0.45 |
| Бедро | 329 / 92 | 0.60 | 0.92 | 0.67 [0.48; 0.80] | 0.773 [0.62; 0.90] | 0.57 |

По типам: sp_pos (10 pos) F1 0.38 / AUC 0.60; sp_axis (17) 0.38 / 0.74; sp_art (35) 0.60 / 0.82;
hip_pos (79) 0.51 / 0.70; hip_roi (16) 0.64 / 0.92. Пороги: sp_pos 0.813, sp_axis 0.756,
sp_art 0.557, hip_pos 0.709, hip_roi 0.891 (в `config.yaml`/`metrics_summary.json`).
Скорость: 499 файлов за 23 с (0.05 с/файл, 2 vCPU); API с бонусами 1.1–2.1 с/файл; 499/499 Success.

## 5. Рабочие рецепты

Сборка и деплой на сервере (все команды — от root, ssh оборачивать в `timeout`):
```bash
cd /opt/neuropeople/densito_rebuild_test/densito_rebuild
(nohup docker build -t densitoai:2.1.0 . > /tmp/build_2.1.0.log 2>&1 &)   # ~10–15 мин
grep -c 'format check: OK' /tmp/build_2.1.0.log; tail -1 /tmp/build_2.1.0.log   # ждём "#NN DONE"
docker rm -f densitoai-api-container
docker run -d --name densitoai-api-container --restart unless-stopped \
  -p 127.0.0.1:8026:8000 -v /opt/neuropeople/densito_api_outputs:/data/output densitoai:2.1.0 api
curl -s localhost:8026/api/health
```
Синхронизация в публичный git:
```bash
cd /tmp/densito_wc || git clone /opt/git/densitoai.git /tmp/densito_wc && cd /tmp/densito_wc
B=/opt/neuropeople/densito_rebuild_test/densito_rebuild
rsync -a --delete --exclude .git --exclude outputs/ --exclude '*.log' --exclude __pycache__ --exclude .pytest_cache $B/ ./
git add -A && git -c user.name='DensitoAI Team' -c user.email='team@densitoai.local' commit -m "..." \
  && git push -q origin HEAD:main && git -C /opt/git/densitoai.git update-server-info
```
Регрессия после любого изменения инференса: прогнать 499 файлов и сравнить CSV с предыдущим
(`diff`), запустить `pytest tests/`, smoke через API на 3 файлах разных размеров.
Обновление веб-интерфейса: скопировать `web/index.html` в `/opt/neuropeople/densitoai-web/`.

Локально без Docker: `pip install -r requirements.txt`, затем
`python src/inference.py --input <папка> --output outputs/results.csv --xlsx --debug-csv`.

## 6. Что сделано (хронология)

2.0.0 — двухконтурная архитектура, контейнер, формат ТЗ, smoke-тест. 2.0.2 — единая модель
бедра, any-модели, DICOM SR, auto-ROI, визуализация, API. 2.1.0 — согласование prob/class,
контентный детектор региона, панель скор/порог, веб-интерфейс, отчёт метрик с ДИ, разбор Q&A,
письмо организаторам, презентация. Коммиты: e2c4ec7 → 85e57d5 → c419edd → 6de7926 → 93a0a3a → 36f09b5.

## 7. Что НЕ сделано / открытые вопросы

- ФИО и контакты команды в презентации и письме — заглушки.
- Обратная связь врача (рентгенолога) по качеству работы ещё не собрана —
  см. `docs/RADIOLOGIST_TEST_BRIEF.md`.
- Список кандидатов на доработку (ранжированный) — `docs/IMPROVEMENT_ANALYSIS.md`.
- Загрузка файлов в веб-интерфейс из облачного headless-браузера зависает (проблема
  окружения тестера, не сервиса; через curl `/api/analyze` работает).
- Ветка `master` в bare-репозитории — лишняя, не трогать без владельца (основная — `main`).

## 8. Правила, о которые уже спотыкались

- Любая правка модели/порогов — только через OOF-метрики с ДИ; финальные модели переобучать
  `train_final_models.py`, обновлять manifest, CHANGELOG, README §10, METRICS_REPORT.
- Не менять строки регионов/нарушений и порядок колонок (`config.yaml`).
- В документации и презентации указывать одни и те же числа (0.775/0.773 и т.д.).
- Не заявлять в документах то, чего нет в коде (пример: раньше README говорил «имя файла не
  используется», а код проверял его первым — исправлено).
- Кириллица в XML/pptx — править через Python, не sed.
