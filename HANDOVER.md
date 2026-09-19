# HANDOVER — DensitoAI (ЛЦТ 2026, задача 4). Версия v3 (19.09.2026), без секретов

> **Обновление 19.09 (вечер):** этот файл — снимок состояния на 18.09 21:40. Актуальное состояние, все закрытые пункты К1–К10 и
> текущие числа (пороги после К3: sp_art 0.602, hip_pos 0.643, hip_roi 0.918; OOF-метрики — `docs/METRICS_REPORT.md`) — в `PLAN.md`
> (раздел «Журнал»), `README.md` §10 и `CHANGELOG.md`. Цифры ниже в разделах 5–6 относятся к 18.09.


## 0. Изменения 19.09 (v3) — читать первым, остальное ниже без изменений (v2 от 18.09 21:40)

### 0.1. Состояние на 19.09 13:05 MSK
- Код и модели с 18.09 21:40 **не менялись**. Контейнер `densitoai:2.1.0` работает, сайт https://neuropeople.pro — кабинет, `/v1/` — лендинг 15.09. Git `main` = `2443738` (до этого `e1bd338`).
- Изменились только документы: `PLAN.md` (+30 строк) и новый каталог `docs/council/` (15 файлов).
- 19.09 проведён **консилиум из 5 моделей** (Claude Fable 5 — ML/CV, Gemini 3.1 Pro — клиника, Claude Opus 5 — продукт/питч, GPT 5.6 Sol — конкуренты/red team, Kimi K3 — данные/надёжность), два раунда: независимые заключения → синтез → взаимная критика → финальный план.
- Итог консилиума — раздел `PLAN.md` «Приоритеты консилиума 19.09»: пункты **К1–К10** (~95–100 ч), список «не делаем». Условия приёмки, спорные вопросы и их решения — `docs/council/SYNTHESIS_ROUND2.md`; раунд 1 — `SYNTHESIS_ROUND1.md`; отчёты моделей — `r1_*.md`, `r2_*.md`; индекс — `docs/council/README.md`. **Порядок К1–К10 заменяет порядок шагов 3–10 PLAN.md** (шаги остаются как справочник). При отставании первыми режутся К7–К9.
- Ключевые находки консилиума (коротко): стэкинг 0.5/0.5 смешивает сильный контур со случайным (sp_axis geom 0.839 / emb 0.491 → итог 0.738) — нужен вентиль по критерию через nested; жюри отбирает топ-10 по прототипу и презентации (5 критериев × 5 баллов), техническая группа ЦДиТ сама разворачивает решения на закрытых данных — проверяемость на стенде №1; ориентир заказчика AUC > 0.81 (высоко > 0.9), инструмент ЦДиТ по РГ ОГК 0.782/0.852 — как контекст, не «мы не хуже»; классификатор дефектов НПКЦ ДиТ: нет SR / два SR — дефект; Arak — CC BY-NC, в именах файлов Chinese OP — ФИО, изображения внешних наборов в репозиторий не класть; конкурсные DICOM в публичный эталонный пакет не класть (синтетические фантомы + sha256 ожидаемого CSV).

### 0.2. Новое указание владельца (19.09 13:02)
«Сделай новый [чат] и приступай к реализации плана, делаем всё что можно параллельно, что не можешь — постепенно». То есть: **параллельные субагенты разрешены**, ждать подтверждений не нужно, работать до конца.

Ограничения параллельности (обязательны):
1. Сервер общий с продакшеном NEUROPEOPLE, свободно ~1–2.5 ГБ RAM. **Одновременно не более одного тяжёлого процесса** на сервере (обучение, nested, инференс на 499 файлах, `docker build`). Тяжёлое — через `nice -n 10`, `OMP_NUM_THREADS=2`. Вычислительные эксперименты (nested, вентиль, калибровка) лучше гнать в песочнице ассистента (8 ГБ RAM, 2 vCPU): скачать `models/`, `outputs/*/features*`, OOF-таблицы и код `src/` вниз по SFTP. Для тяжёлого — RunPod (раздел 2.3, ключ в 12): владелец просил не экономить.
2. **Один git-репозиторий**: коммиты, `docker build`, рестарт контейнера, копирование веба — только оркестратор, последовательно. Субагенты работают в отдельных файлах/каталогах песочницы и отдают готовые файлы; оркестратор синхронизирует на `$B`, тестирует, коммитит.
3. Веб: скриншоты владельцу **до** деплоя (раздел 4.1 остаётся в силе — лендинг v1 + кабинет объединить).
4. Не трогать neuropeople.shop/.site, pm2 (особенно `stt-stream`), контейнеры `np-*`, VPN, бота; nginx — только с бэкапом и `nginx -t`.
5. Не менять строки регионов/нарушений, порядок 9 колонок, имена файлов; изменения моделей/порогов — только через OOF/nested с ДИ; одни и те же числа во всех документах; не заявлять в документах то, чего нет в коде.

### 0.3. Раскладка параллельных дорожек (первая волна)
| Дорожка | Пункты | Где считать | Что отдаёт |
|---|---|---|---|
| A | К1: `verify.sh`, CPU-only образ по digest, `linux/amd64`, лимиты, контрольная офлайн-сборка (сначала офлайн `docker run` готового образа, wheelhouse — если помещается в RAM), синтетические фантомы + sha256 ожидаемого CSV, `verification_report.html` из кода, матрица transfer syntax, аудит лицензий/ПДн, релиз-архив | сервер (оркестратор) + песочница | скрипты, Dockerfile-правки, отчёт |
| B | К2: вентиль по критерию (вес ∈ {0, .25, .5, .75, 1}) + иерархия any→типы, nested repeated GroupKFold (study + pixel_hash); приёмка: прирост ≥ 0.03 в ≥ 7/10 повторов без потери macro-F1; затем пересборка контракта моделей (`models/MODEL_CONTRACT.md`) | песочница (или RunPod) | таблица результатов + предложенный `config.yaml: stacking` |
| C | Веб: лендинг v1 + кабинет (раздел 4.1), чистка заявлений (Grad-CAM, «автокоррекция ROI»), демо без пароля с обезличенным/синтетическим примером, панель истории; затем К7 карточка решения (измеренное против нормы, одна причина, кнопка «переснять»), офлайн `casebook.html` | песочница | `web/index.html`, скриншоты |
| D | К8 аудит «линии × метка» (2 ч; < 30 % кадров с линиями → только флаг) + К9 OOD-gate прототип (fingerprint + Mahalanobis на имеющихся эмбеддингах, FPR ≤ 1 % на OOF, CSV не трогать), правило «эндопротез», study coherence auditor (warning) | песочница | скрипты + отчёт с числами |
| E | К10 SR на исследование (`--sr-study`, `src/dicom_sr.py` есть) + валидатор + хэш оригинала; `DZM_CONFORMANCE.md` (статусы «сделано/частично/не делаем»); К4: `docs/EVIDENCE.md`, JSON Schema в валидаторе, генератор model card из `models/metrics_summary.json` | песочница | код + документы, сгенерированные кодом |
| F | К6 подготовка: офлайн-галерея слепой ревизии (40–60 уникальных кадров по pixel_hash: 30 случайных + 20 расхождений, 10 % повторов, «ответ до показа модели»), форма разметки 4 ориентиров на 40 кадрах, скрипты подсчёта kappa/PCK@10 мм с ДИ. Самого врача пока нет — только инструмент | песочница | HTML + скрипты |

Вторая волна (после результатов B): К3 калибровка cross-fit и три правила порога внутри nested, зона «не уверен» (крыша отказа ≤ 3–5 %, только UI); К5 hip_pos (согласованность меток сторон ≥ 85 % → признаки контекста исследования), sp_axis по определению постановщика; К4 nested ночью → две колонки в README. В конце — К10: заморозка за 48 ч, видео 60–90 с, презентация в шаблоне ЛЦТ, 8–15 вопросов Q&A.

### 0.4. Вспомогательные скрипты (песочница прошлой сессии пропала — пересоздать)
```python
# ssh.py — python3 ssh.py "cmd" [timeout]
import sys, paramiko
HOST, USER, PASS = "5.180.173.14", "root", "<пароль из раздела 12>"
def run(cmd, timeout=25):
    c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS, timeout=15, banner_timeout=15, auth_timeout=15)
    try:
        _, out, err = c.exec_command(cmd, timeout=timeout)
        o, e = out.read().decode("utf-8", "replace"), err.read().decode("utf-8", "replace")
        return out.channel.recv_exit_status(), o, e
    finally:
        c.close()
if __name__ == "__main__":
    rc, o, e = run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 25)
    sys.stdout.write(o + (("\n[stderr]\n" + e) if e else "")); sys.exit(rc)
```
```python
# sftp_put.py — python3 sftp_put.py put local remote | get remote local
import sys, paramiko, os
HOST, USER, PASS = "5.180.173.14", "root", "<пароль из раздела 12>"
mode, a, b = sys.argv[1:4]
t = paramiko.Transport((HOST, 22)); t.connect(username=USER, password=PASS)
s = paramiko.SFTPClient.from_transport(t)
(s.put if mode == "put" else s.get)(a, b); print(mode, a, "->", b); s.close(); t.close()
```
Для каталогов: `tar czf` на сервере → `get` → распаковать (и наоборот). Коммит: rsync `$B` → `/tmp/densito_wc` (excl. `.git outputs logs __pycache__`), `git add -A`, `git -c user.name='DensitoAI Team' -c user.email='team@densitoai.local' commit`, `git push origin HEAD:main`, `git -C /opt/git/densitoai.git update-server-info`. Никогда не запускать `pkill -f '<шаблон>'` в ssh-команде, содержащей этот шаблон.

### 0.5. Артефакты, отданные владельцу 19.09 (имена для обновления версий)
«Консилиум DensitoAI — синтез пяти заключений (раунд 1)», «Консилиум DensitoAI — раунд 2 и финальный приоритетный план», пять отчётов раунда 1 и пять раунда 2 (имена — в `docs/council/README.md`). Полный handover с доступами: `/opt/neuropeople/densito_private/HANDOVER-DensitoAI-v3-FULL.md` (только на сервере, `chmod 600`, в git не попадает — каталог вне `$B`).

---


Обновлено: 18.09.2026, 21:40 MSK. Версия продукта 2.1.0, последний коммит `07f1a3d` (main).
Автор передачи: ассистент (Claude) в аккаунте владельца. Получатель: любой человек или новая
модель-ассистент, у которой нет памяти об этом проекте.

**Как читать.** Раздел 1 — кто заказчик и как с ним работать (обязательно). Раздел 2 — где всё
лежит и как войти. Раздел 3 — что сделано и что задеплоено прямо сейчас. Раздел 4 — что не
сделано и в каком порядке делать. Дальше — устройство продукта, метрики, рецепты, грабли.
После этого файла читать: `PLAN.md` (единый план и журнал), `README.md` (597 строк, полная
документация), `docs/METRICS_REPORT.md`, `docs/ROBUSTNESS_REPORT.md`, `docs/GPU_EXPERIMENT.md`,
`models/MODEL_CONTRACT.md`, `docs/IMPROVEMENT_ANALYSIS.md`.

Секреты (пароли, ключи) — в разделе 12 этого файла. **Файл с разделом 12 нельзя класть в
публичный git.** В репозитории лежит версия без раздела 12 (`HANDOVER.md`).

---

## 1. Заказчик, цель, правила работы

### 1.1. Цель
Победить в конкурсе «Лидеры цифровой трансформации 2026» (Москва), задача 4 — «Сервис ИИ по
оценке качества исследований плотности костей человека» (постановщик — Департамент
здравоохранения Москвы, ДЗМ). Команда — **DensitoAI**. ФИО участников и контакты **до сих пор
неизвестны** — в презентации и письме организаторам стоят заглушки; спросить владельца в конце.

Финальная проверка — экспертами организатора на **закрытом** наборе без участия команды;
метрики считает организатор (F1 и ROC-AUC приоритетны, желательны 95 % ДИ), плюс критерии
ТЗ §8.1–8.5 (формат выгрузки, скорость ≤ 3 мин на исследование, устойчивость, документация,
презентация). ТЗ и разъяснения: `uploaded_attachments/.../4.-DepZdrav.pdf`,
`Raziasneniia-po-voprosam-LTsT_V2.docx`, `Instruktsiia-dlia-uchastnika.pdf`, транскрипт Q&A и
его разбор — `docs/qa/` (`QA_ANALYSIS.md`, `QA_COMPLIANCE_MATRIX.md`, `QA_WRITTEN_CLARIFICATIONS.md`,
`QA_SESSION_TRANSCRIPT.txt`). Шаблон презентации организаторов —
`LTsT2026-Shablon-prezentatsii.pptx` (у владельца в исходных файлах).

### 1.2. Владелец и стиль общения (важно, нарушение — конфликт)
- Пишет **голосовым вводом** и иногда в **неправильной раскладке** (пример: «dct ljcnegs» =
  «все доступы»). Текст искажён — читать благожелательно, переспрашивать только в крайнем случае.
- Нетерпелив, при затягивании — мат. Не обижаться, действовать.
- Ответы: **по-русски, коротко, без восклицательных знаков, без эмодзи**, без «давайте начнём».
- **Сначала продукт, потом бумажки.** Цитата: «сначала продукт, его нужно протестировать, потом
  все бумажки». Презентация и ТЗ рентгенологу — **в самом конце**, когда всё остальное сделано.
- **Работать автономно до конца**, не задавать вопросов, если можно решить самому. Деплой
  разрешён без подтверждения («да, деплой и не спрашивай»).
- **Не экономить ресурсы**: «проект дорогой для меня, поэтому мы здесь максимум ресурсов
  вкладываем» (GPU, модели — бери лучшее).
- **Никаких всплывающих окон/форм для ключей** — владелец злится («хватит усложнять мне жизнь
  своими всплывающими окнами»). Все ключи — в этом файле (раздел 12), разрешение дано явно.
- **Один трекинг-файл** — `PLAN.md`. Владелец жаловался, что файлов слишком много и он запутался.
  Не плодить новые документы без нужды; после каждого шага — короткий отчёт в чате (5–10 строк, что
  сделано, цифры, что дальше).
- **По одному шагу, от простого к сложному**, каждый шаг: код → тест → деплой → коммит → строка в
  PLAN.md → отчёт.
- Владелец **смотрит сайт сам** и сравнивает с тем, что было. Последняя реакция (18.09 21:35):
  «У нас до этого было красиво, и сайт гораздо больше, там и демо можно было запустить» — ему
  **не понравился** нынешний интерфейс «кабинета» по сравнению с первой лендинг-версией (15.09).
  Подробнее — раздел 4.1. Это первое, что надо решить при продолжении.

### 1.3. Что владелец уже сказал про содержание продукта
- «Все снимки в датасете сделаны в Москве на одних и тех же аппаратах» — учитывать: OOD-детектор
  «чужой аппарат» (шаг 7) имеет смысл; на реальном потоке искажения экспозиции не возникают.
- Очень интересует **режим лаборанта** и **полноценный кабинет врача**: очередь, карточка решения,
  кнопки «согласен / исправить / не уверен», журнал решений, панель отделения. Это шаги 5–6.
- Хотел 200–300 дополнительных снимков для разметки рентгенологом: решение — отобрать из открытых
  DXA-наборов (Arak 4020 бедро, DEXA-Osteo 174 позвоночник) снимки со скором у порога
  (active learning), шаг 5.

---

## 2. Где всё лежит, как войти

### 2.1. Сервер Madrid (единственный боевой) — 5.180.173.14
Доступ — раздел 12. 4 ядра, 5 ГБ RAM (свободно ~2 ГБ), диск 118 ГБ (занято 73 %). На нём же
живёт **чужой продакшен NEUROPEOPLE** (pm2-процессы, docker `np-postgres`, `np-redis`, `np-livekit`,
`np-postgres-gemini`, домены neuropeople.shop / neuropeople.site) — **не трогать**, см. раздел 11.

| Что | Где |
|---|---|
| Живой сайт | `https://neuropeople.pro` — basic auth (раздел 12); `/api/` без auth |
| Старая лендинг-версия сайта (15.09) для сравнения | `https://neuropeople.pro/v1/` (копия `/opt/neuropeople/densitoai-web-v1-backup/`, выложена 18.09 21:40; её кнопка загрузки ходит в старый API и работать не будет) |
| Swagger | `https://neuropeople.pro/docs`, `https://neuropeople.pro/openapi.json` |
| Публичный git (bare) | `https://neuropeople.pro/git/densitoai.git`, gitweb `https://neuropeople.pro/git/browse?p=densitoai.git;a=summary`; на диске `/opt/git/densitoai.git`; ветка `main` (есть лишняя `master` — не трогать) |
| Рабочая копия git | `/tmp/densito_wc` (может исчезнуть после ребута → `git clone /opt/git/densitoai.git /tmp/densito_wc`) |
| **Каталог сборки = источник истины кода** | `B=/opt/neuropeople/densito_rebuild_test/densito_rebuild/` |
| Docker | образ `densitoai:2.1.0`, контейнер `densitoai-api-container`, порт `127.0.0.1:8026 → 8000`, `--restart unless-stopped`, healthcheck |
| Выходы API | `/opt/neuropeople/densito_api_outputs` (в контейнере `/data/output`), внутри `jobs/<job_id>/` |
| Веб-статика | `/opt/neuropeople/densitoai-web/index.html` (копия `$B/web/index.html`), `assets/`, старые pptx/pdf презентации |
| Nginx | `/etc/nginx/sites-available/noema-manus` (бэкап `.bak_before_full_replace_20260915`); `/` — auth_basic, `/api/` — `auth_basic off` → 8026, `/docs`, `/openapi.json`, `/git/…` — git-http-backend + gitweb, `/assets/`; htpasswd `/etc/nginx/.htpasswd_densitoai` (бэкап `.bak_<ts>`) |
| Старый v1-бэкенд (15.09, systemd `densitoai-api.service`, venv `/opt/neuropeople/densitoai-api/`) | **выключен и disabled**, заменён контейнером; бэкап `/opt/neuropeople/backups/hackathon_v1/` |
| Внешние датасеты | `/opt/neuropeople/external_datasets/{BUU-LSPINE 890M, MTDDH 1011M, FracAtlas 353M, kaggle/{arak-bone-densitometry-center, dexa-osteo, aasce-miccai-2019-x-ray-dataset} 326M, chinese_osteoporosis/Hip 30M (частично), own_dataset 43M}` |
| Свой датасет на сервере | `/opt/neuropeople/external_datasets/own_dataset/Исследования/...` (1162 dcm, включая «прочее») |
| GPU-чекпоинты | `/opt/neuropeople/gpu_ckpt/proxy_b0/backbone_ep00..59.pth` (копия с RunPod) |
| Приватный хендовер (старый, короткий) | `/opt/neuropeople/HANDOVER_PRIVATE.md`; **этот файл** — `/opt/neuropeople/HANDOVER_FULL_2026-09-18.md` |
| Скрипт живой проверки | `/tmp/live_check.sh` (curl с basic auth и 3 DICOM → `/tmp/live_resp.json`) |

### 2.2. Локальная песочница ассистента (пропадёт вместе с сессией)
`/home/user/workspace/densito_rebuild/` — полная копия кода, моделей, данных (синхронизирована с
`$B` на сервере по состоянию на коммит `07f1a3d`). Датасет владельца распакован в
`/home/user/workspace/dataset_extracted/Датасет/НД_для_обучения/Исследования/<study>/…/*.dcm`
(499 снимков / 100 исследований + xlsx разметки); пути к файлам — в `data/labels_for_embeddings.csv`.
Исходные файлы владельца — `/home/user/workspace/uploaded_attachments/…` (Dataset.zip, ТЗ, шаблон
презентации, доступы, старый хендовер 15.09). **В новом аккаунте всё это надо взять с сервера
(`$B`) или снова загрузить Dataset.zip у владельца** — данные `data/*.csv|npy` есть в `$B/data/`.

### 2.3. RunPod (GPU, оплата поминутно)
- Аккаунт владельца, баланс ≈ **$7.5** (было ~$9, потрачено ≈ $1.5).
- Текущий под **`11rn2rqqig7fwy`**, RTX PRO 4500 ($0.72/ч), состояние **STOPPED** (podStop →
  EXITED). Контейнерный диск `/data` при стопе теряется; том `/workspace` сохранён: `/workspace/ckpt/…`
  (чекпоинты), `/workspace/densito/` (скрипты и кэш тайлов). Возобновить — `podResume` через GraphQL
  `https://api.runpod.io/graphql` (ключ — раздел 12) или в веб-консоли RunPod.
- Старый под `58lzazalfl39hd` (RTX 4090, 15.09) — не используется.
- За хранение остановленного тома идёт небольшая плата (центы в день). Если GPU больше не нужен —
  удалить под, чекпоинты уже на Madrid.

### 2.4. Kaggle
Токен (раздел 12) — для `kaggle datasets download`. Уже скачаны: Arak (4020 DXA бедра PNG),
DEXA-Osteo (174 DXA позвоночника PNG), AASCE (481 позвоночник). Частично — Chinese Osteoporosis
(только Hip, 30 МБ); полный набор ~9800 ROI — не докачан.

---

## 3. Состояние на момент передачи (что задеплоено и работает)

### 3.1. Продукт
- Docker-сервис только на CPU: DICOM (папка / zip / файлы) → CSV/XLSX строго по ТЗ п.2.5 (9 полей)
  + REST API + веб-интерфейс + PNG-визуализация + DICOM SR + предложение исправленного ROI бедра.
- Две области: поясничный отдел позвоночника и проксимальный отдел бедра (левое/правое). Пять
  критериев: `sp_pos` (укладка позвоночника), `sp_axis` (ось), `sp_art` (инородные предметы),
  `hip_pos` (укладка бедра), `hip_roi` (ROI бедра).
- Архитектура двухконтурная: контур A — интерпретируемая геометрия кости (сегментация, ось, углы,
  ширины в мм), контур B — эмбеддинги EfficientNet-B0 (ImageNet; для `sp_pos` — наш GPU-бэкбон
  `densito`), ранговый стэкинг по критерию, any-модель для `quality_prob`.
- Живой сайт отвечает, контейнер `healthy`, 36 API-проверок и тест формата зелёные, стресс-тест
  пройден (раздел 6).

### 3.2. Git-история (main)
```
07f1a3d Step 2 closed: robustness suite, broken zip -> Failure/400, 36 API checks, README benchmark
bcea158 Step 4: hybrid backbones — GPU-pretrained densito B0 for sp_pos; gpu/ scripts, GPU_EXPERIMENT.md
2a47d43 Step 2: per-request result isolation (jobs/<id>), traversal hardening, limits, /api/jobs, API tests
b1b806d web: гибкие колонки очереди, карточка без переполнения
d34cff0 Step 1: кабинет v1 — очередь по риску, карточка решения, XLSX-сводка, API details; безопасный zip
b88bafa docs: PLAN.md
6d6d7f8 docs: HANDOVER, radiologist brief, improvement analysis, independent review
36f09b5 / 93a0a3a docs: presentation (14 слайдов на шаблоне организаторов)
6de7926 / c419edd / 85e57d5 v2.1.0: регион по содержимому, согласованный quality_prob, отчёт метрик с ДИ
e2c4ec7 первая публикация (v2.0.x)
```

### 3.3. Что ещё не в порядке (быстрые пункты)
1. **`docs/METRICS_REPORT.md` и `docs/metrics_oof_full.md` устарели** (12:53, до шага 4): после
   замены бэкбона для `sp_pos` бинарные метрики позвоночника могли измениться. Пересчитать:
   `python src/eval_oof_metrics.py` (обновит `models/metrics_oof_full.json`, `docs/metrics_oof_full.md`),
   затем руками сверить числа в `docs/METRICS_REPORT.md`, README §10 и презентации.
   Актуальные по-критериальные цифры — `models/metrics_summary.json` (18:07) и раздел 6 ниже.
2. В документах местами упомянут **старый пароль сайта `Densito2026`** (`docs/EXPERT_TESTING_GUIDE.md`,
   `docs/RADIOLOGIST_TEST_BRIEF.md`, файл доступов владельца, `HANDOVER_PRIVATE.md`). Пароль изменён
   18.09 ~21:00 (раздел 12). Обновить перед финалом.
3. ФИО/контакты команды — заглушки (презентация `docs/DensitoAI_LCT2026_presentation.pptx`,
   `docs/LETTER_TO_ORGANIZERS.md`).
4. Старая landing-версия сайта нравилась владельцу больше — раздел 4.1.

---

## 4. Что делать дальше (порядок согласован с владельцем)

### 4.1. Сначала — веб-интерфейс: вернуть «красиво и много» (реакция владельца)
История: 15.09 предыдущий ассистент сделал **лендинг** (`/v1/`, 25 КБ): логотип NEUROPEOPLE, описание,
6 карточек кейсов с Grad-CAM (`assets/gradcam_*.png`), блок benchmark-карточек, зона загрузки
(«демо»), ссылки на pptx/pdf презентации. 18.09 12:19 он был заменён на «рабочий» интерфейс v2.1.0
(10 КБ), 18.09 18:22 — на «кабинет v1» (35–37 КБ: очередь по риску, фильтры, KPI, карточка решения,
XLSX, история запросов). Владелец увидел кабинет 18.09 21:35 и сказал, что раньше было красивее и
«сайт гораздо больше, там и демо можно было запустить».

Предлагаемое решение (не реализовано): **объединить** — сделать главную страницу лендингом в
стилистике v1 (логотип, что это, как работает, карточки кейсов с реальными визуализациями из
`outputs/…/viz/*.png`, метрики с ДИ, ссылки на презентацию, git, Swagger, документы) с крупной
кнопкой «Открыть демо / кабинет» → текущий кабинет (перенести в `/app/` или секцией ниже). Кабинет
оставить как есть функционально, но подтянуть визуал под лендинг (шрифты, цвета, отступы, крупные
заголовки). Показать владельцу скриншоты **до** деплоя. Файлы: `$B/web/index.html` (кабинет),
`/opt/neuropeople/densitoai-web-v1-backup/index.html` (лендинг), assets там же.

### 4.2. Дальше по PLAN.md (статусы на 18.09)
| № | Шаг | Статус |
|---|---|---|
| 1 | Быстрые правки удобства (кабинет v1, XLSX-сводка, API details) | ☑ |
| 2 | Надёжность API + стресс-тест | ☑ |
| 3 | Честный риск и зона «не уверен» | ☐ следующий |
| 4 | GPU-бэкбон для редких классов (гибрид) | ☑ |
| 5 | Карточка решения и цикл врача (согласен/исправить/не уверен, JSONL-журнал, экспорт для дообучения, active-learning отбор 200–300 открытых DXA) | ☐ |
| 6 | Режим лаборанта («переснять сейчас») и панель отделения (доля пересъёмок, тренды по дате/аппарату/оператору из тегов DICOM) | ☐ |
| 7 | OOD-детектор чужого аппарата + проверка экспозиции/шума | ☐ |
| 8 | Nested CV и model card | ☐ |
| 9 | Внедрение: watch-folder / DICOM C-STORE, SR через Orthanc | ☐ |
| 10 | Финал: ТЗ рентгенологу (черновик готов `docs/RADIOLOGIST_TEST_BRIEF.md`), правки по заключению, ФИО, презентация | ☐ в самом конце |

Проектные заметки по шагам:

**Шаг 3 (риск).** В `process_file` уже есть per-criterion `*_score`, `*_threshold`, `*_flag`,
`*_p_geom`, `*_p_emb` в debug-строке. Сделать: cross-fit калибровку (Platt/isotonic внутри
GroupKFold по исследованиям в `train_stacked.py`), зону отказа по запасу |score − порог| (в
стресс-тесте 57–76 % переворотов при < 0.15, во всей выборке таких 30 % — компромисс покрытие/
надёжность считать по OOF, risk–coverage кривая), Brier/ECE и reliability diagram → в
`METRICS_REPORT.md`; в debug-CSV и API `details` — `risk_level` (низкий/средний/высокий) и флаг
«требует врача»; в вебе — цвет строки и бейдж. Официальные 9 колонок CSV **не менять**.

**Шаг 5 (цикл врача).** API: `POST /api/jobs/{job}/decisions` (file, criterion, decision ∈
{agree, fix, unsure}, comment, corrected_flags) → `outputs/decisions/decisions.jsonl` (+ по job);
`GET /api/decisions/export` → CSV с меткой врача для `train_stacked.py --extra-labels`. Скрипт
дообучения на подтверждениях. Active learning: прогнать Arak/DEXA-Osteo PNG через инференс (нужно
обёрнуть PNG в псевдо-DICOM или добавить чтение PNG за флагом), взять 200–300 с |score − порог|
минимальным, разложить в папку для рентгенолога вместе с формой разметки (xlsx как у ДЗМ).

**Шаг 6 (лаборант/отделение).** Экран для лаборанта: один снимок → крупный вердикт «переснять
сейчас / ок» + причина + подсказка укладки (тексты уже в `config.yaml` → `actions`), крупные
кнопки. Панель отделения: агрегаты по `outputs/jobs/*/summary.json` и debug-CSV: доля пересъёмок,
типы, тренд по дате (`StudyDate`), аппарату (`StationName`, `DeviceSerialNumber`,
`ManufacturerModelName`), оператору (`OperatorsName`) — теги читать в `read_and_validate`
(сейчас читаются Manufacturer/Model/StudyDate частично, см. `_tag`). Владелец: аппараты в Москве
одни и те же — трендов по аппарату будет мало, по дате/оператору — есть смысл.

**Шаг 7 (OOD).** Fingerprint тегов (Manufacturer=GE, ModelName Lunar, Rows/Cols 300/280/248,
PixelSpacing, BitsStored) + расстояние Mahalanobis в PCA-пространстве эмбеддингов до обучающего
облака + статистики экспозиции (медиана/перцентили пикселей тела, оценка шума по лапласиану) →
предупреждение «нестандартный источник / экспозиция, оценка ненадёжна». Тест: Arak/DEXA-Osteo PNG,
рентгены FracAtlas, искажения из `tests/robustness_suite.py`.

**Шаг 8.** Вложенная GroupKFold (внешняя 5 × внутренняя 3) с подбором порога/PCA внутри; таблица
рядом с текущими цифрами; model card (`models/MODEL_CARD.md`).

**Шаг 9.** Watch-folder (`--watch <dir>`) в `inference.py` + опционально pynetdicom C-STORE SCP;
SR валидировать dciodvfy, round-trip в Orthanc (docker `jodogne/orthanc`).

---

## 5. Устройство продукта

### 5.1. Карта кода (`$B` / `densito_rebuild/`)
```
src/inference.py            ядро (~1400 строк): discover_files/collect inputs (zip, вложенные zip, Zip Slip,
                            кодировки cp866/cp1251, битые архивы → Failure), normalize_pixels (MONOCHROME1,
                            перцентили 1–99 %), read_and_validate, classify_region (теги → ширина 300/280/248 →
                            контентная модель model_region_emb.pkl), ModelBundle/ModelRegistry (pkl + manifest),
                            EmbeddingExtractor (несколько бэкбонов по source: imagenet/densito, зеркалирование),
                            DensitoInference.process_file (строка CSV + debug-строка), .run (пакет, CSV/XLSX/debug,
                            бонус-файлы), CLI main
src/api_server.py           FastAPI: GET /api/health; POST /api/analyze (multipart files, лимиты 500 файлов /
                            512 МБ, ValueError → 400); POST /api/batch (папка на сервере, output только внутри
                            OUTPUT_DIR); GET /api/results/{job}/{name}; GET /api/jobs, /api/jobs/{job};
                            legacy GET /api/results/{name}. Инференс сериализован (lock). Результаты в
                            outputs/jobs/<job_id>/ (results.csv, debug.csv, xlsx, summary.json, viz/, sr/, roi/)
src/geometry_features.py    контур A позвоночник: сегментация (Otsu на сглаженном), ось, центр, металл
src/hip_features.py         контур A бедро: segment_bone_hip (строчно-адаптивный Otsu·0.75), диафиз, угол, ROI, сторона
src/embeddings.py           FrozenBackbone(source) → 1280-d; BACKBONE_FILES = {imagenet: torchvision, densito: models/backbone_densito.pth}
src/embeddings_hip_canonical.py  канонизация бедра (зеркалирование) для эмбеддингов
src/train_stacked.py        OOF 5×GroupKFold по исследованию, per-criterion geom/emb модели, ранговый стэкинг,
                            пороги; EMB_SOURCE_BY_CRITERION = {'sp_pos': 'densito'}; → models/metrics_summary.json
src/train_final_models.py   финальные модели на всех данных → models/*.pkl + manifest (emb_source внутри pickle)
src/eval_oof_metrics.py     отчёт ТЗ §8.4 с бутстрап-ДИ → docs/metrics_oof_full.md, models/metrics_oof_full.json
src/build_dataset.py, extract_all_features.py   разметка xlsx → data/labels_full.csv; признаки → data/geometry_features.csv
src/visualize_report.py     PNG-оверлей (ось, контур, ROI, предметы, панель скор/порог)
src/dicom_sr.py             DICOM Structured Report;  src/auto_roi.py  исправленный ROI бедра
src/train_multilabel.py, hip_eval.py   вспомогательные/исторические
gpu/prepare_cache.py, pretrain_proxy.py, eval_embeddings.py   GPU-предобучение (RunPod), см. docs/GPU_EXPERIMENT.md
web/index.html              кабинет (чистый HTML/JS, textContent везде — XSS закрыт)
config.yaml                 версия 2.1.0, ВСЕ пороги, веса, строки регионов/нарушений, actions, validation, output
models/                     *.pkl (по критерию geom + emb_pca; any-модели по области; model_region_emb.pkl),
                            backbone_densito.pth (16 МБ), manifest с SHA, metrics_summary.json, MODEL_CONTRACT.md,
                            cnn_ml_*.pth — старые v1-модели (не используются)
data/                       labels_full.csv, labels_for_embeddings.csv (пути к dcm!), geometry_features.csv,
                            embeddings.npy (ImageNet), embeddings_densito.npy, embeddings_hip_canonical.npy
tests/test_inference_format.py   формат выгрузки (запускается при docker build)
tests/test_api_isolation.py      36 проверок API (изоляция jobs, traversal, лимиты, мусор, битый zip, параллельные zip)
tests/robustness_suite.py        стресс-тест: --n 80 → docs/ROBUSTNESS_REPORT.md, docs/qa/robustness_results.json
tests/sample_test_zip/           3 DICOM для smoke
Dockerfile, docker-entrypoint.sh (режимы api / batch), docker-compose.yml, build_and_run.sh, requirements.txt
docs/                       см. раздел 2.1 и README §2
```

### 5.2. Ключевые решения
- Регион: теги → ширина кадра (300 = позвоночник, 280/248 = бедро, подтверждено организаторами) →
  при нестандартной ширине контентная модель; имя файла — только подсказка.
- Бедро право/лево — одна модель с зеркалированием; сторона в выгрузку не выводится (ТЗ).
- `quality_class = 1`, если хотя бы один критерий ≥ порога; `violation_type` — все сработавшие через
  `;`; `quality_prob = 0.5·any + 0.5·max(критерии)`, согласована с классом (класс 1 ⇔ prob ≥ 0.5).
- Сбой файла → строка `Failure`, `quality_class 0`, `quality_prob 0.5`, остальные обрабатываются.
- Пороги: `sp_pos 0.791` (prevalence), `sp_axis 0.756`, `sp_art 0.557`, `hip_pos 0.709`, `hip_roi 0.891`
  (F1-optimal OOF) — в `metrics_summary.json` и `config.yaml`.
- Гибрид бэкбонов (шаг 4): `densito` только для `sp_pos`; `hip_pos` на боевом протоколе выигрыша не дал
  (0.635→0.636) — оставлен ImageNet. Инференс грузит оба бэкбона, для позвоночника два эмбеддинга
  (+~0.3 с/файл CPU).

### 5.3. Веб-кабинет (текущий)
Загрузка (drag&drop файлов/zip, прогресс, таймер), KPI-карточки, очередь по риску с фильтрами
(все/нарушения/норма/ошибки, область, поиск), сортировкой и группировкой по исследованию, карточка
решения по снимку (критерии со скорами/порогами/методом, измерения в мм/°/%, ROI и оверлей, действие
принять/проверить/вручную), скачивание CSV/XLSX/debug-CSV, панель «История запросов» (`/api/jobs`),
клавиатура, понятные ошибки.

---

## 6. Метрики (OOF, GroupKFold по исследованию; 499 снимков / 100 исследований)

Актуально по критериям (`models/metrics_summary.json`, 18.09 18:07, после шага 4):

| Критерий | n / pos | AUC контур A | AUC контур B (источник) | AUC стек | F1 OOF | Порог |
|---|---|---|---|---|---|---|
| sp_pos укладка позв. | 166 / 10 | 0.611 | 0.797 (densito) | 0.715 | 0.400 | 0.791 |
| sp_axis ось | 166 / 17 | 0.839 | 0.491 (imagenet) | 0.738 | 0.381 | 0.756 |
| sp_art предметы | 166 / 35 | 0.560 | 0.897 (imagenet) | 0.823 | 0.598 | 0.557 |
| hip_pos укладка бедра | 329 / 79 | 0.694 | 0.635 (imagenet) | 0.704 | 0.511 | 0.709 |
| hip_roi ROI бедра | 329 / 16 | 0.902 | 0.877 (imagenet) | 0.917 | 0.636 | 0.891 |

Бинарные (по области), **до шага 4** (`docs/METRICS_REPORT.md`, пересчитать — п. 3.3.1):
позвоночник ROC-AUC 0.775 [0.64; 0.89], F1 0.662 [0.50; 0.79], Sens 0.80 / Spec 0.65;
бедро ROC-AUC 0.773 [0.62; 0.90], F1 0.667 [0.48; 0.80], Sens 0.60 / Spec 0.92.
Скорость: p50 0.04 с / p95 0.07 с на снимок без бонус-файлов (2 vCPU); в контейнере с
визуализациями/SR/ROI ≈ 2 с/файл; холодный старт ≈ 18 с. 499/499 Success.

Честные ограничения (говорить всегда): 10 положительных для sp_pos (ДИ F1 [0.00; 0.72]); все
цифры — на своей разметке, финальная проверка — рентгенолог (шаг 10) и закрытый набор организатора.

Стресс-тест (`docs/ROBUSTNESS_REPORT.md`, 81 снимок × 11 искажений): формат-инварианты (пересохранение,
удаление тегов и UID, MONOCHROME1, 16 бит) — 0 % переворотов класса; обрезка 4 % — 9 %; resize ±20–25 %
— 14–15 %; шум σ=3 % — 22 %; гамма 0.7/1.4 — 25–28 %. Область ни разу не перепутана, Failure 0 %.
Причина чувствительности: маска тела `img>8` и порог `0.75·Otsu` не инвариантны к шуму/гамме; 57–76 %
переворотов — у снимков с запасом до порога < 0.15. Решено закрывать шагами 3 и 7, признаки не
переобучать до разметки рентгенолога.

GPU-эксперимент (`docs/GPU_EXPERIMENT.md`): пул 15 633 фрагментов (FracAtlas, Arak, MTDDH, BUU-LSPINE,
AASCE, DEXA-Osteo, свои 499 без меток), прокси-задачи (поворот, сдвиг, масштаб, синтетический
металл), EfficientNet-B0, 60 эпох × 11 с. Парная проверка 10 повторов: sp_pos +0.13 AUC (10/10),
hip_pos +0.07 (9/10), sp_art −0.14 (0/10), ось/ROI нейтрально → гибрид.

---

## 7. Рецепты (проверенные команды)

Все ssh — через `sshpass` и `timeout`; у ассистента окно ожидания 30 с, длинные операции запускать
в фоне с логом. Имена с кириллицей/пробелами в inline-ssh ломают кавычки — использовать скрипты
(scp → `bash script`). **Никогда** не делать `pkill -f '<pattern>'` в ssh-команде, содержащей тот же
pattern (убьёт саму ssh-сессию).

```bash
export SSHPASS='<пароль root, раздел 12>'
S="sshpass -e ssh -o StrictHostKeyChecking=no root@5.180.173.14"
B=/opt/neuropeople/densito_rebuild_test/densito_rebuild

# 1. Синхронизация кода из локальной копии на сервер
sshpass -e rsync -a -e "ssh -o StrictHostKeyChecking=no" --exclude __pycache__ src/ root@5.180.173.14:$B/src/
# аналогично tests/ docs/ web/ models/ PLAN.md README.md

# 2. Сборка образа (~1 мин с кэшем, 10–15 мин с нуля), в фоне
$S "cd $B && rm -f /tmp/build.log; setsid nohup bash -c 'docker build -t densitoai:2.1.0 . > /tmp/build.log 2>&1 && echo BUILD_OK >> /tmp/build.log' > /dev/null 2>&1 < /dev/null &"
$S "tail -1 /tmp/build.log"        # ждём BUILD_OK; в логе есть 'format check: OK' от теста формата

# 3. Перезапуск контейнера (разрешено без подтверждения)
$S "docker rm -f densitoai-api-container; docker run -d --name densitoai-api-container --restart unless-stopped -p 127.0.0.1:8026:8000 -v /opt/neuropeople/densito_api_outputs:/data/output densitoai:2.1.0 api"
$S "sleep 20; curl -s http://127.0.0.1:8026/api/health; bash /tmp/live_check.sh"

# 4. Обновление веба
$S "cp $B/web/index.html /opt/neuropeople/densitoai-web/index.html"

# 5. Коммит в публичный git
$S "cd /tmp/densito_wc && rsync -a --delete --exclude .git --exclude outputs --exclude logs --exclude __pycache__ $B/ ./ && git add -A && git -c user.name='DensitoAI Team' -c user.email=team@densitoai.local commit -qm 'msg' && git push -q origin HEAD:main && (cd /opt/git/densitoai.git && git update-server-info)"

# 6. Тесты (локально, в каталоге проекта)
python tests/test_inference_format.py           # ALL CHECKS PASSED
python tests/test_api_isolation.py              # ИТОГ: все проверки пройдены (36)
python -u tests/robustness_suite.py --n 80      # ~5 мин, отчёт в docs/ROBUSTNESS_REPORT.md

# 7. Переобучение после изменения признаков/меток
python src/extract_all_features.py              # → data/geometry_features.csv
python src/train_stacked.py                     # OOF, пороги → models/metrics_summary.json
python src/train_final_models.py                # → models/*.pkl + manifest
python src/eval_oof_metrics.py                  # → docs/metrics_oof_full.md
# затем: инференс на 499 файлах, diff CSV с предыдущим, тесты, README §10, METRICS_REPORT, CHANGELOG

# 8. Локальный запуск без Docker
pip install -r requirements.txt
python src/inference.py --input <папка|zip> --output outputs/results.csv --xlsx --debug-csv
uvicorn api_server:app --app-dir src --port 8000

# 9. RunPod GraphQL (ключ — раздел 12; заголовок Authorization: Bearer <key> или ?api_key=)
curl -s -X POST https://api.runpod.io/graphql -H 'content-type: application/json' \
  -d '{"query":"{ myself { pods { id name desiredStatus runtime { uptimeInSeconds } } } }"}'
# podResume(input:{podId:"11rn2rqqig7fwy", gpuCount:1}) { id desiredStatus } ; podStop(input:{podId:"..."})
```

Регрессия после любого изменения инференса: прогнать все 499 файлов, `diff` CSV с предыдущим,
оба теста, smoke через живой API.

---

## 8. Хронология работы (для понимания, что уже пробовали)

- **15.09** (предыдущий ассистент, v1): лендинг + CNN (15 .pth, `cnn_ml_*`), systemd-сервис, презентация
  черновик. Хендовер `HANDOVER-Khakaton-v1-15.09.2026.md`.
- **18.09 утро–день**: полная пересборка (v2.0 → 2.1.0): двухконтурная архитектура, формат ТЗ, контейнер,
  API, веб, DICOM SR, auto-ROI, отчёт метрик с ДИ, разбор Q&A организаторов, письмо, презентация 14 слайдов,
  независимое ревью (`docs/INDEPENDENT_REVIEW_2026-09-18.md`, `docs/review_fable5.md`).
- **18.09 ~18:00**: владелец попросил: (1) найти большой датасет, (2) внедрять улучшения по одному от простого
  к сложному, (3) рентгенолог — в конце. Создан `PLAN.md`. Глубокий поиск датасетов агентом
  (`docs/DATASETS_DEEP_SEARCH.md`): сырых DXA-DICOM GE Lunar в открытом доступе нет; скачано, что есть.
- **18.09 18:22**: шаг 1 (кабинет v1) задеплоен; попутно исправлены `path_to_study` с tmp-путём, кракозябры
  имён из Windows-zip, Zip Slip.
- **18.09 ~18:40–20:30**: RunPod — под, кэш тайлов, предобучение 60 эпох, парная проверка; стоимость ≈ $1.5.
- **18.09 19:53**: шаг 2 часть 1 (изоляция jobs, лимиты, история).
- **18.09 ~20:50**: шаг 4 — гибридные бэкбоны внедрены и задеплоены (`bcea158`).
- **18.09 ~21:00**: пароль сайта изменён по просьбе владельца.
- **18.09 21:30**: шаг 2 закрыт (стресс-тест, битый zip), `07f1a3d`.
- **18.09 21:35**: владелец недоволен внешним видом сайта, просит подробный хендовер для другого аккаунта.
  Выложена `/v1/` для сравнения. Написан этот файл.

Отвергнутые варианты (не повторять без новых данных): конкатенация эмбеддингов двух бэкбонов
(0.677 vs 0.676); densito-бэкбон для hip_pos на боевом протоколе; densito для sp_art (хуже на 0.14);
альтернативные правила `quality_prob` (см. METRICS_REPORT «компоненты»); использование столбца «Итог»
как таргета; чтение имени файла для стандартных кадров.

---

## 9. Данные и разметка

- Своя разметка: 499 снимков / 100 исследований (xlsx организаторов), ключ соединения — папка
  исследования; «Итог» не используется как таргет; правило «ROI не помечается при некорректной укладке».
  `data/labels_full.csv` (v1 с бэкапом), `data/labels_for_embeddings.csv` (пути).
- Все снимки — GE Lunar, Москва, одни аппараты (слова владельца); ширины 300 (позвоночник), 280/248 (бедро).
- Внешние наборы — таблица в `PLAN.md` §0 и `docs/DATASETS_DEEP_SEARCH.md`. Ни один не содержит нужной
  разметки качества DXA; польза — предобучение, OOD-тест, active learning для рентгенолога.
- Не скачано, но доступно: Zenodo 15352880 (таз/бедро, 76 МБ, прямая ссылка), PelviXNet (760 МБ),
  MosMedData пояснично-крестцовый (e-mail-форма), VinDr-SpineXR (PhysioNet DUA), OAI (заявка).

---

## 10. Артефакты, отданные владельцу в чат (имена для обновления версий)
«PLAN — единый план доработок DensitoAI» (из `PLAN.md`), «HANDOVER — DensitoAI», «Анализ Q&A с
организаторами хакатона», «DensitoAI — презентация ЛЦТ 2026» (pptx), «Анализ улучшений и фишек
DensitoAI», «ТЗ рентгенологу на экспертную оценку DensitoAI». Все исходники — в `$B/docs/`.

---

## 11. Правила и грабли

Безопасность окружения (сервер общий с продакшеном NEUROPEOPLE):
1. Не трогать neuropeople.shop и neuropeople.site, pm2-процессы (особенно `stt-stream`), контейнеры `np-*`,
   Казахстан VPN 195.133.8.45, Амстердам Telegram-бот 185.109.48.82.
2. Перед правкой nginx — бэкап конфига, `nginx -t && systemctl reload nginx`, потом проверить .shop/.site.
3. RAM на сервере мало (~2 ГБ свободно): не запускать обучение и тяжёлый инференс на 499 файлах на сервере
   параллельно со сборкой; обучение — локально у ассистента или на RunPod.
4. Диск 73 % — следить, чистить `outputs/jobs/` старше недели и docker build cache при необходимости.

Методология:
5. Любая правка модели/порогов — только через OOF с ДИ; после — `train_final_models.py`, manifest, CHANGELOG,
   README §10, METRICS_REPORT, презентация — **одни и те же числа везде**.
6. Не менять строки регионов/нарушений, порядок 9 колонок, имена файлов выгрузки (`config.yaml`).
7. Не заявлять в документах то, чего нет в коде.
8. Кириллица в XML/pptx — править через Python, не sed. Шаблон организаторов — обязателен для презентации.

Техника:
9. Фоновые процессы: `setsid nohup ... > log 2>&1 < /dev/null &`, python с `-u`; ждать через `sleep` ≤ 28 с.
10. `pydicom.dcmread(force=True)`; MONOCHROME1 инвертируется в `normalize_pixels`; PixelSpacing может
    отсутствовать — есть дефолты 1.05/0.6 мм.
11. Чекпоинты GPU — state_dict `features.*` torchvision efficientnet_b0, грузить `strict=False`.
12. Сборка образа запускает тест формата; если он падает — образ не собирается (это намеренно).
13. Bare-репозиторий отдаётся nginx через git-http-backend — после push обязателен `git update-server-info`.
14. Загрузка файлов в веб из облачного headless-браузера зависает (окружение тестера, не сервис).

---

## 12. Доступы и ключи

В публичной версии раздел удалён. Полный файл с доступами — у владельца и на сервере
`/opt/neuropeople/HANDOVER_FULL_2026-09-18.md` (не коммитить).
