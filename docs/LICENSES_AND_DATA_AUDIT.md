# Аудит лицензий и данных (DensitoAI 2.1.0)

Дата аудита: 19.09.2026. Метод: `importlib.metadata` по каждой строке `requirements.txt` в тестовой среде
(Python 3.14.3, pip-пакеты тех же версий, кроме отмеченных), поля License-Expression / License / Classifier.
В образе Docker ставятся ровно версии из `requirements.txt`; расхождения версий среды отмечены в столбце «установлено».

## 1. Python-пакеты (`requirements.txt`)

| Пакет | requirements.txt | установлено в стенде | Лицензия | Источник |
|---|---|---|---|---|
| torch | 2.14.0+cpu | 2.14.0+cpu | BSD-3-Clause (метаданные: Apache-2.0 AND BSD-2/3 AND MIT AND ... для сторонних компонентов) | https://pytorch.org |
| torchvision | 0.29.0+cpu | 0.29.0+cpu | BSD | https://github.com/pytorch/vision |
| numpy | 2.5.3 | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | https://numpy.org |
| scipy | 1.18.1 | 1.18.1 | BSD License | https://scipy.org/ |
| scikit-learn | 1.9.1 | 1.9.1 | BSD-3-Clause | https://scikit-learn.org |
| joblib | 1.6.0 | 1.6.0 | BSD-3-Clause | https://joblib.readthedocs.io |
| threadpoolctl | 3.7.0 | 3.7.0 | BSD-3-Clause | https://github.com/joblib/threadpoolctl |
| pandas | 3.0.5 | 3.0.6 | BSD License | https://pandas.pydata.org |
| python-dateutil | 2.9.0.post0 | 2.9.0.post0 | Apache-2.0 OR BSD-3-Clause (Dual License) | https://github.com/dateutil/dateutil |
| pytz | 2026.3.post1 | 2026.3.post1 | MIT | http://pythonhosted.org/pytz |
| pydicom | 3.0.2 | 3.0.2 | MIT License | https://pydicom.github.io/pydicom |
| opencv-python-headless | 5.0.0.93 | 5.0.0.93 | Apache 2.0 | https://github.com/opencv/opencv-python |
| pillow | 12.3.0 | 12.3.0 | MIT-CMU | https://pillow.readthedocs.io/en/stable/releasenotes/index.html |
| scikit-image | 0.26.0 | 0.26.0 | BSD License | https://scikit-image.org |
| imageio | 2.37.4 | 2.37.4 | BSD-2-Clause | https://github.com/imageio/imageio |
| tifffile | 2026.9.15 | 2026.9.15 | BSD-3-Clause | https://www.cgohlke.com |
| networkx | 3.6.1 | 3.6.1 | BSD-3-Clause | https://networkx.org/ |
| lazy-loader | 0.5 | 0.5 | BSD-3-Clause | https://scientific-python.org/specs/spec-0001/ |
| packaging | 26.3 | 26.3 | Apache-2.0 OR BSD-2-Clause | https://packaging.pypa.io/ |
| PyYAML | 6.0.3 | 6.0.3 | MIT | https://pyyaml.org/ |
| openpyxl | 3.1.5 | 3.1.5 | MIT | https://openpyxl.readthedocs.io |
| et_xmlfile | 2.0.0 | 2.0.0 | MIT | https://foss.heptapod.net/openpyxl/et_xmlfile |
| fastapi | 0.141.1 | 0.141.1 | MIT | https://github.com/fastapi/fastapi |
| starlette | 1.6.0 | 1.6.0 | BSD-3-Clause | https://github.com/Kludex/starlette |
| pydantic | 2.12.5 | 2.12.5 | MIT | https://github.com/pydantic/pydantic |
| pydantic-core | 2.41.5 | 2.41.5 | MIT | https://github.com/pydantic/pydantic-core |
| annotated-types | 0.8.0 | 0.8.0 | MIT | https://github.com/annotated-types/annotated-types |
| typing_extensions | 4.16.0 | 4.16.0 | PSF-2.0 | https://github.com/python/typing_extensions/issues |
| anyio | 4.15.1 | 4.15.1 | MIT | https://anyio.readthedocs.io/en/latest/ |
| sniffio | 1.3.1 | 1.3.1 | MIT OR Apache-2.0 | https://github.com/python-trio/sniffio |
| idna | 3.20 | 3.20 | BSD-3-Clause | https://github.com/kjd/idna/blob/master/HISTORY.md |
| uvicorn | 0.53.0 | 0.53.0 | BSD-3-Clause | https://uvicorn.dev/release-notes |
| h11 | 0.16.0 | 0.16.0 | MIT | https://github.com/python-hyper/h11 |
| click | 8.5.0 | 8.5.0 | BSD-3-Clause | https://click.palletsprojects.com/page/changes/ |
| python-multipart | 0.0.32 | 0.0.32 | Apache-2.0 | https://github.com/Kludex/python-multipart |
| filelock | 4.0.0 | 3.32.3 | MIT | https://py-filelock.readthedocs.io |
| fsspec | 2026.7.0 | 2026.7.0 | BSD-3-Clause | https://filesystem-spec.readthedocs.io/en/latest/changelog.html |
| sympy | 1.14.0 | 1.14.0 | BSD | https://sympy.org |
| mpmath | 1.3.0 | 1.3.0 | BSD | http://mpmath.org/ |
| Jinja2 | 3.1.6 | 3.1.6 | BSD License | https://jinja.palletsprojects.com/changes/ |
| MarkupSafe | 3.0.3 | 3.0.3 | BSD-3-Clause | https://palletsprojects.com/donate |

Итог: все 41 пакет — пермиссивные лицензии (BSD, MIT, Apache-2.0, PSF, MIT-CMU). Копилефта нет.
Расхождения стенда с pin-версиями (в образе будут pin-версии): pandas 3.0.6 вместо 3.0.5, filelock 3.32.3 вместо 4.0.0.

Опциональный блок кодеков (закомментирован в `requirements.txt`, в образ по умолчанию не входит):
`pylibjpeg` 2.1.0 (MIT), `pylibjpeg-openjpeg` 2.5.0 (MIT), `pylibjpeg-libjpeg` 2.4.0 — **GPL v3**.
Если включать JPEG Lossless/JPEG-LS, то `pylibjpeg-libjpeg` подпадает под GPL; для JPEG 2000 достаточно
`pylibjpeg` + `pylibjpeg-openjpeg` (оба MIT). Решение — за владельцем продукта.

Веб-ресурсы в образе (добавлено в 2.4.1): swagger-ui (`swagger-ui-bundle.js`, `swagger-ui.css`) лежит в
`web/assets/swagger/`, чтобы Swagger `/docs` открывался без интернета (FastAPI по умолчанию берёт эти файлы
с внешнего CDN). Лицензия swagger-ui — Apache 2.0 (https://github.com/swagger-api/swagger-ui), пермиссивная, копилефта нет;
точная версия — в заголовке файлов.

## 2. Внешние наборы данных и где они использованы

В репозитории и в образе внешних изображений нет (см. п. 3). Внешние наборы использовались только
на GPU-стенде для предобучения бэкбона `models/backbone_densito.pth` прокси-задачами
(`gpu/prepare_cache.py`, словарь SOURCES) и для экспериментов; итоговые классификаторы `models/*.pkl`
обучены только на разметке собственных снимков (`data/geometry_features.csv`, 499 файлов).

| Набор | Лицензия / условия (по странице набора) | Использование в проекте |
|---|---|---|
| BUU-LSPINE (Burapha Univ.) | собственное EULA на странице набора (services.informatics.buu.ac.th/spine): некоммерческое исследовательское использование, без передачи | предобучение бэкбона (`buu_lspine`) |
| MTDDH (ScienceDB / figshare) | CC BY 4.0 | предобучение бэкбона (`mtddh_pelvis`); один аннотированный кадр в `docs/mtddh_sample_annotated.png` |
| FracAtlas (figshare / Sci. Data 2023) | CC BY 4.0 | предобучение бэкбона (`fracatlas`) |
| Arak Bone Densitometry Center (Kaggle-зеркало, medRxiv 2025) | CC BY-NC (некоммерческое) | предобучение бэкбона (`arak_hip_dxa`) |
| DEXA-Osteo (Kaggle) | лицензия на странице Kaggle не подтверждена в этом аудите (доступ требует аккаунт) | предобучение бэкбона (`dexa_osteo_spine`) |
| AASCE MICCAI 2019 (SpineWeb, Kaggle-зеркало) | условия челленджа: исследовательское использование | предобучение бэкбона (`aasce_spine`) |
| Chinese Osteoporosis (Kaggle, Hip / LumbarP) | по PLAN.md — CC BY 4.0; страница набора в аудите не подтверждена | предобучение бэкбона (`cn_hip`, `cn_lumbar_ap`) |

**Требует решения владельца (уточнено 22.09.2026).** Производные работы от данных с ограничением NC (Arak,
CC BY-NC) и EULA (BUU-LSPINE) — это **два** бэкбона: `backbone_densito.pth` и `backbone_densito_inv.pth`
(инвариантный вариант того же пула; см. `docs/EMB_GATE_REPORT.md`). Оба используются в поставках 2.3.0 и 2.4.0:

| Критерий | Источник эмбеддингов (`models/metrics_summary.json` → `emb_source`) | Лицензионный статус источника |
|---|---|---|
| `sp_pos` | `densito` (`models/backbone_densito.pth`) | производная от пула с Arak (CC BY-NC) и BUU-LSPINE (EULA) — некоммерческое |
| `sp_axis` | `densito_inv` (`models/backbone_densito_inv.pth`) | то же ограничение (тот же пул) |
| `sp_art` | `imagenet` (torchvision EfficientNet-B0) | BSD-3-Clause, ограничений нет |
| `hip_pos` | `imagenet` | BSD-3-Clause, ограничений нет |
| `hip_roi` | `imagenet` | BSD-3-Clause, ограничений нет |

Итого ограничение NC/EULA затрагивает два критерия позвоночника из пяти (`sp_pos`, `sp_axis`); три критерия
уже свободны. Варианты для коммерческой поставки: (а) переобучить бэкбон только на CC BY / собственных данных;
(б) перевести `sp_pos` и `sp_axis` на `imagenet` либо на свободный по лицензиям `densito_inv_free`
(`models/backbone_densito_inv_free.pth`, обучен без Arak и BUU-LSPINE) — на `sp_axis` он правило приёмки
не проходит (ΔAUC +0.025 при 10/20 повторах против требуемых 14/20, `docs/EMB_GATE_REPORT.md`), то есть
цена свободы по лицензиям измерена и невелика; (в) получить разрешения правообладателей.
Для конкурсной поставки (исследовательское использование, безвозмездно) текущая конфигурация допустима;
для коммерческого внедрения решение владельца обязательно до передачи заказчику.

Единая формулировка для карточки модели, паспорта и ответов жюри (24.09.2026): Исследовательский прототип: бэкбоны `sp_pos` (`densito`) и `sp_axis` (`densito_inv`) дообучены на пуле с некоммерческими лицензиями (Arak — CC BY-NC, BUU-LSPINE — EULA). Для передачи заказчику — вариант `densito_inv_free` (`models/backbone_densito_inv_free.pth`, без этих наборов), цена измерена на nested-протоколе: `sp_axis` AUC стэка 0.860 → 0.773 (`docs/EMB_GATE_REPORT.md`); для `sp_pos` `densito_inv_free` не измерен, замена на `imagenet` стоила 0.759 → 0.672 (К13, до H2).

Важно: ни один внешний набор не входит в поставляемый образ — в нём только веса
(`models/*.pth`, sha256 в `models/WEIGHTS_SHA256.txt`); изображения внешних наборов в репозиторий не кладутся.

## 3. Изображения и DICOM в репозитории (`find densito_rebuild -name '*.png' -o -name '*.dcm' -o -name '*.jpg'`)

| Файлы | Кол-во | Происхождение | Статус |
|---|---|---|---|
| `tests/sample_test_zip/Для теста/*.dcm`, дубликат `data/sample_test_zip/Для теста/*.dcm` — **удалены 26.09**, в публичном GitHub их нет и в истории | 3 + 3 | образец организаторов «Для теста.zip» (публично выдан участникам), PatientName/ID = Anonymized | ПДн не выявлены (п. 4). В релиз-архив по умолчанию не входит (`WITH_SAMPLE=1` включает); дубликат в `data/` рекомендуется удалить |
| `docs/art_sample_0..5.png`, `docs/sample_art_neg/pos.png`, `docs/sample_axis_ok/violation.png` | 10 | визуализации собственных снимков (данные организаторов, 499 DICOM) с наложенной разметкой | **требует решения: обезличенные визуализации собственных снимков** — оставить только с согласия организаторов или заменить фантомами |
| `docs/hip/*.png` (seg_v3, right/left_hip_*, masks, side_fail*, track_fail, dbg_*, seg_compare, commented_cases) | 16 | то же: отладочные визуализации собственных снимков бедра | **требует решения** (то же) |
| `docs/mtddh_sample_annotated.png` | 1 | кадр внешнего набора MTDDH (CC BY 4.0) с аннотацией | внешнее изображение в репозитории; допустимо с атрибуцией, по правилу проекта — вынести из репозитория или добавить атрибуцию |
| `tests/phantoms/**/*.dcm` (после патча) | 15 | синтетика `tools/make_phantoms.py`, PatientName PHANTOM^SYNTHETIC | без ограничений |
| `*.jpg` | 0 | — | — |

Конкурсных DICOM (499 файлов организаторов) в репозитории нет: `data/geometry_features.csv` и `models/oof_*.csv`
содержат только признаки и пути к файлам на рабочей машине (столбец `file_path`, без ПДн); `data/labels*.csv`,
`data/*.npy` исключаются из релиз-архива (`tools/make_release.sh`). Рекомендация: удалить столбец `file_path`
из `data/geometry_features.csv` перед публикацией (инференс использует только медианы признаков).

Всего PNG в docs: 27 файлов, 9,75 МиБ.

## 4. Проверка ПДн в DICOM (`tools/pii_scan.py`)

Запуск: `python tools/pii_scan.py tests data --md outputs/pii_report.md --strict` (19.09.2026, стенд).
Проверено 21 DICOM (3 `tests/sample_test_zip`, 3 `data/sample_test_zip`, 15 `tests/phantoms`): с выявленными ПДн — 0,
приватных тегов — 0, у всех `PatientIdentityRemoved = YES`. В образце организаторов PatientName/PatientID = `Anonymized`,
даты рождения/учреждение/врачи отсутствуют. Первая версия фантомов имела StudyID `PH01` — сканер пометил как
неочищенный тег, StudyID заменён на `PHANTOM01..04` и фантомы перегенерированы. Полный протокол: `work/A/scratch/pii_report.md`.
Ограничение: пиксели на вшитый текст сканер не анализирует (в экспорте GE Lunar таких аннотаций не наблюдалось).
