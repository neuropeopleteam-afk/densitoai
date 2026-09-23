# Стресс-набор устойчивости входа

Сгенерировано `tools/stress_set.py` 2026-09-23 13:30; окружение: pydicom 3.0.2, numpy 2.5.3, Python 3.12.13; декодер JPEG Baseline: есть (pillow).

Источник всех случаев — синтетические фантомы `tests/phantoms/` (кадры пациентов не используются). Пакет = копия 15 фантомов (с исходными путями) + подкаталог `stress/` со случаями ниже; обрабатывается одним прогоном `src/inference.py`. Правила: `failure_row` — строка Failure по `src/inference.py` (quality_class 0, violation_type пустой, quality_prob 0.5, причина в debug CSV); `same_class` — Success, регион, класс и тип нарушения равны строке исходного фантома в одиночном прогоне, |Δ quality_prob| ≤ 0.001; `success` — Success (кадр изменён, класс не сравнивается).

Прогон пакета: 39 строк на 39 файлов за 8.1 с (бюджет 300 с), код возврата 0. Случаев 24, ожидается Failure 9, Success 15; пройдено 24/24. Строки 15 фантомов в смешанном пакете побитово равны одиночному прогону (без time_of_processing), sha256 bb4e80840f67454e…; max |Δ quality_prob| по правилу same_class 0.0e+00.

| Случай | Описание | Источник | Ожидание | Правило | Статус | Регион | Класс | Тип нарушения | prob | Причина (debug) | Итог |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `truncated_half` | Файл обрезан до половины байтов (заголовок цел, пиксели неполные) | `study_01/CR000000.dcm` | Failure | `failure_row` | Failure | Поясничный отдел позвоночника | 0 | — | 0.5 | ValueError: The number of bytes of pixel data is less than expected (47015 vs 95100 bytes) | OK |
| `zero_bytes` | Файл нулевой длины с расширением .dcm | `study_01/CR000000.dcm` | Failure | `failure_row` | Failure | Проксимальный отдел бедра | 0 | — | 0.5 | ValueError: DICOM has no PixelData | OK |
| `not_dicom_ext` | Текст и случайные байты с расширением .dcm | `study_01/CR000000.dcm` | Failure | `failure_row` | Failure | Проксимальный отдел бедра | 0 | — | 0.5 | ValueError: DICOM has no PixelData | OK |
| `no_pixel_data` | DICOM без PixelData (только заголовок) | `study_01/CR000000.dcm` | Failure | `failure_row` | Failure | Поясничный отдел позвоночника | 0 | — | 0.5 | ValueError: DICOM has no PixelData | OK |
| `no_rows_cols` | DICOM без Rows и Columns при наличии PixelData | `study_01/CR000001.dcm` | Failure | `failure_row` | Failure | Проксимальный отдел бедра | 0 | — | 0.5 | AttributeError: Missing required element: (0028,0011) 'Columns' | OK |
| `no_pixel_spacing` | Без PixelSpacing и ImagerPixelSpacing (паспортная константа аппарата) | `study_03/CR000000.dcm` | Success | `same_class` | Success | Поясничный отдел позвоночника | 1 | Присутствуют посторонние предметы | 0.864132 | — | OK |
| `monochrome1` | PhotometricInterpretation MONOCHROME1 (инвертированные пиксели) | `study_04/CR000000.dcm` | Success | `same_class` | Success | Поясничный отдел позвоночника | 1 | Некорректная укладка | 0.936375 | — | OK |
| `signed16_negative` | 16 бит со знаком (PixelRepresentation 1), значения от −2048 | `study_03/CR000000.dcm` | Success | `same_class` | Success | Поясничный отдел позвоночника | 1 | Присутствуют посторонние предметы | 0.864132 | — | OK |
| `bits8_explicit_le` | 8 бит, Explicit VR Little Endian | `study_01/CR000002.dcm` | Success | `same_class` | Success | Проксимальный отдел бедра | 0 | — | 0.35745 | — | OK |
| `rgb` | RGB, SamplesPerPixel 3, три одинаковых канала | `study_03/CR000000.dcm` | Success | `same_class` | Success | Поясничный отдел позвоночника | 1 | Присутствуют посторонние предметы | 0.864132 | — | OK |
| `tiny_8x8` | Кадр 8×8 (меньше validation.min_rows/min_cols = 64) | `study_01/CR000000.dcm` | Failure | `failure_row` | Failure | Проксимальный отдел бедра | 0 | — | 0.5 | ValueError: image size out of range: 8x8 | OK |
| `huge_frame` | Кадр 4000×3000 (в пределах validation.max 4096), 8 бит | `study_01/CR000000.dcm` | Success | `success` | Success | Поясничный отдел позвоночника | 0 | — | 0.276355 | — | OK |
| `explicit_be` | Explicit VR Big Endian (retired) | `study_02/CR000001.dcm` | Success | `same_class` | Success | Проксимальный отдел бедра | 0 | — | 0.355788 | — | OK |
| `deflated_le` | Deflated Explicit VR Little Endian | `study_03/CR000000.dcm` | Success | `same_class` | Success | Поясничный отдел позвоночника | 1 | Присутствуют посторонние предметы | 0.864132 | — | OK |
| `rle_lossless` | RLE Lossless (кодировщик pydicom) | `study_04/CR000000.dcm` | Success | `same_class` | Success | Поясничный отдел позвоночника | 1 | Некорректная укладка | 0.936375 | — | OK |
| `jpeg_baseline` | JPEG Baseline: Success, если у pydicom есть декодер (Pillow); иначе Failure с причиной | `study_01/CR000000.dcm` | Success | `success` | Success | Поясничный отдел позвоночника | 0 | — | 0.328442 | — | OK |
| `bits12_of_16` | BitsAllocated 16 / BitsStored 12 / HighBit 11 | `study_03/CR000002.dcm` | Success | `same_class` | Success | Проксимальный отдел бедра | 0 | — | 0.355573 | — | OK |
| `modality_ct` | Чужая модальность (Modality CT, SOP Class CT Image Storage) | `study_03/CR000000.dcm` | Success | `same_class` | Success | Поясничный отдел позвоночника | 1 | Присутствуют посторонние предметы | 0.864132 | — | OK |
| `constant_frame` | Все пиксели одинаковые (постоянный кадр) | `study_01/CR000000.dcm` | Failure | `failure_row` | Failure | Поясничный отдел позвоночника | 0 | — | 0.5 | ValueError: constant (blank) image | OK |
| `zip_with_broken` | zip: исправный фантом рядом с обрезанным файлом | `study_01/CR000001.dcm` | Success | `same_class` | Success | Проксимальный отдел бедра | 0 | — | 0.357017 | — | OK |
| `zip_with_broken` | zip: обрезанный файл внутри архива | `study_01/CR000000.dcm` | Failure | `failure_row` | Failure | Поясничный отдел позвоночника | 0 | — | 0.5 | ValueError: The number of bytes of pixel data is less than expected (47015 vs 95100 bytes) | OK |
| `corrupt_zip` | Случайные байты с расширением .zip (не архив) | `study_01/CR000000.dcm` | Failure | `failure_row` | Failure | Проксимальный отдел бедра | 0 | — | 0.5 | ValueError: Архив повреждён или не является zip-файлом; пересоздайте архив и загрузите сно | OK |
| `nested_depth5` | Вложенность каталогов глубиной 5 | `study_04/CR000000.dcm` | Success | `same_class` | Success | Поясничный отдел позвоночника | 1 | Некорректная укладка | 0.936375 | — | OK |
| `cyrillic_spaces` | Кириллица, пробелы и скобки в именах | `study_01/CR000002.dcm` | Success | `same_class` | Success | Проксимальный отдел бедра | 0 | — | 0.35745 | — | OK |

Примечания.

- Причина сбоя пишется в `results_debug.csv` (колонка `error`), в основной CSV — только `Failure`; `study_uid`/`image_uid` строки Failure берутся из тегов, если заголовок читается, иначе — `hash-<sha>` от содержимого файла и папки.
- Файлы фантомов не содержат PixelSpacing, поэтому случай `no_pixel_spacing` дополнительно удаляет и ImagerPixelSpacing; сервис подставляет паспортный размер пикселя 1,05 × 0,6 мм.
- `jpeg_baseline`: кодировщика JPEG у pydicom нет, поток JPEG даёт Pillow (входит в requirements.txt). Ожидание зависит от наличия декодера в окружении и вычисляется при запуске; в `expected_stress.csv` записано ожидание для образа (Pillow есть → Success).
- Огромный кадр строится повтором отсчётов фантома без интерполяции; его класс не сравнивается с исходным (геометрия в пикселях меняется), проверяется только контролируемый Success.
- Набор не доказывает качество на реальных снимках и не покрывает сжатия с потерями иных кодеков (JPEG 2000, JPEG-LS: см. `docs/TRANSFER_SYNTAX_MATRIX.md`).
