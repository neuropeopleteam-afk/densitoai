# Матрица transfer syntax и вариантов кодирования DICOM

Сгенерировано `tests/test_transfer_syntax.py` 2026-09-20 15:03; окружение: pydicom 3.0.2, numpy 2.5.3, Python 3.14.3; кодеки: pylibjpeg —, libjpeg —, openjpeg —, jpeg_ls —, gdcm —.

Источник: 3 файла из `tests/sample_test_zip` (CR000000_ПОП.dcm, CR000000_ППОБ.dcm, CR000001_ЛПОБ.dcm). Критерий «OK»: processing_status = Success, anatomical_region и quality_class совпадают с оригиналом (Implicit VR LE, 8 бит, MONOCHROME2, без PixelSpacing). Δprob — максимальное по файлам |Δ quality_prob|.

Инференс всех вариантов одним прогоном: 6.3 с.

| Вариант | Описание | Записан | Success | Регион | Класс | Тип нарушения | UID | max Δprob | Итог |
|---|---|---|---|---|---|---|---|---|---|
| `implicit_le` | Implicit VR Little Endian (1.2.840.10008.1.2) — как в оригинале | записан | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0.0e+00 | OK |
| `explicit_le` | Explicit VR Little Endian (1.2.840.10008.1.2.1) | записан | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0.0e+00 | OK |
| `explicit_be` | Explicit VR Big Endian (1.2.840.10008.1.2.2, retired) | записан | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0.0e+00 | OK |
| `rle` | RLE Lossless (1.2.840.10008.1.2.5) | записан | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0.0e+00 | OK |
| `jpeg2000_lossless` | JPEG 2000 Lossless (1.2.840.10008.1.2.4.90) | не проверено: RuntimeError: The pixel data encoder for 'JPEG 2000 Image Compression (Lossless Only)' is unavailable because all of its plu | — | — | — | — | — | — | не проверено |
| `jpegls_lossless` | JPEG-LS Lossless (1.2.840.10008.1.2.4.80) | не проверено: RuntimeError: The pixel data encoder for 'JPEG-LS Lossless Image Compression' is unavailable because all of its plugins are  | — | — | — | — | — | — | не проверено |
| `jpeg_lossless_sv1` | JPEG Lossless SV1 (1.2.840.10008.1.2.4.70) | не проверено: NotImplementedError: No pixel data encoders have been implemented for 'JPEG Lossless, Non-Hierarchical, First-Order Prediction (Pro | — | — | — | — | — | — | не проверено |
| `bits16` | 16 бит (BitsAllocated 16 / BitsStored 12), Explicit VR LE | записан | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0.0e+00 | OK |
| `bits16_rescale` | 16 бит + RescaleSlope 0.5 / RescaleIntercept −100 | записан | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0.0e+00 | OK |
| `monochrome1` | MONOCHROME1 (инвертированные пиксели) | записан | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0.0e+00 | OK |
| `pixel_spacing` | С тегом PixelSpacing 1.05/0.6 (в оригинале тега нет) | записан | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0.0e+00 | OK |
| `no_pixel_spacing` | Без PixelSpacing и ImagerPixelSpacing | записан | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0.0e+00 | OK |
| `raw_no_meta` | Без преамбулы и file meta header (raw Implicit VR LE) | записан; pydicom.pixel_array: AttributeError | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0.0e+00 | OK |
| `no_uids` | Без StudyInstanceUID и SOPInstanceUID (ожидается hash-UID, Success) | записан | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 0.0e+00 | OK |

Итог: 11 из 11 проверенных вариантов без расхождений; 3 вариантов не удалось записать текущими кодеками (статус «не проверено»).

Примечания.

- Для `no_uids` UID ожидаемо не совпадают: инференс подставляет `hash-<sha>` от пути; проверяется только Success/регион/класс.
- Варианты с потерями (JPEG Baseline) не включены: изменение пикселей — не вопрос совместимости формата.
- Кодеки для сжатых синтаксисов (pylibjpeg, pylibjpeg-libjpeg, pylibjpeg-openjpeg) нужны только для чтения сжатых DICOM; экспорт GE Lunar Prodigy — несжатый Implicit VR LE. В `requirements.txt` они вынесены отдельным блоком.
