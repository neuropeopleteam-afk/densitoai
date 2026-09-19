## 2.1.1-dev (19.09.2026)
- К3: правило порога по критерию выбрано nested (`config.yaml: thresholds_rule`): sp_art 0.557→0.602, hip_pos 0.709→0.643, hip_roi 0.891→0.918 (sp_pos, sp_axis без изменений); на 499 файлах изменился класс 39 строк (28: 0→1, 11: 1→0). Зона «не уверен» (`needs_review`, `risk_level`, `<crit>_margin`) и Platt-калибровка критериев (`<crit>_p_cal`, `models/calibration.pkl`) — только debug-CSV и API. Калибровка quality_prob не принята (Brier без значимого прироста).
- К7: карточка решения «измерено против нормы», одна причина, подсказка рекомендуемого положения ROI (пунктир, не автокоррекция), режим лаборанта, блок устойчивости, генератор офлайн-casebook (`tools/web/build_casebook.py`).
- К5: `build_dataset.py` определяет сторону бедра анатомическим детектором (плотностная эвристика ошибалась на 80/333); модели не менялись.
- `--sr-study` / `--sr-study-dir`: один DICOM Comprehensive SR на исследование (включая норму), хэши оригинала (sha256 файла и пикселей) в SR и debug-CSV; API: `study_sr`, `study_sr_download`. `tools/validate_sr.py`.
- JSON Schema результата (`schema/`), проверка в `validate_output_csv`, `tests/test_schema.py`; `src/schema_check.py`.
- `tools/make_model_card.py` → `models/MODEL_CARD.md`; `docs/EVIDENCE.md`, `docs/DZM_CONFORMANCE.md`.
- Вентильный стэкинг: `stacking.weights_by_criterion` в config (пусто — поведение 0.5/0.5 без изменений), `tools/nested_gate.py`, `models/nested_gate_decisions.json`, `docs/NESTED_GATE_REPORT.md`. Предсказания на 499 файлах не изменились.
- `--extras` / `src/extras.py` / `models/ood_gate.pkl`: белые линии (флаг), OOD-gate (fingerprint + Mahalanobis, FPR 1 % OOF, 130/130 внешних), «эндопротез», когерентность исследования → `results_extras.csv`, API `details.extras`.
- `tools/verify.sh`, фантомы, digest-pinned образ, verify при сборке; исправлены DICOM без file meta и UID-заглушки, зависящие от пути.
- `tools/review/`: инструмент слепой ревизии для рентгенолога.

# История изменений

## 2.1.0 — 2026-09-18
- `quality_prob` согласована с `quality_class` (смесь any-модели и максимума по критериям,
  монотонное приведение: класс 1 ⇔ prob ≥ 0.5). 79/499 противоречивых строк → 0;
  OOF ROC-AUC бинарной задачи 0.735/0.704 → 0.775/0.773 (v2.1.0); после гибридного бэкбона sp_pos — 0.783/0.773 (позвоночник/бедро).
- Контентный детектор региона для нестандартной ширины кадра (`models/model_region_emb.pkl`).
- Панель визуализации показывает измерения и оценки моделей «скор / порог» по каждому
  критерию (без скрытых эвристических порогов).
- Веб-интерфейс (`web/index.html`): оверлей + ROI-коррекция, DICOM SR, CSV результата.
- `src/eval_oof_metrics.py` — отчёт по метрикам ТЗ §8.4 с 95 % ДИ; `docs/METRICS_REPORT.md`.
- Ответы организаторов (транскрипт Q&A и письменные разъяснения) и их учёт — `docs/qa/`.

## 2.0.2
- Единая модель бедра для обеих сторон (анатомическое определение стороны), any-модели,
  DICOM SR, автокоррекция ROI, визуализация, API `/api/analyze`, `/api/batch`.

## 2.0.0
- Двухконтурная архитектура (геометрия + эмбеддинги EfficientNet-B0), ранг-стэкинг,
  контейнер, формат ТЗ п.2.5, smoke-тест формата.
