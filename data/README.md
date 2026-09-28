# data/ — обучающие таблицы (499 файлов заказчика, 100 исследований)

Здесь лежат производные таблицы обучающего набора. Сервис при инференсе метки отсюда не читает: из
`geometry_features.csv` берутся только медианы признаков (`tools/make_geometry_medians.py`, `src/inference.py`).

## Внимание: в `geometry_features.csv` есть колонки меток

Помимо признаков в `geometry_features.csv` лежат метки разметки и производные от них колонки:
`sp_pos, sp_axis, sp_art, rh_pos, rh_roi, lh_pos, lh_roi, quality_class, applicable, violation_list,
rh_pos_c, rh_roi_c, lh_pos_c, lh_roi_c, hip_pos_c, hip_roi_c`, а также служебные `instance_number, rows, cols`.
Если брать «все числовые колонки» как признаки, модель получает ответ во входе, и любая метрика становится
бессмысленной. Правило: признаки — только списки `CRITERION_GEOMETRY_COLS` / `CRITERION_EXTRA_COLS` из
`src/train_stacked.py` или фильтр `tools/clone_trap.py:feature_columns` (исключает колонки меток, `*_c`, `*class*`,
`*violation*`, `*applicable*` и служебные).

Вынести метки в отдельный файл сейчас нельзя без правки потребителей: их читают из этого файла
`src/train_stacked.py` (мишени `hip_pos_c`, `hip_roi_c`), `tools/nested_gate.py`, `tools/sample_passport.py`,
`tools/web/build_casebook.py`, `tools/web/build_actions_json.py` (`quality_class`), `tools/extras/white_lines_audit.py`,
`src/eval_oof_metrics.py`. Поэтому колонки остаются на месте, а это предупреждение — вместо переноса.

## Клоны кадров

Уникальных кадров (хэш пикселей) 252 из 499: экспорт кладёт в исследование побайтные копии снимка. Любое разбиение
для валидации — только по группам «исследование + хэш пикселей» (`docs/k5/pixel_hashes.csv`). Цена ошибки показана
в `docs/clone_trap.md` (`python tools/clone_trap.py`): одна и та же модель на тех же признаках при случайном
разбиении по файлам получает ROC-AUC в среднем на 0.09–0.38 выше, чем при групповом (таблица признаков 2.5.0).

## Сторона бедра

- `labels_full.csv` — колонка `region` (`right_hip` / `left_hip`) с 24.09 (A2) перегенерирована анатомическим
  детектором (`src/build_dataset.hip_side_anatomical` → `hip_features.detect_hip_side`; `tools/regen_labels_full.py`)
  и совпадает с `hip_side_detected` в `geometry_features.csv` и OOF (`tests/test_organizer_metrics.py`). Порядок строк
  и `file_path` не менялись (порядок совпадает с `embeddings*.npy`).
- `labels_full_v1_density_side.csv` — прежняя версия со стороной по плотностной эвристике (расходится на 80 из 333
  кадров бедра); нужна только `tools/side_label_sensitivity.py` (вариант B) и `src/hip_eval.py` (`pos_old`).
- Колонка `region` в `geometry_features.csv` осталась от прежней версии (плотностная сторона); для стороны там
  используйте `hip_side_detected`.
