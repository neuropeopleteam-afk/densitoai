# Контракт на файлы моделей (`models/*.pkl`) — что ожидает `src/inference.py`

Этот документ — для того, кто сохраняет обученные модели (ML-агент / `train_stacked.py`).
Инференс **терпим**: любой отсутствующий или нечитаемый файл → `WARNING` в лог и
fallback-правило по геометрии, без падения.

## Имена файлов (в `models/`)

| Регион (`region`) | Критерий (`crit`) | Контур A (геометрия)              | Контур B (эмбеддинги + PCA)          |
|-------------------|-------------------|-----------------------------------|--------------------------------------|
| `spine`           | `sp_pos`, `sp_axis`, `sp_art` | `model_spine_<crit>_geom.pkl` | `model_spine_<crit>_emb_pca.pkl` |
| `right_hip`       | `rh_pos`, `rh_roi` | `model_right_hip_<crit>_geom.pkl` | `model_right_hip_<crit>_emb_pca.pkl` |
| `left_hip`        | `lh_pos`, `lh_roi` | `model_left_hip_<crit>_geom.pkl`  | `model_left_hip_<crit>_emb_pca.pkl`  |

Допустимые альтернативы (ищутся, если основного имени нет): для бедра —
`model_hip_pos_*.pkl`, `model_hip_roi_*.pkl` (единая модель на обе стороны; для правого
бедра изображение зеркалируется, если в pickle есть `mirror_right: True`).
Опционально: `model_<region>_any_geom.pkl` / `model_<region>_any_emb_pca.pkl` — отдельная модель
«есть хоть одно нарушение» для `quality_prob`; если её нет, `quality_prob = max` по критериям.

## Формат содержимого pickle (любой из двух)

**Вариант 1 — объект с `predict_proba`** (например, `sklearn.pipeline.Pipeline`
`StandardScaler → [PCA] → LogisticRegression`). Вход: `X.shape == (1, n_features)`.

**Вариант 2 — dict** (предпочтительно, даёт больше метаданных):

```python
{
  "scaler":       StandardScaler,        # обязательно, если использовался при обучении
  "pca":          PCA | None,            # только для emb_pca
  "clf":          estimator с predict_proba (или ключ "model"),
  "feature_cols": ["axis_angle_deg", ...],   # ТОЛЬКО для geom: порядок признаков; если нет —
                                              #   берётся config.yaml: geometry_cols[<crit>]
  "medians":      {"axis_angle_deg": 1.18, ...},  # для импутации NaN (иначе медианы из data/geometry_features.csv)
  "oof_scores":   np.ndarray,            # OOF-предсказания на трейне -> референс для ранг-стэкинга
                                          #   (иначе читается models/oof_stacked_<region>_<crit>.csv, колонки oof_geom / oof_emb)
  "mirror_right": bool,                  # единая модель бедра: правое зеркалируется перед контуром B
  "threshold":    float,                 # порог на стэкнутом скоре (опционально)
  "sklearn_version": "1.9.1",            # информативно
}
```

### Признаки контура A (порядок как в `config.yaml: geometry_cols`)
- `sp_pos`: `center_offset_ratio, bone_width_ratio`
- `sp_axis`: `axis_angle_deg`
- `sp_art`: `metal_metal_area_mm2, metal_metal_max_intensity_gap`
- `rh_pos` / `lh_pos`: `shaft_angle_deg`
- `rh_roi` / `lh_roi`: `edge_distance_ratio, bone_area_ratio`

Признаки вычисляются `geometry_features.py` из uint8-изображения (нормализация как в
`read_dicom_normalized`). Если при обучении использовались **другие** признаки — положите их
список в `feature_cols`, инференс возьмёт значения по имени из словаря признаков
(`extract_geometry`), недостающие → медиана.

### Вход контура B
Вектор 1280-d от `embeddings.FrozenBackbone` (EfficientNet-B0, ImageNet, resize 320×192,
`float32`). Веса backbone лежат в `models/torch_home/` (offline).

## Пороги
`inference.py` берёт порог по критерию: `config.yaml: thresholds[<crit>]` → ключ `"threshold"`
внутри dict-pickle (geom, затем emb) → `models/metrics_summary.json` (`<region>.<crit>.threshold`,
для единой модели бедра также `hip.hip_pos` / `hip.hip_roi` / `hip.pos`) → 0.5 с предупреждением.
OOF-референс для единой модели бедра: `oof_stacked_hip_pos.csv` / `oof_stacked_hip_roi.csv`
(или `oof_scores` внутри pickle). Порог применяется к
**стэкнутому** скору `0.5*rank_geom + 0.5*rank_emb`, где `rank_*` — доля референсных OOF-скоров
≤ текущего (эквивалент `rank(pct=True)` при обучении). Если референс недоступен —
используется сырая вероятность. Если обучаете иначе — обновите `metrics_summary.json`
или пропишите пороги в `config.yaml`.

## Как проверить, что модели подхватились
```bash
python src/inference.py -i tests/sample_test_zip -o outputs/check.csv --debug-csv -v
# в логе: "Model loaded: model_spine_sp_axis_geom.pkl" ...; в outputs/check_debug.csv
# колонки <crit>_method должны быть "stacked_rank_avg" (а не "fallback_rule")
python tests/test_inference_format.py    # формат и устойчивость
```
Проверено на синтетических pickle обоих форматов (dict и Pipeline), а также на битом pickle
(игнорируется с предупреждением).

## Сторона бедра
Сторона определяется в инференсе `hip_features.detect_hip_side` (анатомическое правило), при
сбое — плотностная эвристика. Проверено: на 3 файлах организаторов совпадает с суффиксами
`_ППОБ/_ЛПОБ` без подсказок из имени. Единая модель `model_hip_*` загружается для обоих
внутренних регионов (`right_hip`, `left_hip`) — проверено на синтетических pickle.
