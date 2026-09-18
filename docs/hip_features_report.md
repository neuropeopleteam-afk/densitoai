# Отчёт: переработка ML-признаков для критериев бедра (rh_pos, rh_roi, lh_pos, lh_roi)

Дата: 2026-09-18. Файлы: `src/hip_features.py` (новый), `src/hip_eval.py` (новый, диагностика),
`src/embeddings_hip_canonical.py` (новый, эксперимент), `src/geometry_features.py`,
`src/extract_all_features.py`, `src/train_stacked.py` (изменены). `inference.py`, `Dockerfile`,
`README.md`, `build_dataset.py`, `data/labels_full.csv` — НЕ трогались.

## Метрики ДО / ПОСЛЕ (OOF, repeated GroupKFold 5x5 по study, стэкинг геометрия+эмбеддинги)

ДО — `models/metrics_summary_BEFORE_hip_rework.json` (отдельные модели right_hip / left_hip, метки по
стороне из `labels_full.csv`). ПОСЛЕ — `models/metrics_summary.json`: одна объединённая модель `hip`
(геометрия в канонической ориентации), метрики по сторонам = срез OOF этой модели по стороне,
определённой по изображению; порог общий.

| Критерий | n_pos ДО → ПОСЛЕ | AUC geom ДО → ПОСЛЕ | AUC emb ДО → ПОСЛЕ | AUC stacked ДО → ПОСЛЕ | F1 ДО → ПОСЛЕ |
|---|---|---|---|---|---|
| rh_pos | 45 → 39 | 0.650 → **0.705** | 0.474 → 0.517 | 0.578 → **0.669** | 0.455 → 0.419 |
| rh_roi | 7 → 7 | 0.346 → **1.000** | 0.650 → 0.879 | 0.441 → **0.970** | 0.286 → 0.636 |
| lh_pos | 38 → 40 | 0.332 → **0.671** | 0.626 → 0.737 | 0.483 → **0.731** | 0.384 → 0.592 |
| lh_roi | 4 → 9 | 0.537 → 0.817 | 0.468 → 0.871 | 0.498 → 0.866 | 0.000 → 0.636 |
| hip_pos (merged) | – → 79 | – → 0.694 | – → 0.635 | – → 0.704 | – → 0.511 [CI 0.31–0.67] |
| hip_roi (merged) | – → 16 | – → 0.902 | – → 0.877 | – → 0.917 | – → 0.636 [CI 0.20–0.93] |

n_pos изменились, потому что сторона теперь определяется по изображению (см. ниже), а не по плотности.
Позвоночник не изменился: sp_pos 0.596, sp_axis geom 0.839 / stacked 0.738, sp_art emb 0.897 / stacked 0.823.

Честные оговорки:
* rh_roi (7 позитивов) и lh_roi (9 позитивов, из них пара дубликатов одного исследования и один
  аномально увеличенный снимок) — per-side метрики статистически ненадёжны (bootstrap CI F1 [0, 1]).
  Объединённый hip_roi с 16 позитивами — единственная осмысленная оценка ROI.
* hip_roi порог выбран F1-оптимизацией на OOF (правило >=15 позитивов в train_stacked.py, 16 — на границе).
* F1 rh_pos чуть снизился (0.455 → 0.419) при росте AUC: порог общий для обеих сторон.
* Разница rh_pos/lh_pos не физическая (геометрия канонизирована и распределения признаков у негативов
  обеих сторон совпадают), а следствие шума меток и малого n.

## Что изменено и почему (физика, не тюнинг)

### 1. Сторона бедра определяется по анатомии, а не по плотности (КРИТИЧНО для inference)
`build_dataset.hip_side_by_density` (доля ярких пикселей в половинах кадра) даёт неверную сторону для
80/333 (24%) снимков. Проверка без меток: у 63 из 64 исследований с чётным числом бедренных снимков
анатомическое правило даёт ровно поровну right/left (плотностное — только в 53/64). Правило:
таз всегда медиальнее диафиза, поэтому центроид кости в верхней трети кадра + доля тела у боковых
краёв определяют ориентацию (`hip_features.hip_side_score`; правое бедро = диафиз внизу-слева,
таз вверху-справа). DICOM-теги (PatientOrientation, Laterality) сторону не содержат.
Из-за ограничения scope `labels_full.csv`/`build_dataset.py` не менялись — в `geometry_features.csv`
добавлены колонки `hip_side_detected`, `hip_side_score`, `rh_pos_c, rh_roi_c, lh_pos_c, lh_roi_c,
hip_pos_c, hip_roi_c` (метка из разметка.xlsx для обнаруженной стороны).
**inference.py обязан использовать `hip_features.detect_hip_side`**, т.к. текст violation_type
содержит сторону («…правого/левого бедра»).

### 2. Сегментация кости для бедра (`segment_bone_hip`)
Старый Otsu по всему кадру: дыры в межвертельной области, отсутствие большого вертела, а на снимках
с ярким тазом медуллярный канал выпадал и трекинг «цеплялся» за одну кортикальную стенку 5–10 мм
(`docs/hip/track_fail.png`). Новое: Otsu только по пикселям тела, строчно-адаптивный порог
(отдельно верх 60% / низ 40%, линейный переход), 0.75·Otsu, заливка дыр (с нулевой рамкой —
без неё, если кость касается угла кадра, заливался весь фон), сохранение всех компонент ≥0.5% кадра
(`docs/hip/seg_compare.png`, `docs/hip/seg_v3.png`). Сегментация позвоночника не тронута.

### 3. Каноническая ориентация + трекинг диафиза
Левое бедро зеркалится → все признаки считаются как для правого. Диафиз трекается снизу построчно
(старт — самая широкая полоса в 6 нижних строках), край диафиза аппроксимируется двумя прямыми
итеративно с допуском 2.5 мм (`_fit_shaft`) — так верх диафиза определяется по началу отклонения
контура (вертелы), а не «нижними 40% маски вместе с тазом», как раньше (std угла 12°).

### 4. Признаки позиционирования/ротации (hip_pos)
Набор `femur_solidity, shaft_width_mm, abs_shaft_angle_deg, merge_height_mm, medial_neck_extent_mm`.
Физика наружной ротации: шейка укорачивается в проекции (`medial_neck_extent_mm` ↓, AUC 0.35 ≈ 0.65
инверт.; `merge_height_mm` — высота от верха диафиза до слияния с тазом ↓, 0.34), вырезы силуэта
между большим вертелом/головкой и под головкой заполняются → силуэт выпуклее (`femur_solidity` =
area/convex hull, AUC 0.71 — лучший одиночный признак), проксимальный диафиз шире в проекции
(`shaft_width_mm`, 0.68); отклонение оси диафиза от вертикали (`abs_shaft_angle_deg`, 0.63; в
комментарии разметчика «Отклонение оси» у обоих бёдер угол −4° и −11°). Выбор набора — по OOF
GroupKFold на объединённых данных (0.694) с проверкой, что он работает на обеих сторонах
(right 0.745 / left 0.712); более широкие наборы (+compactness, +signed angle, +вертелы) давали
0.68 merged, но 0.53–0.59 на левой стороне (переобучение). Профиль контура (40 сэмплов по 4 мм над
диафизом → PCA) даёт 0.59–0.61 и в модель не включён; колонки `prof_*` сохранены для диагностики.
Признаки, не давшие сигнала (≈0.5): signed angle, lesser_troch_prominence, greater_troch_offset,
lateral_margin, femur_eccentricity.

### 5. Признаки ROI (hip_roi)
Все ROI-позитивы — короткие сканы (высота 180–261 строк = 190–275 мм против 233–346 у негативов):
область сканирования обрезана и не включает достаточно диафиза ниже малого вертела.
`scan_length_mm` (AUC 0.92 инверт.) + `shaft_len_below_troch_mm` (0.87 инверт.). Старые
`edge_distance_ratio`/`bone_area_ratio` были ≈ случайными.

### 6. Эксперименты без выигрыша (задокументировано, не используется)
* Зеркалированные эмбеддинги EfficientNet-B0 (`data/embeddings_hip_canonical.npy`,
  `src/embeddings_hip_canonical.py`): OOF pos 0.54 vs 0.63 у оригинальных, roi 0.83 vs 0.89 →
  в стэкинге оставлены оригинальные эмбеддинги, зеркалится только геометрия.
* MTDDH (детские рентгенограммы таза): по 150 JSON (прочитаны на месте, без копирования) построено
  референсное распределение (compactness, eccentricity, solidity) подвздошной кости; Mahalanobis-расстояние
  нашей femur-маски до него даёт AUC 0.29 (≈0.71 инверт.) — ровно столько же, сколько собственная
  `femur_solidity`, при анатомически чужом референсе. OOF base+mahal 0.679 vs base 0.671 — в пределах шума.
  Полезный побочный результат — сами shape-признаки, они реализованы без внешнего датасета.
* Комментарии разметчиков: 2 исследования с эндопротезом (металл в маске насыщен) — кандидат на
  отдельное правило «артефакт/имплант», не входит в текущие критерии бедра.

## Артефакты для инспекции
`docs/hip/seg_compare.png`, `seg_v3.png`, `track_fail.png`, `side_fail2.png`, `commented_cases.png`,
`dbg_pos.png`, `dbg_neg.png`, `{right,left}_hip_{neg,pos,roi}.png`; OOF: `models/oof_stacked_hip_hip_pos.csv`,
`models/oof_stacked_hip_hip_roi.csv` (с колонкой `hip_side_detected`).

## Финальные модели (train_final_models.py) и проверка через inference.py

`python3 src/train_final_models.py` (после train_stacked.py) обучает StandardScaler+LR (geom, C=1) и
StandardScaler+PCA32+LR (emb, C=0.1), class_weight=balanced, на 100% валидных данных и сохраняет dict-pickle
по MODEL_CONTRACT.md (scaler, pca, clf, feature_cols, medians, oof_scores, mirror_right, sklearn_version).
Файлы: model_spine_{sp_pos,sp_axis,sp_art}_{geom,emb_pca}.pkl, model_spine_any_*.pkl,
model_hip_{pos,roi}_{geom,emb_pca}.pkl (единая модель, mirror_right=False), model_{right,left}_hip_any_*.pkl
(копии единой hip-any модели), models_manifest.json, per-side oof_stacked_{right,left}_hip_*.csv.

`mirror_right=False`: геометрия канонизируется внутри hip_features.hip_all_features (зеркалится ЛЕВОЕ),
эмбеддинги обучены на оригинальных снимках.

Проверка `python src/inference.py -i tests/sample_test_zip -o /tmp/check.csv --debug-csv -v`:
все 20 pickle загружены ("Model loaded: ..."), все `<crit>_method` = stacked_rank_avg, 0 failures,
tests/test_inference_format.py — ALL CHECKS PASSED.

РАСХОЖДЕНИЕ (в чужом inference.py, не исправлял):
1. `extract_geometry()` для бедра вызывает старый `geometry_features.hip_positioning_features` и отдаёт только
   shaft_angle_deg / edge_distance_ratio / bone_area_ratio. Признаки моих моделей (femur_solidity, shaft_width_mm,
   abs_shaft_angle_deg, merge_height_mm, medial_neck_extent_mm, scan_length_mm, shaft_len_below_troch_mm)
   в feats отсутствуют → все импутируются медианами → p_geom константа (в debug CSV rh_pos_p_geom = lh_pos_p_geom
   = 0.402 и rh_roi = lh_roi = 0.175 для двух РАЗНЫХ снимков; с реальными признаками было бы 0.538 / 0.361 и
   0.029 / 0.036). Контур B (эмбеддинги) работает корректно. Исправление: в `extract_geometry` для бедра
   `from hip_features import hip_all_features; feats.update(hip_all_features(img))`
   (или `geometry_features.extract_all_features(path, 'hip')`).
2. `classify_region` определяет сторону через `hip_side_by_density` (ошибка ~24%) — заменить на
   `hip_features.detect_hip_side(img_u8)`; сторона влияет только на ключ критерия (rh_/lh_) и текст.
