# Карточка модели DensitoAI v2.3.0

Сформировано автоматически `tools/make_model_card.py` 2026-09-20 из `models/metrics_summary.json`, `models/models_manifest.json`, `config.yaml`, `requirements.txt`. Ручные правки не вносить — перегенерировать.

## 1. Назначение

Автоматическая оценка качества денситометрических снимков (DXA, GE Lunar Prodigy) двух областей: «Поясничный отдел позвоночника» и «Проксимальный отдел бедра». Для каждого снимка выдаётся quality_class (0 — норма, 1 — есть нарушение), закрытый список нарушений и quality_prob. Инструмент поддержки контроля качества укладки; не является медицинским изделием и не ставит диагноз. Решение принимает оператор/врач.

Официальные строки нарушений (config.yaml → violations): Не выравнена ось позвоночника; Некорректная область интереса; Некорректная укладка; Присутствуют посторонние предметы.

## 2. Архитектура

Двухконтурный стекинг для каждого критерия (см. `src/inference.py`, `models/MODEL_CONTRACT.md`):

- контур A — геометрические признаки сегментированной кости (логистическая регрессия, `model_<регион>_<критерий>_geom.pkl`);
- контур B — эмбеддинги замороженного EfficientNet-B0 (источник `imagenet` или `densito` — дообученный на внешних DXA-наборах бэкбон `models/backbone_densito.pth`) → PCA → логистическая регрессия (`model_..._emb_pca.pkl`);
- объединение: ранговое усреднение перцентилей относительно OOF-распределения с весами geom=0.5, emb=0.5; quality_prob = 0.5·any-модель + 0.5·max по критериям; consistent_quality_prob=True.
- область снимка: правило по ширине (≥ 290 px — позвоночник), для нестандартных ширин — `model_region_emb.pkl`.

Признаки контура A по фактически обученным моделям (models_manifest.json → feature_cols):

- sp_pos (model_spine_sp_pos_geom.pkl): center_offset_ratio, bone_width_ratio
- sp_axis (model_spine_sp_axis_geom.pkl): axis_angle_deg
- sp_art (model_spine_sp_art_geom.pkl): metal_metal_area_mm2, metal_metal_max_intensity_gap
- any (model_spine_any_geom.pkl): axis_angle_deg, bone_width_ratio, center_offset_ratio, metal_metal_area_mm2, metal_metal_max_intensity_gap
- hip_pos (model_hip_pos_geom.pkl): femur_solidity, shaft_width_mm, abs_shaft_angle_deg, merge_height_mm, medial_neck_extent_mm
- hip_roi (model_hip_roi_geom.pkl): scan_length_mm, shaft_len_below_troch_mm
- any (model_right_hip_any_geom.pkl): abs_shaft_angle_deg, femur_solidity, medial_neck_extent_mm, merge_height_mm, scan_length_mm, shaft_len_below_troch_mm, shaft_width_mm
- any (model_left_hip_any_geom.pkl): abs_shaft_angle_deg, femur_solidity, medial_neck_extent_mm, merge_height_mm, scan_length_mm, shaft_len_below_troch_mm, shaft_width_mm

## 3. Данные

- Обучение и валидация: только DICOM организаторов (набор для обучения; см. `docs/EVIDENCE.md`), разметка — xlsx организаторов, один разметчик. Файлы конфиденциальны и в репозиторий не входят.
- Валидных снимков в OOF-оценке: позвоночник 166, бедро 329 (metrics_summary.json → n_valid).
- Внешние наборы использованы только для предобучения бэкбона `densito` и OOD-теста; в обучении классификаторов и в репозитории их нет (`docs/GPU_EXPERIMENT.md`, `docs/DATASETS_DEEP_SEARCH.md`).

## 4. Протокол оценки

Out-of-fold (OOF) предсказания: повторный GroupKFold с группировкой по исследованию (study_uid), бутстрап доверительных интервалов по исследованиям; пороги подбираются только по OOF (threshold_method в таблице). Подробности и отвергнутые варианты — `docs/EVIDENCE.md`, `docs/METRICS_REPORT.md`.

## 5. Метрики по критериям (OOF)

| Критерий | Назначение | n_valid | n_pos | AUC geom | AUC emb (источник) | AUC стек | Порог (метод) | F1 OOF [95% ДИ] | Примечание |
|---|---|---|---|---|---|---|---|---|---|
| sp_pos | Позвоночник: укладка (центр, симметрия) | 166 | 10 | 0.681 | 0.790 (densito) | 0.734 | 0.822 (prevalence) | 0.476 [0.14; 0.78] |  |
| sp_axis | Позвоночник: ось позвоночника | 166 | 17 | 0.839 | 0.797 (densito_inv) | 0.889 | 0.777 (prevalence_x1.4) | 0.605 [0.31; 0.81] |  |
| sp_art | Позвоночник: посторонние предметы | 166 | 35 | 0.560 | 0.897 (imagenet) | 0.823 | 0.602 (prevalence_x1.4) | 0.535 [0.23; 0.73] |  |
| hip_pos | Бедро (обе стороны, общая модель): укладка | 329 | 79 | 0.710 | 0.635 (imagenet) | 0.725 | 0.658 (prevalence) | 0.432 [0.20; 0.60] |  |
| hip_roi | Бедро (обе стороны, общая модель): область интереса | 329 | 16 | 0.902 | 0.877 (imagenet) | 0.917 | 0.918 (prevalence) | 0.514 [0.00; 0.89] |  |
| rh_pos | Правое бедро: укладка | 161 | 39 | 0.802 | 0.517 (общая hip-модель) | 0.711 | 0.658 (prevalence_shared_hip_model) | 0.435 [0.18; 0.65] |  |
| rh_roi | Правое бедро: область интереса | 161 | 7 | 1.000 | 0.879 (общая hip-модель) | 0.970 | 0.918 (prevalence_shared_hip_model) | 0.375 [0.00; 1.00] | ненадёжно: <10 позитивов |
| lh_pos | Левое бедро: укладка | 168 | 40 | 0.625 | 0.737 (общая hip-модель) | 0.734 | 0.658 (prevalence_shared_hip_model) | 0.430 [0.19; 0.63] |  |
| lh_roi | Левое бедро: область интереса | 168 | 9 | 0.817 | 0.871 (общая hip-модель) | 0.866 | 0.918 (prevalence_shared_hip_model) | 0.632 [0.00; 1.00] | ненадёжно: <10 позитивов |

ДИ — бутстрап по исследованиям, 95 %. F1 считается при пороге из колонки «Порог». Критерии с n_pos < 10 (sp_pos на границе, rh_roi/lh_roi) — оценки неустойчивы: ДИ F1 включает 0.

Nested-оценка (порог и стекинг подобраны внутри внешних фолдов):

Протокол: repeated GroupKFold 5 внешних фолдов × 10 повторов, 3 внутренних; группы — исследование + хэш пикселей; вес стэкинга и порог выбираются только на внутренних фолдах (`tools/nested_gate.py`, `docs/NESTED_GATE_REPORT.md`). AUC — среднее по 10 повторам для базового стэкинга 0.5/0.5 (он и используется); полные таблицы с ДИ — в отчёте.

- sp_pos: nested AUC = 0.702; OOF AUC = 0.734, протокол nested_gate_K2
- sp_axis: nested AUC = 0.860; OOF AUC = 0.889, протокол emb_gate_K13
- sp_art: nested AUC = 0.817; OOF AUC = 0.823, протокол nested_gate_K2
- hip_pos: nested AUC = 0.709; OOF AUC = 0.725, протокол nested_gate_K2
- hip_roi: nested AUC = 0.873; OOF AUC = 0.917, протокол nested_gate_K2

## 6. Пороги и правило решения

- Порог критерия: `config.yaml → thresholds.<критерий>`; если null — из `metrics_summary.json → threshold`; если нет и там — fallback_threshold = 0.5.
- Сейчас в config.yaml: sp_pos=из metrics_summary, sp_axis=из metrics_summary, sp_art=из metrics_summary, rh_pos=из metrics_summary, rh_roi=из metrics_summary, lh_pos=из metrics_summary, lh_roi=из metrics_summary.
- Правило: критерий срабатывает, если стек-скор ≥ порога; quality_class = 1, если сработал хотя бы один критерий области; violation_type — официальные строки сработавших критериев через «;». quality_prob согласован с классом (class 1 → [0.5; 1], class 0 → [0; 0.5)).
- Ошибка чтения файла → строка Failure с quality_class 0, пустым violation_type и quality_prob = 0.5.

## 7. Ограничения

- Мало позитивов: sp_pos n_pos = 10, rh_roi/lh_roi n_pos = 7/9 — ДИ широкие, метрики по этим критериям ориентировочные.
- Один разметчик, один прибор (GE Lunar Prodigy), одна организация — переносимость на другие приборы не проверена на разметке.
- Чувствительность к гамме/шуму (см. `docs/ROBUSTNESS_REPORT.md`): часть решений меняется при искажении яркостной кривой.
- Не медицинское изделие; результат — подсказка для контроля качества укладки, не диагноз.

## 8. Версия и хэши

- Версия пайплайна (config.yaml → version): **2.3.0**; config_hash: **378ee6b98c9b** (тот же пишется в DICOM SR и ответ API).
- Ключевые библиотеки (requirements.txt): torch 2.14.0+cpu, torchvision 0.29.0+cpu, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.0.5, pydicom 3.0.2, opencv-python-headless 5.0.0.93, scikit-image 0.26.0, PyYAML 6.0.3, fastapi 0.141.1.

| Файл модели | Критерий | Признаки / источник | n_pos | sha256[:12] |
|---|---|---|---|---|
| model_spine_sp_pos_geom.pkl | sp_pos | center_offset_ratio, bone_width_ratio | 10 | 61c4f76fe1cc |
| model_spine_sp_pos_emb_pca.pkl | sp_pos | densito | 10 | b905b160ae25 |
| model_spine_sp_axis_geom.pkl | sp_axis | axis_angle_deg | 17 | 297e08c840f3 |
| model_spine_sp_axis_emb_pca.pkl | sp_axis | densito_inv | 17 | 4996fdee1eb7 |
| model_spine_sp_art_geom.pkl | sp_art | metal_metal_area_mm2, metal_metal_max_intensity_gap | 35 | 2d3037b0876a |
| model_spine_sp_art_emb_pca.pkl | sp_art | imagenet | 35 | 54a78475d746 |
| model_spine_any_geom.pkl | any | axis_angle_deg, bone_width_ratio, center_offset_ratio, metal_metal_area_mm2, metal_metal_max_intensity_gap | 60 | ef372ec24204 |
| model_spine_any_emb_pca.pkl | any |  | 60 | 57a0103e1d68 |
| model_hip_pos_geom.pkl | hip_pos | femur_solidity, shaft_width_mm, abs_shaft_angle_deg, merge_height_mm, medial_neck_extent_mm | 79 | 83b1dc39916f |
| model_hip_pos_emb_pca.pkl | hip_pos | imagenet | 79 | 8538d1a7c9ea |
| model_hip_roi_geom.pkl | hip_roi | scan_length_mm, shaft_len_below_troch_mm | 16 | cb8a84174a11 |
| model_hip_roi_emb_pca.pkl | hip_roi | imagenet | 16 | f150b56b2ce9 |
| model_right_hip_any_geom.pkl | any | abs_shaft_angle_deg, femur_solidity, medial_neck_extent_mm, merge_height_mm, scan_length_mm, shaft_len_below_troch_mm, shaft_width_mm | 92 | 10569d3273cf |
| model_right_hip_any_emb_pca.pkl | any |  | 92 | 3f57a44394f9 |
| model_left_hip_any_geom.pkl | any | abs_shaft_angle_deg, femur_solidity, medial_neck_extent_mm, merge_height_mm, scan_length_mm, shaft_len_below_troch_mm, shaft_width_mm | 92 | 10569d3273cf |
| model_left_hip_any_emb_pca.pkl | any |  | 92 | 3f57a44394f9 |
| model_region_emb.pkl | — | классификатор области | — | 66f8a86c2041 |
| backbone_densito.pth | — | бэкбон densito (EfficientNet-B0) | — | f99110664dad |

Источники: `models/metrics_summary.json`, `models/models_manifest.json`, `config.yaml`, `requirements.txt`; процедура валидации и отвергнутые гипотезы — `docs/EVIDENCE.md`.
