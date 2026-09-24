# Б1 (24.09.2026): предварительная регистрация кандидатов «контекст другого бедра визита» для hip_pos

Записано до первого запуска харнесса `tools/hip_pair_context_gate.py`. После запуска этот файл не меняется.

## База (текущая поставка 2.4.0)

- Контур A: `data/geometry_features_canonical.csv` (preprocessing.variant_by_criterion.hip_pos.geom = canonical),
  признаки `femur_solidity, shaft_width_mm, abs_shaft_angle_deg, merge_height_mm, medial_neck_extent_mm`,
  StandardScaler -> LogReg(C=1, balanced), медианы пропусков — по внешнему train.
- Контур B: `data/embeddings.npy` (imagenet), StandardScaler -> PCA(32) -> LogReg(C=0.1, balanced).
- Стэк 0.5·rank_geom + 0.5·rank_emb; ранги test — по референсу inner-OOF (как `percentile_rank` инференса).
- Порог hip_pos и hip_roi — prevalence (`config.yaml: thresholds_rule`) на inner-OOF стэке внешнего train.

## Протокол

- Строки бедра с меткой: n = 329, позитивов 79. Сторона — `hip_side_detected`.
- Группы — компоненты связности (исследование, pixel_hash) из `docs/k5/pixel_hashes.csv`; обе стороны исследования всегда в одном фолде.
- Внешний контур GroupKFold(5, shuffle, random_state = 42 + r), r = 0..19; внутренний GroupKFold(3, shuffle, random_state = 1000·r + k).
- Все базовые модели (hip_pos, hip_roi, any-модель бедра) переобучаются в каждом внешнем фолде.
- Скор другой стороны: для строк train — inner-OOF стэк; для строк test — стэк моделей внешнего фолда.
- R(s) — ранг стэка: для train pct-ранг внутри inner-OOF, для test — доля inner-OOF референса <= s.
- Где другой стороны нет, s' = R(s_own).

## Кандидаты

- (а) `w_inner`: s' = (1 − w)·R(s_own) + w·mean R(s_other), w ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.5} выбирается по AUC на inner-OOF внешнего train;
  при равенстве — меньший w.
- (б) `w_fixed_0.3`: то же с w = 0.3.
- (в) `grok_solidity`: s' = 0.5·R(s_own) + 0.5·R(sol_other), sol_other — среднее `femur_solidity` строк противоположной стороны
  (`data/geometry_features.csv`, как предложено), ранг — по распределению sol_other строк внешнего train; направление «больше = нарушение»
  зафиксировано заранее (физика наружной ротации).
- Порог каждого кандидата — prevalence на его inner-OOF скоре внешнего train.
- Справочно (не кандидаты, для объяснения расхождения с К5): `k5_rule_mean` = 0.5·s_own + 0.5·mean s_other на шкале стэка
  (правило К5) и прогон с контуром A на `data/geometry_features.csv` (база К5).

## Приёмка (все условия)

1. ΔAUC hip_pos ≥ 0.03 не менее чем в 14 из 20 повторов.
2. Средняя macro-F1 hip_pos не ниже базы.
3. AUC hip_roi и AUC бинарной задачи бедра (quality_prob = consistent(0.5·any + 0.5·max(s_pos, s_roi)), класс = OR флагов) не ниже базы в среднем по повторам.
4. `tools/paired_gate.py` на per-repeat скорах hip_pos: вердикт «принят», партия из 3 гипотез (Холм).
5. Отказ, если эффект не переживает удаление одного исследования (jackknife по группам меняет знак) или на исследованиях без второго бедра меняются флаги по иной причине, чем пересчёт порога.
