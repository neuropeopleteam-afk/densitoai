"""
Финальное обучение: Контур A (геометрия) + Контур B (замороженные эмбеддинги)
-> логрегрессия на каждом контуре отдельно -> стэкинг ранговым усреднением.

Честная валидация: repeated GroupKFold (5x5) по study, study-level bootstrap
для ДИ. Пороги: для критериев с >=15 позитивами — максимизация F1 на OOF
(с плато-усреднением), для критериев с <15 позитивами — prevalence-порог
(top-k% по рангу, k = доля позитивов в трейне), без подгонки на шуме.

Стэкинг: score = w_geom*rank(geom) + (1-w_geom)*rank(emb); w_geom по критерию берётся из
config.yaml (stacking.weights_by_criterion, дефолт weight_geom=0.5) и здесь НЕ подбирается —
выбор веса делается только в nested CV (tools/nested_gate.py), см. models/nested_gate_decisions.json.
Пути переопределяются переменными окружения DENSITO_ROOT / DENSITO_DATA_DIR / DENSITO_MODELS_DIR /
DENSITO_CONFIG (по умолчанию — корень репозитория).
"""
import warnings
warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import os
import pickle
from calibration_utils import fit_platt, apply_platt, threshold_by_rule, select_margins, row_reject_rates, is_uncertain
PROJECT_ROOT = Path(os.environ.get("DENSITO_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = Path(os.environ.get("DENSITO_DATA_DIR", PROJECT_ROOT / "data"))
OUT_DIR = Path(os.environ.get("DENSITO_MODELS_DIR", PROJECT_ROOT / "models"))
CONFIG_PATH = Path(os.environ.get("DENSITO_CONFIG", PROJECT_ROOT / "config.yaml"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGION_CRITERIA = {
    'spine': ['sp_pos', 'sp_axis', 'sp_art'],
    # Бедро: ОДНА объединённая модель на left+right (рекомендация Fable 5 про
    # зеркалирование). Геометрические признаки считаются в канонической
    # ориентации (левое бедро зеркалится, см. hip_features.py), сторона
    # определяется по изображению, метка берётся из разметка.xlsx для
    # обнаруженной стороны (колонки hip_pos_c / hip_roi_c в geometry_features.csv).
    # Метрики rh_*/lh_* в metrics_summary.json — это OOF-предсказания той же
    # объединённой модели, разбитые по стороне (порог общий, по стороне НЕ
    # подбирается).
    'hip': ['hip_pos', 'hip_roi'],
}

# критерий -> источник эмбеддингов контура B (см. src/embeddings.py). По умолчанию imagenet.
# 'densito' — наш B0, предобученный на GPU на 15,6 тыс. снимках кости прокси-задачами укладки;
# на нашей разметке он устойчиво лучше ImageNet ТОЛЬКО для укладки позвоночника (sp_pos: AUC
# контура B 0.60 -> 0.80, стек 0.60 -> 0.72 на этом протоколе; в gpu/eval_embeddings.py +0.13,
# 10/10 повторов). Для hip_pos на боевом протоколе (метка по обнаруженной стороне) выигрыша нет
# (0.635 -> 0.636), для артефактов он хуже (-0.14) — там остаётся ImageNet.
# Значение по умолчанию (поведение 2.1.0/2.2.0), если в config.yaml нет секции embeddings.
# К13: источник истины — config.yaml embeddings.source_by_criterion (см. emb_source_config).
EMB_SOURCE_BY_CRITERION = {'sp_pos': 'densito'}
EMB_FILES = {'imagenet': 'embeddings.npy', 'densito': 'embeddings_densito.npy',
             # К13: бэкбоны, обученные на инвариантность эмбеддинга к гамме и шуму
             'densito_inv': 'embeddings_densito_inv.npy',
             'densito_inv_free': 'embeddings_densito_inv_free.npy'}

# К11: вариант предобработки по критерию (config.yaml: preprocessing.variant_by_criterion).
# Источник истины — конфиг; здесь только чтение и прокидывание в metrics_summary.json / pkl.
PREPROC_DEFAULT = {'geom': 'baseline', 'emb': 'baseline'}
GEOM_VARIANT_FILES = {'baseline': 'geometry_features.csv',
                      'mask': 'geometry_features_mask.csv',
                      'canonical': 'geometry_features_canonical.csv'}


def preproc_for(crit):
    """{'geom': вариант, 'emb': вариант} для критерия. enabled: false -> всё baseline (2.1.0)."""
    sec = load_config_section('preprocessing') or {}
    if not sec.get('enabled', False):
        return dict(PREPROC_DEFAULT)
    entry = (sec.get('variant_by_criterion') or {}).get(crit) or {}
    out = dict(PREPROC_DEFAULT)
    for k in ('geom', 'emb'):
        if entry.get(k):
            v = str(entry[k])
            assert v in GEOM_VARIANT_FILES, f"config preprocessing.variant_by_criterion.{crit}.{k}: неизвестный вариант '{v}'"
            out[k] = v
    return out


def emb_file_for(source, variant):
    base = 'embeddings' if source == 'imagenet' else f'embeddings_{source}'
    return f'{base}.npy' if variant == 'baseline' else f'{base}_{variant}.npy'


def load_embeddings_by_source():
    """{(source, variant): np.ndarray} для всех найденных файлов + плоские ключи source
    (= вариант baseline) для совместимости с tools/nested_gate.py и аудитами."""
    out = {}
    for src in EMB_FILES:
        for variant in GEOM_VARIANT_FILES:
            f = DATA_DIR / emb_file_for(src, variant)
            if f.exists():
                arr = np.load(f)
                out[(src, variant)] = arr
                if variant == 'baseline':
                    out[src] = arr
    assert 'imagenet' in out, 'нет data/embeddings.npy'
    return out


def emb_source_config():
    """{критерий: источник эмбеддингов} из config.yaml embeddings.source_by_criterion.

    Источник истины — конфиг; при отсутствии секции берётся EMB_SOURCE_BY_CRITERION
    (поведение до К13). Значение попадает в meta .pkl (emb_source) и в metrics_summary.json,
    согласованность проверяется tools/verify_checks.py."""
    sec = load_config_section('embeddings') or {}
    by = sec.get('source_by_criterion') or {}
    if not by:
        return dict(EMB_SOURCE_BY_CRITERION)
    out = {}
    for crit, src in by.items():
        s = str(src)
        assert s == 'imagenet' or s in EMB_FILES, \
            f"config embeddings.source_by_criterion.{crit}: неизвестный источник '{s}'"
        out[str(crit)] = s
    return out


def emb_source_for(crit, available):
    src = emb_source_config().get(crit, 'imagenet')
    if src not in available:
        print(f"  [warn] эмбеддинги '{src}' для {crit} не найдены — используется imagenet")
        src = 'imagenet'
    return src


def emb_matrix_for(crit, available):
    """(матрица эмбеддингов, source, variant) для критерия с учётом варианта предобработки."""
    src = emb_source_for(crit, available)
    variant = preproc_for(crit)['emb']
    if (src, variant) not in available:
        print(f"  [warn] эмбеддинги '{src}'/'{variant}' для {crit} не найдены — берётся baseline")
        variant = 'baseline'
    return available[(src, variant)], src, variant


# критерий -> колонка метки в geometry_features.csv (по умолчанию совпадают)
CRITERION_LABEL_COL = {'hip_pos': 'hip_pos_c', 'hip_roi': 'hip_roi_c'}
# регион -> какие значения колонки region в geometry_features.csv / labels_for_embeddings.csv
REGION_ROWS = {'spine': ['spine'], 'hip': ['right_hip', 'left_hip']}
# для отчёта по сторонам: (сторона, критерий объединённой модели) -> имя критерия ТЗ
HIP_SIDE_REPORT = {('right', 'hip_pos'): ('right_hip', 'rh_pos'), ('right', 'hip_roi'): ('right_hip', 'rh_roi'),
                   ('left', 'hip_pos'): ('left_hip', 'lh_pos'), ('left', 'hip_roi'): ('left_hip', 'lh_roi')}

# Признак-специфичные наборы для геометрической модели каждого критерия.
# ВАЖНО (найдено эмпирически): использование всех 9 геометрических признаков
# в одной логрегрессии для каждого критерия РАЗРУШАЕТ сигнал сильных
# одиночных признаков из-за переобучения на <20 позитивах (напр. sp_axis:
# AUC 0.84 одним признаком axis_angle_deg -> 0.60 при добавлении 8 шумных).
# Поэтому здесь у каждого критерия — свой минимальный, физически осмысленный
# набор признаков (по определению критерия в ТЗ), не общий для региона.
CRITERION_GEOMETRY_COLS = {
    'sp_pos': ['center_offset_ratio', 'bone_width_ratio'],
    'sp_axis': ['axis_angle_deg'],
    'sp_art': ['metal_metal_area_mm2', 'metal_metal_max_intensity_gap'],
    # --- бедро (hip_features.py, каноническая ориентация) ---
    # Позиционирование/ротация. Физика: при наружной ротации шейка укорачивается
    # в проекции (medial_neck_extent_mm ↓, merge_height_mm ↓), силуэт становится
    # выпуклее (femur_solidity ↑), проксимальный диафиз шире в проекции
    # (shaft_width_mm ↑); отклонение оси диафиза от вертикали (abs_shaft_angle_deg).
    # Набор выбран по OOF GroupKFold на объединённых данных (0.69) и
    # проверен, что работает на ОБЕИХ сторонах (right 0.75 / left 0.71);
    # более широкие наборы (+compactness, +signed angle, +вертелы) давали
    # ~0.68 merged, но разваливались на левой стороне (0.53-0.59) -> переобучение.
    'hip_pos': ['femur_solidity', 'shaft_width_mm', 'abs_shaft_angle_deg',
                'merge_height_mm', 'medial_neck_extent_mm'],
    # Корректность ROI. Все ROI-позитивы — короткие сканы: 16 позитивов имеют
    # высоту 180-261 строк (190-275 мм) — область сканирования обрезана и не
    # содержит достаточно диафиза ниже малого вертела. scan_length_mm один даёт
    # AUC 0.92 (инверт.), shaft_len_below_troch_mm — 0.87 (инверт.).
    # ЧЕСТНО: lh_roi имеет всего 4 позитива в исходной разметке (9 после
    # исправления стороны), из них один — аномально увеличенный снимок и пара
    # дубликатов; per-side метрики lh_roi статистически ненадёжны.
    'hip_roi': ['scan_length_mm', 'shaft_len_below_troch_mm'],
}

N_FOLDS = 5
N_REPEATS = 5
PCA_COMPONENTS = 32  # снижаем размерность эмбеддингов 1280 -> 32 для устойчивости на малых данных

# --- Вентильный стэкинг (К2). Вес контура A по критерию: скор = w*rank(geom) + (1-w)*rank(emb).
# Источник истины — config.yaml: stacking.weights_by_criterion (ключи sp_pos, sp_axis, sp_art,
# hip_pos, hip_roi); по умолчанию stacking.weight_geom (0.5). Значения выбраны ТОЛЬКО в nested
# repeated GroupKFold (tools/nested_gate.py -> models/nested_gate_decisions.json): вентиль принят
# для критерия, если прирост AUC >= 0.03 в >= 7 из 10 повторов и macro-F1 не хуже.
# ВНИМАНИЕ: сам этот скрипт вес НЕ подбирает — иначе OOF-метрики были бы оптимистичны.
DEFAULT_WEIGHT_GEOM = 0.5
NESTED_DECISIONS_FILE = 'nested_gate_decisions.json'
# К13: nested-гейт источника эмбеддингов (tools/emb_gate.py). Нужен здесь потому, что запись К2
# считалась при СВОЁМ источнике эмбеддингов: если источник с тех пор изменён, её AUC к продакшену
# уже не относится, и честную nested-оценку берём из ветки fixed:<источник> этого файла.
EMB_GATE_DECISIONS_FILE = 'emb_gate_decisions.json'


def load_stacking_weights():
    """{crit: w_geom} из config.yaml (weights_by_criterion поверх weight_geom)."""
    default, by_crit = DEFAULT_WEIGHT_GEOM, {}
    if CONFIG_PATH.exists():
        try:
            import yaml
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                st = (yaml.safe_load(f) or {}).get('stacking', {}) or {}
            default = float(st.get('weight_geom', default))
            by_crit = {k: float(v) for k, v in (st.get('weights_by_criterion') or {}).items() if v is not None}
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] config.yaml не прочитан ({e}) — веса стэкинга 0.5/0.5")
    return default, by_crit


# Правило порога по критерию — config.yaml: thresholds_rule.by_criterion (ключи sp_pos, sp_axis, sp_art,
# hip_pos, hip_roi), значения: f1_optimal (F1-опт при >= 15 позитивах, иначе prevalence), prevalence,
# prevalence_x<k>. Выбор правила — ТОЛЬКО по nested-протоколу (tools/calibration_eval.py --stage nested,
# отчёт docs/METRICS_REPORT.md «Калибровка и зона не уверен»), здесь оно не подбирается.
DEFAULT_THRESHOLD_RULE = 'f1_optimal'


def load_config_section(name):
    if CONFIG_PATH.exists():
        try:
            import yaml
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return (yaml.safe_load(f) or {}).get(name, {}) or {}
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] config.yaml не прочитан ({e}) — секция {name} по умолчанию")
    return {}


def threshold_rule_for(crit):
    sec = load_config_section('thresholds_rule')
    by = sec.get('by_criterion') or {}
    return str(by.get(crit) or sec.get('default') or DEFAULT_THRESHOLD_RULE)


def uncertainty_config():
    sec = load_config_section('uncertainty')
    return {'max_reject_rate': float(sec.get('max_reject_rate', 0.05)),
            'quota_step': float(sec.get('quota_step', 0.0025))}


def weight_geom_for(crit):
    default, by_crit = load_stacking_weights()
    return float(np.clip(by_crit.get(crit, default), 0.0, 1.0))


def nested_decision_for(crit):
    """Запись nested-валидации вентиля для критерия (models/nested_gate_decisions.json), если есть."""
    p = OUT_DIR / NESTED_DECISIONS_FILE
    if not p.exists():
        return None
    try:
        with open(p, 'r', encoding='utf-8') as f:
            d = json.load(f)
        for rec in d.get('criteria', []):
            if rec.get('criterion') == crit:
                return rec
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] {p.name} не прочитан: {e}")
    return None


def emb_gate_decision_for(crit):
    """Запись nested-гейта источника эмбеддингов для критерия (models/emb_gate_decisions.json)."""
    p = OUT_DIR / EMB_GATE_DECISIONS_FILE
    if not p.exists():
        return None
    try:
        with open(p, 'r', encoding='utf-8') as f:
            d = json.load(f)
        for rec in d.get('decisions', []):
            if rec.get('criterion') == crit:
                return rec
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] {p.name} не прочитан: {e}")
    return None


def nested_block_for(crit, emb_src, nested):
    """nested-оценка AUC стэка для критерия ПРИ ТЕКУЩЕМ источнике эмбеддингов.

    Оба протокола одинаковы по схеме (внешний GroupKFold 5 × повторы, внутренний 3, порог по
    правилу на inner-OOF) и отличаются тем, что выбирается внутри: К2 — вес вентиля,
    К13 — источник эмбеддингов. Берём ту запись, которая относится к текущему источнику;
    если ни одной нет, поле остаётся пустым, а не заполняется числом от другого бэкбона.
    """
    empty = {'protocol': None, 'mean': None, 'ci': None, 'base_mean': None, 'gain': None,
             'production': None, 'alternative': None, 'alternative_what': None}
    if nested is not None and str(nested.get('emb_source') or 'imagenet') == emb_src:
        # К2: в продакшене стоит базовый стэкинг 0.5/0.5 (вентиль не принят),
        # альтернатива — вес вентиля, выбранный внутри фолдов.
        return {'protocol': 'nested_gate_K2', 'mean': nested.get('auc_gate_mean_over_repeats'),
                'ci': nested.get('auc_gate_ci'), 'base_mean': nested.get('auc_base_mean_over_repeats'),
                'gain': nested.get('n_repeats_gain_ge_0.03'),
                'production': nested.get('auc_base_mean_over_repeats'),
                'alternative': nested.get('auc_gate_mean_over_repeats'),
                'alternative_what': 'вес вентиля по критерию (К2, не принят)'}
    g = emb_gate_decision_for(crit)
    fx = ((g or {}).get('fixed') or {}).get(emb_src)
    if fx:
        # К13: в продакшене стоит текущий источник (ветка fixed), альтернатива — прежний источник.
        return {'protocol': 'emb_gate_K13', 'mean': fx.get('auc_mean_over_repeats'),
                'ci': fx.get('auc_ci'), 'base_mean': (g or {}).get('auc_base_mean_over_repeats'),
                'gain': fx.get('n_repeats_gain_ge_0.03'),
                'production': fx.get('auc_mean_over_repeats'),
                'alternative': (g or {}).get('auc_base_mean_over_repeats'),
                'alternative_what': f"прежний источник эмбеддингов '{(g or {}).get('base_source')}'"}
    print(f"  [warn] нет nested-записи для {crit} при источнике '{emb_src}' — поля nested_* пустые")
    return empty


def prevalence_threshold(y_train, scores_train):
    """Порог = квантиль, соответствующий доле позитивов в трейне (без подгонки на шуме)."""
    prevalence = y_train.mean()
    if prevalence <= 0 or prevalence >= 1:
        return 0.5
    thresh = np.quantile(scores_train, 1 - prevalence)
    return thresh


def f1_optimal_threshold(y_true, scores, min_positives_for_optimization=15):
    """F1-оптимальный порог с плато-усреднением, только если достаточно позитивов."""
    if y_true.sum() < min_positives_for_optimization:
        return None  # сигнал использовать prevalence-порог вместо этого
    thresholds = np.unique(scores)
    best_f1 = -1
    best_thresholds = []
    for t in thresholds:
        preds = (scores >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresholds = [t]
        elif f1 == best_f1:
            best_thresholds.append(t)
    return float(np.mean(best_thresholds))


def study_level_bootstrap_f1(df_region, y_true_col, oof_pred_col, threshold, n_boot=1000, seed=42):
    """Bootstrap ДИ для F1, ресэмплинг на уровне study (не изображения)."""
    rng = np.random.default_rng(seed)
    studies = df_region['study'].unique()
    f1_scores = []
    valid = df_region.dropna(subset=[y_true_col, oof_pred_col])
    for _ in range(n_boot):
        sampled_studies = rng.choice(studies, size=len(studies), replace=True)
        sample_df = pd.concat([valid[valid['study'] == s] for s in sampled_studies], ignore_index=True) \
            if len(sampled_studies) < 200 else None
        # эффективнее: строим индекс через merge count, но для простоты и малых данных ок
        rows = []
        for s in sampled_studies:
            rows.append(valid[valid['study'] == s])
        if not rows:
            continue
        sample = pd.concat(rows, ignore_index=True)
        if sample[y_true_col].sum() == 0 and (sample[oof_pred_col] >= threshold).sum() == 0:
            f1_scores.append(1.0)  # both agree no positives -> trivially perfect on this resample
            continue
        preds = (sample[oof_pred_col] >= threshold).astype(int)
        try:
            f1 = f1_score(sample[y_true_col], preds, zero_division=0)
            f1_scores.append(f1)
        except Exception:
            continue
    if not f1_scores:
        return (0.0, 0.0, 0.0)
    return (float(np.percentile(f1_scores, 2.5)), float(np.mean(f1_scores)), float(np.percentile(f1_scores, 97.5)))


def train_region_stacked(region, criteria):
    print(f"\n=== {region} | criteria: {criteria} ===")
    region_rows = REGION_ROWS.get(region, [region])
    # К11: все варианты контура A, какой брать — решает preproc_for(критерий).
    geom_by_variant = {}
    for variant, fn in GEOM_VARIANT_FILES.items():
        f = DATA_DIR / fn
        if not f.exists():
            continue
        g = pd.read_csv(f)
        geom_by_variant[variant] = g[g['region'].isin(region_rows)].reset_index(drop=True)
    assert 'baseline' in geom_by_variant, 'нет data/geometry_features.csv'
    geom_df = geom_by_variant['baseline']
    for variant, g in geom_by_variant.items():
        assert (g['file_path'].values == geom_df['file_path'].values).all(), \
            f"порядок строк geometry_features варианта '{variant}' не совпадает с baseline"

    # Эмбеддинги: ОРИГИНАЛЬНЫЕ (без зеркалирования). Проверено (hip_eval /
    # embeddings_hip_canonical.py): зеркалирование левого бедра перед
    # EfficientNet-B0 НЕ улучшает OOF AUC (pos 0.54 vs 0.63; roi 0.83 vs 0.89) —
    # ImageNet-признаки не инвариантны к отражению, а объединённая выборка
    # выигрыш не даёт. Поэтому зеркалим только геометрию.
    emb_labels = pd.read_csv(DATA_DIR / 'labels_for_embeddings.csv')
    emb_by_source = load_embeddings_by_source()
    emb_labels['emb_idx'] = np.arange(len(emb_labels))
    region_emb_idx = emb_labels[emb_labels['region'].isin(region_rows)]['emb_idx'].values
    region_emb_by_source = {k: v[region_emb_idx] for k, v in emb_by_source.items()}
    region_embeddings = region_emb_by_source['imagenet']

    assert len(geom_df) == len(region_embeddings), f"Mismatch {len(geom_df)} vs {len(region_embeddings)}"
    assert (emb_labels.loc[region_emb_idx, 'file_path'].values == geom_df['file_path'].values).all(), \
        "embeddings/geometry row order mismatch"

    groups = geom_df['study'].values
    results = {}

    for crit in criteria:
        y = geom_df[CRITERION_LABEL_COL.get(crit, crit)].values
        valid_mask = ~pd.isna(y)
        y_valid = y[valid_mask].astype(int)
        n_pos = y_valid.sum()
        print(f"\n--- criterion: {crit} | n_valid={valid_mask.sum()} | n_pos={n_pos} ---")

        preproc = preproc_for(crit)
        geom_variant = preproc['geom'] if preproc['geom'] in geom_by_variant else 'baseline'
        if geom_variant != preproc['geom']:
            print(f"  [warn] признаки варианта '{preproc['geom']}' не найдены — берётся baseline")
        geom_src_df = geom_by_variant[geom_variant]

        geom_cols = CRITERION_GEOMETRY_COLS[crit]
        X_geom_raw = geom_src_df[geom_cols].values.astype(np.float64)
        col_medians = np.nanmedian(X_geom_raw, axis=0)
        for j in range(X_geom_raw.shape[1]):
            mask_nan = np.isnan(X_geom_raw[:, j])
            X_geom_raw[mask_nan, j] = col_medians[j]

        X_geom = X_geom_raw[valid_mask]
        emb_matrix, emb_src, emb_variant = emb_matrix_for(crit, region_emb_by_source)
        print(f"  контур A: предобработка '{geom_variant}' | контур B: эмбеддинги '{emb_src}'/'{emb_variant}'")
        X_emb = emb_matrix[valid_mask]
        groups_valid = groups[valid_mask]
        studies_valid = geom_df['study'].values[valid_mask]
        file_paths_valid = geom_df['file_path'].values[valid_mask]

        oof_geom = np.full(len(y_valid), np.nan)
        oof_emb = np.full(len(y_valid), np.nan)

        all_repeat_oof_geom = np.zeros((N_REPEATS, len(y_valid)))
        all_repeat_oof_emb = np.zeros((N_REPEATS, len(y_valid)))

        for repeat in range(N_REPEATS):
            gkf = GroupKFold(n_splits=N_FOLDS)
            rng_seed = 42 + repeat
            # shuffle groups order deterministically per repeat for different splits
            rng = np.random.default_rng(rng_seed)
            perm = rng.permutation(len(y_valid))

            for fold, (train_idx_p, val_idx_p) in enumerate(gkf.split(X_geom[perm], groups=groups_valid[perm])):
                train_idx = perm[train_idx_p]
                val_idx = perm[val_idx_p]

                if y_valid[train_idx].sum() < 2 or y_valid[train_idx].sum() == len(train_idx):
                    # skip degenerate fold (no variation to train on)
                    continue

                # Contour A: geometry -> logistic regression
                scaler_g = StandardScaler()
                Xg_train = scaler_g.fit_transform(X_geom[train_idx])
                Xg_val = scaler_g.transform(X_geom[val_idx])
                clf_g = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
                clf_g.fit(Xg_train, y_valid[train_idx])
                pg = clf_g.predict_proba(Xg_val)[:, 1]
                all_repeat_oof_geom[repeat, val_idx] = pg

                # Contour B: embeddings -> PCA -> logistic regression with strong L2
                n_comp = min(PCA_COMPONENTS, len(train_idx) - 1, X_emb.shape[1])
                pca = PCA(n_components=n_comp, random_state=42)
                scaler_e = StandardScaler()
                Xe_train_scaled = scaler_e.fit_transform(X_emb[train_idx])
                Xe_train = pca.fit_transform(Xe_train_scaled)
                Xe_val = pca.transform(scaler_e.transform(X_emb[val_idx]))
                clf_e = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced')
                clf_e.fit(Xe_train, y_valid[train_idx])
                pe = clf_e.predict_proba(Xe_val)[:, 1]
                all_repeat_oof_emb[repeat, val_idx] = pe

        # average across repeats (ignoring NaN from skipped folds)
        oof_geom = np.nanmean(all_repeat_oof_geom, axis=0)
        oof_emb = np.nanmean(all_repeat_oof_emb, axis=0)

        # rank-average stacking с весом по критерию (вентиль К2; вес из config.yaml, НЕ подбирается здесь)
        w_geom = weight_geom_for(crit)
        nested = nested_decision_for(crit)
        nested_block = nested_block_for(crit, emb_src, nested)
        gate_selected_by = 'nested' if (nested is not None and nested.get('accepted')) else 'default'
        rank_geom = pd.Series(oof_geom).rank(pct=True).values
        rank_emb = pd.Series(oof_emb).rank(pct=True).values
        oof_stacked = w_geom * rank_geom + (1.0 - w_geom) * rank_emb
        print(f"  стэкинг: w_geom={w_geom} ({gate_selected_by})")

        # AUC comparison (only if variation)
        aucs = {}
        for name, scores in [('geom', oof_geom), ('emb', oof_emb), ('stacked', oof_stacked)]:
            valid_score = ~np.isnan(scores)
            if valid_score.sum() > 0 and len(np.unique(y_valid[valid_score])) > 1:
                aucs[name] = roc_auc_score(y_valid[valid_score], scores[valid_score])
            else:
                aucs[name] = None
        print(f"  OOF AUC: geom={aucs['geom']}, emb={aucs['emb']}, stacked={aucs['stacked']}")

        # thresholding: правило из config.yaml (thresholds_rule), см. threshold_rule_for
        thr_rule = threshold_rule_for(crit)
        thresh, thresh_method = threshold_by_rule(thr_rule, y_valid, oof_stacked,
                                                  f1_optimal_threshold, prevalence_threshold, min_pos_f1=15)
        print(f"  threshold={thresh} ({thresh_method}; правило {thr_rule})")

        preds = (oof_stacked >= thresh).astype(int)
        f1 = f1_score(y_valid, preds, zero_division=0)
        print(f"  OOF F1 (stacked, threshold={thresh_method}): {f1:.3f}")

        # bootstrap CI (study-level)
        boot_df = pd.DataFrame({'study': studies_valid, 'y': y_valid, 'pred_score': oof_stacked})
        ci_lo, ci_mean, ci_hi = study_level_bootstrap_f1(boot_df, 'y', 'pred_score', thresh, n_boot=500)
        print(f"  F1 95% CI (study-level bootstrap): [{ci_lo:.3f}, {ci_hi:.3f}] (mean={ci_mean:.3f})")

        results[crit] = {
            'n_valid': int(valid_mask.sum()), 'n_pos': int(n_pos),
            'auc_geom': aucs['geom'], 'auc_emb': aucs['emb'], 'auc_stacked': aucs['stacked'], 'emb_source': emb_src,
            'preproc': {'geom': geom_variant, 'emb': emb_variant},
            'threshold': float(thresh) if thresh is not None else None,
            'threshold_method': thresh_method,
            'threshold_rule': thr_rule,
            'n_flag_oof': int(preds.sum()),
            'f1_oof': float(f1),
            'f1_ci_lo': ci_lo, 'f1_ci_hi': ci_hi,
            # вентиль К2: вес контура A и откуда он взят
            'weight_geom': w_geom, 'weight_emb': 1.0 - w_geom, 'gate_selected_by': gate_selected_by,
            # nested-оценка берётся из протокола, который относится к текущему источнику эмбеддингов
            'nested_protocol': nested_block['protocol'],
            # nested_auc_production — оценка ТОГО, ЧТО СТОИТ В ПРОДАКШЕНЕ (её и печатать в документах);
            # nested_auc_alternative — то, с чем сравнивали и что не взяли (или взяли взамен).
            'nested_auc_production': nested_block['production'],
            'nested_auc_alternative': nested_block['alternative'],
            'nested_alternative_what': nested_block['alternative_what'],
            'nested_auc_mean': nested_block['mean'],
            'nested_auc_ci': nested_block['ci'],
            'nested_auc_base_mean': nested_block['base_mean'],
            'nested_repeats_gain_ge_0.03': nested_block['gain'],
        }

        # save OOF details
        oof_out = pd.DataFrame({
            'study': studies_valid, 'file_path': file_paths_valid,
            'y_true': y_valid, 'oof_geom': oof_geom, 'oof_emb': oof_emb,
            'oof_stacked': oof_stacked, 'pred_label': preds,
        })
        if 'hip_side_detected' in geom_df.columns:
            oof_out['hip_side_detected'] = geom_df['hip_side_detected'].values[valid_mask]
        # %.17g — полная точность double: значение, равное порогу, должно остаться
        # равным ему и после чтения CSV (иначе отчёт метрик разойдётся с сервисом)
        oof_out.to_csv(OUT_DIR / f'oof_stacked_{region}_{crit}.csv', index=False,
                       float_format='%.17g')

        # --- per-side отчёт для бедра (та же модель, тот же порог) ---
        if region == 'hip':
            sides = geom_df['hip_side_detected'].values[valid_mask]
            results[crit]['by_side'] = {}
            for side in ['right', 'left']:
                m = sides == side
                ys, n_pos_s = y_valid[m], int(y_valid[m].sum())
                side_aucs = {}
                for name, scores in [('geom', oof_geom), ('emb', oof_emb), ('stacked', oof_stacked)]:
                    sc = scores[m]
                    ok = ~np.isnan(sc)
                    side_aucs[name] = roc_auc_score(ys[ok], sc[ok]) if len(np.unique(ys[ok])) > 1 else None
                f1_s = f1_score(ys, preds[m], zero_division=0)
                boot_s = pd.DataFrame({'study': studies_valid[m], 'y': ys, 'pred_score': oof_stacked[m]})
                lo_s, mean_s, hi_s = study_level_bootstrap_f1(boot_s, 'y', 'pred_score', thresh, n_boot=500)
                print(f"  [{side}] n={m.sum()} n_pos={n_pos_s} AUC geom={side_aucs['geom']}, "
                      f"emb={side_aucs['emb']}, stacked={side_aucs['stacked']}, F1={f1_s:.3f} CI=[{lo_s:.3f},{hi_s:.3f}]")
                results[crit]['by_side'][side] = {
                    'n_valid': int(m.sum()), 'n_pos': n_pos_s,
                    'auc_geom': side_aucs['geom'], 'auc_emb': side_aucs['emb'], 'auc_stacked': side_aucs['stacked'],
                    'threshold': float(thresh), 'threshold_method': thresh_method + '_shared_hip_model',
                    'weight_geom': w_geom, 'weight_emb': 1.0 - w_geom, 'gate_selected_by': gate_selected_by,
                    'f1_oof': float(f1_s), 'f1_ci_lo': lo_s, 'f1_ci_hi': hi_s,
                    'note': ('ненадёжно: <10 позитивов' if n_pos_s < 10 else 'ok'),
                }

    return results


def build_calibration(all_results):
    """Platt (a, b) на OOF-стэке критерия и запас margin: доля OOF-строк с |score - thr| < margin <= max_reject_rate
    по критерию и по строкам региона (общая квота q подбирается вниз от max_reject_rate)."""
    ucfg = uncertainty_config()
    crit_info, margins_data = {}, {}
    for region, criteria in REGION_CRITERIA.items():
        for crit in criteria:
            df = pd.read_csv(OUT_DIR / f'oof_stacked_{region}_{crit}.csv')
            y = df['y_true'].values.astype(int); s = df['oof_stacked'].values.astype(float)
            thr = float(all_results[region][crit]['threshold'])
            a, b = fit_platt(s, y)
            p = apply_platt(s, a, b)
            crit_info[crit] = {'region': region, 'threshold': thr, 'platt': {'a': a, 'b': b},
                               'brier_oof_platt': float(np.mean((p - y) ** 2)),
                               'brier_oof_constant': float(np.mean((y.mean() - y) ** 2)),
                               'p_cal_at_threshold': float(apply_platt(thr, a, b))}
            margins_data[crit] = np.abs(s - thr)
    margins, q = select_margins(margins_data, REGION_CRITERIA, max_reject=ucfg['max_reject_rate'],
                                step=ucfg['quota_step'],
                                quota_scope=ucfg.get('quota_scope', 'per_criterion'))
    rows = row_reject_rates(margins_data, REGION_CRITERIA, margins)
    import math
    margins = {c: (0.0 if abs(v) < 1e-9 else math.ceil(float(v) * 1e9) / 1e9) for c, v in margins.items()}  # вверх до 1e-9
    for crit in crit_info:
        crit_info[crit]['margin'] = float(margins[crit])
        crit_info[crit]['uncertain_rate_oof'] = float(is_uncertain(margins_data[crit], margins[crit]).mean())
    calib = {'format_version': 1, 'kind': 'densito_calibration',
             'criteria': crit_info, 'margin_by_criterion': {c: float(v) for c, v in margins.items()},
             'max_reject_rate': ucfg['max_reject_rate'], 'quota_per_criterion': q,
             'row_uncertain_rate_by_region': rows,
             'any_model_calibration': None,  # не принято по cross-fit (см. docs/METRICS_REPORT.md)
             'note': 'Platt: p = sigmoid(a*score + b) на стэкнутом ранговом скоре; «не уверен» <=> |score - threshold| <= margin.'}
    print(f"\n=== К3: калибровка и запасы === квота по регионам q={q}; доля строк «не уверен» по регионам: {rows}")
    for crit, ci in crit_info.items():
        print(f"  {crit}: Platt a={ci['platt']['a']:.3f} b={ci['platt']['b']:.3f}, Brier OOF {ci['brier_oof_platt']:.4f} "
              f"(константа {ci['brier_oof_constant']:.4f}), margin={ci['margin']:.4f}, не уверен {ci['uncertain_rate_oof']:.3%}")
    return calib


def main():
    all_results = {}
    if '--calibration-only' in sys.argv:
        # пересчитать только calibration.pkl и блок uncertainty по уже сохранённым OOF/порогам
        with open(OUT_DIR / 'metrics_summary.json', 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        all_results.pop('uncertainty', None)
        calib = build_calibration(all_results)
        with open(OUT_DIR / 'calibration.pkl', 'wb') as f:
            pickle.dump(calib, f)
        for region, criteria in REGION_CRITERIA.items():
            for crit in criteria:
                all_results[region][crit]['platt'] = calib['criteria'][crit]['platt']
                all_results[region][crit]['margin'] = calib['criteria'][crit]['margin']
                all_results[region][crit]['uncertain_rate_oof'] = calib['criteria'][crit]['uncertain_rate_oof']
        all_results['uncertainty'] = {k: calib[k] for k in ('max_reject_rate', 'quota_per_criterion',
                                                              'margin_by_criterion', 'row_uncertain_rate_by_region')}
        with open(OUT_DIR / 'metrics_summary.json', 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        return
    for region, criteria in REGION_CRITERIA.items():
        results = train_region_stacked(region, criteria)
        all_results[region] = results

    # Совместимость: блоки right_hip / left_hip с критериями ТЗ (rh_pos, rh_roi,
    # lh_pos, lh_roi) — per-side срез OOF объединённой модели 'hip'.
    if 'hip' in all_results:
        for (side, crit), (reg_name, crit_name) in HIP_SIDE_REPORT.items():
            all_results.setdefault(reg_name, {})[crit_name] = all_results['hip'][crit]['by_side'][side]

    # --- К3: калибровка Platt по критерию и запас «не уверен» (на полном OOF; оценка качества — cross-fit
    # в tools/calibration_eval.py). Файл models/calibration.pkl по контракту MODEL_CONTRACT.md.
    calib = build_calibration(all_results)
    with open(OUT_DIR / 'calibration.pkl', 'wb') as f:
        pickle.dump(calib, f)
    for region, criteria in REGION_CRITERIA.items():
        for crit in criteria:
            all_results[region][crit]['platt'] = calib['criteria'][crit]['platt']
            all_results[region][crit]['margin'] = calib['criteria'][crit]['margin']
            all_results[region][crit]['uncertain_rate_oof'] = calib['criteria'][crit]['uncertain_rate_oof']
    all_results['uncertainty'] = {k: calib[k] for k in ('max_reject_rate', 'quota_per_criterion',
                                                          'margin_by_criterion', 'row_uncertain_rate_by_region')}

    with open(OUT_DIR / 'metrics_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n\n=== SUMMARY ===")
    for region, results in all_results.items():
        if region == 'uncertainty':
            continue
        for crit, m in results.items():
            print(f"{region}.{crit}: n_pos={m['n_pos']}, w_geom={m.get('weight_geom')}, AUC_geom={m['auc_geom']}, "
                  f"AUC_emb={m['auc_emb']}, AUC_stacked={m['auc_stacked']}, "
                  f"F1={m['f1_oof']:.3f} CI=[{m['f1_ci_lo']:.3f},{m['f1_ci_hi']:.3f}]")


if __name__ == '__main__':
    main()
