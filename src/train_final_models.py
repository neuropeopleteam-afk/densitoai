"""
Финальное обучение на 100% данных и сохранение моделей в формате
models/MODEL_CONTRACT.md ("Вариант 2 — dict").

Запускать ПОСЛЕ train_stacked.py: оттуда берутся гиперпараметры (те же
константы) и OOF-скоры (models/oof_stacked_<region>_<crit>.csv), которые кладутся
в pickle как референс для ранг-стэкинга (`oof_scores`).

Сохраняемые файлы:
  model_spine_<crit>_geom.pkl / model_spine_<crit>_emb_pca.pkl   (sp_pos, sp_axis, sp_art)
  model_hip_pos_geom.pkl / model_hip_pos_emb_pca.pkl               (единая модель на обе стороны)
  model_hip_roi_geom.pkl / model_hip_roi_emb_pca.pkl
  model_spine_any_geom.pkl / model_spine_any_emb_pca.pkl           (OR критериев региона)
  model_right_hip_any_*.pkl, model_left_hip_any_*.pkl              (копии единой hip-any модели —
                                                                    inference ищет any-модель
                                                                    только под именем региона)
  oof_stacked_right_hip_rh_*.csv, oof_stacked_left_hip_lh_*.csv    (per-side срез OOF единой модели,
                                                                    чтобы затереть устаревшие файлы
                                                                    старых раздельных моделей)

ВАЖНО про бедро и `mirror_right`:
  * Геометрические признаки бедра (hip_features.hip_all_features) САМИ определяют
    сторону по изображению (hip_features.detect_hip_side) и зеркалят ЛЕВОЕ бедро
    к канонической ориентации правого. Поэтому на инференсе изображение для
    контура A зеркалить НЕ нужно — нужно лишь вызвать hip_all_features(img_u8)
    (или geometry_features.extract_all_features(path, 'hip')).
  * Эмбеддинги (контур B) обучены на ОРИГИНАЛЬНЫХ (незеркалированных) снимках
    обеих сторон: зеркалирование не улучшало OOF AUC (см. train_stacked.py).
    Следовательно `mirror_right = False`: ни правое, ни левое бедро перед
    EfficientNet-B0 зеркалить не надо.
  * Сторона для текста violation_type должна браться из
    hip_features.detect_hip_side(img_u8) (плотностная эвристика ошибается в 24%).
"""
import warnings
warnings.filterwarnings("ignore")
import sys, json, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from train_stacked import (DATA_DIR, OUT_DIR, REGION_CRITERIA, CRITERION_GEOMETRY_COLS, CRITERION_EXTRA_COLS, contour_a_cols,
                           load_embeddings_by_source, emb_source_for,
                           CRITERION_LABEL_COL, REGION_ROWS, HIP_SIDE_REPORT, PCA_COMPONENTS,
                           GEOM_VARIANT_FILES, preproc_for, emb_matrix_for)

# Гиперпараметры — ТЕ ЖЕ, что в train_stacked.train_region_stacked
GEOM_C, EMB_C = 1.0, 0.1

HIP_MIRROR_DOC = (
    "mirror_right=False. Контур A: hip_features.hip_all_features(img_u8) сам определяет сторону "
    "(hip_features.detect_hip_side) и зеркалит ЛЕВОЕ бедро в каноническую ориентацию правого; "
    "изображение перед вызовом не зеркалить. Контур B: эмбеддинг EfficientNet-B0 от ОРИГИНАЛЬНОГО "
    "снимка любой стороны (обучение без зеркалирования)."
)


def _impute(X, medians):
    X = X.astype(np.float64).copy()
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j])
        X[m, j] = medians[j]
    return X


def fit_geom(X, y, cols, medians, oof=None, extra=None):
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, C=GEOM_C, class_weight='balanced').fit(scaler.transform(X), y)
    d = {'scaler': scaler, 'pca': None, 'clf': clf, 'feature_cols': list(cols),
         'medians': {c: float(m) for c, m in zip(cols, medians)},
         'oof_scores': None if oof is None else np.asarray(oof, dtype=np.float64),
         'sklearn_version': sklearn.__version__, 'n_train': int(len(y)), 'n_pos': int(y.sum()),
         'hyperparams': {'C': GEOM_C, 'class_weight': 'balanced'}}
    d.update(extra or {})
    return d


def fit_emb(E, y, oof=None, extra=None, emb_source='imagenet'):
    scaler = StandardScaler().fit(E)
    n_comp = min(PCA_COMPONENTS, len(y) - 1, E.shape[1])
    pca = PCA(n_components=n_comp, random_state=42).fit(scaler.transform(E))
    clf = LogisticRegression(max_iter=1000, C=EMB_C, class_weight='balanced').fit(
        pca.transform(scaler.transform(E)), y)
    d = {'scaler': scaler, 'pca': pca, 'clf': clf, 'feature_cols': None, 'medians': None,
         'oof_scores': None if oof is None else np.asarray(oof, dtype=np.float64),
         'sklearn_version': sklearn.__version__, 'n_train': int(len(y)), 'n_pos': int(y.sum()),
         'hyperparams': {'C': EMB_C, 'class_weight': 'balanced', 'pca_components': n_comp},
         'emb_source': emb_source,
         'embedding': f'embeddings.FrozenBackbone(source={emb_source!r}) (EfficientNet-B0, resize 320x192), 1280-d'}
    d.update(extra or {})
    return d


def save(d, name):
    p = OUT_DIR / name
    with open(p, 'wb') as f:
        pickle.dump(d, f, protocol=4)
    print(f"  saved {name} ({p.stat().st_size/1024:.0f} KB, n={d['n_train']}, n_pos={d['n_pos']})")


def main():
    # К11: все варианты предобработки контура A; по критерию выбирает preproc_for().
    geom_all_by_variant = {v: pd.read_csv(DATA_DIR / fn) for v, fn in GEOM_VARIANT_FILES.items()
                           if (DATA_DIR / fn).exists()}
    geom_all = geom_all_by_variant['baseline']
    emb_labels = pd.read_csv(DATA_DIR / 'labels_for_embeddings.csv')
    emb_by_source = load_embeddings_by_source()
    manifest = {}

    for region, criteria in REGION_CRITERIA.items():
        rows = REGION_ROWS.get(region, [region])
        gdf_by_variant = {v: g[g['region'].isin(rows)].reset_index(drop=True)
                          for v, g in geom_all_by_variant.items()}
        gdf = gdf_by_variant['baseline']
        for v, g in gdf_by_variant.items():
            assert (g['file_path'].values == gdf['file_path'].values).all(), \
                f"порядок строк варианта '{v}' не совпадает с baseline"
        eidx = np.nonzero(emb_labels['region'].isin(rows).values)[0]
        E_by_source = {k: v[eidx] for k, v in emb_by_source.items()}
        E_all = E_by_source['imagenet']
        assert (emb_labels.loc[eidx, 'file_path'].values == gdf['file_path'].values).all()
        print(f"\n=== {region}: {len(gdf)} images ===")

        is_hip = region == 'hip'
        extra = {'mirror_right': False, 'mirror_doc': HIP_MIRROR_DOC,
                 'side_detection': 'hip_features.detect_hip_side'} if is_hip else {}
        any_label = np.zeros(len(gdf), dtype=float); any_seen = np.zeros(len(gdf), dtype=bool)

        for crit in criteria:
            y = gdf[CRITERION_LABEL_COL.get(crit, crit)].values.astype(float)
            valid = ~np.isnan(y)
            any_label[valid] = np.maximum(any_label[valid], y[valid]); any_seen |= valid
            yv = y[valid].astype(int)
            preproc = preproc_for(crit)
            geom_variant = preproc['geom'] if preproc['geom'] in gdf_by_variant else 'baseline'
            cols = contour_a_cols(crit)   # H2: геометрия + признаки из эмбеддинга (sp_pos: synth_pos_logit)
            missing_cols = [c for c in cols if c not in gdf_by_variant[geom_variant].columns]
            assert not missing_cols, f"{crit}: нет колонок {missing_cols} (tools/add_sppos_head_feature.py)"
            Xraw = gdf_by_variant[geom_variant][cols].values.astype(np.float64)
            med = np.nanmedian(Xraw, axis=0)           # медианы по всем строкам региона (как в OOF)
            X = _impute(Xraw, med)[valid]
            E_full, emb_src, emb_variant = emb_matrix_for(crit, E_by_source)
            E = E_full[valid]
            extra_crit = dict(extra, preproc_geom=geom_variant, preproc_emb=emb_variant)
            print(f"  {crit}: контур A '{geom_variant}', контур B '{emb_src}'/'{emb_variant}'")

            oof_path = OUT_DIR / f'oof_stacked_{region}_{crit}.csv'
            oof = pd.read_csv(oof_path) if oof_path.exists() else None
            if oof is not None:
                assert (oof['file_path'].values == gdf['file_path'].values[valid]).all(), 'OOF order mismatch'
            oof_g = None if oof is None else oof['oof_geom'].values
            oof_e = None if oof is None else oof['oof_emb'].values

            base = f'model_{region}_{crit}' if not is_hip else f'model_{crit}'   # model_hip_pos / model_hip_roi
            dg = fit_geom(X, yv, cols, med, oof_g, dict(extra_crit, criterion=crit, region=region))
            de = fit_emb(E, yv, oof_e, dict(extra_crit, criterion=crit, region=region), emb_source=emb_src)
            save(dg, f'{base}_geom.pkl'); save(de, f'{base}_emb_pca.pkl')
            manifest[f'{base}_geom.pkl'] = {'criterion': crit, 'feature_cols': cols, 'n_pos': int(yv.sum())}
            manifest[f'{base}_geom.pkl'].update({'preproc_geom': geom_variant})
            extra_cols = CRITERION_EXTRA_COLS.get(crit) or []
            if extra_cols:
                # H2 (2.4.0): признаки контура A из эмбеддинга — откуда берутся (голова и её эмбеддинг)
                from sppos_head import HEAD_FILE, EMB_SOURCE, EMB_VARIANT
                manifest[f'{base}_geom.pkl']['embedding_features'] = {
                    c: {'head_file': HEAD_FILE, 'emb_source': EMB_SOURCE, 'preproc_emb': EMB_VARIANT} for c in extra_cols}
            manifest[f'{base}_emb_pca.pkl'] = {'criterion': crit, 'n_pos': int(yv.sum()), 'emb_source': emb_src,
                                               'preproc_emb': emb_variant}

            # per-side OOF csv для совместимости с oof_stacked_<region>_<crit>.csv инференса
            if is_hip and oof is not None and 'hip_side_detected' in oof:
                for (side, c), (reg_name, crit_name) in HIP_SIDE_REPORT.items():
                    if c == crit:
                        oof[oof['hip_side_detected'] == side].to_csv(
                            OUT_DIR / f'oof_stacked_{reg_name}_{crit_name}.csv', index=False)

        # --- "есть хоть одно нарушение" = OR критериев региона ---
        # any-модель — только геометрия (CRITERION_GEOMETRY_COLS): признаки из эмбеддинга (H2) сюда не входят,
        # бинарная модель региона в 2.4.0 не менялась.
        ya = any_label[any_seen].astype(int)
        cols_any = sorted({c for cr in criteria for c in CRITERION_GEOMETRY_COLS[cr]})
        Xraw = gdf[cols_any].values.astype(np.float64)
        med = np.nanmedian(Xraw, axis=0)
        dg = fit_geom(_impute(Xraw, med)[any_seen], ya, cols_any, med, None,
                      dict(extra, criterion='any', region=region, label='OR(' + ','.join(criteria) + ')'))
        de = fit_emb(E_all[any_seen], ya, None, dict(extra, criterion='any', region=region))
        names = [region] if not is_hip else ['right_hip', 'left_hip']
        for nm in names:
            save(dg, f'model_{nm}_any_geom.pkl'); save(de, f'model_{nm}_any_emb_pca.pkl')
            manifest[f'model_{nm}_any_geom.pkl'] = {'criterion': 'any', 'feature_cols': cols_any, 'n_pos': int(ya.sum())}
            manifest[f'model_{nm}_any_emb_pca.pkl'] = {'criterion': 'any', 'n_pos': int(ya.sum())}

    with open(OUT_DIR / 'models_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nmanifest: {OUT_DIR / 'models_manifest.json'}")


if __name__ == '__main__':
    main()
