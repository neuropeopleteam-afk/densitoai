"""Извлекает геометрические признаки (Контур A) для всех 499 изображений и
сохраняет в data/geometry_features.csv для последующей оценки разделяющей
способности и обучения логрегрессии/бустинга поверх признаков.
"""
import warnings
warnings.filterwarnings("ignore")
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
from geometry_features import extract_all_features

DATA_CSV = Path("/home/user/workspace/densito_rebuild/data/labels_full.csv")
OUT_CSV = Path("/home/user/workspace/densito_rebuild/data/geometry_features.csv")


def main():
    df = pd.read_csv(DATA_CSV)
    df = df[df['region'].isin(['spine', 'right_hip', 'left_hip'])].reset_index(drop=True)
    print(f"Extracting features for {len(df)} images...")

    rows = []
    t0 = time.time()
    for i, row in df.iterrows():
        feats = extract_all_features(row['file_path'], row['region'])
        feats['study'] = row['study']
        feats['file_path'] = row['file_path']
        rows.append(feats)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(df)} done, {time.time()-t0:.1f}s elapsed")

    feat_df = pd.DataFrame(rows)
    merged = df.merge(feat_df, on=['study', 'file_path', 'region'], how='left')

    # --- Сторона бедра по ИЗОБРАЖЕНИЮ и метки, скорректированные по стороне ---
    # В labels_full.csv сторона (region right_hip/left_hip) назначена эвристикой
    # плотности (build_dataset.hip_side_by_density) и ошибочна для ~24% снимков
    # (проверка: анатомический детектор даёт ровно right+left в 63/64
    # исследованиях с чётным числом бедренных снимков; плотностный — в 53/64).
    # labels_full.csv и build_dataset.py здесь НЕ меняем (порядок строк должен
    # совпадать с embeddings.npy) — добавляем колонки:
    #   hip_side_detected, hip_side_score  — сторона по hip_features.detect_hip_side
    #   rh_pos_c, rh_roi_c, lh_pos_c, lh_roi_c — метки из разметка.xlsx для
    #       обнаруженной стороны (NaN для снимков другой стороны)
    #   hip_pos_c, hip_roi_c — метка "своей" стороны для объединённой модели hip
    from build_dataset import load_labels  # только чтение xlsx
    xl = load_labels()
    xl.index = xl.index.astype(str)
    for c in ['rh_pos_c', 'rh_roi_c', 'lh_pos_c', 'lh_roi_c', 'hip_pos_c', 'hip_roi_c']:
        merged[c] = float('nan')
    is_hip = merged['region'].isin(['right_hip', 'left_hip']) & merged['hip_side_detected'].notna()
    for i in merged.index[is_hip]:
        st = str(merged.at[i, 'study'])
        if st not in xl.index:
            continue
        lab = xl.loc[st]
        side = merged.at[i, 'hip_side_detected']
        pfx = 'rh' if side == 'right' else 'lh'
        merged.at[i, f'{pfx}_pos_c'] = lab[f'{pfx}_pos']
        merged.at[i, f'{pfx}_roi_c'] = lab[f'{pfx}_roi']
        merged.at[i, 'hip_pos_c'] = lab[f'{pfx}_pos']
        merged.at[i, 'hip_roi_c'] = lab[f'{pfx}_roi']
    n_changed = int((is_hip & (merged['region'].str.replace('_hip', '') != merged['hip_side_detected'])).sum())
    print(f"hip side: detected differs from labels_full.csv region for {n_changed}/{int(is_hip.sum())} hip images")
    merged.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(merged)} rows to {OUT_CSV}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
