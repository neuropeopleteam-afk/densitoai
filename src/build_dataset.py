"""
Строит полный датасет уровня изображения с мультилейбл метками
по отдельным критериям (НЕ по агрегированной колонке "Итог"),
как явно указали организаторы на Q&A-сессии.

Выход: data/labels_full.csv с колонками:
  study, file_path, sop_instance_uid, instance_number, rows, cols,
  region (spine/right_hip/left_hip/unknown; сторона бедра — hip_features.detect_hip_side, К5 п.1),
  sp_pos, sp_axis, sp_art, rh_pos, rh_roi, lh_pos, lh_roi (0/1/NaN — по применимости),
  quality_class (0/1, агрегат по критериям применимого региона, НЕ по "Итог"),
  violation_list (текст через ';' по русским названиям, как ожидает выходной формат)
"""
import warnings
warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pydicom

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path("/home/user/workspace/dataset_extracted/Датасет/НД_для_обучения")
STUDIES_DIR = ROOT / "Исследования"
LABELS_XLSX = ROOT / "разметка.xlsx"

VIOLATION_NAMES = {
    'sp_pos': 'Нарушение укладки поясничного отдела позвоночника',
    'sp_axis': 'Нарушение оси позвоночника (отклонение более 5 градусов)',
    'sp_art': 'Наличие посторонних предметов, артефактов или наложений',
    'rh_pos': 'Нарушение позиционирования/ротации правого бедра',
    'rh_roi': 'Нарушение корректности области интереса правого бедра',
    'lh_pos': 'Нарушение позиционирования/ротации левого бедра',
    'lh_roi': 'Нарушение корректности области интереса левого бедра',
}


def load_labels():
    df = pd.read_excel(LABELS_XLSX, header=[0, 1])
    df.columns = ['num', 'study', 'sp_pos', 'sp_axis', 'sp_art', 'rh_pos', 'rh_roi',
                  'lh_pos', 'lh_roi', 'itog_sp', 'itog_rh', 'itog_lh',
                  'itog_comment1', 'itog_comment2', 'gen_sp', 'gen_rh',
                  'gen_lh1', 'gen_lh2', 'gen_lh3']
    df['study'] = df['study'].astype(str).str.strip()
    return df.set_index('study')


def classify_region_from_dims(rows, cols):
    """Классификация региона по размерам изображения (правило,
    подтверждённое организаторами на Q&A: 300px спина, 280/248 бедро)."""
    if cols is None:
        return 'unknown'
    if cols >= 290:
        return 'spine'
    return 'hip_unresolved'  # разрешаем сторону ниже по плотности


def hip_side_by_density(pixel_array):
    img = pixel_array.astype(np.float32)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min()) * 255.0
    img = img.astype(np.uint8)
    h, w = img.shape
    threshold = np.percentile(img, 60)
    left_density = (img[:, :w // 2] > threshold).sum() / img[:, :w // 2].size
    right_density = (img[:, w // 2:] > threshold).sum() / img[:, w // 2:].size
    return 'right_hip' if right_density > left_density else 'left_hip'


def hip_side_anatomical(pixel_array, ds=None):
    """Сторона бедра по анатомии (hip_features.detect_hip_side: таз всегда медиальнее
    диафиза), как в inference.detect_hip_side_region. Плотностная эвристика
    hip_side_by_density ошибается на ~24% снимков (80/333, work/I/REPORT.md, К5 п.1);
    она остаётся только резервом при исключении. Возвращает (region, source)."""
    try:
        from geometry_features import read_dicom_normalized  # noqa: WPS433
        from hip_features import detect_hip_side
        if ds is not None:
            arr = pixel_array.astype(np.float32)
            if getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2') == 'MONOCHROME1':
                arr = arr.max() - arr
            lo, hi = np.percentile(arr, [1, 99])
            img_u8 = (np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255) if hi > lo else np.zeros_like(arr)).astype(np.uint8)
        else:
            img_u8 = pixel_array.astype(np.uint8)
        return detect_hip_side(img_u8) + '_hip', 'anatomical'
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: detect_hip_side failed ({e}) -> density heuristic")
        return hip_side_by_density(pixel_array), 'density'


def main():
    labels = load_labels()
    rows_out = []

    study_dirs = sorted([d for d in STUDIES_DIR.iterdir() if d.is_dir()])
    print(f"Studies found on disk: {len(study_dirs)}, in labels: {len(labels)}")

    for sd in study_dirs:
        study_id = sd.name
        if study_id not in labels.index:
            print(f"WARNING: study {study_id} not in labels xlsx, skipping")
            continue
        lab = labels.loc[study_id]

        dcm_files = sorted(sd.rglob("*.dcm"))
        # first pass: read all + get dims
        parsed = []
        for f in dcm_files:
            try:
                ds = pydicom.dcmread(str(f), force=True)
                pixel_array = ds.pixel_array
                rows_, cols_ = int(getattr(ds, 'Rows', 0)), int(getattr(ds, 'Columns', 0))
                parsed.append({
                    'file': f, 'ds': ds, 'pixel_array': pixel_array,
                    'rows': rows_, 'cols': cols_,
                    'sop_uid': str(getattr(ds, 'SOPInstanceUID', '')),
                    'instance_number': int(getattr(ds, 'InstanceNumber', 0)),
                })
            except Exception as e:
                print(f"ERROR reading {f}: {e}")

        # classify region per file
        hip_items = []
        for item in parsed:
            region = classify_region_from_dims(item['rows'], item['cols'])
            if region == 'hip_unresolved':
                # К5 п.1: сторона по анатомии (детектор), плотность — только резерв
                region, item['side_source'] = hip_side_anatomical(item['pixel_array'], item['ds'])
            item['region'] = region

        for item in parsed:
            region = item['region']
            if region == 'spine':
                crits = {'sp_pos': lab['sp_pos'], 'sp_axis': lab['sp_axis'], 'sp_art': lab['sp_art']}
            elif region == 'right_hip':
                crits = {'rh_pos': lab['rh_pos'], 'rh_roi': lab['rh_roi']}
            elif region == 'left_hip':
                crits = {'lh_pos': lab['lh_pos'], 'lh_roi': lab['lh_roi']}
            else:
                crits = {}

            crit_vals = [v for v in crits.values() if pd.notna(v)]
            if crit_vals:
                quality_class = 1 if any(v == 1 for v in crit_vals) else 0
                violations = [VIOLATION_NAMES[k] for k, v in crits.items() if pd.notna(v) and v == 1]
                violation_list = '; '.join(violations)
                applicable = True
            else:
                quality_class = np.nan
                violation_list = ''
                applicable = False

            row = {
                'study': study_id,
                'file_path': str(item['file']),
                'sop_instance_uid': item['sop_uid'],
                'instance_number': item['instance_number'],
                'rows': item['rows'],
                'cols': item['cols'],
                'region': region,
                'applicable': applicable,
                'quality_class': quality_class,
                'violation_list': violation_list,
            }
            for k in ['sp_pos', 'sp_axis', 'sp_art', 'rh_pos', 'rh_roi', 'lh_pos', 'lh_roi']:
                row[k] = crits.get(k, np.nan)
            rows_out.append(row)

    out_df = pd.DataFrame(rows_out)
    out_path = Path("/home/user/workspace/densito_rebuild/data/labels_full.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} rows to {out_path}")
    print(out_df['region'].value_counts())
    print(out_df.groupby('region')['quality_class'].value_counts(dropna=False))


if __name__ == '__main__':
    main()
