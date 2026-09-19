"""К5 п.3: патчи 128x128 нативного разрешения у малого вертела -> эмбеддинг EfficientNet-B0 (imagenet), контур B2."""
import os; os.environ['OMP_NUM_THREADS']='1'; os.environ['TORCH_HOME']='models/torch_home'
import sys, time, numpy as np, pandas as pd, cv2, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'src')
import torch; torch.set_num_threads(1)
from geometry_features import read_dicom_normalized
from hip_features import segment_bone_hip, hip_side_score, track_femur, _fit_shaft, PIXEL_SPACING_X_MM as SX, PIXEL_SPACING_Y_MM as SY
from scipy.signal import find_peaks
from embeddings import FrozenBackbone
from torchvision import transforms
PATCH = 128

def lesser_troch_xy(mask_c):
    """(x, y, source) в канонических координатах: пик медиального контура над диафизом; fallback — медиальный край верха диафиза."""
    tr = track_femur(mask_c)
    if tr is None or len(tr['y']) < 15: return None
    y, L, R, W = tr['y'], tr['left'], tr['right'], tr['width']; n = len(y)
    shaft_top, (al, bl), (ar, br) = _fit_shaft(y, L, R, SX, SY)
    yf = y.astype(float); dev_med = (R - np.polyval([ar, br], yf)) * SX
    w0 = float(np.median(W[:shaft_top]))
    jump = np.nonzero(np.diff(R) * SX > 15.0)[0] + 1; wide = np.nonzero(W > 3.0 * w0)[0]
    cands = [i for i in list(jump) + list(wide) if i > shaft_top]; end = min(cands) if cands else n
    dm = dev_med[shaft_top:end]
    if len(dm) >= 5:
        dm_s = np.convolve(np.pad(dm, 2, mode='edge'), np.ones(5)/5, mode='valid')
        peaks, props = find_peaks(dm_s, prominence=1.0)
        for pk in peaks:
            if dm_s[pk] > 1.5:
                k = shaft_top + pk; return int(R[k]), int(y[k]), 'peak'
    k = shaft_top - 1
    return int(R[k]), int(y[k]), 'shaft_top'

def crop(img, x, y, s=PATCH):
    h, w = img.shape; pad = s//2
    P = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    return P[y:y+s, x:x+s]

def main():
    t0=time.time()
    lab = pd.read_csv('data/labels_for_embeddings.csv')
    hip = lab[lab.region.isin(['right_hip','left_hip'])].reset_index(drop=True)
    bb = FrozenBackbone('imagenet')
    bb.tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    embs, meta, tiles = [], [], []
    for i, r in hip.iterrows():
        img, _ = read_dicom_normalized(r.file_path)
        mask = segment_bone_hip(img); side = 'right' if hip_side_score(img, mask) >= 0 else 'left'
        img_c = img if side=='right' else np.ascontiguousarray(img[:, ::-1]); mask_c = mask if side=='right' else np.ascontiguousarray(mask[:, ::-1])
        lt = lesser_troch_xy(mask_c)
        if lt is None:
            x, y, src = img.shape[1]//2, img.shape[0]//2, 'none'
        else: x, y, src = lt
        patch = crop(img_c, x, y)      # каноническая ориентация (медиальный край всегда справа), нативное разрешение
        embs.append(bb.extract(patch)); meta.append(dict(file_path=r.file_path, side=side, x_canon=x, y_canon=y, src=src))
        if len(tiles) < 24 and i % 14 == 0:
            t = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR); cv2.putText(t, src[:5], (2,12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1); tiles.append(t)
        if (i+1) % 50 == 0: print(i+1, f'{time.time()-t0:.0f}s', flush=True)
    np.save('outputs/k5/out/emb_troch_patch_hip.npy', np.stack(embs))
    m = pd.DataFrame(meta); m.to_csv('outputs/k5/out/troch_patch_meta.csv', index=False)
    print(m.src.value_counts().to_dict(), f'{time.time()-t0:.0f}s')
    grid = np.vstack([np.hstack(tiles[j:j+6]) for j in range(0, 24, 6)]) if len(tiles)>=24 else np.hstack(tiles)
    cv2.imwrite('outputs/k5/out/img/troch_patches_grid.png', grid)
if __name__ == '__main__': main()
