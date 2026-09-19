"""
extras.py — экспериментальные дополнительные флаги DensitoAI (страница «Дополнительно», results_extras.csv).

Ничего из этого модуля не влияет на 9 колонок основного CSV. Все функции чистые, без
исключений наружу (возвращают словари с полями *_error при сбое).

Функции:
  white_lines(img_u8)                 — детектор «белых линий» (ROI-графика софта аппарата):
                                        пиксели 254–255 после нормализации + морфология
                                        (длинные тонкие горизонтальные/вертикальные сегменты,
                                        контраст с обеих сторон линии).
  exposure_stats(img_u8)              — статистики экспозиции (медиана/перцентили тела, шум лапласианом).
  fingerprint(tags)                   — fingerprint DICOM-тегов GE Lunar Prodigy.
  OODGate / load_ood_gate / ood_score — Mahalanobis в PCA-пространстве эмбеддингов imagenet.
  endoprosthesis(img_u8, region)      — правило «эндопротез» по маске очень плотных пикселей.
  pixel_hash(ds)                      — sha1 сырых пикселей (дубликаты кадров).
  study_coherence(rows)               — аудитор согласованности исследования (warning).
  compute_extras_for_rows(rows, debug_rows, images) — сборка results_extras.csv.

Параметры гейта: models/ood_gate.pkl (см. OOD_GATE_PKL_VERSION).
"""
from __future__ import annotations

import hashlib
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

EXTRAS_VERSION = "0.1.0"
OOD_GATE_PKL_VERSION = "ood_gate/1"

EXTRAS_COLUMNS = ["file", "image_uid", "white_lines_flag", "white_lines_len_px", "ood_flag",
                  "ood_mahalanobis", "ood_fingerprint_ok", "ood_reason", "endoprosthesis_suspected",
                  "axis_curved_flag", "study_warnings"]
# Выраженный изгиб оси столба: std остатка (линейная − квадратичная подгонка оси, px) выше 99 % квантиля
# размеченной нормы (106 кадров позвоночника, data/geometry_features.csv: q99 = 9.0 px). Не диагноз, только подсказка.
AXIS_CURVATURE_MAX_PX = 9.0


# --------------------------------------------------------------------------- #
# 1. Белые линии
# --------------------------------------------------------------------------- #
def _runs_1d(vec: np.ndarray) -> List[Tuple[int, int]]:
    """Отрезки подряд идущих True: [(start, end_exclusive), ...]."""
    v = np.concatenate([[0], vec.astype(np.int8), [0]])
    d = np.diff(v)
    starts = np.nonzero(d == 1)[0]
    ends = np.nonzero(d == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def white_lines(img_u8: np.ndarray, sat_level: int = 254, min_len: int = 20, max_thick: int = 3,
                side_contrast: int = 100) -> Dict[str, Any]:
    """
    Детектор линий ROI-графики: пиксели >= sat_level (после нормализации в uint8),
    длинные (>= min_len) строго горизонтальные/вертикальные отрезки толщиной <= max_thick px,
    у которых по ОБЕ стороны (на расстоянии max_thick+1 px) интенсивность ниже линии
    не менее чем на side_contrast (отсекает саттурированные гребни кортикального слоя,
    у которых хотя бы одна сторона тоже яркая).

    Возвращает: has_lines, n_segments, n_horizontal, n_vertical, total_len_px, frac_px,
    sat_frac (доля пикселей >= sat_level до морфологии), segments (список (ориентация, y/x, start, end)).
    """
    try:
        img = np.asarray(img_u8)
        if img.ndim != 2:
            raise ValueError("expected 2-D uint8 image")
        h, w = img.shape
        sat = img >= sat_level
        sat_frac = float(sat.mean())
        segs: List[Tuple[str, int, int, int]] = []
        line_mask = np.zeros_like(sat, dtype=bool)
        off = max_thick + 1
        imgf = img.astype(np.float32)

        def _scan(mat_sat: np.ndarray, mat_img: np.ndarray, orient: str):
            n, m = mat_sat.shape
            for i in range(n):
                row = mat_sat[i]
                if row.sum() < min_len:
                    continue
                for s, e in _runs_1d(row):
                    if e - s < min_len:
                        continue
                    # толщина: сколько соседних строк повторяют этот отрезок (>= 90 % покрытия)
                    thick = 1
                    for di in (1, -1):
                        j = i + di
                        while 0 <= j < n and mat_sat[j, s:e].mean() >= 0.9:
                            thick += 1
                            j += di
                    if thick > max_thick:
                        continue
                    # контраст с обеих сторон
                    lo_i, hi_i = i - off, i + off
                    if lo_i < 0 or hi_i >= n:
                        continue  # линия у самого края — не считаем (поле экспорта)
                    line_val = float(mat_img[i, s:e].mean())
                    side_a = float(mat_img[lo_i, s:e].mean())
                    side_b = float(mat_img[hi_i, s:e].mean())
                    if line_val - side_a < side_contrast or line_val - side_b < side_contrast:
                        continue
                    segs.append((orient, i, s, e))

        _scan(sat, imgf, "h")
        _scan(sat.T, imgf.T, "v")
        total = 0
        for orient, i, s, e in segs:
            total += e - s
            if orient == "h":
                line_mask[i, s:e] = True
            else:
                line_mask[s:e, i] = True
        n_h = sum(1 for sgm in segs if sgm[0] == "h")
        n_v = len(segs) - n_h
        return {"has_lines": bool(segs), "n_segments": int(len(segs)), "n_horizontal": int(n_h),
                "n_vertical": int(n_v), "total_len_px": int(total),
                "frac_px": float(line_mask.mean()), "sat_frac": sat_frac, "segments": segs}
    except Exception as e:  # noqa: BLE001
        return {"has_lines": False, "n_segments": 0, "n_horizontal": 0, "n_vertical": 0,
                "total_len_px": 0, "frac_px": 0.0, "sat_frac": 0.0, "segments": [], "error": str(e)}


# --------------------------------------------------------------------------- #
# 2. Экспозиция
# --------------------------------------------------------------------------- #
def exposure_stats(img_u8: np.ndarray, body_thresh: int = 8) -> Dict[str, float]:
    """Медиана/перцентили пикселей тела (> body_thresh), доля тела, шум по лапласиану
    (медиана |Laplacian| внутри тела — устойчивее среднего), доля насыщенных."""
    img = np.asarray(img_u8)
    body = img > body_thresh
    px = img[body].astype(np.float32)
    if px.size < 50:
        px = img.astype(np.float32).ravel()
    lap = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
    lap_body = np.abs(lap[body]) if body.sum() > 50 else np.abs(lap).ravel()
    return {
        "body_frac": float(body.mean()),
        "body_p05": float(np.percentile(px, 5)),
        "body_p50": float(np.percentile(px, 50)),
        "body_p95": float(np.percentile(px, 95)),
        "body_mean": float(px.mean()),
        "body_std": float(px.std()),
        "lap_noise": float(np.median(lap_body)),
        "sat_frac": float((img >= 254).mean()),
    }


# --------------------------------------------------------------------------- #
# 3. Fingerprint DICOM-тегов
# --------------------------------------------------------------------------- #
FINGERPRINT_RULES = {
    "Manufacturer_contains": "GE",
    "ManufacturerModelName_contains_any": ("Lunar", "Prodigy"),
    "Columns_in": (300, 280, 248),
    "Rows_range": (150, 450),
    "BitsStored_eq": 8,
    "PhotometricInterpretation_eq": "MONOCHROME2",
    "SamplesPerPixel_eq": 1,
    "Modality_in": ("CR", "OT", "DX", "RG", ""),
}


def fingerprint(tags: Dict[str, Any]) -> Dict[str, Any]:
    """tags: словарь тегов (строки или числа). Отсутствующий текстовый тег -> статус 'incomplete'
    (не противоречие). Возвращает ok (нет противоречий), status ('ok'|'mismatch'|'incomplete'), reasons."""
    mism: List[str] = []
    missing: List[str] = []

    def _s(k):
        v = tags.get(k, None)
        return None if v is None or str(v).strip() == "" else str(v).strip()

    def _i(k):
        v = tags.get(k, None)
        try:
            return None if v is None or str(v).strip() == "" else int(float(str(v)))
        except Exception:  # noqa: BLE001
            return None

    man = _s("Manufacturer")
    if man is None:
        missing.append("Manufacturer")
    elif "GE" not in man.upper():
        mism.append(f"Manufacturer={man}")
    model = _s("ManufacturerModelName")
    if model is None:
        missing.append("ManufacturerModelName")
    elif not any(t.lower() in model.lower() for t in FINGERPRINT_RULES["ManufacturerModelName_contains_any"]):
        mism.append(f"Model={model}")
    cols, rows = _i("Columns"), _i("Rows")
    if cols is None or rows is None:
        missing.append("Rows/Columns")
    else:
        if cols not in FINGERPRINT_RULES["Columns_in"]:
            mism.append(f"Columns={cols}")
        lo, hi = FINGERPRINT_RULES["Rows_range"]
        if not (lo <= rows <= hi):
            mism.append(f"Rows={rows}")
    bs = _i("BitsStored")
    if bs is None:
        missing.append("BitsStored")
    elif bs != FINGERPRINT_RULES["BitsStored_eq"]:
        mism.append(f"BitsStored={bs}")
    ph = _s("PhotometricInterpretation")
    if ph is None:
        missing.append("PhotometricInterpretation")
    elif ph != FINGERPRINT_RULES["PhotometricInterpretation_eq"]:
        mism.append(f"Photometric={ph}")
    spp = _i("SamplesPerPixel")
    if spp is not None and spp != 1:
        mism.append(f"SamplesPerPixel={spp}")
    mod = _s("Modality")
    if mod is not None and mod not in FINGERPRINT_RULES["Modality_in"]:
        mism.append(f"Modality={mod}")

    status = "mismatch" if mism else ("incomplete" if missing else "ok")
    return {"ok": not mism, "status": status, "mismatch": mism, "missing": missing}


def tags_from_dataset(ds) -> Dict[str, Any]:
    out = {}
    for k in ("Manufacturer", "ManufacturerModelName", "Columns", "Rows", "BitsStored",
              "PhotometricInterpretation", "SamplesPerPixel", "Modality", "StudyDate", "PatientID",
              "StudyInstanceUID", "SOPInstanceUID", "SoftwareVersions"):
        try:
            v = getattr(ds, k, None)
            out[k] = None if v is None else str(v)
        except Exception:  # noqa: BLE001
            out[k] = None
    return out


# --------------------------------------------------------------------------- #
# 4. OOD-gate: Mahalanobis в PCA-пространстве
# --------------------------------------------------------------------------- #
class OODGate:
    """
    Mahalanobis-гейт по эмбеддингам imagenet (1280-d).
    Основная статистика: StandardScaler -> ковариация Ledoit-Wolf в полном 1280-d пространстве
    -> расстояние Махаланобиса (method='lw_full'). Вторичная (отчётная): StandardScaler -> PCA(k)
    -> Ledoit-Wolf -> T² в PCA-пространстве (по ТЗ совета; сама по себе слабее, т.к. OOD-энергия
    лежит в остаточном подпространстве — см. OOD_GATE_REPORT.md).
    Пороги = квантиль q OOF-расстояний (GroupKFold по study), для каждой статистики отдельно.
    """

    def __init__(self, k: int = 32, quantile: float = 0.99, seed: int = 0):
        self.k, self.quantile, self.seed = k, quantile, seed
        self.params: Optional[Dict[str, Any]] = None

    @staticmethod
    def _fit_params(X: np.ndarray, k: int) -> Dict[str, Any]:
        from sklearn.covariance import LedoitWolf
        from sklearn.decomposition import PCA
        mu0 = X.mean(0)
        sd0 = X.std(0) + 1e-6
        Z = (X - mu0) / sd0
        lw_full = LedoitWolf().fit(Z)
        pca = PCA(n_components=k, random_state=0).fit(Z)
        P = pca.transform(Z)
        lw = LedoitWolf().fit(P)
        return {"scaler_mean": mu0, "scaler_scale": sd0,
                "prec_full": lw_full.precision_.astype(np.float32), "shrinkage_full": float(lw_full.shrinkage_),
                "pca_components": pca.components_.astype(np.float32), "pca_mean": pca.mean_,
                "mu": P.mean(0), "cov_inv": lw.precision_, "shrinkage": float(lw.shrinkage_), "k": int(k)}

    @staticmethod
    def _dist(params: Dict[str, Any], X: np.ndarray) -> np.ndarray:
        """Основное расстояние: Ledoit-Wolf Mahalanobis в полном стандартизованном пространстве."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Z = (X - params["scaler_mean"]) / params["scaler_scale"]
        m2 = np.einsum("ij,jk,ik->i", Z, params["prec_full"].astype(np.float64), Z)
        return np.sqrt(np.maximum(m2, 0.0))

    @staticmethod
    def _dist_pca(params: Dict[str, Any], X: np.ndarray) -> np.ndarray:
        """Вторичное расстояние: T² (Mahalanobis) в PCA(k)-пространстве."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Z = (X - params["scaler_mean"]) / params["scaler_scale"]
        P = (Z - params["pca_mean"]) @ params["pca_components"].astype(np.float64).T
        D = P - params["mu"]
        m2 = np.einsum("ij,jk,ik->i", D, params["cov_inv"], D)
        return np.sqrt(np.maximum(m2, 0.0))

    def oof_distances(self, X: np.ndarray, groups: Sequence, n_splits: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """(oof_full, oof_pca): расстояния каждого кадра по параметрам, обученным без его study."""
        from sklearn.model_selection import GroupKFold
        X = np.asarray(X, dtype=np.float64)
        oof = np.full(len(X), np.nan); oof_p = np.full(len(X), np.nan)
        gkf = GroupKFold(n_splits=n_splits)
        for tr, te in gkf.split(X, groups=groups):
            p = self._fit_params(X[tr], self.k)
            oof[te] = self._dist(p, X[te]); oof_p[te] = self._dist_pca(p, X[te])
        return oof, oof_p

    def fit(self, X: np.ndarray, groups: Sequence, n_splits: int = 5,
            exposure: Optional[np.ndarray] = None, exposure_names: Optional[List[str]] = None) -> "OODGate":
        X = np.asarray(X, dtype=np.float64)
        oof, oof_p = self.oof_distances(X, groups, n_splits)
        thr = float(np.quantile(oof, self.quantile)); thr_p = float(np.quantile(oof_p, self.quantile))
        p = self._fit_params(X, self.k)
        p.update({"version": OOD_GATE_PKL_VERSION, "method": "lw_full", "threshold": thr, "threshold_pca": thr_p,
                  "quantile": self.quantile, "n_train": int(len(X)), "n_splits": int(n_splits),
                  "oof_distances": oof.astype(np.float32), "oof_distances_pca": oof_p.astype(np.float32),
                  "oof_fpr": float((oof > thr).mean()), "oof_fpr_pca": float((oof_p > thr_p).mean()),
                  "insample_distances": self._dist(p, X).astype(np.float32),
                  "emb_source": "imagenet", "emb_dim": int(X.shape[1])})
        if exposure is not None:
            # пороги экспозиции: двусторонние 0.5 % / 99.5 % квантили (суммарно ~1 % по каждому признаку)
            E = np.asarray(exposure, dtype=np.float64)
            p["exposure_names"] = list(exposure_names or [f"e{i}" for i in range(E.shape[1])])
            p["exposure_lo"] = np.quantile(E, 0.005, axis=0)
            p["exposure_hi"] = np.quantile(E, 0.995, axis=0)
        self.params = p
        return self

    def distance(self, emb: np.ndarray) -> float:
        assert self.params is not None
        return float(self._dist(self.params, np.asarray(emb).reshape(1, -1))[0])

    def distance_pca(self, emb: np.ndarray) -> float:
        assert self.params is not None
        return float(self._dist_pca(self.params, np.asarray(emb).reshape(1, -1))[0])

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self.params, f)

    @classmethod
    def load(cls, path: Path) -> "OODGate":
        with open(path, "rb") as f:
            p = pickle.load(f)
        if p.get("version") != OOD_GATE_PKL_VERSION:
            raise ValueError(f"ood_gate.pkl version mismatch: {p.get('version')}")
        g = cls(k=int(p["k"]), quantile=float(p["quantile"]))
        g.params = p
        return g


EXPOSURE_KEYS = ["body_frac", "body_p05", "body_p50", "body_p95", "lap_noise"]

_GATE_CACHE: Dict[str, OODGate] = {}


def load_ood_gate(path: Optional[Path] = None) -> Optional[OODGate]:
    p = Path(path) if path else Path(__file__).resolve().parent.parent / "models" / "ood_gate.pkl"
    key = str(p)
    if key not in _GATE_CACHE:
        try:
            _GATE_CACHE[key] = OODGate.load(p)
        except Exception:  # noqa: BLE001
            return None
    return _GATE_CACHE[key]


def ood_score(emb: Optional[np.ndarray], tags: Optional[Dict[str, Any]], gate: Optional[OODGate] = None,
              exposure: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Итоговое правило: ood_flag = fingerprint mismatch OR mahalanobis > threshold.
    Экспозиция вне 0.5–99.5 % — отдельный мягкий признак (в reason, не в flag).
    Отсутствие тегов (PNG, теги удалены) -> fingerprint_ok=True со статусом 'incomplete' в reason."""
    gate = gate or load_ood_gate()
    reasons: List[str] = []
    fp = fingerprint(tags or {})
    fp_ok = bool(fp["ok"])
    if fp["mismatch"]:
        reasons.append("fingerprint: " + ", ".join(fp["mismatch"]))
    elif fp["missing"]:
        reasons.append("tags incomplete: " + ", ".join(fp["missing"]))
    maha = None
    maha_pca = None
    maha_flag = False
    if gate is not None and emb is not None:
        try:
            maha = gate.distance(emb)
            maha_pca = gate.distance_pca(emb)
            maha_flag = maha > float(gate.params["threshold"])
            if maha_flag:
                reasons.append(f"mahalanobis {maha:.1f} > {gate.params['threshold']:.1f}")
        except Exception as e:  # noqa: BLE001
            reasons.append(f"mahalanobis error: {e}")
    elif gate is None:
        reasons.append("ood_gate.pkl not loaded")
    else:
        reasons.append("no embedding")
    exp_flag = False
    if exposure and gate is not None and "exposure_lo" in gate.params:
        names = gate.params["exposure_names"]
        vals = np.array([exposure.get(n, np.nan) for n in names], dtype=float)
        lo, hi = gate.params["exposure_lo"], gate.params["exposure_hi"]
        bad = [n for n, v, a, b in zip(names, vals, lo, hi) if np.isfinite(v) and (v < a or v > b)]
        if bad:
            exp_flag = True
            reasons.append("exposure out of range: " + ", ".join(bad))
    flag = (not fp_ok) or maha_flag
    return {"ood_flag": bool(flag), "mahalanobis": None if maha is None else round(float(maha), 3),
            "mahalanobis_pca": None if maha_pca is None else round(float(maha_pca), 3),
            "fingerprint_ok": fp_ok, "fingerprint_status": fp["status"], "mahalanobis_flag": bool(maha_flag),
            "exposure_flag": bool(exp_flag), "reason": "; ".join(reasons) if reasons else "ok"}


# --------------------------------------------------------------------------- #
# 5. Эндопротез
# --------------------------------------------------------------------------- #
# Пороги выбраны на 499 (work/D/out/EXTRAS_STATUS.md, п. 3): максимум среди 331 не-протезных бёдер —
# полутолщина 9.7 px, площадь 830 px; реальный протез (#82/#83) — 33 px, 10 472–12 010 px.
# Валидировано на 1 положительном исследовании из 2 (второе, #40/#45, имеет иную GE-сигнатуру: тёмное кольцо вокруг металла).
ENDO_DEFAULTS = {"dense_level": 250, "min_area_px": 2000, "min_half_thick_px": 12.0, "min_extent_rows": 40}


def endoprosthesis(img_u8: np.ndarray, region: str = "hip", params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Правило «эндопротез»: очень плотные пиксели (>= dense_level после нормализации) образуют
    СПЛОШНОЙ компонент (полутолщина по distance transform >= min_half_thick_px — кортикальный
    гребень насыщается лентой 1–3 px, металл — сплошным телом) достаточной площади, протяжённый
    по вертикали (оси диафиза) >= min_extent_rows. Для позвоночника не применяется (флаг False).
    Возвращает метрики самого «толстого» компонента и флаг.
    """
    p = dict(ENDO_DEFAULTS)
    if params:
        p.update(params)
    out = {"endoprosthesis_suspected": False, "dense_area_px": 0, "max_half_thick_px": 0.0,
           "extent_rows": 0, "n_dense_components": 0}
    try:
        img = np.asarray(img_u8)
        dense = (img >= p["dense_level"]).astype(np.uint8)
        dense = cv2.morphologyEx(dense, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(dense, connectivity=8)
        if n <= 1:
            return out
        dt = cv2.distanceTransform(dense, cv2.DIST_L2, 3)
        best = None
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 20:
                continue
            comp = lab == i
            half = float(dt[comp].max())
            ext = int(stats[i, cv2.CC_STAT_HEIGHT])
            cand = (half, area, ext)
            if best is None or cand > best:
                best = cand
        out["n_dense_components"] = int(n - 1)
        if best is None:
            return out
        half, area, ext = best
        out.update({"dense_area_px": area, "max_half_thick_px": round(half, 2), "extent_rows": ext})
        if not str(region).endswith("hip") and region != "hip":
            return out
        out["endoprosthesis_suspected"] = bool(area >= p["min_area_px"] and half >= p["min_half_thick_px"]
                                               and ext >= p["min_extent_rows"])
        return out
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)
        return out


# --------------------------------------------------------------------------- #
# 6. Study coherence
# --------------------------------------------------------------------------- #
def pixel_hash(ds_or_array) -> str:
    try:
        arr = ds_or_array.pixel_array if hasattr(ds_or_array, "pixel_array") else np.asarray(ds_or_array)
        return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return ""


def _short_hash(s: Optional[str]) -> str:
    return "" if not s else hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:10]


def study_coherence(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    rows: словари с ключами study_uid, image_uid, region ('spine'|'right_hip'|'left_hip'|...),
    pixel_hash, study_date, patient_hash (любой может отсутствовать).
    Возвращает {study_uid: [предупреждения]}. Уровень — warning, на CSV не влияет.
    """
    by_study: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_study[str(r.get("study_uid", ""))].append(r)
    res: Dict[str, List[str]] = {}
    for suid, rs in by_study.items():
        w: List[str] = []
        regions = Counter(str(r.get("region", "")) for r in rs)
        for reg, lim in (("spine", 1), ("right_hip", 1), ("left_hip", 1)):
            if regions.get(reg, 0) > lim:
                w.append(f"{reg} x{regions[reg]} (ожидается <= {lim})")
        if regions.get("spine", 0) == 0:
            w.append("нет позвоночника")
        if regions.get("right_hip", 0) + regions.get("left_hip", 0) == 0:
            w.append("нет бедра")
        hashes = Counter(str(r.get("pixel_hash", "")) for r in rs if r.get("pixel_hash"))
        dups = {h: c for h, c in hashes.items() if c > 1}
        if dups:
            w.append(f"дубликаты кадров (pixel hash): {sum(c - 1 for c in dups.values())} лишних, {len(dups)} групп")
        dates = {str(r.get("study_date", "")) for r in rs if r.get("study_date")}
        if len(dates) > 1:
            w.append(f"разные StudyDate: {len(dates)}")
        pats = {str(r.get("patient_hash", "")) for r in rs if r.get("patient_hash")}
        if len(pats) > 1:
            w.append(f"разные PatientID: {len(pats)}")
        res[suid] = w
    return res


# --------------------------------------------------------------------------- #
# 7. Сборка results_extras.csv
# --------------------------------------------------------------------------- #
def compute_file_extras(img_u8: Optional[np.ndarray], emb: Optional[np.ndarray], tags: Dict[str, Any],
                        region: str, gate: Optional[OODGate] = None) -> Dict[str, Any]:
    """Все пофайловые флаги (без study-уровня)."""
    out: Dict[str, Any] = {}
    if img_u8 is None:
        out.update({"white_lines_flag": False, "white_lines_len_px": 0, "endoprosthesis_suspected": False})
        exp = None
    else:
        wl = white_lines(img_u8)
        out["white_lines_flag"] = bool(wl["has_lines"])
        out["white_lines_len_px"] = int(wl["total_len_px"])
        out["white_lines_n_segments"] = int(wl["n_segments"])
        ep = endoprosthesis(img_u8, region)
        out["endoprosthesis_suspected"] = bool(ep["endoprosthesis_suspected"])
        out["endo_dense_area_px"] = ep["dense_area_px"]
        out["endo_half_thick_px"] = ep["max_half_thick_px"]
        exp = exposure_stats(img_u8)
        out.update({f"exp_{k}": round(v, 4) for k, v in exp.items()})
    o = ood_score(emb, tags, gate=gate, exposure=exp)
    out.update({"ood_flag": o["ood_flag"], "ood_mahalanobis": o["mahalanobis"],
                "ood_fingerprint_ok": o["fingerprint_ok"], "ood_fingerprint_status": o["fingerprint_status"],
                "ood_mahalanobis_pca32": o["mahalanobis_pca"],
                "ood_exposure_flag": o["exposure_flag"], "ood_reason": o["reason"]})
    return out


def compute_extras_for_rows(rows: List[Dict[str, Any]], debug_rows: List[Dict[str, Any]],
                            images: List[Optional[Dict[str, Any]]], gate: Optional[OODGate] = None,
                            models_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    rows       — строки основного CSV (9 колонок) в порядке файлов;
    debug_rows — соответствующие debug-словари (internal_region и т.п.);
    images     — по файлу: {'img_u8', 'emb', 'tags', 'pixel_hash'} или None (Failure).
    Возвращает строки results_extras.csv (EXTRAS_COLUMNS + служебные поля extra_*).
    """
    gate = gate or load_ood_gate(Path(models_dir) / "ood_gate.pkl" if models_dir else None)
    per_file: List[Dict[str, Any]] = []
    coh_input: List[Dict[str, Any]] = []
    for row, dbg, im in zip(rows, debug_rows, images):
        region = str((dbg or {}).get("internal_region", ""))
        im = im or {}
        tags = im.get("tags") or {}
        fx = compute_file_extras(im.get("img_u8"), im.get("emb"), tags, region, gate=gate)
        try:
            fx["axis_curved_flag"] = bool(region == "spine" and float((dbg or {}).get("feat_curvature")) > AXIS_CURVATURE_MAX_PX)
        except (TypeError, ValueError):
            fx["axis_curved_flag"] = False
        fx["file"] = row.get("path_to_study", "")
        fx["image_uid"] = row.get("image_uid", "")
        per_file.append(fx)
        coh_input.append({"study_uid": row.get("study_uid", ""), "image_uid": row.get("image_uid", ""),
                          "region": region, "pixel_hash": im.get("pixel_hash", ""),
                          "study_date": tags.get("StudyDate"), "patient_hash": _short_hash(tags.get("PatientID"))})
    coh = study_coherence(coh_input)
    out: List[Dict[str, Any]] = []
    for row, fx in zip(rows, per_file):
        fx["study_warnings"] = " | ".join(coh.get(str(row.get("study_uid", "")), []))
        ordered = {k: fx.get(k) for k in EXTRAS_COLUMNS}
        ordered.update({k: v for k, v in fx.items() if k not in EXTRAS_COLUMNS})
        out.append(ordered)
    return out


def write_extras_csv(extras_rows: List[Dict[str, Any]], path: Path) -> None:
    import pandas as pd
    df = pd.DataFrame(extras_rows)
    cols = [c for c in EXTRAS_COLUMNS if c in df.columns] + [c for c in df.columns if c not in EXTRAS_COLUMNS]
    df = df.reindex(columns=cols) if len(df) else pd.DataFrame(columns=EXTRAS_COLUMNS)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
