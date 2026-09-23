"""
H2 (2.4.0): признак контура A `synth_pos_logit` для критерия sp_pos (укладка позвоночника).

Что это. Голова «дефект укладки» (`models/head_densito_synth.pth`) — небольшой MLP поверх замороженного
бэкбона densito (EfficientNet-B0, `models/backbone_densito.pth`). Голова обучена на GPU на синтетических
смещениях кадра без меток разметки (work/gpu_sppos/gpu/sppos_synth_train.py, голова head v3, зерно 0):
на вход подаётся канонический 1280-мерный эмбеддинг кадра (тот же, что использует контур B критерия
sp_pos: вариант предобработки `canonical`, источник `densito`, препроцессинг 320x192 из src/embeddings.py),
на выходе 4 числа; первое — логит «кадр смещён». Именно этот логит (`out[:, 0]`) и есть признак
`synth_pos_logit`; он подаётся третьим признаком в геометрическую модель контура A sp_pos
(`models/model_spine_sp_pos_geom.pkl`, StandardScaler -> LogReg) вместе с center_offset_ratio и
bone_width_ratio. Принят по действующему правилу nested (ΔAUC >= 0.03 в 20/20 повторов);
калиброванный гейт при 6 положительных исследованиях мощности не имеет — см. docs/NESTED_GATE_REPORT.md, часть 5.

Стоимость. Эмбеддинг densito канонического кадра инференс уже считает для контура B sp_pos; голова —
два матричных умножения 1280x256 и 256x4, время на снимок не меняется заметно.

Детерминизм. CPU, fp32, без dropout (eval), фиксированное число потоков (как у остальных моделей).
Отсутствие файла головы — явная ошибка (FileNotFoundError с пояснением), а не молчаливый ноль.

Использование:
    from sppos_head import SynthPosHead, FEATURE_NAME
    head = SynthPosHead(models_dir / "head_densito_synth.pth")
    logit = head.logit(emb1280)              # float
    logits = head.logit_many(E)              # (n,) для матрицы (n, 1280)

Сверка без torch (numpy): `SynthPosHead.load_weights_numpy(path)` + `logit_numpy(weights, E)` — читает
те же тензоры напрямую из zip-контейнера torch.save; используется тестом tests/test_sppos_head.py и
tools/add_sppos_head_feature.py --backend numpy в песочнице без torch. Оба пути дают одинаковые числа
(с точностью fp32 -> fp64, |Δ| < 1e-5 по логиту).
"""
from __future__ import annotations

import io
import os
import pickletools
import zipfile
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

FEATURE_NAME = "synth_pos_logit"
HEAD_FILE = "head_densito_synth.pth"
# на каком эмбеддинге определена голова: вариант предобработки и источник бэкбона
EMB_VARIANT = "canonical"
EMB_SOURCE = "densito"
EMB_DIM = 1280
HIDDEN = 256
N_OUT = 4
# критерии, для которых признак считается (только позвоночник)
CRITERIA = ("sp_pos",)
REGIONS = ("spine",)

# ключи state_dict головы: nn.Sequential(Dropout(0.2), Linear(1280, 256), GELU(), Linear(256, 4))
STATE_KEYS = ("1.weight", "1.bias", "3.weight", "3.bias")
STATE_SHAPES = {"1.weight": (HIDDEN, EMB_DIM), "1.bias": (HIDDEN,), "3.weight": (N_OUT, HIDDEN), "3.bias": (N_OUT,)}


class HeadMissingError(FileNotFoundError):
    """Файла головы нет: критерий sp_pos не может быть посчитан по контракту 2.4.0."""


def head_path(models_dir: Optional[Path] = None) -> Path:
    """models/head_densito_synth.pth; каталог моделей — как у инференса (DENSITO_MODELS_DIR)."""
    if models_dir is None:
        env = os.environ.get("DENSITO_MODELS_DIR")
        models_dir = Path(env) if env else Path(__file__).resolve().parent.parent / "models"
    return Path(models_dir) / HEAD_FILE


def _missing_message(path: Path) -> str:
    return (f"Файл головы укладки не найден: {path}. Признак контура A `{FEATURE_NAME}` критерия sp_pos "
            f"(версия 2.4.0) требует models/{HEAD_FILE}; без него критерий sp_pos не считается. "
            f"Проверьте поставку: `python tools/hash_weights.py --check` и models/models_manifest.json.")


def _check_state(sd: Dict[str, np.ndarray], path: Path) -> None:
    missing = [k for k in STATE_KEYS if k not in sd]
    if missing:
        raise RuntimeError(f"{path}: в state_dict головы нет ключей {missing} (ожидались {list(STATE_KEYS)})")
    for k, shape in STATE_SHAPES.items():
        got = tuple(int(x) for x in np.shape(sd[k]))
        if got != shape:
            raise RuntimeError(f"{path}: тензор {k} имеет форму {got}, ожидалась {shape}")


class SynthPosHead:
    """Голова «дефект укладки» на torch (боевой путь инференса)."""

    def __init__(self, path: Optional[Path] = None, num_threads: Optional[int] = None):
        self.path = Path(path) if path is not None else head_path()
        if not self.path.exists():
            raise HeadMissingError(_missing_message(self.path))
        import torch
        import torch.nn as nn
        torch.manual_seed(0)
        torch.set_num_threads(num_threads if num_threads else max(1, min(8, os.cpu_count() or 1)))
        sd = torch.load(self.path, map_location="cpu")
        if not isinstance(sd, dict):
            raise RuntimeError(f"{self.path}: ожидался state_dict (dict), получен {type(sd).__name__}")
        _check_state({k: v.numpy() for k, v in sd.items()}, self.path)
        self.model = nn.Sequential(nn.Dropout(0.2), nn.Linear(EMB_DIM, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, N_OUT))
        self.model.load_state_dict({k: sd[k].float() for k in STATE_KEYS}, strict=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self._torch = torch

    def forward(self, E: np.ndarray) -> np.ndarray:
        """(n, 1280) -> (n, 4): [логит дефекта, смещение x, смещение y, масштаб] (сырые выходы)."""
        E = np.asarray(E, dtype=np.float32)
        if E.ndim == 1:
            E = E[None, :]
        if E.shape[1] != EMB_DIM:
            raise ValueError(f"ожидался эмбеддинг размерности {EMB_DIM}, получено {E.shape}")
        with self._torch.no_grad():
            out = self.model(self._torch.from_numpy(np.ascontiguousarray(E)))
        return out.cpu().numpy().astype(np.float64)

    def logit_many(self, E: np.ndarray) -> np.ndarray:
        return self.forward(E)[:, 0]

    def logit(self, emb: np.ndarray) -> float:
        """Логит «дефект укладки» для одного эмбеддинга (float; NaN невозможен при конечном входе)."""
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim != 1 or emb.shape[0] != EMB_DIM or not np.all(np.isfinite(emb)):
            raise ValueError(f"эмбеддинг должен быть конечным вектором длины {EMB_DIM}, получено {emb.shape}")
        return float(self.forward(emb)[0, 0])

    # ---- путь без torch (сверка, тесты, песочница)
    @staticmethod
    def load_weights_numpy(path: Optional[Path] = None) -> Dict[str, np.ndarray]:
        """Читает тензоры головы из zip-контейнера torch.save без torch (fp32 little-endian).
        Разбирает data.pkl только на уровне опкодов pickle (без выполнения), сопоставляя ключ -> storage -> форма."""
        path = Path(path) if path is not None else head_path()
        if not path.exists():
            raise HeadMissingError(_missing_message(path))
        with zipfile.ZipFile(path) as z:
            pkl_name = next(n for n in z.namelist() if n.endswith("/data.pkl"))
            prefix = pkl_name[: -len("/data.pkl")]
            ops = [(op.name, arg) for op, arg, _ in pickletools.genops(io.BytesIO(z.read(pkl_name)))]
            # шаблон записи: BINUNICODE <ключ> ... GLOBAL torch FloatStorage, BINUNICODE <storage_id>, ...
            #                BINPERSID, BININT <offset>, <shape ints>, TUPLEk (shape), <stride ints>, TUPLEk (stride)
            out: Dict[str, np.ndarray] = {}
            i = 0
            while i < len(ops):
                name, arg = ops[i]
                if name == "BINUNICODE" and arg in STATE_KEYS:
                    key = arg
                    # storage id — единственная строка между ключом и BINPERSID, кроме 'storage'/'cpu'
                    # (те после первого тензора идут через BINGET и в строках не повторяются)
                    j = i + 1
                    strings = []
                    while ops[j][0] != "BINPERSID":
                        if ops[j][0] in ("BINUNICODE", "SHORT_BINUNICODE") and ops[j][1] not in ("storage", "cpu"):
                            strings.append(str(ops[j][1]))
                        j += 1
                    if len(strings) != 1:
                        raise RuntimeError(f"{path}: не удалось определить storage для {key}: {strings}")
                    storage_id = strings[0]
                    # после BINPERSID: offset, затем целые формы до TUPLE*
                    j += 1
                    ints = []
                    while not ops[j][0].startswith("TUPLE") and ops[j][0] != "EMPTY_TUPLE":
                        if ops[j][0].startswith("BININT"):
                            ints.append(int(ops[j][1]))
                        j += 1
                    offset, shape = ints[0], tuple(ints[1:])
                    raw = z.read(f"{prefix}/data/{storage_id}")
                    arr = np.frombuffer(raw, dtype="<f4")
                    n = int(np.prod(shape)) if shape else 1
                    out[key] = arr[offset: offset + n].reshape(shape).astype(np.float64)
                    i = j
                i += 1
        _check_state(out, path)
        return out


def logit_numpy(weights: Dict[str, np.ndarray], E: np.ndarray) -> np.ndarray:
    """Тот же прямой проход в numpy (fp64): Linear -> GELU (точная, через erf) -> Linear, столбец 0."""
    from scipy.special import erf
    E = np.asarray(E, dtype=np.float64)
    if E.ndim == 1:
        E = E[None, :]
    h = E @ weights["1.weight"].T + weights["1.bias"]
    h = 0.5 * h * (1.0 + erf(h / np.sqrt(2.0)))
    out = h @ weights["3.weight"].T + weights["3.bias"]
    return out[:, 0]


def logit_to_prob(logit: np.ndarray) -> np.ndarray:
    """sigmoid — для сверки с p_defect в work/gpu_sppos/outputs/scores_synth_head_v3.csv."""
    x = np.asarray(logit, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def needs_head(feature_cols: Sequence[str]) -> bool:
    return FEATURE_NAME in set(feature_cols or [])
