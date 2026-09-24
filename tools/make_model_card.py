#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор карточки модели: models/MODEL_CARD.md из фактических файлов проекта.

Источники (только они, никаких чисел «из головы»):
  models/metrics_summary.json   — OOF-метрики по критериям, пороги, ДИ, n_pos
  models/models_manifest.json   — список .pkl, признаки, источник эмбеддингов
  config.yaml                   — версия, строки регионов/нарушений, правило стекинга
  requirements.txt              — версии ключевых библиотек
  models/*.pkl, backbone        — sha256 (первые 12 символов)

Nested-оценка: если в metrics_summary.json (на верхнем уровне или в блоке региона) есть
ключ ``nested_auc_mean`` (опционально ``nested_auc_std``, ``nested_note``) — печатается,
иначе строка «nested: не рассчитано». Куда вставлять: metrics_summary.json ->
{"nested": {"nested_auc_mean": ..., "nested_auc_std": ..., "nested_note": "5x5 nested GroupKFold, ..."}}
или в каждый блок критерия (``spine.sp_pos.nested_auc_mean``).

Запуск: python tools/make_model_card.py [--out models/MODEL_CARD.md]
"""
import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

try:
    from inference import config_hash as _config_hash  # noqa: E402
except Exception:  # noqa: BLE001
    def _config_hash(cfg):
        return hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:12]

CRIT_TITLE = {
    "sp_pos": "Позвоночник: укладка (центр, симметрия)",
    "sp_axis": "Позвоночник: ось позвоночника",
    "sp_art": "Позвоночник: посторонние предметы",
    "hip_pos": "Бедро (обе стороны, общая модель): укладка",
    "hip_roi": "Бедро (обе стороны, общая модель): область интереса",
    "rh_pos": "Правое бедро: укладка",
    "rh_roi": "Правое бедро: область интереса",
    "lh_pos": "Левое бедро: укладка",
    "lh_roi": "Левое бедро: область интереса",
}
KEY_LIBS = ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "pandas", "pydicom",
            "opencv-python-headless", "scikit-image", "PyYAML", "fastapi")


def sha12(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def f3(x, nd=3):
    return "—" if x is None else f"{float(x):.{nd}f}"


def nested_line(block: dict) -> str:
    # оценка поставки: nested_auc_production; nested_auc_mean — синоним для старых файлов (A2: до 24.09 у sp_art,
    # hip_pos, hip_roi в nested_auc_mean стояла отвергнутая альтернатива, поэтому production — первым)
    val = block.get("nested_auc_production", block.get("nested_auc_mean")) if isinstance(block, dict) else None
    if val is not None:
        s = f"nested AUC = {f3(val)}"
        if block.get("nested_auc_std") is not None:
            s += f" ± {f3(block['nested_auc_std'])}"
        if block.get("nested_note"):
            s += f" ({block['nested_note']})"
        return s
    return "nested: не рассчитано"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "models" / "MODEL_CARD.md"))
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args(argv)
    root = Path(a.root)

    cfg = yaml.safe_load(open(root / "config.yaml", encoding="utf-8"))
    metrics = json.load(open(root / "models" / "metrics_summary.json", encoding="utf-8"))
    manifest = json.load(open(root / "models" / "models_manifest.json", encoding="utf-8"))
    req = {}
    for line in open(root / "requirements.txt", encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "==" in line:
            k, v = line.split("==", 1)
            req[k.strip()] = v.strip()

    version = str(cfg.get("version", "?"))
    chash = _config_hash(cfg)
    st = cfg.get("stacking", {})
    thr_cfg = cfg.get("thresholds", {})

    L = []
    w = L.append
    w(f"# Карточка модели DensitoAI v{version}")
    w("")
    w(f"Сформировано автоматически `tools/make_model_card.py` {date.today().isoformat()} из "
      "`models/metrics_summary.json`, `models/models_manifest.json`, `config.yaml`, `requirements.txt`. "
      "Ручные правки не вносить — перегенерировать.")
    w("")
    w("## 1. Назначение")
    w("")
    w("Автоматическая оценка качества денситометрических снимков (DXA, GE Lunar Prodigy) двух областей: "
      f"«{cfg['regions']['spine']}» и «{cfg['regions']['hip']}». Для каждого снимка выдаётся quality_class (0 — норма, "
      "1 — есть нарушение), закрытый список нарушений и quality_prob. Инструмент поддержки контроля качества "
      "укладки; не является медицинским изделием и не ставит диагноз. Решение принимает оператор/врач.")
    w("")
    w('**Лицензии.** Исследовательский прототип: бэкбоны `sp_pos` (`densito`) и `sp_axis` (`densito_inv`) дообучены на пуле с некоммерческими лицензиями (Arak — CC BY-NC, BUU-LSPINE — EULA). Для передачи заказчику — вариант `densito_inv_free` (`models/backbone_densito_inv_free.pth`, без этих наборов), цена измерена на nested-протоколе: `sp_axis` AUC стэка 0.860 → 0.773 (`docs/EMB_GATE_REPORT.md`); для `sp_pos` `densito_inv_free` не измерен, замена на `imagenet` стоила 0.759 → 0.672 (К13, до H2).')
    w("")
    w("Официальные строки нарушений (config.yaml → violations): " +
      "; ".join(sorted(set(cfg["violations"].values()))) + ".")
    w("")
    w("## 2. Архитектура")
    w("")
    w("Двухконтурный стекинг для каждого критерия (см. `src/inference.py`, `models/MODEL_CONTRACT.md`):")
    w("")
    w("- контур A — геометрические признаки сегментированной кости (логистическая регрессия, `model_<регион>_<критерий>_geom.pkl`);")
    w("- контур B — эмбеддинги замороженного EfficientNet-B0 → PCA → логистическая регрессия (`model_..._emb_pca.pkl`). "
      "Источник эмбеддингов выбран по критерию (К13, nested-протокол): `imagenet` — веса torchvision (BSD-3-Clause); "
      "`densito` — `models/backbone_densito.pth`, дообучен на пуле внешних DXA/рентген-наборов; `densito_inv` — "
      "`models/backbone_densito_inv.pth`, инвариантный вариант того же пула. В пул входят Arak (CC BY-NC) и "
      "BUU-LSPINE (некоммерческое EULA), поэтому критерии на `densito`/`densito_inv` (`sp_pos`, `sp_axis`) "
      "ограничены исследовательским использованием; `sp_art`, `hip_pos`, `hip_roi` работают на `imagenet` и "
      "ограничений не имеют. Разбор и варианты — `docs/LICENSES_AND_DATA_AUDIT.md`, свободный по лицензиям "
      "`backbone_densito_inv_free.pth` в образе тоже есть;")
    w(f"- объединение: ранговое усреднение перцентилей относительно OOF-распределения с весами geom={st.get('weight_geom')}, "
      f"emb={st.get('weight_emb')}; quality_prob = {st.get('any_blend_weight_model')}·any-модель + "
      f"{1 - float(st.get('any_blend_weight_model', 0.5)):.1f}·{st.get('any_violation_aggregation')} по критериям; "
      f"consistent_quality_prob={st.get('consistent_quality_prob')}.")
    w("- область снимка: правило по ширине (≥ {0} px — позвоночник), для нестандартных ширин — `model_region_emb.pkl`.".format(
        cfg["regions"].get("spine_min_cols")))
    w("")
    w("Признаки контура A по фактически обученным моделям (models_manifest.json → feature_cols):")
    w("")
    for name, info in manifest.items():
        if name.endswith("_geom.pkl") and info.get("feature_cols"):
            w(f"- {info.get('criterion', name)} ({name}): {', '.join(info['feature_cols'])}")
    w("")
    w("## 3. Данные")
    w("")
    w("- Обучение и валидация: только DICOM организаторов (набор для обучения; см. `docs/EVIDENCE.md`), "
      "разметка — xlsx организаторов, один разметчик. Файлы конфиденциальны и в репозиторий не входят.")
    n_sp = metrics.get("spine", {}).get("sp_pos", {}).get("n_valid")
    n_hip = metrics.get("hip", {}).get("hip_pos", {}).get("n_valid")
    w(f"- Валидных снимков в OOF-оценке: позвоночник {n_sp}, бедро {n_hip} (metrics_summary.json → n_valid).")
    w("- Внешние наборы использованы только для предобучения бэкбона `densito` и OOD-теста; в обучении классификаторов "
      "и в репозитории их нет (`docs/GPU_EXPERIMENT.md`, `docs/DATASETS_DEEP_SEARCH.md`).")
    w("")
    w("## 4. Протокол оценки")
    w("")
    w("Out-of-fold (OOF) предсказания: повторный GroupKFold с группировкой по исследованию (study_uid), "
      "бутстрап доверительных интервалов по исследованиям; пороги подбираются только по OOF "
      "(threshold_method в таблице). Подробности и отвергнутые варианты — `docs/EVIDENCE.md`, `docs/METRICS_REPORT.md`.")
    w("")
    w("## 5. Метрики по критериям (OOF)")
    w("")
    w("| Критерий | Назначение | n_valid | n_pos | AUC geom | AUC emb (источник) | AUC стек | Порог (метод) | F1 OOF [95% ДИ] | Примечание |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    order = [("spine", "sp_pos"), ("spine", "sp_axis"), ("spine", "sp_art"), ("hip", "hip_pos"), ("hip", "hip_roi"),
             ("right_hip", "rh_pos"), ("right_hip", "rh_roi"), ("left_hip", "lh_pos"), ("left_hip", "lh_roi")]
    for region, crit in order:
        m = metrics.get(region, {}).get(crit)
        if not m:
            continue
        note = m.get("note", "")
        if note == "ok":
            note = ""
        if m.get("n_pos", 0) < 10 and "ненадёжно" not in note:
            note = (note + "; " if note else "") + "ненадёжно: <10 позитивов"
        w(f"| {crit} | {CRIT_TITLE.get(crit, crit)} | {m.get('n_valid')} | {m.get('n_pos')} | {f3(m.get('auc_geom'))} | "
          f"{f3(m.get('auc_emb'))} ({m.get('emb_source') or 'общая hip-модель'}) | {f3(m.get('auc_stacked'))} | "
          f"{f3(m.get('threshold'))} ({m.get('threshold_method', '—')}) | "
          f"{f3(m.get('f1_oof'))} [{f3(m.get('f1_ci_lo'), 2)}; {f3(m.get('f1_ci_hi'), 2)}] | {note} |")
    w("")
    w("ДИ — бутстрап по исследованиям, 95 %. F1 считается при пороге из колонки «Порог». "
      "Критерии с n_pos < 10 (sp_pos на границе, rh_roi/lh_roi) — оценки неустойчивы: ДИ F1 включает 0.")
    w("")
    w("Nested-оценка (порог и стекинг подобраны внутри внешних фолдов):")
    w("")
    w("Протокол: repeated GroupKFold 5 внешних фолдов × 10 повторов, 3 внутренних; группы — исследование + хэш пикселей; "
      "вес стэкинга и порог выбираются только на внутренних фолдах (`tools/nested_gate.py`, `docs/NESTED_GATE_REPORT.md`). "
      "AUC — среднее по 10 повторам для базового стэкинга 0.5/0.5 (он и используется); полные таблицы с ДИ — в отчёте.")
    w("")
    any_nested = False
    for region, crit in order:
        m = metrics.get(region, {}).get(crit)
        if m and (m.get("nested_auc_production") or m.get("nested_auc_base_mean")) is not None:
            any_nested = True
            # печатаем оценку того, что стоит в продакшене (К13 сменил источник эмбеддингов для sp_axis,
            # и для него nested-оценка берётся из emb_gate, а не из К2)
            prod = m.get("nested_auc_production", m.get("nested_auc_base_mean"))
            src = f", протокол {m['nested_protocol']}" if m.get("nested_protocol") else ""
            w(f"- {crit}: nested AUC = {f3(prod)}; OOF AUC = {f3(m.get('auc_stacked'))}{src}")
    if not any_nested:
        w("- nested: не рассчитано")
    w("")
    w("## 6. Пороги и правило решения")
    w("")
    w("- Порог критерия: `config.yaml → thresholds.<критерий>`; если null — из `metrics_summary.json → threshold`; "
      f"если нет и там — fallback_threshold = {cfg.get('fallback_threshold')}.")
    w("- Сейчас в config.yaml: " + ", ".join(f"{k}={'из metrics_summary' if v is None else v}" for k, v in thr_cfg.items()) + ".")
    w("- Правило: критерий срабатывает, если стек-скор ≥ порога; quality_class = 1, если сработал хотя бы один критерий области; "
      "violation_type — официальные строки сработавших критериев через «;». quality_prob согласован с классом "
      "(class 1 → [0.5; 1], class 0 → [0; 0.5)).")
    w("- Ошибка чтения файла → строка Failure с quality_class 0, пустым violation_type и quality_prob = "
      f"{cfg['output'].get('fallback_quality_prob')}.")
    w("")
    w("## 7. Ограничения")
    w("")
    w("- Мало позитивов: sp_pos n_pos = {0}, rh_roi/lh_roi n_pos = {1}/{2} — ДИ широкие, метрики по этим критериям ориентировочные.".format(
        metrics.get("spine", {}).get("sp_pos", {}).get("n_pos"),
        metrics.get("right_hip", {}).get("rh_roi", {}).get("n_pos"),
        metrics.get("left_hip", {}).get("lh_roi", {}).get("n_pos")))
    w("- Один разметчик, один прибор (GE Lunar Prodigy), одна организация — переносимость на другие приборы не проверена на разметке.")
    w("- Чувствительность к гамме/шуму (см. `docs/ROBUSTNESS_REPORT.md`): часть решений меняется при искажении яркостной кривой.")
    w("- Не медицинское изделие; результат — подсказка для контроля качества укладки, не диагноз.")
    w("")
    w("## 8. Версия и хэши")
    w("")
    w(f"- Версия пайплайна (config.yaml → version): **{version}**; config_hash: **{chash}** (тот же пишется в DICOM SR и ответ API).")
    w("- Ключевые библиотеки (requirements.txt): " + ", ".join(f"{k} {req[k]}" for k in KEY_LIBS if k in req) + ".")
    w("")
    w("| Файл модели | Критерий | Признаки / источник | n_pos | sha256[:12] |")
    w("|---|---|---|---|---|")
    for name, info in manifest.items():
        if not name.endswith(".pkl"):
            continue
        p = root / "models" / name
        feats = ", ".join(info.get("feature_cols", [])) or info.get("emb_source", "")
        w(f"| {name} | {info.get('criterion', '—')} | {feats} | {info.get('n_pos', '—')} | {sha12(p) if p.exists() else 'файл отсутствует'} |")
    for extra in ("model_region_emb.pkl", "backbone_densito.pth"):
        p = root / "models" / extra
        if p.exists() and extra not in manifest:
            w(f"| {extra} | — | {'классификатор области' if 'region' in extra else 'бэкбон densito (EfficientNet-B0)'} | — | {sha12(p)} |")
    w("")
    w("Источники: `models/metrics_summary.json`, `models/models_manifest.json`, `config.yaml`, `requirements.txt`; "
      "процедура валидации и отвергнутые гипотезы — `docs/EVIDENCE.md`.")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"MODEL_CARD -> {out} ({len(L)} строк)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
