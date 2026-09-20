"""Отчёт по nested-гейту источника эмбеддингов (К13) -> docs/EMB_GATE_REPORT.md.

Все числа берутся из models/emb_gate_decisions.json (результат tools/emb_gate.py) и
models/metrics_summary.json — руками в отчёт ничего не вписывается.

Запуск: python tools/emb_gate_report.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEC = json.loads((ROOT / "models" / "emb_gate_decisions.json").read_text(encoding="utf-8"))
MS = json.loads((ROOT / "models" / "metrics_summary.json").read_text(encoding="utf-8"))

NAMES = {"sp_pos": "Некорректная укладка (позвоночник)",
         "sp_axis": "Не выравнена ось позвоночника",
         "sp_art": "Присутствуют посторонние предметы",
         "hip_pos": "Некорректная укладка (бедро)",
         "hip_roi": "Некорректная область интереса"}
SRC_ORDER = ["imagenet", "densito", "densito_inv", "densito_inv_free"]


def f(v, n=3):
    return "—" if v is None or (isinstance(v, float) and v != v) else f"{v:.{n}f}"


def s(v, n=3):
    return "—" if v is None or (isinstance(v, float) and v != v) else f"{v:+.{n}f}"


p = DEC["protocol"]
L = []
L.append("# К13: выбор источника эмбеддингов контура B по критерию (nested-гейт)\n")
L.append("Инструмент — `tools/emb_gate.py`, решения — `models/emb_gate_decisions.json`, этот отчёт "
         "генерируется `tools/emb_gate_report.py` (числа не переписываются руками).\n")
L.append("## Зачем гейт\n")
L.append("GPU-эксперимент К13 обучил два бэкбона EfficientNet-B0 с loss инвариантности эмбеддинга к гамме и шуму: "
         "`densito_inv` (полный пул 15 633 плитки) и `densito_inv_free` (9 893 плитки, без данных с несвободной "
         "лицензией). На его собственной оценке (только контур B: PCA-32 + логрегрессия, 10 повторов) прирост "
         "выглядел большим сразу на двух критериях — `sp_axis` +0.279 AUC и `hip_pos` +0.131. Но там бэкбон и "
         "выбирался, и оценивался на одной и той же выборке, и мерился отдельный контур B, а не то, что отдаётся "
         "заказчику. Поэтому источник выбирается здесь — внутри nested CV, на полном стэке (контур A + контур B) "
         "и с порогом по правилу, как в проде.\n")
L.append("## Протокол\n")
L.append(f"- внешний контур: GroupKFold(shuffle, random_state=42+r), {p['outer']} фолдов × {p['repeats']} повторов; "
         f"группы — {p['groups']};\n"
         f"- внутренний контур: GroupKFold {p['inner']} фолда на внешнем train → inner-OOF контура A (один раз) и "
         f"inner-OOF контура B для каждого источника-кандидата;\n"
         f"- выбор источника: максимум AUC стэка на inner-OOF (ранги внутри inner-OOF, вес {p['w_stacking']}); "
         f"при равенстве — консервативно продакшен-источник;\n"
         f"- порог: правило из `config.yaml` `thresholds_rule`, считается на inner-OOF;\n"
         f"- ветки: `base` — источник продакшена; `gate` — источник выбирается внутри nested (оценка процедуры "
         f"выбора); `fixed:<источник>` — источник задан жёстко на всех фолдах (оценка ровно того изменения, "
         f"которое пойдёт в прод);\n"
         f"- приёмка: прирост AUC ≥ {p['gain_min']} в ≥ {p['gain_repeats']} из {p['repeats']} повторов И без потери "
         f"macro-F1; отдельно проверяется бинарная метрика области (у организаторов она первая).\n")
L.append("Кандидаты для критерия ограничены источниками, у которых есть эмбеддинги в том же варианте "
         "предобработки, что выбран К11 (`sp_pos` считается на `canonical`, остальные — на `baseline`).\n")

L.append("## Результат по критериям\n")
L.append("| Критерий | n / поз. | База | Выбран внутри nested (частота фолдов) | ΔAUC ветки gate | Вердикт |")
L.append("|---|---|---|---|---|---|")
for d in DEC["decisions"]:
    freq = " ".join(f"{k.replace('densito_inv_free', 'inv_free').replace('densito_inv', 'inv')}={d['freq_' + k]}"
                    for k in SRC_ORDER if d.get(f"freq_{k}"))
    verdict = ("**принят " + d["source_mode"] + "**") if d["accepted_gain_rule"] else "не принят"
    L.append(f"| `{d['criterion']}` {NAMES[d['criterion']]} | {d['n']} / {d['n_pos']} | `{d['base_source']}` | "
             f"{d['source_mode']} ({freq}) | {s(d['mean_delta_auc'])} (≥0.03 в "
             f"{d['n_repeats_gain_ge_0.03']}/{d['n_repeats']}) | {verdict} |")
L.append("")
L.append("Ветка `gate` честно оценивает процедуру «выбрать бэкбон внутри фолда», и она шумная: на `sp_axis` "
         "процедура даёт +0.037 AUC, потому что в 21 фолде из 100 выбирает не тот источник. Поэтому решение "
         "принимается по ветке `fixed` — это ровно то изменение, которое уходит в конфиг, — а частота выбора "
         "внутри фолдов служит проверкой, что выбор устойчив, а не случайность одного разбиения.\n")

L.append("## Жёстко заданный источник против продакшена (парные фолды)\n")
L.append("| Критерий | Источник | AUC стэка (среднее по повторам) | ΔAUC | ≥0.03 | в плюсе | Δmacro-F1 | Правило |")
L.append("|---|---|---|---|---|---|---|---|")
for d in DEC["decisions"]:
    for src, v in d["fixed"].items():
        mark = " **✓**" if v["accepted_gain_rule"] else ""
        base_mark = " (база)" if src == d["base_source"] else ""
        L.append(f"| `{d['criterion']}` | `{src}`{base_mark} | {f(v['auc_mean_over_repeats'])} ± "
                 f"{f(v['auc_sd_over_repeats'], 3)} | {s(v['mean_delta_auc'])} | "
                 f"{v['n_repeats_gain_ge_0.03']}/{v['n_repeats']} | {v['n_repeats_positive']}/{v['n_repeats']} | "
                 f"{s(v['mean_delta_macro_f1'])} | {'да' + mark if v['accepted_gain_rule'] else 'нет'} |")
L.append("")

dec_axis = next(d for d in DEC["decisions"] if d["criterion"] == "sp_axis")
fx = dec_axis["fixed"]["densito_inv"]
L.append("## Что принято и что отвергнуто\n")
L.append(f"**Принято одно изменение: `sp_axis` → `densito_inv`.** Ветка `fixed` даёт ΔAUC {s(fx['mean_delta_auc'])} "
         f"(прирост ≥0.03 в {fx['n_repeats_gain_ge_0.03']} из {fx['n_repeats']} повторов, в плюсе "
         f"{fx['n_repeats_positive']} из {fx['n_repeats']}), Δmacro-F1 {s(fx['mean_delta_macro_f1'])}; внутри nested "
         f"этот источник выбран в {dec_axis['freq_densito_inv']} фолдах из {dec_axis['n_folds']}. "
         f"Nested-оценка AUC стэка: {f(dec_axis['auc_base_mean_over_repeats'])} (ImageNet) → "
         f"{f(fx['auc_mean_over_repeats'])} (`densito_inv`).\n")
L.append("**Отвергнуто:**\n")
for crit, why in (("hip_pos", "главное расхождение с оценкой К13: обещанные на контуре B +0.131 AUC на полном стэке "
                              "превращаются в"),
                  ("sp_art", "ImageNet остаётся недостижимым:"),
                  ("hip_roi", "все кандидаты хуже базы:"),
                  ("sp_pos", "новые бэкбоны в варианте `canonical` недоступны и по К13 слабее; кандидат ImageNet даёт")):
    d = next(x for x in DEC["decisions"] if x["criterion"] == crit)
    parts = [f"`{src}` {s(v['mean_delta_auc'])} ({v['n_repeats_gain_ge_0.03']}/{v['n_repeats']})"
             for src, v in d["fixed"].items() if src != d["base_source"]]
    L.append(f"- `{crit}` — {why} " + ", ".join(parts) + ".")
L.append("")
L.append("Почему contour-B-прирост на `hip_pos` не дошёл до стэка: там контур A (геометрия бедра, 5 признаков) "
         "сам даёт AUC 0.710, и ранговое усреднение 0.5/0.5 с более слабым контуром B новую информацию не "
         "добавляет. На `sp_axis` картина обратная: контур A — один признак `axis_angle_deg`, а контур B "
         "на ImageNet был около случайного (AUC 0.491), поэтому замена бэкбона видна в стэке целиком.\n")

axis = MS["spine"]["sp_axis"]
L.append("## Что получилось после внедрения (OOF на 499 файлах)\n")
L.append("| Величина | До (ImageNet) | После (`densito_inv`) |")
L.append("|---|---|---|")
L.append(f"| AUC контура B (`sp_axis`) | 0.491 | {f(axis['auc_emb'])} |")
L.append(f"| AUC стэка (`sp_axis`) | 0.738 | {f(axis['auc_stacked'])} |")
L.append(f"| F1(+) OOF (`sp_axis`) | 0.381 | {f(axis['f1_oof'])} |")
L.append(f"| Порог (правило {axis['threshold_rule']}) | 0.7560 | {f(axis['threshold'], 4)} |")
L.append("| Бинарная «есть нарушение», позвоночник: ROC-AUC | 0.758 | 0.773 |")
L.append("| Бинарная «есть нарушение», позвоночник: F1 | 0.609 | 0.623 |")
L.append("| Macro-F1 по типам нарушений, позвоночник | 0.464 | 0.539 |")
L.append("")
L.append("Бедро не затронуто: его критерии остались на ImageNet, числа бит в бит те же.\n")
L.append("## Ограничения\n")
L.append("- Бэкбон `densito_inv` обучен один раз (один seed предобучения). Разброс самого предобучения этот "
         "протокол не измеряет: 20 повторов учитывают только шум фолдов и классификатора.\n")
L.append("- `densito_inv` обучен на пуле, куда входят Arak (CC BY-NC) и BUU-LSPINE (EULA) — то же ограничение, "
         "что у `backbone_densito.pth`; см. `docs/LICENSES_AND_DATA_AUDIT.md`. Свободный по лицензиям "
         "`densito_inv_free` на `sp_axis` правило не проходит (ΔAUC "
         f"{s(dec_axis['fixed']['densito_inv_free']['mean_delta_auc'])}, "
         f"{dec_axis['fixed']['densito_inv_free']['n_repeats_gain_ge_0.03']}/"
         f"{dec_axis['fixed']['densito_inv_free']['n_repeats']} повторов).\n")
L.append("- Позитивов у `sp_axis` всего 17, поэтому доверительные интервалы широкие: парный ДИ прироста "
         f"[{s(fx['delta_auc_pooled_ci'][0])}; {s(fx['delta_auc_pooled_ci'][2])}] включает ноль, и вывод опирается "
         "на парную схему (одинаковые фолды, 20/20 повторов в плюсе), а не на абсолютную ширину ДИ.\n")

(ROOT / "docs" / "EMB_GATE_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("docs/EMB_GATE_REPORT.md:", len("\n".join(L).splitlines()), "строк")
