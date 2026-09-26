#!/usr/bin/env python3
"""Сводка слепой ревизии (лёгкий набор, страница /review/): врачи ↔ разметка ↔ сервис ↔ между врачами.

Вход:  JSON-ответы страницы (POST /api/review), ключ набора tools/review/kit_light_manifest.json,
       вердикты сервиса из web/review/index.html (DATA.service — OOF текущей версии).
Дубликаты: повторная отправка той же сессии (совпадают shown_at и ms всех показов или session_id) — отбрасывается
с пометкой, в счёт идёт первая отправка.
Счёт:  первый показ каждого кадра; «затрудняюсь ответить» исключается из согласия и считается отдельно.
Только стандартная библиотека (на хосте нет pandas/numpy).

Запуск на сервере (из корня B):
  python3 tools/review/compute_light_agreement.py --responses /opt/neuropeople/densito_api_outputs/review \
      --out docs/REVIEW_DOCTORS.md [--name <файл>=<подпись> ...] [--exclude <файл> ...]
"""
import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CRIT_NAMES = {"sp_pos": "укладка позвоночника", "sp_axis": "ось позвоночника", "sp_art": "посторонние предметы",
              "hip_pos": "укладка бедра", "hip_roi": "область интереса бедра"}
GROUP_NAMES = {"norm:spine": "норма, позвоночник", "norm:hip": "норма, бедро", "viol:sp_pos": "нарушение: укладка позвоночника",
               "viol:sp_axis": "нарушение: ось позвоночника", "viol:sp_art": "нарушение: посторонние предметы",
               "viol:hip_pos": "нарушение: укладка бедра", "viol:hip_roi": "нарушение: область интереса бедра"}


def norm_code(code: str) -> str:
    return re.sub(r"^(rh|lh)_", "hip_", code or "")


def msk(iso: str) -> str:
    try:
        t = dt.datetime.strptime(iso[:19], "%Y-%m-%dT%H:%M:%S") + dt.timedelta(hours=3)
        return t.strftime("%d.%m %H:%M")
    except Exception:  # noqa: BLE001
        return iso or "—"


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - adj) / den, (centre + adj) / den)


def kappa(a, b):
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def frac(k, n, ci=True):
    if n == 0:
        return "—"
    s = f"{k}/{n} ({100 * k / n:.0f} %)"
    if ci:
        lo, hi = wilson(k, n)
        s += f" [{100 * lo:.0f}; {100 * hi:.0f}]"
    return s


def fk(v):
    return "—" if v is None else f"{v:.2f}"


def load_service(page: Path):
    html = page.read_text(encoding="utf-8")
    m = re.search(r"^const DATA = (.+);$", html, re.M)
    if not m:
        raise SystemExit(f"DATA не найден в {page}")
    data = json.loads(m.group(1))
    return data["service"], data["kit_sha256"], data.get("service_version", "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", type=Path, required=True, help="каталог с JSON-ответами /api/review")
    ap.add_argument("--manifest", type=Path, default=HERE / "kit_light_manifest.json")
    ap.add_argument("--page", type=Path, default=ROOT / "web" / "review" / "index.html")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "REVIEW_DOCTORS.md")
    ap.add_argument("--name", action="append", default=[], help="файл=подпись (например 2026...json=врач 1)")
    ap.add_argument("--exclude", action="append", default=[], help="имя файла, не учитывать")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    frames = {f["file"]: f for f in manifest["frames"]}
    order = {s["file"]: s["idx"] for s in manifest["shows_full"] if not s["is_repeat"]}
    service, kit_sha, svc_version = load_service(args.page)
    if kit_sha != manifest["kit_sha256"]:
        raise SystemExit("kit_sha256 страницы и манифеста не совпадают")
    names = dict(x.split("=", 1) for x in args.name)

    def label_bad(f):
        return any(v == 1 for v in frames[f]["labels"].values())

    def label_codes(f):
        return [c for c, v in frames[f]["labels"].items() if v == 1]

    def svc_bad(f):
        return int(service[f]["quality_class"]) == 1

    def svc_codes(f):
        return [norm_code(c["code"]) for c in service[f].get("criteria", []) if c.get("flag")]

    # ---- загрузка ответов, дубликаты -------------------------------------------------------------
    docs, skipped, dups, seen = [], [], [], {}
    for p in sorted(args.responses.glob("*.json")):
        if p.name in args.exclude:
            skipped.append((p.name, "исключён вручную"))
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("kit_sha256") != kit_sha:
            skipped.append((p.name, "другой набор"))
            continue
        answers = d.get("answers") or []
        if len(answers) < len(manifest["shows_full"]):
            skipped.append((p.name, f"неполный: {len(answers)} ответов"))
            continue
        sig = hashlib.sha256(json.dumps([(a.get("idx"), a.get("file"), a.get("shown_at"), a.get("ms")) for a in answers],
                                        sort_keys=True).encode()).hexdigest()
        keys = [sig] + ([f"sid:{d['session_id']}"] if d.get("session_id") else [])
        prev = next((seen[k] for k in keys if k in seen), None)
        if prev:
            dups.append((p.name, prev, msk(d.get("finished_at", ""))))
            continue
        for k in keys:
            seen[k] = p.name
        docs.append({"file": p.name, "data": d, "sha": hashlib.sha256(p.read_bytes()).hexdigest()[:12]})
    docs.sort(key=lambda x: x["data"].get("finished_at", ""))
    for i, doc in enumerate(docs, 1):
        base = names.get(doc["file"], f"врач {i}")
        rv = (doc["data"].get("reviewer") or "").strip()
        doc["name"] = base if not rv or rv == base else f"{base} ({rv})"
        first, repeats = {}, []
        for a in sorted(doc["data"]["answers"], key=lambda a: a["idx"]):
            if a["file"] in first:
                repeats.append((first[a["file"]], a))
            else:
                first[a["file"]] = a
        doc["first"], doc["repeats"] = first, repeats
        doc["certain"] = [f for f in frames if f in first and first[f]["verdict"] in ("ok", "bad")]
        doc["unsure"] = [f for f in frames if f in first and first[f]["verdict"] not in ("ok", "bad")]

    def d_bad(doc, f):
        return doc["first"][f]["verdict"] == "bad"

    # ---- сводка по врачам --------------------------------------------------------------------------
    L = []
    L.append(f"# Слепая ревизия рентгенологов — сводка ({'предварительно, ' if len(docs) < 3 else ''}{len(docs)} врач(а/ей))\n")
    L.append(f"Сформировано {dt.datetime.now().strftime('%d.%m.%Y %H:%M')} скриптом `tools/review/compute_light_agreement.py`. "
             f"Сервис {svc_version} (OOF), набор `kit_sha256 {kit_sha[:12]}…`, {len(frames)} уникальных кадров, "
             f"{len(manifest['shows_full'])} показов, {len(manifest['repeat_frames_full'])} повтора. Разметка — организаторов. "
             "Счёт по первому показу кадра; «затрудняюсь ответить» исключён из согласия. Врач отвечал до показа вывода сервиса.\n")
    L.append("## Источники\n")
    for doc in docs:
        d = doc["data"]
        L.append(f"- **{doc['name']}** — `{doc['file']}` (sha256 {doc['sha']}…), {msk(d.get('started_at',''))}–{msk(d.get('finished_at',''))} MSK, "
                 f"{len(d['answers'])} ответов, «затрудняюсь»: {len(doc['unsure'])}"
                 + (f", session_id `{d['session_id']}`" if d.get("session_id") else "") + ".")
    for name, prev, t in dups:
        L.append(f"- `{name}` ({t} MSK) — **повторная отправка той же сессии**, что `{prev}` (совпадают время показа и длительность всех ответов); в счёт не идёт.")
    for name, why in skipped:
        L.append(f"- `{name}` — пропущен: {why}.")
    L.append("")

    L.append("## Сводка\n")
    L.append("| | " + " | ".join(doc["name"] for doc in docs) + " | Сервис (OOF) |")
    L.append("|---|" + "---|" * (len(docs) + 1))
    row_lbl, row_sens, row_fa, row_svc, row_k_lbl, row_k_svc, row_reason = [], [], [], [], [], [], []
    for doc in docs:
        cert = doc["certain"]
        agree = sum(d_bad(doc, f) == label_bad(f) for f in cert)
        pos = [f for f in cert if label_bad(f)]
        neg = [f for f in cert if not label_bad(f)]
        row_lbl.append(frac(agree, len(cert)))
        row_sens.append(frac(sum(d_bad(doc, f) for f in pos), len(pos), ci=False))
        row_fa.append(frac(sum(d_bad(doc, f) for f in neg), len(neg), ci=False))
        row_svc.append(frac(sum(d_bad(doc, f) == svc_bad(f) for f in cert), len(cert)))
        row_k_lbl.append(fk(kappa([d_bad(doc, f) for f in cert], [label_bad(f) for f in cert])))
        row_k_svc.append(fk(kappa([d_bad(doc, f) for f in cert], [svc_bad(f) for f in cert])))
        bads = [f for f in cert if d_bad(doc, f) and doc["first"][f].get("reason")]
        row_reason.append(f"{sum(doc['first'][f]['reason'] in label_codes(f) for f in bads)}/{len(bads)} с разметкой, "
                          f"{sum(doc['first'][f]['reason'] in svc_codes(f) for f in bads)}/{len(bads)} с сервисом")
    allf = list(frames)
    s_agree = sum(svc_bad(f) == label_bad(f) for f in allf)
    s_pos = [f for f in allf if label_bad(f)]
    s_neg = [f for f in allf if not label_bad(f)]
    L.append("| Совпадение с разметкой (нарушение есть/нет) | " + " | ".join(row_lbl) + f" | {frac(s_agree, len(allf))} |")
    L.append("| Нашёл нарушений из размеченных | " + " | ".join(row_sens) + f" | {frac(sum(svc_bad(f) for f in s_pos), len(s_pos), ci=False)} |")
    L.append("| Ложных тревог на нормах | " + " | ".join(row_fa) + f" | {frac(sum(svc_bad(f) for f in s_neg), len(s_neg), ci=False)} |")
    L.append("| Каппа Коэна с разметкой | " + " | ".join(row_k_lbl) + f" | {fk(kappa([svc_bad(f) for f in allf], [label_bad(f) for f in allf]))} |")
    L.append("| Совпадение с сервисом | " + " | ".join(row_svc) + " | — |")
    L.append("| Каппа Коэна с сервисом | " + " | ".join(row_k_svc) + " | — |")
    L.append("| Причина нарушения совпала (когда врач сказал «неправильно») | " + " | ".join(row_reason) + " | — |")
    L.append("\nВ квадратных скобках — 95 % доверительный интервал (Уилсон). Сервис в столбце справа посчитан на всех кадрах набора; "
             "в строках врачей сравнение идёт только по кадрам с определённым ответом врача.\n")

    # ---- между врачами -----------------------------------------------------------------------------
    if len(docs) >= 2:
        L.append("## Между врачами\n")
        L.append("| Пара | Совпали по вердикту | Каппа |")
        L.append("|---|---|---|")
        for a, b in itertools.combinations(docs, 2):
            both = [f for f in frames if f in a["first"] and f in b["first"] and f in a["certain"] and f in b["certain"]]
            agree = sum(d_bad(a, f) == d_bad(b, f) for f in both)
            L.append(f"| {a['name']} ↔ {b['name']} | {frac(agree, len(both))} | {fk(kappa([d_bad(a, f) for f in both], [d_bad(b, f) for f in both]))} |")
        # большинство
        maj = {}
        for f in frames:
            votes = [d_bad(doc, f) for doc in docs if f in doc["certain"]]
            if len(votes) >= 2 and votes.count(True) != votes.count(False):
                maj[f] = votes.count(True) > votes.count(False)
        if maj:
            fs = list(maj)
            L.append(f"\nБольшинство врачей (определено на {len(fs)} кадрах): с разметкой {frac(sum(maj[f] == label_bad(f) for f in fs), len(fs))}, "
                     f"с сервисом {frac(sum(maj[f] == svc_bad(f) for f in fs), len(fs))}; сервис на тех же кадрах с разметкой "
                     f"{frac(sum(svc_bad(f) == label_bad(f) for f in fs), len(fs))}.")
        unanimous = [f for f in frames if all(f in doc["certain"] for doc in docs) and len({d_bad(doc, f) for doc in docs}) == 1]
        ceil = [f for f in unanimous if d_bad(docs[0], f) != label_bad(f)]
        vs_svc = [f for f in unanimous if d_bad(docs[0], f) != svc_bad(f)]
        L.append(f"\n**Потолок разметки** — все врачи единодушны и расходятся с разметкой: {len(ceil)} из {len(unanimous)} единодушных кадров"
                 + (": " + "; ".join(f"показ {order[f]} ({GROUP_NAMES.get(frames[f]['group'], frames[f]['group'])}, врачи: {'нарушение' if d_bad(docs[0], f) else 'норма'})" for f in ceil) if ceil else "") + ".")
        L.append(f"\n**Все врачи против сервиса**: {len(vs_svc)} кадров"
                 + (": " + "; ".join(f"показ {order[f]} (сервис: {service[f]['violation_type'] if svc_bad(f) else 'норма'}, врачи: {'нарушение' if d_bad(docs[0], f) else 'норма'})" for f in vs_svc) if vs_svc else "") + ".\n")

    # ---- по группам --------------------------------------------------------------------------------
    L.append("## По группам набора (кадров с определённым ответом / сказали «нарушение»)\n")
    L.append("| Группа | Кадров | " + " | ".join(doc["name"] for doc in docs) + " | Сервис |")
    L.append("|---|---|" + "---|" * (len(docs) + 1))
    for g in GROUP_NAMES:
        fs = [f for f in frames if frames[f]["group"] == g]
        if not fs:
            continue
        cells = []
        for doc in docs:
            c = [f for f in fs if f in doc["certain"]]
            cells.append(f"{sum(d_bad(doc, f) for f in c)}/{len(c)}")
        L.append(f"| {GROUP_NAMES[g]} | {len(fs)} | " + " | ".join(cells) + f" | {sum(svc_bad(f) for f in fs)}/{len(fs)} |")
    L.append("")

    # ---- карта по кадрам ----------------------------------------------------------------------------
    L.append("## Карта по кадрам (первый показ)\n")
    L.append("| Показ | Кадр | Группа | Разметка | Сервис | " + " | ".join(doc["name"] for doc in docs) + " |")
    L.append("|---|---|---|---|---|" + "---|" * len(docs))
    for f in sorted(frames, key=lambda x: order[x]):
        lab = ", ".join(CRIT_NAMES.get(c, c) for c in label_codes(f)) or "норма"
        sv = (", ".join(CRIT_NAMES.get(c, c) for c in svc_codes(f)) or service[f].get("violation_type", "?")) if svc_bad(f) else "норма"
        cells = []
        for doc in docs:
            a = doc["first"].get(f)
            if not a:
                cells.append("—")
                continue
            v = {"ok": "норма", "bad": "нарушение", "unsure": "затрудняюсь"}.get(a["verdict"], a["verdict"])
            if a["verdict"] == "bad" and a.get("reason"):
                v += f": {CRIT_NAMES.get(a['reason'], a['reason'])}"
            rep = next((r for r in doc["repeats"] if r[0]["file"] == f), None)
            if rep and rep[1]["verdict"] != rep[0]["verdict"]:
                v += f" (повтор: {'нарушение' if rep[1]['verdict'] == 'bad' else 'норма' if rep[1]['verdict'] == 'ok' else 'затрудняюсь'})"
            if a.get("wording_ok") is False:
                v += " ✎"
            cells.append(v)
        L.append(f"| {order[f]} | `{f[:13]}` | {GROUP_NAMES.get(frames[f]['group'], frames[f]['group'])} | {lab} | {sv} | " + " | ".join(cells) + " |")
    L.append("\n✎ — врач отметил формулировку сервиса по этому кадру как непонятную или опасную.\n")

    # ---- повторы, фразы, свободные ответы ----------------------------------------------------------
    L.append("## Самосогласие на повторах\n")
    for doc in docs:
        same = sum(r[0]["verdict"] == r[1]["verdict"] for r in doc["repeats"])
        L.append(f"- {doc['name']}: {same} из {len(doc['repeats'])} повторов совпали" + ("" if same == len(doc["repeats"]) else "; расхождения: " + "; ".join(
            f"показ {r[0]['idx']}→{r[1]['idx']} ({r[0]['verdict']}→{r[1]['verdict']})" for r in doc["repeats"] if r[0]["verdict"] != r[1]["verdict"])) + ".")
    L.append("\n## Формулировки сервиса (так говорить можно / нельзя)\n")
    phr_ids = []
    for doc in docs:
        for p in doc["data"].get("phrases", []):
            if p["id"] not in phr_ids:
                phr_ids.append(p["id"])
    L.append("| Фраза | " + " | ".join(doc["name"] for doc in docs) + " |")
    L.append("|---|" + "---|" * len(docs))
    for pid in phr_ids:
        text = next((p["text"] for doc in docs for p in doc["data"].get("phrases", []) if p["id"] == pid), pid)
        cells = []
        for doc in docs:
            p = next((x for x in doc["data"].get("phrases", []) if x["id"] == pid), None)
            if p is None or p.get("acceptable") is None:
                cells.append("—")
            else:
                cells.append(("можно" if p["acceptable"] else "**нельзя**") + (f" — {p['comment']}" if p.get("comment") else ""))
        L.append(f"| {text[:90]}{'…' if len(text) > 90 else ''} | " + " | ".join(cells) + " |")
    L.append("\n## Свободные ответы и комментарии\n")
    qs = {"first_screen": "Что непонятно с первого экрана сервиса", "not_trusted": "Чему не поверили",
          "missing": "Чего не хватает для работы отделения", "excess": "Что лишнее"}
    for doc in docs:
        L.append(f"**{doc['name']}**")
        ft = doc["data"].get("free_text") or {}
        for k, q in qs.items():
            if (ft.get(k) or "").strip():
                L.append(f"- {q}: «{ft[k].strip()}»")
        for a in sorted(doc["data"]["answers"], key=lambda a: a["idx"]):
            if (a.get("comment") or "").strip():
                L.append(f"- показ {a['idx']} ({'нарушение: ' + CRIT_NAMES.get(a['reason'], a['reason']) if a['verdict'] == 'bad' and a.get('reason') else {'ok': 'норма', 'unsure': 'затрудняюсь'}.get(a['verdict'], a['verdict'])}): «{a['comment'].strip()}»")
        L.append("")
    L.append("## Оговорки\n")
    L.append(f"- {len(frames)} уникальных кадров одного аппарата, набор составлен намеренно (половина — нарушения по разметке); проценты не переносить на поток отделения.")
    L.append(f"- {len(docs)} врач(а/ей); мнение каждого — одно прохождение без обсуждения. Ни модель, ни пороги по этим мнениям не менялись: оценка организаторов идёт по их разметке.")
    L.append("- Сервис показан в режиме OOF (вне обучающего фолда) той же версии, что в бою.")
    notes = Path(__file__).with_name("review_notes.md")  # ручные оговорки к конкретному раунду (консилиум 26.09)
    if notes.is_file():
        L += [x for x in notes.read_text(encoding="utf-8").splitlines() if x.strip()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"готово: {args.out} — врачей {len(docs)}, дубликатов {len(dups)}, пропущено {len(skipped)}")


if __name__ == "__main__":
    main()
