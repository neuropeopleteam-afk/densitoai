#!/usr/bin/env python3
"""Экспертная проверка (src/expert_review.py): выборка, слепота, ответы, отчёт о совпадении.

Запуск:  python tests/test_expert_review.py     (код 0 — пройдено)
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["DENSITO_REGISTRY_OPEN"] = "1"
import registry as R  # noqa: E402
import expert_review as E  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


def main():
    d = Path(tempfile.mkdtemp(prefix="densito_expert_"))
    jobs = d / "jobs"
    reg = R.Registry(d)
    er = E.ExpertReview(reg, jobs)
    doc = reg.check_session("open|doctor|%D0%9F%D0%B5%D1%82%D1%80%D0%BE%D0%B2%D0%B0")
    print("1. выборка")
    check("пустой журнал — понятная ошибка", raises(ValueError, er.create_set, doc, 10))
    # 12 снимков: 6 с нарушением (укладка позвоночника), 6 в норме; у 10 есть чистый кадр
    job = "20260920_100000_eeeeee"
    rows = []
    for i in range(12):
        viol = i < 6
        rows.append({"path_to_study": f"f{i}.dcm", "study_uid": f"1.2.{i}", "image_uid": f"1.2.{i}.1",
                     "anatomical_region": "Поясничный отдел позвоночника", "quality_class": "1" if viol else "0",
                     "violation_type": "Некорректная укладка" if viol else "", "quality_prob": "0.9" if viol else "0.1",
                     "processing_status": "Success", "region_supported": True, "details": {"internal_region": "spine"}})
    reg.index_rows(job, rows, {})
    (jobs / job / "bonus").mkdir(parents=True)
    for i in range(10):
        (jobs / job / "bonus" / f"row{i:04d}_frame.png").write_bytes(b"\x89PNG")
    s = er.create_set(doc, 8, seed=1)
    check("в выборке 8 снимков", s["n"] == 8 and len(s["items"]) == 8)
    check("только снимки с чистым кадром", all(it["study_uid"] not in ("1.2.10", "1.2.11") for it in s["items"]))
    check("слепая выдача: решения сервиса нет", "service_class" not in str(s) and "Некорректная" not in str(s))
    sv = er.get_set(s["id"], blind=False)
    n_viol = sum(1 for it in sv["items"] if it["service_class"] == "1")
    check("стратификация: половина с нарушением", n_viol == 4, str(n_viol))
    check("у позвоночника три критерия", [c["code"] for c in s["items"][0]["criteria"]] == ["sp_pos", "sp_axis", "sp_art"])
    check("кадр отдаётся", er.frame(s["id"], 1).is_file())
    check("чужой номер кадра — KeyError", raises(KeyError, er.frame, s["id"], 99))
    print("2. ответы")
    check("неполный ответ отклонён", raises(ValueError, er.answer, s["id"], 1, doc, {"sp_pos": "ok"}))
    check("ответ без имени эксперта отклонён", raises(ValueError, er.answer, s["id"], 1, {"login": "", "name": "", "role": "doctor"},
                                                       {"sp_pos": "ok", "sp_axis": "ok", "sp_art": "ok"}))
    # врач совпадает с сервисом по укладке везде, кроме одного пропуска и одного ложного срабатывания
    items = {it["pos"]: it for it in sv["items"]}
    flip_fn = next(p for p, it in items.items() if it["service_class"] == "1")
    flip_fp = next(p for p, it in items.items() if it["service_class"] == "0")
    for p, it in items.items():
        svc = it["service_class"] == "1"
        exp = (not svc) if p in (flip_fn, flip_fp) else svc
        er.answer(s["id"], p, doc, {"sp_pos": "violation" if exp else "ok", "sp_axis": "ok", "sp_art": "unsure"}, "к")
    check("повторный ответ заменяет прежний", er.answer(s["id"], flip_fn, doc, {"sp_pos": "ok", "sp_axis": "ok", "sp_art": "unsure"})["ok"])
    check("мои ответы — 8", len(er.my_answers(s["id"], "Петрова")) == 8)
    print("3. отчёт")
    rp = er.report(s["id"])
    c = next(x for x in rp["criteria"] if x["code"] == "sp_pos")
    check("укладка: TP 3, FN 1, FP 1, TN 3", (c["tp"], c["fn"], c["fp"], c["tn"]) == (3, 1, 1, 3), str(c))
    check("укладка: совпадение 75 %", c["agreement"] == 0.75)
    check("чувствительность и специфичность 0,75 с интервалами", c["sensitivity"] == 0.75 and c["specificity"] == 0.75 and c["sensitivity_ci"])
    a = next(x for x in rp["criteria"] if x["code"] == "sp_axis")
    check("ось: сервис и врач — везде норма, специфичность 1", a["specificity"] == 1.0 and a["sensitivity"] is None)
    u = next(x for x in rp["criteria"] if x["code"] == "sp_art")
    check("«не могу оценить» не входит в расчёт", u["n"] == 0 and u["unsure"] == 8)
    check("расхождения перечислены", len(rp["disagreements"]) == 2)
    check("каппа посчитана", c["kappa"] == 0.5, str(c["kappa"]))
    csvt = er.export_csv(s["id"])
    check("CSV: строка на критерий", csvt.count("\n") == 1 + 8 * 3)
    check("оценки видны в карточке исследования", len(er.for_study(items[1]["study_uid"])) == 1)
    print("4. своя слепая проверка (загрузил — оценил — увидел решение)")
    reg.hidden_jobs = er.hidden_jobs
    org = reg.check_session("open|doctor|%D0%9E%D1%80%D0%B3%D0%B0%D0%BD%D0%B8%D0%B7%D0%B0%D1%82%D0%BE%D1%80")  # «Организатор»
    job2 = "20260926_120000_ffffff"
    rows2 = []
    for i in range(6):
        hip = i >= 3
        rows2.append({"path_to_study": f"o{i}.dcm", "study_uid": f"9.9.{i}", "image_uid": f"9.9.{i}.1",
                      "anatomical_region": "Проксимальный отдел бедра" if hip else "Поясничный отдел позвоночника",
                      "quality_class": "1" if i in (0, 3) else "0", "violation_type": "Некорректная укладка" if i in (0, 3) else "",
                      "quality_prob": "0.8", "processing_status": "Failure" if i == 5 else "Success", "region_supported": i != 4,
                      "details": {"internal_region": "right_hip" if hip else "spine"}})
    reg.index_rows(job2, rows2, {})
    (jobs / job2 / "bonus").mkdir(parents=True)
    for i in range(6):
        (jobs / job2 / "bonus" / f"row{i:04d}_frame.png").write_bytes(b"\x89PNG" + bytes([i]))
    check("без ФИО эксперта набор не создаётся", raises(ValueError, er.create_set_from_job, reg.check_session("open|doctor|"), job2))
    o = er.create_set_from_job(org, job2, "Свои снимки", n_files=6, seed=3, versions={"model_version": "2.4.0", "config_hash": "abc"})
    check("в набор вошли все 4 годных снимка, без отбора", o["n"] == 4, str(o["n"]))
    check("отказ и чужая область посчитаны отдельно", o["params"]["skipped"]["failure"] == 1 and o["params"]["skipped"]["unsupported"] == 1)
    check("хэш набора и версия записаны", len(o["params"]["set_hash"]) == 12 and o["params"]["config_hash"] == "abc")
    check("слепая выдача: нет study_uid и решения", all(it["study_uid"] is None for it in o["items"]) and "service_class" not in str(o))
    sr = reg.search()
    mine = [x for x in sr["studies"] if x["study_uid"].startswith("9.9.")]
    check("журнал: снимки загрузки видны", len(mine) == 6, str(len(mine)))
    check("журнал: решение скрыто до завершения", all(x["verdict"] == "hidden" and not x["violations"] and x["n_violation"] == 0
                                                     for x in mine if x["study_uid"] not in ("9.9.4", "9.9.5")))
    check("журнал: фильтр по нарушению не раскрывает скрытое",
          not [x for x in reg.search(violation="укладка")["studies"] if x["study_uid"].startswith("9.9.")])
    d = reg.study("9.9.0", org, job_token=lambda j: "tok")
    check("карточка: без вердикта и без ссылки на результат", all(im.get("hidden") and not im["quality_class"] and not im["card_url"]
                                                                  for im in d["images"]))
    check("отчёт до завершения закрыт", not er.report_allowed(o["id"], "Организатор"))
    check("завершить, не оценив все, нельзя", raises(ValueError, er.finish, o["id"], org))
    lg = er.create_set(doc, 8, seed=2)
    check("выборка из журнала не берёт снимки незавершённой слепой проверки",
          all(not it["study_uid"].startswith("9.9.") for it in er.get_set(lg["id"], blind=False)["items"]))
    for it in o["items"]:
        crit = {c["code"]: "violation" if it["pos"] == 1 else "ok" for c in it["criteria"]}
        er.answer(o["id"], it["pos"], org, crit)
    check("до завершения ответ можно поправить", er.answer(o["id"], 1, org, {c["code"]: "ok" for c in o["items"][0]["criteria"]})["ok"])
    check("завершение", er.finish(o["id"], org)["ok"])
    check("после завершения ответ не меняется", raises(ValueError, er.answer, o["id"], 1, org, {c["code"]: "ok" for c in o["items"][0]["criteria"]}))
    check("отчёт открылся", er.report_allowed(o["id"], "Организатор"))
    rp2 = er.report(o["id"])
    check("отчёт: таблица по каждому снимку", len(rp2["per_image"]) == 4 and all(x["expert"] == "норма" for x in rp2["per_image"]))
    sr2 = [x for x in reg.search()["studies"] if x["study_uid"] in ("9.9.0", "9.9.3")]
    check("журнал после завершения показывает решение сервиса", sr2 and all(x["verdict"] == "violation" for x in sr2), str([x["verdict"] for x in sr2]))
    check("журнал: оценки эксперта в карточке", len(er.for_study("9.9.0")) == 1)
    doc2 = reg.check_session("open|doctor|%D0%98%D0%B2%D0%B0%D0%BD%D0%BE%D0%B2")  # «Иванов»
    for it in o["items"]:
        er.answer(o["id"], it["pos"], doc2, {c["code"]: "violation" for c in it["criteria"]})
    ir = er.report(o["id"])["inter_reader"]
    check("согласие экспертов между собой посчитано", len(ir) == 1 and ir[0]["n"] == 4 and ir[0]["agreement"] == 0.0, str(ir))
    d2 = reg.study("9.9.0", org, reveal=True)
    check("открытый стенд: ФИО не раскрываются", d2["pii"] and d2["pii"]["patient_name"] == "скрыто на открытом стенде")
    print("\nИТОГ:", "OK" if not FAILED else f"FAIL ({len(FAILED)})")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
