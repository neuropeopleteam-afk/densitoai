#!/usr/bin/env python3
"""Второй тур «врач + сервис» (2.5) и правила слепоты 2.4.1 (P2, P3).

Через HTTP-маршруты mount (TestClient), без моделей:
  P2. загрузка помечается скрытой до индексации (hold_job): в журнале решения скрыты, пока загрузивший не завершил;
  P3. второй участник завершил раньше создателя -> второй тур закрыт ему и создателю (403), отчёт и answers.csv
      закрыты; до своего «Завершить» второй тур закрыт и создателю;
  1. после «Завершить» создателем второй тур открыт; в нём только снимки и критерии, где определённый ответ
     первого тура разошёлся с сервисом («не могу оценить» не спорный), решение сервиса — только по ним;
  2. первый тур неизменен: POST /answers после «Завершить» -> 400, строки expert_answers до и после второго тура
     совпадают, метрики сервиса в отчёте (первый тур) не меняются;
  3. второй тур: ответ по неспорному снимку -> 404, без ответа по спорному критерию -> 400, после завершения
     второго тура -> 400; пропуск допустим;
  4. отчёт: сколько изменено и в какую сторону, совпадение с сервисом до и после (числа сценария);
  5. answers.csv: первые 10 колонок прежние, 4 колонки второго тура в конце, оба тура в одной строке;
  6. выборка из журнала: второй тур только после «Завершить» этого эксперта;
  7. страница /expert/: второй тур, без утверждений о превосходстве пары «врач + сервис».

Запуск:  python tests/test_expert_second_round.py     (код 0 — пройдено)
"""
import csv
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["DENSITO_REGISTRY_OPEN"] = "1"
import registry as R  # noqa: E402
import expert_review as E  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def sess(name: str) -> str:
    return "open|doctor|" + quote(name)


def index_job(reg, jobs: Path, job: str, n: int) -> None:
    """Снимки позвоночника: чётные — «Некорректная укладка» по решению сервиса, нечётные — норма."""
    rows = []
    for i in range(n):
        viol = i % 2 == 0
        rows.append({"path_to_study": f"f{i}.dcm", "study_uid": f"1.8.{job[-3:]}.{i}", "image_uid": f"1.8.{job[-3:]}.{i}.1",
                     "anatomical_region": "Поясничный отдел позвоночника", "quality_class": "1" if viol else "0",
                     "violation_type": "Некорректная укладка" if viol else "", "quality_prob": "0.9" if viol else "0.1",
                     "processing_status": "Success", "region_supported": True, "details": {"internal_region": "spine"}})
    (jobs / job / "bonus").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (jobs / job / "bonus" / f"row{i:04d}_frame.png").write_bytes(b"\x89PNG" + bytes([i + 1]) * 8)
    reg.index_rows(job, rows, {})


def first_rows(reg, sid):
    with reg._conn() as c:
        return [tuple(r) for r in c.execute("SELECT set_id,pos,reviewer,answers,comment,at FROM expert_answers "
                                            "WHERE set_id=? ORDER BY pos, reviewer", (sid,))]


def main() -> int:
    d = Path(tempfile.mkdtemp(prefix="densito_second_"))
    jobs = d / "jobs"
    reg = R.Registry(d)
    er = E.ExpertReview(reg, jobs)
    reg.hidden_jobs = er.hidden_jobs
    app = FastAPI()
    R.mount(app, reg, lambda job: None)
    E.mount(app, reg, er)
    c = TestClient(app)
    A, B = "Петрова Анна Сергеевна", "Сидоров Павел Олегович"
    ua = reg.check_session(sess(A))
    HA, HB = {"X-Registry-Session": sess(A)}, {"X-Registry-Session": sess(B)}

    print("P2. скрытие загрузки до индексации")
    job = "20260927_130000_ddd333"
    er.hold_job(job, A)
    index_job(reg, jobs, job, 4)
    mine = [x for x in reg.search("", limit=200)["studies"] if job in x["jobs"]]
    check("журнал: решения скрыты сразу после индексации", mine and all(x["verdict"] == "hidden" for x in mine))
    s = er.create_set_from_job(ua, job, "Свои снимки", seed=3)
    sid, n = s["id"], s["n"]
    check("набор на 4 снимка", n == 4, str(n))
    svc = {it["pos"]: it["service_class"] for it in er.get_set(sid, blind=False)["items"]}

    print("P3. второй участник завершил раньше создателя")
    for pos in range(1, n + 1):
        c.post(f"/api/expert/sets/{sid}/answers", json={"pos": pos, "answers": {"sp_pos": "ok", "sp_axis": "ok", "sp_art": "ok"}}, headers=HB)
    check("второй: «Завершить»", c.post(f"/api/expert/sets/{sid}/finish", headers=HB).status_code == 200)
    for who, h in (("второму", HB), ("создателю", HA)):
        r = c.get(f"/api/expert/sets/{sid}/second", headers=h)
        check(f"второй тур закрыт {who} (403)", r.status_code == 403 and "Некорректная" not in r.text and "service" not in r.text,
              f"{r.status_code} {r.text[:120]}")
        check(f"отчёт закрыт {who} (403)", c.get(f"/api/expert/sets/{sid}/report", headers=h).status_code == 403)
        check(f"answers.csv закрыт {who} (403)", c.get(f"/api/expert/sets/{sid}/answers.csv", headers=h).status_code == 403)
    r = c.post(f"/api/expert/sets/{sid}/second/answers", json={"pos": 1, "answers": {"sp_pos": "ok"}}, headers=HB)
    check("ответ второго тура до открытия отчёта -> 403", r.status_code == 403, str(r.status_code))
    check("finish второго тура до открытия отчёта -> 403", c.post(f"/api/expert/sets/{sid}/second/finish", headers=HB).status_code == 403)

    print("1. создатель: первый тур, «Завершить», второй тур")
    # создатель: укладка — «нарушение» везде (спор там, где сервис «норма»), ось — «не могу оценить», предметы — «норма»
    ansA = {"sp_pos": "violation", "sp_axis": "unsure", "sp_art": "ok"}
    for pos in range(1, n + 1):
        check(f"создатель: ответ {pos}", c.post(f"/api/expert/sets/{sid}/answers", json={"pos": pos, "answers": ansA}, headers=HA).status_code == 200)
    check("до «Завершить» второй тур закрыт создателю", c.get(f"/api/expert/sets/{sid}/second", headers=HA).status_code == 403)
    check("создатель: «Завершить»", c.post(f"/api/expert/sets/{sid}/finish", headers=HA).status_code == 200)
    mine = [x for x in reg.search("", limit=200)["studies"] if job in x["jobs"]]
    check("журнал: решения открылись после завершения создателем", mine and all(x["verdict"] != "hidden" for x in mine))
    r = c.get(f"/api/expert/sets/{sid}/second", headers=HA)
    q = r.json() if r.status_code == 200 else {}
    disputed = sorted(p for p, k in svc.items() if k != "1")
    check("второй тур открыт создателю", r.status_code == 200, f"{r.status_code} {r.text[:120]}")
    check("во втором туре только спорные снимки", sorted(x["pos"] for x in q.get("items", [])) == disputed,
          str([x["pos"] for x in q.get("items", [])]))
    check("спорный критерий только укладка («не могу оценить» не спорный)",
          all([cc["code"] for cc in x["criteria"]] == ["sp_pos"] for x in q.get("items", [])))
    check("решение сервиса показано по спорному критерию",
          all(x["criteria"][0]["service"] == "ok" and x["criteria"][0]["first"] == "violation" for x in q.get("items", [])))
    r1_before = first_rows(reg, sid)
    rep_before = c.get(f"/api/expert/sets/{sid}/report", params={"reviewer": A}, headers=HA).json()

    print("2-3. первый тур неизменен, ответы второго тура")
    r = c.post(f"/api/expert/sets/{sid}/answers", json={"pos": 1, "answers": {"sp_pos": "ok", "sp_axis": "ok", "sp_art": "ok"}}, headers=HA)
    check("первый тур после «Завершить» не меняется (400)", r.status_code == 400, str(r.status_code))
    agreed = [p for p in svc if svc[p] == "1"][0]
    r = c.post(f"/api/expert/sets/{sid}/second/answers", json={"pos": agreed, "answers": {"sp_pos": "ok"}}, headers=HA)
    check("неспорный снимок во втором туре -> 404", r.status_code == 404, str(r.status_code))
    r = c.post(f"/api/expert/sets/{sid}/second/answers", json={"pos": disputed[0], "answers": {"sp_axis": "ok"}}, headers=HA)
    check("без ответа по спорному критерию -> 400", r.status_code == 400, str(r.status_code))
    r = c.post(f"/api/expert/sets/{sid}/second/answers", json={"pos": disputed[0], "answers": {"sp_pos": "ok"}, "comment": "согласна"}, headers=HA)
    check("изменение ответа во втором туре", r.status_code == 200, r.text[:120])
    r = c.post(f"/api/expert/sets/{sid}/second/answers", json={"pos": disputed[1], "answers": {"sp_pos": "violation"}}, headers=HA)
    check("оставить свой ответ во втором туре", r.status_code == 200, r.text[:120])
    check("строки первого тура не изменились", first_rows(reg, sid) == r1_before)
    q2 = c.get(f"/api/expert/sets/{sid}/second", headers=HA).json()
    check("второй тур хранит ответы отдельно", {x["pos"]: x["second"] for x in q2["items"]}
          == {disputed[0]: {"sp_pos": "ok"}, disputed[1]: {"sp_pos": "violation"}})

    print("4. отчёт второго тура")
    rep = c.get(f"/api/expert/sets/{sid}/report", params={"reviewer": A}, headers=HA).json()
    sr = rep["second_round"]
    check("спорных 2, пересмотрено 2, изменено 1", (sr["disputed"], sr["reviewed"], sr["changed"]) == (2, 2, 1), str(sr))
    check("направление: нарушение -> норма 1", sr["directions"]["нарушение → норма"] == 1
          and sum(sr["directions"].values()) == 1 and sr["changed_to_service"] == 1)
    # определённые ответы: укладка 4 (совпало 2) + предметы 4 (совпало 4) = 6 из 8; после второго тура 7 из 8
    check("совпадение с сервисом до: 6 из 8", (sr["agreement_before"]["agree"], sr["agreement_before"]["n"]) == (6, 8))
    check("совпадение с сервисом после: 7 из 8", (sr["agreement_after"]["agree"], sr["agreement_after"]["n"]) == (7, 8))
    check("метрики сервиса (первый тур) не изменились", rep["criteria"] == rep_before["criteria"]
          and rep["any_violation"] == rep_before["any_violation"])
    check("пояснение: второй тур не слепой", "не слепой" in sr["note"] and "только по первому туру" in sr["note"])
    check("finish второго тура", c.post(f"/api/expert/sets/{sid}/second/finish", headers=HA).status_code == 200)
    r = c.post(f"/api/expert/sets/{sid}/second/answers", json={"pos": disputed[1], "answers": {"sp_pos": "ok"}}, headers=HA)
    check("после завершения второго тура ответы не меняются (400)", r.status_code == 400, str(r.status_code))
    check("второй тур завершён у создателя", A in c.get(f"/api/expert/sets/{sid}/report", headers=HA).json()["second_round"]["reviewers_finished_second"])
    rb = c.get(f"/api/expert/sets/{sid}/second", headers=HB).json()
    check("второму участнику после завершения создателем второй тур открыт, спорные — нарушения сервиса",
          sorted(x["pos"] for x in rb["items"]) == sorted(p for p in svc if svc[p] == "1"))

    print("5. answers.csv")
    r = c.get(f"/api/expert/sets/{sid}/answers.csv", headers=HA)
    rows = list(csv.reader(io.StringIO(r.text.lstrip("\ufeff")), delimiter=";"))
    check("первые 10 колонок прежние", rows[0][:10] == ["set_id", "pos", "study_uid", "region", "reviewer", "criterion",
                                                        "expert", "service", "comment", "at"], str(rows[0]))
    check("колонки второго тура в конце", rows[0][10:] == ["round2_shown_service", "expert_round2", "comment_round2", "at_round2"])
    byk = {(int(x[1]), x[4], x[5]): x for x in rows[1:]}
    x = byk[(disputed[0], A, "sp_pos")]
    check("строка: первый тур «нарушение», второй «норма»", x[6] == "нарушение" and x[10] == "да" and x[11] == "норма" and x[12] == "согласна")
    x = byk[(agreed, A, "sp_pos")]
    check("неспорный критерий: второй тур не показывался", x[10] == "нет" and x[11] == "")
    x = byk[(disputed[0], A, "sp_axis")]
    check("«не могу оценить» не спорный", x[6] == "не могу оценить" and x[10] == "нет")
    check("все строки по 14 колонок", all(len(x) == 14 for x in rows))

    print("6. выборка из журнала")
    job3 = "20260927_140000_eee444"
    index_job(reg, jobs, job3, 8)
    s3 = er.create_set(ua, 4, seed=2)
    sid3 = s3["id"]
    check("до «Завершить» второй тур закрыт", c.get(f"/api/expert/sets/{sid3}/second", headers=HA).status_code == 403)
    for it in s3["items"]:
        c.post(f"/api/expert/sets/{sid3}/answers", json={"pos": it["pos"], "answers": {"sp_pos": "violation", "sp_axis": "ok", "sp_art": "ok"}}, headers=HA)
    c.post(f"/api/expert/sets/{sid3}/finish", headers=HA)
    r = c.get(f"/api/expert/sets/{sid3}/second", headers=HA)
    check("после «Завершить» второй тур открыт", r.status_code == 200, f"{r.status_code}")
    check("другому эксперту без своего «Завершить» закрыт", c.get(f"/api/expert/sets/{sid3}/second", headers=HB).status_code == 403)

    print("7. страница /expert/")
    html = (ROOT / "web" / "expert" / "index.html").read_text(encoding="utf-8")
    check("есть второй тур", "/second" in html and "Второй тур" in html and "function renderSecond" in html)
    check("нет утверждений о превосходстве пары", ("луч" + "ше") not in html)
    check("пропуск возможен", "Пропустить" in html)

    print("\nИТОГ:", "OK" if not FAILED else f"FAIL ({len(FAILED)}): " + "; ".join(FAILED))
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
