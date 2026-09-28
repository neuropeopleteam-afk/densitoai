#!/usr/bin/env python3
"""Слепая проверка на своих снимках (own_upload), 2.4.1: P3 и счётчик no_frame.

P3. Решения сервиса по набору own_upload открываются только после того, как оценку завершил загрузивший (создатель
    набора), независимо от того, кто запрашивает. Сценарий с двумя участниками через HTTP-маршруты mount:
      * второй участник оценил все снимки и нажал «Завершить», создатель — нет -> отчёт, answers.csv и
        report_allowed закрыты обоим; в выдаче набора нет решений сервиса;
      * создатель завершил -> открыто обоим.
    До 2.4.1 второй участник, пройдя набор, видел ответы модели до завершения загрузившим.
no_frame. Текст ошибки «нет снимков для оценки» называет число строк без сохранённого кадра.

Запуск:  python tests/test_expert_owner_report.py     (код 0 — пройдено)
"""
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


def index_job(reg, jobs: Path, job: str, n: int, frames: int) -> None:
    rows = []
    for i in range(n):
        viol = i % 2 == 0
        rows.append({"path_to_study": f"f{i}.dcm", "study_uid": f"1.9.{job[-3:]}.{i}", "image_uid": f"1.9.{job[-3:]}.{i}.1",
                     "anatomical_region": "Поясничный отдел позвоночника", "quality_class": "1" if viol else "0",
                     "violation_type": "Некорректная укладка" if viol else "", "quality_prob": "0.9" if viol else "0.1",
                     "processing_status": "Success", "region_supported": True, "details": {"internal_region": "spine"}})
    reg.index_rows(job, rows, {})
    (jobs / job / "bonus").mkdir(parents=True, exist_ok=True)
    for i in range(frames):
        (jobs / job / "bonus" / f"row{i:04d}_frame.png").write_bytes(b"\x89PNG" + bytes([i]) * 8)


def main() -> int:
    d = Path(tempfile.mkdtemp(prefix="densito_owner_"))
    jobs = d / "jobs"
    reg = R.Registry(d)
    er = E.ExpertReview(reg, jobs)
    reg.hidden_jobs = er.hidden_jobs
    app = FastAPI()
    R.mount(app, reg, lambda job: None)
    E.mount(app, reg, er)
    c = TestClient(app)

    creator_name, second_name = "Петрова Анна Сергеевна", "Сидоров Павел Олегович"
    creator, second = reg.check_session(sess(creator_name)), reg.check_session(sess(second_name))
    H1, H2 = {"X-Registry-Session": sess(creator_name)}, {"X-Registry-Session": sess(second_name)}

    print("1. счётчик no_frame в тексте ошибки")
    index_job(reg, jobs, "20260927_100000_aaa000", 3, 0)
    try:
        er.create_set_from_job(creator, "20260927_100000_aaa000")
        msg = ""
    except ValueError as e:
        msg = str(e)
    check("ошибка называет строки без кадра", "нет снимков для оценки" in msg and "без сохранённого кадра 3" in msg, msg)

    print("2. отчёт своей проверки: второй участник закончил раньше создателя")
    job = "20260927_110000_bbb111"
    index_job(reg, jobs, job, 4, 4)
    s = er.create_set_from_job(creator, job, "Свои снимки", seed=5)
    sid, n = s["id"], s["n"]
    check("набор создан на 4 снимка", n == 4, str(n))
    ans = {"sp_pos": "ok", "sp_axis": "ok", "sp_art": "violation"}
    for pos in range(1, n + 1):
        r = c.post(f"/api/expert/sets/{sid}/answers", json={"pos": pos, "answers": ans}, headers=H2)
        check(f"второй: ответ {pos}", r.status_code == 200, r.text[:200])
    r = c.post(f"/api/expert/sets/{sid}/finish", headers=H2)
    check("второй: «Завершить»", r.status_code == 200, r.text[:200])
    check("report_allowed для второго закрыт", not er.report_allowed(sid, second_name))
    check("report_allowed для создателя закрыт", not er.report_allowed(sid, creator_name))
    for who, h in (("второму", H2), ("создателю", H1)):
        r = c.get(f"/api/expert/sets/{sid}/report", headers=h)
        check(f"отчёт закрыт {who} (403)", r.status_code == 403, f"{r.status_code} {r.text[:120]}")
        r = c.get(f"/api/expert/sets/{sid}/report", params={"reviewer": second_name}, headers=h)
        check(f"отчёт по ответам второго закрыт {who} (403)", r.status_code == 403, f"{r.status_code}")
        r = c.get(f"/api/expert/sets/{sid}/answers.csv", headers=h)
        check(f"answers.csv закрыт {who} (403)", r.status_code == 403, f"{r.status_code}")
        r = c.get(f"/api/expert/sets/{sid}", headers=h)
        body = r.text
        check(f"выдача набора {who} без решений сервиса", r.status_code == 200 and "service_class" not in body
              and "Некорректная" not in body and '"study_uid":null' in body.replace(" ", ""), body[:200])
    lst = c.get("/api/expert/sets", headers=H2).text
    check("список наборов без решений сервиса", "Некорректная" not in lst and "service_class" not in lst)
    card = reg.search("", limit=200)
    mine = [x for x in card["studies"] if job in x["jobs"]]
    check("журнал: решения загрузки скрыты", mine and all(x["verdict"] == "hidden" for x in mine))

    print("3. создатель завершил — отчёт открыт обоим")
    for pos in range(1, n + 1):
        r = c.post(f"/api/expert/sets/{sid}/answers", json={"pos": pos, "answers": ans}, headers=H1)
        check(f"создатель: ответ {pos}", r.status_code == 200, r.text[:200])
    check("создатель ещё не нажал «Завершить» — закрыто", c.get(f"/api/expert/sets/{sid}/report", headers=H2).status_code == 403)
    r = c.post(f"/api/expert/sets/{sid}/finish", headers=H1)
    check("создатель: «Завершить»", r.status_code == 200, r.text[:200])
    check("report_allowed открыт обоим", er.report_allowed(sid, second_name) and er.report_allowed(sid, creator_name))
    for who, h in (("второму", H2), ("создателю", H1)):
        r = c.get(f"/api/expert/sets/{sid}/report", headers=h)
        check(f"отчёт открыт {who} (200)", r.status_code == 200 and r.json()["n_answers"] == 2 * n, f"{r.status_code} {r.text[:120]}")
        r = c.get(f"/api/expert/sets/{sid}/answers.csv", headers=h)
        check(f"answers.csv открыт {who} (200)", r.status_code == 200 and "нарушение" in r.text, f"{r.status_code}")
    mine = [x for x in reg.search("", limit=200)["studies"] if job in x["jobs"]]
    check("журнал: решения открылись", mine and all(x["verdict"] != "hidden" for x in mine))

    print("4. выборка из журнала (не own_upload) — отчёт открыт как раньше")
    job3 = "20260927_120000_ccc222"
    index_job(reg, jobs, job3, 8, 8)
    s3 = er.create_set(creator, 4, seed=2)
    check("create_set: report_allowed открыт", er.report_allowed(s3["id"], second_name))

    print("\nИТОГ:", "OK" if not FAILED else f"FAIL ({len(FAILED)}): " + "; ".join(FAILED))
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
