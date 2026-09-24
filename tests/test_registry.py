#!/usr/bin/env python3
"""Журнал исследований (src/registry.py): доступ, поиск, маскирование ПДн, статусы, комментарии, аудит.

Запуск:  python tests/test_registry.py      (код 0 — все проверки пройдены)
Проверки маршрутов API выполняются, если импортируется api_server (в образе — да).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import registry as R  # noqa: E402

FAILED: list = []


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


def rows(n_study: str, names):
    out = []
    for i, (f, cls, viol, prob, reg) in enumerate(names):
        out.append({"path_to_study": f, "study_uid": n_study, "image_uid": f"{n_study}.{i}",
                    "anatomical_region": "Поясничный отдел позвоночника" if reg == "spine" else "Проксимальный отдел бедра",
                    "quality_class": cls, "violation_type": viol, "quality_prob": prob, "processing_status": "Success",
                    "region_supported": True,
                    "details": {"internal_region": reg, "criteria": [{"code": "x", "uncertain": prob == "0.55"}]}})
    return out


def main() -> int:
    d = Path(tempfile.mkdtemp(prefix="densito_reg_"))
    reg = R.Registry(d)
    print("1. выключен без учётных записей")
    check("enabled() = False", reg.enabled() is False)
    check("вход без учётных записей запрещён", raises(PermissionError, reg.login, "x", "y"))

    print("2. учётные записи и сессии")
    reg.add_user("vrach", "Pass-1234", "Петрова А. В.", "doctor")
    reg.add_user("lab", "Pass-5678", "Сидоров К. Л.", "lab")
    reg.add_user("adm", "Pass-9999", "Админ", "admin")
    check("enabled() = True", reg.enabled())
    check("пароль хранится хэшем", "Pass-1234" not in (d / "registry_users.json").read_text(encoding="utf-8"))
    check("права файла учётных записей 0600", oct((d / "registry_users.json").stat().st_mode & 0o777) == "0o600")
    check("короткий пароль отклонён", raises(ValueError, reg.add_user, "u2", "123", "", "lab"))
    check("неверный пароль", raises(PermissionError, reg.login, "vrach", "wrong"))
    s_doc = reg.login("vrach", "Pass-1234")["session"]
    s_lab = reg.login("lab", "Pass-5678")["session"]
    s_adm = reg.login("adm", "Pass-9999")["session"]
    doc, lab, adm = reg.check_session(s_doc), reg.check_session(s_lab), reg.check_session(s_adm)
    check("сессия врача", doc["role"] == "doctor")
    check("подделанная сессия отклонена", raises(PermissionError, reg.check_session, s_doc[:-4] + "AAAA"))
    check("пустая сессия отклонена", raises(PermissionError, reg.check_session, ""))
    for _ in range(R.LOCK_AFTER):
        raises(PermissionError, reg.login, "lab", "bad")
    check("блокировка после 5 неудачных попыток", raises(PermissionError, reg.login, "lab", "Pass-5678"))
    reg._fails.clear()

    print("3. индексирование и поиск")
    tags = {0: {"PatientName": "Иванова^Мария^Петровна", "PatientID": "PID004821", "PatientBirthDate": "19580312",
                "StudyDate": "20260915", "AccessionNumber": "ACC77123", "StationName": "PRODIGY1"},
            1: {"PatientName": "Иванова^Мария^Петровна", "PatientID": "PID004821", "StudyDate": "20260915"}}
    reg.index_rows("20260915_100000_aaaaaa", rows("1.2.3", [("a.dcm", "1", "Некорректная укладка", "0.93", "spine"),
                                                             ("b.dcm", "0", "", "0.2", "hip")]), tags)
    reg.index_rows("20260920_100000_bbbbbb", rows("1.2.4", [("c.dcm", "0", "", "0.55", "spine")]),
                   {0: {"PatientName": "Семёнов^Пётр", "PatientID": "77", "StudyDate": "20260920"}})
    reg.index_rows("20260921_100000_cccccc", rows("1.2.5", [("d.dcm", "0", "", "0.1", "hip")]),
                   {0: {"PatientName": "Anonymized", "PatientID": "Anonymized", "StudyDate": "Anonymized"}})
    r = reg.search("иванова")
    check("поиск по фамилии в нижнем регистре", r["total"] == 1 and r["studies"][0]["study_uid"] == "1.2.3")
    check("поиск «семенов» находит «Семёнов» (ё → е)", reg.search("семенов")["total"] == 1)
    check("поиск по имени и фамилии", reg.search("мария иванова")["total"] == 1)
    check("поиск по номеру направления", reg.search("acc77")["total"] == 1)
    check("поиск по имени файла", reg.search("c.dcm")["total"] == 1)
    check("фильтр по дате (ДД.ММ.ГГГГ)", reg.search("", "16.09.2026", "30.09.2026")["total"] == 2)
    check("фильтр по дате до", reg.search("", "", "15.09.2026")["total"] == 1)
    check("неверная дата -> ValueError", raises(ValueError, reg.search, "", "31-31-2026"))
    check("фильтр «с нарушением»", [s["study_uid"] for s in reg.search(result="violation")["studies"]] == ["1.2.3"])
    check("фильтр «не уверен»", [s["study_uid"] for s in reg.search(result="uncertain")["studies"]] == ["1.2.4"])
    check("фильтр по области hip", {s["study_uid"] for s in reg.search(region="hip")["studies"]} == {"1.2.3", "1.2.5"})
    check("фильтр по типу нарушения", reg.search(violation="укладка")["total"] == 1)
    s0 = reg.search("иванова")["studies"][0]
    check("ФИО в списке маскировано", s0["patient"] == "Иванова М. П.", s0["patient"])
    check("ID в списке маскирован", s0["patient_id"] == "…4821", s0["patient_id"])
    check("полного ФИО и даты рождения нет в выдаче поиска",
          "Мария" not in str(reg.search("")) and "19580312" not in str(reg.search("")))
    check("«Anonymized» показано как «обезличено»",
          next(s for s in reg.search("")["studies"] if s["study_uid"] == "1.2.5")["patient"] == "обезличено")
    check("повторная индексация той же задачи не дублирует строки",
          reg.index_rows("20260915_100000_aaaaaa", rows("1.2.3", [("a.dcm", "1", "Некорректная укладка", "0.93", "spine"),
                                                                   ("b.dcm", "0", "", "0.2", "hip")]), tags) == 2
          and reg.search("иванова")["studies"][0]["n_images"] == 2)

    print("4. карточка, раскрытие ПДн и аудит")
    st = reg.study("1.2.3", doc, job_token=lambda j: "TOK")
    check("карточка без ПДн по умолчанию", st["pii"] is None and "Мария" not in str(st))
    check("ссылка на карточку задачи с кодом доступа", st["images"][0]["card_url"].endswith("/TOK"))
    st = reg.study("1.2.3", doc, reveal=True)
    check("раскрытие ПДн по запросу", st["pii"]["patient_name"] == "Иванова Мария Петровна" and st["pii"]["birth_date"] == "12.03.1958")
    au = reg.audit_list()
    check("просмотр ПДн записан в журнал доступа", any(a["action"] == "pii_view" and a["login"] == "vrach" for a in au))
    check("неизвестное исследование -> KeyError", raises(KeyError, reg.study, "9.9.9", doc))

    print("5. комментарии и статусы")
    c1 = reg.add_comment("1.2.3", doc, "Переснять: ротация бедра, стопа не внутрь", "1.2.3.1")
    c2 = reg.add_comment("1.2.3", lab, "Переснято 16.09, загружено повторно")
    check("комментарии врача и лаборанта", c1["role"] == "doctor" and c2["role"] == "lab")
    check("пустой комментарий отклонён", raises(ValueError, reg.add_comment, "1.2.3", doc, "   "))
    check("слишком длинный комментарий отклонён", raises(ValueError, reg.add_comment, "1.2.3", doc, "x" * (R.MAX_COMMENT + 1)))
    check("комментарий к чужому снимку отклонён", raises(ValueError, reg.add_comment, "1.2.3", doc, "t", "1.2.4.0"))
    check("HTML сохраняется как текст", reg.add_comment("1.2.4", doc, "<script>alert(1)</script>")["text"].startswith("<script>"))
    check("врач: «нужна пересъёмка»", reg.set_status("1.2.3", doc, "retake")["status"] == "retake")
    check("лаборант не может «принять»", raises(PermissionError, reg.set_status, "1.2.3", lab, "accepted"))
    check("лаборант: «переснято»", reg.set_status("1.2.3", lab, "retaken")["status"] == "retaken")
    check("неизвестный статус", raises(ValueError, reg.set_status, "1.2.3", doc, "zzz"))
    s0 = reg.search("иванова")["studies"][0]
    check("статус и число комментариев в списке", s0["status"] == "retaken" and s0["n_comments"] == 2)
    check("фильтр по статусу", reg.search(status="retaken")["total"] == 1 and reg.search(status="new")["total"] == 2)
    check("фильтр «есть комментарии»", reg.search(has_comments="1")["total"] == 2)
    st = reg.study("1.2.3", doc)
    check("история статусов", [x["status"] for x in st["status_log"]] == ["retake", "retaken"])
    check("выгрузка CSV без полного ФИО", "Мария" not in reg.export_csv(reg.search("")["studies"]))

    print("5б. открытый режим (DENSITO_REGISTRY_OPEN=1)")
    d2 = Path(tempfile.mkdtemp(prefix="densito_reg_open_"))
    os.environ["DENSITO_REGISTRY_OPEN"] = "1"
    try:
        ro = R.Registry(d2)
        check("открытый режим включает журнал без учётных записей", ro.enabled() and ro.open_mode)
        u = ro.check_session("open|lab|%D0%A1%D0%B8%D0%B4%D0%BE%D1%80%D0%BE%D0%B2")
        check("роль и имя из заголовка", u["role"] == "lab" and u["name"] == "Сидоров", str(u))
        check("admin в открытом режиме недоступен", ro.check_session("open|admin|x")["role"] == "doctor")
        check("пустой заголовок — врач", ro.check_session("")["role"] == "doctor")
        check("управляющие символы и теги из имени убраны", "<" not in ro.check_session("open|doctor|%3Cb%3E%00x")["name"])
        ro.index_rows("20260915_100000_dddddd", rows("9.9.1", [("z.dcm", "1", "Некорректная укладка", "0.9", "spine")]),
                      {0: {"PatientName": "Петров^Иван", "StudyDate": "20260915"}})
        ro.add_comment("9.9.1", u, "проверка")
        check("комментарий в открытом режиме подписан", ro.study("9.9.1", u)["comments"][0]["author"] == "Сидоров")
        check("лаборант в открытом режиме не может «принять»", raises(PermissionError, ro.set_status, "9.9.1", u, "accepted"))
    finally:
        os.environ.pop("DENSITO_REGISTRY_OPEN", None)

    print("6. маршруты API")
    try:
        os.environ["DENSITO_OUTPUT_DIR"] = str(Path(tempfile.mkdtemp(prefix="densito_reg_api_")))
        import api_server as A  # noqa: E402
    except Exception as e:  # noqa: BLE001
        print(f"  SKIP api_server не импортируется в этом окружении ({type(e).__name__})")
    else:
        paths = {getattr(r, "path", "") for r in A.app.routes}
        for p in ("/api/registry/status", "/api/registry/login", "/api/registry/studies",
                  "/api/registry/studies/{study_uid}", "/api/registry/studies/{study_uid}/comments",
                  "/api/registry/studies/{study_uid}/status", "/api/registry/audit"):
            check(f"маршрут {p}", p in paths)
        # FastAPI должен видеть Request/Header как служебные параметры, а не как параметры запроса (иначе 422)
        for r in A.app.routes:
            if str(getattr(r, "path", "")).startswith("/api/registry/") and hasattr(r, "dependant"):
                q = [p.name for p in r.dependant.query_params]
                bad = [n for n in q if n in ("request", "x_registry_session")]
                check(f"{r.path}: служебные параметры не в строке запроса", not bad, str(bad))
        from fastapi import HTTPException
        fn = next(r.endpoint for r in A.app.routes if getattr(r, "path", "") == "/api/registry/studies")
        try:
            fn(x_registry_session=None)
            ok = False
        except HTTPException as e:
            ok = e.status_code == 503
        check("без учётных записей журнал закрыт (503)", ok)
        A.REGISTRY.add_user("t1", "Pass-0000", "Т", "doctor")
        try:
            fn(x_registry_session="bad")
            ok = False
        except HTTPException as e:
            ok = e.status_code == 401
        check("без сессии — 401", ok)
        st = A.registry_status() if hasattr(A, "registry_status") else next(
            r.endpoint for r in A.app.routes if getattr(r, "path", "") == "/api/registry/status")()
        check("статус журнала не раскрывает учётные записи", "t1" not in str(st))

    print("\nИТОГ:", "OK" if not FAILED else f"FAIL ({len(FAILED)}): " + "; ".join(FAILED))
    return 0 if not FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
