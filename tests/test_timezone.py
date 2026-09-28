#!/usr/bin/env python3
"""Часовой пояс контейнера (2.4.1, Б7).

В образе python:slim часовой пояс — UTC, а журнал, экспертная проверка, приёмник DICOM и карточки запросов
пишут время через time.strftime (местное). Врач в Москве видел время на 3 часа раньше. Исправление: в Dockerfile
ENV TZ=MSK-3 (POSIX-строка, работает без tzdata). Кабинет (web/index.html) брал дату и время истории через
toISOString — это UTC, и сравнивал с местными датами журнала; теперь — местное время браузера (localIso).

Проверяется:
  * Dockerfile задаёт TZ=MSK-3 в ENV;
  * при TZ=MSK-3 registry._now(), expert_review._now(), dicom_receiver._now() = UTC+3 (в отдельном процессе);
  * в src/*.py нет явного UTC (gmtime, utcnow, utcfromtimestamp, timezone.utc), которое сравнивалось бы с местным;
  * в web/index.html нет toISOString (UTC) — дата истории и фильтры «за N дней» в местном времени;
  * localIso() из web/index.html при TZ=MSK-3 даёт UTC+3 (если в системе есть node; иначе пункт пропускается).

Запуск:  python tests/test_timezone.py     (код 0 — пройдено)
"""
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILED = []


def check(name, cond, detail=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def dockerfile_env() -> dict:
    """Переменные из инструкций ENV (с продолжениями строк «\\» и комментариями внутри блока)."""
    env, cont = {}, False
    for ln in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines():
        st = ln.strip()
        if cont and st.startswith("#"):
            continue  # Docker пропускает строки-комментарии внутри продолжения
        if st.startswith("ENV ") or cont:
            body = st[4:] if st.startswith("ENV ") and not cont else st
            cont = body.endswith("\\")
            for k, v in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=(\S+)", body.rstrip("\\")):
                env[k] = v
        else:
            cont = False
    return env


def main() -> int:
    print("1. Dockerfile")
    env = dockerfile_env()
    check("ENV TZ=MSK-3", env.get("TZ") == "MSK-3", str(env.get("TZ")))

    print("2. _now() при TZ=MSK-3 = UTC+3")
    code = ("import sys, json, time; sys.path.insert(0, %r); time.tzset(); import registry, expert_review, dicom_receiver;"
            "print(json.dumps({'registry': registry._now(), 'expert_review': expert_review._now(),"
            " 'dicom_receiver': dicom_receiver._now(), 'utc': time.strftime('%%Y-%%m-%%dT%%H:%%M:%%S', time.gmtime())}))"
            % str(ROOT / "src"))
    e = dict(os.environ, TZ=env.get("TZ") or "UTC", OMP_NUM_THREADS="1")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=e, timeout=120)
    check("процесс с TZ из Dockerfile отработал", p.returncode == 0, p.stderr[-300:])
    if p.returncode == 0:
        got = json.loads(p.stdout.strip().splitlines()[-1])
        utc = dt.datetime.fromisoformat(got.pop("utc"))
        for mod, s in got.items():
            delta = (dt.datetime.fromisoformat(s) - utc).total_seconds()
            check(f"{mod}._now() = UTC+3", abs(delta - 3 * 3600) <= 5, f"{s} против UTC {utc}")

    print("3. явного UTC в коде нет")
    bad = []
    for f in sorted((ROOT / "src").glob("*.py")):
        for i, ln in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            code_part = ln.split("#", 1)[0]
            if re.search(r"\bgmtime\b|\butcnow\b|\butcfromtimestamp\b|timezone\.utc|datetime\.UTC\b", code_part):
                bad.append(f"{f.name}:{i}")
    check("src/*.py: без gmtime/utcnow/utcfromtimestamp/timezone.utc", not bad, str(bad))
    web = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    web_code = "\n".join(ln.split("//", 1)[0] if ln.strip().startswith("//") else ln for ln in web.splitlines())
    uses = [m.start() for m in re.finditer(r"\.toISOString\(", web_code)]
    check("web/index.html: без toISOString (UTC) в коде", not uses, f"{len(uses)} мест")
    check("web/index.html: localIso() для истории и фильтров", "created_at: localIso()" in web and "localIso(new Date(Date.now() - n * 864e5))" in web)

    print("4. localIso() в браузерном коде")
    node = shutil.which("node")
    m = re.search(r"^function localIso\(d\) \{.*\}$", web, flags=re.M)
    if not node:
        print("  SKIP node не установлен")
    elif not m:
        check("функция localIso найдена", False)
    else:
        js = m.group(0) + "\nconst d=new Date(); console.log(JSON.stringify({l: localIso(d), u: d.toISOString().slice(0,19)}));"
        q = subprocess.run([node, "-e", js], capture_output=True, text=True, env=dict(os.environ, TZ="MSK-3"), timeout=60)
        if q.returncode != 0:
            check("node отработал", False, q.stderr[-200:])
        else:
            r = json.loads(q.stdout)
            delta = (dt.datetime.fromisoformat(r["l"]) - dt.datetime.fromisoformat(r["u"])).total_seconds()
            check("localIso() при TZ=MSK-3 = UTC+3", abs(delta - 3 * 3600) <= 2, str(r))

    print("\nИТОГ:", "OK" if not FAILED else f"FAIL ({len(FAILED)}): " + "; ".join(FAILED))
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
