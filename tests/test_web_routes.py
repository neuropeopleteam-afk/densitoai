#!/usr/bin/env python3
"""Страницы веб-интерфейса и Swagger без интернета (2.4.1: Б1, Б5).

Б1. Кабинет ссылается на docs.html, страница документации — на index.html; до 2.4.1 оба адреса отдавали 404.
Б5. Встроенная страница /docs FastAPI грузила Swagger UI с cdn.jsdelivr.net и без интернета (закрытый контур
    заказчика) была пустой. Теперь swagger-ui-dist лежит в web/assets/swagger/ (Apache 2.0, LICENSE рядом):
    HTML /docs не содержит внешних адресов, скрипт/стили/иконка отдаются сервером (200), /openapi.json — как раньше.

Запуск:  python tests/test_web_routes.py     (код 0 — пройдено)
"""
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["DENSITO_OUTPUT_DIR"] = tempfile.mkdtemp(prefix="densito_webroutes_")

from fastapi.testclient import TestClient  # noqa: E402
import api_server as A  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def main() -> int:
    c = TestClient(A.app)
    print("1. страницы по именам файлов (Б1)")
    for url, marker in (("/index.html", "DensitoAI"), ("/docs.html", "index.html"), ("/", "docs.html")):
        r = c.get(url)
        check(f"{url}: 200, text/html", r.status_code == 200 and r.headers.get("content-type", "").startswith("text/html"),
              f"{r.status_code} {r.headers.get('content-type')}")
        check(f"{url}: содержимое страницы", marker in r.text)
    check("/index.html = /", c.get("/index.html").content == c.get("/").content)
    check("/docs.html = web/docs.html", c.get("/docs.html").content == (ROOT / "web" / "docs.html").read_bytes())
    # ссылки между страницами разрешаются в существующие маршруты
    for page in ("/", "/docs.html"):
        for href in set(re.findall(r'href="([a-z_]+\.html)"', c.get(page).text)):
            check(f"ссылка {page} -> {href} открывается", c.get("/" + href).status_code == 200)

    print("2. Swagger UI без внешних ресурсов (Б5)")
    r = c.get("/docs")
    html = r.text
    check("/docs: 200, text/html", r.status_code == 200 and "text/html" in r.headers.get("content-type", ""))
    ext = re.findall(r"""(?:src|href|url)\s*[:=]\s*["']?(?:https?:)?//[^"'\s>]+""", html) + re.findall(r"https?://", html)
    check("в HTML /docs нет внешних адресов (http://, https://, //)", not ext, str(ext[:3]))
    check("validatorUrl выключен (без обращения к validator.swagger.io)", '"validatorUrl": null' in html)
    assets = re.findall(r"""(?:src|href)=["'](/assets/swagger/[^"']+)["']""", html)
    check("скрипт, стили и иконка — из /assets/swagger/", {Path(a).name for a in assets} >= {
        "swagger-ui-bundle.js", "swagger-ui.css", "favicon-32x32.png"}, str(assets))
    for a in assets:
        ra = c.get(a)
        check(f"{a}: 200", ra.status_code == 200 and len(ra.content) > 500, f"{ra.status_code} {len(ra.content)}")
    js = c.get("/assets/swagger/swagger-ui-bundle.js")
    check("swagger-ui-bundle.js — javascript", "javascript" in js.headers.get("content-type", "") and b"SwaggerUIBundle" in js.content)
    check("swagger-ui.css — text/css", "text/css" in c.get("/assets/swagger/swagger-ui.css").headers.get("content-type", ""))
    sw = ROOT / "web" / "assets" / "swagger"
    lic = (sw / "LICENSE").read_text(encoding="utf-8") if (sw / "LICENSE").is_file() else ""
    check("LICENSE Apache 2.0 рядом с ассетами", "Apache License" in lic and "Version 2.0" in lic)
    ver = (sw / "VERSION.txt").read_text(encoding="utf-8") if (sw / "VERSION.txt").is_file() else ""
    check("версия swagger-ui-dist закреплена (VERSION.txt)", re.search(r"swagger-ui-dist \d+\.\d+\.\d+", ver) is not None)
    o = c.get("/openapi.json")
    check("/openapi.json: 200, есть /api/analyze", o.status_code == 200 and "/api/analyze" in o.json().get("paths", {}))
    check("служебные страницы не попали в схему API", not {"/docs", "/docs.html", "/index.html"} & set(o.json().get("paths", {})))

    print("\nИТОГ:", "OK" if not FAILED else f"FAIL ({len(FAILED)}): " + "; ".join(FAILED))
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
