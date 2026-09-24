#!/usr/bin/env python3
"""Одностраничный паспорт решения DensitoAI (пункт О). Числа — только из models/metrics_summary.json,
docs/METRICS_REPORT.md, docs/PERFORMANCE.md (см. сноски внизу страницы)."""
import json
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from reportlab.lib.styles import ParagraphStyle

F = "/usr/share/fonts/truetype/noto/"
pdfmetrics.registerFont(TTFont("NS", F + "NotoSans-Regular.ttf"))
pdfmetrics.registerFont(TTFont("NSB", F + "NotoSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("NSS", F + "NotoSans-SemiBold.ttf"))

TEXT = colors.HexColor("#28251D"); MUTED = colors.HexColor("#6B6A65"); ACC = colors.HexColor("#01696F")
LINE = colors.HexColor("#D4D1CA"); SURF = colors.HexColor("#F3F2EE")

_MS = [Path(__file__).resolve().parents[2] / "models" / "metrics_summary.json", Path(__file__).with_name("metrics_summary.json")]
M = json.load(open(next(p for p in _MS if p.exists()), encoding="utf-8"))

s_title = ParagraphStyle("t", fontName="NSB", fontSize=17, leading=20, textColor=TEXT)
s_sub = ParagraphStyle("s", fontName="NS", fontSize=8.6, leading=11.2, textColor=MUTED)
s_h = ParagraphStyle("h", fontName="NSB", fontSize=9.6, leading=12, textColor=ACC, spaceBefore=5, spaceAfter=2)
s_b = ParagraphStyle("b", fontName="NS", fontSize=8.2, leading=10.6, textColor=TEXT)
s_cell = ParagraphStyle("c", fontName="NS", fontSize=7.9, leading=9.6, textColor=TEXT)
s_cellb = ParagraphStyle("cb", fontName="NSS", fontSize=7.9, leading=9.6, textColor=TEXT)
s_note = ParagraphStyle("n", fontName="NS", fontSize=6.9, leading=8.6, textColor=MUTED)


def P(t, st=s_b):
    return Paragraph(t, st)


def fmt(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def build(out):
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=11 * mm,
                            bottomMargin=9 * mm, title="DensitoAI — паспорт решения (ЛЦТ 2026, задача 4)",
                            author="Perplexity Computer")
    W = A4[0] - 28 * mm
    st = []
    st.append(P("DensitoAI — паспорт решения", s_title))
    st.append(P("ЛЦТ 2026, задача 4 (ДЗМ / ЦДиТ): оценка технического качества денситометрических исследований. "
                "Команда DensitoAI. Версия сервиса 2.4.0, хэш конфигурации 1f12392d7373, 24.09.2026.", s_sub))
    st.append(Spacer(1, 3))

    st.append(P("Задача и результат", s_h))
    st.append(P("На вход — DICOM денситометра GE Lunar Prodigy (папка, zip, отдельные файлы или приём по DICOM). На выход — "
                "CSV строго в 9 колонках по ТЗ п. 2.5, плюс XLSX, технический CSV, DICOM SR и PNG-оверлеи. Две области "
                "(поясничный отдел позвоночника, проксимальный отдел бедра), пять критериев: укладка, ось позвоночника, "
                "посторонние предметы; укладка и область интереса бедра. Полностью локально: Docker на CPU, без внешних API."))

    st.append(P("Как устроено", s_h))
    st.append(P("Два независимых контура на каждый критерий: А — измеренная геометрия кости (ось, центр, края поля, "
                "металл; логистическая регрессия на 1–5 признаках), Б — эмбеддинги свёрточной сети. Ранговое усреднение "
                "0,5 / 0,5, порог по распространённости нарушения на обучающей выборке, зона «не уверен» у порога. "
                "Каждое решение объяснимо: карточка показывает скор и порог каждого критерия и измерения в градусах и мм."))

    st.append(P("Качество (out-of-fold, 100 исследований / 499 файлов заказчика; ДИ 95 % — бутстрап по исследованиям)", s_h))
    rows = [["Критерий", "n / с нарушением", "ROC-AUC", "F1", "ДИ 95 % F1", "Порог"]]
    names = {"sp_pos": "Укладка, позвоночник", "sp_axis": "Ось позвоночника", "sp_art": "Посторонние предметы",
             "hip_pos": "Укладка, бедро", "hip_roi": "Область интереса, бедро"}
    for reg, crit in (("spine", "sp_pos"), ("spine", "sp_axis"), ("spine", "sp_art"), ("hip", "hip_pos"), ("hip", "hip_roi")):
        m = M[reg][crit]
        rows.append([P(names[crit], s_cellb), P(f"{m['n_valid']} / {m['n_pos']}", s_cell), P(fmt(m["auc_stacked"]), s_cell),
                     P(fmt(m["f1_oof"]), s_cell), P(f"[{fmt(m['f1_ci_lo'], 2)}; {fmt(m['f1_ci_hi'], 2)}]", s_cell),
                     P(fmt(m["threshold"]), s_cell)])
    t = Table(rows, colWidths=[W * 0.30, W * 0.16, W * 0.12, W * 0.10, W * 0.18, W * 0.14])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "NSS"), ("FONTSIZE", (0, 0), (-1, 0), 7.6), ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("BACKGROUND", (0, 0), (-1, 0), SURF), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2)]))
    st.append(t)
    st.append(Spacer(1, 2))
    st.append(P("Задача «есть нарушение» целиком: позвоночник ROC-AUC 0,813, F1 0,662; бедро ROC-AUC 0,760, F1 0,516. "
                "Положительных примеров мало (укладка позвоночника — 10 кадров в 6 исследованиях), поэтому интервалы широкие; "
                "мы их показываем, а не прячем.", s_b))

    st.append(P("Как проверяли", s_h))
    st.append(P("Любое изменение модели, порога или признака принималось только через вложенную повторную кросс-валидацию "
                "по группам (5 внешних фолдов, 10–20 повторов, группа = исследование и хэш пикселей, чтобы клоны кадров не "
                "попадали в обе части) и калиброванную парную проверку. Одно исключение указано явно: новый признак укладки "
                "позвоночника дал прирост ROC-AUC +0,119 во всех 20 повторах, но у парной проверки на 6 положительных "
                "исследованиях не хватает мощности (нижняя граница ДИ90 = -0,004) — признак принят решением команды, и это "
                "написано в отчётах. Дополнительно: поверка измерений на фантомах (ось — средняя ошибка 0,35°, центр — "
                "0,09 мм), стресс-набор битых и нестандартных входов (18 / 18), проверка на 904 снимках чужих аппаратов; впечатанная разметка "
                "денситометра убирается до оценки (на 499 кадрах с синтетической разметкой смена класса 29,3 % без очистки и 4,2 % с очисткой)."))

    st.append(P("Что в поставке", s_h))
    st.append(P("Docker-образ (CPU) и исходники с контрольными суммами; офлайн-проверка «из коробки» без сети (18 / 18 проверок); "
                "REST API с кодом доступа к результатам; веб-кабинет врача и лаборанта: очередь по риску, карточка решения, "
                "слой сегментации, предложение области интереса бедра с решением специалиста, сводка по партии для "
                "заведующего, журнал исследований с поиском по ФИО и дате, статусами и комментариями (ФИО маскированы, "
                "просмотр фиксируется); DICOM SR и SC-серия; приём по DICOM; отказ по неподдерживаемой области и чужому аппарату."))

    st.append(P("Скорость и ресурсы", s_h))
    st.append(P("Один снимок на боевом сервере: медиана 0,54 с, 95-й перцентиль 1,76 с (позвоночник — 2,85 с); 499 файлов — 368 с; "
                "память процесса 536 МиБ, образ 2,57 ГБ; офлайн-проверка проходит при ограничении 2 vCPU и 3 ГБ памяти."))

    st.append(P("Чего мы не обещаем", s_h))
    st.append(P("Это не медицинское изделие и не диагноз: сервис оценивает техническое качество укладки и области интереса, "
                "решение принимает врач. Обучение и проверка — на одном аппарате (GE Lunar Prodigy) одной организации, "
                "252 уникальных кадра; на других аппаратах результат не определён (на 904 чужих PNG сервис без фильтра "
                "аппарата давал 70–92 % «нарушений», поэтому такие снимки теперь получают отказ). Пропускную способность "
                "отделения и экономию пересъёмок не обещаем — это требует проверки в работе."))

    st.append(Spacer(1, 4))
    st.append(P("Источники чисел: models/metrics_summary.json, docs/METRICS_REPORT.md, docs/NESTED_GATE_REPORT.md (часть 5), "
                "docs/MEASUREMENT_CHECK.md, docs/PERFORMANCE.md, docs/EXTERNAL_DXA.md — в репозитории решения; "
                "демо-стенд https://neuropeople.pro.", s_note))
    doc.build(st)


if __name__ == "__main__":
    build(str(Path(__file__).with_name("DensitoAI_passport.pdf")))
