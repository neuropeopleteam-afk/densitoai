#!/usr/bin/env python3
"""Экспертная проверка сервиса в отделении: слепая оценка врачом выборки из журнала исследований и отчёт о
совпадении с решениями сервиса.

Зачем. Метрики поставки посчитаны на одном аппарате и одном разметчике. Отделение должно иметь возможность само
проверить сервис на своих снимках и своих врачах, не веря разработчику на слово. Заведующий формирует проверку —
случайную выборку снимков из журнала (стратифицированную по заключению сервиса, чтобы были и «нормы», и
«нарушения»); врач оценивает каждый снимок по критериям области, не видя вердикта сервиса; отчёт по каждому
критерию показывает совпадение, чувствительность и специфичность сервиса относительно врача с 95 % интервалами
Уилсона и каппу Коэна. Ответы хранятся в той же базе журнала (OUTPUT_DIR/registry.sqlite3) и видны в карточке
исследования. Модель по ответам НЕ дообучается: их можно выгрузить CSV как разметку для следующей проверенной версии.

Кадр для слепой оценки — чистое изображение без отметок сервиса (bonus/rowNNNN_frame.png), сохраняется при
загрузке (save_frames). Для задач, загруженных до появления этой функции, кадра нет — такие снимки в выборку не
попадают.

Проверка на своих снимках (режим own_upload). Врач или организатор загружает свои DICOM на странице проверки;
сервис обрабатывает их тем же конвейером, снимки сразу попадают в журнал, но решение сервиса скрыто и в проверке, и
в журнале, пока загрузивший не оценит все снимки и не нажмёт «Завершить». После завершения ответы не меняются,
открывается отчёт, а в журнале — вердикты сервиса и оценки эксперта. В набор входят все загруженные снимки
поддерживаемой области без отбора по решению сервиса; ошибки чтения и чужие аппараты перечисляются отдельно и в
знаменатель не входят.

Второй тур «врач + сервис» (2.5). После «Завершить» ответы первого тура зафиксированы. Эксперт может открыть второй
тур: ему показывается решение сервиса только по тем критериям, где его определённый ответ («норма» или
«нарушение») разошёлся с сервисом; он оставляет свой ответ или меняет его. Ответы второго тура хранятся отдельно
(expert_second), первый тур не переписывается. Доступ — по тем же правилам слепоты: второй тур открыт только
эксперту, завершившему свой первый тур, и только когда отчёт по набору разрешён (report_allowed, P3). Отчёт
показывает, сколько ответов изменено и в какую сторону, и совпадение с сервисом до и после второго тура; основные
метрики сервиса считаются только по первому, слепому туру. Второй тур не слепой, поэтому совпадение после него —
не проверка сервиса, а описание того, как врач пользуется подсказкой.
"""
import csv
import hashlib
import io
import json
import math
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

CRITERIA = {
    "spine": [("sp_pos", "Укладка пациента", "Некорректная укладка"),
              ("sp_axis", "Ось позвоночника", "Не выравнена ось позвоночника"),
              ("sp_art", "Посторонние предметы", "Присутствуют посторонние предметы")],
    "hip": [("hip_pos", "Укладка пациента", "Некорректная укладка"),
            ("hip_roi", "Область интереса", "Некорректная область интереса")],
}
ANSWERS = {"ok": "норма", "violation": "нарушение", "unsure": "не могу оценить"}
MAX_SET = 200
FRAME_SUFFIX = "_frame.png"
OWN = "own_upload"

SCHEMA = """
CREATE TABLE IF NOT EXISTS expert_sets (
  id INTEGER PRIMARY KEY, title TEXT, created_at TEXT, created_by TEXT, params TEXT, n INTEGER
);
CREATE TABLE IF NOT EXISTS expert_items (
  set_id INTEGER, pos INTEGER, job_id TEXT, row_idx INTEGER, image_uid TEXT, study_uid TEXT, region TEXT,
  service_class TEXT, service_violations TEXT, PRIMARY KEY (set_id, pos)
);
CREATE TABLE IF NOT EXISTS expert_answers (
  id INTEGER PRIMARY KEY, set_id INTEGER, pos INTEGER, reviewer TEXT, role TEXT, answers TEXT, comment TEXT, at TEXT,
  UNIQUE (set_id, pos, reviewer)
);
CREATE TABLE IF NOT EXISTS expert_finish (
  set_id INTEGER, reviewer TEXT, at TEXT, PRIMARY KEY (set_id, reviewer)
);
CREATE TABLE IF NOT EXISTS expert_second (
  id INTEGER PRIMARY KEY, set_id INTEGER, pos INTEGER, reviewer TEXT, answers TEXT, comment TEXT, at TEXT,
  UNIQUE (set_id, pos, reviewer)
);
CREATE TABLE IF NOT EXISTS expert_second_finish (
  set_id INTEGER, reviewer TEXT, at TEXT, PRIMARY KEY (set_id, reviewer)
);
CREATE TABLE IF NOT EXISTS expert_hold (
  job_id TEXT PRIMARY KEY, at TEXT, by TEXT
);
"""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _region(internal: str) -> str:
    return "hip" if "hip" in (internal or "") else "spine"


def wilson(k: int, n: int) -> Optional[List[float]]:
    if n <= 0:
        return None
    z = 1.959964
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def kappa(tp: int, fp: int, fn: int, tn: int) -> Optional[float]:
    n = tp + fp + fn + tn
    if n == 0:
        return None
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    return None if pe >= 1 else round((po - pe) / (1 - pe), 3)


def save_frames(rows: List[Dict[str, Any]], tmp: Path, job_dir: Path, files: Optional[List[Any]] = None) -> int:
    """Сохранить чистый кадр (без отметок сервиса) каждой строки для слепой экспертной оценки.
    `files[i]` — фактический файл строки i (DensitoInference.last_row_files; для zip — файл во временном
    каталоге распаковки); без него файл ищется по path_to_study относительно каталога загрузки."""
    import cv2
    import pydicom
    import inference
    out = Path(job_dir) / "bonus"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, r in enumerate(rows):
        rel = str(r.get("path_to_study") or "")
        real = [Path(files[i])] if files and i < len(files) and files[i] else []
        cands = ([Path(tmp) / rel] + ([Path(tmp) / Path(*Path(rel).parts[1:])] if len(Path(rel).parts) > 1 else [])) if rel else []
        src = next((c for c in real + cands if c.is_file()), None)
        if src is None:
            continue
        try:
            img = inference.normalize_pixels(pydicom.dcmread(str(src), force=True))
            cv2.imwrite(str(out / f"row{i:04d}{FRAME_SUFFIX}"), img)
            n += 1
        except Exception:  # noqa: BLE001 — кадр для проверки не должен ломать ответ
            continue
    return n


class ExpertReview:
    def __init__(self, registry, jobs_dir: Path):
        self.reg = registry
        self.jobs_dir = Path(jobs_dir)
        self._lock = threading.Lock()
        with self.reg._conn() as c:
            c.executescript(SCHEMA)

    def frame_path(self, job_id: str, row_idx: int) -> Path:
        return self.jobs_dir / job_id / "bonus" / f"row{int(row_idx):04d}{FRAME_SUFFIX}"

    # ---- формирование выборки
    def create_set(self, user: Dict[str, Any], n: int = 40, region: str = "", date_from: str = "", date_to: str = "",
                   title: str = "", seed: Optional[int] = None) -> Dict[str, Any]:
        from registry import parse_date
        n = max(4, min(int(n or 40), MAX_SET))
        df, dt = parse_date(date_from), parse_date(date_to)
        with self.reg._conn() as c:
            rows = [dict(r) for r in c.execute(
                "SELECT job_id,row_idx,image_uid,study_uid,internal_region,quality_class,violation_type,study_date,created_at "
                "FROM images WHERE processing_status='Success' AND region_supported=1 ORDER BY created_at DESC")]
        hidden = self.hidden_jobs()
        seen, pool = set(), []
        for r in rows:  # последняя загрузка каждого снимка, только с сохранённым чистым кадром
            k = r["image_uid"] or f"{r['job_id']}:{r['row_idx']}"
            if k in seen:
                continue
            seen.add(k)
            if r["job_id"] in hidden:  # снимки чужой незавершённой слепой проверки не раскрываем через отчёт
                continue
            if region in ("spine", "hip") and _region(r["internal_region"]) != region:
                continue
            d = r["study_date"] or (r["created_at"] or "")[:10].replace("-", "")
            if (df and d < df) or (dt and d > dt):
                continue
            if self.frame_path(r["job_id"], r["row_idx"]).is_file():
                pool.append(r)
        if len(pool) < 4:
            raise ValueError(f"в журнале недостаточно снимков для проверки (подходит {len(pool)}; нужны снимки, "
                             f"загруженные после включения экспертной проверки)")
        rnd = random.Random(seed if seed is not None else int(time.time()))
        viol = [r for r in pool if r["quality_class"] == "1"]
        norm = [r for r in pool if r["quality_class"] != "1"]
        rnd.shuffle(viol); rnd.shuffle(norm)
        half = min(n // 2, len(viol))
        pick = viol[:half] + norm[:n - half]
        if len(pick) < n:
            pick += viol[half:half + (n - len(pick))]
        rnd.shuffle(pick)
        params = {"n": n, "region": region, "date_from": date_from, "date_to": date_to, "stratified": True,
                  "pool": len(pool), "pool_violation": len(viol)}
        with self._lock, self.reg._conn() as c:
            cur = c.execute("INSERT INTO expert_sets(title,created_at,created_by,params,n) VALUES(?,?,?,?,?)",
                            (title or f"Проверка от {time.strftime('%d.%m.%Y %H:%M')}", _now(),
                             user.get("name") or user.get("login"), json.dumps(params, ensure_ascii=False), len(pick)))
            sid = cur.lastrowid
            c.executemany("INSERT INTO expert_items VALUES(?,?,?,?,?,?,?,?,?)",
                          [(sid, i + 1, r["job_id"], r["row_idx"], r["image_uid"], r["study_uid"], _region(r["internal_region"]),
                            r["quality_class"], r["violation_type"]) for i, r in enumerate(pick)])
        self.reg.audit(user.get("login", "?"), "expert_set_create", str(sid), json.dumps(params, ensure_ascii=False))
        return self.get_set(sid, blind=True)

    def create_set_from_job(self, user: Dict[str, Any], job_id: str, title: str = "", n_files: Optional[int] = None,
                            seed: Optional[int] = None, versions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Слепая проверка на своих снимках: набор — все снимки одной загрузки (job_id), без отбора по решению
        сервиса. Решение сервиса скрыто, пока загрузивший не завершит оценку."""
        reviewer = (user.get("name") or "").strip()[:60]
        if not reviewer or reviewer == user.get("login") or reviewer.lower() in ("врач", "лаборант", "администратор"):
            raise ValueError("укажите ФИО эксперта — им подписываются оценки")
        with self.reg._conn() as c:
            rows = [dict(r) for r in c.execute(
                "SELECT job_id,row_idx,file_name,image_uid,study_uid,internal_region,quality_class,violation_type,"
                "processing_status,region_supported FROM images WHERE job_id=? ORDER BY row_idx", (job_id,))]
        skipped = {"failure": 0, "unsupported": 0, "duplicate": 0, "no_frame": 0}
        seen, pick, hashes = set(), [], {}
        for r in rows:
            if r["processing_status"] != "Success":
                skipped["failure"] += 1; continue
            if not r["region_supported"]:
                skipped["unsupported"] += 1; continue
            k = r["image_uid"] or f"{r['job_id']}:{r['row_idx']}"
            if k in seen:
                skipped["duplicate"] += 1; continue
            fp = self.frame_path(r["job_id"], r["row_idx"])
            if not fp.is_file():
                skipped["no_frame"] += 1; continue
            h = hashlib.sha256(fp.read_bytes()).hexdigest()
            if h in hashes:  # тот же кадр под другим именем
                skipped["duplicate"] += 1; continue
            seen.add(k); hashes[h] = 1; pick.append(r)
        if not pick:
            raise ValueError(f"нет снимков для оценки: из {len(rows)} строк ошибок чтения {skipped['failure']}, "
                             f"вне поддерживаемой области или с чужого аппарата {skipped['unsupported']}, "
                             f"повторов {skipped['duplicate']}, без сохранённого кадра {skipped['no_frame']}")
        if len(pick) > MAX_SET:
            raise ValueError(f"в одной проверке не больше {MAX_SET} снимков, загружено {len(pick)}")
        seed = seed if seed is not None else int(time.time())
        random.Random(seed).shuffle(pick)
        set_hash = hashlib.sha256("".join(sorted(hashes)).encode()).hexdigest()[:12]
        params = {"mode": OWN, "job_id": job_id, "stratified": False, "seed": seed, "n_files": n_files,
                  "n_rows": len(rows), "skipped": skipped, "set_hash": set_hash,
                  "model_version": (versions or {}).get("model_version"), "config_hash": (versions or {}).get("config_hash")}
        with self._lock, self.reg._conn() as c:
            cur = c.execute("INSERT INTO expert_sets(title,created_at,created_by,params,n) VALUES(?,?,?,?,?)",
                            (title.strip()[:120] if title and title.strip() else f"Свои снимки от {time.strftime('%d.%m.%Y %H:%M')}",
                             _now(), reviewer, json.dumps(params, ensure_ascii=False), len(pick)))
            sid = cur.lastrowid
            c.executemany("INSERT INTO expert_items VALUES(?,?,?,?,?,?,?,?,?)",
                          [(sid, i + 1, r["job_id"], r["row_idx"], r["image_uid"], r["study_uid"], _region(r["internal_region"]),
                            r["quality_class"], r["violation_type"]) for i, r in enumerate(pick)])
        self.reg.audit(user.get("login", "?"), "expert_own_upload", str(sid), json.dumps(params, ensure_ascii=False))
        return self.get_set(sid, blind=True)

    def _params(self, c, sid: int) -> Dict[str, Any]:
        s = c.execute("SELECT created_by, params FROM expert_sets WHERE id=?", (sid,)).fetchone()
        if not s:
            raise KeyError(sid)
        return {**json.loads(s["params"] or "{}"), "_creator": s["created_by"]}

    def finished(self, sid: int, reviewer: str) -> bool:
        with self.reg._conn() as c:
            return bool(c.execute("SELECT 1 FROM expert_finish WHERE set_id=? AND reviewer=?", (sid, reviewer)).fetchone())

    def finish(self, sid: int, user: Dict[str, Any]) -> Dict[str, Any]:
        reviewer = (user.get("name") or user.get("login") or "").strip()[:60]
        with self.reg._conn() as c:
            s = c.execute("SELECT n FROM expert_sets WHERE id=?", (sid,)).fetchone()
            if not s:
                raise KeyError(sid)
            done = c.execute("SELECT COUNT(*) FROM expert_answers WHERE set_id=? AND reviewer=?", (sid, reviewer)).fetchone()[0]
        if done < s["n"]:
            raise ValueError(f"оценены не все снимки: {done} из {s['n']}")
        with self._lock, self.reg._conn() as c:
            c.execute("INSERT OR IGNORE INTO expert_finish(set_id,reviewer,at) VALUES(?,?,?)", (sid, reviewer, _now()))
        self.reg.audit(user.get("login", "?"), "expert_finish", str(sid), reviewer)
        return {"ok": True, "set_id": sid, "reviewer": reviewer}

    def report_allowed(self, sid: int, requester: str = "") -> bool:
        """Можно ли отдавать решения сервиса по набору (отчёт, answers.csv, get_set(blind=False)).

        Режим own_upload (2.4.1, P3): только после того, как оценку завершил именно загрузивший (создатель
        набора). Кто запрашивает — не важно: второй участник, пройдя набор раньше создателя, ответы модели
        не видит. `requester` оставлен для совместимости вызовов и на решение не влияет. Для выборки из
        журнала (create_set) решения сервиса и так видны в журнале — отчёт открыт."""
        with self.reg._conn() as c:
            p = self._params(c, sid)
            if p.get("mode") != OWN:
                return True
            return bool(c.execute("SELECT 1 FROM expert_finish WHERE set_id=? AND reviewer=?",
                                  (sid, p["_creator"])).fetchone())

    def hold_job(self, job_id: str, by: str = "") -> None:
        """Скрыть решения загрузки в журнале ДО её индексации (2.4.1, P2). Запись фиксируется (commit) до того,
        как строки загрузки появятся в таблице images, поэтому любой читатель журнала, увидевший снимки, видит и
        скрытие. Снимается только завершением своей слепой проверки загрузившим (см. hidden_jobs); если набор
        создать не удалось (нет ФИО, нет кадров), загрузка так и остаётся скрытой."""
        with self._lock, self.reg._conn() as c:
            c.execute("INSERT OR IGNORE INTO expert_hold(job_id, at, by) VALUES(?,?,?)", (job_id, _now(), by[:60]))

    def hidden_jobs(self) -> set:
        """Загрузки, решения по которым скрыты: своя слепая проверка ещё не завершена загрузившим, или загрузка
        помечена скрытой при приёме (hold_job), а завершённого своего набора по ней нет."""
        out, released = set(), set()
        with self.reg._conn() as c:
            for s in c.execute("SELECT id, created_by, params FROM expert_sets WHERE params LIKE ?", (f'%"{OWN}"%',)):
                p = json.loads(s["params"] or "{}")
                if p.get("mode") != OWN:
                    continue
                if c.execute("SELECT 1 FROM expert_finish WHERE set_id=? AND reviewer=?", (s["id"], s["created_by"])).fetchone():
                    released.add(p.get("job_id"))
                else:
                    out.add(p.get("job_id"))
            for r in c.execute("SELECT job_id FROM expert_hold"):
                if r["job_id"] not in released:
                    out.add(r["job_id"])
        return out

    def list_sets(self) -> List[Dict[str, Any]]:
        with self.reg._conn() as c:
            sets = [dict(r) for r in c.execute("SELECT * FROM expert_sets ORDER BY id DESC")]
            for s in sets:
                s["params"] = json.loads(s["params"] or "{}")
                s["reviewers"] = [dict(r) for r in c.execute(
                    "SELECT reviewer, COUNT(*) AS n FROM expert_answers WHERE set_id=? GROUP BY reviewer", (s["id"],))]
                s["finished"] = [r[0] for r in c.execute("SELECT reviewer FROM expert_finish WHERE set_id=?", (s["id"],))]
        return sets

    def get_set(self, sid: int, blind: bool = True) -> Dict[str, Any]:
        with self.reg._conn() as c:
            s = c.execute("SELECT * FROM expert_sets WHERE id=?", (sid,)).fetchone()
            if not s:
                raise KeyError(sid)
            items = [dict(r) for r in c.execute("SELECT * FROM expert_items WHERE set_id=? ORDER BY pos", (sid,))]
        own = json.loads(s["params"] or "{}").get("mode") == OWN
        out = []
        for it in items:
            # в своей слепой проверке идентификатор исследования не отдаём: по нему нельзя найти вердикт в журнале
            x = {"pos": it["pos"], "region": it["region"], "study_uid": None if (blind and own) else it["study_uid"],
                 "criteria": [{"code": k, "title": t} for k, t, _ in CRITERIA[it["region"]]],
                 "frame_url": f"/api/expert/sets/{sid}/frame/{it['pos']}.png"}
            if not blind:
                x.update(service_class=it["service_class"], service_violations=it["service_violations"])
            out.append(x)
        return {"id": s["id"], "title": s["title"], "created_at": s["created_at"], "created_by": s["created_by"],
                "params": json.loads(s["params"] or "{}"), "n": s["n"], "items": out, "answers_legend": ANSWERS}

    def frame(self, sid: int, pos: int) -> Path:
        with self.reg._conn() as c:
            it = c.execute("SELECT job_id,row_idx FROM expert_items WHERE set_id=? AND pos=?", (sid, pos)).fetchone()
        if not it:
            raise KeyError(pos)
        p = self.frame_path(it["job_id"], it["row_idx"])
        if not p.is_file():
            raise KeyError(pos)
        return p

    # ---- ответы
    def answer(self, sid: int, pos: int, user: Dict[str, Any], answers: Dict[str, str], comment: str = "") -> Dict[str, Any]:
        reviewer = (user.get("name") or user.get("login") or "").strip()[:60]
        if not reviewer:
            raise ValueError("укажите имя эксперта")
        with self.reg._conn() as c:
            it = c.execute("SELECT region FROM expert_items WHERE set_id=? AND pos=?", (sid, pos)).fetchone()
        if not it:
            raise KeyError(pos)
        if self.finished(sid, reviewer):
            raise ValueError("оценка завершена: ответы зафиксированы и не меняются")
        allowed = {k for k, _, _ in CRITERIA[it["region"]]}
        clean = {k: v for k, v in (answers or {}).items() if k in allowed and v in ANSWERS}
        if set(clean) != allowed:
            raise ValueError("нужно оценить все критерии снимка")
        comment = (comment or "").replace("\x00", "").strip()[:1000]
        with self._lock, self.reg._conn() as c:
            c.execute("INSERT OR REPLACE INTO expert_answers(set_id,pos,reviewer,role,answers,comment,at) VALUES(?,?,?,?,?,?,?)",
                      (sid, pos, reviewer, user.get("role", ""), json.dumps(clean, ensure_ascii=False), comment, _now()))
        return {"ok": True, "pos": pos, "reviewer": reviewer}

    def my_answers(self, sid: int, reviewer: str) -> Dict[int, Dict[str, Any]]:
        with self.reg._conn() as c:
            return {r["pos"]: {"answers": json.loads(r["answers"]), "comment": r["comment"]} for r in c.execute(
                "SELECT pos, answers, comment FROM expert_answers WHERE set_id=? AND reviewer=?", (sid, reviewer))}

    # ---- второй тур «врач + сервис» (2.5)
    @staticmethod
    def _svc(it: Dict[str, Any]) -> Dict[str, str]:
        viol = set(filter(None, (it.get("service_violations") or "").split(";")))
        return {code: ("violation" if vname in viol else "ok") for code, _, vname in CRITERIA[it["region"]]}

    def second_allowed(self, sid: int, reviewer: str) -> bool:
        """Второй тур: эксперт завершил свой первый тур И отчёт по набору разрешён (P3 для своих снимков)."""
        return bool(reviewer) and self.finished(sid, reviewer) and self.report_allowed(sid, reviewer)

    def _disputed(self, sid: int, reviewer: str) -> Dict[int, Dict[str, Any]]:
        """{pos: {item, first, codes, service}}: критерии, где определённый ответ первого тура не совпал с сервисом."""
        s = self.get_set(sid, blind=False)
        items = {it["pos"]: it for it in s["items"]}
        first = self.my_answers(sid, reviewer)
        out = {}
        for pos, a in sorted(first.items()):
            it = items.get(pos)
            if not it:
                continue
            svc = self._svc(it)
            codes = [c for c, _, _ in CRITERIA[it["region"]] if a["answers"].get(c) in ("ok", "violation")
                     and a["answers"].get(c) != svc[c]]
            if codes:
                out[pos] = {"item": it, "first": a["answers"], "codes": codes, "service": svc}
        return out

    def second_answers(self, sid: int, reviewer: str) -> Dict[int, Dict[str, Any]]:
        with self.reg._conn() as c:
            return {r["pos"]: {"answers": json.loads(r["answers"]), "comment": r["comment"], "at": r["at"]} for r in c.execute(
                "SELECT pos, answers, comment, at FROM expert_second WHERE set_id=? AND reviewer=?", (sid, reviewer))}

    def second_finished(self, sid: int, reviewer: str) -> bool:
        with self.reg._conn() as c:
            return bool(c.execute("SELECT 1 FROM expert_second_finish WHERE set_id=? AND reviewer=?",
                                  (sid, reviewer)).fetchone())

    def second_round(self, sid: int, reviewer: str) -> Dict[str, Any]:
        """Снимки второго тура: решение сервиса показывается только по спорным критериям."""
        if not self.second_allowed(sid, reviewer):
            raise PermissionError("второй тур открывается после «Завершить» первого тура и после того, как "
                                  "загрузивший завершил свою оценку")
        dis = self._disputed(sid, reviewer)
        mine = self.second_answers(sid, reviewer)
        titles = {k: t for reg in CRITERIA.values() for k, t, _ in reg}
        items = []
        for pos, d in dis.items():
            it = d["item"]
            items.append({"pos": pos, "region": it["region"], "frame_url": f"/api/expert/sets/{sid}/frame/{pos}.png",
                          "criteria": [{"code": c, "title": titles[c], "first": d["first"][c], "service": d["service"][c]}
                                       for c in d["codes"]],
                          "second": (mine.get(pos) or {}).get("answers"), "comment": (mine.get(pos) or {}).get("comment")})
        return {"set_id": sid, "reviewer": reviewer, "n_items": len(items),
                "n_criteria": sum(len(x["criteria"]) for x in items), "items": items,
                "finished": self.second_finished(sid, reviewer), "answers_legend": ANSWERS,
                "note": "Показаны только снимки и критерии, где ваш ответ первого тура разошёлся с сервисом. Ответы "
                        "первого тура зафиксированы; ответ второго тура хранится отдельно. Можно оставить свой ответ, "
                        "изменить его или пропустить снимок."}

    def second_answer(self, sid: int, pos: int, user: Dict[str, Any], answers: Dict[str, str],
                      comment: str = "") -> Dict[str, Any]:
        reviewer = (user.get("name") or user.get("login") or "").strip()[:60]
        if not self.second_allowed(sid, reviewer):
            raise PermissionError("второй тур недоступен: первый тур не завершён или отчёт по набору ещё закрыт")
        if self.second_finished(sid, reviewer):
            raise ValueError("второй тур завершён: ответы зафиксированы и не меняются")
        dis = self._disputed(sid, reviewer)
        if pos not in dis:
            raise KeyError(pos)
        allowed = set(dis[pos]["codes"])
        clean = {k: v for k, v in (answers or {}).items() if k in allowed and v in ANSWERS}
        if set(clean) != allowed:
            raise ValueError("нужно ответить по всем спорным критериям снимка")
        comment = (comment or "").replace("\x00", "").strip()[:1000]
        with self._lock, self.reg._conn() as c:
            c.execute("INSERT OR REPLACE INTO expert_second(set_id,pos,reviewer,answers,comment,at) VALUES(?,?,?,?,?,?)",
                      (sid, pos, reviewer, json.dumps(clean, ensure_ascii=False), comment, _now()))
        return {"ok": True, "pos": pos, "reviewer": reviewer}

    def second_finish(self, sid: int, user: Dict[str, Any]) -> Dict[str, Any]:
        """Завершить второй тур. Неотвеченные спорные снимки считаются пропущенными (остаётся ответ первого тура)."""
        reviewer = (user.get("name") or user.get("login") or "").strip()[:60]
        if not self.second_allowed(sid, reviewer):
            raise PermissionError("второй тур недоступен: первый тур не завершён или отчёт по набору ещё закрыт")
        with self._lock, self.reg._conn() as c:
            c.execute("INSERT OR IGNORE INTO expert_second_finish(set_id,reviewer,at) VALUES(?,?,?)", (sid, reviewer, _now()))
        self.reg.audit(user.get("login", "?"), "expert_second_finish", str(sid), reviewer)
        return {"ok": True, "set_id": sid, "reviewer": reviewer}

    def second_summary(self, sid: int, reviewer: str = "") -> Dict[str, Any]:
        """Сводка второго тура: изменения и направления, совпадение с сервисом до и после (по критериям).
        Учитываются только эксперты, завершившие первый тур."""
        s = self.get_set(sid, blind=False)
        items = {it["pos"]: it for it in s["items"]}
        with self.reg._conn() as c:
            q1 = "SELECT pos, reviewer, answers FROM expert_answers WHERE set_id=?" + (" AND reviewer=?" if reviewer else "")
            first = [dict(r) for r in c.execute(q1, (sid, reviewer) if reviewer else (sid,))]
            second = {(r["pos"], r["reviewer"]): json.loads(r["answers"]) for r in c.execute(
                "SELECT pos, reviewer, answers FROM expert_second WHERE set_id=?", (sid,))}
            fin2 = {r[0] for r in c.execute("SELECT reviewer FROM expert_second_finish WHERE set_id=?", (sid,))}
        fin1 = {r for r in {a["reviewer"] for a in first} if self.finished(sid, r)}
        dirs = {"нарушение → норма": 0, "норма → нарушение": 0, "нарушение → не могу оценить": 0,
                "норма → не могу оценить": 0}
        n_dis = n_rev = n_changed = n_to_service = 0
        before = {"agree": 0, "n": 0}
        after = {"agree": 0, "n": 0}
        per_rev: Dict[str, Dict[str, int]] = {}
        rows = []
        for a in first:
            it = items.get(a["pos"])
            if not it or a["reviewer"] not in fin1:
                continue
            svc = self._svc(it)
            ans1 = json.loads(a["answers"])
            ans2 = second.get((a["pos"], a["reviewer"])) or {}
            pr = per_rev.setdefault(a["reviewer"], {"disputed": 0, "reviewed": 0, "changed": 0})
            for code, title, _ in CRITERIA[it["region"]]:
                e1 = ans1.get(code)
                if e1 not in ("ok", "violation"):
                    continue
                before["n"] += 1
                before["agree"] += int(e1 == svc[code])
                e2 = ans2.get(code, e1) if e1 != svc[code] else e1
                if e2 in ("ok", "violation"):
                    after["n"] += 1
                    after["agree"] += int(e2 == svc[code])
                if e1 == svc[code]:
                    continue
                n_dis += 1
                pr["disputed"] += 1
                if code in ans2:
                    n_rev += 1
                    pr["reviewed"] += 1
                if e2 != e1:
                    n_changed += 1
                    pr["changed"] += 1
                    dirs[f"{ANSWERS[e1]} → {ANSWERS[e2]}"] += 1
                    n_to_service += int(e2 == svc[code])
                rows.append({"pos": a["pos"], "reviewer": a["reviewer"], "criterion": title, "region": it["region"],
                             "first": ANSWERS[e1], "service": ANSWERS[svc[code]],
                             "second": ANSWERS.get(ans2[code]) if code in ans2 else "пропущен"})

        def share(d):
            return {"agree": d["agree"], "n": d["n"], "agreement": round(d["agree"] / d["n"], 3) if d["n"] else None,
                    "agreement_ci": wilson(d["agree"], d["n"])}
        return {"reviewers_finished_first": sorted(fin1), "reviewers_finished_second": sorted(fin1 & fin2),
                "disputed": n_dis, "reviewed": n_rev, "skipped": n_dis - n_rev, "changed": n_changed,
                "changed_to_service": n_to_service, "directions": dirs, "per_reviewer": per_rev,
                "agreement_before": share(before), "agreement_after": share(after), "rows": rows[:300],
                "note": "Второй тур не слепой: эксперт видел решение сервиса по спорным критериям. Совпадение после "
                        "второго тура описывает, как эксперт использует подсказку сервиса, и не является проверкой "
                        "сервиса; чувствительность, специфичность и каппа выше считаются только по первому туру. "
                        "Единица счёта — критерий снимка с определённым ответом («норма» или «нарушение»)."}

    # ---- отчёт
    def report(self, sid: int, reviewer: str = "") -> Dict[str, Any]:
        s = self.get_set(sid, blind=False)
        items = {it["pos"]: it for it in s["items"]}
        with self.reg._conn() as c:
            q = "SELECT pos, reviewer, answers FROM expert_answers WHERE set_id=?" + (" AND reviewer=?" if reviewer else "")
            ans = [dict(r) for r in c.execute(q, (sid, reviewer) if reviewer else (sid,))]
        cells: Dict[str, Dict[str, int]] = {}
        any_cell = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "unsure": 0}
        disagreements = []
        for a in ans:
            it = items.get(a["pos"])
            if not it:
                continue
            viol = set(filter(None, (it["service_violations"] or "").split(";")))
            answers = json.loads(a["answers"])
            exp_any, svc_any, unsure_any = False, it["service_class"] == "1", False
            for code, title, vname in CRITERIA[it["region"]]:
                e = answers.get(code)
                svc = vname in viol
                key = f"{it['region']}:{code}"
                cc = cells.setdefault(key, {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "unsure": 0, "title": title, "region": it["region"]})
                if e == "unsure":
                    cc["unsure"] += 1; unsure_any = True
                    continue
                exp = e == "violation"
                exp_any = exp_any or exp
                cc["tp" if exp and svc else "fp" if svc else "fn" if exp else "tn"] += 1
                if exp != svc:
                    disagreements.append({"pos": a["pos"], "reviewer": a["reviewer"], "criterion": title,
                                          "region": it["region"], "expert": "нарушение" if exp else "норма",
                                          "service": "нарушение" if svc else "норма", "study_uid": it["study_uid"]})
            if unsure_any and not exp_any:
                any_cell["unsure"] += 1
            else:
                any_cell["tp" if exp_any and svc_any else "fp" if svc_any else "fn" if exp_any else "tn"] += 1

        def stats(cc: Dict[str, int]) -> Dict[str, Any]:
            tp, fp, fn, tn = cc["tp"], cc["fp"], cc["fn"], cc["tn"]
            n = tp + fp + fn + tn
            return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "unsure": cc["unsure"],
                    "agreement": round((tp + tn) / n, 3) if n else None, "agreement_ci": wilson(tp + tn, n),
                    "sensitivity": round(tp / (tp + fn), 3) if tp + fn else None, "sensitivity_ci": wilson(tp, tp + fn),
                    "specificity": round(tn / (tn + fp), 3) if tn + fp else None, "specificity_ci": wilson(tn, tn + fp),
                    "kappa": kappa(tp, fp, fn, tn)}
        crit = []
        for reg in ("spine", "hip"):
            for code, title, _ in CRITERIA[reg]:
                cc = cells.get(f"{reg}:{code}")
                if cc:
                    crit.append({"region": reg, "code": code, "title": title, **stats(cc)})
        # согласие экспертов между собой («есть нарушение / норма» по снимку), если экспертов больше одного
        by_rev: Dict[str, Dict[int, Optional[bool]]] = {}
        for a in ans:
            it = items.get(a["pos"])
            if not it:
                continue
            v = json.loads(a["answers"]).values()
            by_rev.setdefault(a["reviewer"], {})[a["pos"]] = (True if "violation" in v else (None if "unsure" in v else False))
        pairs = []
        revs = sorted(by_rev)
        for i in range(len(revs)):
            for j in range(i + 1, len(revs)):
                a1, a2 = by_rev[revs[i]], by_rev[revs[j]]
                common = [p for p in a1 if p in a2 and a1[p] is not None and a2[p] is not None]
                if not common:
                    continue
                tp = sum(1 for p in common if a1[p] and a2[p]); tn = sum(1 for p in common if not a1[p] and not a2[p])
                fp = sum(1 for p in common if not a1[p] and a2[p]); fn = sum(1 for p in common if a1[p] and not a2[p])
                pairs.append({"a": revs[i], "b": revs[j], "n": len(common), "agreement": round((tp + tn) / len(common), 3),
                              "agreement_ci": wilson(tp + tn, len(common)), "kappa": kappa(tp, fp, fn, tn)})
        per_image = []
        if s["params"].get("mode") == OWN:
            for p, it in sorted(items.items()):
                e = by_rev.get(reviewer or s["created_by"], {}).get(p, "нет ответа")
                per_image.append({"pos": p, "region": it["region"],
                                  "expert": "нет ответа" if e == "нет ответа" else ("не могу оценить" if e is None else ("нарушение" if e else "норма")),
                                  "service": "нарушение" if it["service_class"] == "1" else "норма",
                                  "service_violations": [v for v in (it["service_violations"] or "").split(";") if v]})
        return {"set": {k: s[k] for k in ("id", "title", "created_at", "created_by", "n", "params")},
                "reviewers": sorted({a["reviewer"] for a in ans}), "n_answers": len(ans),
                "inter_reader": pairs, "per_image": per_image,
                "any_violation": stats(any_cell), "criteria": crit, "disagreements": disagreements[:300],
                "second_round": self.second_summary(sid, reviewer),
                "note": "Чувствительность и специфичность сервиса считаются относительно оценки врача; «не могу оценить» "
                        "в расчёт не входит. Интервалы — Уилсона 95 %. Модель по этим ответам не дообучается."}

    def export_csv(self, sid: int) -> str:
        s = self.get_set(sid, blind=False)
        items = {it["pos"]: it for it in s["items"]}
        with self.reg._conn() as c:
            ans = [dict(r) for r in c.execute("SELECT * FROM expert_answers WHERE set_id=? ORDER BY pos, reviewer", (sid,))]
            second = {(r["pos"], r["reviewer"]): dict(r) for r in c.execute("SELECT * FROM expert_second WHERE set_id=?", (sid,))}
        buf = io.StringIO()
        w = csv.writer(buf, delimiter=";")
        # 2.5: колонки второго тура — в конце, первые 10 колонок прежние (первый тур)
        w.writerow(["set_id", "pos", "study_uid", "region", "reviewer", "criterion", "expert", "service", "comment", "at",
                    "round2_shown_service", "expert_round2", "comment_round2", "at_round2"])
        for a in ans:
            it = items.get(a["pos"]) or {}
            viol = set(filter(None, (it.get("service_violations") or "").split(";")))
            s2 = second.get((a["pos"], a["reviewer"])) or {}
            a2 = json.loads(s2["answers"]) if s2 else {}
            fin = self.finished(sid, a["reviewer"])
            for code, title, vname in CRITERIA.get(it.get("region", "spine"), []):
                e = json.loads(a["answers"]).get(code, "")
                svc = "violation" if vname in viol else "ok"
                shown = fin and e in ("ok", "violation") and e != svc
                e2 = a2.get(code, "") if shown else ""
                w.writerow([sid, a["pos"], it.get("study_uid", ""), it.get("region", ""), a["reviewer"], code,
                            ANSWERS.get(e, e), "нарушение" if vname in viol else "норма", a["comment"], a["at"],
                            "да" if shown else "нет", ANSWERS.get(e2, e2),
                            s2.get("comment", "") if e2 else "", s2.get("at", "") if e2 else ""])
        return "\ufeff" + buf.getvalue()

    def for_study(self, study_uid: str) -> List[Dict[str, Any]]:
        """Экспертные оценки снимков исследования — для карточки в журнале."""
        with self.reg._conn() as c:
            rows = [dict(r) for r in c.execute(
                "SELECT a.set_id, a.pos, a.reviewer, a.answers, a.comment, a.at, i.region, i.image_uid "
                "FROM expert_answers a JOIN expert_items i ON a.set_id=i.set_id AND a.pos=i.pos WHERE i.study_uid=? "
                "ORDER BY a.at", (study_uid,))]
        titles = {k: t for reg in CRITERIA.values() for k, t, _ in reg}
        for r in rows:
            r["answers"] = {titles.get(k, k): ANSWERS.get(v, v) for k, v in json.loads(r["answers"]).items()}
        return rows


def mount(app, reg, er: ExpertReview) -> None:
    from fastapi import Header, HTTPException, Request
    from fastapi.responses import FileResponse, Response

    def need(session: Optional[str]) -> Dict[str, Any]:
        if not reg.enabled():
            raise HTTPException(503, "Журнал исследований не настроен — экспертная проверка работает поверх него.")
        try:
            return reg.check_session(session)
        except PermissionError as e:
            raise HTTPException(401, str(e))

    @app.get("/api/expert/sets")
    def expert_sets(x_registry_session: Optional[str] = Header(None)):
        need(x_registry_session)
        return {"sets": er.list_sets()}

    @app.post("/api/expert/sets")
    async def expert_create(request: Request, x_registry_session: Optional[str] = Header(None)):
        u = need(x_registry_session)
        try:
            b = await request.json()
            return er.create_set(u, b.get("n", 40), b.get("region", ""), b.get("date_from", ""), b.get("date_to", ""),
                                 b.get("title", ""))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/expert/sets/{sid}")
    def expert_get(sid: int, x_registry_session: Optional[str] = Header(None)):
        u = need(x_registry_session)
        try:
            s = er.get_set(sid, blind=True)
        except KeyError:
            raise HTTPException(404, "проверка не найдена")
        me = (u.get("name") or u.get("login") or "").strip()[:60]
        s["my_answers"] = er.my_answers(sid, me)
        s["my_finished"] = er.finished(sid, me)
        return s

    @app.post("/api/expert/sets/{sid}/finish")
    def expert_finish(sid: int, x_registry_session: Optional[str] = Header(None)):
        u = need(x_registry_session)
        try:
            return er.finish(sid, u)
        except KeyError:
            raise HTTPException(404, "проверка не найдена")
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/expert/sets/{sid}/frame/{pos}.png")
    def expert_frame(sid: int, pos: int, x_registry_session: Optional[str] = Header(None), s: str = ""):
        need(x_registry_session or s)
        try:
            return FileResponse(str(er.frame(sid, pos)), media_type="image/png")
        except KeyError:
            raise HTTPException(404, "кадр не найден")

    @app.post("/api/expert/sets/{sid}/answers")
    async def expert_answer(sid: int, request: Request, x_registry_session: Optional[str] = Header(None)):
        u = need(x_registry_session)
        try:
            b = await request.json()
            return er.answer(sid, int(b.get("pos", 0)), u, b.get("answers") or {}, b.get("comment", ""))
        except KeyError:
            raise HTTPException(404, "снимок не найден в проверке")
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e))

    # ---- второй тур «врач + сервис» (2.5): только после своего «Завершить» и при открытом отчёте (P3)
    @app.get("/api/expert/sets/{sid}/second")
    def expert_second(sid: int, x_registry_session: Optional[str] = Header(None)):
        u = need(x_registry_session)
        me = (u.get("name") or u.get("login") or "").strip()[:60]
        try:
            return er.second_round(sid, me)
        except KeyError:
            raise HTTPException(404, "проверка не найдена")
        except PermissionError as e:
            raise HTTPException(403, str(e))

    @app.post("/api/expert/sets/{sid}/second/answers")
    async def expert_second_answer(sid: int, request: Request, x_registry_session: Optional[str] = Header(None)):
        u = need(x_registry_session)
        try:
            b = await request.json()
            return er.second_answer(sid, int(b.get("pos", 0)), u, b.get("answers") or {}, b.get("comment", ""))
        except PermissionError as e:
            raise HTTPException(403, str(e))
        except KeyError:
            raise HTTPException(404, "снимка нет во втором туре")
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e))

    @app.post("/api/expert/sets/{sid}/second/finish")
    def expert_second_finish(sid: int, x_registry_session: Optional[str] = Header(None)):
        u = need(x_registry_session)
        try:
            return er.second_finish(sid, u)
        except PermissionError as e:
            raise HTTPException(403, str(e))
        except KeyError:
            raise HTTPException(404, "проверка не найдена")

    @app.get("/api/expert/sets/{sid}/report")
    def expert_report(sid: int, reviewer: str = "", x_registry_session: Optional[str] = Header(None)):
        u = need(x_registry_session)
        try:
            if not er.report_allowed(sid, (u.get("name") or "").strip()[:60]):
                raise HTTPException(403, "отчёт откроется, когда загрузивший оценит все снимки и нажмёт «Завершить»")
            return er.report(sid, reviewer)
        except KeyError:
            raise HTTPException(404, "проверка не найдена")

    @app.get("/api/expert/sets/{sid}/answers.csv")
    def expert_csv(sid: int, x_registry_session: Optional[str] = Header(None)):
        u = need(x_registry_session)
        try:
            if not er.report_allowed(sid, (u.get("name") or "").strip()[:60]):
                raise HTTPException(403, "выгрузка откроется, когда загрузивший оценит все снимки и нажмёт «Завершить»")
            txt = er.export_csv(sid)
        except KeyError:
            raise HTTPException(404, "проверка не найдена")
        return Response(content=txt.encode("utf-8"), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f"attachment; filename=expert_set_{sid}.csv"})

    @app.get("/api/expert/study/{study_uid}")
    def expert_for_study(study_uid: str, x_registry_session: Optional[str] = Header(None)):
        need(x_registry_session)
        return {"reviews": er.for_study(study_uid)}
