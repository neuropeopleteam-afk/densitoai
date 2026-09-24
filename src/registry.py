#!/usr/bin/env python3
"""Журнал исследований отделения: поиск по снимкам, фильтры, статусы и комментарии врача и лаборанта.

Зачем. Кабинет по умолчанию ведёт историю в браузере того, кто загрузил партию: один пользователь не
видит чужих запросов (ТЗ §2.9). Отделению нужно другое — общий журнал: лаборант снял и загрузил,
врач нашёл исследование по фамилии или дате, оставил комментарий «переснять с ротацией внутрь»,
лаборант отметил «переснято». Журнал — отдельный, явно включаемый контур с учётными записями.

Что хранится. SQLite-файл в томе выходов (OUTPUT_DIR/registry.sqlite3), внутри контура учреждения,
без внешних сервисов. По каждому снимку — результат оценки (класс, нарушения, quality_prob, зона
«не уверен») и теги DICOM для поиска: ФИО, идентификатор пациента, дата рождения, пол, дата и время
исследования, номер направления (Accession), аппарат. Пиксели и файлы не копируются: карточка
снимка открывается из каталога задачи по её коду доступа.

Персональные данные (152-ФЗ). ФИО и идентификатор в списке маскируются («Иванова М. П.», «…4821»).
Полные значения отдаются только по явному запросу в карточке исследования, и каждый такой просмотр
пишется в журнал доступа (кто, когда, что). Поиск по фамилии работает по нормализованной строке
(регистр, «ё» → «е»). Журнал выключен, пока техгруппа не создала учётные записи:
    python src/registry.py add-user <логин> --name "Петрова А. В." --role doctor|lab|admin

Доступ. Вход по логину и паролю (PBKDF2-SHA256, 200 000 итераций), сессия — подписанный HMAC токен
в заголовке X-Registry-Session (не cookie — нет CSRF), срок 12 ч. После 5 неудачных попыток логин
блокируется на 60 с. Лаборант может ставить статусы «переснято» и «спорно», врач и администратор —
любые; журнал доступа к ПДн видит только администратор.

Модуль не зависит от FastAPI: api_server.py подключает маршруты функцией mount(app, ...).
Стандартная библиотека + pydicom (уже в образе).
"""

import argparse
import base64
import csv
import getpass
import hashlib
import hmac
import io
import json
import os
import re
import secrets
import sqlite3
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROLES = {"doctor": "врач", "lab": "лаборант", "admin": "администратор"}
STATUSES = {
    "new": "новое",
    "retake": "нужна пересъёмка",
    "retaken": "переснято",
    "accepted": "принято",
    "disputed": "спорно",
}
LAB_STATUSES = {"retaken", "disputed"}
SESSION_TTL_S = 12 * 3600
PBKDF2_ITER = 200_000
MAX_COMMENT = 2000
MAX_COMMENTS_PER_STUDY = 500
LOCK_AFTER = 5
LOCK_S = 60
ANON_VALUES = {"", "anonymized", "anonymous", "anon", "none", "null", "unknown", "обезличено"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
  id INTEGER PRIMARY KEY,
  job_id TEXT NOT NULL, row_idx INTEGER NOT NULL,
  file_name TEXT, study_uid TEXT, image_uid TEXT,
  region TEXT, internal_region TEXT, quality_class TEXT, violation_type TEXT, quality_prob REAL,
  uncertain INTEGER DEFAULT 0, processing_status TEXT, region_supported INTEGER DEFAULT 1,
  patient_name TEXT, patient_name_norm TEXT, patient_id TEXT, birth_date TEXT, sex TEXT,
  study_date TEXT, study_time TEXT, accession TEXT, station TEXT, manufacturer TEXT, model TEXT,
  created_at TEXT, source TEXT,
  UNIQUE(job_id, row_idx)
);
CREATE INDEX IF NOT EXISTS ix_img_study ON images(study_uid);
CREATE INDEX IF NOT EXISTS ix_img_name ON images(patient_name_norm);
CREATE INDEX IF NOT EXISTS ix_img_date ON images(study_date);
CREATE TABLE IF NOT EXISTS study_status (
  study_uid TEXT PRIMARY KEY, status TEXT NOT NULL, login TEXT, author TEXT, role TEXT, at TEXT
);
CREATE TABLE IF NOT EXISTS status_log (
  id INTEGER PRIMARY KEY, study_uid TEXT, status TEXT, login TEXT, author TEXT, role TEXT, at TEXT
);
CREATE TABLE IF NOT EXISTS comments (
  id INTEGER PRIMARY KEY, study_uid TEXT NOT NULL, image_uid TEXT, login TEXT, author TEXT, role TEXT,
  text TEXT NOT NULL, at TEXT
);
CREATE INDEX IF NOT EXISTS ix_com_study ON comments(study_uid);
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY, at TEXT, login TEXT, action TEXT, target TEXT, detail TEXT
);
"""


# --------------------------------------------------------------------------- #
# нормализация и маскирование
# --------------------------------------------------------------------------- #
def _clean(v: Any) -> str:
    s = "" if v is None else str(v)
    s = s.replace("\x00", " ").strip()
    return "" if s.lower() in ANON_VALUES else s


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower().replace("ё", "е").replace("^", " ")
    return re.sub(r"\s+", " ", s).strip()


def name_parts(name: str) -> List[str]:
    return [p for p in re.split(r"[\^\s]+", _clean(name)) if p]


def display_name(name: str) -> str:
    return " ".join(name_parts(name))


def mask_name(name: str) -> str:
    p = name_parts(name)
    if not p:
        return "обезличено"
    return p[0] + ("" if len(p) == 1 else " " + " ".join(x[0].upper() + "." for x in p[1:3]))


def mask_id(pid: str) -> str:
    pid = _clean(pid)
    if not pid:
        return ""
    return "…" + pid[-4:] if len(pid) > 4 else "…"


def fmt_date(d: str) -> str:
    d = _clean(d)
    return f"{d[6:8]}.{d[4:6]}.{d[0:4]}" if re.fullmatch(r"\d{8}", d) else d


def parse_date(s: Optional[str]) -> Optional[str]:
    """ДД.ММ.ГГГГ / ГГГГ-ММ-ДД / ГГГГММДД -> ГГГГММДД."""
    s = (s or "").strip()
    if not s:
        return None
    m = re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", s)
    if m:
        return f"{m.group(3)}{int(m.group(2)):02d}{int(m.group(1)):02d}"
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return m.group(1) + m.group(2) + m.group(3)
    if re.fullmatch(r"\d{8}", s):
        return s
    raise ValueError(f"дата «{s}» не распознана (ожидается ДД.ММ.ГГГГ)")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# --------------------------------------------------------------------------- #
# учётные записи и сессии
# --------------------------------------------------------------------------- #
def hash_password(pw: str, salt: Optional[bytes] = None, it: int = PBKDF2_ITER) -> str:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), salt, it)
    return f"pbkdf2_sha256${it}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(pw: str, stored: str) -> bool:
    try:
        algo, it, salt, dk = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        got = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), base64.b64decode(salt), int(it))
        return hmac.compare_digest(got, base64.b64decode(dk))
    except Exception:  # noqa: BLE001
        return False


class Registry:
    def __init__(self, output_dir: Path, users_file: Optional[Path] = None, secret: Optional[str] = None):
        self.dir = Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.dir / "registry.sqlite3"
        self.users_file = Path(users_file) if users_file else self.dir / "registry_users.json"
        self._secret = (secret or "").encode() or self._load_secret()
        self._lock = threading.Lock()
        self._fails: Dict[str, Tuple[int, float]] = {}
        with self._conn() as c:
            c.executescript(SCHEMA)

    # ---- хранилище
    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self.db_path), timeout=10)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def _load_secret(self) -> bytes:
        p = self.dir / ".registry_secret"
        if not p.exists():
            fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(secrets.token_hex(32))
        return p.read_text().strip().encode()

    # ---- пользователи
    def users(self) -> Dict[str, Dict[str, Any]]:
        try:
            data = json.loads(self.users_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
        out = {}
        for u in data.get("users", []):
            if u.get("login") and u.get("pbkdf2") and u.get("role") in ROLES and not u.get("disabled"):
                out[str(u["login"])] = u
        return out

    def enabled(self) -> bool:
        return bool(self.users())

    def add_user(self, login: str, password: str, name: str, role: str) -> None:
        if role not in ROLES:
            raise ValueError(f"роль: {', '.join(ROLES)}")
        if not re.fullmatch(r"[A-Za-z0-9_.\-]{2,32}", login):
            raise ValueError("логин: латиница, цифры, _.- (2–32 символа)")
        if len(password) < 8:
            raise ValueError("пароль не короче 8 символов")
        try:
            data = json.loads(self.users_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = {"users": []}
        data["users"] = [u for u in data.get("users", []) if u.get("login") != login]
        data["users"].append({"login": login, "name": name or login, "role": role, "pbkdf2": hash_password(password)})
        tmp = self.users_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.replace(self.users_file)

    def login(self, login: str, password: str) -> Dict[str, Any]:
        login = (login or "").strip()
        n, until = self._fails.get(login, (0, 0.0))
        if until > time.time():
            raise PermissionError(f"слишком много попыток, повторите через {int(until - time.time()) + 1} с")
        u = self.users().get(login)
        if not u or not verify_password(password or "", u["pbkdf2"]):
            n += 1
            self._fails[login] = (n, time.time() + LOCK_S if n >= LOCK_AFTER else 0.0)
            if n >= LOCK_AFTER:
                self._fails[login] = (0, time.time() + LOCK_S)
            self.audit(login or "?", "login_failed", "", "")
            raise PermissionError("неверный логин или пароль")
        self._fails.pop(login, None)
        exp = int(time.time()) + SESSION_TTL_S
        payload = f"{login}|{exp}"
        sig = hmac.new(self._secret, payload.encode(), hashlib.sha256).hexdigest()
        token = base64.urlsafe_b64encode(f"{payload}|{sig}".encode()).decode()
        self.audit(login, "login", "", "")
        return {"session": token, "expires_at": exp, "user": self.public_user(u)}

    @staticmethod
    def public_user(u: Dict[str, Any]) -> Dict[str, Any]:
        return {"login": u["login"], "name": u.get("name") or u["login"], "role": u["role"],
                "role_title": ROLES[u["role"]]}

    def check_session(self, token: Optional[str]) -> Dict[str, Any]:
        try:
            login, exp, sig = base64.urlsafe_b64decode((token or "").encode()).decode().split("|")
            good = hmac.new(self._secret, f"{login}|{exp}".encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(sig, good) or int(exp) < time.time():
                raise ValueError
            u = self.users().get(login)
            if not u:
                raise ValueError
            return u
        except Exception:  # noqa: BLE001
            raise PermissionError("сессия недействительна, войдите заново")

    def audit(self, login: str, action: str, target: str, detail: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO audit(at,login,action,target,detail) VALUES(?,?,?,?,?)",
                      (_now(), login, action, target, detail[:500]))

    # ---- индексирование
    def index_rows(self, job_id: str, rows: List[Dict[str, Any]], tags_by_row: Dict[int, Dict[str, str]],
                   created_at: Optional[str] = None, source: str = "upload") -> int:
        created_at = created_at or _now()
        recs = []
        for i, r in enumerate(rows):
            t = tags_by_row.get(i, {})
            det = r.get("details") if isinstance(r.get("details"), dict) else {}
            crit = det.get("criteria") if isinstance(det.get("criteria"), list) else []
            unc = bool(det.get("uncertain_criteria")) or any(bool(c.get("uncertain")) for c in crit if isinstance(c, dict))
            try:
                prob = float(r.get("quality_prob"))
            except Exception:  # noqa: BLE001
                prob = None
            name = _clean(t.get("PatientName"))
            recs.append((
                job_id, i, str(r.get("path_to_study") or ""), str(r.get("study_uid") or ""), str(r.get("image_uid") or ""),
                str(r.get("anatomical_region") or ""), str(det.get("internal_region") or ""),
                str(r.get("quality_class") or ""), str(r.get("violation_type") or ""), prob, int(unc),
                str(r.get("processing_status") or ""), int(r.get("region_supported") not in (False, "False", 0, "0")),
                display_name(name), norm_text(display_name(name)), _clean(t.get("PatientID")),
                _clean(t.get("PatientBirthDate")), _clean(t.get("PatientSex")), _clean(t.get("StudyDate")),
                _clean(t.get("StudyTime")), _clean(t.get("AccessionNumber")), _clean(t.get("StationName")),
                _clean(t.get("Manufacturer")), _clean(t.get("ManufacturerModelName")), created_at, source))
        with self._lock, self._conn() as c:
            c.executemany("""INSERT OR REPLACE INTO images(job_id,row_idx,file_name,study_uid,image_uid,region,
                internal_region,quality_class,violation_type,quality_prob,uncertain,processing_status,region_supported,
                patient_name,patient_name_norm,patient_id,birth_date,sex,study_date,study_time,accession,station,
                manufacturer,model,created_at,source) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", recs)
        return len(recs)

    # ---- поиск
    def search(self, q: str = "", date_from: Optional[str] = None, date_to: Optional[str] = None,
               region: str = "", result: str = "", status: str = "", violation: str = "",
               has_comments: str = "", limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        where, args = [], []
        for tok in norm_text(q).split():
            like = f"%{tok}%"
            where.append("(patient_name_norm LIKE ? OR lower(patient_id) LIKE ? OR lower(accession) LIKE ? "
                         "OR lower(file_name) LIKE ? OR study_uid LIKE ?)")
            args += [like, like, like, like, like]
        df, dt = parse_date(date_from), parse_date(date_to)
        eff_date = "COALESCE(NULLIF(study_date,''), replace(substr(created_at,1,10),'-',''))"
        if df:
            where.append(f"{eff_date} >= ?"); args.append(df)
        if dt:
            where.append(f"{eff_date} <= ?"); args.append(dt)
        if region in ("spine", "hip"):
            where.append("internal_region LIKE ?"); args.append(f"%{region}%")
        if violation:
            where.append("violation_type LIKE ?"); args.append(f"%{violation}%")
        base = " AND ".join(where) if where else "1=1"
        # отбираем исследования, у которых хотя бы один снимок подходит под текст/дату/область
        with self._conn() as c:
            uids = [r[0] for r in c.execute(f"SELECT DISTINCT study_uid FROM images WHERE {base} AND study_uid!=''", args)]
            studies = [self._study_summary(c, u) for u in uids]
        if result:
            studies = [s for s in studies if result in s["flags"]]
        if status in STATUSES:
            studies = [s for s in studies if s["status"] == status]
        if has_comments == "1":
            studies = [s for s in studies if s["n_comments"] > 0]
        studies.sort(key=lambda s: (s["date_sort"], s["last_upload"]), reverse=True)
        total = len(studies)
        limit = max(1, min(int(limit or 50), 200))
        page = studies[int(offset or 0): int(offset or 0) + limit]
        return {"total": total, "offset": int(offset or 0), "limit": limit, "studies": page}

    def _study_summary(self, c: sqlite3.Connection, uid: str) -> Dict[str, Any]:
        imgs = [dict(r) for r in c.execute("SELECT * FROM images WHERE study_uid=? ORDER BY created_at DESC, row_idx", (uid,))]
        # последняя загрузка каждого снимка (повторная загрузка того же файла не дублирует строку)
        seen, latest = set(), []
        for im in imgs:
            k = im["image_uid"] or f"{im['job_id']}:{im['row_idx']}"
            if k in seen:
                continue
            seen.add(k); latest.append(im)
        st = c.execute("SELECT * FROM study_status WHERE study_uid=?", (uid,)).fetchone()
        ncom = c.execute("SELECT COUNT(*) FROM comments WHERE study_uid=?", (uid,)).fetchone()[0]
        lastc = c.execute("SELECT author, role, text, at FROM comments WHERE study_uid=? ORDER BY id DESC LIMIT 1", (uid,)).fetchone()
        name = next((im["patient_name"] for im in latest if im["patient_name"]), "")
        pid = next((im["patient_id"] for im in latest if im["patient_id"]), "")
        sdate = next((im["study_date"] for im in latest if im["study_date"]), "")
        flags = set()
        for im in latest:
            if im["processing_status"] and im["processing_status"] != "Success":
                flags.add("fail")
            elif not im["region_supported"]:
                flags.add("unsupported")
            elif im["quality_class"] == "1":
                flags.add("violation")
            else:
                flags.add("ok")
            if im["uncertain"]:
                flags.add("uncertain")
        viol = sorted({v for im in latest for v in (im["violation_type"] or "").split(";") if v})
        regions = sorted({im["region"] for im in latest if im["region"]})
        probs = [im["quality_prob"] for im in latest if im["quality_prob"] is not None]
        last_upload = max((im["created_at"] or "" for im in latest), default="")
        return {
            "study_uid": uid,
            "patient": mask_name(name),
            "patient_id": mask_id(pid),
            "has_pii": bool(name or pid),
            "study_date": fmt_date(sdate) or (last_upload[:10] and fmt_date(last_upload[:10].replace("-", ""))),
            "study_date_source": "DICOM" if sdate else "дата загрузки",
            "date_sort": sdate or last_upload[:10].replace("-", ""),
            "last_upload": last_upload,
            "regions": regions,
            "n_images": len(latest),
            "n_violation": sum(1 for im in latest if im["quality_class"] == "1"),
            "max_prob": round(max(probs), 3) if probs else None,
            "violations": viol,
            "flags": sorted(flags),
            "verdict": "fail" if "fail" in flags else ("violation" if "violation" in flags else
                        ("unsupported" if flags == {"unsupported"} else "ok")),
            "status": st["status"] if st else "new",
            "status_title": STATUSES[st["status"] if st else "new"],
            "status_by": (f"{st['author']} ({ROLES.get(st['role'], st['role'])})" if st else ""),
            "n_comments": ncom,
            "last_comment": ({"author": lastc["author"], "role": ROLES.get(lastc["role"], lastc["role"]),
                              "text": lastc["text"][:140], "at": lastc["at"]} if lastc else None),
            "jobs": sorted({im["job_id"] for im in latest}),
            "station": next((im["station"] for im in latest if im["station"]), ""),
            "accession": mask_id(next((im["accession"] for im in latest if im["accession"]), "")),
        }

    def study(self, uid: str, user: Dict[str, Any], reveal: bool = False,
              job_token: Optional[Any] = None) -> Dict[str, Any]:
        with self._conn() as c:
            if not c.execute("SELECT 1 FROM images WHERE study_uid=?", (uid,)).fetchone():
                raise KeyError(uid)
            s = self._study_summary(c, uid)
            imgs = [dict(r) for r in c.execute(
                "SELECT job_id,row_idx,file_name,image_uid,region,quality_class,violation_type,quality_prob,uncertain,"
                "processing_status,region_supported,created_at,patient_name,patient_id,birth_date,sex,accession "
                "FROM images WHERE study_uid=? ORDER BY created_at DESC, row_idx", (uid,))]
            com = [dict(r) for r in c.execute(
                "SELECT id,image_uid,author,role,text,at FROM comments WHERE study_uid=? ORDER BY id", (uid,))]
            log = [dict(r) for r in c.execute(
                "SELECT status,author,role,at FROM status_log WHERE study_uid=? ORDER BY id", (uid,))]
        pii = None
        if reveal:
            im = next((x for x in imgs if x["patient_name"] or x["patient_id"]), imgs[0])
            pii = {"patient_name": im["patient_name"] or "обезличено", "patient_id": im["patient_id"],
                   "birth_date": fmt_date(im["birth_date"]), "sex": im["sex"], "accession": im["accession"]}
            self.audit(user["login"], "pii_view", uid, "ФИО, ID, дата рождения, направление")
        for im in imgs:
            for k in ("patient_name", "patient_id", "birth_date", "sex", "accession"):
                im.pop(k, None)
            tok = job_token(im["job_id"]) if job_token else None
            im["card_url"] = f"/#app/job/{im['job_id']}/{tok}" if tok else None
        for x in com:
            x["role_title"] = ROLES.get(x["role"], x["role"])
        for x in log:
            x["status_title"] = STATUSES.get(x["status"], x["status"])
            x["role_title"] = ROLES.get(x["role"], x["role"])
        return {**s, "images": imgs, "comments": com, "status_log": log, "pii": pii,
                "statuses": STATUSES, "allowed_statuses": self.allowed_statuses(user)}

    @staticmethod
    def allowed_statuses(user: Dict[str, Any]) -> List[str]:
        return sorted(LAB_STATUSES) if user["role"] == "lab" else [k for k in STATUSES if k != "new"] + ["new"]

    # ---- действия
    def add_comment(self, uid: str, user: Dict[str, Any], text: str, image_uid: Optional[str] = None) -> Dict[str, Any]:
        text = (text or "").replace("\x00", "").strip()
        if not text:
            raise ValueError("пустой комментарий")
        if len(text) > MAX_COMMENT:
            raise ValueError(f"комментарий длиннее {MAX_COMMENT} символов")
        with self._lock, self._conn() as c:
            if not c.execute("SELECT 1 FROM images WHERE study_uid=?", (uid,)).fetchone():
                raise KeyError(uid)
            if image_uid and not c.execute("SELECT 1 FROM images WHERE study_uid=? AND image_uid=?", (uid, image_uid)).fetchone():
                raise ValueError("снимок не относится к исследованию")
            if c.execute("SELECT COUNT(*) FROM comments WHERE study_uid=?", (uid,)).fetchone()[0] >= MAX_COMMENTS_PER_STUDY:
                raise ValueError("превышен предел комментариев по исследованию")
            cur = c.execute("INSERT INTO comments(study_uid,image_uid,login,author,role,text,at) VALUES(?,?,?,?,?,?,?)",
                            (uid, image_uid or None, user["login"], user.get("name") or user["login"], user["role"], text, _now()))
            cid = cur.lastrowid
        return {"id": cid, "author": user.get("name"), "role": user["role"], "role_title": ROLES[user["role"]],
                "text": text, "image_uid": image_uid, "at": _now()}

    def set_status(self, uid: str, user: Dict[str, Any], status: str) -> Dict[str, Any]:
        if status not in STATUSES:
            raise ValueError("неизвестный статус")
        if status not in self.allowed_statuses(user):
            raise PermissionError(f"роль «{ROLES[user['role']]}» не может ставить статус «{STATUSES[status]}»")
        with self._lock, self._conn() as c:
            if not c.execute("SELECT 1 FROM images WHERE study_uid=?", (uid,)).fetchone():
                raise KeyError(uid)
            row = (uid, status, user["login"], user.get("name") or user["login"], user["role"], _now())
            c.execute("INSERT OR REPLACE INTO study_status(study_uid,status,login,author,role,at) VALUES(?,?,?,?,?,?)", row)
            c.execute("INSERT INTO status_log(study_uid,status,login,author,role,at) VALUES(?,?,?,?,?,?)", row)
        return {"status": status, "status_title": STATUSES[status]}

    def audit_list(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT at,login,action,target,detail FROM audit ORDER BY id DESC LIMIT ?",
                                               (max(1, min(limit, 2000)),))]

    def export_csv(self, studies: List[Dict[str, Any]]) -> str:
        buf = io.StringIO()
        w = csv.writer(buf, delimiter=";")
        w.writerow(["study_uid", "пациент (маска)", "дата", "области", "снимков", "с нарушением", "нарушения",
                    "статус", "комментариев"])
        for s in studies:
            w.writerow([s["study_uid"], s["patient"], s["study_date"], ", ".join(s["regions"]), s["n_images"],
                        s["n_violation"], "; ".join(s["violations"]), s["status_title"], s["n_comments"]])
        return "\ufeff" + buf.getvalue()


# --------------------------------------------------------------------------- #
# теги DICOM для индексирования
# --------------------------------------------------------------------------- #
TAGS = ("PatientName", "PatientID", "PatientBirthDate", "PatientSex", "StudyDate", "StudyTime",
        "AccessionNumber", "StationName", "Manufacturer", "ManufacturerModelName", "SOPInstanceUID")


def read_tags(rows: List[Dict[str, Any]], root: Path) -> Dict[int, Dict[str, str]]:
    """Теги по строкам: сначала по path_to_study, затем по SOPInstanceUID среди файлов каталога загрузки."""
    try:
        import pydicom
    except Exception:  # noqa: BLE001
        return {}

    def tags_of(p: Path) -> Dict[str, str]:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
        except Exception:  # noqa: BLE001
            return {}
        return {t: str(getattr(ds, t, "") or "") for t in TAGS}

    out: Dict[int, Dict[str, str]] = {}
    by_sop: Optional[Dict[str, Dict[str, str]]] = None
    root = Path(root)
    for i, r in enumerate(rows):
        rel = str(r.get("path_to_study") or "")
        cands = [root / rel] + ([root / Path(*Path(rel).parts[1:])] if len(Path(rel).parts) > 1 else [])
        p = next((x for x in cands if rel and x.is_file()), None)
        t = tags_of(p) if p else {}
        if not t or (r.get("image_uid") and t.get("SOPInstanceUID") and t["SOPInstanceUID"] != str(r.get("image_uid"))):
            if by_sop is None:
                by_sop = {}
                n = 0
                for f in root.rglob("*"):
                    if f.is_file() and n < 5000:
                        n += 1
                        tt = tags_of(f)
                        if tt.get("SOPInstanceUID"):
                            by_sop[tt["SOPInstanceUID"]] = tt
            t = by_sop.get(str(r.get("image_uid") or ""), t)
        if t:
            out[i] = t
    return out


# --------------------------------------------------------------------------- #
# подключение к FastAPI
# --------------------------------------------------------------------------- #
def mount(app, reg: Registry, job_token_reader, log=None) -> None:
    from fastapi import Header, HTTPException, Request
    from fastapi.responses import Response

    def need(session: Optional[str]) -> Dict[str, Any]:
        if not reg.enabled():
            raise HTTPException(503, "Журнал исследований не настроен: техгруппа создаёт учётные записи "
                                     "командой python src/registry.py add-user (см. README).")
        try:
            return reg.check_session(session)
        except PermissionError as e:
            raise HTTPException(401, str(e))

    @app.get("/api/registry/status")
    def registry_status():
        """Включён ли журнал (есть ли учётные записи). Без персональных данных."""
        return {"enabled": reg.enabled(), "roles": ROLES, "statuses": STATUSES}

    @app.post("/api/registry/login")
    async def registry_login(request: Request):
        if not reg.enabled():
            raise HTTPException(503, "Журнал исследований не настроен.")
        try:
            body = await request.json()
            return reg.login(str(body.get("login", "")), str(body.get("password", "")))
        except PermissionError as e:
            raise HTTPException(401, str(e))
        except Exception:  # noqa: BLE001
            raise HTTPException(400, "тело запроса: JSON {login, password}")

    @app.get("/api/registry/me")
    def registry_me(x_registry_session: Optional[str] = Header(None)):
        return reg.public_user(need(x_registry_session))

    @app.get("/api/registry/studies")
    def registry_search(q: str = "", date_from: str = "", date_to: str = "", region: str = "", result: str = "",
                        status: str = "", violation: str = "", has_comments: str = "", limit: int = 50, offset: int = 0,
                        x_registry_session: Optional[str] = Header(None)):
        """Поиск исследований: q — фамилия/имя/ID пациента/номер направления/имя файла; даты ДД.ММ.ГГГГ;
        region spine|hip; result violation|ok|fail|uncertain|unsupported; status new|retake|retaken|accepted|disputed."""
        need(x_registry_session)
        try:
            return reg.search(q, date_from, date_to, region, result, status, violation, has_comments, limit, offset)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/registry/studies.csv")
    def registry_export(q: str = "", date_from: str = "", date_to: str = "", region: str = "", result: str = "",
                        status: str = "", violation: str = "", has_comments: str = "",
                        x_registry_session: Optional[str] = Header(None)):
        u = need(x_registry_session)
        try:
            res = reg.search(q, date_from, date_to, region, result, status, violation, has_comments, 200, 0)
        except ValueError as e:
            raise HTTPException(400, str(e))
        reg.audit(u["login"], "export_csv", "", f"{res['total']} исследований, ФИО маскированы")
        return Response(content=reg.export_csv(res["studies"]).encode("utf-8"), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": "attachment; filename=registry.csv"})

    @app.get("/api/registry/studies/{study_uid}")
    def registry_study(study_uid: str, reveal: int = 0, x_registry_session: Optional[str] = Header(None)):
        u = need(x_registry_session)
        try:
            return reg.study(study_uid, u, reveal=bool(reveal), job_token=job_token_reader)
        except KeyError:
            raise HTTPException(404, "исследование не найдено")

    @app.post("/api/registry/studies/{study_uid}/comments")
    async def registry_comment(study_uid: str, request: Request, x_registry_session: Optional[str] = Header(None)):
        u = need(x_registry_session)
        try:
            body = await request.json()
            return reg.add_comment(study_uid, u, str(body.get("text", "")), body.get("image_uid") or None)
        except KeyError:
            raise HTTPException(404, "исследование не найдено")
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/registry/studies/{study_uid}/status")
    async def registry_set_status(study_uid: str, request: Request, x_registry_session: Optional[str] = Header(None)):
        u = need(x_registry_session)
        try:
            body = await request.json()
            return reg.set_status(study_uid, u, str(body.get("status", "")))
        except KeyError:
            raise HTTPException(404, "исследование не найдено")
        except PermissionError as e:
            raise HTTPException(403, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/registry/audit")
    def registry_audit(limit: int = 200, x_registry_session: Optional[str] = Header(None)):
        u = need(x_registry_session)
        if u["role"] != "admin":
            raise HTTPException(403, "журнал доступа доступен администратору")
        return {"audit": reg.audit_list(limit)}


# --------------------------------------------------------------------------- #
# CLI для техгруппы
# --------------------------------------------------------------------------- #
def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Журнал исследований DensitoAI: учётные записи и переиндексация")
    ap.add_argument("--output-dir", default=os.environ.get("DENSITO_OUTPUT_DIR", "/data/output"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add-user"); a.add_argument("login"); a.add_argument("--name", default="")
    a.add_argument("--role", choices=sorted(ROLES), required=True)
    a.add_argument("--password-stdin", action="store_true", help="прочитать пароль из stdin (для скриптов)")
    sub.add_parser("list-users")
    sub.add_parser("reindex", help="добавить в журнал задачи из jobs/ (без тегов DICOM — загрузки уже удалены)")
    args = ap.parse_args(list(argv) if argv is not None else None)
    reg = Registry(Path(args.output_dir))
    if args.cmd == "add-user":
        pw = input() if args.password_stdin else getpass.getpass("Пароль: ")
        reg.add_user(args.login, pw.strip(), args.name, args.role)
        print(f"учётная запись {args.login} ({ROLES[args.role]}) сохранена в {reg.users_file}")
    elif args.cmd == "list-users":
        for u in reg.users().values():
            print(u["login"], ROLES[u["role"]], u.get("name", ""))
    elif args.cmd == "reindex":
        n = 0
        for d in sorted((Path(args.output_dir) / "jobs").glob("*")):
            sj = d / "summary.json"
            if sj.is_file():
                card = json.loads(sj.read_text(encoding="utf-8"))
                n += reg.index_rows(d.name, card.get("rows", []), {}, card.get("created_at"), source="reindex")
        print(f"строк в журнале добавлено/обновлено: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
