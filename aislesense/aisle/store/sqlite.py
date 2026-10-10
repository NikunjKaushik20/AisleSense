"""SQLite event store under var/. One writer (the session runner), readers open their own connection."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
import time
from pathlib import Path

from aisle.schema import Event, VAR_DIR

DDL = """
CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, site TEXT, mode TEXT, started REAL, meta TEXT);
CREATE TABLE IF NOT EXISTS events(session TEXT, cam TEXT, t REAL, track TEXT, x REAL, y REAL, conf REAL);
CREATE INDEX IF NOT EXISTS ix_events_st ON events(session, t);
"""


class Store:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else VAR_DIR / "aisle.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._tx() as c:
            c.executescript(DDL)
            c.execute("PRAGMA journal_mode=WAL")
        self._w = self._conn()
        self._w.execute("PRAGMA synchronous=NORMAL")
        self._buf, self._last_flush = [], time.monotonic()

    def _conn(self):
        return sqlite3.connect(self.path, timeout=30, check_same_thread=False)

    @contextmanager
    def _tx(self):
        c = self._conn()
        try:
            with c:
                yield c
        finally:
            c.close()

    def close(self) -> None:
        self.flush()
        self._w.close()

    def new_session(self, site: str, mode: str, meta: dict | None = None) -> str:
        sid = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}-{site}"
        with self._lock, self._tx() as c:
            c.execute("INSERT INTO sessions VALUES(?,?,?,?,?)", (sid, site, mode, time.time(), json.dumps(meta or {})))
        return sid

    def insert(self, events, flush_rows: int = 5000, flush_s: float = 0.5) -> int:
        """Buffered write; rows become visible to readers after flush (<= flush_s later)."""
        rows = [e.as_row() for e in events]
        with self._lock:
            self._buf += rows
            if len(self._buf) >= flush_rows or time.monotonic() - self._last_flush >= flush_s:
                self._flush_locked()
        return len(rows)

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if self._buf:
            with self._w:
                self._w.executemany("INSERT INTO events VALUES(?,?,?,?,?,?,?)", self._buf)
            self._buf = []
        self._last_flush = time.monotonic()

    def events(self, session: str, t0: float | None = None, t1: float | None = None) -> list:
        q, a = "SELECT session,cam,t,track,x,y,conf FROM events WHERE session=?", [session]
        if t0 is not None:
            q += " AND t>=?"; a.append(t0)
        if t1 is not None:
            q += " AND t<?"; a.append(t1)
        q += " ORDER BY t"
        with self._tx() as c:
            return [Event.from_row(r) for r in c.execute(q, a)]

    def sessions(self) -> list:
        with self._tx() as c:
            return [dict(id=r[0], site=r[1], mode=r[2], started=r[3], meta=json.loads(r[4] or "{}"))
                    for r in c.execute("SELECT * FROM sessions ORDER BY started")]
