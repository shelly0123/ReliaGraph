"""Small persistent SQLite JSON cache for expensive local extraction calls."""
from __future__ import annotations
from pathlib import Path
import json
import sqlite3


class SQLiteJSONCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=20) as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("CREATE TABLE IF NOT EXISTS extraction_cache (cache_key TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)")

    def get(self, key: str):
        with sqlite3.connect(self.path, timeout=20) as con:
            row = con.execute("SELECT value_json FROM extraction_cache WHERE cache_key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value) -> None:
        with sqlite3.connect(self.path, timeout=20) as con:
            con.execute("INSERT OR REPLACE INTO extraction_cache(cache_key,value_json,updated_at) VALUES(?,?,CURRENT_TIMESTAMP)",
                        (key, json.dumps(value, ensure_ascii=False, separators=(",", ":"))))

    def clear(self) -> None:
        with sqlite3.connect(self.path, timeout=20) as con:
            con.execute("DELETE FROM extraction_cache")
