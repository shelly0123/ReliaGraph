"""Reset ReliaGraph's local account while preserving knowledge-base data.

Run only after stopping the ReliaGraph server. This clears accounts and sessions
from auth.sqlite3 only; it does not touch the graph, uploads, registry, or documents.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_HOME = Path(
    os.environ.get(
        "RELIAGRAPH_DATA_DIR",
        os.environ.get("TOPICWEAVE_DATA_DIR", ROOT / "data" / "dashboard"),
    )
)
AUTH_DB = DATA_HOME / "auth.sqlite3"


def main() -> int:
    if not AUTH_DB.exists():
        print("No ReliaGraph login database exists yet.")
        print("Start run_reliagraph_windows.bat; the first account setup should appear.")
        return 0

    try:
        with sqlite3.connect(AUTH_DB, timeout=10) as source:
            tables = {
                row[0]
                for row in source.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if "users" not in tables or "sessions" not in tables:
                print(f"The file does not look like a ReliaGraph account database: {AUTH_DB}")
                print("No changes were made.")
                return 2

            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup_path = AUTH_DB.with_name(f"auth.backup-{stamp}.sqlite3")
            with sqlite3.connect(backup_path) as backup:
                source.backup(backup)

            source.execute("DELETE FROM sessions")
            source.execute("DELETE FROM users")
            source.commit()

        print("ReliaGraph login reset completed.")
        print(f"Account database backup: {backup_path}")
        print("Only usernames/password hashes and login sessions were cleared.")
        print("Your documents and knowledge graph were not modified.")
        print("Now run run_reliagraph_windows.bat and create a new local account.")
        return 0
    except (OSError, sqlite3.Error) as exc:
        print(f"Could not reset login: {exc}")
        print("Close ReliaGraph and retry. No data outside the account database is changed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
