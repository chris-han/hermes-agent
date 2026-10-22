from storage import hosted_postgres as hosted
import sqlite3
from pathlib import Path

from runtime_paths import sqlite_db_path

DB_PATH = sqlite_db_path("idempotency.db", ".semantier/idempotency.db", "idempotency.db")


def init_db():
    if hosted.enabled():
        raise hosted.HostedStorageUnavailable("legacy idempotency owner requires explicit hosted migration")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS events (org_id TEXT, hash TEXT, PRIMARY KEY (org_id, hash))")
    conn.commit()
    conn.close()


def seen(org_id: str, hash_val: str) -> bool:
    if hosted.enabled():
        raise hosted.HostedStorageUnavailable("legacy idempotency owner requires explicit hosted migration")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("SELECT 1 FROM events WHERE org_id=? AND hash=?", (org_id, hash_val))
    result = c.fetchone()
    conn.close()
    return result is not None


def commit(org_id: str, hash_val: str):
    if hosted.enabled():
        raise hosted.HostedStorageUnavailable("legacy idempotency owner requires explicit hosted migration")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO events(org_id, hash) VALUES(?,?)", (org_id, hash_val))
    conn.commit()
    conn.close()
