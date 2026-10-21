"""Durable media retirement candidates survive transcript deletion and failed cleanup."""
import hashlib
from hermes_state_sql import statement as _storage_statement, is_postgres, execute_many
import json

PREFIX = 'gateway.media.retired.v1.'


def retire_media(conn, payload, request_id=''):
    from gateway.session_ingress_media import admission_media_references, hosted_document_references
    references = [*admission_media_references(payload), *hosted_document_references(request_id, payload),
                  *payload.get('api_turn_v1', {}).get('media', [])]
    for reference in references:
        key = PREFIX + hashlib.sha256(reference['path'].encode()).hexdigest()
        conn.execute(_storage_statement(sqlite='INSERT OR IGNORE INTO state_meta(key,value) VALUES(?,?)', postgresql='INSERT INTO state_meta(key,value) VALUES(%s,%s) ON CONFLICT DO NOTHING', connection=conn),
                     (key, json.dumps(reference)))


def collect_retired_media(db):
    """Release retired candidates no admission or retained transcript row owns; drop collected keys.
    Synchronous SQLite plus one messages pass: async callers must run it off the event loop."""
    from gateway.session_ingress_media import release_unheld_media
    with db._read_ctx() as conn:
        # Primary-key range, not LIKE: the common no-candidate case costs one index probe.
        rows = conn.execute(_storage_statement(sqlite='SELECT key,value FROM state_meta WHERE key>=? AND key<?', postgresql='SELECT key,value FROM state_meta WHERE key>=%s AND key<%s', connection=conn),
                            (PREFIX, PREFIX[:-1] + chr(ord(PREFIX[-1]) + 1))).fetchall()
    if rows:
        release_unheld_media(db, [json.loads(row['value']) for row in rows])
        from pathlib import Path
        gone = [(row['key'],) for row in rows if not Path(json.loads(row['value'])['path']).exists()]
        if gone:
            db._execute_write(lambda conn: execute_many(conn,_storage_statement(sqlite='DELETE FROM state_meta WHERE key=?', postgresql='DELETE FROM state_meta WHERE key=%s', connection=conn), gone))
