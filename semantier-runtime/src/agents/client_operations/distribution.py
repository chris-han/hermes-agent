"""Caddy origin request facts, not users or installed population."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit,unquote


def ingest_logs(store,root):
    root=Path(root)
    for path in sorted(root.glob('access*.log*')):
        if path.suffix=='.gz' or path.is_symlink() or not path.is_file():continue
        with path.open('rb') as stream:
            stat=path.stat();first=stream.readline();stream.seek(0)
            # Inode survives renaming/rotation. Prefix identity prevents reuse of a
            # recycled inode from silently skipping a new source generation.
            source=hashlib.sha256(f'{stat.st_dev}:{stat.st_ino}:'.encode()+first).hexdigest()
            with store.connection() as c:
                c.execute('INSERT INTO access_sources(source_id,source_path) VALUES(%s,%s) ON CONFLICT DO NOTHING',(source,path.name))
                state=c.execute('SELECT * FROM access_sources WHERE source_id=%s FOR UPDATE',(source,)).fetchone()
                cursor=state['cursor_bytes']
                if stat.st_size<cursor:
                    c.execute('UPDATE access_sources SET gap=true WHERE source_id=%s',(source,));continue
                stream.seek(cursor)
                for _ in range(10000):
                    offset=stream.tell();line=stream.readline(1024*1024+1)
                    if not line or not line.endswith(b'\n'):break
                    try:
                        record=json.loads(line);request=record['request'];uri=urlsplit(record.get('artifact_path') or request['uri']).path
                        method=request['method'];status=int(record['status']);size=int(record['size'])
                        stamp=datetime.fromtimestamp(float(record['ts']),timezone.utc)
                        if method not in {'GET','HEAD'} or size<0:raise ValueError('unsupported request')
                    except (ValueError,KeyError,TypeError,OverflowError):
                        c.execute('UPDATE access_sources SET gap=true WHERE source_id=%s',(source,));cursor=stream.tell();continue
                    # No IP, credentials, query strings, headers or personal IDs persist.
                    c.execute('INSERT INTO access_events VALUES(%s,%s,%s,%s,%s,%s,%s,now()) ON CONFLICT DO NOTHING',(source,offset,unquote(uri).lstrip('/'),method,status,size,stamp));cursor=stream.tell()
                c.execute('UPDATE access_sources SET cursor_bytes=%s,last_ingested_at=now(),source_path=%s WHERE source_id=%s',(cursor,path.name,source))
    # Raw retention is limited to the dedicated, declared log root only.
    for path in root.glob('access*.log*'):
        if path.name!='access.log' and not path.is_symlink() and path.is_file() and datetime.now(timezone.utc).timestamp()-path.stat().st_mtime>30*86400:path.unlink()


def summary(store,keys,start,end):
    with store.connection() as c:
        sources=c.execute('SELECT source_id,gap,last_ingested_at FROM access_sources').fetchall()
        rows=c.execute('SELECT artifact_key,method,status,count(*) AS requests,sum(bytes) AS bytes FROM access_events WHERE received_at>=%s AND received_at<%s AND (%s::text[] IS NULL OR artifact_key=ANY(%s)) GROUP BY artifact_key,method,status ORDER BY artifact_key,method,status',(start,end,keys,keys)).fetchall()
        return {'items':rows,'coverage':'no_source' if not sources else 'partial' if any(s['gap'] for s in sources) else 'collected_sources_only','sources':sources,'window_basis':'received_at_utc','window_start':start,'window_end':end,'limits':'Origin requests and response bytes; retries/ranges/caches do not establish people or installations.'}
