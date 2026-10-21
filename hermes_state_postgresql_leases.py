"""Native lease claims on the caller's genuine PostgreSQL transaction."""
import socket
from psycopg import sql


def stale_lease(row,now):
    from hermes_state import _compression_lock_holder_process_is_dead
    return (float(row['expires_at'])<=now or
            (row['owner_host']==socket.gethostname() and
             _compression_lock_holder_process_is_dead(row['holder'])))


def claim_lease_row(conn,table,key_column,key,holder,now,expires_at):
    if (table,key_column) not in {('session_turn_leases','conversation_id'),('compression_locks','session_id')}:
        raise ValueError('Unknown native lease owner')
    relation=sql.Identifier(conn.hermes_scope.schema,table)
    column=sql.Identifier(key_column)
    old=conn.execute(sql.SQL('SELECT holder,expires_at,owner_host FROM {} WHERE {}=%s FOR UPDATE').format(relation,column),(key,)).fetchone()
    reclaimed=None
    if old and stale_lease(old,now):
        conn.execute(sql.SQL('DELETE FROM {} WHERE {}=%s AND holder=%s').format(relation,column),(key,old['holder']))
        reclaimed=old['holder']
    # An exact existing owner's acquisition is idempotent, including its original expiry.
    # Renewal is an explicit operation; acquisition must not silently extend ownership.
    conn.execute(sql.SQL('INSERT INTO {}({},holder,acquired_at,expires_at,owner_host) VALUES(%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING').format(relation,column),(key,holder,now,expires_at,socket.gethostname()))
    owner=conn.execute(sql.SQL('SELECT holder FROM {} WHERE {}=%s').format(relation,column),(key,)).fetchone()
    return owner is not None and owner['holder']==holder,reclaimed
