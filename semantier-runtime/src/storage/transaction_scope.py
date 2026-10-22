"""Real DB transaction scopes for retained store objects.

psycopg Connection context managers close the connection; SQLite's do not.
These scopes commit/rollback the actual connection without changing its SQL API.
"""
from contextlib import contextmanager
import threading

from storage.hosted_postgres import is_postgres


@contextmanager
def transaction_scope(connection):
    try:
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


class StoreMutationLock:
    """Preserve a store's read/decide/write lock across hosted processes."""
    def __init__(self,connection):
        self.connection=connection
        self._thread_lock=threading.RLock()
        self._depth=0

    def __enter__(self):
        self._thread_lock.acquire()
        try:
            if self._depth==0 and is_postgres(self.connection):
                self.connection.execute("SELECT pg_advisory_xact_lock(hashtext(current_schema()),hashtext('authority-write'))")
            self._depth+=1
            return self
        except BaseException:
            self._thread_lock.release()
            raise

    def __exit__(self,kind,error,traceback):
        self._depth-=1
        try:
            if self._depth==0 and is_postgres(self.connection):
                if kind is None:self.connection.commit()
                else:self.connection.rollback()
        finally:self._thread_lock.release()
