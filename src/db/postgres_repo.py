"""Optional Postgres-ready repository scaffold.
This module intentionally avoids importing heavy dependencies so that
SQLite usage stays dependency-free. Plug psycopg2 or SQLAlchemy as needed.
"""

from typing import List, Dict

from .repository import Repository


class PostgresRepository(Repository):
    def __init__(self, conn_str: str):
        self.conn_str = conn_str
        # Implement actual connection logic when dependency available.

    def get_samples(self, split: str) -> List[Dict]:
        raise NotImplementedError("PostgresRepository requires psycopg2/SQLAlchemy implementation")
