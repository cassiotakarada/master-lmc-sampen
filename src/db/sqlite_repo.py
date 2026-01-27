import os
import sqlite3
from typing import List, Dict, Optional, Sequence

from .models import rows_to_samples
from .repository import Repository
from ..utils.logging import get_logger


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT NOT NULL,
    label INTEGER NOT NULL,
    split TEXT NOT NULL
);
"""


class SQLiteRepository(Repository):
    def __init__(self, db_path: str, logger=None):
        self.db_path = db_path
        self.logger = logger or get_logger()
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.logger.info("Connecting to SQLite db at %s", db_path)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(SCHEMA_SQL)
            conn.commit()
            row = conn.execute("SELECT COUNT(*) FROM samples").fetchone()
            count = row[0] if row else 0
        self.logger.info("Ensured samples table exists; current row count=%d", count)

    def get_samples(self, split: str) -> List[Dict]:
        query = "SELECT id, image_path, label, split FROM samples"
        params: Sequence = []
        if split != "all":
            query += " WHERE split=?"
            params = [split]
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        samples = rows_to_samples(rows)
        result = []
        dropped = 0
        for s in samples:
            if s.image_path is None or s.label is None:
                dropped += 1
                continue
            result.append({"image": s.image_path, "label": s.label})
        if dropped:
            self.logger.warning("Dropped %d invalid rows missing path/label for split=%s", dropped, split)
        self.logger.info("Fetched %d samples for split=%s", len(result), split)
        return result

    def insert_samples(self, rows: List[Dict]) -> None:
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO samples (image_path, label, split) VALUES (?, ?, ?)",
                [(r["image_path"], r["label"], r.get("split", "train")) for r in rows],
            )
            conn.commit()
        self.logger.info("Inserted %d rows", len(rows))


def bootstrap_sqlite(db_path: str, image_paths: List[str], labels: Optional[List[int]] = None, split_ratio: float = 0.8, logger=None) -> None:
    """Create a small sqlite db with provided images. Labels default to zeros."""
    repo = SQLiteRepository(db_path, logger=logger)
    labels = labels or [0 for _ in image_paths]
    split_index = int(len(image_paths) * split_ratio)
    rows = []
    for idx, (img, label) in enumerate(zip(image_paths, labels)):
        split = "train" if idx < split_index else "val"
        rows.append({"image_path": img, "label": int(label), "split": split})
    repo.insert_samples(rows)
    (logger or get_logger()).info("Bootstrapped sqlite with %d images (split_ratio=%.2f)", len(rows), split_ratio)
