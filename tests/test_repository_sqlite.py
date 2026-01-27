import os
import tempfile

from src.db.sqlite_repo import SQLiteRepository


def test_repository_get_samples():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        repo = SQLiteRepository(db_path)
        rows = [
            {"image_path": "/tmp/a.png", "label": 0, "split": "train"},
            {"image_path": "/tmp/b.png", "label": 1, "split": "train"},
            {"image_path": "/tmp/c.png", "label": 0, "split": "val"},
        ]
        repo.insert_samples(rows)
        train = repo.get_samples("train")
        assert len(train) == 2
        assert set(train[0].keys()) == {"image", "label"}
        all_samples = repo.get_all_samples()
        assert len(all_samples) == 3
