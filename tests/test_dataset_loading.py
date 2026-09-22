import os
import tempfile

import numpy as np
import torch
from PIL import Image

from src.data.transforms import build_transforms
from src.data.dataset import create_datasets, create_loaders
from src.db.sqlite_repo import SQLiteRepository


def create_image(path: str) -> None:
    arr = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def test_dataset_loading_shape():
    with tempfile.TemporaryDirectory() as tmpdir:
        img1 = os.path.join(tmpdir, "img1.png")
        img2 = os.path.join(tmpdir, "img2.png")
        create_image(img1)
        create_image(img2)

        db_path = os.path.join(tmpdir, "test.db")
        repo = SQLiteRepository(db_path)
        repo.insert_samples([
            {"image_path": img1, "label": 0, "split": "train"},
            {"image_path": img2, "label": 1, "split": "train"},
        ])
        train_samples = repo.get_samples("train")

        transforms = build_transforms()
        train_ds, _ = create_datasets(train_samples, [], transforms, cache_rate=0.0)
        train_loader, _, _ = create_loaders(train_ds, None, batch_size=2, num_workers=0)

        batch = next(iter(train_loader))
        images = batch["image"]
        # 3 canais, nao 1: desde a F3b o pipeline replica o canal cinza para 3, porque
        # as redes do torchvision esperam RGB (ver src/data/transforms.py).
        assert images.shape == (2, 3, 224, 224)
        # os tres canais tem de ser copias identicas do mesmo cinza
        assert torch.equal(images[:, 0], images[:, 1])
        assert torch.equal(images[:, 0], images[:, 2])
        assert batch["label"].dtype == torch.long
