import os
from dataclasses import dataclass
from typing import Optional


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(key)
    return value if value is not None else default


@dataclass
class TrainConfig:
    db_path: str = _env("DB_PATH", "results/demo.db")
    data_root: str = _env("DATA_ROOT", "data/MedNIST")
    results_dir: str = _env("RESULTS_DIR", "monai_wieghts")
    run_id: str = _env("RUN_ID", "run_mednist_50")
    batch_size: int = int(_env("BATCH_SIZE", "64"))
    epochs: int = int(_env("EPOCHS", "50"))
    lr: float = float(_env("LR", "1e-3"))
    weight_decay: float = float(_env("WEIGHT_DECAY", "1e-4"))
    num_workers: int = int(_env("NUM_WORKERS", "0"))
    cache_rate: float = float(_env("CACHE_RATE", "0.0"))
    in_channels: int = int(_env("IN_CHANNELS", "1"))
    num_classes: int = int(_env("NUM_CLASSES", "6"))
    seed: int = int(_env("SEED", "42"))
    save_every_epoch: bool = _env("SAVE_EVERY_EPOCH", "1") == "1"
    val_split: float = float(_env("VAL_SPLIT", "0.2"))
    log_level: str = _env("LOG_LEVEL", "INFO")
    debug: bool = _env("DEBUG", "0") == "1"
    sample_log_count: int = int(_env("SAMPLE_LOG_COUNT", "5"))

    def to_dict(self) -> dict:
        return self.__dict__.copy()
