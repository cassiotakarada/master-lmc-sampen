import os
from dataclasses import dataclass
from typing import Optional


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(key)
    return value if value is not None else default


@dataclass
class TrainConfig:
    # Conjunto de dados: "mednist" (6 classes, cinza) ou "pathmnist" (9 classes, RGB).
    # O numero de classes segue o dataset (ver NUM_CLASSES_POR_DATASET em main_train.py).
    dataset: str = _env("DATASET", "mednist")
    # .npz oficial do PathMNIST 64x64 (MedMNIST+, Zenodo 10519652)
    pathmnist_file: str = _env("PATHMNIST_FILE", "~/.medmnist/pathmnist_64.npz")
    db_path: str = _env("DB_PATH", "results/demo.db")
    data_root: str = _env("DATA_ROOT", "data/MedNIST")
    results_dir: str = _env("RESULTS_DIR", "monai_weights")
    run_id: str = _env("RUN_ID", "run_mednist_50")
    batch_size: int = int(_env("BATCH_SIZE", "64"))
    epochs: int = int(_env("EPOCHS", "50"))
    lr: float = float(_env("LR", "1e-3"))
    weight_decay: float = float(_env("WEIGHT_DECAY", "1e-4"))
    num_workers: int = int(_env("NUM_WORKERS", "0"))
    cache_rate: float = float(_env("CACHE_RATE", "0.0"))
    # 3 canais: o MedNIST replica o canal cinza (ver src/data/transforms.py); o PathMNIST ja e RGB
    in_channels: int = int(_env("IN_CHANNELS", "3"))
    num_classes: int = int(_env("NUM_CLASSES", "6"))
    seed: int = int(_env("SEED", "42"))
    save_every_epoch: bool = _env("SAVE_EVERY_EPOCH", "1") == "1"
    # --- particionamento dos dados (F3) ---
    # data_seed decide QUEM cai em treino/validacao/teste. Fica FIXA entre runs, para
    # que os runs sejam comparaveis e nenhuma amostra troque de conjunto entre eles.
    # Nao confundir com `seed` acima, que e a semente do treino (pesos e ordem dos lotes).
    # Particao CONGELADA em disco (F3b): deixou de depender de biblioteca externa
    split_file: str = _env("SPLIT_FILE", "data/mednist_split.json")
    data_seed: int = int(_env("DATA_SEED", "0"))
    val_frac: float = float(_env("VAL_FRAC", "0.1"))
    test_frac: float = float(_env("TEST_FRAC", "0.1"))
    # fracao do TREINO efetivamente usada; 1.0 = tudo. Valores pequenos (ex. 0.05)
    # servem para induzir overfitting inequivoco (run de controle da F6).
    train_fraction: float = float(_env("TRAIN_FRACTION", "1.0"))
    # Ruido colorido nas IMAGENS de treino (ver src/data/image_noise.py).
    # "nenhum" | "branco" (beta 0) | "rosa" (beta 1) | "marrom" (beta 2). Fixo por imagem.
    image_noise: str = _env("IMAGE_NOISE", "nenhum")
    # Expoente do espectro de potencia (~1/f^beta). Se definido, substitui o da cor.
    image_noise_beta: Optional[float] = (
        float(_env("IMAGE_NOISE_BETA")) if _env("IMAGE_NOISE_BETA") is not None else None
    )
    # Intensidade: desvio padrao do ruido na escala do pixel (imagens em [0, 1]).
    image_noise_sigma: float = float(_env("IMAGE_NOISE_SIGMA", "0.1"))
    # Semente do ruido. Fica SEPARADA de `data_seed` (particao, FIXA) e de `seed` (pesos e
    # ordem dos lotes): repetir o experimento trocando so `seed` manteria exatamente o mesmo
    # ruido em cada imagem. None = usa `seed`.
    noise_seed: Optional[int] = (
        int(_env("NOISE_SEED")) if _env("NOISE_SEED") is not None else None
    )
    # Avalia o proprio TREINO em eval() a cada epoca. Custa uma passada extra, mas e a
    # unica forma de distinguir overfitting real (treino alto, val baixa) de artefato de
    # BatchNorm (os dois baixos). Essencial nos runs de subamostragem.
    eval_train: bool = _env("EVAL_TRAIN", "0") == "1"
    # Aumento de dados (espelhamento horizontal) no treino. Os experimentos atuais rodam
    # sem ele (--no_augment), para que so o regime estudado mude entre os runs.
    augment: bool = _env("AUGMENT", "1") == "1"
    # --- monitor de complexidade ao vivo (F4) ---
    live_complexity: bool = _env("LIVE_COMPLEXITY", "1") == "1"
    flatten_order: str = _env("FLATTEN_ORDER", "n_major")   # n_major = n1x1, n1x2, ...
    lmc_bins: int = int(_env("LMC_BINS", "100"))            # fixado pelo estudo da F2
    sampen_r_factor: float = float(_env("SAMPEN_R_FACTOR", "0.20"))
    # --- deteccao de overfitting (F5) ---
    patience: int = int(_env("PATIENCE", "3"))
    overfit_delta: float = float(_env("OVERFIT_DELTA", "0.0"))
    # Desligado por padrao: parar cedo impediria observar grokking (melhora
    # tardia da validacao) e truncaria a trajetoria de LMC/SampEn que queremos ver.
    early_stop: bool = _env("EARLY_STOP", "0") == "1"
    log_level: str = _env("LOG_LEVEL", "INFO")
    debug: bool = _env("DEBUG", "0") == "1"
    sample_log_count: int = int(_env("SAMPLE_LOG_COUNT", "5"))
    model_name: str = _env("MODEL_NAME", "densenet121")
    image_size: int = int(_env("IMAGE_SIZE", "224"))

    def to_dict(self) -> dict:
        return self.__dict__.copy()
