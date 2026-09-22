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
    results_dir: str = _env("RESULTS_DIR", "monai_weights")
    run_id: str = _env("RUN_ID", "run_mednist_50")
    batch_size: int = int(_env("BATCH_SIZE", "64"))
    epochs: int = int(_env("EPOCHS", "50"))
    lr: float = float(_env("LR", "1e-3"))
    weight_decay: float = float(_env("WEIGHT_DECAY", "1e-4"))
    num_workers: int = int(_env("NUM_WORKERS", "0"))
    cache_rate: float = float(_env("CACHE_RATE", "0.0"))
    # 3 canais: o pipeline replica o canal cinza (ver src/data/transforms.py)
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
    # Ruido de rotulo: forma CORRETA de induzir overfitting (ver src/data/label_noise.py).
    # Mantem o dataset inteiro, logo o BatchNorm continua saudavel. Só o TREINO e afetado.
    label_noise: float = float(_env("LABEL_NOISE", "0.0"))
    # Semente do SORTEIO do ruido. Fica SEPARADA das outras duas porque sao tres fontes
    # de variacao distintas: `data_seed` decide a particao (FIXA), `seed` decide pesos
    # iniciais e ordem dos lotes, e esta decide QUAIS rotulos sao corrompidos. Repetir o
    # experimento de ruido trocando so `seed` manteria exatamente os mesmos rotulos
    # errados -- mediria sensibilidade a inicializacao, nao replicacao do experimento.
    # None = usa data_seed (comportamento anterior; mantem o run f6_ruido30 reproduzivel).
    noise_seed: Optional[int] = (
        int(_env("NOISE_SEED")) if _env("NOISE_SEED") is not None else None
    )
    # Avalia o proprio TREINO em eval() a cada epoca. Custa uma passada extra, mas e a
    # unica forma de distinguir overfitting real (treino alto, val baixa) de artefato de
    # BatchNorm (os dois baixos). Essencial no run de ruido de rotulo.
    eval_train: bool = _env("EVAL_TRAIN", "0") == "1"
    # Aumento de dados (espelhamento horizontal) no treino. Desligue no run de ruido de
    # rotulo: aumento COMBATE memorizacao, que e justamente o que queremos provocar la.
    augment: bool = _env("AUGMENT", "1") == "1"
    # --- monitor de complexidade ao vivo (F4) ---
    live_complexity: bool = _env("LIVE_COMPLEXITY", "1") == "1"
    flatten_order: str = _env("FLATTEN_ORDER", "n_major")   # n_major = n1x1, n1x2, ...
    lmc_bins: int = int(_env("LMC_BINS", "100"))            # fixado pelo estudo da F2
    sampen_r_factor: float = float(_env("SAMPEN_R_FACTOR", "0.20"))
    live_sampen2d: bool = _env("LIVE_SAMPEN2D", "1") == "1"
    # m=1 e obrigatorio numa densa de poucas linhas: com m=2 a SampEn2D fica indefinida
    sampen2d_m: int = int(_env("SAMPEN2D_M", "1"))
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
