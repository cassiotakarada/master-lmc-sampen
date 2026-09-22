import argparse
import logging
import os
import shutil
import sys
import time
from typing import List

import numpy as np
import torch
import torchvision
from torch.utils.data import Subset

from .config import TrainConfig
from .data.transforms import build_transforms
from .data.dataset import create_loaders
from .data.label_noise import RuidoDeRotulo
from .data.mednist import MedNISTDataset, ParticaoMedNIST
from .data.split import resumo_particao, stratified_subset_indices
from .training.model import build_model
from .training.trainer import Trainer
from .training.complexity_monitor import ComplexityMonitor, LimiaresStatus
from .training.checkpoints import save_initial_weights, save_config
from .utils.param_types import save_param_types
from .utils.paths import ensure_dir
from .utils.seed import set_seed
from .utils.logging import setup_logger, get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treinador de classificacao 2D (PyTorch puro)")
    parser.add_argument("--db_path", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--cache_rate", type=float, default=None)
    parser.add_argument("--in_channels", type=int, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_every_epoch", action="store_true", default=None)
    parser.add_argument("--data_seed", type=int, default=None,
                        help="semente da PARTICAO treino/val/teste (fixa entre runs)")
    parser.add_argument("--val_frac", type=float, default=None)
    parser.add_argument("--test_frac", type=float, default=None)
    parser.add_argument("--train_fraction", type=float, default=None,
                        help="fracao do treino a usar (1.0 = tudo)")
    parser.add_argument("--noise_seed", type=int, default=None,
                        help="semente do sorteio do ruido de rotulo; omitida = usa data_seed")
    parser.add_argument("--label_noise", type=float, default=None,
                        help="fracao dos rotulos de TREINO sorteados ao acaso (0.30 = 30%%)")
    parser.add_argument("--eval_train", action="store_true", default=None,
                        help="avalia o treino em eval() a cada epoca (checagem anti-BatchNorm)")
    parser.add_argument("--no_augment", dest="augment", action="store_false", default=None,
                        help="desliga o aumento de dados (use no run de ruido de rotulo)")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--flatten_order", type=str, default=None,
                        choices=["n_major", "x_major"],
                        help="ordem do achatamento da densa: n_major = n1x1,n1x2,... (padrao)")
    parser.add_argument("--lmc_bins", type=int, default=None)
    parser.add_argument("--sampen_r_factor", type=float, default=None)
    parser.add_argument("--no_live_complexity", dest="live_complexity",
                        action="store_false", default=None,
                        help="desliga o calculo de LMC/SampEn durante o treino")
    parser.add_argument("--no_live_sampen2d", dest="live_sampen2d",
                        action="store_false", default=None)
    parser.add_argument("--sampen2d_m", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--overfit_delta", type=float, default=None)
    parser.add_argument("--early_stop", action="store_true", default=None,
                        help="para quando val_loss nao melhora ha `patience` epocas "
                             "(desligado por padrao: impediria observar grokking)")
    parser.add_argument("--log_level", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=None)
    parser.add_argument("--sample_log_count", type=int, default=None)
    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()
    for field in cfg.__dataclass_fields__:  # type: ignore[attr-defined]
        arg_val = getattr(args, field, None)
        if arg_val is not None:
            setattr(cfg, field, arg_val)
    if getattr(args, "debug", None):
        cfg.debug = True
        cfg.log_level = "DEBUG"
    if getattr(args, "log_level", None):
        cfg.log_level = args.log_level
    return cfg


def log_dataset_samples(dataset, split: str, n: int) -> None:
    logger = get_logger()
    preview = min(n, len(dataset))
    for idx in range(preview):
        item = dataset[idx]
        logger.info("Sample %s | idx=%d label=%s shape=%s", split, idx, item.get("label"), np.array(item.get("image")).shape)


def prepare_mednist_datasets(cfg: TrainConfig, transforms_treino, transforms_aval):
    """Carrega os tres conjuntos do MedNIST a partir da particao CONGELADA.

    A particao vive em `data/mednist_split.json` (ver scripts/congelar_particao.py).
    Ela deixou de depender da implementacao interna do MONAI e passou a ser um dado
    versionado do projeto -- qualquer pessoa reproduz exatamente a mesma divisao.

    Apenas o TREINO recebe aumento de dados; validacao e teste usam a regua fixa.
    Se `cfg.train_fraction < 1.0`, so o treino e subamostrado, de forma estratificada.
    """
    logger = get_logger()
    particao = ParticaoMedNIST(cfg.split_file)
    logger.info("Particao congelada carregada de %s | %s", cfg.split_file, particao.resumo())

    train_ds = MedNISTDataset("train", transform=transforms_treino, particao=particao)
    val_ds = MedNISTDataset("val", transform=transforms_aval, particao=particao)
    test_ds = MedNISTDataset("test", transform=transforms_aval, particao=particao)

    if cfg.label_noise > 0:
        semente_ruido = cfg.noise_seed if cfg.noise_seed is not None else cfg.data_seed
        train_ds = RuidoDeRotulo(train_ds, cfg.label_noise, cfg.num_classes, seed=semente_ruido)
        r = train_ds.relatorio
        logger.info(
            "RUIDO DE ROTULO no treino: %.1f%% sorteados -> %d de %d rotulos efetivamente "
            "ERRADOS (%.2f%%). Validacao e teste permanecem INTACTOS.",
            r["fracao_sorteada"] * 100, r["n_efetivamente_errados"],
            len(train_ds), r["fracao_efetiva"] * 100,
        )
        logger.info("Semente do SORTEIO do ruido noise_seed=%d (independente de seed=%d)",
                    semente_ruido, cfg.seed)
        logger.info(
            "Se a rede IGNORAR o ruido: val_acc alta e train_eval_acc ~= %.0f%% (a fracao "
            "de rotulos corretos). Se DECORAR: train_eval_acc sobe rumo a 100%% e val_acc CAI. "
            "E essa separacao que caracteriza overfitting.",
            (1 - r["fracao_efetiva"]) * 100
        )

    n_train_original = len(train_ds)
    if cfg.train_fraction < 1.0:
        indices = stratified_subset_indices(train_ds.rotulos(), cfg.train_fraction, seed=cfg.data_seed)
        train_ds = Subset(train_ds, indices)
        logger.info(
            "Treino subamostrado (train_fraction=%.3f): %d -> %d amostras, estratificado por classe",
            cfg.train_fraction, n_train_original, len(train_ds),
        )

    logger.info("Particao dos dados | %s", resumo_particao(len(train_ds), len(val_ds), len(test_ds)))
    logger.info(
        "Semente da PARTICAO data_seed=%d (fixa, congelada em disco) | semente do TREINO seed=%d (varia)",
        cfg.data_seed, cfg.seed,
    )
    logger.info(
        "Disciplina dos conjuntos: treino ajusta os pesos | validacao escolhe a epoca "
        "(nunca entra no gradiente) | teste e olhado UMA UNICA VEZ, no fim"
    )
    return train_ds, val_ds, test_ds


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    cfg = merge_config(args)
    set_seed(cfg.seed)

    run_dir_path = os.path.join(cfg.results_dir, cfg.run_id)
    existed = os.path.exists(run_dir_path)
    run_dir = ensure_dir(run_dir_path)
    logger = setup_logger(run_dir, level="DEBUG" if cfg.debug else cfg.log_level)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Startup: python=%s torch=%s torchvision=%s device=%s seed=%d log_level=%s",
        sys.version.replace("\n", " "),
        torch.__version__,
        torchvision.__version__,
        device,
        cfg.seed,
        logging.getLevelName(logger.level),
    )
    logger.info("Run directory: %s (existed=%s)", run_dir, existed)
    logger.info("Configuration: %s", cfg.to_dict())

    try:
        transforms_treino = build_transforms(size=(cfg.image_size, cfg.image_size), treino=cfg.augment)
        transforms_aval = build_transforms(size=(cfg.image_size, cfg.image_size), treino=False)
        logger.info("Transformacoes | treino: %s", transforms_treino)
        logger.info("Transformacoes | validacao/teste (sem aumento): %s", transforms_aval)
        logger.info("Expected modality: grayscale in_channels=%d image_size=%d", cfg.in_channels, cfg.image_size)

        train_ds, val_ds, test_ds = prepare_mednist_datasets(cfg, transforms_treino, transforms_aval)
        log_dataset_samples(train_ds, "train", cfg.sample_log_count)
        log_dataset_samples(val_ds, "val", cfg.sample_log_count)
        log_dataset_samples(test_ds, "test", cfg.sample_log_count)

        train_loader, val_loader, test_loader = create_loaders(
            train_ds, val_ds, test_ds,
            batch_size=cfg.batch_size, num_workers=cfg.num_workers, seed=cfg.seed, logger=logger,
        )

        # --- trava contra o artefato de BatchNorm (ver plan.md, achado da F3) ---
        # As estatisticas moveis do BatchNorm partem de media 0 / variancia 1 e so
        # convergem ao longo de muitas atualizacoes. Com poucos passos de treino elas
        # ficam ruins, e o modelo desaba em eval() -- inclusive sobre o PROPRIO treino.
        # Isso imita overfitting de forma convincente e levaria a medir o fenomeno errado.
        passos_por_epoca = len(train_loader) if train_loader is not None else 0
        passos_totais = passos_por_epoca * cfg.epochs
        logger.info("Passos de treino: %d por epoca x %d epocas = %d atualizacoes do BatchNorm",
                    passos_por_epoca, cfg.epochs, passos_totais)
        if passos_totais < 200:
            logger.warning(
                "POUCOS PASSOS (%d < 200): as estatisticas moveis do BatchNorm podem nao "
                "convergir. O modelo pode desabar em eval() por artefato, NAO por overfitting. "
                "Antes de interpretar o resultado, confira a acuracia do proprio TREINO em eval(). "
                "Para induzir overfitting, prefira ruido de rotulo (mantem o dataset inteiro) "
                "a reduzir train_fraction.",
                passos_totais,
            )

        set_seed(cfg.seed)  # re-anchor immediately before weight init for reproducible initial weights
        model = build_model(cfg.model_name, in_channels=cfg.in_channels, num_classes=cfg.num_classes)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        has_nan = any(torch.isnan(p).any().item() for p in model.parameters())
        logger.info(
            "Model: %s in_channels=%d num_classes=%d total_params=%d trainable_params=%d has_nan_init=%s",
            cfg.model_name,
            cfg.in_channels,
            cfg.num_classes,
            total_params,
            trainable_params,
            has_nan,
        )

        save_initial_weights(model, os.path.join(run_dir, "initial_weights.pth"), logger=logger)
        snapshot = cfg.to_dict()
        snapshot["particao"] = {
            "n_treino": len(train_ds), "n_validacao": len(val_ds), "n_teste": len(test_ds),
            "resumo": resumo_particao(len(train_ds), len(val_ds), len(test_ds)),
        }
        save_config(snapshot, os.path.join(run_dir, "config.json"), logger=logger)
        pt_path = os.path.join(run_dir, "param_types.json")
        save_param_types(model, pt_path, logger=logger)
        for _prefix in ("initial_weights", "bef_weights", "aft_weights"):
            shutil.copy(pt_path, os.path.join(run_dir, f"{_prefix}_param_types.json"))

        monitor = None
        if cfg.live_complexity:
            monitor = ComplexityMonitor(
                order=cfg.flatten_order,
                n_bins=cfg.lmc_bins,
                r_scale=cfg.sampen_r_factor,
                calcular_2d=cfg.live_sampen2d,
                sampen2d_m=cfg.sampen2d_m,
                limiares=LimiaresStatus(patience=cfg.patience),
                logger=logger,
            )
            amostra = monitor.medir(model)
            logger.info(
                "Monitor ao vivo LIGADO | camada=%s [%d neuronios x %d entradas = %d pesos, "
                "todos usados] | ordem=%s | bins=%d | r=%.2f*sigma | 2D=%s",
                amostra["camada"], amostra["n_neuronios"], amostra["n_entradas"],
                amostra["n_pesos"], cfg.flatten_order, cfg.lmc_bins,
                cfg.sampen_r_factor, cfg.live_sampen2d,
            )
        else:
            logger.info("Monitor ao vivo DESLIGADO (--no_live_complexity)")

        tb_log_dir = os.path.join("runs", cfg.run_id)
        trainer = Trainer(
            model=model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            epochs=cfg.epochs,
            results_dir=run_dir,
            save_every_epoch=cfg.save_every_epoch,
            tb_log_dir=tb_log_dir,
            logger=logger,
            test_loader=test_loader,
            monitor=monitor,
            patience=cfg.patience,
            overfit_delta=cfg.overfit_delta,
            early_stop=cfg.early_stop,
            eval_train=cfg.eval_train,
        )

        trainer.fit()

        # O teste e consultado UMA UNICA VEZ, aqui, depois que o treino terminou e com
        # os pesos da melhor epoca de validacao -- nunca os da ultima epoca.
        trainer.evaluate_test()
        if trainer.best_val_acc is not None and cfg.epochs >= 50 and trainer.best_val_acc < 0.30:
            logger.warning("Validation accuracy is below 0.30 after %d epochs. Consider reviewing transforms, learning rate, or batch size.", cfg.epochs)
        logger.info(
            "Run complete: best_epoch=%s best_val_loss=%s best_val_acc=%s test=%s history=%s param_types=%s bef_weights=%s",
            trainer.best_epoch,
            trainer.best_val_loss,
            trainer.best_val_acc,
            trainer.test_metrics,
            os.path.join(run_dir, "history.csv"),
            os.path.join(run_dir, "param_types.json"),
            os.path.join(run_dir, "bef_weights.pth"),
        )
        elapsed = time.perf_counter() - start_time
        logger.info("Total runtime: %.2fs", elapsed)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training failed: %s", exc)
        fail_path = os.path.join(run_dir, "FAILED.txt")
        with open(fail_path, "w", encoding="utf-8") as f:
            f.write(f"Training failed: {exc}\n")
        raise


if __name__ == "__main__":
    main()
