"""Sequential experiment runner for monai_weights/.

Each entry in EXPERIMENTS becomes one training run. Runs are skipped if their
output dir already has a final history.csv with at least `epochs` rows, so this
script is safe to re-launch.

Two GPUs: launch one worker per GPU with --claim, e.g.
    CUDA_VISIBLE_DEVICES=0 python run_sweep.py --only pm_ --claim &
    CUDA_VISIBLE_DEVICES=1 python run_sweep.py --only pm_ --claim &
Each worker takes the next run nobody has claimed yet (claim = mkdir in
monai_weights/.fila/, which is atomic), so the queue balances itself.
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional


PYTHON = os.environ.get("SWEEP_PYTHON", "myenv/bin/python")
RESULTS_ROOT = "monai_weights"


@dataclass
class Experiment:
    run_id: str
    model_name: str
    image_size: int
    seed: int
    epochs: int
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    train_fraction: float = 1.0
    image_noise: str = "nenhum"          # "branco" | "rosa" | "marrom" (src/data/image_noise.py)
    image_noise_sigma: float = 0.1
    noise_seed: Optional[int] = None
    augment: bool = True
    eval_train: bool = False
    num_workers: int = 4
    dataset: str = "mednist"
    save_every_epoch: bool = True

    def to_cli(self) -> List[str]:
        cli = [
            "--results_dir", RESULTS_ROOT,
            "--run_id", self.run_id,
            "--model_name", self.model_name,
            "--image_size", str(self.image_size),
            "--seed", str(self.seed),
            "--epochs", str(self.epochs),
            "--batch_size", str(self.batch_size),
            "--lr", str(self.lr),
            "--weight_decay", str(self.weight_decay),
            "--train_fraction", str(self.train_fraction),
            "--image_noise", self.image_noise,
            "--image_noise_sigma", str(self.image_noise_sigma),
            "--num_workers", str(self.num_workers),
            "--dataset", self.dataset,
        ]
        if self.save_every_epoch:
            cli.append("--save_every_epoch")
        if self.noise_seed is not None:
            cli += ["--noise_seed", str(self.noise_seed)]
        if not self.augment:
            cli.append("--no_augment")
        if self.eval_train:
            cli.append("--eval_train")
        return cli


# ===========================================================================
# Controles: treino normal (rotulos e imagens intactos)
# ===========================================================================
# 60 epocas, sem aumento de dados, sem weight decay e com eval_train: a mesma
# configuracao dos regimes de subamostragem e de ruido nas imagens, para que o regime
# seja a unica diferenca. Os runs ja existem em monai_weights/ (o runner os pula).
EXPERIMENTS: List[Experiment] = [
    Experiment(run_id=f"f6_ctrlpar_seed{_s}", model_name="resnet18", image_size=64, seed=_s,
               epochs=60, augment=False, weight_decay=0.0, eval_train=True)
    for _s in (42, 1, 7, 13, 23, 99, 101, 202)
] + [
    Experiment(run_id=f"dn121_ctrlpar_seed{_s}", model_name="densenet121", image_size=64, seed=_s,
               epochs=60, augment=False, weight_decay=0.0, eval_train=True)
    for _s in (13, 23, 99, 101, 202)
]

# ===========================================================================
# Ruido colorido nas IMAGENS de treino (src/data/image_noise.py)
# ===========================================================================
# Branco (potencia plana) e rosa (potencia ~ 1/f), FIXO por imagem. Mesma configuracao
# dos controles pareados f6_ctrlpar_seed*: 60 epocas, sem aumento de dados, sem weight
# decay e com eval_train -- assim o ruido e a unica diferenca, e os controles ja existem.
# noise_seed = seed, como nas demais replicacoes.
#
# SIGMA_RUIDO: intensidade na escala do pixel [0, 1]. Valores definidos: 0,05, 0,1 e 0,2
# (rosa e branco), os mesmos no MedNIST e no PathMNIST.
SIGMA_RUIDO = (0.05, 0.1, 0.2)
for _cor in ("branco", "rosa"):
    for _sig in SIGMA_RUIDO:
        for _s in (42, 1, 7, 13, 23):
            EXPERIMENTS.append(Experiment(
                run_id=f"f9_{_cor}_s{round(_sig * 100):03d}_seed{_s}", model_name="resnet18",
                image_size=64, seed=_s, epochs=60, image_noise=_cor, image_noise_sigma=_sig,
                noise_seed=_s, augment=False, weight_decay=0.0, eval_train=True))



# ===========================================================================
# PathMNIST (segundo dataset, decisao de 2026-10-23): as MESMAS situacoes do MedNIST
# ===========================================================================
# 64x64 (pathmnist_64.npz do MedMNIST+), particao oficial, 9 classes, RGB. Mesma receita
# dos controles pareados: 60 epocas, sem aumento de dados, sem weight decay, eval_train.
# Prefixo pm_ para nao misturar com os runs do MedNIST.
PM = dict(dataset="pathmnist", image_size=64, augment=False, weight_decay=0.0, eval_train=True)
SEMENTES_PM = (42, 1, 7, 13, 23)

EXPERIMENTS += [
    Experiment(run_id=f"pm_ctrl_seed{_s}", model_name="resnet18", seed=_s, epochs=60, **PM)
    for _s in (42, 1, 7, 13, 23, 99, 101, 202)
] + [
    Experiment(run_id=f"pm_dn121_ctrl_seed{_s}", model_name="densenet121", seed=_s, epochs=60, **PM)
    for _s in (13, 23, 99, 101, 202)
]

# Subamostragem. Controle: 89.996 imagens = 704 passos/epoca x 60 = 42.240 passos.
#   fracao  n_treino  passos/epoca  epocas (passos igualados = ceil(42.240 / passos))
#   10 %    9.000     71            595
#    2 %    1.800     15            2.816
#    1 %      901      8            5.280
# Curta: 60 epocas (menos dados E menos passos). Igualada: so menos dados. A igualada
# nao grava checkpoint por epoca (milhares de epocas x 45 MB); LMC/SampEn ja ficam em
# complexity_live.csv.
EPOCAS_IGUALADAS_PM = {0.10: 595, 0.02: 2816, 0.01: 5280}
for _f, _ep in EPOCAS_IGUALADAS_PM.items():
    _tag = f"{round(_f * 100):02d}"
    for _s in SEMENTES_PM:
        EXPERIMENTS.append(Experiment(run_id=f"pm_frac{_tag}_seed{_s}", model_name="resnet18",
                                      seed=_s, epochs=60, train_fraction=_f, **PM))
        EXPERIMENTS.append(Experiment(run_id=f"pm_frac{_tag}eq_seed{_s}", model_name="resnet18",
                                      seed=_s, epochs=_ep, train_fraction=_f,
                                      save_every_epoch=False, **PM))

# Ruido branco e rosa nas imagens de treino, mesmos sigmas do MedNIST. O mesmo campo 2D
# e somado aos 3 canais RGB (src/data/image_noise.py).
for _cor in ("branco", "rosa"):
    for _sig in SIGMA_RUIDO:
        for _s in SEMENTES_PM:
            EXPERIMENTS.append(Experiment(
                run_id=f"pm_{_cor}_s{round(_sig * 100):03d}_seed{_s}", model_name="resnet18",
                seed=_s, epochs=60, image_noise=_cor, image_noise_sigma=_sig, noise_seed=_s, **PM))


def is_complete(run_dir: str, epochs: int) -> bool:
    history = os.path.join(run_dir, "history.csv")
    if not os.path.isfile(history):
        return False
    try:
        with open(history) as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            rows = sum(1 for _ in reader)
        return rows >= epochs
    except Exception:
        return False


def claim(run_id: str) -> bool:
    """Reserva o run para este worker. mkdir e atomico: so um worker consegue."""
    fila = os.path.join(RESULTS_ROOT, ".fila")
    os.makedirs(fila, exist_ok=True)
    try:
        os.mkdir(os.path.join(fila, run_id))
        return True
    except FileExistsError:
        return False


def run_one(exp: Experiment, dry_run: bool, use_claim: bool = False) -> int:
    run_dir = os.path.join(RESULTS_ROOT, exp.run_id)
    log_path = os.path.join(RESULTS_ROOT, f"{exp.run_id}_train.log")

    if is_complete(run_dir, exp.epochs):
        print(f"[SKIP] {exp.run_id} already has {exp.epochs}+ epochs in history.csv")
        return 0
    if use_claim and not dry_run and not claim(exp.run_id):
        print(f"[SKIP] {exp.run_id} claimed by another worker")
        return 0

    cmd = [PYTHON, "-m", "src.main_train"] + exp.to_cli()
    print(f"[RUN ] {exp.run_id} -> {' '.join(cmd)}")
    if dry_run:
        return 0

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    start = time.perf_counter()
    with open(log_path, "w") as logf:
        logf.write(f"# command: {' '.join(cmd)}\n")
        logf.flush()
        env = os.environ.copy()
        if not exp.save_every_epoch:
            # o padrao do TrainConfig (SAVE_EVERY_EPOCH=1) so e desligado pelo ambiente
            env["SAVE_EVERY_EPOCH"] = "0"
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    elapsed = time.perf_counter() - start
    print(f"[{'OK  ' if proc.returncode == 0 else 'FAIL'}] {exp.run_id} rc={proc.returncode} elapsed={elapsed:.0f}s log={log_path}")
    return proc.returncode


def custo_relativo(exp: Experiment) -> float:
    """Estimativa grosseira do tempo: passos de treino + avaliacoes por epoca."""
    passos = exp.epochs * exp.train_fraction
    avaliacoes = exp.epochs * 0.3          # validacao (+ eval_train) a cada epoca
    peso = 2.4 if exp.model_name == "densenet121" else 1.0
    return peso * (passos + avaliacoes)


def main():
    p = argparse.ArgumentParser(description="Sequential experiment runner")
    p.add_argument("--dry_run", action="store_true", help="Print commands without running")
    p.add_argument("--only", default=None, help="Comma-separated run_ids to include (substring match)")
    p.add_argument("--claim", action="store_true",
                   help="reserve each run in monai_weights/.fila/ (one worker per GPU)")
    p.add_argument("--longest_first", action="store_true",
                   help="run the longest experiments first (better balance across GPUs)")
    args = p.parse_args()

    targets = EXPERIMENTS
    if args.only:
        wants = [s.strip() for s in args.only.split(",") if s.strip()]
        targets = [e for e in EXPERIMENTS if any(w in e.run_id for w in wants)]
        if not targets:
            print(f"No experiments match --only={args.only}")
            return 1

    if args.longest_first:
        targets = sorted(targets, key=custo_relativo, reverse=True)

    print(f"Sweep: {len(targets)} experiment(s)")
    failures = 0
    for exp in targets:
        rc = run_one(exp, args.dry_run, args.claim)
        if rc != 0:
            failures += 1
    print(f"Done. failures={failures}/{len(targets)}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
