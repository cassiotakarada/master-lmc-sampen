"""Sequential experiment runner for monai_weights/.

Each entry in EXPERIMENTS becomes one training run. Runs are skipped if their
output dir already has a final history.csv with at least `epochs` rows, so this
script is safe to re-launch.
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
    label_noise: float = 0.0
    noise_seed: Optional[int] = None
    augment: bool = True
    eval_train: bool = False
    num_workers: int = 4

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
            "--label_noise", str(self.label_noise),
            "--num_workers", str(self.num_workers),
            "--save_every_epoch",
        ]
        if self.noise_seed is not None:
            cli += ["--noise_seed", str(self.noise_seed)]
        if not self.augment:
            cli.append("--no_augment")
        if self.eval_train:
            cli.append("--eval_train")
        return cli


# ===========================================================================
# Grid da fase F6 (plan.md)
# ===========================================================================
# Runs anteriores a F3b usaram a ResNet-18 do MONAI e NAO sao comparaveis com
# estes; formam uma linha de base separada. Prefixo "f6_" para nao confundir.
#
#  - 3 sementes x 40 epocas: treino normal, para a trajetoria tipica.
#  - 1 run de ruido de rotulo: o unico onde ha overfitting inequivoco.
#    Nele o aumento de dados e o weight decay sao DESLIGADOS de proposito --
#    ambos combatem memorizacao, que e justamente o fenomeno que queremos
#    provocar. `eval_train` liga a checagem anti-BatchNorm (ver plan.md/F3).
EXPERIMENTS: List[Experiment] = [
    Experiment(run_id="f6_seed42", model_name="resnet18", image_size=64, seed=42, epochs=40),
    Experiment(run_id="f6_seed1",  model_name="resnet18", image_size=64, seed=1,  epochs=40),
    Experiment(run_id="f6_seed7",  model_name="resnet18", image_size=64, seed=7,  epochs=40),
    # Mais 2 controles (n=5 dos dois lados): o contraste ruido x controle e o resultado
    # central da F6, e com apenas 3 controles o teste estatistico ficava limitado por eles.
    # Mesmas 40 epocas, mesmo aumento de dados e mesmo weight decay dos 3 primeiros --
    # trocar qualquer um deles quebraria a comparabilidade dentro do proprio grupo.
    Experiment(run_id="f6_seed13", model_name="resnet18", image_size=64, seed=13, epochs=40),
    Experiment(run_id="f6_seed23", model_name="resnet18", image_size=64, seed=23, epochs=40),
    Experiment(run_id="f6_ruido30", model_name="resnet18", image_size=64, seed=42, epochs=60,
               label_noise=0.30, augment=False, weight_decay=0.0, eval_train=True),
    # Replicacao do run de ruido (ressalva da F6: "n = 1 run com overfitting").
    # Variam DUAS coisas por semente: a inicializacao/ordem dos lotes (`seed`) e QUAIS
    # rotulos sao corrompidos (`noise_seed`). So mudar `seed` repetiria o experimento
    # sobre exatamente os mesmos rotulos errados -- nao seria replicacao de verdade.
    # A particao (data_seed=0) continua congelada, senao os runs deixariam de ser
    # comparaveis entre si e com os controles.
    Experiment(run_id="f6_ruido30_seed1", model_name="resnet18", image_size=64, seed=1, epochs=60,
               label_noise=0.30, noise_seed=1, augment=False, weight_decay=0.0, eval_train=True),
    Experiment(run_id="f6_ruido30_seed7", model_name="resnet18", image_size=64, seed=7, epochs=60,
               label_noise=0.30, noise_seed=7, augment=False, weight_decay=0.0, eval_train=True),
    # n=5: com cinco repeticoes da para reportar intervalo de confianca em vez de media +- desvio.
    Experiment(run_id="f6_ruido30_seed13", model_name="resnet18", image_size=64, seed=13, epochs=60,
               label_noise=0.30, noise_seed=13, augment=False, weight_decay=0.0, eval_train=True),
    Experiment(run_id="f6_ruido30_seed23", model_name="resnet18", image_size=64, seed=23, epochs=60,
               label_noise=0.30, noise_seed=23, augment=False, weight_decay=0.0, eval_train=True),
]


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


def run_one(exp: Experiment, dry_run: bool) -> int:
    run_dir = os.path.join(RESULTS_ROOT, exp.run_id)
    log_path = os.path.join(RESULTS_ROOT, f"{exp.run_id}_train.log")

    if is_complete(run_dir, exp.epochs):
        print(f"[SKIP] {exp.run_id} already has {exp.epochs}+ epochs in history.csv")
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
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    print(f"[{'OK  ' if proc.returncode == 0 else 'FAIL'}] {exp.run_id} rc={proc.returncode} elapsed={elapsed:.0f}s log={log_path}")
    return proc.returncode


def main():
    p = argparse.ArgumentParser(description="Sequential experiment runner")
    p.add_argument("--dry_run", action="store_true", help="Print commands without running")
    p.add_argument("--only", default=None, help="Comma-separated run_ids to include (substring match)")
    args = p.parse_args()

    targets = EXPERIMENTS
    if args.only:
        wants = [s.strip() for s in args.only.split(",") if s.strip()]
        targets = [e for e in EXPERIMENTS if any(w in e.run_id for w in wants)]
        if not targets:
            print(f"No experiments match --only={args.only}")
            return 1

    print(f"Sweep: {len(targets)} experiment(s)")
    failures = 0
    for exp in targets:
        rc = run_one(exp, args.dry_run)
        if rc != 0:
            failures += 1
    print(f"Done. failures={failures}/{len(targets)}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
