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
    # Semente 99: INEDITA de proposito. As outras cinco (42, 1, 7, 13, 23) participaram da
    # calibracao dos limiares do status -- ainda que 13 e 23 so como verificacao, os dados
    # delas ja existiam quando a grade foi buscada. Estes dois runs sao o primeiro teste
    # dos limiares em dados que nao existiam no momento da escolha, e rodando pelo caminho
    # de producao (status ao vivo no log), nao por replay.
    Experiment(run_id="f6_ruido30_seed99", model_name="resnet18", image_size=64, seed=99, epochs=60,
               label_noise=0.30, noise_seed=99, augment=False, weight_decay=0.0, eval_train=True),
    Experiment(run_id="f6_seed99", model_name="resnet18", image_size=64, seed=99, epochs=40),
    # Mais duas prospectivas (101 e 202). Com a 99 sozinha, "antecedencia +3" tanto podia
    # ser a regua real quanto azar de uma semente -- n=1 nao distingue as duas coisas.
    # Os controles vao junto para que a afirmacao "nao acusa treino saudavel" tambem seja
    # testada prospectivamente, e nao so em 1 run.
    Experiment(run_id="f6_ruido30_seed101", model_name="resnet18", image_size=64, seed=101, epochs=60,
               label_noise=0.30, noise_seed=101, augment=False, weight_decay=0.0, eval_train=True),
    Experiment(run_id="f6_seed101", model_name="resnet18", image_size=64, seed=101, epochs=40),
    Experiment(run_id="f6_ruido30_seed202", model_name="resnet18", image_size=64, seed=202, epochs=60,
               label_noise=0.30, noise_seed=202, augment=False, weight_decay=0.0, eval_train=True),
    Experiment(run_id="f6_seed202", model_name="resnet18", image_size=64, seed=202, epochs=40),
]

# ===========================================================================
# Niveis de ruido: o indicador vale so para 30 %? (maior lacuna que restava da F6)
# ===========================================================================
# Ate aqui TODO o resultado veio de um unico nivel de ruido. Sem variar isso, "a SampEn2D
# antecipa o overfitting" e uma afirmacao sobre 30 % de rotulos sorteados, nao sobre
# overfitting.
#
# As sementes 42, 1 e 7 sao REUSADAS de proposito: com a mesma inicializacao e a mesma
# ordem de lotes dos runs de 30 %, a unica coisa que muda entre os tres niveis e o ruido.
# Sementes novas misturariam duas fontes de variacao e a comparacao ficaria mais fraca.
#
# Fracao efetivamente ERRADA (o sorteio inclui a classe verdadeira, entao e f*5/6):
#   15 % sorteados -> 12,5 % errados   (sinal mais fraco: teste de sensibilidade)
#   30 % sorteados -> 25,0 % errados   (ja rodado)
#   50 % sorteados -> 41,7 % errados   (sinal mais forte: teste de saturacao)
for _s in (42, 1, 7):
    for _f, _tag in ((0.15, "15"), (0.50, "50")):
        EXPERIMENTS.append(Experiment(
            run_id=f"f6_ruido{_tag}_seed{_s}", model_name="resnet18", image_size=64,
            seed=_s, epochs=60, label_noise=_f, noise_seed=_s,
            augment=False, weight_decay=0.0, eval_train=True))

# ===========================================================================
# Outra arquitetura: DenseNet-121 (ultima generalizacao que faltava)
# ===========================================================================
# ATENCAO AO PREFIXO: estes runs NAO comecam com "f6_" de proposito. Os scripts de
# analise varrem `monai_weights/f6_*`, e misturar duas arquiteturas na mesma tabela
# agregada daria medias sem sentido. O prefixo `dn121_` os mantem invisiveis para
# aquelas varreduras; a comparacao entre arquiteturas e feita por script proprio.
#
# A camada densa muda de 6x512 (ResNet-18) para 6x1024 (DenseNet-121) -- o dobro de
# pesos, e uma matriz ainda mais assimetrica. Se o indicador so funcionasse na forma
# especifica da ResNet, e aqui que isso apareceria.
#
# Mesmas sementes e mesmo nivel de ruido (30 %) dos runs de referencia, para que a
# arquitetura seja a unica diferenca. ~31 s/epoca contra ~10 s da ResNet-18.
for _s in (42, 1, 7):
    EXPERIMENTS.append(Experiment(
        run_id=f"dn121_ruido30_seed{_s}", model_name="densenet121", image_size=64,
        seed=_s, epochs=60, label_noise=0.30, noise_seed=_s,
        augment=False, weight_decay=0.0, eval_train=True))
    EXPERIMENTS.append(Experiment(
        run_id=f"dn121_seed{_s}", model_name="densenet121", image_size=64,
        seed=_s, epochs=40))


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
