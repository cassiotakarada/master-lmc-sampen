"""
Monta o pacote para o orientador: pesos da camada densa + graficos + tabelas, em pastas.

Os checkpoints completos somam ~200 GB; o escopo do trabalho e so a camada densa, entao
o pacote leva apenas ela (fc.weight na ResNet-18, classifier.weight na DenseNet-121),
epoca a epoca, num .npz por treino -- poucos MB no total.

Estrutura gerada:
  <saida>/
    LEIA-ME.md
    graficos/  comparativos/  (figuras da analise descritiva)
               por_treino/<arquitetura>/<regime>/semente_XX.png
    tabelas/   (CSVs da analise descritiva)
    pesos/<arquitetura>/<regime>/semente_XX/
        camada_densa_por_epoca.npz   pesos [n_epocas, n_classes, n_entradas] + epocas
        metricas_por_epoca.csv       LMC, SampEn, SampEn2D, perdas e acuracias
        config.json, test_metrics.json
  <saida>.zip
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

# (arquitetura, regime na pasta, padrao dos runs, descricao)
GRUPOS = [
    ("resnet18", "controle",                      "f6_ctrlpar_seed*",  "100% dos dados, rotulos corretos, 60 epocas"),
    ("resnet18", "ruido15",                       "f6_ruido15_seed*",  "15% dos rotulos de treino sorteados, 60 epocas"),
    ("resnet18", "ruido30",                       "f6_ruido30*",       "30% dos rotulos de treino sorteados, 60 epocas"),
    ("resnet18", "ruido50",                       "f6_ruido50_seed*",  "50% dos rotulos de treino sorteados, 60 epocas"),
    ("resnet18", "subamostragem10_curta",         "f6_frac10_seed*",   "10% dos dados, 60 epocas (2.220 passos)"),
    ("resnet18", "subamostragem02_curta",         "f6_frac02_seed*",   "2% dos dados, 200 epocas (1.600 passos)"),
    ("resnet18", "subamostragem01_curta",         "f6_frac01_seed*",   "1% dos dados, 200 epocas (800 passos)"),
    ("resnet18", "subamostragem10_passos_iguais", "f6_frac10eq_seed*", "10% dos dados, 599 epocas (~22.140 passos)"),
    ("resnet18", "subamostragem02_passos_iguais", "f6_frac02eq_seed*", "2% dos dados, 2.768 epocas (~22.140 passos)"),
    ("resnet18", "subamostragem01_passos_iguais", "f6_frac01eq_seed*", "1% dos dados, 5.535 epocas (~22.140 passos)"),
    ("densenet121", "controle",                   "dn121_ctrlpar_seed*", "100% dos dados, rotulos corretos, 60 epocas"),
    ("densenet121", "ruido30",                    "dn121_ruido30_seed*", "30% dos rotulos de treino sorteados, 60 epocas"),
]


def peso_denso(caminho: str) -> np.ndarray:
    """A unica matriz 2D com sufixo .weight do state_dict e a camada densa."""
    import torch
    sd = torch.load(caminho, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    densas = [v for k, v in sd.items() if torch.is_tensor(v) and v.ndim == 2 and k.endswith(".weight")]
    if len(densas) != 1:
        raise RuntimeError(f"esperava 1 camada densa em {caminho}, achei {len(densas)}")
    return densas[0].float().numpy()


def processar_run(args) -> str:
    run_dir, destino, figura = args
    os.makedirs(destino, exist_ok=True)
    cks = sorted(glob.glob(os.path.join(run_dir, "epoch_*.pth")),
                 key=lambda p: int(re.search(r"epoch_(\d+)", p).group(1)))
    if cks:
        epocas = np.array([int(re.search(r"epoch_(\d+)", p).group(1)) for p in cks])
        pesos = np.stack([peso_denso(p) for p in cks])
        obs = "um registro por epoca"
    else:
        # runs de passos igualados: sem checkpoint por epoca (ver subamostragem_passos_igualados.sh)
        # bef = pesos da epoca de menor val_loss (a mesma avaliada no teste);
        # aft = primeira subida do val_loss depois do melhor ate entao (costuma ser cedo)
        teste = json.load(open(os.path.join(run_dir, "test_metrics.json")))
        log = open(os.path.join(run_dir, "run.log"), encoding="utf-8").read()
        aft = re.search(r"Saved aft_weights at epoch (\d+)", log)
        momentos = [("initial_weights.pth", 0), ("bef_weights.pth", int(teste["epoca_avaliada"])),
                    ("aft_weights.pth", int(aft.group(1)) if aft else -1)]
        momentos = sorted((m for m in momentos if os.path.isfile(os.path.join(run_dir, m[0]))),
                          key=lambda m: m[1])
        pesos = np.stack([peso_denso(os.path.join(run_dir, f)) for f, _ in momentos])
        epocas = np.array([e for _, e in momentos])
        obs = ("so 3 momentos: inicial (epoca 0), primeira subida do val_loss (aft) e epoca de "
               "menor val_loss (bef); a serie por epoca de LMC/SampEn esta no CSV")
    np.savez_compressed(os.path.join(destino, "camada_densa_por_epoca.npz"),
                        pesos=pesos, epocas=epocas, observacao=np.array(obs))

    live = pd.read_csv(os.path.join(run_dir, "complexity_live.csv"))
    hist = pd.read_csv(os.path.join(run_dir, "history.csv"))
    live = live.drop(columns=[c for c in ("val_loss", "val_accuracy", "status", "d_lmc", "d_sampen",
                                          "incl_lmc_rel", "overhead_s") if c in live.columns])
    cols = [c for c in ("epoch", "train_loss", "val_loss", "val_accuracy", "train_eval_loss",
                        "train_eval_accuracy") if c in hist.columns]
    met = live.merge(hist[cols], on="epoch", how="left")
    met.to_csv(os.path.join(destino, "metricas_por_epoca.csv"), index=False)
    for f in ("config.json", "test_metrics.json"):
        if os.path.isfile(os.path.join(run_dir, f)):
            shutil.copy2(os.path.join(run_dir, f), destino)

    figura_run(met, os.path.basename(run_dir), figura)
    return f"{run_dir} -> {pesos.shape}"


def figura_run(df: pd.DataFrame, nome: str, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURFACE, TEXT_1, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=SURFACE)
    x = df["epoch"]
    paineis = [("Acurácia", [("val_accuracy", "validação", "#0b0b0b"),
                             ("train_eval_accuracy", "treino (eval)", "#e34948")]),
               ("LMC", [("lmc", "LMC", "#2a78d6")]),
               ("SampEn2D", [("sampen2d", "SampEn2D", "#1baf7a")])]
    for ax, (titulo, series) in zip(axes, paineis):
        for col, rot, cor in series:
            if col in df:
                ax.plot(x, df[col], color=cor, linewidth=1.8, label=rot)
        ax.set_facecolor(SURFACE)
        ax.set_title(titulo, color=TEXT_1, fontsize=11, loc="left")
        ax.set_xlabel("Época", color=TEXT_2, fontsize=9)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(colors=TEXT_2, labelsize=8.5)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        if len(series) > 1:
            ax.legend(frameon=False, fontsize=8.5, labelcolor=TEXT_2)
    fig.suptitle(nome, color=TEXT_1, fontsize=12, x=0.006, ha="left", weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    plt.close(fig)


LEIA_ME = """# Pacote para avaliação — LMC e SampEn na camada densa

Gerado em {data} por `scripts/montar_pacote_orientador.py` (repositório master-lmc-sampen).

## O que tem aqui

| Pasta | Conteúdo |
|---|---|
| `graficos/comparativos/` | Figuras da análise descritiva: trajetórias médias de LMC, SampEn 1D e SampEn2D por regime, por época e por passos de otimização |
| `graficos/por_treino/` | Um gráfico por treino (acurácia, LMC e SampEn2D por época), organizado como `pesos/` |
| `tabelas/` | `comportamento_por_run.csv` (resumo de cada treino), `comparacoes_vs_controle.csv` (Mann-Whitney de cada regime contra o controle), `trajetorias_medias.csv` |
| `pesos/<arquitetura>/<regime>/semente_XX/` | Pesos da camada densa e métricas de cada treino |

## Dentro de cada `semente_XX/`

- `camada_densa_por_epoca.npz`: abra com `np.load(...)`. Chaves `pesos` (forma `[n_registros, 6, n_entradas]`; 512 entradas na ResNet-18, 1024 na DenseNet-121), `epocas` e `observacao`. O bias não está incluído.
- `metricas_por_epoca.csv`: LMC, entropia, desequilíbrio, SampEn 1D (ordem por neurônio e por entrada), SampEn2D, perdas e acurácias, época a época.
- `config.json`: todos os hiperparâmetros do treino. `test_metrics.json`: desempenho no conjunto de teste, avaliado uma única vez.

Nos treinos `subamostragem*_passos_iguais` não foram salvos checkpoints por época (seriam ~120 GB por treino). O `.npz` deles traz 3 momentos (inicial, primeira subida da perda de validação e época de menor perda de validação, em ordem de época); a série completa de LMC e SampEn por época está no CSV, calculada durante o treino.

## Regimes

{tabela}

Configuração comum: MedNIST (partição fixa 47.164 / 5.895 / 5.895), imagens 64×64, AdamW (lr = 0,001, weight decay = 0), lote 128, sem aumento de dados, sem weight decay. LMC com 100 bins; SampEn 1D com m = 2 e r = 0,2·DP; SampEn2D com m = 1. Achatamento da camada densa por neurônio (`n1x1, n1x2, …`).

Exemplo em Python:

```python
import numpy as np, pandas as pd
d = np.load("pesos/resnet18/ruido30/semente_42/camada_densa_por_epoca.npz")
W = d["pesos"]        # [60, 6, 512]
m = pd.read_csv("pesos/resnet18/ruido30/semente_42/metricas_por_epoca.csv")
```
"""


def main() -> int:
    p = argparse.ArgumentParser(description="Monta o pacote do orientador")
    p.add_argument("--root", default="monai_weights")
    p.add_argument("--analise", default="monai_weights/f8_comportamento")
    p.add_argument("--saida", required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()

    if os.path.exists(a.saida):
        raise SystemExit(f"{a.saida} ja existe; apague ou escolha outro nome")
    tarefas, linhas = [], []
    for arq, regime, padrao, desc in GRUPOS:
        runs = sorted(d for d in glob.glob(os.path.join(a.root, padrao)) if os.path.isdir(d))
        sementes = []
        for d in runs:
            semente = json.load(open(os.path.join(d, "config.json")))["seed"]
            sementes.append(semente)
            tarefas.append((d, os.path.join(a.saida, "pesos", arq, regime, f"semente_{semente:02d}"),
                            os.path.join(a.saida, "graficos", "por_treino", arq, regime,
                                         f"semente_{semente:02d}.png")))
        linhas.append(f"| {arq} | `{regime}` | {len(runs)} | {', '.join(map(str, sorted(sementes)))} | {desc} |")

    with ProcessPoolExecutor(a.workers) as ex:
        for msg in ex.map(processar_run, tarefas):
            print(msg, flush=True)

    comp = os.path.join(a.saida, "graficos", "comparativos")
    tab = os.path.join(a.saida, "tabelas")
    os.makedirs(comp, exist_ok=True)
    os.makedirs(tab, exist_ok=True)
    for f in glob.glob(os.path.join(a.analise, "*.png")):
        shutil.copy2(f, comp)
    for f in glob.glob(os.path.join(a.analise, "*.csv")):
        shutil.copy2(f, tab)

    tabela = ("| Arquitetura | Pasta | Treinos | Sementes | Descrição |\n|---|---|---|---|---|\n"
              + "\n".join(linhas))
    with open(os.path.join(a.saida, "LEIA-ME.md"), "w", encoding="utf-8") as f:
        f.write(LEIA_ME.format(data=pd.Timestamp.now().strftime("%d/%m/%Y"), tabela=tabela))

    zip_base = a.saida.rstrip("/")
    shutil.make_archive(zip_base, "zip", root_dir=os.path.dirname(zip_base),
                        base_dir=os.path.basename(zip_base))
    print(f"\nPasta: {a.saida}\nZip:   {zip_base}.zip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
