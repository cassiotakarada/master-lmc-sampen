"""
Monta o pacote para o orientador: pesos da camada densa em CSV + todos os graficos + tabelas.

Os checkpoints completos somam ~200 GB; o escopo do trabalho e so a camada densa, entao
o pacote leva apenas ela (fc.weight na ResNet-18, classifier.weight na DenseNet-121),
epoca a epoca, em CSV.

Formato dos pesos: uma linha por epoca, uma coluna por peso, na ordem por neuronio
(n1_x1, n1_x2, ..., n1_x512, n2_x1, ...) -- o "caminho" entrada x_i -> neuronio n_j.
O bias nao entra.

Estrutura gerada:
  <saida>/
    LEIA-ME.md
    pesos/
      todos_os_pesos_resnet18.csv      regime, semente, epoca, n1_x1 ... (todos os treinos)
      todos_os_pesos_densenet121.csv
      <arquitetura>/<regime>/semente_XX/
        pesos_camada_densa.csv         epoca, n1_x1 ...
        metricas_por_epoca.csv         LMC, SampEn, SampEn2D, perdas e acuracias
        config.json, test_metrics.json
    graficos/
      comparativos/                    trajetorias medias por regime
      por_treino/<arquitetura>/<regime>/semente_XX.png
      analises_anteriores/<pasta>/     figuras das fases F6/F7 (enquadramento antigo)
    tabelas/                           CSVs da analise descritiva
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

# figuras das fases anteriores (mesmos runs; marcam o "sinal" do detector, hoje fora do escopo)
ANALISES_ANTERIORES = ["f6_analise", "f6_analise_n8", "f6_analise_dn121", "f7_pareado_resnet",
                       "f7_pareado_dn121", "sensibilidade"]


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


def colunas_pesos(n_neur: int, n_ent: int) -> list[str]:
    return [f"n{j}_x{i}" for j in range(1, n_neur + 1) for i in range(1, n_ent + 1)]


def processar_run(args) -> str:
    run_dir, destino, figura = args
    os.makedirs(destino, exist_ok=True)
    cks = sorted(glob.glob(os.path.join(run_dir, "epoch_*.pth")),
                 key=lambda p: int(re.search(r"epoch_(\d+)", p).group(1)))
    if cks:
        inicial = os.path.join(run_dir, "initial_weights.pth")
        if os.path.isfile(inicial):
            cks = [inicial] + cks
        epocas = [0 if p.endswith("initial_weights.pth") else int(re.search(r"epoch_(\d+)", p).group(1))
                  for p in cks]
        pesos = np.stack([peso_denso(p) for p in cks])
    else:
        # runs de passos igualados: sem checkpoint por epoca (ver subamostragem_passos_igualados.sh).
        # bef = epoca de menor val_loss (a avaliada no teste); aft = primeira subida do val_loss.
        teste = json.load(open(os.path.join(run_dir, "test_metrics.json")))
        log = open(os.path.join(run_dir, "run.log"), encoding="utf-8").read()
        aft = re.search(r"Saved aft_weights at epoch (\d+)", log)
        momentos = [("initial_weights.pth", 0), ("bef_weights.pth", int(teste["epoca_avaliada"])),
                    ("aft_weights.pth", int(aft.group(1)) if aft else -1)]
        momentos = sorted((m for m in momentos if os.path.isfile(os.path.join(run_dir, m[0]))),
                          key=lambda m: m[1])
        pesos = np.stack([peso_denso(os.path.join(run_dir, f)) for f, _ in momentos])
        epocas = [e for _, e in momentos]

    n, n_neur, n_ent = pesos.shape
    tab = pd.DataFrame(pesos.reshape(n, -1), columns=colunas_pesos(n_neur, n_ent))
    tab.insert(0, "epoca", epocas)
    tab.to_csv(os.path.join(destino, "pesos_camada_densa.csv"), index=False, float_format="%.7g")

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
    """Seis paineis, um eixo y cada: perda, acuracia, LMC, entropia, SampEn 1D, SampEn2D."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURFACE, TEXT_1, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    paineis = [
        ("Perda", [("train_loss", "treino", "#e34948"), ("val_loss", "validação", "#0b0b0b")]),
        ("Acurácia", [("train_eval_accuracy", "treino (eval)", "#e34948"),
                      ("val_accuracy", "validação", "#0b0b0b")]),
        ("LMC", [("lmc", "LMC", "#2a78d6")]),
        ("Entropia de Shannon do histograma (bits)", [("entropia", "H", "#4a3aa7")]),
        ("SampEn 1D", [("sampen", "ordem por neurônio", "#eb6834"),
                       ("sampen_x_major", "ordem por entrada", "#eda100")]),
        ("SampEn2D", [("sampen2d", "SampEn2D", "#1baf7a")]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor=SURFACE)
    x = df["epoch"]
    for ax, (titulo, series) in zip(axes.ravel(), paineis):
        for col, rot, cor in series:
            if col in df:
                ax.plot(x, df[col], color=cor, linewidth=1.6, label=rot)
        if titulo == "Perda":
            ax.set_yscale("log")
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
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def juntar_pesos(saida: str, arq: str, itens: list[tuple[str, int, str]]) -> None:
    """Um CSV com todos os treinos da arquitetura: regime, semente, epoca, pesos."""
    destino = os.path.join(saida, "pesos", f"todos_os_pesos_{arq}.csv")
    primeiro = True
    for regime, semente, pasta in itens:
        df = pd.read_csv(os.path.join(pasta, "pesos_camada_densa.csv"))
        df.insert(0, "semente", semente)
        df.insert(0, "regime", regime)
        df.to_csv(destino, mode="w" if primeiro else "a", header=primeiro, index=False,
                  float_format="%.7g")
        primeiro = False


LEIA_ME = """# Pacote para avaliação — LMC e SampEn na camada densa

Gerado em {data} por `scripts/montar_pacote_orientador.py` (repositório master-lmc-sampen).
Só a camada densa entra no pacote (é o objeto da pesquisa): `fc.weight` na ResNet-18
(6 neurônios × 512 entradas = 3.072 pesos) e `classifier.weight` na DenseNet-121
(6 × 1.024 = 6.144 pesos). O bias não está incluído.

## Pesos (`pesos/`)

- **`todos_os_pesos_resnet18.csv`** e **`todos_os_pesos_densenet121.csv`**: todos os treinos
  num arquivo só. Uma linha por época de cada treino. Colunas: `regime`, `semente`, `epoca` e
  um peso por coluna.
- **`<arquitetura>/<regime>/semente_XX/pesos_camada_densa.csv`**: os mesmos pesos, separados
  por treino. Colunas: `epoca` e um peso por coluna.

Nome das colunas de peso: `nJ_xI` é o peso do caminho **entrada I → neurônio J**. A ordem é
por neurônio: `n1_x1, n1_x2, …, n1_x512, n2_x1, …` (a mesma ordem usada no cálculo da SampEn 1D).
`epoca = 0` é o peso inicial, antes do treino, quando existe.

Nos treinos `subamostragem*_passos_iguais` não foram salvos pesos de todas as épocas (seriam
~120 GB por treino). O CSV deles tem 3 linhas: a época 0 (inicial), a primeira época em que a
perda de validação subiu e a época de menor perda de validação. A série completa de LMC e
SampEn desses treinos está em `metricas_por_epoca.csv`, calculada durante o treino.

Na mesma pasta de cada treino:
- `metricas_por_epoca.csv`: LMC, entropia, desequilíbrio, SampEn 1D (ordem por neurônio e por
  entrada), SampEn2D, perdas e acurácias, época a época.
- `config.json`: todos os hiperparâmetros. `test_metrics.json`: desempenho no conjunto de teste.

## Gráficos (`graficos/`)

| Pasta | O que mostra |
|---|---|
| `por_treino/<arquitetura>/<regime>/semente_XX.png` | Um gráfico por treino com 6 painéis: perda, acurácia, LMC, entropia, SampEn 1D e SampEn2D por época |
| `comparativos/` | Média ± desvio entre sementes: trajetórias de cada regime por época e por passos (`trajetoria_*`), níveis de ruído 15/30/50% (`niveis_de_ruido`), DenseNet-121 controle × ruído (`densenet121_controle_vs_ruido`) e sensibilidade aos parâmetros das medidas (`sensibilidade_parametros`) |
| `analises_anteriores/` | Figuras das fases anteriores (F6/F7) com os mesmos treinos. Elas marcam o "sinal" de um detector de overfitting, abordagem que saiu do escopo; servem como registro |

## Tabelas (`tabelas/`)

| Arquivo | O que é |
|---|---|
| `comportamento_por_run.csv` | Uma linha por treino: valor inicial, final, amplitude relativa, época do pico e do vale de LMC, SampEn 1D e SampEn2D, além das acurácias finais |
| `comparacoes_vs_controle.csv` | Cada regime comparado ao controle (teste de Mann-Whitney): média, desvio, razão e valor-p do valor final e da amplitude de cada medida |
| `trajetorias_medias.csv` | As curvas médias (e desvio) de cada regime, por época e por passos, usadas nos gráficos comparativos |
| `sensibilidade_resumo.csv` | Para cada valor de parâmetro (bins da LMC; r da SampEn 1D e 2D): médias de controle e ruído 30%, razão e valor-p |
| `sensibilidade_por_epoca.csv` | Os valores época a época que geram o gráfico de sensibilidade |

Nas comparações, cada regime da DenseNet-121 é comparado com o controle da própria DenseNet-121 (coluna `controle`).

Amplitude relativa = (máximo − mínimo) / |média| da trajetória da medida.

## Regimes

{tabela}

Configuração comum: MedNIST (partição fixa 47.164 / 5.895 / 5.895), imagens 64×64, AdamW
(lr = 0,001, weight decay = 0), lote 128, sem aumento de dados. LMC com 100 bins; SampEn 1D
com m = 2 e r = 0,2·DP; SampEn2D com m = 1.
"""


def main() -> int:
    p = argparse.ArgumentParser(description="Monta o pacote do orientador")
    p.add_argument("--root", default="monai_weights")
    p.add_argument("--analise", nargs="+",
                   default=["monai_weights/f8_comportamento", "monai_weights/f8_sensibilidade"])
    p.add_argument("--saida", required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()

    if os.path.exists(a.saida):
        raise SystemExit(f"{a.saida} ja existe; apague ou escolha outro nome")
    tarefas, linhas, por_arq = [], [], {}
    for arq, regime, padrao, desc in GRUPOS:
        runs = sorted(d for d in glob.glob(os.path.join(a.root, padrao)) if os.path.isdir(d))
        sementes = []
        for d in runs:
            semente = json.load(open(os.path.join(d, "config.json")))["seed"]
            sementes.append(semente)
            pasta = os.path.join(a.saida, "pesos", arq, regime, f"semente_{semente:02d}")
            tarefas.append((d, pasta, os.path.join(a.saida, "graficos", "por_treino", arq, regime,
                                                   f"semente_{semente:02d}.png")))
            por_arq.setdefault(arq, []).append((regime, semente, pasta))
        linhas.append(f"| {arq} | `{regime}` | {len(runs)} | {', '.join(map(str, sorted(sementes)))} | {desc} |")

    with ProcessPoolExecutor(a.workers) as ex:
        for msg in ex.map(processar_run, tarefas):
            print(msg, flush=True)
    for arq, itens in por_arq.items():
        juntar_pesos(a.saida, arq, itens)

    comp = os.path.join(a.saida, "graficos", "comparativos")
    tab = os.path.join(a.saida, "tabelas")
    os.makedirs(comp, exist_ok=True)
    os.makedirs(tab, exist_ok=True)
    for pasta in a.analise:
        for f in glob.glob(os.path.join(pasta, "*.png")):
            shutil.copy2(f, comp)
        for f in glob.glob(os.path.join(pasta, "*.csv")):
            shutil.copy2(f, tab)
    for nome in ANALISES_ANTERIORES:
        figs = glob.glob(os.path.join(a.root, nome, "*.png"))
        if figs:
            dest = os.path.join(a.saida, "graficos", "analises_anteriores", nome)
            os.makedirs(dest, exist_ok=True)
            for f in figs:
                shutil.copy2(f, dest)

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
