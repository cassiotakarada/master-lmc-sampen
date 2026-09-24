# 🧠 MedMNIST Neural Network Complexity Analysis

This project analyzes the complexity of convolutional neural networks (CNNs) trained on [MedMNIST](https://medmnist.com/) datasets using entropy-based metrics. It evaluates models using both **Sample Entropy** and **LMC Complexity**, exports **TensorBoard** scalars, and supports inference-based neuron analysis.

---

## 📦 Project Setup

### 1. ✅ Prerequisites

- Python 3.8+
- pip

### 2. 🔧 Create a virtual environment

```bash
python -m venv myenv
source myenv/bin/activate      # Linux/macOS
myenv\Scripts\activate         # Windows
```

### 3. 📦 Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Use

Run everything using the interactive tool:

```bash
python run_tool.py
```

You will see a menu:

```
1. Extract model weights after training
2. Export TensorBoard scalars
3. Analyze inference outputs
4. Analyze weights complexity (LMC & Sample Entropy)
```

### 1️⃣ Extract Model Weights (Training)

```bash
# inside tool or manually
python medmnist_weights_extraction.py
```

- Prompts for number of epochs
- Automatically creates a folder like `results/1_3` containing:
  - CNN/ResNet model weights (`*.pth`)
  - Inference results (`*_inference_outputs.pt`, `*_inference_mean.json`)
  - Summary of run (`summary.csv`)

TensorBoard logs are saved in:

- `runs/pathmnist_28_1_3/`, `runs/organmnist3d_64_1_3/`, etc.

### 2️⃣ Export TensorBoard Scalars

```bash
python export_tensorboard_scalars.py
```

- Prompts for folder like `1_3`
- Extracts scalars and saves as:
  - `results/1_3/tensorboard_scalars.csv`

### 3️⃣ Analyze Inference Outputs

```bash
python analyze_inference_outputs.py
```

- Prompts for folder like `1_3`
- Outputs:
  - `inference_summary.csv` with mean activations
  - Mean barplot: `*_mean_plot.png`
  - Optional neuron heatmaps: `*_heatmap.png`

### 4️⃣ Analyze Weight Complexity

```bash
python data_analysis_summary.py
```

- Prompts for folder like `1_3`
- Computes entropy metrics:
  - `weights_entropy_results.csv`
  - Plots: LMC barplot, SampEn barplot, LMC by layer type

---

## 📁 Folder Structure

```
project/
├── results/
│   └── 1_3/
│       ├── *.pth (model weights)
│       ├── summary.csv
│       ├── tensorboard_scalars.csv
│       ├── inference_summary.csv
│       ├── *_inference_mean.json
│       ├── *_inference_outputs.pt
│       ├── *_mean_plot.png / *_heatmap.png
│       └── weights_entropy_results.csv
│
├── runs/
│   └── pathmnist_28_1_3/ (TensorBoard event files)
│   └── organmnist3d_64_1_3/ (etc.)
│
├── run_tool.py
├── medmnist_weights_extraction.py
├── export_tensorboard_scalars.py
├── analyze_inference_outputs.py
├── data_analysis_summary.py
└── requirements.txt
```

---

## 📊 Metrics Explained

| Metric             | Description                                       |
| ------------------ | ------------------------------------------------- |
| **Sample Entropy** | Measures unpredictability/irregularity of weights |
| **LMC Complexity** | Product of Shannon entropy and disequilibrium     |
| **Mean Inference** | Average neuron activation after running inference |

---

## 🧩 Notes

- Works with both **2D (PathMNIST)** and **3D (OrganMNIST3D)** datasets
- Supports both ResNet and simple CNN architectures
- All outputs are saved under `results/{run_id}_{epochs}/`
- Use `tensorboard --logdir runs` to visualize training logs

### MedNIST quickstart

```bash
./myenv/bin/python -m src.main_train --epochs 40 --run_id meu_run --model_name resnet18 --image_size 64
```

> ⚠️ Use `./myenv/bin/python -m pip`, **nunca** `./myenv/bin/pip`: os scripts wrapper de
> `myenv/bin/` (pip, pytest, torchrun…) têm gravado o caminho de outro diretório do projeto
> e executam no ambiente errado. O mesmo vale para `pytest` — use `python -m pytest`.

## 🔧 Stack: PyTorch puro (sem MONAI)

Desde 2026-09-21 (fase F3b) o projeto **não usa mais MONAI**. Tudo é PyTorch/torchvision:

| antes | agora |
|---|---|
| `monai.apps.MedNISTDataset` | `src/data/mednist.py` + partição congelada |
| `monai.networks.nets` | `torchvision.models` |
| `monai.transforms` | `torchvision.transforms` |
| `monai.data.DataLoader` | `torch.utils.data.DataLoader` |

A **partição está congelada** em `data/mednist_split.json` — a lista exata de arquivos de
cada conjunto, gerada uma única vez a partir do MONAI antes da remoção
(`scripts/congelar_particao.py`). Ela deixou de depender da implementação interna de uma
biblioteca e virou um dado versionado: qualquer pessoa reproduz a mesma divisão.

As imagens em escala de cinza são replicadas para 3 canais, porque as redes do torchvision
esperam RGB. A camada densa analisada continua sendo `fc` com forma **[6, 512] = 3.072 pesos**,
idêntica à de antes.

> Checkpoints anteriores a F3b foram treinados com a ResNet-18 do MONAI, que **não** é
> numericamente idêntica à do torchvision. Runs novos formam uma nova linha de base.

---

## 🔬 Particionamento dos dados (treino / validação / teste)

O projeto usa as **três seções oficiais do MedNIST**, sem recortes manuais:

| Conjunto | n | Para que serve | Regra |
|----------|---|----------------|-------|
| treino | 47.164 (80 %) | ajustar os pesos | único que entra no `backward()` |
| validação | 5.895 (10 %) | escolher a época e detectar overfitting | **nunca** entra no gradiente |
| teste | 5.895 (10 %) | medir o desempenho final | **olhado uma única vez**, no fim |

O teste é avaliado por `Trainer.evaluate_test()`, depois que o treino termina, recarregando os
pesos da época de menor `val_loss` — nunca os da última época. O resultado vai para
`test_metrics.json`, com a época avaliada registrada.

### Duas sementes diferentes, de propósito

| Flag | O que controla | Varia entre runs? |
|------|----------------|-------------------|
| `--data_seed` (default 0) | **quem** cai em treino/validação/teste | **Não** — fixa, senão os runs não são comparáveis e uma amostra trocaria de conjunto entre runs |
| `--seed` (default 42) | inicialização dos pesos e ordem dos lotes | **Sim** — é o ponto de repetir o experimento |

### Flags úteis

```bash
--data_seed 0          # semente da partição (mantenha fixa)
--val_frac 0.1 --test_frac 0.1
--train_fraction 1.0   # fração do treino usada; <1.0 subamostra de forma estratificada
```

> ⚠️ **Cuidado com `--train_fraction` pequeno.** Com poucos passos de treino, as estatísticas
> móveis do BatchNorm não convergem e o modelo desaba em `eval()` — inclusive sobre o próprio
> conjunto de treino. Isso **imita overfitting de forma convincente** sem ser overfitting. O
> código emite `WARNING` abaixo de 200 passos. Para induzir overfitting de verdade, prefira
> **ruído de rótulo**, que mantém o dataset inteiro.

## 🧮 Complexidade: uma única implementação

Todas as medidas (LMC, SampEn 1D/2D, MSE) vivem em **`src/complexity.py`**. Nenhum outro
arquivo deve redefini-las. Pacote autocontido para compartilhar:
`deliverables/complexity_reference/` (gerado por `python deliverables/build_deliverable.py`).

Convenções fixadas: `n_bins=100`, `m=2`, `r=0.20·σ`, ordem de achatamento `n_major`
(`n1x1, n1x2, …`). Ver `plan.md` §2 para a justificativa.

---

🙋‍♂️ Feel free to fork, contribute or suggest ideas (like comparing architectures, new entropy metrics, or UI dashboards).
# master-lmc-sampen
