# thesis-complexity-nn

Experimento central da dissertação de mestrado — FFCLRP-USP, Programa de Pós-Graduação em Computação Aplicada.

**Hipótese:** A complexidade dos pesos de uma rede neural profunda evolui de forma sistemática ao longo do treinamento e constitui um marcador objetivo de overfitting.

**Medidas:** LMC Complexity (por camada, por checkpoint) + Sample Entropy (trajetória de normas L2 por filtro/neurônio).

## Setup rápido

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke test (~ 5 min na GPU)

```bash
python scripts/diagnostic.py
```

## Rodar grid completo

```bash
python -m src.orchestrator --config config/experiment_grid.yaml
```

## Compilar tese

```bash
bash scripts/build_thesis.sh
```

## Estrutura

```
thesis-complexity-nn/
├── src/
│   ├── complexity.py     # LMC + SampEn
│   ├── weights.py        # agrupamento de camadas + WeightSnapshot
│   ├── training.py       # loop de treinamento + checkpointing
│   ├── orchestrator.py   # runner YAML-driven + resumable
│   └── writer.py         # thesis writer via claude -p
├── scripts/
│   ├── diagnostic.py     # smoke test
│   └── build_thesis.sh   # compila PDF
├── config/
│   └── experiment_grid.yaml
├── results/
│   ├── checkpoints/      # temporário — deletado após extração de métricas
│   ├── metrics/          # métricas LMC/SampEn por run (permanente)
│   └── results.csv
├── tese/                 # dissertação LaTeX (abnTeX2)
└── tests/                # unit tests
```
