# legacy/

Scripts da fase MedMNIST/MONAI, anteriores ao pipeline atual. Mantidos aqui
para referência histórica; **não fazem parte do fluxo de trabalho corrente** e
podem depender de dados que já não existem no repositório.

| Arquivo | Observação |
|---------|------------|
| `medmnist_weights_extraction.py` | extração de pesos da fase MedMNIST |
| `data_analysis_summary.py` | análise agregada da mesma fase |
| `medmnist-jup.ipynb` | notebook exploratório |
| `monai_wieghts_plot.py` | lia a pasta `monai_wieghts/`, **removida em 2026-09-24** |

O pipeline atual é `src/main_train.py` + `src/training/trainer.py`, com as
medidas em `src/complexity.py` e as análises em `scripts/`.
