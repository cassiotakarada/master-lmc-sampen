#!/usr/bin/env bash
# Subamostragem com PASSOS IGUALADOS ao controle (f6_ctrlpar: 60 epocas x 369 passos = 22.140).
# Mesma configuracao dos f6_frac*: sem aumento de dados, sem weight decay, eval_train ligado.
# Sem checkpoint por epoca (SAVE_EVERY_EPOCH=0): 2.768 epocas x 45 MB dariam ~120 GB por run,
# e LMC/SampEn ja sao gravadas ao vivo em complexity_live.csv.
#
#   fracao  n_treino  passos/epoca  epocas
#   10 %    4.717     37            599
#    2 %      944      8            2768
#    1 %      471      4            5535
#
# Uso: scripts/subamostragem_passos_igualados.sh <gpu> <fracao:epocas> [<fracao:epocas> ...]
set -u
cd "$(dirname "$0")/.."
PY="${PY:-/home/users/u7594034/Área de trabalho/Masters/venv/bin/python}"
GPU="$1"; shift
SEEDS="42 1 7 13 23"
for item in "$@"; do
  frac="${item%%:*}"; ep="${item##*:}"
  tag=$(printf "%02d" "$("$PY" -c "print(round($frac*100))")")
  for s in $SEEDS; do
    rid="f6_frac${tag}eq_seed${s}"
    if [ -f "monai_weights/${rid}/test_metrics.json" ]; then echo "[pula] $rid ja concluido"; continue; fi
    echo "[$(date '+%F %T')] GPU $GPU -> $rid ($ep epocas)"
    CUDA_VISIBLE_DEVICES="$GPU" SAVE_EVERY_EPOCH=0 "$PY" -m src.main_train \
      --results_dir monai_weights --run_id "$rid" --model_name resnet18 --image_size 64 \
      --seed "$s" --epochs "$ep" --batch_size 128 --lr 0.001 --weight_decay 0.0 \
      --train_fraction "$frac" --num_workers 4 --no_augment --eval_train \
      > "monai_weights/${rid}_train.log" 2>&1 || echo "[ERRO] $rid falhou (ver monai_weights/${rid}_train.log)"
  done
done
echo "[$(date '+%F %T')] fila da GPU $GPU terminou"
