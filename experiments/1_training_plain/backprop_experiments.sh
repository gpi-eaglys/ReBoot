#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# import var: 'EXP_DATA_DIR'
source ${SCRIPT_DIR}/../0_common/paths.sh

pushd "$SCRIPT_DIR" > /dev/null

# MLP-1
python3 backprop_plain.py MNIST           --subsample 10000 --resize 14 --num-layers 2 --num-hidden 32 --lr 5e-3 --weight-decay 1e-3 --num-runs 10 --data-path "${EXP_DATA_DIR}"
python3 backprop_plain.py FASHION_MNIST   --subsample 10000 --resize 14 --num-layers 2 --num-hidden 32 --lr 5e-3 --weight-decay 1e-3 --num-runs 10 --data-path "${EXP_DATA_DIR}"
python3 backprop_plain.py KUZUSHIJI_MNIST --subsample 10000 --resize 14 --num-layers 2 --num-hidden 32 --lr 5e-3 --weight-decay 1e-3 --num-runs 10 --data-path "${EXP_DATA_DIR}"
python3 backprop_plain.py LETTER_RECOGNITION                            --num-layers 2 --num-hidden 32 --lr 5e-3 --weight-decay 1e-3 --num-runs 10 --data-path "${EXP_DATA_DIR}"
python3 backprop_plain.py BREAST_CANCER     --weight-decay 0.0          --num-layers 2 --num-hidden 32 --batch-size 16 --lr 5e-3     --num-runs 10 --data-path "${EXP_DATA_DIR}"
python3 backprop_plain.py HEART_DISEASE     --weight-decay 5e-3         --num-layers 2 --num-hidden 32 --batch-size  8 --lr 1e-2     --num-runs 10 --data-path "${EXP_DATA_DIR}"

# MLP-2
python3 backprop_plain.py MNIST           --subsample 10000 --resize 14 --num-layers 3 --num-hidden 64 32 --lr 1e-3 --weight-decay 5e-3 --num-runs 10 --data-path "${EXP_DATA_DIR}"
python3 backprop_plain.py FASHION_MNIST   --subsample 10000 --resize 14 --num-layers 3 --num-hidden 64 32 --lr 5e-4 --weight-decay 5e-3 --num-runs 10 --data-path "${EXP_DATA_DIR}"
python3 backprop_plain.py KUZUSHIJI_MNIST --subsample 10000 --resize 14 --num-layers 3 --num-hidden 64 32 --lr 5e-4 --weight-decay 0.0  --num-runs 10 --data-path "${EXP_DATA_DIR}"
python3 backprop_plain.py LETTER_RECOGNITION                            --num-layers 3 --num-hidden 64 32 --lr 5e-3 --weight-decay 1e-3 --num-runs 10 --data-path "${EXP_DATA_DIR}"
python3 backprop_plain.py BREAST_CANCER   --batch-size 16               --num-layers 3 --num-hidden 64 32 --lr 1e-3 --weight-decay 1e-2 --num-runs 10 --data-path "${EXP_DATA_DIR}"
python3 backprop_plain.py HEART_DISEASE   --batch-size 8                --num-layers 3 --num-hidden 64 32 --lr 5e-3 --weight-decay 0.0  --num-runs 10 --data-path "${EXP_DATA_DIR}"

# MLP-3
python3 backprop_plain.py MNIST            --subsample 10000 --resize 14 --num-layers 4 --num-hidden 128 64 32 --lr 5e-4 --weight-decay 1e-3 --num-runs 10 --data-path "${EXP_DATA_DIR}"
python3 backprop_plain.py FASHION_MNIST    --subsample 10000 --resize 14 --num-layers 4 --num-hidden 128 64 32 --lr 5e-4 --weight-decay 1e-3 --num-runs 10 --data-path "${EXP_DATA_DIR}"
python3 backprop_plain.py KUZUSHIJI_MNIST  --subsample 10000 --resize 14 --num-layers 4 --num-hidden 128 64 32 --lr 5e-4 --weight-decay 5e-4 --num-runs 10 --data-path "${EXP_DATA_DIR}"
python3 backprop_plain.py LETTER_RECOGNITION                             --num-layers 4 --num-hidden 128 64 32 --lr 1e-3 --weight-decay 0.0  --num-runs 10 --data-path "${EXP_DATA_DIR}"
python3 backprop_plain.py BREAST_CANCER    --batch-size 16               --num-layers 4 --num-hidden 128 64 32 --lr 1e-3 --weight-decay 1e-2 --num-runs 10 --data-path "${EXP_DATA_DIR}"
python3 backprop_plain.py HEART_DISEASE    --batch-size  8               --num-layers 4 --num-hidden 128 64 32 --lr 5e-3 --weight-decay 0.0  --num-runs 10 --data-path "${EXP_DATA_DIR}"

popd
