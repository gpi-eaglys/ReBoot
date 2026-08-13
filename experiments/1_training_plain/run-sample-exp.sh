#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# import var: 'EXP_DATA_DIR'
source ${SCRIPT_DIR}/../0_common/paths.sh


pushd "$SCRIPT_DIR" > /dev/null

python backprop_plain.py MNIST --subsample 10000 --resize 14 --num-layers 2 --num-hidden 32 --lr 5e-3 --weight-decay 1e-3 --num-runs 1 --data-path "${EXP_DATA_DIR}"

popd > /dev/null
