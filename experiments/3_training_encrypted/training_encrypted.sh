#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# import var: 'EXP_DATA_DIR'
source ${SCRIPT_DIR}/../0_common/paths.sh

pushd "$SCRIPT_DIR" > /dev/null


python3 training_encrypted.py MNIST  \
        --subsample 10000 --resize 14  \
        --num-layers 2 --num-hidden 32  \
        --lr 1e-3 --weight-decay 1e-3 --num-runs 10  \
        --data-path "${EXP_DATA_DIR}"  \
        --config-dir "${EXP_CONF_DIR}"

popd
