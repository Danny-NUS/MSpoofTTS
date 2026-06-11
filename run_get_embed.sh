#!/bin/bash

DATA_DIR="/data2/minh_duc/from_hf/libritts/new.test.clean"
OUT_ROOT="/data2/minh_duc/neutts_eval/dis_drift1000"
MAX_UTTS=1000

SCHEMES=("orig")

mkdir -p $OUT_ROOT

for SCHEME in "${SCHEMES[@]}"
do
    echo "Running scheme: $SCHEME"

    python get_hidden_embed.py \
        --data_dir $DATA_DIR \
        --scheme $SCHEME \
        --max_utts $MAX_UTTS \
        --output $OUT_ROOT/${SCHEME}_gap.npz

done

echo "Done."