#!/bin/bash
# CUDA_VISIBLE_DEVICES=2 ./run_eval_one_scheme.sh

# ========================================
# Allowed schemes:
#   - ASR_GT
#   - orig
#   - recon
#   - ras_k50_win25
#   - dis
#   - ras_dis
#   - eas
# ========================================

SCHEME="ras_dis"
MAX_UTTS=100
N_SYN=1
SEED=42

python3 eval_one_scheme.py \
    --scheme ${SCHEME} \
    --n_syn_per_utt ${N_SYN} \
    --seed ${SEED} \
    --max_utts ${MAX_UTTS} \
    --nisqa
