#!/bin/bash
# Example:
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
#   - eas_dis
#   - hier
#   - ras_hier
#   - eas_hier
# ========================================

# ========================
# DATASET CONFIG
# ========================

DATA_DIR="/data/datasets/libritts_test_clean"
RESULT_ROOT="/data/results/libritts_test_clean"

# ========================
# EXPERIMENT CONFIG
# ========================

SCHEME="ASR_GT"

# ASR model options:
#   openai/whisper-large-v3
#   facebook/wav2vec2-large-960h
#   facebook/wav2vec2-large-960h-lv60-self
ASR_MODEL="facebook/wav2vec2-large-960h"

MAX_UTTS=100
N_SYN=1
SEED=42

# ========================
# RUN
# ========================

echo "======================================"
echo "Dataset: ${DATA_DIR}"
echo "Result root: ${RESULT_ROOT}"
echo "Scheme: ${SCHEME}"
echo "ASR model: ${ASR_MODEL}"
echo "Max utts: ${MAX_UTTS}"
echo "Seed: ${SEED}"
echo "======================================"

python3 eval_one_scheme.py \
    --scheme ${SCHEME} \
    --asr ${ASR_MODEL} \
    --data_dir ${DATA_DIR} \
    --result_root ${RESULT_ROOT} \
    --n_syn_per_utt ${N_SYN} \
    --seed ${SEED} \
    --max_utts ${MAX_UTTS} \
    --nisqa