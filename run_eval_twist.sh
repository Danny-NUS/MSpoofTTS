#!/bin/bash
# Example:
# CUDA_VISIBLE_DEVICES=2 ./run_eval_twist.sh

# ========================================
# Allowed schemes:
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
#   - rank_hier
#   - rank_ras_hier
#   - rank_eas_hier
# ========================================

# ========================
# DATASET CONFIG
# ========================

# Kaldi dataset (reference voice pool)
DATA_DIR="/data2/minh_duc/from_hf/librispeech_asr/test.clean"

# Tongue twister text file (each line = one sentence)
TWIST_TXT="/data2/minh_duc/from_hf/twistlist/test.txt"

RESULT_ROOT="/data2/minh_duc/neutts_eval/twistlist_test"

# ========================
# EXPERIMENT CONFIG
# ========================

SCHEME="rank_eas_hier"

# ASR model options:
#   openai/whisper-large-v3
#   facebook/wav2vec2-large-960h
#   facebook/wav2vec2-large-960h-lv60-self
ASR_MODEL="openai/whisper-large-v3"
# ASR_MODEL="facebook/wav2vec2-large-960h"

MAX_LINES=-1     # -1 means use all twist lines
MAX_REFS=100      # -1 means load all ref utterances

N_SYN=1
SEED=42

# ========================
# RUN
# ========================

echo "======================================"
echo "Reference dataset: ${DATA_DIR}"
echo "Twist txt: ${TWIST_TXT}"
echo "Result root: ${RESULT_ROOT}"
echo "Scheme: ${SCHEME}"
echo "ASR model: ${ASR_MODEL}"
echo "Max lines: ${MAX_LINES}"
echo "Max refs: ${MAX_REFS}"
echo "Seed: ${SEED}"
echo "======================================"

python3 eval_twist.py \
    --scheme ${SCHEME} \
    --asr ${ASR_MODEL} \
    --data_dir ${DATA_DIR} \
    --twist_txt ${TWIST_TXT} \
    --result_root ${RESULT_ROOT} \
    --n_syn_per_line ${N_SYN} \
    --seed ${SEED} \
    --max_lines ${MAX_LINES} \
    --max_refs ${MAX_REFS} \
    --nisqa