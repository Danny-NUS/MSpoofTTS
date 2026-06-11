#!/bin/bash

BASE="/data2/minh_duc/neutts_eval"
OUT="/data2/minh_duc/mos"

SCHEMES=("orig" "ras_k50_win25" "eas" "rank_ras_hier" "rank_eas_hier")

# =========================
# LIBRISPEECH
# =========================
LIBRISPEECH_IDS=(
"61-70968-0000"
"121-121726-0000"
"237-126133-0000"
"672-122797-0000"
"5142-33396-0000"
"7021-79730-0000"
"7729-102255-0000"
"8224-274381-0002"
"8455-210777-0000"
"8555-284447-0000"
)

for utt in "${LIBRISPEECH_IDS[@]}"; do
    mkdir -p "$OUT/Librispeech/$utt"
    for scheme in "${SCHEMES[@]}"; do
        src=$(find "$BASE/librispeech/syn/$scheme/wav" -maxdepth 1 -name "${utt}__ref_*__k0.wav")
        if [ -f "$src" ]; then
            cp "$src" "$OUT/Librispeech/$utt/${scheme}.wav"
        else
            echo "Missing Librispeech $utt $scheme"
        fi
    done
done


# =========================
# LIBRITTS
# =========================
LIBRITTS_IDS=(
"121_127105_000007_000002"
"1089_134686_000032_000007"
"1580_141083_000002_000002"
"1995_1826_000005_000001"
"2300_131720_000002_000001"
"2961_961_000002_000000"
"3570_5694_000005_000003"
"3729_6852_000003_000004"
"4077_13751_000006_000005"
"4970_29093_000004_000000"
)

for utt in "${LIBRITTS_IDS[@]}"; do
    mkdir -p "$OUT/Libritts/$utt"
    for scheme in "${SCHEMES[@]}"; do
        src=$(find "$BASE/libritts/syn/$scheme/wav" -maxdepth 1 -name "${utt}__ref_*__k0.wav")
        if [ -f "$src" ]; then
            cp "$src" "$OUT/Libritts/$utt/${scheme}.wav"
        else
            echo "Missing Libritts $utt $scheme"
        fi
    done
done


# =========================
# TWISTLIST
# =========================
TWIST_IDS=(
"twist_000000"
"twist_000001"
"twist_000006"
"twist_000007"
"twist_000013"
"twist_000016"
"twist_000021"
"twist_000025"
)

for utt in "${TWIST_IDS[@]}"; do
    mkdir -p "$OUT/Twistlist/$utt"
    for scheme in "${SCHEMES[@]}"; do
        src=$(find "$BASE/twistlist/syn/$scheme/wav" -maxdepth 1 -name "${utt}__ref_*__k0.wav")
        if [ -f "$src" ]; then
            cp "$src" "$OUT/Twistlist/$utt/${scheme}.wav"
        else
            echo "Missing Twistlist $utt $scheme"
        fi
    done
done

echo "Finished building MOS folder."