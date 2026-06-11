#!/usr/bin/env python3
import argparse
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from neutts import NeuTTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# CONFIG
# =========================

SEGMENT_SIZES = ["10", "25", "50", "UTT"]
MIN_TOKEN_THRESHOLD = 200
K_WINDOWS = 5

REGION_ANCHORS = {
    "EARLY": 0.15,
    "MID": 0.50,
    "LATE": 0.85,
}

REGION_LABELS = {
    "EARLY": 0,
    "MID": 1,
    "LATE": 2,
    "UTT": -1,
}

# =========================
# DATA LOADER
# =========================

def load_kaldi_dataset(data_dir: Path, max_utts=None):
    wav_map = {}
    text_map = {}
    spk_map = {}

    with open(data_dir / "wav.scp") as f:
        for line in f:
            utt, path = line.strip().split(maxsplit=1)
            wav_map[utt] = path

    with open(data_dir / "text") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            text_map[parts[0]] = parts[1] if len(parts) > 1 else ""

    with open(data_dir / "utt2spk") as f:
        for line in f:
            utt, spk = line.strip().split()
            spk_map[utt] = spk

    utts = sorted(set(wav_map) & set(text_map) & set(spk_map))
    if max_utts is not None:
        utts = utts[:max_utts]

    examples = []
    for utt in utts:
        examples.append({
            "utt_id": utt,
            "speaker_id": spk_map[utt],
            "wav_path": wav_map[utt],
            "text": text_map[utt],
        })

    return examples


# =========================
# MULTI-WINDOW EXTRACTION
# =========================

def extract_multi_segments(hidden: torch.Tensor, region: str, size: int):
    """
    hidden: [Len, D]
    returns list of pooled vectors
    """

    L = hidden.shape[0]
    if L < size:
        return []

    anchor = int(REGION_ANCHORS[region] * L)

    step = max(1, size // 4)
    half = K_WINDOWS // 2

    vectors = []

    for i in range(-half, half + 1):
        center = anchor + i * step

        start = center - size // 2
        end = start + size

        if start < 0:
            start = 0
            end = size
        if end > L:
            end = L
            start = L - size

        if start < 0 or end > L:
            continue

        seg = hidden[start:end]
        vectors.append(seg.mean(dim=0))

    return vectors


# =========================
# MAIN PIPELINE
# =========================

def run_visualization(
    scheme: str,
    data_dir: Path,
    output_path: Path,
    max_utts: int,
):
    random.seed(42)
    torch.manual_seed(42)

    examples = load_kaldi_dataset(data_dir, max_utts)
    print(f"Loaded {len(examples)} utterances")

    spk2utts = defaultdict(list)
    for ex in examples:
        spk2utts[ex["speaker_id"]].append(ex)

    tts = NeuTTS(
        backbone_repo="neuphonic/neutts-nano",
        backbone_device="cuda",
        codec_repo="neuphonic/neucodec",
        codec_device="cuda",
        use_dis="dis" in scheme,
        use_hier="hier" in scheme,
    )

    syn_vectors = []
    gt_vectors = []
    region_labels = []
    size_labels = []
    utt_ids = []

    for spk, utts in tqdm(spk2utts.items(), desc="Speakers"):
        if len(utts) < 2:
            continue

        for ex in tqdm(utts, desc=f"Utts spk={spk}", leave=False):
            utt_id = ex["utt_id"]
            text = ex["text"]
            gt_path = ex["wav_path"]

            ref_pool = [u for u in utts if u["utt_id"] != utt_id]
            if not ref_pool:
                continue

            ref = ref_pool[0]  # deterministic

            result = tts.infer_distribution_gap(
                text=text,
                ref_text=ref["text"],
                ref_path=ref["wav_path"],
                gt_path=gt_path,
                sampling_scheme=scheme,
            )

            Hs = result["Hs_speech"].detach().cpu()
            Hg = result["Hg_speech"].detach().cpu()

            if Hs.shape[0] < MIN_TOKEN_THRESHOLD or Hg.shape[0] < MIN_TOKEN_THRESHOLD:
                continue

            for size_str in SEGMENT_SIZES:

                if size_str == "UTT":
                    syn_vec = Hs.mean(dim=0)
                    gt_vec = Hg.mean(dim=0)

                    syn_vectors.append(syn_vec.numpy())
                    gt_vectors.append(gt_vec.numpy())
                    region_labels.append(REGION_LABELS["UTT"])
                    size_labels.append("UTT")
                    utt_ids.append(utt_id)

                else:
                    size = int(size_str)

                    for region in ["EARLY", "MID", "LATE"]:

                        syn_segs = extract_multi_segments(Hs, region, size)
                        gt_segs = extract_multi_segments(Hg, region, size)

                        min_k = min(len(syn_segs), len(gt_segs))

                        for k in range(min_k):
                            syn_vectors.append(syn_segs[k].numpy())
                            gt_vectors.append(gt_segs[k].numpy())
                            region_labels.append(REGION_LABELS[region])
                            size_labels.append(size)
                            utt_ids.append(utt_id)

    syn_arr = np.stack(syn_vectors)
    gt_arr = np.stack(gt_vectors)
    region_arr = np.array(region_labels)
    size_arr = np.array(size_labels)
    utt_ids = np.array(utt_ids)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        syn=syn_arr,
        gt=gt_arr,
        region=region_arr,
        size=size_arr,
        utt_ids=utt_ids,
    )

    print(f"Saved {len(region_arr)} segments to {output_path}")


# =========================
# CLI
# =========================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--scheme", type=str, default="orig")
    p.add_argument("--max_utts", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_visualization(
        scheme=args.scheme,
        data_dir=Path(args.data_dir),
        output_path=Path(args.output),
        max_utts=args.max_utts,
    )