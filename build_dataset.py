#!/usr/bin/env python3

import os
import random
import ctypes
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import torch
import torchaudio
import soundfile as sf
from datasets import load_dataset, Audio
from tqdm import tqdm

# =========================
# ENV BOOTSTRAP (CRITICAL)
# =========================

os.environ["PATH"] = os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"]
os.environ["LD_LIBRARY_PATH"] = os.path.expanduser("~/.local/lib") + ":" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["TORCHCODEC_DISABLE"] = "1"

ctypes.CDLL(os.path.expanduser("~/.local/lib/libespeak-ng.so"))

from neutts import NeuTTS

# =========================
# CONFIG
# =========================

REAL_ROOT = Path("/data2/minh_duc/from_hf/libritts/test.clean")
SYN_ROOT  = Path("/data2/minh_duc/neutts/libritts/test.clean")

N_SYN_PER_UTT = 3
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

# =========================
# UTILS
# =========================

def ensure_dirs(root: Path):
    (root / "wav").mkdir(parents=True, exist_ok=True)
    (root / "codes").mkdir(parents=True, exist_ok=True)

def speech_ids_to_codes(speech_ids):
    if hasattr(speech_ids, "tolist"):
        speech_ids = speech_ids.tolist()
    return "".join(f"<|speech_{i}|>" for i in speech_ids)

def save_wav(path, wav, sr):
    sf.write(path, wav, sr)

def resample_24k_to_16k(wav_24k):
    wav = torch.tensor(wav_24k)
    wav_16k = torchaudio.functional.resample(
        wav,
        orig_freq=24000,
        new_freq=16000
    )
    return wav_16k.cpu().numpy()

def log_msg(log_f, msg):
    ts = datetime.now().isoformat(timespec="seconds")
    log_f.write(f"[{ts}] {msg}\n")
    log_f.flush()

# =========================
# PROGRESS FROM wav.scp
# =========================

def load_real_done_ids(wav_scp_path: Path):
    done = set()
    if not wav_scp_path.exists():
        return done
    with open(wav_scp_path) as f:
        for line in f:
            utt_id = line.strip().split()[0]
            done.add(utt_id)
    return done

def load_syn_src2children(wav_scp_path: Path):
    """
    Returns:
      src_id -> set(child_utt_ids)
    """
    src2children = defaultdict(set)

    if not wav_scp_path.exists():
        return src2children

    with open(wav_scp_path) as f:
        for line in f:
            utt_id = line.strip().split()[0]
            if "_ref_" in utt_id:
                src_id = utt_id.split("_ref_")[0]
                src2children[src_id].add(utt_id)

    return src2children

# =========================
# STEP 1 — REAL DATA
# =========================

def build_real_dataset(ds):
    print("Building real dataset...")
    ensure_dirs(REAL_ROOT)

    wav_scp_path = REAL_ROOT / "wav.scp"
    utt2codes_path = REAL_ROOT / "utt2codes"
    text_path = REAL_ROOT / "text"
    log_path = REAL_ROOT / "process.log"

    done_ids = load_real_done_ids(wav_scp_path)

    wav_scp = open(wav_scp_path, "a")
    utt2codes = open(utt2codes_path, "a")
    text_f = open(text_path, "a")
    log_f = open(log_path, "a")

    tts = NeuTTS(
        backbone_repo="neuphonic/neutts-nano",
        backbone_device="cuda",
        codec_repo="neuphonic/neucodec",
        codec_device="cuda"
    )

    for ex in tqdm(ds, desc="Real utts"):
        utt_id = ex["id"]
        if utt_id in done_ids:
            continue

        try:
            wav_path = REAL_ROOT / "wav" / f"{utt_id}.wav"
            codes_path = REAL_ROOT / "codes" / f"{utt_id}.txt"

            save_wav(wav_path, ex["audio"]["array"], 16000)

            code_ids = tts.encode_reference(str(wav_path))
            code_str = speech_ids_to_codes(code_ids)

            with open(codes_path, "w") as f:
                f.write(code_str)

            wav_scp.write(f"{utt_id} {wav_path}\n")
            utt2codes.write(f"{utt_id} {codes_path}\n")
            text_f.write(f"{utt_id} {ex['text_normalized']}\n")

            wav_scp.flush()
            utt2codes.flush()
            text_f.flush()

            log_msg(log_f, f"REAL OK {utt_id}")

        except Exception as e:
            log_msg(log_f, f"REAL FAIL {utt_id} : {e}")

    wav_scp.close()
    utt2codes.close()
    text_f.close()
    log_f.close()

# =========================
# STEP 2 — SYNTHETIC DATA
# =========================

def build_synthetic_dataset(ds):
    print("Building synthetic dataset...")
    ensure_dirs(SYN_ROOT)

    wav_scp_path = SYN_ROOT / "wav.scp"
    utt2codes_path = SYN_ROOT / "utt2codes"
    text_path = SYN_ROOT / "text"
    log_path = SYN_ROOT / "process.log"

    src2children = load_syn_src2children(wav_scp_path)

    wav_scp = open(wav_scp_path, "a")
    utt2codes = open(utt2codes_path, "a")
    text_f = open(text_path, "a")
    log_f = open(log_path, "a")

    spk2utts = defaultdict(list)
    for ex in ds:
        spk2utts[ex["speaker_id"]].append(ex)

    tts = NeuTTS(
        backbone_repo="neuphonic/neutts-nano",
        backbone_device="cuda",
        codec_repo="neuphonic/neucodec",
        codec_device="cuda"
    )

    for spk, utts in tqdm(spk2utts.items(), desc="Speakers"):
        if len(utts) < 2:
            continue

        for ex in tqdm(utts, desc=f"Utts spk={spk}", leave=False):
            src_id = ex["id"]
            src_text = ex["text_normalized"]

            existing = src2children.get(src_id, set())
            remaining = N_SYN_PER_UTT - len(existing)

            if remaining <= 0:
                continue

            candidates = [u for u in utts if u["id"] != src_id]
            refs = random.sample(candidates, min(remaining, len(candidates)))

            for ref in refs:
                new_id = f"{src_id}_ref_{ref['id']}"
                if new_id in existing:
                    continue

                try:
                    ref_wav_path = REAL_ROOT / "wav" / f"{ref['id']}.wav"
                    ref_codes = tts.encode_reference(str(ref_wav_path))

                    wav_24k, code_str = tts.infer(src_text, ref_codes, ref["text_normalized"], return_codes=True)
                    wav_16k = resample_24k_to_16k(wav_24k)

                    wav_path = SYN_ROOT / "wav" / f"{new_id}.wav"
                    codes_path = SYN_ROOT / "codes" / f"{new_id}.txt"

                    save_wav(wav_path, wav_16k, 16000)

                    # code_ids = tts.encode_reference(str(wav_path))
                    # code_str = speech_ids_to_codes(code_ids)

                    with open(codes_path, "w") as f:
                        f.write(code_str)

                    wav_scp.write(f"{new_id} {wav_path}\n")
                    utt2codes.write(f"{new_id} {codes_path}\n")
                    text_f.write(f"{new_id} {src_text}\n")

                    wav_scp.flush()
                    utt2codes.flush()
                    text_f.flush()

                    src2children[src_id].add(new_id)
                    log_msg(log_f, f"SYN OK {new_id}")

                except Exception as e:
                    log_msg(log_f, f"SYN FAIL {new_id} : {e}")

    wav_scp.close()
    utt2codes.close()
    text_f.close()
    log_f.close()

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    print("Loading LibriTTS test.clean once...")
    ds = load_dataset(
        "mythicinfinity/libritts",
        "clean",
        split="test.clean"
    ).cast_column("audio", Audio(sampling_rate=16000))

    # DEBUG: limit size (remove later)
    # ds = ds.select(range(min(8, len(ds))))

    build_real_dataset(ds)
    build_synthetic_dataset(ds)
