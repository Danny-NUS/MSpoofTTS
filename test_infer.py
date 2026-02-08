#!/usr/bin/env python3

import os
import random
import ctypes
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Optional

import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm
from datasets import load_dataset, Audio
from jiwer import wer, cer
from transformers import WhisperProcessor, WhisperForConditionalGeneration

import re
from statistics import mean

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

SYN_ROOT = Path("/data2/minh_duc/neutts/libritts/infer.org.test.clean")

ASR_ROOT = Path("/data2/minh_duc/neutts/libritts/asr.infer.test")
ASR_TAG  = "whisper-large-v3"

N_SYN_PER_UTT = 1
SEED = 42
MAX_UTTS: Optional[int] = 100   # set e.g. 100 for debugging

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
torch.manual_seed(SEED)

# =========================
# UTILS
# =========================

def ensure_dirs():
    (SYN_ROOT / "wav").mkdir(parents=True, exist_ok=True)

    asr_dir = ASR_ROOT / ASR_TAG
    asr_dir.mkdir(parents=True, exist_ok=True)

    return asr_dir

def save_wav(path: Path, wav, sr: int):
    sf.write(path, wav, sr)

def resample_24k_to_16k(wav_24k):
    wav = torch.tensor(wav_24k)
    wav_16k = torchaudio.functional.resample(
        wav,
        orig_freq=24000,
        new_freq=16000
    )
    return wav_16k.cpu().numpy()

def log_msg(log_f, msg: str):
    ts = datetime.now().isoformat(timespec="seconds")
    log_f.write(f"[{ts}] {msg}\n")
    log_f.flush()

# =========================
# LOAD HF DATASET
# =========================

def load_hf_libritts(max_utts: Optional[int] = None):
    ds = load_dataset(
        "mythicinfinity/libritts",
        "clean",
        split="test.clean"
    ).cast_column("audio", Audio(sampling_rate=16000))

    if max_utts is not None:
        ds = ds.select(range(min(max_utts, len(ds))))

    return [
        {
            "utt_id": ex["id"],
            "speaker_id": ex["speaker_id"],
            "text": ex["text_normalized"],
            "audio": ex["audio"]["array"],
        }
        for ex in ds
    ]

# =========================
# WHISPER ASR
# =========================

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class WhisperASR:
    def __init__(self):
        self.processor = WhisperProcessor.from_pretrained(
            "openai/whisper-large-v3"
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3"
        ).to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def transcribe(self, wav_16k):
        inputs = self.processor(
            wav_16k,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(DEVICE)

        pred_ids = self.model.generate(
            **inputs,
            language="en",
            task="transcribe"
        )
        return self.processor.decode(
            pred_ids[0],
            skip_special_tokens=True
        )

# =========================
# BUILD + EVAL
# =========================

def build_and_eval(examples):
    asr_dir = ensure_dirs()

    # ---- files ----
    wav_scp = open(SYN_ROOT / "wav.scp", "a")
    text_f  = open(SYN_ROOT / "text", "a")
    log_f   = open(SYN_ROOT / "process.log", "a")
    asr_f   = open(asr_dir / "results.jsonl", "a")

    # ---- metrics accumulators ----
    wer_gt_list  = []
    wer_syn_list = []
    cer_gt_list  = []
    cer_syn_list = []

    # ---- write ASR metadata once ----
    meta_path = asr_dir / "meta.json"
    meta = {
        "asr_model": ASR_TAG,
        "date": datetime.now().isoformat(timespec="seconds"),
        "sampling_strategy": "neutts_default",
        "notes": "GT vs SYN ASR evaluation",
    }

    # ---- group by speaker ----
    spk2utts = defaultdict(list)
    for ex in examples:
        spk2utts[ex["speaker_id"]].append(ex)

    # ---- models ----
    tts = NeuTTS(
        backbone_repo="neuphonic/neutts-nano",
        backbone_device="cuda",
        codec_repo="neuphonic/neucodec",
        codec_device="cuda"
    )

    asr = WhisperASR()

    # ---- main loop ----
    for spk, utts in tqdm(spk2utts.items(), desc="Speakers"):
        if len(utts) < 2:
            continue

        for ex in tqdm(utts, desc=f"Utts spk={spk}", leave=False):
            src_id   = ex["utt_id"]
            src_text = ex["text"]

            refs = random.sample(
                [u for u in utts if u["utt_id"] != src_id],
                min(N_SYN_PER_UTT, len(utts) - 1)
            )

            for ref in refs:
                new_id = f"{src_id}_ref_{ref['utt_id']}"

                try:
                    # ---- prepare ref wav ----
                    tmp_ref = SYN_ROOT / "wav" / "_tmp_ref.wav"
                    save_wav(tmp_ref, ref["audio"], 16000)

                    # ---- synthesis ----
                    ref_codes = tts.encode_reference(str(tmp_ref))
                    wav_24k = tts.infer(
                        src_text,
                        ref_codes,
                        ref["text"]
                    )
                    wav_16k = resample_24k_to_16k(wav_24k)

                    wav_path = SYN_ROOT / "wav" / f"{new_id}.wav"
                    save_wav(wav_path, wav_16k, 16000)

                    # ---- ASR ----
                    asr_gt_raw  = asr.transcribe(ex["audio"])
                    asr_syn_raw = asr.transcribe(wav_16k)

                    # ---- normalize ----
                    ref_n = normalize_text(src_text)
                    gt_n  = normalize_text(asr_gt_raw)
                    syn_n = normalize_text(asr_syn_raw)

                    # ---- metrics ----
                    wer_gt  = wer(ref_n, gt_n)
                    wer_syn = wer(ref_n, syn_n)
                    cer_gt  = cer(ref_n, gt_n)
                    cer_syn = cer(ref_n, syn_n)

                    wer_gt_list.append(wer_gt)
                    wer_syn_list.append(wer_syn)
                    cer_gt_list.append(cer_gt)
                    cer_syn_list.append(cer_syn)

                    # ---- log JSON ----
                    record = {
                        "utt_id": src_id,
                        "speaker_id": spk,
                        "ref_text": src_text,
                        "asr_gt_text": asr_gt_raw,
                        "asr_syn_text": asr_syn_raw,
                        "wer_gt": wer_gt,
                        "wer_syn": wer_syn,
                        "cer_gt": cer_gt,
                        "cer_syn": cer_syn,
                    }

                    asr_f.write(json.dumps(record) + "\n")
                    asr_f.flush()

                    wav_scp.write(f"{new_id} {wav_path}\n")
                    text_f.write(f"{new_id} {src_text}\n")

                    log_msg(log_f, f"OK {new_id}")

                    tmp_ref.unlink(missing_ok=True)

                except Exception as e:
                    log_msg(log_f, f"FAIL {new_id} : {e}")

    # ---- finalize metadata with averages ----
    if wer_gt_list:
        meta.update({
            "num_samples": len(wer_gt_list),
            "avg_wer_gt":  mean(wer_gt_list),
            "avg_wer_syn": mean(wer_syn_list),
            "avg_cer_gt":  mean(cer_gt_list),
            "avg_cer_syn": mean(cer_syn_list),
            "avg_delta_wer": mean(wer_syn_list) - mean(wer_gt_list),
            "avg_delta_cer": mean(cer_syn_list) - mean(cer_gt_list),
        })

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    wav_scp.close()
    text_f.close()
    log_f.close()
    asr_f.close()


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    print("Loading LibriTTS from HuggingFace...")
    examples = load_hf_libritts(max_utts=MAX_UTTS)
    print(f"Loaded {len(examples)} utterances")

    build_and_eval(examples)
