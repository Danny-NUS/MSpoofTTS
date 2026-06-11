#!/usr/bin/env python3
import os
import re
import json
import time
import random
import ctypes
import argparse
import tempfile
import math
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm
from jiwer import wer, cer
from jiwer import Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip
from num2words import num2words

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

# =========================
# ENV BOOTSTRAP (CRITICAL)
# =========================

os.environ["PATH"] = os.path.expanduser("~/.local/bin") + ":" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = os.path.expanduser("~/.local/lib") + ":" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["TORCHCODEC_DISABLE"] = "1"

ctypes.CDLL(os.path.expanduser("~/.local/lib/libespeak-ng.so"))

from neutts import NeuTTS  # noqa: E402

# =========================
# CONFIG
# =========================

SYN_ROOT: Path
ASR_ROOT: Path

DEFAULT_N_SYN_PER_LINE = 1
DEFAULT_SEED = 42
DEFAULT_MAX_LINES: Optional[int] = None  # None => full file
DEFAULT_MAX_REFS: Optional[int] = None   # None => full kaldi utts

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Schemes allowed for TTS synthesis (NO ASR_GT)
sampling_schemes = [
    "orig", "ras_k50_win25", "dis", "ras_dis", "eas",
    "recon", "eas_dis", "hier", "ras_hier", "eas_hier",
    "rank_hier", "rank_ras_hier", "rank_eas_hier",
]

# =========================
# Optional NISQA
# =========================

_HAS_NISQA = False
try:
    from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment  # type: ignore
    _HAS_NISQA = True
except Exception:
    _HAS_NISQA = False

# =========================
# Speaker Similarity (WavLM x-vector)
# =========================

def _finite_or_zero(x: Optional[float]) -> Optional[float]:
    """Return 0.0 if x is nan/inf. Preserve None."""
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return 0.0
    return x if math.isfinite(x) else 0.0


def _sanitize_wav(wav: torch.Tensor) -> torch.Tensor:
    wav = wav.clone()
    wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    wav = wav.clamp(-1.0, 1.0)
    return wav


class SpeakerSimilarity:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-base-plus-sv"
        )
        self.model = WavLMForXVector.from_pretrained(
            "microsoft/wavlm-base-plus-sv"
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def embedding(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        wav = _sanitize_wav(wav)

        if sr != 16000:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)

        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        inputs = self.feature_extractor(
            wav.squeeze(0).cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)

        outputs = self.model(input_values)

        if hasattr(outputs, "xvector"):
            emb = outputs.xvector.squeeze(0)
        elif hasattr(outputs, "embeddings"):
            emb = outputs.embeddings.squeeze(0)
        else:
            raise RuntimeError("Unexpected WavLM output format")

        emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

        denom = emb.norm(p=2)
        if (not torch.isfinite(denom)) or denom.item() < 1e-12:
            return torch.zeros_like(emb)

        return emb / denom

    @torch.no_grad()
    def similarity(self, wav_ref: torch.Tensor, sr_ref: int, wav_syn: torch.Tensor, sr_syn: int) -> float:
        emb_ref = self.embedding(wav_ref, sr_ref)
        emb_syn = self.embedding(wav_syn, sr_syn)
        sim = torch.dot(emb_ref, emb_syn)
        val = float(sim.item())
        return val if math.isfinite(val) else 0.0


# =========================
# TEXT NORMALIZATION (WER SAFE)
# =========================

_jiwer_transform = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip(),
])

_number_regex = re.compile(r"\b\d+\b")


def _expand_number(match):
    try:
        return num2words(int(match.group(0)))
    except Exception:
        return match.group(0)


def normalize_text(text: str) -> str:
    text = _jiwer_transform(text)
    text = _number_regex.sub(_expand_number, text)
    text = _jiwer_transform(text)
    return text


# =========================
# CUDA tracking
# =========================

def reset_cuda_peak():
    if DEVICE.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_cuda_peak_mb() -> Optional[float]:
    if DEVICE.startswith("cuda") and torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated()
        return float(peak) / (1024 ** 2)
    return None


def cuda_sync_if_needed() -> None:
    if DEVICE.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


# =========================
# IO UTILS
# =========================

def save_wav(path: Path, wav: torch.Tensor, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if wav.ndim == 2 and wav.shape[0] == 1:
        wav = wav.squeeze(0)
    sf.write(str(path), wav.detach().cpu().numpy(), sr)


def resample(wav: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    if orig_sr == new_sr:
        return wav
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    wav_rs = torchaudio.functional.resample(wav, orig_freq=orig_sr, new_freq=new_sr)
    return wav_rs.squeeze(0)


def audio_duration_sec(wav: torch.Tensor, sr: int) -> float:
    return float(int(wav.numel())) / float(sr)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log_msg(log_f, msg: str) -> None:
    ts = now_iso()
    log_f.write(f"[{ts}] {msg}\n")
    log_f.flush()


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def load_done_keys(done_path: Path) -> set:
    if not done_path.exists():
        return set()
    with open(done_path, "r") as f:
        return set(line.strip() for line in f if line.strip())


def append_done_key(done_path: Path, key: str) -> None:
    done_path.parent.mkdir(parents=True, exist_ok=True)
    with open(done_path, "a") as f:
        f.write(key + "\n")
        f.flush()


def make_key(line_id: str, ref_id: str, syn_idx: int) -> str:
    return f"{line_id}__ref={ref_id}__k={syn_idx}"


# =========================
# LOAD TONGUE TWISTER LINES
# =========================

def load_twist_lines(txt_path: Path, max_lines: Optional[int]) -> List[Dict[str, Any]]:
    assert txt_path.exists(), f"Missing twist txt: {txt_path}"
    lines: List[str] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            lines.append(s)
            if max_lines is not None and len(lines) >= max_lines:
                break

    out: List[Dict[str, Any]] = []
    for i, s in enumerate(lines):
        line_id = f"twist_{i:06d}"
        out.append({"line_id": line_id, "text": s})
    return out


# =========================
# LOAD KALDI DATASET (REF VOICE POOL)
#   NEW: speaker-balanced sampling using utt2spk
# =========================

def _balanced_sample_utts(
    spk2utts: Dict[str, List[str]],
    target_n: int,
) -> List[str]:
    """
    Speaker-balanced sampling (round-robin across speakers).
    Assumes random.seed() already set by caller for reproducibility.
    """
    spk_list = list(spk2utts.keys())
    random.shuffle(spk_list)
    for spk in spk_list:
        random.shuffle(spk2utts[spk])

    # round-robin pick
    picks: List[str] = []
    i = 0
    while len(picks) < target_n and len(spk_list) > 0:
        spk = spk_list[i % len(spk_list)]
        if len(spk2utts[spk]) == 0:
            # remove empty speakers
            spk_list.remove(spk)
            if len(spk_list) == 0:
                break
            continue

        picks.append(spk2utts[spk].pop())  # pop one utt from that speaker
        i += 1

    return picks


def load_kaldi_dataset(data_dir: Path, max_utts: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Expects:
      wav.scp
      text
      utt2spk

    Returns list of:
      {utt_id, speaker_id, text, audio_16k (np array)}

    NEW behavior:
      - If max_utts is None: load all (old behavior)
      - If max_utts is set: use utt2spk to do speaker-balanced sampling, then only load those wavs.
    """
    wav_scp_path = data_dir / "wav.scp"
    text_path = data_dir / "text"
    utt2spk_path = data_dir / "utt2spk"

    assert wav_scp_path.exists(), f"Missing {wav_scp_path}"
    assert text_path.exists(), f"Missing {text_path}"
    assert utt2spk_path.exists(), f"Missing {utt2spk_path}"

    # Load wav.scp
    wav_map: Dict[str, str] = {}
    with open(wav_scp_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt, path = line.split(maxsplit=1)
            wav_map[utt] = path

    # Load text
    text_map: Dict[str, str] = {}
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            utt = parts[0]
            txt = parts[1] if len(parts) > 1 else ""
            text_map[utt] = txt

    # Load utt2spk
    spk_map: Dict[str, str] = {}
    spk2utts_all: Dict[str, List[str]] = defaultdict(list)
    with open(utt2spk_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt, spk = line.split()
            spk_map[utt] = spk
            spk2utts_all[spk].append(utt)

    # Intersect valid utts
    valid_utts = set(wav_map) & set(text_map) & set(spk_map)
    if len(valid_utts) == 0:
        raise RuntimeError(f"No valid utterances after intersecting wav.scp/text/utt2spk in {data_dir}")

    # Filter spk2utts to valid utts only
    spk2utts: Dict[str, List[str]] = {}
    for spk, utts in spk2utts_all.items():
        kept = [u for u in utts if u in valid_utts]
        if kept:
            spk2utts[spk] = kept

    # Decide which utts to load
    if max_utts is None:
        utt_ids = sorted(list(valid_utts))
    else:
        # speaker-balanced sample
        utt_ids = _balanced_sample_utts({k: v.copy() for k, v in spk2utts.items()}, target_n=max_utts)

        # if still not enough (e.g., max_utts > total valid), fall back to all valid
        if len(utt_ids) < max_utts and len(valid_utts) > len(utt_ids):
            remaining = list(valid_utts - set(utt_ids))
            random.shuffle(remaining)
            utt_ids.extend(remaining[: (max_utts - len(utt_ids))])

        utt_ids = utt_ids[:max_utts]

    # Load audio only for selected utts
    examples: List[Dict[str, Any]] = []
    for utt in tqdm(utt_ids, desc="Loading kaldi wavs (selected)"):
        wav_path = wav_map[utt]
        wav, sr = torchaudio.load(wav_path)
        if wav.ndim > 1:
            wav = wav.mean(dim=0)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        examples.append({
            "utt_id": utt,
            "speaker_id": spk_map[utt],
            "text": text_map[utt],
            "audio_16k": wav.numpy(),
        })

    return examples


# =========================
# ASR
# =========================

class BaseASR:
    def transcribe_16k(self, wav_16k: torch.Tensor) -> str:
        raise NotImplementedError


class WhisperASR(BaseASR):
    def __init__(self, model_id: str):
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def transcribe_16k(self, wav_16k: torch.Tensor) -> str:
        inputs = self.processor(
            wav_16k.detach().cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).to(DEVICE)

        pred_ids = self.model.generate(
            **inputs,
            language="en",
            task="transcribe"
        )
        return self.processor.decode(pred_ids[0], skip_special_tokens=True)


class Wav2Vec2ASR(BaseASR):
    def __init__(self, model_id: str):
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def transcribe_16k(self, wav_16k: torch.Tensor) -> str:
        inputs = self.processor(
            wav_16k.detach().cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).to(DEVICE)

        logits = self.model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(pred_ids)[0]
        return transcription.lower()


def load_asr_model(asr_tag: str) -> BaseASR:
    tag = asr_tag.lower()
    if "whisper" in tag:
        print(f"Loading Whisper ASR: {asr_tag}")
        return WhisperASR(asr_tag)
    if "wav2vec2" in tag:
        print(f"Loading Wav2Vec2 ASR: {asr_tag}")
        return Wav2Vec2ASR(asr_tag)
    raise ValueError(f"Unsupported ASR tag: {asr_tag}")


# =========================
# NISQA
# =========================

class NISQAMetric:
    def __init__(self, sr: int):
        if not _HAS_NISQA:
            raise RuntimeError("torchmetrics NonIntrusiveSpeechQualityAssessment not available")
        self.sr = sr
        self.metric = NonIntrusiveSpeechQualityAssessment(sr).to(DEVICE)

    @torch.no_grad()
    def __call__(self, wav: torch.Tensor) -> Dict[str, float]:
        wav = wav.to(DEVICE)
        scores = self.metric(wav)  # (5,)
        scores = scores.detach().float().cpu().tolist()
        return {
            "nisqa_overall": float(scores[0]),
            "nisqa_noisiness": float(scores[1]),
            "nisqa_discontinuity": float(scores[2]),
            "nisqa_coloration": float(scores[3]),
            "nisqa_loudness": float(scores[4]),
        }


# =========================
# SUMMARY (incremental)
# =========================

def init_or_load_summary(summary_path: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
    existing = load_json(summary_path)
    if existing is not None:
        existing.setdefault("running", {})
        r = existing["running"]
        r.setdefault("count", 0)
        r.setdefault("sum_wer", 0.0)
        r.setdefault("sum_cer", 0.0)

        r.setdefault("sum_sim_same", 0.0)
        r.setdefault("count_sim_same", 0)
        r.setdefault("sum_sim_diff", 0.0)
        r.setdefault("count_sim_diff", 0)
        r.setdefault("sum_sim_margin", 0.0)
        r.setdefault("count_sim_margin", 0)

        r.setdefault("sum_rtf", 0.0)
        r.setdefault("count_rtf", 0)
        r.setdefault("nisqa", {"count": 0, "sum_overall": 0.0, "sum_noisiness": 0.0,
                               "sum_discontinuity": 0.0, "sum_coloration": 0.0, "sum_loudness": 0.0})
        r.setdefault("sum_gpu_peak_mb", 0.0)
        r.setdefault("count_gpu_peak", 0)
        existing.setdefault("meta", meta)
        return existing

    return {
        "meta": meta,
        "last_updated": now_iso(),
        "num_samples": 0,
        "avg_wer": None,
        "avg_cer": None,
        "avg_sim_same": None,
        "avg_sim_diff": None,
        "avg_sim_margin": None,
        "avg_rtf": None,
        "avg_nisqa_overall": None,
        "avg_nisqa_noisiness": None,
        "avg_nisqa_discontinuity": None,
        "avg_nisqa_coloration": None,
        "avg_nisqa_loudness": None,
        "avg_gpu_peak_mem_mb": None,
        "running": {
            "count": 0,
            "sum_wer": 0.0,
            "sum_cer": 0.0,

            "sum_sim_same": 0.0,
            "count_sim_same": 0,
            "sum_sim_diff": 0.0,
            "count_sim_diff": 0,
            "sum_sim_margin": 0.0,
            "count_sim_margin": 0,

            "sum_rtf": 0.0,
            "count_rtf": 0,

            "sum_gpu_peak_mb": 0.0,
            "count_gpu_peak": 0,

            "nisqa": {
                "count": 0,
                "sum_overall": 0.0,
                "sum_noisiness": 0.0,
                "sum_discontinuity": 0.0,
                "sum_coloration": 0.0,
                "sum_loudness": 0.0,
            },
        },
    }


def update_summary(summary: Dict[str, Any], record: Dict[str, Any]) -> None:
    r = summary["running"]
    r["count"] += 1
    r["sum_wer"] += float(record["wer"])
    r["sum_cer"] += float(record["cer"])

    sim_same = _finite_or_zero(record.get("sim_same", None))
    if sim_same is not None:
        r["sum_sim_same"] += float(sim_same)
        r["count_sim_same"] += 1

    sim_diff = _finite_or_zero(record.get("sim_diff", None))
    if sim_diff is not None:
        r["sum_sim_diff"] += float(sim_diff)
        r["count_sim_diff"] += 1

    sim_margin = _finite_or_zero(record.get("sim_margin", None))
    if sim_margin is not None:
        r["sum_sim_margin"] += float(sim_margin)
        r["count_sim_margin"] += 1

    rtf = record.get("rtf", None)
    if rtf is not None:
        r["sum_rtf"] += float(rtf)
        r["count_rtf"] += 1

    gpu = record.get("gpu_peak_mem_mb", None)
    if gpu is not None:
        r["sum_gpu_peak_mb"] += float(gpu)
        r["count_gpu_peak"] += 1

    if "nisqa_overall" in record:
        rn = r["nisqa"]
        rn["count"] += 1
        rn["sum_overall"] += float(record["nisqa_overall"])
        rn["sum_noisiness"] += float(record["nisqa_noisiness"])
        rn["sum_discontinuity"] += float(record["nisqa_discontinuity"])
        rn["sum_coloration"] += float(record["nisqa_coloration"])
        rn["sum_loudness"] += float(record["nisqa_loudness"])

    summary["num_samples"] = r["count"]
    summary["avg_wer"] = r["sum_wer"] / r["count"]
    summary["avg_cer"] = r["sum_cer"] / r["count"]

    summary["avg_sim_same"] = (r["sum_sim_same"] / r["count_sim_same"]) if r["count_sim_same"] > 0 else None
    summary["avg_sim_diff"] = (r["sum_sim_diff"] / r["count_sim_diff"]) if r["count_sim_diff"] > 0 else None
    summary["avg_sim_margin"] = (r["sum_sim_margin"] / r["count_sim_margin"]) if r["count_sim_margin"] > 0 else None
    summary["avg_rtf"] = (r["sum_rtf"] / r["count_rtf"]) if r["count_rtf"] > 0 else None
    summary["avg_gpu_peak_mem_mb"] = (r["sum_gpu_peak_mb"] / r["count_gpu_peak"]) if r["count_gpu_peak"] > 0 else None

    rn = r["nisqa"]
    if rn["count"] > 0:
        summary["avg_nisqa_overall"] = rn["sum_overall"] / rn["count"]
        summary["avg_nisqa_noisiness"] = rn["sum_noisiness"] / rn["count"]
        summary["avg_nisqa_discontinuity"] = rn["sum_discontinuity"] / rn["count"]
        summary["avg_nisqa_coloration"] = rn["sum_coloration"] / rn["count"]
        summary["avg_nisqa_loudness"] = rn["sum_loudness"] / rn["count"]
    else:
        summary["avg_nisqa_overall"] = None
        summary["avg_nisqa_noisiness"] = None
        summary["avg_nisqa_discontinuity"] = None
        summary["avg_nisqa_coloration"] = None
        summary["avg_nisqa_loudness"] = None

    summary["last_updated"] = now_iso()


# =========================
# PATHS PER SCHEME
# =========================

def get_scheme_dirs(scheme: str, asr_tag: str) -> Tuple[Path, Path]:
    syn_wav_dir = SYN_ROOT / scheme / "wav"
    asr_scheme_dir = ASR_ROOT / asr_tag / scheme
    syn_wav_dir.mkdir(parents=True, exist_ok=True)
    asr_scheme_dir.mkdir(parents=True, exist_ok=True)
    return syn_wav_dir, asr_scheme_dir


# =========================
# MAIN EVAL (twist -> TTS -> ASR)
# =========================

def run_scheme(
    scheme: str,
    twist_lines: List[Dict[str, Any]],
    ref_examples: List[Dict[str, Any]],
    n_syn_per_line: int,
    seed: int,
    max_lines: Optional[int],
    max_refs: Optional[int],
    enable_nisqa: bool,
    asr_tag: str,
) -> None:
    allowed = set(sampling_schemes)
    assert scheme in allowed, f"scheme={scheme} not in allowed={sorted(allowed)}"

    random.seed(seed)
    torch.manual_seed(seed)

    syn_wav_dir, asr_scheme_dir = get_scheme_dirs(scheme, asr_tag)

    # files
    log_f = open(asr_scheme_dir / "process.log", "a", encoding="utf-8")
    results_path = asr_scheme_dir / "results.jsonl"
    asr_f = open(results_path, "a", encoding="utf-8")
    done_path = asr_scheme_dir / "done_keys.txt"
    done = load_done_keys(done_path)

    meta = {
        "asr_model": asr_tag,
        "date_started": now_iso(),
        "scheme": scheme,
        "sampling_schemes_allowed": sampling_schemes,
        "n_syn_per_line": n_syn_per_line,
        "max_lines": max_lines,
        "seed": seed,
        "device": DEVICE,
        "notes": "Tongue-twister eval: prompt text from twist_txt, reference voice from Kaldi dataset (speaker-balanced ref subset when max_refs is set).",
        "syn_root": str(SYN_ROOT),
        "asr_root": str(ASR_ROOT),
        "max_refs": max_refs,
    }
    summary_path = asr_scheme_dir / "summary.json"
    summary = init_or_load_summary(summary_path, meta)
    atomic_write_json(summary_path, summary)

    # group refs by speaker for anchor/negative sampling
    spk2utts = defaultdict(list)
    for ex in ref_examples:
        spk2utts[ex["speaker_id"]].append(ex)

    spk_list = sorted(list(spk2utts.keys()))
    if len(spk_list) == 0:
        raise RuntimeError("No speakers found in Kaldi ref dataset.")

    # models
    asr = load_asr_model(asr_tag)

    use_dis = "dis" in scheme
    use_hier = "hier" in scheme

    tts = NeuTTS(
        backbone_repo="neuphonic/neutts-nano",
        backbone_device="cuda",
        codec_repo="neuphonic/neucodec",
        codec_device="cuda",
        use_dis=use_dis,
        use_hier=use_hier,
    )

    nisqa_metric = None
    if enable_nisqa and _HAS_NISQA:
        nisqa_metric = NISQAMetric(sr=24000)
        log_msg(log_f, "NISQA enabled (24kHz).")
    elif enable_nisqa and not _HAS_NISQA:
        log_msg(log_f, "WARNING: NISQA requested but torchmetrics NISQA not available. Skipping NISQA.")

    sim_model = SpeakerSimilarity(device=DEVICE)

    def sample_ref() -> Dict[str, Any]:
        spk = random.choice(spk_list)
        return random.choice(spk2utts[spk])

    for line_ex in tqdm(twist_lines, desc="Twist lines"):
        line_id = line_ex["line_id"]
        gt_text = line_ex["text"]

        chosen_refs: List[Dict[str, Any]] = []
        seen_ref_utts = set()
        attempts = 0
        while len(chosen_refs) < n_syn_per_line and attempts < n_syn_per_line * 10:
            attempts += 1
            ref = sample_ref()
            if ref["utt_id"] in seen_ref_utts:
                continue
            seen_ref_utts.add(ref["utt_id"])
            chosen_refs.append(ref)

        if len(chosen_refs) == 0:
            continue

        for syn_idx, ref in enumerate(chosen_refs):
            ref_id = ref["utt_id"]
            spk = ref["speaker_id"]

            key = make_key(line_id, ref_id=ref_id, syn_idx=syn_idx)
            if key in done:
                continue

            try:
                with tempfile.TemporaryDirectory() as td:
                    tmp_ref = Path(td) / "ref.wav"
                    wav_ref_16k = torch.tensor(ref["audio_16k"], dtype=torch.float32)
                    save_wav(tmp_ref, wav_ref_16k, 16000)
                    ref_codes = tts.encode_reference(str(tmp_ref))

                reset_cuda_peak()
                cuda_sync_if_needed()
                t0 = time.perf_counter()

                wav_24k = tts.infer(
                    gt_text,
                    ref_codes,
                    ref["text"],
                    sampling_scheme=scheme
                )

                cuda_sync_if_needed()
                t1 = time.perf_counter()

                gpu_peak_mb = get_cuda_peak_mb()
                gen_time = float(t1 - t0)

                wav_24k_t = torch.tensor(wav_24k, dtype=torch.float32).view(-1)
                dur = audio_duration_sec(wav_24k_t, 24000)
                rtf = gen_time / max(dur, 1e-6)

                wav_path = syn_wav_dir / f"{line_id}__ref_{ref_id}__k{syn_idx}.wav"
                save_wav(wav_path, wav_24k_t, 24000)

                wav_16k_t = resample(wav_24k_t, orig_sr=24000, new_sr=16000)
                asr_pred = asr.transcribe_16k(wav_16k_t)

                gt_n = normalize_text(gt_text)
                pred_n = normalize_text(asr_pred)
                w = wer(gt_n, pred_n)
                c = cer(gt_n, pred_n)

                sim_same = None
                sim_diff = None
                sim_margin = None

                anchor_pool = [u for u in spk2utts[spk] if u["utt_id"] != ref_id]
                if len(anchor_pool) > 0:
                    anchor_ex = random.choice(anchor_pool)
                    wav_anchor_16k = torch.tensor(anchor_ex["audio_16k"], dtype=torch.float32)
                    sim_same = sim_model.similarity(wav_anchor_16k, 16000, wav_16k_t, 16000)

                    other_spks = [s for s in spk_list if s != spk]
                    if len(other_spks) > 0:
                        spk_neg = random.choice(other_spks)
                        neg_ex = random.choice(spk2utts[spk_neg])
                        wav_neg_16k = torch.tensor(neg_ex["audio_16k"], dtype=torch.float32)
                        sim_diff = sim_model.similarity(wav_neg_16k, 16000, wav_16k_t, 16000)
                        sim_margin = (sim_same - sim_diff) if (sim_same is not None and sim_diff is not None) else None

                sim_same = _finite_or_zero(sim_same)
                sim_diff = _finite_or_zero(sim_diff)
                sim_margin = _finite_or_zero(sim_margin)

                record: Dict[str, Any] = {
                    "key": key,
                    "scheme": scheme,
                    "line_id": line_id,
                    "ref_speaker_id": spk,
                    "ref_utt_id": ref_id,
                    "ref_text": ref["text"],
                    "gt_text": gt_text,
                    "asr_text": asr_pred,
                    "wer": float(w),
                    "cer": float(c),
                    "audio_path": str(wav_path),
                    "audio_sr": 24000,
                    "duration_sec": float(dur),
                    "gen_time_sec": float(gen_time),
                    "rtf": float(rtf),
                    "sim_same": sim_same,
                    "sim_diff": sim_diff,
                    "sim_margin": sim_margin,
                    "gpu_peak_mem_mb": gpu_peak_mb,
                }

                if nisqa_metric is not None:
                    try:
                        record.update(nisqa_metric(wav_24k_t))
                    except Exception as ne:
                        log_msg(log_f, f"WARN NISQA failed key={key}: {ne}")

                asr_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                asr_f.flush()

                append_done_key(done_path, key)
                done.add(key)

                update_summary(summary, record)
                atomic_write_json(summary_path, summary)

                log_msg(
                    log_f,
                    f"OK {scheme} line={line_id} ref={ref_id} k={syn_idx} "
                    f"wer={w:.4f} cer={c:.4f} rtf={rtf:.3f}"
                )

            except Exception as e:
                log_msg(log_f, f"FAIL {scheme} line={line_id} ref={ref_id} k={syn_idx} : {e}")

    summary.setdefault("meta", {})
    summary["meta"]["date_finished"] = now_iso()
    atomic_write_json(summary_path, summary)

    log_msg(log_f, f"DONE scheme={scheme} num_samples={summary.get('num_samples')}")
    log_f.close()
    asr_f.close()


# =========================
# MAIN
# =========================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--result_root",
        type=str,
        required=True,
        help="Root directory to store results (will create syn/ and asr/ inside)"
    )

    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to Kaldi-style dataset directory (contains wav.scp, text, utt2spk). Used as reference voice pool."
    )

    p.add_argument(
        "--twist_txt",
        type=str,
        required=True,
        help="Path to tongue-twister .txt file. Each non-empty line is one prompt."
    )

    p.add_argument("--scheme", type=str, required=True, choices=sampling_schemes)
    p.add_argument("--asr", type=str, required=True,
                   help="ASR model tag, e.g. whisper-large-v3 or wav2vec2-large-960h")

    p.add_argument("--n_syn_per_line", type=int, default=DEFAULT_N_SYN_PER_LINE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)

    p.add_argument("--max_lines", type=int,
                   default=(DEFAULT_MAX_LINES if DEFAULT_MAX_LINES is not None else -1),
                   help="Max number of twist lines to evaluate (-1 = all).")

    p.add_argument("--max_refs", type=int,
                   default=(DEFAULT_MAX_REFS if DEFAULT_MAX_REFS is not None else -1),
                   help="Max number of kaldi ref utterances to load (-1 = all).")

    p.add_argument("--nisqa", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # IMPORTANT: seed before selecting refs so speaker-balanced subset is reproducible
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    max_lines = None if args.max_lines == -1 else args.max_lines
    max_refs = None if args.max_refs == -1 else args.max_refs

    RESULT_ROOT = Path(args.result_root)
    SYN_ROOT = RESULT_ROOT / "syn"
    ASR_ROOT = RESULT_ROOT / "asr"

    SYN_ROOT.mkdir(parents=True, exist_ok=True)
    ASR_ROOT.mkdir(parents=True, exist_ok=True)

    twist_txt = Path(args.twist_txt)
    print(f"Loading twist lines from {twist_txt}")
    twist_lines = load_twist_lines(twist_txt, max_lines=max_lines)
    print(f"Loaded {len(twist_lines)} twist lines")

    data_dir = Path(args.data_dir)
    print(f"Loading Kaldi ref dataset from {data_dir}")
    ref_examples = load_kaldi_dataset(data_dir, max_utts=max_refs)
    print(f"Loaded {len(ref_examples)} ref utterances")

    run_scheme(
        scheme=args.scheme,
        twist_lines=twist_lines,
        ref_examples=ref_examples,
        n_syn_per_line=args.n_syn_per_line,
        seed=args.seed,
        max_lines=max_lines,
        max_refs=max_refs,
        enable_nisqa=args.nisqa,
        asr_tag=args.asr
    )