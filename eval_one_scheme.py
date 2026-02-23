#!/usr/bin/env python3
import os
import re
import json
import time
import random
import ctypes
import argparse
import tempfile
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

import torch
import torchaudio
import torch.nn.functional as F
from speechbrain.pretrained import EncoderClassifier
import soundfile as sf
from tqdm import tqdm
from datasets import load_dataset, Audio
from jiwer import wer, cer
from jiwer import Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip
from num2words import num2words

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

# =========================
# ENV BOOTSTRAP (CRITICAL)
# =========================

os.environ["PATH"] = os.path.expanduser("~/.local/bin") + ":" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = os.path.expanduser("~/.local/lib") + ":" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["TORCHCODEC_DISABLE"] = "1"

ctypes.CDLL(os.path.expanduser("~/.local/lib/libespeak-ng.so"))

from neutts import NeuTTS  

# =========================
# CONFIG 
# =========================

SYN_ROOT = Path("/data2/minh_duc/neutts/libritts/infer100/syn.test.clean")
ASR_ROOT = Path("/data2/minh_duc/neutts/libritts/infer100/asr.test.clean")
ASR_TAG = "whisper-large-v3"

# default knobs (override via CLI)
DEFAULT_N_SYN_PER_UTT = 1
DEFAULT_SEED = 42
DEFAULT_MAX_UTTS: Optional[int] = 100  # set None for full

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Schemes allowed for TTS synthesis
sampling_schemes = ["orig", "ras_k50_win25", "dis", "ras_dis", "eas", 
                    "recon", "eas_dis", "hier", "ras_hier", "eas_hier"]  # add more here

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
# Speaker Similarity: WAV-LM for speaker verification
# =========================

class SpeakerSimilarity:
    def __init__(self, device="cuda"):
        self.device = device

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-base-plus-sv"
        )

        self.model = WavLMForXVector.from_pretrained(
            "microsoft/wavlm-base-plus-sv"
        ).to(device)

        self.model.eval()

    @torch.no_grad()
    def embedding(self, wav: torch.Tensor, sr: int):
        """
        wav: [T] float tensor
        """

        # resample to 16k (required)
        if sr != 16000:
            wav = torchaudio.functional.resample(
                wav.unsqueeze(0), sr, 16000
            ).squeeze(0)

        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        wav = wav.to(self.device)

        inputs = self.feature_extractor(
            wav.squeeze(0).cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )

        input_values = inputs.input_values.to(self.device)

        outputs = self.model(input_values)

        # xvector shape: [1, 512]
        if hasattr(outputs, "xvector"):
            emb = outputs.xvector.squeeze(0)
        elif hasattr(outputs, "embeddings"):
            emb = outputs.embeddings.squeeze(0)
        else:
            raise RuntimeError("Unexpected WavLM output format")

        # L2 normalize (important for cosine stability)
        emb = F.normalize(emb, p=2, dim=0)

        return emb

    @torch.no_grad()
    def similarity(self, wav_ref, sr_ref, wav_syn, sr_syn):
        emb_ref = self.embedding(wav_ref, sr_ref)
        emb_syn = self.embedding(wav_syn, sr_syn)

        sim = torch.dot(emb_ref, emb_syn)

        return float(sim.item())


# =========================
# UTILS
# =========================

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
    """
    Standard ASR normalization:
    - lowercase
    - remove punctuation
    - normalize numbers to words
    - collapse spaces
    """
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


def save_wav(path: Path, wav: torch.Tensor, sr: int) -> None:
    """
    wav: torch.Tensor shape [T] or [1, T] (float)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if wav.ndim == 2 and wav.shape[0] == 1:
        wav = wav.squeeze(0)
    sf.write(str(path), wav.detach().cpu().numpy(), sr)


def resample(wav: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    if orig_sr == new_sr:
        return wav
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)  # [1, T]
    wav_rs = torchaudio.functional.resample(wav, orig_freq=orig_sr, new_freq=new_sr)
    return wav_rs.squeeze(0)


def audio_duration_sec(wav: torch.Tensor, sr: int) -> float:
    n = int(wav.numel())
    return float(n) / float(sr)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log_msg(log_f, msg: str) -> None:
    ts = now_iso()
    log_f.write(f"[{ts}] {msg}\n")
    log_f.flush()


def cuda_sync_if_needed() -> None:
    if DEVICE.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


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


def make_key(utt_id: str, ref_id: str, syn_idx: int) -> str:
    return f"{utt_id}__ref={ref_id}__k={syn_idx}"


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
            "audio_16k": ex["audio"]["array"],  # numpy
        }
        for ex in ds
    ]


# =========================
# WHISPER ASR
# =========================

class WhisperASR:
    def __init__(self, tag: str):
        # currently tag controls directory naming; model id is fixed to whisper-large-v3
        self.tag = tag
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3"
        ).to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def transcribe_16k(self, wav_16k: torch.Tensor) -> str:
        """
        wav_16k: torch.Tensor [T], float, 16kHz
        """
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


# =========================
# NISQA
# =========================

class NISQAMetric:
    """
    Wrap torchmetrics NISQA. Returns 5 scores:
    overall MOS, noisiness, discontinuity, coloration, loudness
    """
    def __init__(self, sr: int):
        if not _HAS_NISQA:
            raise RuntimeError("torchmetrics NonIntrusiveSpeechQualityAssessment not available")
        self.sr = sr
        self.metric = NonIntrusiveSpeechQualityAssessment(sr).to(DEVICE)

    @torch.no_grad()
    def __call__(self, wav: torch.Tensor) -> Dict[str, float]:
        # expects [T] float
        wav = wav.to(DEVICE)
        scores = self.metric(wav)  # shape (5,)
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
    """
    Keeps running sums to support O(1) update without rereading results.jsonl.
    """
    existing = load_json(summary_path)
    if existing is not None:
        # back-compat: ensure required fields exist
        existing.setdefault("running", {})
        r = existing["running"]
        r.setdefault("count", 0)
        r.setdefault("sum_wer", 0.0)
        r.setdefault("sum_cer", 0.0)
        r.setdefault("sum_sim", 0.0)
        r.setdefault("count_sim", 0)
        r.setdefault("sum_rtf", 0.0)
        r.setdefault("count_rtf", 0)
        r.setdefault("nisqa", {"count": 0, "sum_overall": 0.0, "sum_noisiness": 0.0,
                               "sum_discontinuity": 0.0, "sum_coloration": 0.0, "sum_loudness": 0.0})
        r.setdefault("sum_gpu_peak_mb", 0.0)
        r.setdefault("count_gpu_peak", 0)
        existing.setdefault("meta", meta)
        return existing

    summary = {
        "meta": meta,
        "last_updated": now_iso(),
        "num_samples": 0,
        "avg_wer": None,
        "avg_cer": None,
        "avg_sim_score": None,
        "avg_rtf": None,
        "avg_nisqa_overall": None,
        "avg_nisqa_noisiness": None,
        "avg_nisqa_discontinuity": None,
        "avg_nisqa_coloration": None,
        "avg_nisqa_loudness": None,
        "running": {
            "count": 0,
            "sum_wer": 0.0,
            "sum_cer": 0.0,
            "sum_sim": 0.0,
            "count_sim": 0,
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
    return summary


def update_summary(summary: Dict[str, Any], record: Dict[str, Any]) -> None:
    r = summary["running"]
    r["count"] += 1
    r["sum_wer"] += float(record["wer"])
    r["sum_cer"] += float(record["cer"])

    sim = record.get("sim_score", None)
    if sim is not None:
        r["sum_sim"] += float(sim)
        r["count_sim"] += 1

    rtf = record.get("rtf", None)
    if rtf is not None:
        r["sum_rtf"] += float(rtf)
        r["count_rtf"] += 1

    gpu = record.get("gpu_peak_mem_mb", None)
    if gpu is not None:
        r["sum_gpu_peak_mb"] += float(gpu)
        r["count_gpu_peak"] += 1

    # NISQA
    if "nisqa_overall" in record:
        rn = r["nisqa"]
        rn["count"] += 1
        rn["sum_overall"] += float(record["nisqa_overall"])
        rn["sum_noisiness"] += float(record["nisqa_noisiness"])
        rn["sum_discontinuity"] += float(record["nisqa_discontinuity"])
        rn["sum_coloration"] += float(record["nisqa_coloration"])
        rn["sum_loudness"] += float(record["nisqa_loudness"])

    # publish averages
    summary["num_samples"] = r["count"]
    summary["avg_wer"] = r["sum_wer"] / r["count"]
    summary["avg_cer"] = r["sum_cer"] / r["count"]

    if r["count_sim"] > 0:
        summary["avg_sim_score"] = r["sum_sim"] / r["count_sim"]
    else:
        summary["avg_sim_score"] = None

    if r["count_rtf"] > 0:
        summary["avg_rtf"] = r["sum_rtf"] / r["count_rtf"]
    else:
        summary["avg_rtf"] = None

    if r["count_gpu_peak"] > 0:
        summary["avg_gpu_peak_mem_mb"] = (
            r["sum_gpu_peak_mb"] / r["count_gpu_peak"]
        )
    else:
        summary["avg_gpu_peak_mem_mb"] = None

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

def get_scheme_dirs(scheme: str) -> Tuple[Path, Path]:
    """
    Returns:
      syn_wav_dir: SYN_ROOT/<scheme>/wav
      asr_scheme_dir: ASR_ROOT/<ASR_TAG>/<scheme>
    """
    syn_wav_dir = SYN_ROOT / scheme / "wav"
    asr_scheme_dir = ASR_ROOT / ASR_TAG / scheme
    syn_wav_dir.mkdir(parents=True, exist_ok=True)
    asr_scheme_dir.mkdir(parents=True, exist_ok=True)
    return syn_wav_dir, asr_scheme_dir


# =========================
# MAIN EVAL (one scheme)
# =========================

def run_scheme(
    scheme: str,
    examples: List[Dict[str, Any]],
    n_syn_per_utt: int,
    seed: int,
    max_utts: Optional[int],
    enable_nisqa: bool,
) -> None:
    allowed = set(["ASR_GT"] + sampling_schemes)
    assert scheme in allowed, f"scheme={scheme} not in allowed={sorted(allowed)}"

    random.seed(seed)
    torch.manual_seed(seed)

    syn_wav_dir, asr_scheme_dir = get_scheme_dirs(scheme)

    # files
    log_f = open(asr_scheme_dir / "process.log", "a")
    results_path = asr_scheme_dir / "results.jsonl"
    asr_f = open(results_path, "a")
    done_path = asr_scheme_dir / "done_keys.txt"
    done = load_done_keys(done_path)

    # meta + summary
    meta = {
        "asr_model": ASR_TAG,
        "date_started": now_iso(),
        "scheme": scheme,
        "sampling_schemes_allowed": sampling_schemes,
        "n_syn_per_utt": n_syn_per_utt,
        "max_utts": max_utts,
        "seed": seed,
        "device": DEVICE,
        "notes": "Incremental one-scheme evaluation. Audio in SYN_ROOT; metrics in ASR_ROOT/ASR_TAG/scheme.",
        "syn_root": str(SYN_ROOT),
        "asr_root": str(ASR_ROOT),
    }
    summary_path = asr_scheme_dir / "summary.json"
    summary = init_or_load_summary(summary_path, meta)
    atomic_write_json(summary_path, summary)  # ensure exists early

    # group by speaker
    spk2utts = defaultdict(list)
    for ex in examples:
        spk2utts[ex["speaker_id"]].append(ex)

    # models
    asr = WhisperASR(tag=ASR_TAG)

    tts = None
    if scheme != "ASR_GT":
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

    # main loop
    for spk, utts in tqdm(spk2utts.items(), desc="Speakers"):
        if scheme != "ASR_GT" and len(utts) < 2:
            continue

        for ex in tqdm(utts, desc=f"Utts spk={spk}", leave=False):
            utt_id = ex["utt_id"]
            gt_text = ex["text"]
            try:
                wav_gt_16k = torch.tensor(ex["audio_16k"], dtype=torch.float32)

            except Exception as e:
                log_msg(log_f, f"FAIL ASR_GT utt={utt_id} : {e}")

            # For ASR_GT, we run exactly once per utt (no refs).
            if scheme == "ASR_GT":
                key = make_key(utt_id, ref_id="NONE", syn_idx=0)
                if key in done:
                    continue

                try:
                    wav_gt_16k = torch.tensor(ex["audio_16k"], dtype=torch.float32)

                    # Save GT audio for audit (optional but helpful)
                    wav_path = syn_wav_dir / f"{utt_id}.wav"
                    if not wav_path.exists():
                        save_wav(wav_path, wav_gt_16k, 16000)

                    # ASR
                    asr_pred = asr.transcribe_16k(wav_gt_16k)

                    gt_n = normalize_text(gt_text)
                    pred_n = normalize_text(asr_pred)

                    w = wer(gt_n, pred_n)
                    c = cer(gt_n, pred_n)

                    record = {
                        "key": key,
                        "scheme": scheme,
                        "utt_id": utt_id,
                        "speaker_id": spk,
                        "ref_utt_id": None,
                        "ref_text": None,
                        "gt_text": gt_text,
                        "asr_text": asr_pred,
                        "wer": float(w),
                        "cer": float(c),
                        "audio_path": str(wav_path),
                        "audio_sr": 16000,
                        "duration_sec": audio_duration_sec(wav_gt_16k, 16000),
                        "gen_time_sec": None,
                        "rtf": None,
                        "sim_score": None,
                        "gpu_peak_mem_mb": None,
                    }

                    asr_f.write(json.dumps(record) + "\n")
                    asr_f.flush()

                    append_done_key(done_path, key)
                    done.add(key)

                    update_summary(summary, record)
                    atomic_write_json(summary_path, summary)

                    log_msg(log_f, f"OK ASR_GT utt={utt_id} wer={w:.4f} cer={c:.4f}")

                except Exception as e:
                    log_msg(log_f, f"FAIL ASR_GT utt={utt_id} : {e}")

                continue  # next utt

            # ---- TTS schemes: pick refs from same speaker ----
            assert tts is not None

            # pick up to n_syn_per_utt distinct reference utterances
            ref_pool = [u for u in utts if u["utt_id"] != utt_id]
            if len(ref_pool) == 0:
                continue
            refs = random.sample(ref_pool, k=min(n_syn_per_utt, len(ref_pool)))

            for syn_idx, ref in enumerate(refs):
                ref_id = ref["utt_id"]
                key = make_key(utt_id, ref_id=ref_id, syn_idx=syn_idx)
                if key in done:
                    continue

                try:
                    # prepare ref wav on disk (NeuTTS wants path)
                    with tempfile.TemporaryDirectory() as td:
                        tmp_ref = Path(td) / "ref.wav"
                        # ref audio is 16k from HF
                        wav_ref_16k = torch.tensor(ref["audio_16k"], dtype=torch.float32)
                        save_wav(tmp_ref, wav_ref_16k, 16000)

                        # encode reference
                        ref_codes = tts.encode_reference(str(tmp_ref))

                    # TTS inference timing (wall-clock, with cuda sync)
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

                    # ensure tensor
                    wav_24k_t = torch.tensor(wav_24k, dtype=torch.float32)
                    if wav_24k_t.ndim > 1:
                        wav_24k_t = wav_24k_t.view(-1)

                    dur = audio_duration_sec(wav_24k_t, 24000)
                    rtf = gen_time / max(dur, 1e-6)

                    # save synthesized 24k for audit
                    wav_path = syn_wav_dir / f"{utt_id}__ref_{ref_id}__k{syn_idx}.wav"
                    save_wav(wav_path, wav_24k_t, 24000)

                    # ASR (needs 16k)
                    wav_16k_t = resample(wav_24k_t, orig_sr=24000, new_sr=16000)
                    asr_pred = asr.transcribe_16k(wav_16k_t)

                    gt_n = normalize_text(gt_text)
                    pred_n = normalize_text(asr_pred)

                    w = wer(gt_n, pred_n)
                    c = cer(gt_n, pred_n)

                    # SIM score
                    sim_score = 0.0
                    sim_score = sim_model.similarity(
                        wav_ref_16k,
                        16000,
                        wav_16k_t,
                        16000
                    )


                    record = {
                        "key": key,
                        "scheme": scheme,
                        "utt_id": utt_id,
                        "speaker_id": spk,
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
                        "sim_score": sim_score,
                        "gpu_peak_mem_mb": gpu_peak_mb,
                    }

                    # NISQA (optional)
                    if nisqa_metric is not None:
                        try:
                            record.update(nisqa_metric(wav_24k_t))
                        except Exception as ne:
                            # don't kill run if NISQA fails on a sample
                            log_msg(log_f, f"WARN NISQA failed key={key}: {ne}")

                    asr_f.write(json.dumps(record) + "\n")
                    asr_f.flush()

                    append_done_key(done_path, key)
                    done.add(key)

                    update_summary(summary, record)
                    atomic_write_json(summary_path, summary)

                    log_msg(
                        log_f,
                        f"OK {scheme} utt={utt_id} ref={ref_id} k={syn_idx} "
                        f"wer={w:.4f} cer={c:.4f} rtf={rtf:.3f}"
                    )

                except Exception as e:
                    log_msg(log_f, f"FAIL {scheme} utt={utt_id} ref={ref_id} k={syn_idx} : {e}")

    # finalize meta timestamps
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
    allowed = ["ASR_GT"] + sampling_schemes
    p = argparse.ArgumentParser()
    p.add_argument(
        "--scheme",
        type=str,
        required=True,
        help=f"One scheme to run. Allowed: {allowed}",
    )
    p.add_argument("--n_syn_per_utt", type=int, default=DEFAULT_N_SYN_PER_UTT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max_utts", type=int, default=(DEFAULT_MAX_UTTS if DEFAULT_MAX_UTTS is not None else -1))
    p.add_argument("--nisqa", action="store_true", help="Enable NISQA (requires torchmetrics).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    max_utts = None if args.max_utts == -1 else args.max_utts

    allowed = set(["ASR_GT"] + sampling_schemes)
    if args.scheme not in allowed:
        raise ValueError(f"--scheme {args.scheme} not in {sorted(allowed)}")

    print("Loading LibriTTS from HuggingFace...")
    examples = load_hf_libritts(max_utts=max_utts)
    print(f"Loaded {len(examples)} utterances")

    run_scheme(
        scheme=args.scheme,
        examples=examples,
        n_syn_per_utt=args.n_syn_per_utt,
        seed=args.seed,
        max_utts=max_utts,
        enable_nisqa=args.nisqa,
    )
