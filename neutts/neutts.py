import os
from typing import Generator
from pathlib import Path
import librosa
import numpy as np
import torch
import re
import platform
import glob
import warnings
from collections import deque
import math
import random
from phonemizer.backend import EspeakBackend
from neucodec import NeuCodec, DistillNeuCodec
from transformers import AutoTokenizer, AutoModelForCausalLM

from Discriminator import SegmentTokenDiscriminator
from Discriminator import StridedSegmentTokenDiscriminator


def _configure_espeak_library():
    """Auto-detect and configure espeak library on macOS."""
    if platform.system() != "Darwin":
        return  # Only needed on macOS

    # Common Homebrew installation paths
    search_paths = [
        "/opt/homebrew/Cellar/espeak/*/lib/libespeak.*.dylib",  # Apple Silicon
        "/usr/local/Cellar/espeak/*/lib/libespeak.*.dylib",  # Intel
        "/opt/homebrew/Cellar/espeak-ng/*/lib/libespeak-ng.*.dylib",  # Apple Silicon
        "/usr/local/Cellar/espeak-ng/*/lib/libespeak-ng.*.dylib",
    ]

    for pattern in search_paths:
        matches = glob.glob(pattern)
        if matches:
            try:
                from phonemizer.backend.espeak.wrapper import EspeakWrapper

                EspeakWrapper.set_library(matches[0])
                return
            except Exception:
                # If this fails, phonemizer will try its default detection
                pass


# Call before using phonemizer
_configure_espeak_library()


def _linear_overlap_add(frames: list[np.ndarray], stride: int) -> np.ndarray:
    # original impl --> https://github.com/facebookresearch/encodec/blob/main/encodec/utils.py
    assert len(frames)
    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]

    total_size = 0
    for i, frame in enumerate(frames):
        frame_end = stride * i + frame.shape[-1]
        total_size = max(total_size, frame_end)

    sum_weight = np.zeros(total_size, dtype=dtype)
    out = np.zeros(*shape, total_size, dtype=dtype)

    offset: int = 0
    for frame in frames:
        frame_length = frame.shape[-1]
        t = np.linspace(0, 1, frame_length + 2, dtype=dtype)[1:-1]
        weight = np.abs(0.5 - (t - 0.5))

        out[..., offset : offset + frame_length] += weight * frame
        sum_weight[offset : offset + frame_length] += weight
        offset += stride
    assert sum_weight.min() > 0
    return out / sum_weight


def _sample_top_k(
    logits: torch.Tensor,   # (B, vocab)
    top_k: int,
    temperature: float,
) -> torch.Tensor:          # (B, 1)

    if temperature != 1.0:
        logits = logits / temperature

    if top_k is not None and top_k > 0:
        values, indices = torch.topk(logits, top_k, dim=-1)
        probs = torch.softmax(values, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        return indices.gather(-1, sampled)
    else:
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


def nucleus_sampling(
    weighted_scores: torch.Tensor,
    top_p: float = 0.8,
    top_k: int = 25,
) -> int:
    probs = torch.softmax(weighted_scores, dim=0)
    sorted_probs, sorted_idx = probs.sort(descending=True, stable=True)

    cum_prob = 0.0
    kept_probs = []
    kept_idx = []

    for i in range(len(sorted_idx)):
        if cum_prob < top_p and len(kept_probs) < top_k:
            cum_prob += sorted_probs[i].item()
            kept_probs.append(sorted_probs[i])
            kept_idx.append(sorted_idx[i])
        else:
            break

    kept_probs = torch.stack(kept_probs)
    kept_idx = torch.stack(kept_idx)

    sampled = torch.multinomial(kept_probs, 1)
    return kept_idx[sampled].item()


def random_sampling(
    weighted_scores: torch.Tensor,
) -> int:
    probs = torch.softmax(weighted_scores, dim=0)
    return torch.multinomial(probs, 1).item()


def ras_sampling(
    weighted_scores: torch.Tensor,
    decoded_tokens: list[int],
    *,
    top_p: float = 0.8,
    top_k: int = 50,
    win_size: int = 20,
    tau_r: float = 0.1,
) -> int:
    """
    Repetition Aware Sampling (VALL-E 2 style)
    """
    top_id = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)

    if len(decoded_tokens) > 0:
        window = decoded_tokens[-win_size:]
        rep_num = (torch.tensor(window, device=weighted_scores.device) == top_id).sum().item()
        if rep_num >= win_size * tau_r:
            weighted_scores = weighted_scores.clone()
            weighted_scores[top_id] = -float("inf")
            top_id = random_sampling(weighted_scores)

    return top_id


class EASPenalty:
    """
    Multi-instance temporal penalty memory.

    - Each time a token appears in EAS top-k, a new instance is added.
    - Each instance decays exponentially.
    - Instances expire after `window` steps.
    - Total penalty per token is capped at `cap`.
    """

    def __init__(
        self,
        vocab_size: int,
        alpha: float = 0.2,
        beta: float = 0.7,
        window: int = 15,
        cap: float = 0.8,
        device: torch.device | None = None,
    ):
        self.alpha = alpha
        self.beta = beta
        self.window = window
        self.cap = cap
        self.device = device

        self.instances = []  # list of dicts: {id, rank, age}
        self.vocab_size = vocab_size

    def step(self):
        # Increase age
        for inst in self.instances:
            inst["age"] += 1

        # Remove expired
        self.instances = [
            inst for inst in self.instances
            if inst["age"] <= self.window
        ]

    def update_cluster(self, token_ids: torch.Tensor):
        """
        token_ids: tensor of top-k LM ids (ordered by rank)
        """
        for rank, tok in enumerate(token_ids.tolist()):
            self.instances.append({
                "id": tok,
                "rank": rank,
                "age": 0,
            })

    def build_penalty_vector(self) -> torch.Tensor:
        penalty = torch.zeros(self.vocab_size, device=self.device)

        for inst in self.instances:
            rank_scale = 1.0 / (1.0 + inst["rank"])
            value = self.alpha * rank_scale * (self.beta ** inst["age"])
            penalty[inst["id"]] += value

        # Hard cap per token
        penalty.clamp_(max=self.cap)

        return penalty

    def clone(self):
        """
        Deep copy of penalty memory state.
        """
        new = EASPenalty(
            vocab_size=self.vocab_size,
            alpha=self.alpha,
            beta=self.beta,
            window=self.window,
            cap=self.cap,
            device=self.device,
        )

        # Deep copy instances (list of small dicts)
        new.instances = [inst.copy() for inst in self.instances]

        return new


class NeuTTS:

    def __init__(
        self,
        backbone_repo="neuphonic/neutts-nano",
        backbone_device="cpu",
        codec_repo="neuphonic/neucodec",
        codec_device="cpu",
        use_dis=False,
        use_hier=False,
    ):

        # Consts
        self.sample_rate = 24_000
        self.max_context = 2048 # 2560
        self.hop_length = 480
        self.streaming_overlap_frames = 1
        self.streaming_frames_per_chunk = 25
        self.streaming_lookforward = 5
        self.streaming_lookback = 50
        self.streaming_stride_samples = self.streaming_frames_per_chunk * self.hop_length

        # ggml & onnx flags
        self._is_quantized_model = False
        self._is_onnx_codec = False

        # HF tokenizer
        self.tokenizer = None

        # Load phonemizer + models
        print("Loading phonemizer...")
        self.phonemizer = EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )

        self._load_backbone(backbone_repo, backbone_device)

        self._load_codec(codec_repo, codec_device)

        # Load watermarker (optional)
        try:
            import perth

            self.watermarker = perth.PerthImplicitWatermarker()
        except (ImportError, AttributeError) as e:
            warnings.warn(
                f"Perth watermarking unavailable: {e}. "
                "Audio will not be watermarked. "
                "Install with: pip install perth>=0.2.0"
            )
            self.watermarker = None

        if use_dis or use_hier:
            checkpoint_path = "/data2/minh_duc/TTS_spoofing/Segment_discriminator_len50/version_0/checkpoints/epoch=4-step=29675.ckpt"
            self.discriminator50 = SegmentTokenDiscriminator(segment_len=50,
                                    vocab_size=65536,
                                    d_model=256,
                                    nhead=8,
                                    num_layers=4,
                                    dim_feedforward=1024,
                                    dropout=0.1,
                                    )
            state = torch.load(checkpoint_path, map_location="cpu")
            self.discriminator50.load_state_dict(state["model_state_dict"])
            self.discriminator50.eval()
            self.discriminator50.to(self.backbone.device)

        if use_hier:
            checkpoint_path = "/data2/minh_duc/TTS_spoofing/Strided_discriminator_seg50_scale10/version_0/epochepoch=4.ckpt"
            self.discriminator50s10 = StridedSegmentTokenDiscriminator(
                                        segment_len=50,   # 50
                                        scale=10,               # 50 / 25 / 10
                                        vocab_size=65536,
                                        d_model=256,
                                        nhead=8,
                                        num_layers=4,
                                        dim_feedforward=1024,
                                        dropout=0.1,
                                        )
            state = torch.load(checkpoint_path, map_location="cpu")
            self.discriminator50s10.load_state_dict(state["model_state_dict"])
            self.discriminator50s10.eval()
            self.discriminator50s10.to(self.backbone.device)

            checkpoint_path = "/data2/minh_duc/TTS_spoofing/Strided_discriminator_seg50_scale25/version_0/epochepoch=4.ckpt"
            self.discriminator50s25 = StridedSegmentTokenDiscriminator(
                                        segment_len=50,   # 50
                                        scale=25,               # 50 / 25 / 10
                                        vocab_size=65536,
                                        d_model=256,
                                        nhead=8,
                                        num_layers=4,
                                        dim_feedforward=1024,
                                        dropout=0.1,
                                        )
            state = torch.load(checkpoint_path, map_location="cpu")
            self.discriminator50s25.load_state_dict(state["model_state_dict"])
            self.discriminator50s25.eval()
            self.discriminator50s25.to(self.backbone.device)

            checkpoint_path = "/data2/minh_duc/TTS_spoofing/Segment_discriminator_len25/version_0/epochepoch=4.ckpt"
            self.discriminator25 = SegmentTokenDiscriminator(segment_len=25,
                                    vocab_size=65536,
                                    d_model=256,
                                    nhead=8,
                                    num_layers=4,
                                    dim_feedforward=1024,
                                    dropout=0.1,
                                    )
            state = torch.load(checkpoint_path, map_location="cpu")
            self.discriminator25.load_state_dict(state["model_state_dict"])
            self.discriminator25.eval()
            self.discriminator25.to(self.backbone.device)

            checkpoint_path = "/data2/minh_duc/TTS_spoofing/Segment_discriminator_len10/version_0/epochepoch=4.ckpt"
            self.discriminator10 = SegmentTokenDiscriminator(segment_len=10,
                                    vocab_size=65536,
                                    d_model=256,
                                    nhead=8,
                                    num_layers=4,
                                    dim_feedforward=1024,
                                    dropout=0.1,
                                    )
            state = torch.load(checkpoint_path, map_location="cpu")
            self.discriminator10.load_state_dict(state["model_state_dict"])
            self.discriminator10.eval()
            self.discriminator10.to(self.backbone.device)

        # Speech token range (verified contiguous)
        self.speech_start_id = 128262
        self.speech_vocab_size = 65536
        self.speech_end_id = self.speech_start_id + self.speech_vocab_size # exclusive


    def _load_backbone(self, backbone_repo, backbone_device):
        print(f"Loading backbone from: {backbone_repo} on {backbone_device} ...")

        # GGUF loading
        if backbone_repo.endswith("gguf"):

            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise ImportError(
                    "Failed to import `llama_cpp`. "
                    "Please install it with:\n"
                    "    pip install llama-cpp-python"
                ) from e

            # If backbone_repo is a local file path, load it directly with llama.cpp
            if os.path.isfile(backbone_repo):
                self.backbone = Llama(
                    model_path=backbone_repo,
                    verbose=False,
                    n_gpu_layers=-1 if backbone_device == "gpu" else 0,
                    n_ctx=self.max_context,
                    mlock=True,
                    flash_attn=True if backbone_device == "gpu" else False,
                )
            else:
                # Fallback: treat it as a HF repo id (keeps original behavior if ever needed)
                self.backbone = Llama.from_pretrained(
                    repo_id=backbone_repo,
                    filename="*.gguf",
                    verbose=False,
                    n_gpu_layers=-1 if backbone_device == "gpu" else 0,
                    n_ctx=self.max_context,
                    mlock=True,
                    flash_attn=True if backbone_device == "gpu" else False,
                )

            self._is_quantized_model = True

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_repo)
            self.backbone = AutoModelForCausalLM.from_pretrained(backbone_repo).to(
                torch.device(backbone_device)
            )

    def _load_codec(self, codec_repo, codec_device):

        print(f"Loading codec from: {codec_repo} on {codec_device} ...")

        # 1) Local ONNX path (offline, recommended for embedded)
        if codec_repo.endswith(".onnx") and os.path.isfile(codec_repo):
            try:
                from neucodec import NeuCodecOnnxDecoder
            except ImportError as e:
                raise ImportError(
                    "Failed to import NeuCodecOnnxDecoder. "
                    "Make sure `neucodec` and `onnxruntime` are installed."
                ) from e

            self.codec = NeuCodecOnnxDecoder(codec_repo)
            self._is_onnx_codec = True

        # 2) Original HF-based behavior (use only if you really want remote download)
        match codec_repo:
            case "neuphonic/neucodec":
                self.codec = NeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(codec_device)
            case "neuphonic/distill-neucodec":
                self.codec = DistillNeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(codec_device)
            case "neuphonic/neucodec-onnx-decoder":

                if codec_device != "cpu":
                    raise ValueError("Onnx decoder only currently runs on CPU.")

                try:
                    from neucodec import NeuCodecOnnxDecoder
                except ImportError as e:
                    raise ImportError(
                        "Failed to import the onnx decoder."
                        " Ensure you have onnxruntime installed as well as neucodec >= 0.0.4."
                    ) from e

                self.codec = NeuCodecOnnxDecoder.from_pretrained(codec_repo)
                self._is_onnx_codec = True

            case _:
                raise ValueError(
                    "Invalid codec repo! Must be one of:"
                    " 'neuphonic/neucodec', 'neuphonic/distill-neucodec',"
                    " 'neuphonic/neucodec-onnx-decoder'."
                )

    def infer(self, text: str, ref_codes: np.ndarray | torch.Tensor, ref_text: str, 
              return_codes: bool = False, sampling_scheme: str = "orig") -> np.ndarray:
        """
        Perform inference to generate speech from text using the TTS model and reference audio.

        Args:
            text (str): Input text to be converted to speech.
            ref_codes (np.ndarray | torch.tensor): Encoded reference.
            ref_text (str): Reference text for reference audio. Defaults to None.
        Returns:
            np.ndarray: Generated speech waveform.
        """

        # Generate tokens
        if self._is_quantized_model:
            output_str = self._infer_ggml(ref_codes, ref_text, text)
        else:
            prompt_ids = self._apply_chat_template(ref_codes, ref_text, text)
            output_str = self._infer_torch(prompt_ids, sampling_scheme)

        # Decode
        wav = self._decode(output_str)
        watermarked_wav = (
            wav
            if self.watermarker is None
            else self.watermarker.apply_watermark(wav, sample_rate=24_000)
        )

        if return_codes:
            return watermarked_wav, output_str
        return watermarked_wav

    def infer_stream(
        self, text: str, ref_codes: np.ndarray | torch.Tensor, ref_text: str
    ) -> Generator[np.ndarray, None, None]:
        """
        Perform streaming inference to generate speech from
            text using the TTS model and reference audio.

        Args:
            text (str): Input text to be converted to speech.
            ref_codes (np.ndarray | torch.tensor): Encoded reference.
            ref_text (str): Reference text for reference audio. Defaults to None.
        Yields:
            np.ndarray: Generated speech waveform.
        """

        if self._is_quantized_model:
            return self._infer_stream_ggml(ref_codes, ref_text, text)

        else:
            raise NotImplementedError("Streaming is not implemented for the torch backend!")

    def encode_reference(self, ref_audio_path: str | Path):
        wav, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        with torch.no_grad():
            ref_codes = self.codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
        return ref_codes

    def _decode(self, codes: str):

        # Extract speech token IDs using regex
        speech_ids = [int(num) for num in re.findall(r"<\|speech_(\d+)\|>", codes)]

        if len(speech_ids) > 0:

            # Onnx decode
            if self._is_onnx_codec:
                codes = np.array(speech_ids, dtype=np.int32)[np.newaxis, np.newaxis, :]
                recon = self.codec.decode_code(codes)

            # Torch decode
            else:
                with torch.no_grad():
                    codes = torch.tensor(speech_ids, dtype=torch.long)[None, None, :].to(
                        self.codec.device
                    )
                    recon = self.codec.decode_code(codes).cpu().numpy()

            return recon[0, 0, :]
        else:
            raise ValueError("No valid speech tokens found in the output.")

    def _to_phones(self, text: str) -> str:
        phones = self.phonemizer.phonemize([text])
        phones = phones[0].split()
        phones = " ".join(phones)
        return phones

    def _apply_chat_template(
        self, ref_codes: list[int], ref_text: str, input_text: str
    ) -> list[int]:

        input_text = self._to_phones(ref_text) + " " + self._to_phones(input_text)
        speech_replace = self.tokenizer.convert_tokens_to_ids("<|SPEECH_REPLACE|>")
        speech_gen_start = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
        text_replace = self.tokenizer.convert_tokens_to_ids("<|TEXT_REPLACE|>")
        text_prompt_start = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_START|>")
        text_prompt_end = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_END|>")

        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        chat = """user: Convert the text to speech:<|TEXT_REPLACE|>\nassistant:<|SPEECH_REPLACE|>"""
        ids = self.tokenizer.encode(chat)

        text_replace_idx = ids.index(text_replace)
        ids = (
            ids[:text_replace_idx]
            + [text_prompt_start]
            + input_ids
            + [text_prompt_end]
            + ids[text_replace_idx + 1 :]  # noqa
        )

        speech_replace_idx = ids.index(speech_replace)
        codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes])
        codes = self.tokenizer.encode(codes_str, add_special_tokens=False)
        ids = ids[:speech_replace_idx] + [speech_gen_start] + list(codes)

        return ids
    
    @torch.no_grad()
    def _backbone_generate(
        self,
        prompt_tensor: torch.Tensor,
        *,
        max_length: int | None = None,        # keep old API
        max_new_tokens: int | None = None,    # new API 
        eos_token_id: int,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        use_cache: bool = True,
        min_new_tokens: int = 0,
        past_key_values=None,               
        return_pkv: bool = False,  
    ) -> torch.Tensor:
        """
        Explicit autoregressive generation for HF LLaMA-style models.

        Args:
            prompt_tensor: (1, T) full prefix tokens
            past_key_values: PKV corresponding to prompt_tensor (optional).
                            If provided, we will start by feeding only the last token.
            max_new_tokens: number of new tokens to generate (preferred)
            max_length: absolute total length cap (backward compatible)
        Returns:
            if return_pkv:
                (output_tokens, pkv)
            else:
                output_tokens
        """
        device = prompt_tensor.device
        input_ids = prompt_tensor
        pkv = past_key_values

        # Determine how many steps to run
        if max_new_tokens is not None:
            steps = max_new_tokens
        else:
            # Backward compatible path: max_length is total length cap
            if max_length is None:
                raise ValueError("Either max_new_tokens or max_length must be provided.")
            cur_len = input_ids.shape[1]
            steps = max(0, max_length - cur_len)

        generated = []

        for _ in range(steps):
            if pkv is None:
                outputs = self.backbone(
                    input_ids=input_ids,
                    use_cache=use_cache,
                )
            else:
                # With pkv, feed only the last token of the prefix/stream
                outputs = self.backbone(
                    input_ids=input_ids[:, -1:],
                    past_key_values=pkv,
                    use_cache=use_cache,
                )

            logits = outputs.logits[:, -1, :]  # (1, vocab)
            pkv = outputs.past_key_values

            if do_sample:
                logits = self._mask_to_speech_only(
                    logits.squeeze(0),
                    eos_token_id,
                ).unsqueeze(0)

                next_token = _sample_top_k(
                    logits,
                    top_k=top_k,
                    temperature=temperature,
                )
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            next_token = next_token.view(1, 1)
            generated.append(next_token)

            # EOS handling (respect min_new_tokens)
            if next_token.item() == eos_token_id and len(generated) >= min_new_tokens:
                input_ids = next_token
                break

            input_ids = next_token

        if generated:
            gen = torch.cat(generated, dim=1)                     # (1, N)
            out = torch.cat([prompt_tensor, gen], dim=1)          # (1, T+N)
        else:
            out = prompt_tensor

        if return_pkv:
            return out, pkv
        return out

    def _mask_to_speech_only(
        self,
        scores: torch.Tensor,     # (vocab,)
        eos_token_id: int,
    ) -> torch.Tensor:
        """
        Keep only speech tokens [speech_start_id, speech_end_id) and eos_token_id.
        Everything else -> -inf.
        """
        masked = scores.clone()
        masked[:] = -float("inf")

        # allow speech range
        masked[self.speech_start_id : self.speech_end_id] = scores[self.speech_start_id : self.speech_end_id]

        # allow EOS (may be just below speech_start_id like 128261)
        masked[eos_token_id] = scores[eos_token_id]

        return masked
    
    def _lm_to_speech_id_or_none(self, lm_id: int) -> int | None:
        if self.speech_start_id <= lm_id < self.speech_end_id:
            return lm_id - self.speech_start_id
        return None

    @torch.no_grad()
    def _ras_generate(
        self,
        prompt_tensor: torch.Tensor,
        *,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        eos_token_id: int,
        temperature: float = 1.0,
        use_cache: bool = True,
        min_new_tokens: int = 0,
        past_key_values=None,
        return_pkv: bool = False,
        # RAS params
        top_p: float = 0.8,
        top_k: int = 50,
        win_size: int = 20,
        tau_r: float = 0.1,
    ):
        device = prompt_tensor.device
        input_ids = prompt_tensor
        pkv = past_key_values

        generated = []
        decoded_speech_ids = []  # LOCAL ONLY

        # determine number of steps
        if max_new_tokens is not None:
            steps = max_new_tokens
        else:
            if max_length is None:
                raise ValueError("Either max_new_tokens or max_length must be provided.")
            cur_len = input_ids.shape[1]
            steps = max(0, max_length - cur_len)

        for _ in range(steps):

            if pkv is None:
                outputs = self.backbone(input_ids=input_ids, use_cache=use_cache)
            else:
                outputs = self.backbone(
                    input_ids=input_ids[:, -1:],
                    past_key_values=pkv,
                    use_cache=use_cache,
                )

            logits = outputs.logits[:, -1, :]
            pkv = outputs.past_key_values

            scores = logits.squeeze(0)

            if temperature != 1.0:
                scores = scores / max(temperature, 1e-8)

            scores = self._mask_to_speech_only(scores, eos_token_id=eos_token_id)

            lm_hist = [sid + self.speech_start_id for sid in decoded_speech_ids]

            next_id = ras_sampling(
                scores,
                lm_hist,
                top_p=top_p,
                top_k=top_k,
                win_size=win_size,
                tau_r=tau_r,
            )

            next_token = torch.tensor([[next_id]], device=device)
            generated.append(next_token)

            sid = self._lm_to_speech_id_or_none(next_id)
            if sid is not None:
                decoded_speech_ids.append(sid)

            if next_id == eos_token_id and len(generated) >= min_new_tokens:
                break

            input_ids = next_token

        if generated:
            gen = torch.cat(generated, dim=1)
            out = torch.cat([prompt_tensor, gen], dim=1)
        else:
            out = prompt_tensor

        if return_pkv:
            return out, pkv
        return out
        
    @torch.no_grad()
    def _eas_generate(
        self,
        prompt_tensor: torch.Tensor,
        *,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        eos_token_id: int,
        temperature: float = 1.0,
        use_cache: bool = True,
        min_new_tokens: int = 0,
        # EAS params
        eas_top_k: int = 3,     # cluster size
        alpha: float = 0.2,
        beta: float = 0.7,
        window: int = 15,
        cap: float = 0.8,
        top_p: float = 0.8,     # sampling nucleus
        sample_top_k: int = 25,
    ):
        """
        Entropy-Aware Sampling with multi-instance temporal penalty memory.
        """

        device = prompt_tensor.device
        input_ids = prompt_tensor
        past_key_values = None
        generated = []

        vocab_size = self.backbone.config.vocab_size

        penalty_memory = EASPenalty(
            vocab_size=vocab_size,
            alpha=alpha,
            beta=beta,
            window=window,
            cap=cap,
            device=device,
        )

        cur_len = input_ids.shape[1]

        # determine number of steps
        steps = 0
        if max_new_tokens is not None:
            steps = max_new_tokens
        else:
            if max_length is None:
                raise ValueError("Either max_new_tokens or max_length must be provided.")
            cur_len = input_ids.shape[1]
            steps = max(0, max_length - cur_len)

        for _ in range(steps):

            # ---- forward ----
            if past_key_values is None:
                outputs = self.backbone(input_ids=input_ids, use_cache=use_cache)
            else:
                outputs = self.backbone(
                    input_ids=input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )

            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            scores = logits.squeeze(0)

            if temperature != 1.0:
                scores = scores / temperature

            # ---- restrict to speech + EOS ----
            scores = self._mask_to_speech_only(scores, eos_token_id=eos_token_id)

            # ---- build penalty vector ----
            penalty = penalty_memory.build_penalty_vector()
            penalty[eos_token_id] = 0.0

            # ---- apply penalty ----
            penalized_scores = scores - penalty

            # ---- sampling (nucleus or hybrid) ----
            next_id = nucleus_sampling(
                penalized_scores,
                top_p=top_p,
                top_k=sample_top_k,
            )

            next_token = torch.tensor([[next_id]], device=device)
            generated.append(next_token)

            # stop
            if next_id == eos_token_id and len(generated) >= min_new_tokens:
                break

            # =====================================================
            # PENALTY UPDATE
            # =====================================================

            # 1) advance time
            penalty_memory.step()

            # 2) compute EAS cluster from penalized scores
            values, indices = torch.topk(penalized_scores, eas_top_k)

            # 3) update memory with new cluster
            penalty_memory.update_cluster(indices)

            input_ids = next_token

        if generated:
            gen = torch.cat(generated, dim=1)
            return torch.cat([prompt_tensor, gen], dim=1)
        else:
            return prompt_tensor

    @torch.no_grad()
    def _generate_chunk(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_id: int,
        temperature: float = 1.0,
        top_k: int = 50,
        past_key_values=None,
    ):
        generated = []
        pkv = past_key_values

        for _ in range(max_new_tokens):
            outputs = self.backbone(
                input_ids=input_ids if pkv is None else input_ids[:, -1:],
                past_key_values=pkv,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            pkv = outputs.past_key_values

            logits = logits / temperature
            if top_k > 0:
                v, idx = torch.topk(logits, top_k)
                probs = torch.softmax(v, dim=-1)
                next_id = idx[0, torch.multinomial(probs[0], 1)]
            else:
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1)

            next_id = next_id.view(1, 1)
            generated.append(next_id)
            input_ids = next_id

            if next_id.item() == eos_token_id:
                break

        if not generated:
            return None, past_key_values

        return torch.cat(generated, dim=1), pkv

    # Normal top-k sampling + discriminator guide chunk selections
    @torch.no_grad()
    def _dis_sampling(
        self,
        prompt_tensor: torch.Tensor,
        *,
        max_length: int,
        eos_token_id: int,
        temperature: float = 1.0,
        top_k: int = 50,
        warmup_len: int = 20,
        segment_len: int = 50,
        n_beams: int = 3,
        max_lm_steps: int = 400,   # safety cap per chunk
    ):
        device = prompt_tensor.device
        output = prompt_tensor

        # -------- warmup (LM tokens, no discriminator) --------
        warmup, _ = self._generate_chunk(
            output,
            max_new_tokens=warmup_len,
            eos_token_id=eos_token_id,
            temperature=temperature,
            top_k=top_k,
            past_key_values=None,  # TODO: later, thread pkv
        )
        if warmup is None:
            return output

        output = torch.cat([output, warmup], dim=1)
        if warmup[0, -1].item() == eos_token_id:
            return output

        # -------- discriminator-guided chunks --------
        while output.shape[1] < max_length:

            # Each candidate stores:
            # - lm_ids: list[int] tokens to append
            # - speech_ids: list[int] collected speech token ids (shifted)
            # - ended_with_eos: bool
            # - ended_by_cap: bool (hit max_lm_steps without eos and without reaching segment_len)
            candidates = []

            for _ in range(n_beams):
                input_ids = output
                past_key_values = None

                lm_ids = []
                speech_ids = []
                ended_with_eos = False
                ended_by_cap = False

                for step in range(max_lm_steps):
                    outputs = self.backbone(
                        input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    logits = outputs.logits[:, -1, :] / max(temperature, 1e-8)
                    past_key_values = outputs.past_key_values

                    if top_k > 0:
                        v, idx = torch.topk(logits, top_k)
                        probs = torch.softmax(v, dim=-1)
                        next_id = idx[0, torch.multinomial(probs[0], 1)].item()
                    else:
                        probs = torch.softmax(logits, dim=-1)
                        next_id = torch.multinomial(probs, 1).item()

                    lm_ids.append(next_id)
                    input_ids = torch.tensor([[next_id]], device=device)

                    # ---- EOS handling ----
                    if next_id == eos_token_id:
                        ended_with_eos = True
                        break

                    # ---- collect speech tokens only ----
                    if self.speech_start_id <= next_id < self.speech_end_id:
                        speech_ids.append(next_id - self.speech_start_id)
                        if len(speech_ids) == segment_len:
                            break

                # If we exited because we exhausted max_lm_steps without eos and without full segment
                if (not ended_with_eos) and (len(speech_ids) < segment_len) and (len(lm_ids) >= max_lm_steps):
                    ended_by_cap = True

                candidates.append(
                    {
                        "lm_ids": lm_ids,
                        "speech_ids": speech_ids,
                        "ended_with_eos": ended_with_eos,
                        "ended_by_cap": ended_by_cap,
                    }
                )

            # Partition candidates
            full = [c for c in candidates if len(c["speech_ids"]) == segment_len]
            partial = [c for c in candidates if len(c["speech_ids"]) < segment_len]  # includes eos + cap

            # ----------------------------
            # Your heuristic:
            # 1) if all < 50 -> choose longer one (by speech_len, tie-break lm_len)
            # 2) if only 1 == 50 -> continue with that one
            # 3) if >=2 == 50 -> discriminator compare
            # ----------------------------

            # Case 1: all less than segment_len (no full chunk)
            if len(full) == 0:
                if len(partial) == 0:
                    return output  # nothing generated at all (shouldn't happen)

                # choose longer partial: max speech_len, tie-break by lm_len
                best_partial = max(
                    partial,
                    key=lambda c: (len(c["speech_ids"]), len(c["lm_ids"]))
                )

                chosen_lm_ids = best_partial["lm_ids"]
                if not chosen_lm_ids:
                    return output

                chosen_lm_tensor = torch.tensor([chosen_lm_ids], device=device)
                output = torch.cat([output, chosen_lm_tensor], dim=1)

                # If EOS happened, we're done.
                # If ended_by_cap (no eos, no full segment), we also stop to avoid looping forever.
                if best_partial["ended_with_eos"] or best_partial["ended_by_cap"]:
                    return output[:, :max_length] if output.shape[1] > max_length else output

                # Otherwise (rare), keep going, but still guard max_length
                if output.shape[1] >= max_length:
                    return output[:, :max_length]
                continue

            # Case 2: exactly one full chunk -> continue with it (no discriminator)
            if len(full) == 1:
                chosen_lm_ids = full[0]["lm_ids"]
                if not chosen_lm_ids:
                    return output

                output = torch.cat([output, torch.tensor([chosen_lm_ids], device=device)], dim=1)
                if output.shape[1] >= max_length:
                    return output[:, :max_length]
                continue

            # Case 3: >=2 full chunks -> discriminator selection among full ones
            beam_speech_ids = [c["speech_ids"] for c in full]
            speech_tensor = torch.tensor(beam_speech_ids, device=device)  # [B, 50]
            scores = self.discriminator50(speech_tensor)
            best = scores.argmax().item()

            chosen_lm_ids = full[best]["lm_ids"]
            if not chosen_lm_ids:
                return output

            output = torch.cat([output, torch.tensor([chosen_lm_ids], device=device)], dim=1)

            if output.shape[1] >= max_length:
                return output[:, :max_length]

        return output[:, :max_length]

    # RAS sampling + discriminator guide chunk selections
    @torch.no_grad()
    def _ras_dis_sampling(
        self,
        prompt_tensor: torch.Tensor,
        *,
        max_length: int,
        eos_token_id: int,
        temperature: float = 1.0,
        warmup_len: int = 20,
        segment_len: int = 50,
        n_beams: int = 3,
        max_lm_steps: int = 400,   # safety cap per chunk
        # RAS params
        top_p: float = 0.8,
        top_k: int = 50,
        win_size: int = 25,
        tau_r: float = 0.1,
    ):
        device = prompt_tensor.device
        output = prompt_tensor

        # Rolling global repetition history (speech-id space) across the whole generation
        global_speech_hist: deque[int] = deque(maxlen=win_size)

        # -------- warmup (RAS, no discriminator) --------
        warmup_out = self._ras_generate(
            output,
            max_new_tokens=warmup_len,
            eos_token_id=eos_token_id,
            temperature=temperature,
            use_cache=True,
            min_new_tokens=warmup_len,  # force exact warmup_len
            top_p=top_p,
            top_k=top_k,
            win_size=win_size,
            tau_r=tau_r,
        )

        warmup = warmup_out[:, output.shape[1]:]
        if warmup.numel() == 0:
            return output

        output = torch.cat([output, warmup], dim=1)

        # Update global history with warmup speech tokens (exclude EOS)
        warmup_ids = warmup.squeeze(0).tolist()
        for tid in warmup_ids:
            if tid == eos_token_id:
                break
            if self.speech_start_id <= tid < self.speech_end_id:
                global_speech_hist.append(tid - self.speech_start_id)

        if output[0, -1].item() == eos_token_id:
            return output[:, :max_length]

        # -------- discriminator-guided chunks with RAS sampling --------
        while output.shape[1] < max_length:

            candidates = []

            for _ in range(n_beams):
                input_ids = output
                past_key_values = None

                lm_ids: list[int] = []
                speech_ids: list[int] = []  # speech-id space [0..65535]

                # Seed local history with global rolling history (speech-id space)
                decoded_speech_hist: list[int] = list(global_speech_hist)

                ended_with_eos = False
                ended_by_cap = False

                for _step in range(max_lm_steps):
                    outputs = self.backbone(
                        input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    logits = outputs.logits[:, -1, :]   # (1, vocab)
                    past_key_values = outputs.past_key_values

                    scores = logits.squeeze(0)
                    if temperature != 1.0:
                        scores = scores / max(temperature, 1e-8)

                    # restrict to speech range + EOS
                    scores = self._mask_to_speech_only(scores, eos_token_id=eos_token_id)

                    # RAS expects decoded tokens in LM-id space
                    decoded_lm_hist = [sid + self.speech_start_id for sid in decoded_speech_hist]

                    next_id = ras_sampling(
                        scores,
                        decoded_lm_hist,
                        top_p=top_p,
                        top_k=top_k,
                        win_size=win_size,
                        tau_r=tau_r,
                    )

                    lm_ids.append(next_id)
                    input_ids = torch.tensor([[next_id]], device=device)

                    # EOS handling
                    if next_id == eos_token_id:
                        ended_with_eos = True
                        break

                    # collect speech tokens (should always be speech due to mask, but keep safe)
                    if self.speech_start_id <= next_id < self.speech_end_id:
                        sid = next_id - self.speech_start_id
                        speech_ids.append(sid)

                        # update local hist (which already contains seeded global hist)
                        decoded_speech_hist.append(sid)
                        if len(decoded_speech_hist) > win_size:
                            decoded_speech_hist = decoded_speech_hist[-win_size:]

                        if len(speech_ids) == segment_len:
                            break

                if (not ended_with_eos) and (len(speech_ids) < segment_len) and (len(lm_ids) >= max_lm_steps):
                    ended_by_cap = True

                candidates.append(
                    {
                        "lm_ids": lm_ids,
                        "speech_ids": speech_ids,
                        "ended_with_eos": ended_with_eos,
                        "ended_by_cap": ended_by_cap,
                    }
                )

            # Partition candidates
            full = [c for c in candidates if len(c["speech_ids"]) == segment_len]
            partial = [c for c in candidates if len(c["speech_ids"]) < segment_len]  # includes EOS + cap

            # ---- SAME heuristic as your fixed dis_sampling ----

            # Case 1: no full chunk -> choose best partial (longer speech_len, tie-break lm_len)
            if len(full) == 0:
                if len(partial) == 0:
                    return output[:, :max_length]

                best_partial = max(partial, key=lambda c: (len(c["speech_ids"]), len(c["lm_ids"])))
                chosen_lm_ids = best_partial["lm_ids"]
                if not chosen_lm_ids:
                    return output[:, :max_length]

                output = torch.cat([output, torch.tensor([chosen_lm_ids], device=device)], dim=1)

                # Update global history with the chosen partial's speech ids
                for sid in best_partial["speech_ids"]:
                    global_speech_hist.append(sid)

                # stop if EOS or cap (prevents looping)
                if best_partial["ended_with_eos"] or best_partial["ended_by_cap"]:
                    return output[:, :max_length]

                if output.shape[1] >= max_length:
                    return output[:, :max_length]
                continue

            # Case 2: exactly one full chunk -> continue with it (no discriminator)
            if len(full) == 1:
                chosen = full[0]
                chosen_lm_ids = chosen["lm_ids"]
                if not chosen_lm_ids:
                    return output[:, :max_length]

                output = torch.cat([output, torch.tensor([chosen_lm_ids], device=device)], dim=1)

                # Update global history with chosen chunk speech ids
                for sid in chosen["speech_ids"]:
                    global_speech_hist.append(sid)

                if output.shape[1] >= max_length:
                    return output[:, :max_length]
                continue

            # Case 3: >=2 full chunks -> discriminator among full ones
            beam_speech_ids = [c["speech_ids"] for c in full]
            speech_tensor = torch.tensor(beam_speech_ids, device=device)  # [B, segment_len]
            disc_scores = self.discriminator50(speech_tensor)
            best = disc_scores.argmax().item()

            chosen = full[best]
            chosen_lm_ids = chosen["lm_ids"]
            if not chosen_lm_ids:
                return output[:, :max_length]

            output = torch.cat([output, torch.tensor([chosen_lm_ids], device=device)], dim=1)

            # Update global history with chosen chunk speech ids
            for sid in chosen["speech_ids"]:
                global_speech_hist.append(sid)

            if output.shape[1] >= max_length:
                return output[:, :max_length]

        return output[:, :max_length]

    @torch.no_grad()
    def _eas_dis_generate(
        self,
        prompt_tensor: torch.Tensor,
        *,
        max_length: int,
        eos_token_id: int,
        temperature: float = 1.0,
        warmup_len: int = 20,
        segment_len: int = 50,
        n_beams: int = 3,
        max_lm_steps: int = 400,
        # EAS params
        eas_top_k: int = 3,
        alpha: float = 0.2,
        beta: float = 0.7,
        window: int = 15,
        cap: float = 0.8,
        top_p: float = 0.8,
        sample_top_k: int = 25,
    ):
        device = prompt_tensor.device
        output = prompt_tensor

        vocab_size = self.backbone.config.vocab_size

        # ============================================================
        # NEW: global penalty memory that persists across chunks
        # ============================================================
        global_memory = EASPenalty(
            vocab_size=vocab_size,
            alpha=alpha,
            beta=beta,
            window=window,
            cap=cap,
            device=device,
        )

        # -------- warmup (no discriminator, use EAS) --------
        # Minimal change: do warmup here using *global_memory* (instead of _eas_generate),
        # so the warmup updates carry into later chunks.
        generated = []
        input_ids = output
        past_key_values = None

        for _ in range(warmup_len):
            outputs = self.backbone(
                input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            scores = logits.squeeze(0)
            if temperature != 1.0:
                scores = scores / temperature

            scores = self._mask_to_speech_only(scores, eos_token_id)

            penalty = global_memory.build_penalty_vector()
            penalty[eos_token_id] = 0.0
            penalized_scores = scores - penalty

            next_id = nucleus_sampling(
                penalized_scores,
                top_p=top_p,
                top_k=sample_top_k,
            )

            generated.append(next_id)
            input_ids = torch.tensor([[next_id]], device=device)

            if next_id == eos_token_id:
                break

            # ---- EAS update ----
            global_memory.step()
            _, indices = torch.topk(penalized_scores, eas_top_k)
            global_memory.update_cluster(indices)

        if generated:
            output = torch.cat([output, torch.tensor([generated], device=device)], dim=1)

        if output[0, -1].item() == eos_token_id:
            return output[:, :max_length]

        # -------- discriminator-guided chunks --------
        while output.shape[1] < max_length:

            candidates = []

            for _ in range(n_beams):
                input_ids = output
                past_key_values = None

                lm_ids: list[int] = []
                speech_ids: list[int] = []
                ended_with_eos = False
                ended_by_cap = False

                # ============================================================
                # NEW: per-beam memory starts as a COPY of global_memory state
                # (Requires EASPenalty.clone() method)
                # ============================================================
                penalty_memory = global_memory.clone()

                for _step in range(max_lm_steps):

                    outputs = self.backbone(
                        input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                    logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values

                    scores = logits.squeeze(0)
                    if temperature != 1.0:
                        scores = scores / temperature

                    scores = self._mask_to_speech_only(scores, eos_token_id)

                    penalty = penalty_memory.build_penalty_vector()
                    penalty[eos_token_id] = 0.0
                    penalized_scores = scores - penalty

                    next_id = nucleus_sampling(
                        penalized_scores,
                        top_p=top_p,
                        top_k=sample_top_k,
                    )

                    lm_ids.append(next_id)
                    input_ids = torch.tensor([[next_id]], device=device)

                    if next_id == eos_token_id:
                        ended_with_eos = True
                        break

                    if self.speech_start_id <= next_id < self.speech_end_id:
                        speech_ids.append(next_id - self.speech_start_id)
                        if len(speech_ids) == segment_len:
                            break

                    # ---- EAS update ----
                    penalty_memory.step()
                    _, indices = torch.topk(penalized_scores, eas_top_k)
                    penalty_memory.update_cluster(indices)

                if (not ended_with_eos) and (len(speech_ids) < segment_len) and (len(lm_ids) >= max_lm_steps):
                    ended_by_cap = True

                candidates.append(
                    {
                        "lm_ids": lm_ids,
                        "speech_ids": speech_ids,
                        "ended_with_eos": ended_with_eos,
                        "ended_by_cap": ended_by_cap,
                        # NEW: keep the resulting memory so we can commit it if chosen
                        "penalty_memory": penalty_memory,
                    }
                )

            full = [c for c in candidates if len(c["speech_ids"]) == segment_len]
            partial = [c for c in candidates if len(c["speech_ids"]) < segment_len]

            # Case 1: no full
            if len(full) == 0:
                if len(partial) == 0:
                    return output[:, :max_length]

                best_partial = max(partial, key=lambda c: (len(c["speech_ids"]), len(c["lm_ids"])))
                chosen_lm_ids = best_partial["lm_ids"]
                if not chosen_lm_ids:
                    return output[:, :max_length]

                output = torch.cat([output, torch.tensor([chosen_lm_ids], device=device)], dim=1)

                # NEW: commit global memory to the chosen candidate's state
                global_memory = best_partial["penalty_memory"]

                if best_partial["ended_with_eos"] or best_partial["ended_by_cap"]:
                    return output[:, :max_length]
                continue

            # Case 2: exactly one full
            if len(full) == 1:
                chosen = full[0]
                output = torch.cat([output, torch.tensor([chosen["lm_ids"]], device=device)], dim=1)

                # NEW: commit
                global_memory = chosen["penalty_memory"]

                if output.shape[1] >= max_length:
                    return output[:, :max_length]
                continue

            # Case 3: >=2 full → discriminator
            beam_speech_ids = [c["speech_ids"] for c in full]
            speech_tensor = torch.tensor(beam_speech_ids, device=device)
            disc_scores = self.discriminator50(speech_tensor)
            best = disc_scores.argmax().item()

            chosen = full[best]
            output = torch.cat([output, torch.tensor([chosen["lm_ids"]], device=device)], dim=1)

            # NEW: commit
            global_memory = chosen["penalty_memory"]

            if output.shape[1] >= max_length:
                return output[:, :max_length]

        return output[:, :max_length]
    
    @torch.no_grad()
    def _hier_generate(
        self,
        prompt_tensor: torch.Tensor,
        *,
        max_length: int,
        eos_token_id: int,
        temperature: float = 1.0,
        top_k: int = 50,
        warmup_len: int = 20,
        segment_len: int = 50,        # should be 50
        n_beams: int = 6,             # B0 (outer beams at start of chunk)
        inner_beams_10: int | None = None,  # B1
        inner_beams_25: int | None = None,  # B2
        max_lm_steps: int = 400,      # cap per candidate per chunk
        # Phase-A weights for the final 50 selection
        w50: float = 1.0,
        w50s25: float = 0.5,
        w50s10: float = 0.25,
    ):
        device = prompt_tensor.device
        output = prompt_tensor

        # -------- warmup (LM tokens, no discriminator) --------
        warmup, _ = self._generate_chunk(
            output,
            max_new_tokens=warmup_len,
            eos_token_id=eos_token_id,
            temperature=temperature,
            top_k=top_k,
            past_key_values=None,
        )
        if warmup is None:
            return output

        output = torch.cat([output, warmup], dim=1)
        if output[0, -1].item() == eos_token_id:
            return output[:, :max_length]

        # defaults for inner pruning sizes
        if inner_beams_10 is None:
            inner_beams_10 = max(2, math.ceil(n_beams / 2))
        if inner_beams_25 is None:
            inner_beams_25 = max(2, math.ceil(inner_beams_10 / 2))

        def _sample_next_id(logits_1vocab: torch.Tensor) -> int:
            # logits_1vocab: (1, vocab)
            scores = logits_1vocab.squeeze(0) / max(temperature, 1e-8)

            # critical: restrict to speech + EOS only
            scores = self._mask_to_speech_only(scores, eos_token_id=eos_token_id)

            if top_k > 0:
                v, idx = torch.topk(scores, top_k)
                probs = torch.softmax(v, dim=-1)
                next_id = idx[torch.multinomial(probs, 1)].item()
            else:
                probs = torch.softmax(scores, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
            return next_id

        def _advance_to_speech_len(c: dict, target_speech_len: int) -> dict:
            while (
                (not c["ended_with_eos"])
                and (not c["ended_by_cap"])
                and (len(c["speech_ids"]) < target_speech_len)
            ):
                if c["lm_steps_used"] >= max_lm_steps:
                    c["ended_by_cap"] = True
                    break

                outputs = self.backbone(
                    input_ids=c["input_ids"] if c["pkv"] is None else c["input_ids"][:, -1:],
                    past_key_values=c["pkv"],
                    use_cache=True,
                )
                logits = outputs.logits[:, -1, :]  # (1, vocab)
                c["pkv"] = outputs.past_key_values

                next_id = _sample_next_id(logits)

                c["lm_ids"].append(next_id)
                c["lm_steps_used"] += 1
                c["input_ids"] = torch.tensor([[next_id]], device=device)

                if next_id == eos_token_id:
                    c["ended_with_eos"] = True
                    break

                if self.speech_start_id <= next_id < self.speech_end_id:
                    c["speech_ids"].append(next_id - self.speech_start_id)

            return c

        def _best_longest(cands: list[dict]) -> dict:
            # choose longer partial: max speech_len, tie-break lm_len
            return max(cands, key=lambda c: (len(c["speech_ids"]), len(c["lm_ids"])))

        def _select_topk_by_disc(full: list[dict], disc_fn, k_keep: int, expected_len: int) -> list[dict]:
            # full all have expected_len speech_ids
            speech_tensor = torch.tensor([c["speech_ids"] for c in full], device=device)  # [B, expected_len]
            s = disc_fn(speech_tensor)
            k = min(k_keep, s.shape[0])
            idx = torch.topk(s, k=k, dim=0).indices.tolist()
            return [full[i] for i in idx]

        # -------- main loop: chunk by chunk --------
        while output.shape[1] < max_length:

            # ---- init candidates ----
            candidates = []
            for _ in range(n_beams):
                candidates.append(
                    {
                        "input_ids": output,   # first forward uses full prefix (baseline behavior)
                        "pkv": None,
                        "lm_ids": [],
                        "speech_ids": [],
                        "ended_with_eos": False,
                        "ended_by_cap": False,
                        "lm_steps_used": 0,
                    }
                )

            # ======================
            # Stage 10 (apply 3-case)
            # ======================
            for c in candidates:
                _advance_to_speech_len(c, target_speech_len=10)

            full10 = [c for c in candidates if len(c["speech_ids"]) == 10]
            partial10 = [c for c in candidates if len(c["speech_ids"]) < 10]

            if len(full10) == 0:
                best_partial = _best_longest(candidates)
                chosen = best_partial["lm_ids"]
                if not chosen:
                    return output[:, :max_length]
                output = torch.cat([output, torch.tensor([chosen], device=device)], dim=1)
                if best_partial["ended_with_eos"] or best_partial["ended_by_cap"] or output.shape[1] >= max_length:
                    return output[:, :max_length]
                continue

            if len(full10) == 1:
                survivors_10 = [full10[0]]
            else:
                survivors_10 = _select_topk_by_disc(full10, self.discriminator10, inner_beams_10, expected_len=10)

            # ======================
            # Stage 25 (apply 3-case)
            # ======================
            for c in survivors_10:
                _advance_to_speech_len(c, target_speech_len=25)

            full25 = [c for c in survivors_10 if len(c["speech_ids"]) == 25]
            partial25 = [c for c in survivors_10 if len(c["speech_ids"]) < 25]

            if len(full25) == 0:
                best_partial = _best_longest(survivors_10)
                chosen = best_partial["lm_ids"]
                if not chosen:
                    return output[:, :max_length]
                output = torch.cat([output, torch.tensor([chosen], device=device)], dim=1)
                if best_partial["ended_with_eos"] or best_partial["ended_by_cap"] or output.shape[1] >= max_length:
                    return output[:, :max_length]
                continue

            if len(full25) == 1:
                survivors_25 = [full25[0]]
            else:
                survivors_25 = _select_topk_by_disc(full25, self.discriminator25, inner_beams_25, expected_len=25)

            # ======================
            # Stage 50 (apply 3-case)
            # ======================
            for c in survivors_25:
                _advance_to_speech_len(c, target_speech_len=segment_len)

            full50 = [c for c in survivors_25 if len(c["speech_ids"]) == segment_len]
            partial50 = [c for c in survivors_25 if len(c["speech_ids"]) < segment_len]

            if len(full50) == 0:
                best_partial = _best_longest(survivors_25)
                chosen = best_partial["lm_ids"]
                if not chosen:
                    return output[:, :max_length]
                output = torch.cat([output, torch.tensor([chosen], device=device)], dim=1)
                if best_partial["ended_with_eos"] or best_partial["ended_by_cap"] or output.shape[1] >= max_length:
                    return output[:, :max_length]
                continue

            if len(full50) == 1:
                chosen = full50[0]["lm_ids"]
                if not chosen:
                    return output[:, :max_length]
                output = torch.cat([output, torch.tensor([chosen], device=device)], dim=1)
                if output.shape[1] >= max_length:
                    return output[:, :max_length]
                continue

            # >=2 full50 -> combined 50 discriminators
            speech_tensor = torch.tensor([c["speech_ids"] for c in full50], device=device)  # [B, 50]

            s50 = self.discriminator50(speech_tensor)
            s50s25 = self.discriminator50s25(speech_tensor)
            s50s10 = self.discriminator50s10(speech_tensor)

            p50 = torch.sigmoid(s50)
            p50s25 = torch.sigmoid(s50s25)
            p50s10 = torch.sigmoid(s50s10)

            combined = (w50 * p50) + (w50s25 * p50s25) + (w50s10 * p50s10)
            best = combined.argmax().item()

            chosen = full50[best]["lm_ids"]
            if not chosen:
                return output[:, :max_length]

            output = torch.cat([output, torch.tensor([chosen], device=device)], dim=1)
            if output.shape[1] >= max_length:
                return output[:, :max_length]

        return output[:, :max_length]

    @torch.no_grad()
    def _ras_hier_generate(
        self,
        prompt_tensor: torch.Tensor,
        *,
        max_length: int,
        eos_token_id: int,
        temperature: float = 1.0,
        top_k: int = 50,
        warmup_len: int = 20,
        segment_len: int = 50,
        n_beams: int = 6,
        inner_beams_10: int | None = None,
        inner_beams_25: int | None = None,
        max_lm_steps: int = 400,
        w50: float = 1.0,
        w50s25: float = 0.5,
        w50s10: float = 0.25,
        top_p: float = 0.8,
        win_size: int = 25,
        tau_r: float = 0.1,
    ):
        device = prompt_tensor.device
        output = prompt_tensor

        # =========================
        # NEW: global rolling RAS history (speech-id space)
        # =========================
        global_ras_hist: deque[int] = deque(maxlen=win_size)

        # -------- warmup (RAS, no discriminator) --------
        warmup_out = self._ras_generate(
            output,
            max_new_tokens=warmup_len,
            eos_token_id=eos_token_id,
            temperature=temperature,
            use_cache=True,
            min_new_tokens=warmup_len,
            top_p=top_p,
            top_k=top_k,
            win_size=win_size,
            tau_r=tau_r,
        )
        warmup = warmup_out[:, output.shape[1]:]
        if warmup is None or warmup.numel() == 0:
            return output

        output = torch.cat([output, warmup], dim=1)

        # =========================
        # NEW: update global history from warmup (exclude EOS)
        # =========================
        for tid in warmup.squeeze(0).tolist():
            if tid == eos_token_id:
                break
            if self.speech_start_id <= tid < self.speech_end_id:
                global_ras_hist.append(tid - self.speech_start_id)

        if output[0, -1].item() == eos_token_id:
            return output[:, :max_length]

        if inner_beams_10 is None:
            inner_beams_10 = max(2, math.ceil(n_beams / 2))
        if inner_beams_25 is None:
            inner_beams_25 = max(2, math.ceil(inner_beams_10 / 2))

        def _sample_next_id_ras(scores_1vocab: torch.Tensor, speech_hist_ids: list[int]) -> int:
            logits = scores_1vocab.squeeze(0)
            if temperature != 1.0:
                logits = logits / max(temperature, 1e-8)

            logits = self._mask_to_speech_only(logits, eos_token_id=eos_token_id)

            lm_hist = [sid + self.speech_start_id for sid in speech_hist_ids]
            return ras_sampling(
                logits,
                lm_hist,
                top_p=top_p,
                top_k=top_k,
                win_size=win_size,
                tau_r=tau_r,
            )

        def _advance_to_speech_len(c: dict, target_speech_len: int) -> dict:
            while (
                (not c["ended_with_eos"])
                and (not c["ended_by_cap"])
                and (len(c["speech_ids"]) < target_speech_len)
            ):
                if c["lm_steps_used"] >= max_lm_steps:
                    c["ended_by_cap"] = True
                    break

                outputs = self.backbone(
                    input_ids=c["input_ids"] if c["pkv"] is None else c["input_ids"][:, -1:],
                    past_key_values=c["pkv"],
                    use_cache=True,
                )
                logits = outputs.logits[:, -1, :]
                c["pkv"] = outputs.past_key_values

                # =========================
                # CHANGED: RAS uses c["ras_hist"] (global-seeded rolling history),
                # not c["speech_ids"] (chunk-only)
                # =========================
                next_id = _sample_next_id_ras(logits, c["ras_hist"])

                c["lm_ids"].append(next_id)
                c["lm_steps_used"] += 1
                c["input_ids"] = torch.tensor([[next_id]], device=device)

                if next_id == eos_token_id:
                    c["ended_with_eos"] = True
                    break

                if self.speech_start_id <= next_id < self.speech_end_id:
                    sid = next_id - self.speech_start_id

                    # chunk-local (for stage lengths + discriminator inputs)
                    c["speech_ids"].append(sid)

                    # rolling RAS history
                    c["ras_hist"].append(sid)
                    if len(c["ras_hist"]) > win_size:
                        c["ras_hist"] = c["ras_hist"][-win_size:]

            return c

        def _best_longest(cands: list[dict]) -> dict:
            return max(cands, key=lambda c: (len(c["speech_ids"]), len(c["lm_ids"])))

        def _select_topk_by_disc(full: list[dict], disc_fn, k_keep: int, expected_len: int) -> list[dict]:
            speech_tensor = torch.tensor([c["speech_ids"] for c in full], device=device)
            s = disc_fn(speech_tensor)
            k = min(k_keep, s.shape[0])
            idx = torch.topk(s, k=k, dim=0).indices.tolist()
            return [full[i] for i in idx]

        # -------- main loop: chunk by chunk --------
        while output.shape[1] < max_length:

            candidates = []
            for _ in range(n_beams):
                candidates.append(
                    {
                        "input_ids": output,
                        "pkv": None,
                        "lm_ids": [],
                        "speech_ids": [],                 # chunk-only
                        "ras_hist": list(global_ras_hist),# NEW: seeded from global rolling history
                        "ended_with_eos": False,
                        "ended_by_cap": False,
                        "lm_steps_used": 0,
                    }
                )

            # Stage 10
            for c in candidates:
                _advance_to_speech_len(c, target_speech_len=10)

            full10 = [c for c in candidates if len(c["speech_ids"]) == 10]
            partial10 = [c for c in candidates if len(c["speech_ids"]) < 10]

            if len(full10) == 0:
                best_partial = _best_longest(candidates)
                chosen = best_partial["lm_ids"]
                if not chosen:
                    return output[:, :max_length]
                output = torch.cat([output, torch.tensor([chosen], device=device)], dim=1)

                # NEW: update global history with chosen chunk speech ids
                for sid in best_partial["speech_ids"]:
                    global_ras_hist.append(sid)

                if best_partial["ended_with_eos"] or best_partial["ended_by_cap"] or output.shape[1] >= max_length:
                    return output[:, :max_length]
                continue

            if len(full10) == 1:
                survivors_10 = [full10[0]]
            else:
                survivors_10 = _select_topk_by_disc(full10, self.discriminator10, inner_beams_10, expected_len=10)

            # Stage 25
            for c in survivors_10:
                _advance_to_speech_len(c, target_speech_len=25)

            full25 = [c for c in survivors_10 if len(c["speech_ids"]) == 25]
            partial25 = [c for c in survivors_10 if len(c["speech_ids"]) < 25]

            if len(full25) == 0:
                best_partial = _best_longest(survivors_10)
                chosen = best_partial["lm_ids"]
                if not chosen:
                    return output[:, :max_length]
                output = torch.cat([output, torch.tensor([chosen], device=device)], dim=1)

                # NEW: update global history
                for sid in best_partial["speech_ids"]:
                    global_ras_hist.append(sid)

                if best_partial["ended_with_eos"] or best_partial["ended_by_cap"] or output.shape[1] >= max_length:
                    return output[:, :max_length]
                continue

            if len(full25) == 1:
                survivors_25 = [full25[0]]
            else:
                survivors_25 = _select_topk_by_disc(full25, self.discriminator25, inner_beams_25, expected_len=25)

            # Stage 50
            for c in survivors_25:
                _advance_to_speech_len(c, target_speech_len=segment_len)

            full50 = [c for c in survivors_25 if len(c["speech_ids"]) == segment_len]
            partial50 = [c for c in survivors_25 if len(c["speech_ids"]) < segment_len]

            if len(full50) == 0:
                best_partial = _best_longest(survivors_25)
                chosen = best_partial["lm_ids"]
                if not chosen:
                    return output[:, :max_length]
                output = torch.cat([output, torch.tensor([chosen], device=device)], dim=1)

                # NEW: update global history
                for sid in best_partial["speech_ids"]:
                    global_ras_hist.append(sid)

                if best_partial["ended_with_eos"] or best_partial["ended_by_cap"] or output.shape[1] >= max_length:
                    return output[:, :max_length]
                continue

            if len(full50) == 1:
                chosen_ids = full50[0]["lm_ids"]
                if not chosen_ids:
                    return output[:, :max_length]
                output = torch.cat([output, torch.tensor([chosen_ids], device=device)], dim=1)

                # NEW: update global history with selected full chunk
                for sid in full50[0]["speech_ids"]:
                    global_ras_hist.append(sid)

                if output.shape[1] >= max_length:
                    return output[:, :max_length]
                continue

            # >=2 full50 -> combined 50 discriminators (unchanged)
            speech_tensor = torch.tensor([c["speech_ids"] for c in full50], device=device)

            s50 = self.discriminator50(speech_tensor)
            s50s25 = self.discriminator50s25(speech_tensor)
            s50s10 = self.discriminator50s10(speech_tensor)

            p50 = torch.sigmoid(s50)
            p50s25 = torch.sigmoid(s50s25)
            p50s10 = torch.sigmoid(s50s10)

            combined = (w50 * p50) + (w50s25 * p50s25) + (w50s10 * p50s10)
            best = combined.argmax().item()

            chosen_ids = full50[best]["lm_ids"]
            if not chosen_ids:
                return output[:, :max_length]

            output = torch.cat([output, torch.tensor([chosen_ids], device=device)], dim=1)

            # NEW: update global history with selected best chunk
            for sid in full50[best]["speech_ids"]:
                global_ras_hist.append(sid)

            if output.shape[1] >= max_length:
                return output[:, :max_length]

        return output[:, :max_length]

    @torch.no_grad()
    def _eas_hier_generate(
        self,
        prompt_tensor: torch.Tensor,
        *,
        max_length: int,
        eos_token_id: int,
        temperature: float = 1.0,
        top_k: int = 50,  # kept for compatibility
        warmup_len: int = 20,
        segment_len: int = 50,
        n_beams: int = 6,
        inner_beams_10: int | None = None,
        inner_beams_25: int | None = None,
        max_lm_steps: int = 400,
        # Stage-50 weights
        w50: float = 1.0,
        w50s25: float = 0.5,
        w50s10: float = 0.25,
        # -------- EAS params --------
        eas_top_k: int = 3,
        alpha: float = 0.2,
        beta: float = 0.7,
        window: int = 15,
        cap: float = 0.8,
        top_p: float = 0.8,
        sample_top_k: int = 25,
    ):
        device = prompt_tensor.device
        output = prompt_tensor

        vocab_size = self.backbone.config.vocab_size

        # ============================================================
        # NEW: global EAS memory (persists across chunks)
        # ============================================================
        global_memory = EASPenalty(
            vocab_size=vocab_size,
            alpha=alpha,
            beta=beta,
            window=window,
            cap=cap,
            device=device,
        )

        # ============================================================
        # EAS sampling helper (logits -> next_id) + updates memory
        # ============================================================
        def _sample_next_id_eas(scores_1vocab: torch.Tensor, penalty_memory: EASPenalty) -> int:
            logits = scores_1vocab.squeeze(0)
            if temperature != 1.0:
                logits = logits / max(temperature, 1e-8)

            logits = self._mask_to_speech_only(logits, eos_token_id=eos_token_id)

            penalty = penalty_memory.build_penalty_vector()
            penalty[eos_token_id] = 0.0
            penalized_scores = logits - penalty

            next_id = nucleus_sampling(
                penalized_scores,
                top_p=top_p,
                top_k=sample_top_k,
            )

            # ----- penalty update -----
            penalty_memory.step()
            _, indices = torch.topk(penalized_scores, eas_top_k)
            penalty_memory.update_cluster(indices)

            return next_id

        # ============================================================
        # NEW: Warmup using global_memory (so warmup state carries over)
        # ============================================================
        warmup_generated: list[int] = []
        input_ids = output
        pkv = None
        for _ in range(warmup_len):
            outs = self.backbone(
                input_ids=input_ids if pkv is None else input_ids[:, -1:],
                past_key_values=pkv,
                use_cache=True,
            )
            logits = outs.logits[:, -1, :]
            pkv = outs.past_key_values

            next_id = _sample_next_id_eas(logits, global_memory)
            warmup_generated.append(next_id)
            input_ids = torch.tensor([[next_id]], device=device)

            if next_id == eos_token_id:
                break

        if warmup_generated:
            output = torch.cat([output, torch.tensor([warmup_generated], device=device)], dim=1)

        if output[0, -1].item() == eos_token_id:
            return output[:, :max_length]

        if inner_beams_10 is None:
            inner_beams_10 = max(2, math.ceil(n_beams / 2))
        if inner_beams_25 is None:
            inner_beams_25 = max(2, math.ceil(inner_beams_10 / 2))

        # ============================================================
        # Advance helper (PKV preserved exactly like hier)
        # ============================================================
        def _advance_to_speech_len(c: dict, target_speech_len: int) -> dict:
            while (
                (not c["ended_with_eos"])
                and (not c["ended_by_cap"])
                and (len(c["speech_ids"]) < target_speech_len)
            ):
                if c["lm_steps_used"] >= max_lm_steps:
                    c["ended_by_cap"] = True
                    break

                outs = self.backbone(
                    input_ids=c["input_ids"] if c["pkv"] is None else c["input_ids"][:, -1:],
                    past_key_values=c["pkv"],
                    use_cache=True,
                )
                logits = outs.logits[:, -1, :]
                c["pkv"] = outs.past_key_values

                next_id = _sample_next_id_eas(logits, c["penalty_memory"])

                c["lm_ids"].append(next_id)
                c["lm_steps_used"] += 1
                c["input_ids"] = torch.tensor([[next_id]], device=device)

                if next_id == eos_token_id:
                    c["ended_with_eos"] = True
                    break

                if self.speech_start_id <= next_id < self.speech_end_id:
                    c["speech_ids"].append(next_id - self.speech_start_id)

            return c

        def _best_longest(cands):
            return max(cands, key=lambda c: (len(c["speech_ids"]), len(c["lm_ids"])))

        def _select_topk_by_disc(full, disc_fn, k_keep):
            speech_tensor = torch.tensor([c["speech_ids"] for c in full], device=device)
            scores = disc_fn(speech_tensor)
            k = min(k_keep, scores.shape[0])
            idx = torch.topk(scores, k=k).indices.tolist()
            return [full[i] for i in idx]

        # ============================================================
        # Main chunk loop
        # ============================================================
        while output.shape[1] < max_length:

            candidates = []
            for _ in range(n_beams):
                candidates.append({
                    "input_ids": output,
                    "pkv": None,
                    "lm_ids": [],
                    "speech_ids": [],
                    "ended_with_eos": False,
                    "ended_by_cap": False,
                    "lm_steps_used": 0,
                    # NEW: seed each beam from global memory snapshot
                    "penalty_memory": global_memory.clone(),
                })

            # ======================
            # Stage 10
            # ======================
            for c in candidates:
                _advance_to_speech_len(c, 10)

            full10 = [c for c in candidates if len(c["speech_ids"]) == 10]
            if len(full10) == 0:
                best_cand = _best_longest(candidates)
                chosen = best_cand["lm_ids"]
                if not chosen:
                    return output[:, :max_length]
                output = torch.cat([output, torch.tensor([chosen], device=device)], dim=1)

                # NEW: commit global memory to chosen path
                global_memory = best_cand["penalty_memory"]

                if best_cand["ended_with_eos"] or best_cand["ended_by_cap"]:
                    return output[:, :max_length]
                continue

            survivors_10 = full10 if len(full10) == 1 else _select_topk_by_disc(full10, self.discriminator10, inner_beams_10)

            # ======================
            # Stage 25
            # ======================
            for c in survivors_10:
                _advance_to_speech_len(c, 25)

            full25 = [c for c in survivors_10 if len(c["speech_ids"]) == 25]
            if len(full25) == 0:
                best_cand = _best_longest(survivors_10)
                chosen = best_cand["lm_ids"]
                if not chosen:
                    return output[:, :max_length]
                output = torch.cat([output, torch.tensor([chosen], device=device)], dim=1)

                # NEW: commit
                global_memory = best_cand["penalty_memory"]

                continue

            survivors_25 = full25 if len(full25) == 1 else _select_topk_by_disc(full25, self.discriminator25, inner_beams_25)

            # ======================
            # Stage 50
            # ======================
            for c in survivors_25:
                _advance_to_speech_len(c, segment_len)

            full50 = [c for c in survivors_25 if len(c["speech_ids"]) == segment_len]

            if len(full50) == 0:
                best_cand = _best_longest(survivors_25)
                chosen_cand = best_cand
                chosen = best_cand["lm_ids"]
            elif len(full50) == 1:
                chosen_cand = full50[0]
                chosen = full50[0]["lm_ids"]
            else:
                speech_tensor = torch.tensor([c["speech_ids"] for c in full50], device=device)

                p50 = torch.sigmoid(self.discriminator50(speech_tensor))
                p50s25 = torch.sigmoid(self.discriminator50s25(speech_tensor))
                p50s10 = torch.sigmoid(self.discriminator50s10(speech_tensor))

                combined = (w50 * p50) + (w50s25 * p50s25) + (w50s10 * p50s10)
                best_idx = combined.argmax().item()
                chosen_cand = full50[best_idx]
                chosen = chosen_cand["lm_ids"]

            if not chosen:
                return output[:, :max_length]

            output = torch.cat([output, torch.tensor([chosen], device=device)], dim=1)

            # NEW: commit memory from the selected candidate
            global_memory = chosen_cand["penalty_memory"]

            if output.shape[1] >= max_length:
                return output[:, :max_length]

        return output[:, :max_length]

    def _rank_sum_select(self, p50, p50s25, p50s10):
        """
        Unweighted rank-sum (Borda) fusion.
        Lower rank is better (0 = best).
        Tie-break:
            1) better rank under p50
            2) higher raw p50 score
        Returns:
            best index (int)
        """
        scores = [p50, p50s25, p50s10]
        N = p50.size(0)

        ranks = []

        for s in scores:
            # Flatten in case shape is [B,1]
            s = s.view(-1)

            # Sort descending (higher prob = better)
            order = torch.argsort(s, descending=True)

            # Create rank tensor: 0 best, N-1 worst
            r = torch.empty_like(order)
            r[order] = torch.arange(N, device=s.device)

            ranks.append(r)

        # Unweighted rank sum
        total_rank = ranks[0] + ranks[1] + ranks[2]

        min_rank = total_rank.min()
        candidates = torch.where(total_rank == min_rank)[0]

        # If unique best, return directly
        if candidates.numel() == 1:
            return int(candidates.item())

        # Tie-break by discriminator50 rank
        rank50 = ranks[0][candidates]
        best_rank50 = rank50.min()
        candidates = candidates[rank50 == best_rank50]

        if candidates.numel() == 1:
            return int(candidates.item())

        # Final tie-break by raw p50 score
        best = candidates[torch.argmax(p50.view(-1)[candidates])]
        return int(best.item())
    
    @torch.no_grad()
    def _rank_hier_generate(
        self,
        prompt_tensor: torch.Tensor,
        *,
        max_length: int,
        eos_token_id: int,
        temperature: float = 1.0,
        top_k: int = 50,
        warmup_len: int = 20,
        segment_len: int = 50,        # should be 50
        n_beams: int = 6,             # B0 (outer beams at start of chunk)
        inner_beams_10: int | None = None,  # B1
        inner_beams_25: int | None = None,  # B2
        max_lm_steps: int = 400,      # cap per candidate per chunk
        # Phase-A weights for the final 50 selection
        w50: float = 1.0,
        w50s25: float = 0.5,
        w50s10: float = 0.25,
    ):
        device = prompt_tensor.device
        output = prompt_tensor

        # -------- warmup (LM tokens, no discriminator) --------
        warmup, _ = self._generate_chunk(
            output,
            max_new_tokens=warmup_len,
            eos_token_id=eos_token_id,
            temperature=temperature,
            top_k=top_k,
            past_key_values=None,
        )
        if warmup is None:
            return output

        output = torch.cat([output, warmup], dim=1)
        if output[0, -1].item() == eos_token_id:
            return output[:, :max_length]

        # defaults for inner pruning sizes
        if inner_beams_10 is None:
            inner_beams_10 = max(2, math.ceil(n_beams / 2))
        if inner_beams_25 is None:
            inner_beams_25 = max(2, math.ceil(inner_beams_10 / 2))

        def _sample_next_id(logits_1vocab: torch.Tensor) -> int:
            # logits_1vocab: (1, vocab)
            scores = logits_1vocab.squeeze(0) / max(temperature, 1e-8)

            # critical: restrict to speech + EOS only
            scores = self._mask_to_speech_only(scores, eos_token_id=eos_token_id)

            if top_k > 0:
                v, idx = torch.topk(scores, top_k)
                probs = torch.softmax(v, dim=-1)
                next_id = idx[torch.multinomial(probs, 1)].item()
            else:
                probs = torch.softmax(scores, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
            return next_id

        def _advance_to_speech_len(c: dict, target_speech_len: int) -> dict:
            while (
                (not c["ended_with_eos"])
                and (not c["ended_by_cap"])
                and (len(c["speech_ids"]) < target_speech_len)
            ):
                if c["lm_steps_used"] >= max_lm_steps:
                    c["ended_by_cap"] = True
                    break

                outputs = self.backbone(
                    input_ids=c["input_ids"] if c["pkv"] is None else c["input_ids"][:, -1:],
                    past_key_values=c["pkv"],
                    use_cache=True,
                )
                logits = outputs.logits[:, -1, :]  # (1, vocab)
                c["pkv"] = outputs.past_key_values

                next_id = _sample_next_id(logits)

                c["lm_ids"].append(next_id)
                c["lm_steps_used"] += 1
                c["input_ids"] = torch.tensor([[next_id]], device=device)

                if next_id == eos_token_id:
                    c["ended_with_eos"] = True
                    break

                if self.speech_start_id <= next_id < self.speech_end_id:
                    c["speech_ids"].append(next_id - self.speech_start_id)

            return c

        def _best_longest(cands: list[dict]) -> dict:
            # choose longer partial: max speech_len, tie-break lm_len
            return max(cands, key=lambda c: (len(c["speech_ids"]), len(c["lm_ids"])))

        def _select_topk_by_disc(full: list[dict], disc_fn, k_keep: int, expected_len: int) -> list[dict]:
            # full all have expected_len speech_ids
            speech_tensor = torch.tensor([c["speech_ids"] for c in full], device=device)  # [B, expected_len]
            s = disc_fn(speech_tensor)
            k = min(k_keep, s.shape[0])
            idx = torch.topk(s, k=k, dim=0).indices.tolist()
            return [full[i] for i in idx]

        # -------- main loop: chunk by chunk --------
        while output.shape[1] < max_length:

            # ---- init candidates ----
            candidates = []
            for _ in range(n_beams):
                candidates.append(
                    {
                        "input_ids": output,   # first forward uses full prefix (baseline behavior)
                        "pkv": None,
                        "lm_ids": [],
                        "speech_ids": [],
                        "ended_with_eos": False,
                        "ended_by_cap": False,
                        "lm_steps_used": 0,
                    }
                )

            # ======================
            # Stage 10 (apply 3-case)
            # ======================
            for c in candidates:
                _advance_to_speech_len(c, target_speech_len=10)

            full10 = [c for c in candidates if len(c["speech_ids"]) == 10]
            partial10 = [c for c in candidates if len(c["speech_ids"]) < 10]

            if len(full10) == 0:
                best_partial = _best_longest(candidates)
                chosen = best_partial["lm_ids"]
                if not chosen:
                    return output[:, :max_length]
                output = torch.cat([output, torch.tensor([chosen], device=device)], dim=1)
                if best_partial["ended_with_eos"] or best_partial["ended_by_cap"] or output.shape[1] >= max_length:
                    return output[:, :max_length]
                continue

            if len(full10) == 1:
                survivors_10 = [full10[0]]
            else:
                survivors_10 = _select_topk_by_disc(full10, self.discriminator10, inner_beams_10, expected_len=10)

            # ======================
            # Stage 25 (apply 3-case)
            # ======================
            for c in survivors_10:
                _advance_to_speech_len(c, target_speech_len=25)

            full25 = [c for c in survivors_10 if len(c["speech_ids"]) == 25]
            partial25 = [c for c in survivors_10 if len(c["speech_ids"]) < 25]

            if len(full25) == 0:
                best_partial = _best_longest(survivors_10)
                chosen = best_partial["lm_ids"]
                if not chosen:
                    return output[:, :max_length]
                output = torch.cat([output, torch.tensor([chosen], device=device)], dim=1)
                if best_partial["ended_with_eos"] or best_partial["ended_by_cap"] or output.shape[1] >= max_length:
                    return output[:, :max_length]
                continue

            if len(full25) == 1:
                survivors_25 = [full25[0]]
            else:
                survivors_25 = _select_topk_by_disc(full25, self.discriminator25, inner_beams_25, expected_len=25)

            # ======================
            # Stage 50 (apply 3-case)
            # ======================
            for c in survivors_25:
                _advance_to_speech_len(c, target_speech_len=segment_len)

            full50 = [c for c in survivors_25 if len(c["speech_ids"]) == segment_len]
            partial50 = [c for c in survivors_25 if len(c["speech_ids"]) < segment_len]

            if len(full50) == 0:
                best_partial = _best_longest(survivors_25)
                chosen = best_partial["lm_ids"]
                if not chosen:
                    return output[:, :max_length]
                output = torch.cat([output, torch.tensor([chosen], device=device)], dim=1)
                if best_partial["ended_with_eos"] or best_partial["ended_by_cap"] or output.shape[1] >= max_length:
                    return output[:, :max_length]
                continue

            if len(full50) == 1:
                chosen = full50[0]["lm_ids"]
                if not chosen:
                    return output[:, :max_length]
                output = torch.cat([output, torch.tensor([chosen], device=device)], dim=1)
                if output.shape[1] >= max_length:
                    return output[:, :max_length]
                continue

            # >=2 full50 -> combined 50 discriminators
            speech_tensor = torch.tensor([c["speech_ids"] for c in full50], device=device)  # [B, 50]

            s50 = self.discriminator50(speech_tensor)
            s50s25 = self.discriminator50s25(speech_tensor)
            s50s10 = self.discriminator50s10(speech_tensor)

            p50 = torch.sigmoid(s50)
            p50s25 = torch.sigmoid(s50s25)
            p50s10 = torch.sigmoid(s50s10)

            combined = (w50 * p50) + (w50s25 * p50s25) + (w50s10 * p50s10)
            best = self._rank_sum_select(p50, p50s25, p50s10)

            chosen = full50[best]["lm_ids"]
            if not chosen:
                return output[:, :max_length]

            output = torch.cat([output, torch.tensor([chosen], device=device)], dim=1)
            if output.shape[1] >= max_length:
                return output[:, :max_length]

        return output[:, :max_length]

    @torch.no_grad()
    def _rank_ras_hier_generate(
        self,
        prompt_tensor: torch.Tensor,
        *,
        max_length: int,
        eos_token_id: int,
        temperature: float = 1.0,
        top_k: int = 50,
        warmup_len: int = 20,
        segment_len: int = 50,
        n_beams: int = 6,
        inner_beams_10: int | None = None,
        inner_beams_25: int | None = None,
        max_lm_steps: int = 400,
        w50: float = 1.0,
        w50s25: float = 0.5,
        w50s10: float = 0.25,
        top_p: float = 0.8,
        win_size: int = 25,
        tau_r: float = 0.1,
    ):
        device = prompt_tensor.device
        output = prompt_tensor

        # =========================
        # NEW: global rolling RAS history (speech-id space)
        # =========================
        global_ras_hist: deque[int] = deque(maxlen=win_size)

        # -------- warmup (RAS, no discriminator) --------
        warmup_out = self._ras_generate(
            output,
            max_new_tokens=warmup_len,
            eos_token_id=eos_token_id,
            temperature=temperature,
            use_cache=True,
            min_new_tokens=warmup_len,
            top_p=top_p,
            top_k=top_k,
            win_size=win_size,
            tau_r=tau_r,
        )
        warmup = warmup_out[:, output.shape[1]:]
        if warmup is None or warmup.numel() == 0:
            return output

        output = torch.cat([output, warmup], dim=1)

        # =========================
        # NEW: update global history from warmup (exclude EOS)
        # =========================
        for tid in warmup.squeeze(0).tolist():
            if tid == eos_token_id:
                break
            if self.speech_start_id <= tid < self.speech_end_id:
                global_ras_hist.append(tid - self.speech_start_id)

        if output[0, -1].item() == eos_token_id:
            return output[:, :max_length]

        if inner_beams_10 is None:
            inner_beams_10 = max(2, math.ceil(n_beams / 2))
        if inner_beams_25 is None:
            inner_beams_25 = max(2, math.ceil(inner_beams_10 / 2))

        def _sample_next_id_ras(scores_1vocab: torch.Tensor, speech_hist_ids: list[int]) -> int:
            logits = scores_1vocab.squeeze(0)
            if temperature != 1.0:
                logits = logits / max(temperature, 1e-8)

            logits = self._mask_to_speech_only(logits, eos_token_id=eos_token_id)

            lm_hist = [sid + self.speech_start_id for sid in speech_hist_ids]
            return ras_sampling(
                logits,
                lm_hist,
                top_p=top_p,
                top_k=top_k,
                win_size=win_size,
                tau_r=tau_r,
            )

        def _advance_to_speech_len(c: dict, target_speech_len: int) -> dict:
            while (
                (not c["ended_with_eos"])
                and (not c["ended_by_cap"])
                and (len(c["speech_ids"]) < target_speech_len)
            ):
                if c["lm_steps_used"] >= max_lm_steps:
                    c["ended_by_cap"] = True
                    break

                outputs = self.backbone(
                    input_ids=c["input_ids"] if c["pkv"] is None else c["input_ids"][:, -1:],
                    past_key_values=c["pkv"],
                    use_cache=True,
                )
                logits = outputs.logits[:, -1, :]
                c["pkv"] = outputs.past_key_values

                # =========================
                # CHANGED: RAS uses c["ras_hist"] (global-seeded rolling history),
                # not c["speech_ids"] (chunk-only)
                # =========================
                next_id = _sample_next_id_ras(logits, c["ras_hist"])

                c["lm_ids"].append(next_id)
                c["lm_steps_used"] += 1
                c["input_ids"] = torch.tensor([[next_id]], device=device)

                if next_id == eos_token_id:
                    c["ended_with_eos"] = True
                    break

                if self.speech_start_id <= next_id < self.speech_end_id:
                    sid = next_id - self.speech_start_id

                    # chunk-local (for stage lengths + discriminator inputs)
                    c["speech_ids"].append(sid)

                    # rolling RAS history
                    c["ras_hist"].append(sid)
                    if len(c["ras_hist"]) > win_size:
                        c["ras_hist"] = c["ras_hist"][-win_size:]

            return c

        def _best_longest(cands: list[dict]) -> dict:
            return max(cands, key=lambda c: (len(c["speech_ids"]), len(c["lm_ids"])))

        def _select_topk_by_disc(full: list[dict], disc_fn, k_keep: int, expected_len: int) -> list[dict]:
            speech_tensor = torch.tensor([c["speech_ids"] for c in full], device=device)
            s = disc_fn(speech_tensor)
            k = min(k_keep, s.shape[0])
            idx = torch.topk(s, k=k, dim=0).indices.tolist()
            return [full[i] for i in idx]

        # -------- main loop: chunk by chunk --------
        while output.shape[1] < max_length:

            candidates = []
            for _ in range(n_beams):
                candidates.append(
                    {
                        "input_ids": output,
                        "pkv": None,
                        "lm_ids": [],
                        "speech_ids": [],                  # chunk-only
                        "ras_hist": list(global_ras_hist), # seeded from global rolling history
                        "ended_with_eos": False,
                        "ended_by_cap": False,
                        "lm_steps_used": 0,
                    }
                )

            # =========================
            # Stage 10
            # =========================
            for c in candidates:
                _advance_to_speech_len(c, target_speech_len=10)

            # NEW: EOS priority (stage 10)
            eos10 = [c for c in candidates if c["ended_with_eos"]]
            if len(eos10) >= 1:
                chosen_c = random.choice(eos10) if len(eos10) >= 2 else eos10[0]
                chosen_ids = chosen_c["lm_ids"]
                if not chosen_ids:
                    return output[:, :max_length]

                output = torch.cat([output, torch.tensor([chosen_ids], device=device)], dim=1)

                # update global history (exclude EOS already handled in speech_ids)
                for sid in chosen_c["speech_ids"]:
                    global_ras_hist.append(sid)

                return output[:, :max_length]

            full10 = [c for c in candidates if len(c["speech_ids"]) == 10]
            partial10 = [c for c in candidates if len(c["speech_ids"]) < 10]

            if len(full10) == 0:
                best_partial = _best_longest(candidates)
                chosen = best_partial["lm_ids"]
                if not chosen:
                    return output[:, :max_length]
                output = torch.cat([output, torch.tensor([chosen], device=device)], dim=1)

                for sid in best_partial["speech_ids"]:
                    global_ras_hist.append(sid)

                if best_partial["ended_with_eos"] or best_partial["ended_by_cap"] or output.shape[1] >= max_length:
                    return output[:, :max_length]
                continue

            if len(full10) == 1:
                survivors_10 = [full10[0]]
            else:
                survivors_10 = _select_topk_by_disc(full10, self.discriminator10, inner_beams_10, expected_len=10)

            # =========================
            # Stage 25
            # =========================
            for c in survivors_10:
                _advance_to_speech_len(c, target_speech_len=25)

            # NEW: EOS priority (stage 25)
            eos25 = [c for c in survivors_10 if c["ended_with_eos"]]
            if len(eos25) >= 1:
                chosen_c = random.choice(eos25) if len(eos25) >= 2 else eos25[0]
                chosen_ids = chosen_c["lm_ids"]
                if not chosen_ids:
                    return output[:, :max_length]

                output = torch.cat([output, torch.tensor([chosen_ids], device=device)], dim=1)

                for sid in chosen_c["speech_ids"]:
                    global_ras_hist.append(sid)

                return output[:, :max_length]

            full25 = [c for c in survivors_10 if len(c["speech_ids"]) == 25]
            partial25 = [c for c in survivors_10 if len(c["speech_ids"]) < 25]

            if len(full25) == 0:
                best_partial = _best_longest(survivors_10)
                chosen = best_partial["lm_ids"]
                if not chosen:
                    return output[:, :max_length]
                output = torch.cat([output, torch.tensor([chosen], device=device)], dim=1)

                for sid in best_partial["speech_ids"]:
                    global_ras_hist.append(sid)

                if best_partial["ended_with_eos"] or best_partial["ended_by_cap"] or output.shape[1] >= max_length:
                    return output[:, :max_length]
                continue

            if len(full25) == 1:
                survivors_25 = [full25[0]]
            else:
                survivors_25 = _select_topk_by_disc(full25, self.discriminator25, inner_beams_25, expected_len=25)

            # =========================
            # Stage 50
            # =========================
            for c in survivors_25:
                _advance_to_speech_len(c, target_speech_len=segment_len)

            # NEW: EOS priority (stage 50)
            eos50 = [c for c in survivors_25 if c["ended_with_eos"]]
            if len(eos50) >= 1:
                chosen_c = random.choice(eos50) if len(eos50) >= 2 else eos50[0]
                chosen_ids = chosen_c["lm_ids"]
                if not chosen_ids:
                    return output[:, :max_length]

                output = torch.cat([output, torch.tensor([chosen_ids], device=device)], dim=1)

                for sid in chosen_c["speech_ids"]:
                    global_ras_hist.append(sid)

                return output[:, :max_length]

            full50 = [c for c in survivors_25 if len(c["speech_ids"]) == segment_len]
            partial50 = [c for c in survivors_25 if len(c["speech_ids"]) < segment_len]

            if len(full50) == 0:
                best_partial = _best_longest(survivors_25)
                chosen = best_partial["lm_ids"]
                if not chosen:
                    return output[:, :max_length]
                output = torch.cat([output, torch.tensor([chosen], device=device)], dim=1)

                for sid in best_partial["speech_ids"]:
                    global_ras_hist.append(sid)

                if best_partial["ended_with_eos"] or best_partial["ended_by_cap"] or output.shape[1] >= max_length:
                    return output[:, :max_length]
                continue

            if len(full50) == 1:
                chosen_ids = full50[0]["lm_ids"]
                if not chosen_ids:
                    return output[:, :max_length]
                output = torch.cat([output, torch.tensor([chosen_ids], device=device)], dim=1)

                for sid in full50[0]["speech_ids"]:
                    global_ras_hist.append(sid)

                if output.shape[1] >= max_length:
                    return output[:, :max_length]
                continue

            # >=2 full50 and no eos -> do ranking
            speech_tensor = torch.tensor([c["speech_ids"] for c in full50], device=device)

            s50 = self.discriminator50(speech_tensor)
            s50s25 = self.discriminator50s25(speech_tensor)
            s50s10 = self.discriminator50s10(speech_tensor)

            p50 = torch.sigmoid(s50)
            p50s25 = torch.sigmoid(s50s25)
            p50s10 = torch.sigmoid(s50s10)

            best = self._rank_sum_select(p50, p50s25, p50s10)

            chosen_ids = full50[best]["lm_ids"]
            if not chosen_ids:
                return output[:, :max_length]

            output = torch.cat([output, torch.tensor([chosen_ids], device=device)], dim=1)

            for sid in full50[best]["speech_ids"]:
                global_ras_hist.append(sid)

            if output.shape[1] >= max_length:
                return output[:, :max_length]

        return output[:, :max_length]

    @torch.no_grad()
    def _rank_eas_hier_generate(
        self,
        prompt_tensor: torch.Tensor,
        *,
        max_length: int,
        eos_token_id: int,
        temperature: float = 1.0,
        top_k: int = 50,  # kept for compatibility
        warmup_len: int = 20,
        segment_len: int = 50,
        n_beams: int = 6,
        inner_beams_10: int | None = None,
        inner_beams_25: int | None = None,
        max_lm_steps: int = 400,
        # Stage-50 weights
        w50: float = 1.0,
        w50s25: float = 0.5,
        w50s10: float = 0.25,
        # -------- EAS params --------
        eas_top_k: int = 3,
        alpha: float = 0.2,
        beta: float = 0.7,
        window: int = 15,
        cap: float = 0.8,
        top_p: float = 0.8,
        sample_top_k: int = 25,
    ):
        device = prompt_tensor.device
        output = prompt_tensor

        vocab_size = self.backbone.config.vocab_size

        # ============================================================
        # Global EAS memory (persists across chunks)
        # ============================================================
        global_memory = EASPenalty(
            vocab_size=vocab_size,
            alpha=alpha,
            beta=beta,
            window=window,
            cap=cap,
            device=device,
        )

        # ============================================================
        # EAS sampling helper (logits -> next_id) + updates memory
        # Update rule: include BOTH (top-k cluster) and (sampled token)
        # ============================================================
        def _sample_next_id_eas(scores_1vocab: torch.Tensor, penalty_memory: EASPenalty) -> int:
            logits = scores_1vocab.squeeze(0)
            if temperature != 1.0:
                logits = logits / max(temperature, 1e-8)

            logits = self._mask_to_speech_only(logits, eos_token_id=eos_token_id)

            penalty = penalty_memory.build_penalty_vector()
            penalty[eos_token_id] = 0.0
            penalized_scores = logits - penalty

            # sample next token
            next_id = nucleus_sampling(
                penalized_scores,
                top_p=top_p,
                top_k=sample_top_k,
            )

            # ----- penalty update -----
            # 1) step/decay
            penalty_memory.step()

            # 2) update "cluster" from top-k of penalized scores
            #    NOTE: use eas_top_k here, not sample_top_k (by design)
            _, topk_ids = torch.topk(penalized_scores, k=eas_top_k)
            penalty_memory.update_cluster(topk_ids)

            # 3) ALSO update memory with the actually sampled token (trajectory-aware)
            #    We assume EASPenalty implements update_token(int) or update_tokens(tensor/list).
            #    If you don't have it yet, add it to EASPenalty (recommended).
            if next_id != eos_token_id:
                if hasattr(penalty_memory, "update_token"):
                    penalty_memory.update_token(next_id)
                elif hasattr(penalty_memory, "update_tokens"):
                    penalty_memory.update_tokens(torch.tensor([next_id], device=device))
                else:
                    # Fallback: treat sampled token as a singleton "cluster"
                    penalty_memory.update_cluster(torch.tensor([next_id], device=device))

            return next_id

        # ============================================================
        # Warmup using global_memory (so warmup state carries over)
        # ============================================================
        warmup_generated: list[int] = []
        input_ids = output
        pkv = None
        for _ in range(warmup_len):
            outs = self.backbone(
                input_ids=input_ids if pkv is None else input_ids[:, -1:],
                past_key_values=pkv,
                use_cache=True,
            )
            logits = outs.logits[:, -1, :]
            pkv = outs.past_key_values

            next_id = _sample_next_id_eas(logits, global_memory)
            warmup_generated.append(next_id)
            input_ids = torch.tensor([[next_id]], device=device)

            if next_id == eos_token_id:
                break

        if warmup_generated:
            output = torch.cat([output, torch.tensor([warmup_generated], device=device)], dim=1)

        if output[0, -1].item() == eos_token_id:
            return output[:, :max_length]

        if inner_beams_10 is None:
            inner_beams_10 = max(2, math.ceil(n_beams / 2))
        if inner_beams_25 is None:
            inner_beams_25 = max(2, math.ceil(inner_beams_10 / 2))

        # ============================================================
        # Advance helper (PKV preserved exactly like hier)
        # ============================================================
        def _advance_to_speech_len(c: dict, target_speech_len: int) -> dict:
            while (
                (not c["ended_with_eos"])
                and (not c["ended_by_cap"])
                and (len(c["speech_ids"]) < target_speech_len)
            ):
                if c["lm_steps_used"] >= max_lm_steps:
                    c["ended_by_cap"] = True
                    break

                outs = self.backbone(
                    input_ids=c["input_ids"] if c["pkv"] is None else c["input_ids"][:, -1:],
                    past_key_values=c["pkv"],
                    use_cache=True,
                )
                logits = outs.logits[:, -1, :]
                c["pkv"] = outs.past_key_values

                next_id = _sample_next_id_eas(logits, c["penalty_memory"])

                c["lm_ids"].append(next_id)
                c["lm_steps_used"] += 1
                c["input_ids"] = torch.tensor([[next_id]], device=device)

                if next_id == eos_token_id:
                    c["ended_with_eos"] = True
                    break

                if self.speech_start_id <= next_id < self.speech_end_id:
                    c["speech_ids"].append(next_id - self.speech_start_id)

            return c

        def _best_longest(cands):
            return max(cands, key=lambda c: (len(c["speech_ids"]), len(c["lm_ids"])))

        def _select_topk_by_disc(full, disc_fn, k_keep):
            speech_tensor = torch.tensor([c["speech_ids"] for c in full], device=device)
            scores = disc_fn(speech_tensor)
            k = min(k_keep, scores.shape[0])
            idx = torch.topk(scores, k=k).indices.tolist()
            return [full[i] for i in idx]

        # ============================================================
        # Main chunk loop  (EAS-HIER with EOS priority at ALL stages)
        # ============================================================
        while output.shape[1] < max_length:

            candidates = []
            for _ in range(n_beams):
                candidates.append({
                    "input_ids": output,
                    "pkv": None,
                    "lm_ids": [],
                    "speech_ids": [],
                    "ended_with_eos": False,
                    "ended_by_cap": False,
                    "lm_steps_used": 0,
                    # seed each beam from global memory snapshot
                    "penalty_memory": global_memory.clone(),
                })

            # ======================
            # Stage 10
            # ======================
            for c in candidates:
                _advance_to_speech_len(c, 10)

            # EOS priority (stage 10)
            eos10 = [c for c in candidates if c["ended_with_eos"]]
            if len(eos10) >= 1:
                chosen_cand = random.choice(eos10) if len(eos10) >= 2 else eos10[0]
                chosen = chosen_cand["lm_ids"]
                if not chosen:
                    return output[:, :max_length]
                output = torch.cat([output, torch.tensor([chosen], device=device)], dim=1)

                # commit memory from chosen EOS path
                global_memory = chosen_cand["penalty_memory"]
                return output[:, :max_length]

            full10 = [c for c in candidates if len(c["speech_ids"]) == 10]
            if len(full10) == 0:
                best_cand = _best_longest(candidates)
                chosen = best_cand["lm_ids"]
                if not chosen:
                    return output[:, :max_length]
                output = torch.cat([output, torch.tensor([chosen], device=device)], dim=1)

                # commit global memory to chosen path
                global_memory = best_cand["penalty_memory"]

                if best_cand["ended_with_eos"] or best_cand["ended_by_cap"]:
                    return output[:, :max_length]
                continue

            survivors_10 = full10 if len(full10) == 1 else _select_topk_by_disc(
                full10, self.discriminator10, inner_beams_10
            )

            # ======================
            # Stage 25
            # ======================
            for c in survivors_10:
                _advance_to_speech_len(c, 25)

            # EOS priority (stage 25)
            eos25 = [c for c in survivors_10 if c["ended_with_eos"]]
            if len(eos25) >= 1:
                chosen_cand = random.choice(eos25) if len(eos25) >= 2 else eos25[0]
                chosen = chosen_cand["lm_ids"]
                if not chosen:
                    return output[:, :max_length]
                output = torch.cat([output, torch.tensor([chosen], device=device)], dim=1)

                # commit memory from chosen EOS path
                global_memory = chosen_cand["penalty_memory"]
                return output[:, :max_length]

            full25 = [c for c in survivors_10 if len(c["speech_ids"]) == 25]
            if len(full25) == 0:
                best_cand = _best_longest(survivors_10)
                chosen = best_cand["lm_ids"]
                if not chosen:
                    return output[:, :max_length]
                output = torch.cat([output, torch.tensor([chosen], device=device)], dim=1)

                # commit
                global_memory = best_cand["penalty_memory"]
                continue

            survivors_25 = full25 if len(full25) == 1 else _select_topk_by_disc(
                full25, self.discriminator25, inner_beams_25
            )

            # ======================
            # Stage 50
            # ======================
            for c in survivors_25:
                _advance_to_speech_len(c, segment_len)

            # EOS priority (stage 50)
            eos50 = [c for c in survivors_25 if c["ended_with_eos"]]
            if len(eos50) >= 1:
                chosen_cand = random.choice(eos50) if len(eos50) >= 2 else eos50[0]
                chosen = chosen_cand["lm_ids"]
                if not chosen:
                    return output[:, :max_length]
                output = torch.cat([output, torch.tensor([chosen], device=device)], dim=1)

                # commit memory from chosen EOS path
                global_memory = chosen_cand["penalty_memory"]
                return output[:, :max_length]

            full50 = [c for c in survivors_25 if len(c["speech_ids"]) == segment_len]

            if len(full50) == 0:
                best_cand = _best_longest(survivors_25)
                chosen_cand = best_cand
                chosen = best_cand["lm_ids"]
            elif len(full50) == 1:
                chosen_cand = full50[0]
                chosen = full50[0]["lm_ids"]
            else:
                speech_tensor = torch.tensor([c["speech_ids"] for c in full50], device=device)

                p50 = torch.sigmoid(self.discriminator50(speech_tensor))
                p50s25 = torch.sigmoid(self.discriminator50s25(speech_tensor))
                p50s10 = torch.sigmoid(self.discriminator50s10(speech_tensor))

                best = self._rank_sum_select(p50, p50s25, p50s10)
                chosen_cand = full50[best]
                chosen = chosen_cand["lm_ids"]

            if not chosen:
                return output[:, :max_length]

            output = torch.cat([output, torch.tensor([chosen], device=device)], dim=1)

            # commit memory from the selected candidate
            global_memory = chosen_cand["penalty_memory"]

            if output.shape[1] >= max_length:
                return output[:, :max_length]

        return output[:, :max_length]

    def _infer_torch(self, prompt_ids: list[int], sampling_scheme: str = "orig") -> str:
        prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(self.backbone.device)
        speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        with torch.no_grad():
            if sampling_scheme == "orig":
                output_tokens = self.backbone.generate(
                    prompt_tensor,
                    max_length=self.max_context,
                    eos_token_id=speech_end_id,
                    do_sample=True,
                    temperature=1.0,
                    top_k=50,
                    use_cache=True,
                    min_new_tokens=50,
                )
            elif sampling_scheme == "recon":
                output_tokens = self._backbone_generate(
                    prompt_tensor,
                    max_length=self.max_context,
                    eos_token_id=speech_end_id,
                    do_sample=True,
                    temperature=1.0,
                    top_k=50,
                    use_cache=True,
                    min_new_tokens=50,
                )
            elif sampling_scheme == "ras_k50_win25":
                output_tokens = self._ras_generate(
                    prompt_tensor,
                    max_length=self.max_context,
                    eos_token_id=speech_end_id,
                    temperature=1.0,
                    min_new_tokens=50,
                    top_p=0.8,
                    top_k=50,
                    win_size=25,
                    tau_r=0.1,
                )
            elif sampling_scheme == "dis":
                output_tokens = self._dis_sampling(
                    prompt_tensor,
                    max_length=self.max_context,
                    eos_token_id=speech_end_id,
                    temperature=1.0,
                    top_k=50,
                    warmup_len=20,
                    segment_len=50,
                    n_beams=3,
                )
            elif sampling_scheme == "ras_dis":
                output_tokens = self._ras_dis_sampling(
                    prompt_tensor,
                    max_length=self.max_context,
                    eos_token_id=speech_end_id,
                    temperature=1.0,
                    warmup_len=20,
                    segment_len=50,
                    n_beams=3,
                    top_p=0.8,
                    top_k=50,
                    win_size=25,
                    tau_r=0.1,
                )
            elif sampling_scheme == "eas":
                output_tokens = self._eas_generate(
                    prompt_tensor,
                    max_length=self.max_context,
                    eos_token_id=speech_end_id,
                    temperature=1.0,
                    min_new_tokens=50,
                    eas_top_k=3,
                    alpha=0.18,
                    beta=0.8,
                    window=15,
                    cap=0.7,
                    top_p=0.8,
                    sample_top_k=50,
                )
            elif sampling_scheme == "eas_dis":
                output_tokens = self._eas_dis_generate(
                    prompt_tensor,
                    max_length=self.max_context,
                    eos_token_id=speech_end_id,
                    temperature=1.0,

                    warmup_len=20,
                    segment_len=50,
                    n_beams=3,

                    # EAS params
                    eas_top_k=3,
                    alpha=0.2,          # slightly smaller than eas-only
                    beta=0.8,
                    window=15,
                    cap=0.7,             # slightly softer cap than standalone eas
                    top_p=0.8,
                    sample_top_k=50,
                )
            elif sampling_scheme == "hier":
                output_tokens = self._hier_generate(
                    prompt_tensor,
                    max_length=self.max_context,
                    eos_token_id=speech_end_id,
                    temperature=1.0,
                    top_k=50,
                    warmup_len=20,
                    segment_len=50,
                    n_beams=8,          # try 6 or 8
                    inner_beams_10=5,   # e.g. keep 3 after disc10
                    inner_beams_25=3,   # keep 2 after disc25
                    max_lm_steps=400,
                    w50=1.0,
                    w50s25=0.0,
                    w50s10=0.0,
                )
            elif sampling_scheme == "ras_hier":
                output_tokens = self._ras_hier_generate(
                    prompt_tensor,
                    max_length=self.max_context,
                    eos_token_id=speech_end_id,

                    # === Base ===
                    temperature=1.0,
                    warmup_len=20,
                    segment_len=50,

                    # === Hierarchy ===
                    n_beams=8,
                    inner_beams_10=5,
                    inner_beams_25=3,
                    max_lm_steps=400,

                    # === Stage-50 weights (SAFE first) ===
                    w50=1.0,
                    w50s25=0.0,
                    w50s10=0.0,

                    # === RAS (MILD version first) ===
                    top_p=0.8,
                    top_k=50,
                    win_size=25,
                    tau_r=0.1,
                )
            elif sampling_scheme == "eas_hier":
                output_tokens = self._eas_hier_generate(
                    prompt_tensor,
                    max_length=self.max_context,
                    eos_token_id=speech_end_id,
                    temperature=1.0,

                    # ---- same hierarchy settings as baseline ----
                    warmup_len=20,
                    segment_len=50,
                    n_beams=8,
                    inner_beams_10=5,
                    inner_beams_25=3,
                    max_lm_steps=400,

                    w50=1.0,
                    w50s25=0.0,
                    w50s10=0.0,

                    # ---- EAS params (start mild) ----
                    eas_top_k=3,
                    alpha=0.2,          # slightly softer than standalone
                    beta=0.7,
                    window=15,
                    cap=0.8,
                    top_p=0.8,
                    sample_top_k=50,     # IMPORTANT: smaller than 50 for stability
                )
            elif sampling_scheme == "rank_hier":
                output_tokens = self._rank_hier_generate(
                    prompt_tensor,
                    max_length=self.max_context,
                    eos_token_id=speech_end_id,
                    temperature=1.0,
                    top_k=50,
                    warmup_len=20,
                    segment_len=50,
                    n_beams=8,          # try 6 or 8
                    inner_beams_10=5,   # e.g. keep 3 after disc10
                    inner_beams_25=3,   # keep 2 after disc25
                    max_lm_steps=400,
                    w50=1.0,
                    w50s25=0.0,
                    w50s10=0.0,
                )
            elif sampling_scheme == "rank_ras_hier":
                output_tokens = self._rank_ras_hier_generate(
                    prompt_tensor,
                    max_length=self.max_context,
                    eos_token_id=speech_end_id,

                    # === Base ===
                    temperature=1.0,
                    warmup_len=20,
                    segment_len=50,

                    # === Hierarchy ===
                    n_beams=8,
                    inner_beams_10=5,
                    inner_beams_25=3,
                    max_lm_steps=400,

                    # === Stage-50 weights (SAFE first) ===
                    w50=1.0,
                    w50s25=0.0,
                    w50s10=0.0,

                    # === RAS (MILD version first) ===
                    top_p=0.8,
                    top_k=50,
                    win_size=25,
                    tau_r=0.1,
                )
            elif sampling_scheme == "rank_eas_hier":
                output_tokens = self._rank_eas_hier_generate(
                    prompt_tensor,
                    max_length=self.max_context,
                    eos_token_id=speech_end_id,
                    temperature=1.0,

                    # ---- same hierarchy settings as baseline ----
                    warmup_len=20,
                    segment_len=50,
                    n_beams=8,
                    inner_beams_10=5,
                    inner_beams_25=3,
                    max_lm_steps=400,

                    w50=1.0,
                    w50s25=0.0,
                    w50s10=0.0,

                    # ---- EAS params (start mild) ----
                    eas_top_k=3,
                    alpha=0.2,          # slightly softer than standalone
                    beta=0.7,
                    window=15,
                    cap=0.8,
                    top_p=0.8,
                    sample_top_k=50,     # IMPORTANT: smaller than 50 for stability
                )

        input_length = prompt_tensor.shape[-1]
        output_str = self.tokenizer.decode(
            output_tokens[0, input_length:].cpu().numpy().tolist(), add_special_tokens=False
        )
        return output_str

    def _infer_ggml(self, ref_codes: list[int], ref_text: str, input_text: str) -> str:
        ref_text = self._to_phones(ref_text)
        input_text = self._to_phones(input_text)

        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])
        prompt = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text} {input_text}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )
        output = self.backbone(
            prompt,
            max_tokens=self.max_context,
            temperature=1.0,
            top_k=50,
            stop=["<|SPEECH_GENERATION_END|>"],
        )
        output_str = output["choices"][0]["text"]
        return output_str

    def _infer_stream_ggml(
        self, ref_codes: torch.Tensor, ref_text: str, input_text: str
    ) -> Generator[np.ndarray, None, None]:
        ref_text = self._to_phones(ref_text)
        input_text = self._to_phones(input_text)

        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])
        prompt = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text} {input_text}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )

        audio_cache: list[np.ndarray] = []
        token_cache: list[str] = [f"<|speech_{idx}|>" for idx in ref_codes]
        n_decoded_samples: int = 0
        n_decoded_tokens: int = len(ref_codes)

        for item in self.backbone(
            prompt,
            max_tokens=self.max_context,
            temperature=1.0,
            top_k=50,
            stop=["<|SPEECH_GENERATION_END|>"],
            stream=True,
        ):
            output_str = item["choices"][0]["text"]
            token_cache.append(output_str)

            if (
                len(token_cache[n_decoded_tokens:])
                >= self.streaming_frames_per_chunk + self.streaming_lookforward
            ):

                # decode chunk
                tokens_start = max(
                    n_decoded_tokens - self.streaming_lookback - self.streaming_overlap_frames, 0
                )
                tokens_end = (
                    n_decoded_tokens
                    + self.streaming_frames_per_chunk
                    + self.streaming_lookforward
                    + self.streaming_overlap_frames
                )
                sample_start = (n_decoded_tokens - tokens_start) * self.hop_length
                sample_end = (
                    sample_start
                    + (self.streaming_frames_per_chunk + 2 * self.streaming_overlap_frames)
                    * self.hop_length
                )
                curr_codes = token_cache[tokens_start:tokens_end]
                recon = self._decode("".join(curr_codes))
                recon = (
                    recon
                    if self.watermarker is None
                    else self.watermarker.apply_watermark(recon, sample_rate=24_000)
                )
                recon = recon[sample_start:sample_end]
                audio_cache.append(recon)

                # postprocess
                processed_recon = _linear_overlap_add(
                    audio_cache, stride=self.streaming_stride_samples
                )
                new_samples_end = len(audio_cache) * self.streaming_stride_samples
                processed_recon = processed_recon[n_decoded_samples:new_samples_end]
                n_decoded_samples = new_samples_end
                n_decoded_tokens += self.streaming_frames_per_chunk
                yield processed_recon

        # final decoding handled seperately as non-constant chunk size
        remaining_tokens = len(token_cache) - n_decoded_tokens
        if len(token_cache) > n_decoded_tokens:
            tokens_start = max(
                len(token_cache)
                - (self.streaming_lookback + self.streaming_overlap_frames + remaining_tokens),
                0,
            )
            sample_start = (
                len(token_cache) - tokens_start - remaining_tokens - self.streaming_overlap_frames
            ) * self.hop_length
            curr_codes = token_cache[tokens_start:]
            recon = self._decode("".join(curr_codes))
            recon = (
                recon
                if self.watermarker is None
                else self.watermarker.apply_watermark(recon, sample_rate=24_000)
            )
            recon = recon[sample_start:]
            audio_cache.append(recon)

            processed_recon = _linear_overlap_add(audio_cache, stride=self.streaming_stride_samples)
            processed_recon = processed_recon[n_decoded_samples:]
            yield processed_recon

    def infer_distribution_gap(
        self,
        text: str,
        ref_text: str,
        ref_path: str | Path,
        gt_path: str | Path,
        sampling_scheme: str = "orig",
    ):
        """
        Return:
            ys: synthetic speech token list
            yg: ground-truth speech token list
            Hs_speech: [len(ys), hidden_dim]
            Hg_speech: [len(yg), hidden_dim]
        """

        device = next(self.codec.parameters()).device

        # ============================================================
        # 1) Encode reference speech (for conditioning prompt)
        # ============================================================
        ref_codes = self.encode_reference(ref_path)
        if torch.is_tensor(ref_codes):
            ref_codes_list = ref_codes.cpu().tolist()
        else:
            ref_codes_list = ref_codes

        # ============================================================
        # 2) Generate synthetic speech tokens
        # ============================================================
        ys = self.infer(
            text=text,
            ref_codes=ref_codes,
            ref_text=ref_text,
            return_codes=True,
            sampling_scheme=sampling_scheme,
        )

        if isinstance(ys, np.ndarray):
            ys = ys.tolist()
        elif torch.is_tensor(ys):
            ys = ys.cpu().tolist()

        # ============================================================
        # 3) Encode ground-truth speech tokens
        # ============================================================
        yg = self.encode_reference(gt_path)
        if torch.is_tensor(yg):
            yg = yg.cpu().tolist()

        # ============================================================
        # Helper to build embedding prompt and extract speech hidden
        # ============================================================
        def encode_and_extract(speech_codes: list[int]):

            # Build prefix (text + reference codes)
            prefix_ids = self._apply_chat_template(
                ref_codes=ref_codes_list,
                ref_text=ref_text,
                input_text=text,
            )

            # Convert speech codes into token ids
            codes_str = "".join([f"<|speech_{i}|>" for i in speech_codes])
            speech_ids = self.tokenizer.encode(
                codes_str,
                add_special_tokens=False
            )

            input_ids = prefix_ids + speech_ids

            speech_start = len(prefix_ids)
            speech_end = speech_start + len(speech_ids)

            input_tensor = torch.tensor(
                input_ids,
                dtype=torch.long,
                device=device
            ).unsqueeze(0)

            with torch.no_grad():
                outputs = self.backbone(
                    input_ids=input_tensor,
                    output_hidden_states=True,
                    use_cache=False,
                )

            hidden = outputs.hidden_states[-1]  # [1, L, D]

            # 🔥 CHOP HERE
            speech_hidden = hidden[:, speech_start:speech_end, :]  # [1, Len, D]

            return speech_hidden.squeeze(0)  # [Len, D]

        # ============================================================
        # 4) Extract both
        # ============================================================
        Hs_speech = encode_and_extract(ys)
        Hg_speech = encode_and_extract(yg)

        # ============================================================
        # 5) Return ready-to-use tensors
        # ============================================================
        return {
            "ys": ys,
            "yg": yg,
            "Hs_speech": Hs_speech,   # [len(ys), D]
            "Hg_speech": Hg_speech,   # [len(yg), D]
        }