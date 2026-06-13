# MSpoofTTS note: this file is the upstream NeuTTS runtime with a small set of
# explicitly marked additions for guided decoding. Original NeuTTS behavior is
# preserved when `sampling_scheme="orig"` and `use_dis/use_hier` are left False.
#
# MSpoofTTS additions in this file:
# - Discriminator and Hugging Face checkpoint-loader imports.
# - EAS helpers: `nucleus_sampling` and `EASPenalty`.
# - Optional discriminator-loading args in `NeuTTS.__init__`.
# - Speech-token masking helpers and `rank_eas_hier` decoding.
# - `sampling_scheme` routing in `infer` / `_infer_torch`.

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
import math
import random
from phonemizer.backend import EspeakBackend
from neucodec import NeuCodec, DistillNeuCodec
from transformers import AutoTokenizer, AutoModelForCausalLM

# MSpoofTTS (new): discriminator models and checkpoint loading.
from Discriminator import SegmentTokenDiscriminator
from Discriminator import StridedSegmentTokenDiscriminator
from mspooftts.checkpoints import load_discriminator_state_dict


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


# MSpoofTTS (new): EAS sampling helper.
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


# MSpoofTTS (new): temporal penalty memory for EAS.
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
        # MSpoofTTS (new): leave these defaults unchanged for vanilla NeuTTS.
        use_dis=False,
        use_hier=False,
        discriminator_repo="Chanson-0803/MSpoofTTS",
        discriminator_revision=None,
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
            # MSpoofTTS loads discriminator checkpoints lazily so vanilla NeuTTS
            # inference remains unchanged and does not require extra downloads.
            self.discriminator50 = SegmentTokenDiscriminator(
                segment_len=50,
                vocab_size=65536,
                d_model=256,
                nhead=8,
                num_layers=4,
                dim_feedforward=1024,
                dropout=0.1,
            )
            state = load_discriminator_state_dict(
                "segment_len50",
                repo_id=discriminator_repo,
                revision=discriminator_revision,
            )
            self.discriminator50.load_state_dict(state)
            self.discriminator50.eval()
            self.discriminator50.to(self.backbone.device)

        if use_hier:
            # Guided decoding uses five separately trained token discriminators:
            # three contiguous spans plus two skip-sampled 50-token views.
            self.discriminator50s10 = StridedSegmentTokenDiscriminator(
                segment_len=50,
                scale=10,
                vocab_size=65536,
                d_model=256,
                nhead=8,
                num_layers=4,
                dim_feedforward=1024,
                dropout=0.1,
            )
            state = load_discriminator_state_dict(
                "strided_seg50_scale10",
                repo_id=discriminator_repo,
                revision=discriminator_revision,
            )
            self.discriminator50s10.load_state_dict(state)
            self.discriminator50s10.eval()
            self.discriminator50s10.to(self.backbone.device)

            self.discriminator50s25 = StridedSegmentTokenDiscriminator(
                segment_len=50,
                scale=25,
                vocab_size=65536,
                d_model=256,
                nhead=8,
                num_layers=4,
                dim_feedforward=1024,
                dropout=0.1,
            )
            state = load_discriminator_state_dict(
                "strided_seg50_scale25",
                repo_id=discriminator_repo,
                revision=discriminator_revision,
            )
            self.discriminator50s25.load_state_dict(state)
            self.discriminator50s25.eval()
            self.discriminator50s25.to(self.backbone.device)

            self.discriminator25 = SegmentTokenDiscriminator(
                segment_len=25,
                vocab_size=65536,
                d_model=256,
                nhead=8,
                num_layers=4,
                dim_feedforward=1024,
                dropout=0.1,
            )
            state = load_discriminator_state_dict(
                "segment_len25",
                repo_id=discriminator_repo,
                revision=discriminator_revision,
            )
            self.discriminator25.load_state_dict(state)
            self.discriminator25.eval()
            self.discriminator25.to(self.backbone.device)

            self.discriminator10 = SegmentTokenDiscriminator(
                segment_len=10,
                vocab_size=65536,
                d_model=256,
                nhead=8,
                num_layers=4,
                dim_feedforward=1024,
                dropout=0.1,
            )
            state = load_discriminator_state_dict(
                "segment_len10",
                repo_id=discriminator_repo,
                revision=discriminator_revision,
            )
            self.discriminator10.load_state_dict(state)
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

    # MSpoofTTS (new): keep generation in the codec-token range.
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

    # MSpoofTTS (new): EAS generation without discriminator reranking.
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
        # Scheme: EAS generation (entropy-aware sampling with decay penalty memory).

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


    # MSpoofTTS (new): combine multi-resolution discriminator scores.
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
    

    # MSpoofTTS (new): hierarchical decoding from the paper.
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
        # Scheme: Rank-EAS hierarchical generation (EAS + multi-stage discriminator ranking).
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
        # Main chunk loop: generate candidate continuations, prune them
        # at 10/25 tokens, then rank complete 50-token chunks with the
        # multi-resolution spoof scores described in the paper.
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
        # MSpoofTTS (new): `orig` follows the NeuTTS baseline path; other
        # schemes opt into the added EAS / discriminator-guided decoding.
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
            elif sampling_scheme == "eas":
                # EAS scheme
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
            elif sampling_scheme == "rank_eas_hier":
                # Rank-EAS scheme
                output_tokens = self._rank_eas_hier_generate(
                    prompt_tensor,
                    max_length=self.max_context,
                    eos_token_id=speech_end_id,
                    temperature=1.0,

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
                    sample_top_k=50,
                )
            else:
                raise ValueError(
                    f"Unsupported sampling_scheme `{sampling_scheme}`. "
                    "Use one of: orig, eas, rank_eas_hier."
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
