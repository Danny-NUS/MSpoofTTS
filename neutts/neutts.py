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
from phonemizer.backend import EspeakBackend
from neucodec import NeuCodec, DistillNeuCodec
from transformers import AutoTokenizer, AutoModelForCausalLM

from Discriminator import SegmentTokenDiscriminator


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
        logits: torch.Tensor,
        top_k: int,
        temperature: float,
    ) -> torch.Tensor:
        """
        logits: (vocab,)
        returns: sampled token id (scalar tensor)
        """
        if temperature != 1.0:
            logits = logits / temperature

        if top_k is not None and top_k > 0:
            values, indices = torch.topk(logits, top_k)
            probs = torch.softmax(values, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            return indices[sampled]
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


class NeuTTS:

    def __init__(
        self,
        backbone_repo="neuphonic/neutts-nano",
        backbone_device="cpu",
        codec_repo="neuphonic/neucodec",
        codec_device="cpu",
    ):

        # Consts
        self.sample_rate = 24_000
        self.max_context = 2048
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

        checkpoint_path = "/data2/minh_duc/TTS_spoofing/Segment_discriminator_len50/version_0/checkpoints/epoch=4-step=29675.ckpt"
        self.discriminator = SegmentTokenDiscriminator(segment_len=50,
                                vocab_size=65536,
                                d_model=256,
                                nhead=8,
                                num_layers=4,
                                dim_feedforward=1024,
                                dropout=0.1,
                                )
        state = torch.load(checkpoint_path, map_location="cpu")
        self.discriminator.load_state_dict(state["model_state_dict"])
        self.discriminator.eval()
        self.discriminator.to(self.backbone.device)

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
        max_length: int,
        eos_token_id: int,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        use_cache: bool = True,
        min_new_tokens: int = 0,
    ) -> torch.Tensor:
        """
        Explicit autoregressive generation for HF LLaMA-style models.

        Args:
            prompt_tensor: (1, T)
        Returns:
            output_tokens: (1, T + N)
        """
        device = prompt_tensor.device
        input_ids = prompt_tensor
        past_key_values = None

        generated = []
        cur_len = input_ids.shape[1]

        for step in range(max_length - cur_len):
            if past_key_values is None:
                outputs = self.backbone(
                    input_ids=input_ids,
                    use_cache=use_cache,
                )
            else:
                outputs = self.backbone(
                    input_ids=input_ids[:, -1:],  # only last token
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )

            logits = outputs.logits[:, -1, :]  # (1, vocab)
            past_key_values = outputs.past_key_values

            if do_sample:
                next_token = _sample_top_k(
                    logits.squeeze(0),
                    top_k=top_k,
                    temperature=temperature,
                )
            else:
                next_token = torch.argmax(logits, dim=-1)

            next_token = next_token.view(1, 1)
            generated.append(next_token)

            # EOS handling (respect min_new_tokens)
            if (
                next_token.item() == eos_token_id
                and len(generated) >= min_new_tokens
            ):
                break

            input_ids = next_token

        if generated:
            generated = torch.cat(generated, dim=1)
            return torch.cat([prompt_tensor, generated], dim=1)
        else:
            return prompt_tensor

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
        max_length: int,
        eos_token_id: int,
        temperature: float = 1.0,
        use_cache: bool = True,
        min_new_tokens: int = 0,
        # RAS params
        top_p: float = 0.8,
        top_k: int = 50,
        win_size: int = 20,
        tau_r: float = 0.1,
    ) -> torch.Tensor:
        """
        Explicit autoregressive generation with Repetition Aware Sampling (RAS),
        restricted to speech tokens + eos_token_id.
        """
        input_ids = prompt_tensor
        past_key_values = None

        generated = []

        # Track repetition in *speech-id* space (0..65535)
        decoded_speech_ids: list[int] = []

        cur_len = input_ids.shape[1]

        for _ in range(max_length - cur_len):
            if past_key_values is None:
                outputs = self.backbone(input_ids=input_ids, use_cache=use_cache)
            else:
                outputs = self.backbone(
                    input_ids=input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )

            logits = outputs.logits[:, -1, :]  # (1, vocab)
            past_key_values = outputs.past_key_values

            scores = logits.squeeze(0)

            if temperature != 1.0:
                scores = scores / temperature

            # ---- restrict to speech range + EOS ----
            scores = self._mask_to_speech_only(scores, eos_token_id=eos_token_id)

            # ---- RAS on masked scores ----
            next_id = ras_sampling(
                scores,
                # IMPORTANT: repetition window in speech-id space,
                # but ras_sampling expects same ID space as scores.
                # So we pass LM IDs history only for *speech tokens*:
                [sid + self.speech_start_id for sid in decoded_speech_ids],
                top_p=top_p,
                top_k=top_k,
                win_size=win_size,
                tau_r=tau_r,
            )

            next_token = torch.tensor([[next_id]], device=prompt_tensor.device)
            generated.append(next_token)

            # update speech repetition history (speech tokens only)
            sid = self._lm_to_speech_id_or_none(next_id)
            if sid is not None:
                decoded_speech_ids.append(sid)

            # stop on EOS once min_new_tokens satisfied
            if next_id == eos_token_id and len(generated) >= min_new_tokens:
                break

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

    # Normal sampling + discriminator guide chunk selections
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
            past_key_values=None,
        )

        if warmup is None:
            return output

        output = torch.cat([output, warmup], dim=1)
        if warmup[0, -1].item() == eos_token_id:
            return output

        # -------- discriminator-guided chunks --------
        while output.shape[1] < max_length:

            beam_lm_chunks = []
            beam_speech_ids = []
            eos_lm_chunks = []   # store EOS-ending beams

            for _ in range(n_beams):
                input_ids = output
                past_key_values = None

                lm_ids = []
                speech_ids = []

                for _ in range(max_lm_steps):
                    outputs = self.backbone(
                        input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    logits = outputs.logits[:, -1, :] / temperature
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
                        eos_lm_chunks.append(lm_ids)
                        break

                    # ---- collect speech tokens only ----
                    if self.speech_start_id <= next_id < self.speech_end_id:
                        speech_ids.append(next_id - self.speech_start_id)
                        if len(speech_ids) == segment_len:
                            break

                if len(speech_ids) == segment_len:
                    beam_lm_chunks.append(lm_ids)
                    beam_speech_ids.append(speech_ids)

            if not beam_lm_chunks:
                return output

            # ---- discriminator selection ----
            speech_tensor = torch.tensor(beam_speech_ids, device=device)  # [B, 50]
            scores = self.discriminator(speech_tensor)
            best = scores.argmax().item()

            chosen_lm_ids = beam_lm_chunks[best]
            chosen_lm_tensor = torch.tensor([chosen_lm_ids], device=device)

            # ---- append chosen LM tokens ----
            output = torch.cat([output, chosen_lm_tensor], dim=1)

            # length guard
            if output.shape[1] >= max_length:
                return output[:, :max_length]
            
        # If any beam ended with EOS, accept and stop
        if eos_lm_chunks:
            chosen = eos_lm_chunks[0]   # or random / shortest / highest prob later
            return torch.cat([output, torch.tensor([chosen], device=device)], dim=1)

        # Otherwise, must have full speech chunks
        if not beam_lm_chunks:
            return output
        

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

        # -------- warmup (plain top-k sample, no discriminator, no RAS) --------
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
        if warmup[0, -1].item() == eos_token_id:
            return output

        # -------- discriminator-guided chunks with RAS sampling --------
        while output.shape[1] < max_length:

            beam_lm_chunks: list[list[int]] = []
            beam_speech_ids: list[list[int]] = []
            eos_lm_chunks: list[list[int]] = []

            for _ in range(n_beams):
                input_ids = output
                past_key_values = None

                lm_ids: list[int] = []
                speech_ids: list[int] = []              # collected speech ids (0..65535) for discriminator
                decoded_speech_hist: list[int] = []     # repetition window history in speech-id space

                for _ in range(max_lm_steps):
                    outputs = self.backbone(
                        input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    logits = outputs.logits[:, -1, :]   # (1, vocab)
                    past_key_values = outputs.past_key_values

                    scores = (logits.squeeze(0) / temperature)

                    # ---- restrict to speech range + EOS ----
                    scores = self._mask_to_speech_only(scores, eos_token_id=eos_token_id)

                    # ---- RAS expects "decoded_tokens" in same id-space as scores (LM ids).
                    # We maintain repetition history in speech-id space (0..65535),
                    # then map to LM ids by adding speech_start_id.
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

                    # ---- EOS handling ----
                    if next_id == eos_token_id:
                        eos_lm_chunks.append(lm_ids)
                        break

                    # ---- collect speech tokens only ----
                    sid = self._lm_to_speech_id_or_none(next_id)
                    if sid is not None:
                        speech_ids.append(sid)
                        decoded_speech_hist.append(sid)

                        if len(speech_ids) == segment_len:
                            break

                # only keep full speech chunks for discriminator scoring
                if len(speech_ids) == segment_len:
                    beam_lm_chunks.append(lm_ids)
                    beam_speech_ids.append(speech_ids)

            # If any beam ended with EOS, accept and stop (your intended semantics)
            if eos_lm_chunks:
                chosen = eos_lm_chunks[0]  # could choose shortest / highest score later
                return torch.cat([output, torch.tensor([chosen], device=device)], dim=1)

            if not beam_lm_chunks:
                return output

            # ---- discriminator selection ----
            speech_tensor = torch.tensor(beam_speech_ids, device=device)  # [B, 50] in [0..65535]
            scores = self.discriminator(speech_tensor)
            best = scores.argmax().item()

            chosen_lm_ids = beam_lm_chunks[best]
            chosen_lm_tensor = torch.tensor([chosen_lm_ids], device=device)

            # ---- append chosen LM tokens ----
            output = torch.cat([output, chosen_lm_tensor], dim=1)

            if output.shape[1] >= max_length:
                return output[:, :max_length]

        return output

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