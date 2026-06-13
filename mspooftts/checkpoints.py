"""Checkpoint loading helpers for MSpoofTTS discriminator inference."""

from __future__ import annotations

from pathlib import Path
import torch

DEFAULT_DISCRIMINATOR_REPO = "Chanson-0803/MSpoofTTS"

# (new) Logical discriminator role -> expected Hugging Face checkpoint filename.
# If checkpoint filenames change on the model hub, update this table only.
CHECKPOINT_FILES = {
    "segment_len10": "checkpoints/segment_len10.ckpt",
    "segment_len25": "checkpoints/segment_len25.ckpt",
    "segment_len50": "checkpoints/segment_len50.ckpt",
    "strided_seg50_scale10": "checkpoints/strided_seg50_scale10.ckpt",
    "strided_seg50_scale25": "checkpoints/strided_seg50_scale25.ckpt",
}

FALLBACK_PATTERNS = {
    key: tuple(Path(filename).stem.lower().split("_"))
    for key, filename in CHECKPOINT_FILES.items()
}

_SUPPORTED_SUFFIXES = (".ckpt", ".pt", ".pth", ".bin")


def _normalise_state_dict(state):
    """Accept Lightning, plain PyTorch, and prefixed module state dicts."""
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            candidate = state.get(key)
            if isinstance(candidate, dict):
                state = candidate
                break

    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a PyTorch state dict.")

    cleaned = {}
    for key, value in state.items():
        if key.startswith("model."):
            key = key[len("model."):]
        cleaned[key] = value
    return cleaned


def _resolve_hf_filename(alias: str, repo_id: str, revision: str | None = None) -> str:
    try:
        from huggingface_hub import list_repo_files
    except ImportError as exc:
        raise ImportError(
            "Install `huggingface_hub` to load MSpoofTTS discriminator checkpoints."
        ) from exc

    files = list_repo_files(repo_id=repo_id, revision=revision)
    expected = CHECKPOINT_FILES[alias]
    if expected in files:
        return expected

    patterns = FALLBACK_PATTERNS[alias]
    candidates = [
        path for path in files
        if path.lower().endswith(_SUPPORTED_SUFFIXES)
        and all(pattern in Path(path).stem.lower() for pattern in patterns)
    ]
    if candidates:
        return sorted(candidates)[0]

    ckpts = [path for path in files if path.lower().endswith(_SUPPORTED_SUFFIXES)]
    raise FileNotFoundError(f"No checkpoint for `{alias}` in `{repo_id}`. Found: {ckpts}")


def load_discriminator_state_dict(
    alias: str,
    *,
    repo_id: str = DEFAULT_DISCRIMINATOR_REPO,
    revision: str | None = None,
    local_path: str | Path | None = None,
):
    """Load one MSpoofTTS discriminator checkpoint from a local path or Hugging Face."""
    if alias not in CHECKPOINT_FILES:
        raise ValueError(f"Unknown discriminator checkpoint alias: {alias}")

    if local_path is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("Install `huggingface_hub` to load MSpoofTTS checkpoints.") from exc

        filename = _resolve_hf_filename(alias, repo_id, revision)
        local_path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)

    state = torch.load(local_path, map_location="cpu")
    return _normalise_state_dict(state)
