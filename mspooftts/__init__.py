"""MSpoofTTS additions layered on top of the upstream NeuTTS runtime."""

from .checkpoints import load_discriminator_state_dict

__all__ = ["load_discriminator_state_dict"]
