# Discriminator/strided.py
import torch
import torch.nn as nn

from .base import BaseTokenDiscriminator


class StridedSegmentTokenDiscriminator(BaseTokenDiscriminator):
    """
    Single-scale discriminator:
      - input segment_len (e.g., 50)
      - choose a target scale (e.g., 25 or 10)
      - downsample tokens by fixed stride BEFORE encoding
      - one encoder pass, one logit
    """

    def __init__(
        self,
        segment_len: int = 50,
        scale: int = 50,   # 50 / 25 / 10
        **kwargs,
    ):
        assert segment_len % scale == 0, \
            f"segment_len ({segment_len}) must be divisible by scale ({scale})"

        super().__init__(max_len=segment_len, **kwargs)

        self.segment_len = segment_len
        self.scale = scale
        self.stride = segment_len // scale

        d_model = kwargs["d_model"]
        dropout = kwargs["dropout"]

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: LongTensor [B, segment_len] (e.g., [B, 50])
        returns: logits [B]
        """
        assert tokens.dim() == 2 and tokens.size(1) == self.segment_len, \
            f"Expected [B, {self.segment_len}], got {tokens.shape}"

        # Downsample in token space
        x = tokens[:, ::self.stride]  # [B, scale]
        assert x.size(1) == self.scale, \
            f"Downsample produced length {x.size(1)} != scale {self.scale}"

        h = self.encode(x)        # [B, scale, D]
        h = h.mean(dim=1)         # [B, D]
        return self.classifier(h).squeeze(-1)
