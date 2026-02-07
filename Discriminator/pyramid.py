# Discriminator/pyramid.py
from typing import Sequence, Dict
import torch
import torch.nn as nn

from .base import BaseTokenDiscriminator


class PyramidTokenDiscriminator(BaseTokenDiscriminator):
    """
    Multi-resolution discriminator using temporal pyramid pooling.
    Information is intentionally destroyed at lower resolutions.
    """

    def __init__(
        self,
        vocab_size: int = 65536,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        segment_len: int = 50,
        horizons: Sequence[int] = (50, 25, 10),
    ):
        assert max(horizons) == segment_len, \
            "segment_len must equal max(horizons)"

        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=segment_len,
        )

        self.segment_len = segment_len
        self.horizons = tuple(sorted(horizons, reverse=True))

        # ---- pooling ops (explicit, non-learnable baseline) ----
        self.pool_50_to_25 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.pool_50_to_10 = nn.AvgPool1d(kernel_size=5, stride=5)

        # ---- heads per horizon ----
        self.heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
            for h in self.horizons
        })

    def forward(self, tokens: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        tokens: LongTensor [B, 50]
        returns:
            dict {horizon: logits[B]}
        """
        B, T = tokens.shape
        assert T == self.segment_len, \
            f"Expected tokens of length {self.segment_len}, got {T}"

        # ---- Encode once ----
        h = self.encode(tokens)  # [B, 50, D]

        # ---- Prepare pyramid ----
        # [B, T, D] -> [B, D, T] for pooling
        h_t = h.transpose(1, 2)

        h50 = h                          # [B, 50, D]
        h25 = self.pool_50_to_25(h_t).transpose(1, 2)  # [B, 25, D]
        h10 = self.pool_50_to_10(h_t).transpose(1, 2)  # [B, 10, D]

        reps = {
            50: h50,
            25: h25,
            10: h10,
        }

        outputs: Dict[int, torch.Tensor] = {}

        for h_len in self.horizons:
            h_rep = reps[h_len]                  # [B, h, D]
            h_mean = h_rep.mean(dim=1)           # [B, D]
            logits = self.heads[str(h_len)](h_mean).squeeze(-1)
            outputs[h_len] = logits

        return outputs
