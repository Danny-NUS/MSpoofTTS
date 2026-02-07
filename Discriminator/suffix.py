# Discriminator/suffix.py
from typing import Sequence, Dict
import torch
import torch.nn as nn

from .base import BaseTokenDiscriminator


class SuffixTokenDiscriminator(BaseTokenDiscriminator):
    """
    Multi-horizon discriminator using suffix truncation.
    Each head sees the last K hidden states.
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

        # one head per horizon
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
        tokens: LongTensor [B, segment_len]
        returns:
            dict {horizon: logits[B]}
        """
        B, T = tokens.shape
        assert T == self.segment_len, \
            f"Expected tokens of length {self.segment_len}, got {T}"

        # Encode full segment once
        h = self.encode(tokens)  # [B, 50, D]

        outputs: Dict[int, torch.Tensor] = {}

        for horizon in self.horizons:
            # take last `horizon` hidden states
            h_suffix = h[:, -horizon:, :]      # [B, h, D]
            h_mean = h_suffix.mean(dim=1)      # [B, D]
            logits = self.heads[str(horizon)](h_mean).squeeze(-1)
            outputs[horizon] = logits

        return outputs
