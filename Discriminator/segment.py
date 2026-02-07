# Discriminator/segment.py
import torch
import torch.nn as nn
from .base import BaseTokenDiscriminator

class SegmentTokenDiscriminator(BaseTokenDiscriminator):
    def __init__(self, segment_len=50, **kwargs):
        super().__init__(max_len=segment_len, **kwargs)
        self.segment_len = segment_len
        self.classifier = nn.Sequential(
            nn.Linear(kwargs["d_model"], kwargs["d_model"]),
            nn.ReLU(),
            nn.Dropout(kwargs["dropout"]),
            nn.Linear(kwargs["d_model"], 1),
        )

    def forward(self, tokens: torch.Tensor):
        assert tokens.shape[1] == self.segment_len
        h = self.encode(tokens)           # [B, 50, D]
        h = h.mean(dim=1)
        return self.classifier(h).squeeze(-1)
