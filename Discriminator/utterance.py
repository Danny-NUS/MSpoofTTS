# Discriminator/utterance.py
import torch
import torch.nn as nn
from .base import BaseTokenDiscriminator

class TokenDiscriminator(BaseTokenDiscriminator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classifier = nn.Sequential(
            nn.Linear(kwargs["d_model"], kwargs["d_model"]),
            nn.ReLU(),
            nn.Dropout(kwargs["dropout"]),
            nn.Linear(kwargs["d_model"], 1),
        )

    def forward(self, padded, padding_mask):
        h = self.encode(padded, padding_mask)
        mask = (~padding_mask).unsqueeze(-1)
        h = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.classifier(h).squeeze(-1)
