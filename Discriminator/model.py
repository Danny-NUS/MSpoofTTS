# Obsolete, for reference only
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class FeedForwardModule(nn.Module):
    def __init__(self, d_model, dim_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_ff),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConvolutionModule(nn.Module):
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D]
        x = self.layer_norm(x)
        x = x.transpose(1, 2)           # [B, D, T]
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)           # [B, T, D]
        return x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model]
        """
        return x + self.pe[:, : x.size(1)]


class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_ff,
        dropout,
        conv_kernel=31,
    ):
        super().__init__()

        self.ffn1 = FeedForwardModule(d_model, dim_ff, dropout)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.conv = ConvolutionModule(d_model, conv_kernel, dropout)
        self.ffn2 = FeedForwardModule(d_model, dim_ff, dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        # Macaron FFN (½)
        x = x + 0.5 * self.ffn1(x)

        # Self-attention
        attn_out, _ = self.self_attn(
            x, x, x, key_padding_mask=padding_mask
        )
        x = x + self.dropout(attn_out)

        # Convolution
        x = x + self.conv(x)

        # Macaron FFN (½)
        x = x + 0.5 * self.ffn2(x)

        return self.norm(x)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        nhead,
        dim_ff,
        dropout,
        conv_kernel=31,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    d_model, nhead, dim_ff, dropout, conv_kernel
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, padding_mask):
        for layer in self.layers:
            x = layer(x, padding_mask)
        return x


class TokenDiscriminator(nn.Module):
    def __init__(
        self,
        vocab_size: int = 65536,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 2000, 
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model, max_len=max_len
        )

        self.encoder = ConformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_ff=dim_feedforward,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        """
        sequences: list of 1D LongTensor, each shape [T_i]
        Returns:
            logits: Tensor of shape [B]
        """

        device = sequences[0].device
        batch_size = len(sequences)
        lengths = torch.tensor(
            [seq.size(0) for seq in sequences], device=device
        )
        max_len = lengths.max().item()

        # ---- Padding ----
        padded = torch.zeros(
            batch_size, max_len, dtype=torch.long, device=device
        )
        padding_mask = torch.ones(
            batch_size, max_len, dtype=torch.bool, device=device
        )  # True = PAD

        for i, seq in enumerate(sequences):
            padded[i, : seq.size(0)] = seq
            padding_mask[i, : seq.size(0)] = False

        # ---- Embedding + PE ----
        x = self.embedding(padded)           # [B, T, D]
        x = self.positional_encoding(x)

        # ---- Conformer Encoder ----
        h = self.encoder(x, padding_mask)    # [B, T, D]

        # ---- Masked mean pooling ----
        mask = (~padding_mask).unsqueeze(-1)     # [B, T, 1]
        h_sum = (h * mask).sum(dim=1)             # [B, D]
        denom = mask.sum(dim=1).clamp(min=1)      # [B, 1]
        h_mean = h_sum / denom                    # [B, D]

        # ---- Classification ----
        logits = self.classifier(h_mean).squeeze(-1)  # [B]
        return logits


class SegmentTokenDiscriminator(nn.Module):
    def __init__(
        self,
        vocab_size: int = 65536,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        segment_len: int = 50,
    ):
        super().__init__()

        self.segment_len = segment_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model, max_len=segment_len
        )

        self.encoder = ConformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_ff=dim_feedforward,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: LongTensor of shape [B, 50]
        Returns:
            logits: Tensor of shape [B]
        """

        # ---- Sanity check (optional but nice) ----
        assert tokens.dim() == 2 and tokens.size(1) == self.segment_len, \
            f"Expected [B, {self.segment_len}] tokens, got {tokens.shape}"

        # ---- Embedding + PE ----
        x = self.embedding(tokens)        # [B, 50, D]
        x = self.positional_encoding(x)   # [B, 50, D]

        # ---- Conformer Encoder ----
        # No padding mask needed
        h = self.encoder(x, padding_mask=None)  # [B, 50, D]

        # ---- Mean pooling over segment ----
        h_mean = h.mean(dim=1)  # [B, D]

        # ---- Classification ----
        logits = self.classifier(h_mean).squeeze(-1)  # [B]
        return logits
