# Discriminator/conformer.py
import torch.nn as nn
from .modules import FeedForwardModule, ConvolutionModule

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
