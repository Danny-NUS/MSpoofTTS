# Discriminator/base.py
import torch.nn as nn
from .modules import SinusoidalPositionalEncoding
from .conformer import ConformerEncoder

class BaseTokenDiscriminator(nn.Module):
    """
    Shared embedding + positional encoding + Conformer encoder.
    Child classes decide how to pool / project.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        self.encoder = ConformerEncoder(
            num_layers, d_model, nhead, dim_feedforward, dropout
        )

    def encode(self, tokens, padding_mask=None):
        x = self.embedding(tokens)
        x = self.positional_encoding(x)
        return self.encoder(x, padding_mask)
