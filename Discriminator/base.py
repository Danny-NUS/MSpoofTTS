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
        vocab_size: int = 65636,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 50,
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
