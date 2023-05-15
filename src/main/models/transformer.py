import torch
import torch.nn as nn
import torch.nn.functional as F

from src.main.models.position import PositionalEncoding


class EncoderBlock(nn.Module):
    """
    A single encoder block for the transformer encoder. Includes a
    MultiHeadAttention layer followed by a FeedForward layer.

    Parameters:
        - d_model (int): dimension of input.
        - num_heads (int): number of attention heads.
        - dropout (float): dropout probability.
    """
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super(EncoderBlock, self).__init__()
        # self-attention.
        self.attention = nn.MultiheadAttention(embed_dim=d_model,
                                               num_heads=num_heads)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        # feed forward.
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, d_model)
        )
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor):
        x_attn = self.dropout1(self.norm1(x + self.attention(x, x, x)[0]))
        x_out = self.dropout2(self.norm2(x_attn + self.feedforward(x_attn)))
        return x_out


class Encoder(nn.Module):
    """
    A sequence of EncoderBlocks used to create an encoder.

    Parameters:
        - input_dim (int): the dimension of the input.
        - num_layers (int): number of encoder blocks.
        - d_model (int): dimension of input.
        - num_heads (int): number of attention heads.
        - num_classes (int): number of output classes.
        - max_len (int): length of the input sequence.
        - dropout (float): dropout probability.
    """
    def __init__(
            self,
            input_dim: int,
            num_layers: int = 6,
            d_model: int = 512,
            num_heads: int = 8,
            num_classes: int = 10,
            max_len: int = 256,
            dropout: float = 0.1
    ):
        super(Encoder, self).__init__()
        self.linear_in = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(max_len, d_model)
        self.blocks = nn.ModuleList([
          EncoderBlock(d_model, num_heads, dropout)
          for _ in range(num_layers)
        ])
        self.linear_out = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor):
        x_in = self.linear_in(x)
        x_enc = self.positional_encoding(x_in)
        for block in self.blocks:
            x_enc = block(x_enc)
        x_out = self.linear_out(x_enc)
        x_out = F.softmax(x_out)
        return x_out
