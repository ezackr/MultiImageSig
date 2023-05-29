from math import floor
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # attention block.
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        # feed-forward block.
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout)
        )

    def forward(self, x: torch.Tensor):
        # attention block.
        norm_x = self.norm1(x)
        x_attn = x + self.attention(norm_x, norm_x, norm_x)[0]
        # feed-forward block.
        norm_x = self.norm2(x_attn)
        x_out = x_attn + self.feedforward(norm_x)
        return x_out


class AttentionEncoder(nn.Module):
    """
    An attention-based encoder for image classification. Applies two
    convolutional layers, a sequence of transformer blocks, and then
    feed-forward layers for classification.

    Parameters:
        - input_shape (Tuple[int, int]): the dimension of the input. The tuple
            represents (seq_len, input_dim).
        - num_layers (int): number of encoder blocks.
        - num_heads (int): number of attention heads.
        - num_classes (int): number of output classes.
        - dropout (float): dropout probability.
    """
    def __init__(
            self,
            input_shape: Tuple[int, int],
            num_layers: int = 6,
            num_heads: int = 8,
            num_classes: int = 10,
            dropout: float = 0.1
    ):
        super(AttentionEncoder, self).__init__()
        seq_len, sig_dim = input_shape
        # convolutional layers.
        self.conv1 = nn.Conv1d(
            in_channels=sig_dim,
            out_channels=32,
            kernel_size=(3,),
        )
        self.pool1 = nn.MaxPool1d(kernel_size=3)
        self.seq_len = floor((((seq_len - 2) - 3) // 3) + 1)
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3,)
        )
        self.pool2 = nn.MaxPool1d(kernel_size=3)
        self.seq_len = floor((((self.seq_len - 2) - 3) // 3) + 1)

        # transformer blocks.
        self.encoder_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.encoder_blocks.append(EncoderBlock(
                d_model=64,
                num_heads=num_heads,
                dropout=dropout
            ))

        # fully-connected layers.
        self.norm = nn.LayerNorm(normalized_shape=64)
        self.dropout = nn.Dropout(p=dropout)
        self.encoder_linear = nn.Linear(in_features=64, out_features=1)
        self.out_linear = nn.Sequential(
            nn.Linear(in_features=64, out_features=50),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=50, out_features=num_classes),
            nn.Softmax()
        )

    def forward(self, x: torch.Tensor):
        # convolutional layers.
        x = x.permute(0, 2, 1)
        x_enc = self.pool1(self.conv1(x))
        x_enc = self.pool2(self.conv2(x_enc))
        x_enc = x_enc.permute(0, 2, 1)
        # transformer layers.
        for block in self.encoder_blocks:
            x_enc = block(x_enc)
        # fully-connected layers.
        x_rep = self.dropout(self.norm(x_enc))
        x_attn = F.softmax(self.encoder_linear(x_rep), dim=1)
        x_out = torch.matmul(x_attn.permute(0, 2, 1), x_rep).squeeze(dim=1)
        x_out = self.out_linear(x_out)
        return x_out
