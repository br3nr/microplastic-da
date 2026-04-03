import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Optional
import torch.nn.functional as F
import math


# Gradient reversal for DANN
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)


# Code based on https://github.com/maxjcohen/transformer && Attention is All You Need
class Transformer(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_channel: int,
        d_model: int,
        d_output: int,
        q: int,
        v: int,
        h: int,
        N: int,
        dropout: float = 0.3,
        pe: bool = False,
        bottleneck_dim: int = 256,
    ):
        super().__init__()

        self._d_input = d_input
        self._d_channel = d_channel
        self._d_model = d_model
        self._pe = pe
        self._bottleneck_dim = bottleneck_dim

        self.layers_encoding = nn.ModuleList(
            [Encoder(d_model, q, v, h, dropout=dropout) for _ in range(N)]
        )

        self._embedding = nn.Linear(self._d_channel, d_model)

        # Bottleneck: flattened features -> smaller dim.
        # Avoids out-of-memory issues in cuda.
        self._bottleneck = nn.Sequential(
            nn.Linear(d_model * d_input, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self._linear = nn.Linear(bottleneck_dim, d_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoding = self._embedding(x)

        # Positional encoding
        if self._pe:
            pe = torch.ones_like(encoding[0])
            position = torch.arange(0, self._d_input, device=x.device).unsqueeze(-1)
            temp = torch.arange(0, self._d_model, 2, device=x.device).float()
            temp = temp * -(math.log(10000) / self._d_model)
            temp = torch.exp(temp).unsqueeze(0)
            temp = torch.matmul(position.float(), temp)
            pe[:, 0::2] = torch.sin(temp)
            pe[:, 1::2] = torch.cos(temp)
            encoding = encoding + pe

        for layer in self.layers_encoding:
            encoding = layer(encoding)

        # Flatten and project
        flattened = encoding.reshape(encoding.shape[0], -1)
        bottleneck_features = self._bottleneck(flattened)

        return bottleneck_features

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        return self._linear(features)

    @property
    def feature_dim(self) -> int:
        return self._bottleneck_dim


class Encoder(nn.Module):
    """Encoder block from Attention is All You Need."""

    def __init__(self, d_model: int, q: int, v: int, h: int, dropout: float = 0.3):
        super().__init__()

        self._selfAttention = MultiHeadAttention(d_model, q, v, h)
        self._feedForward = PositionwiseFeedForward(d_model)

        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)

        self._dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self attention
        residual = x
        x = self._selfAttention(query=x, key=x, value=x)
        x = self._dropout(x)
        x = self._layerNorm1(x + residual)

        # Feed forward
        residual = x
        x = self._feedForward(x)
        x = self._dropout(x)
        x = self._layerNorm2(x + residual)
        return x


class MultiHeadAttention(nn.Module):
    """Multi Head Attention block from Attention is All You Need."""

    def __init__(self, d_model: int, q: int, v: int, h: int):
        super().__init__()

        self._q = q
        self._h = h

        # Query, keys and value matrices
        self._W_q = nn.Linear(d_model, q * h)
        self._W_k = nn.Linear(d_model, q * h)
        self._W_v = nn.Linear(d_model, v * h)

        self._W_o = nn.Linear(v * h, d_model)
        self._scores = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[str] = None,
    ) -> torch.Tensor:

        Q = torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0)
        K = torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0)
        V = torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0)

        # Scaled dot product
        self._scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self._q)
        self._scores = F.softmax(self._scores, dim=-1)
        attention = torch.matmul(self._scores, V)
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)
        self_attention = self._W_o(attention_heads)
        return self_attention


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed Forward Network block from Attention is All You Need."""

    def __init__(self, d_model: int, d_ff: Optional[int] = 2048):
        super().__init__()
        self._linear1 = nn.Linear(d_model, d_ff)
        self._linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear2(F.relu(self._linear1(x)))
