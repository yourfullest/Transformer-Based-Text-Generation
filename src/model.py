from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        positions = torch.arange(max_len).unsqueeze(1)
        div_terms = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(positions * div_terms)
        encoding[:, 1::2] = torch.cos(positions * div_terms[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.encoding[:, : x.size(1)])


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(query.size(0), query.size(1), -1)
        return self.out_proj(context)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.dropout(self.self_attn(normed, normed, normed, src_mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None,
        src_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.dropout(self.self_attn(normed, normed, normed, tgt_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), memory, memory, src_mask))
        x = x + self.dropout(self.ff(self.norm3(x)))
        return x


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 512,
        max_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        return (src != self.pad_id).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        seq_len = tgt.size(1)
        padding_mask = (tgt != self.pad_id).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=tgt.device))
        return padding_mask & causal_mask.unsqueeze(0).unsqueeze(0)

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src_mask = self.make_src_mask(src)
        x = self.position(self.token_embedding(src) * math.sqrt(self.d_model))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.encoder_norm(x), src_mask

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        tgt_mask = self.make_tgt_mask(tgt)
        x = self.position(self.token_embedding(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.decoder_norm(x)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        memory, src_mask = self.encode(src)
        decoded = self.decode(tgt, memory, src_mask)
        return self.output(decoded)
