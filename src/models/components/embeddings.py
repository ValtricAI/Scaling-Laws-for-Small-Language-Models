"""Embedding and positional encoding implementations."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Applies rotation to query and key vectors based on position,
    enabling relative position encoding with absolute position information.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
    ):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        # Build cache
        self._build_cache(max_position_embeddings)

    def _build_cache(self, seq_len: int):
        """Build sin/cos cache for positions."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cos and sin for rotary embedding.

        Args:
            x: Input tensor (batch, heads, seq_len, head_dim)
            position_ids: Position indices (batch, seq_len)

        Returns:
            Tuple of (cos, sin) tensors
        """
        seq_len = x.shape[2]

        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)

        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        return cos.to(x.dtype), sin.to(x.dtype)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to queries and keys.

    Args:
        q: Query tensor (batch, heads, seq_len, head_dim)
        k: Key tensor (batch, heads, seq_len, head_dim)
        cos: Cosine values
        sin: Sine values

    Returns:
        Tuple of rotated (q, k)
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class ALiBi(nn.Module):
    """Attention with Linear Biases (ALiBi).

    Adds position-dependent linear bias to attention scores,
    enabling extrapolation to longer sequences.
    """

    def __init__(
        self,
        num_attention_heads: int,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()

        self.num_attention_heads = num_attention_heads

        # Compute slopes for each head
        slopes = self._get_slopes(num_attention_heads)
        self.register_buffer("slopes", slopes)

        # Build bias cache
        self._build_cache(max_position_embeddings)

    def _get_slopes(self, n: int) -> torch.Tensor:
        """Get ALiBi slopes for n heads."""

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return torch.tensor(get_slopes_power_of_2(n))

        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return torch.tensor(
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )

    def _build_cache(self, seq_len: int):
        """Build position bias cache."""
        positions = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
        positions = positions.abs().neg().unsqueeze(0)  # (1, seq_len, seq_len)

        alibi = self.slopes.unsqueeze(1).unsqueeze(1) * positions
        self.register_buffer("alibi", alibi)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Get ALiBi bias for given sequence length."""
        if seq_len > self.alibi.shape[-1]:
            self._build_cache(seq_len)

        return self.alibi[:, :seq_len, :seq_len]
