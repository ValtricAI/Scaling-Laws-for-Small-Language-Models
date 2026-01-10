"""Linear Attention implementation.

Linear attention replaces softmax attention with a kernel-based approximation,
reducing complexity from O(n^2) to O(n).
"""

import math
from typing import Optional, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttention(nn.Module):
    """Linear Attention with feature map approximation.

    Uses phi(x) feature maps instead of softmax for O(n) complexity.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: Optional[int] = None,
        feature_map: str = "elu",
        eps: float = 1e-6,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.eps = eps

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

        # Feature map function
        self.feature_map = self._get_feature_map(feature_map)

    def _get_feature_map(self, name: str) -> Callable:
        """Get feature map function by name."""
        if name == "elu":
            return lambda x: F.elu(x) + 1
        elif name == "relu":
            return F.relu
        elif name == "identity":
            return lambda x: x
        else:
            raise ValueError(f"Unknown feature map: {name}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        # Project queries, keys, values
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        # Apply feature map
        q = self.feature_map(q)
        k = self.feature_map(k)

        # Linear attention: O(n) complexity
        # Instead of: softmax(Q @ K^T) @ V
        # We compute: (phi(Q) @ (phi(K)^T @ V)) / (phi(Q) @ sum(phi(K)))

        # Compute KV product: (seq_len, head_dim) @ (head_dim, seq_len) -> accumulated
        kv = torch.einsum("bhnd,bhnv->bhdv", k, v)  # (batch, heads, head_dim, value_dim)

        # Compute normalizer
        k_sum = k.sum(dim=2)  # (batch, heads, head_dim)

        # Compute output
        qkv = torch.einsum("bhnd,bhdv->bhnv", q, kv)  # (batch, heads, seq_len, value_dim)
        normalizer = torch.einsum("bhnd,bhd->bhn", q, k_sum).unsqueeze(-1) + self.eps

        attn_output = qkv / normalizer

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None
