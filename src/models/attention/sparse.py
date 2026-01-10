"""Sparse Attention implementation.

Sparse attention patterns reduce complexity by limiting which tokens
can attend to each other.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAttention(nn.Module):
    """Block-sparse attention with local and global patterns.

    Combines local attention (each token attends to nearby tokens)
    with global attention (special tokens attend to all).
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: Optional[int] = None,
        block_size: int = 64,
        num_global_tokens: int = 0,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.block_size = block_size
        self.num_global_tokens = num_global_tokens

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

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

        # Create sparse attention mask
        sparse_mask = self._create_sparse_mask(seq_len, hidden_states.device)

        # Combine with input attention mask if provided
        if attention_mask is not None:
            sparse_mask = sparse_mask + attention_mask

        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights + sparse_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None

    def _create_sparse_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create block-sparse attention mask.

        Each token can attend to:
        1. Tokens in the same block (local attention)
        2. Global tokens (first num_global_tokens positions)
        """
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)

        # Local attention: block diagonal
        for i in range(0, seq_len, self.block_size):
            end = min(i + self.block_size, seq_len)
            mask[i:end, i:end] = 0

        # Global attention: first tokens can attend to all
        if self.num_global_tokens > 0:
            mask[:, :self.num_global_tokens] = 0
            mask[:self.num_global_tokens, :] = 0

        # Causal mask
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1
        )
        mask = torch.maximum(mask, causal_mask)

        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
