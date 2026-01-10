"""Tests for model implementations."""

import pytest
import torch

from src.models.attention.grouped_query import GroupedQueryAttention
from src.models.attention.linear_attention import LinearAttention
from src.models.components.ffn import SwiGLU, GatedMLP
from src.models.components.embeddings import RotaryEmbedding


class TestGroupedQueryAttention:
    """Tests for Grouped Query Attention."""

    def test_init(self):
        attn = GroupedQueryAttention(
            hidden_size=256,
            num_attention_heads=8,
            num_kv_heads=2,
        )
        assert attn.num_attention_heads == 8
        assert attn.num_kv_heads == 2
        assert attn.num_groups == 4

    def test_forward(self):
        attn = GroupedQueryAttention(
            hidden_size=256,
            num_attention_heads=8,
            num_kv_heads=2,
        )
        x = torch.randn(2, 16, 256)
        output, _ = attn(x)
        assert output.shape == x.shape

    def test_kv_cache(self):
        attn = GroupedQueryAttention(
            hidden_size=256,
            num_attention_heads=8,
            num_kv_heads=2,
        )
        x = torch.randn(2, 16, 256)
        _, kv = attn(x, use_cache=True)
        assert kv is not None
        assert len(kv) == 2


class TestLinearAttention:
    """Tests for Linear Attention."""

    def test_init(self):
        attn = LinearAttention(
            hidden_size=256,
            num_attention_heads=8,
        )
        assert attn.num_attention_heads == 8

    def test_forward(self):
        attn = LinearAttention(
            hidden_size=256,
            num_attention_heads=8,
        )
        x = torch.randn(2, 16, 256)
        output, _ = attn(x)
        assert output.shape == x.shape


class TestSwiGLU:
    """Tests for SwiGLU FFN."""

    def test_init(self):
        ffn = SwiGLU(hidden_size=256, intermediate_size=1024)
        assert ffn.gate_proj.in_features == 256
        assert ffn.gate_proj.out_features == 1024

    def test_forward(self):
        ffn = SwiGLU(hidden_size=256, intermediate_size=1024)
        x = torch.randn(2, 16, 256)
        output = ffn(x)
        assert output.shape == x.shape


class TestRotaryEmbedding:
    """Tests for Rotary Position Embedding."""

    def test_init(self):
        rope = RotaryEmbedding(dim=64, max_position_embeddings=2048)
        assert rope.dim == 64

    def test_forward(self):
        rope = RotaryEmbedding(dim=64, max_position_embeddings=2048)
        x = torch.randn(2, 8, 16, 64)
        cos, sin = rope(x)
        assert cos.shape[-1] == 64
        assert sin.shape[-1] == 64
