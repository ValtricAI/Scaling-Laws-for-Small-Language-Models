"""Efficient Transformer architecture inspired by MobileLLM.

Key design principles from MobileLLM:
1. Deep and thin: More layers with smaller hidden size
2. Grouped Query Attention: Reduce KV cache memory
3. SwiGLU activation: Better quality than ReLU/GELU
4. Embedding sharing: Tie input and output embeddings
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn

from src.models.attention.grouped_query import GroupedQueryAttention
from src.models.components.ffn import SwiGLU
from src.models.components.embeddings import RotaryEmbedding


class TransformerBlock(nn.Module):
    """Single transformer block with GQA and SwiGLU."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()

        self.attention = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
        )

        self.ffn = SwiGLU(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

        self.input_layernorm = nn.RMSNorm(hidden_size, eps=layer_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        hidden_states = residual + hidden_states

        # Pre-norm FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value


class EfficientTransformer(nn.Module):
    """Efficient transformer model inspired by MobileLLM architecture."""

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 960,
        num_layers: int = 32,
        num_attention_heads: int = 15,
        num_kv_heads: int = 5,
        intermediate_size: Optional[int] = None,
        max_position_embeddings: int = 2048,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-6,
        tie_word_embeddings: bool = True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tie_word_embeddings = tie_word_embeddings

        if intermediate_size is None:
            intermediate_size = hidden_size * 4

        # Embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.rotary_emb = RotaryEmbedding(
            dim=hidden_size // num_attention_heads,
            max_position_embeddings=max_position_embeddings,
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_kv_heads=num_kv_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.RMSNorm(hidden_size, eps=layer_norm_eps)

        # LM head (shared with embeddings if tie_word_embeddings)
        if not tie_word_embeddings:
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # Create causal mask
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)

        # Forward through layers
        present_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values else None

            hidden_states, present_key_value = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )

            if use_cache:
                present_key_values.append(present_key_value)

        hidden_states = self.norm(hidden_states)

        # LM head
        if self.tie_word_embeddings:
            logits = torch.matmul(hidden_states, self.embed_tokens.weight.T)
        else:
            logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return type("Output", (), {
            "loss": loss,
            "logits": logits,
            "past_key_values": present_key_values,
        })()
