"""Hybrid State Space Model architecture.

Combines attention layers with state space model (SSM) layers
for efficient long-context modeling.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class S4Layer(nn.Module):
    """Simplified S4 (Structured State Space) layer.

    A linear state space model with structured matrices for
    efficient sequence modeling.
    """

    def __init__(
        self,
        hidden_size: int,
        state_size: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.state_size = state_size

        # State space parameters
        self.A = nn.Parameter(torch.randn(state_size, state_size) * 0.01)
        self.B = nn.Parameter(torch.randn(state_size, hidden_size) * 0.01)
        self.C = nn.Parameter(torch.randn(hidden_size, state_size) * 0.01)
        self.D = nn.Parameter(torch.ones(hidden_size))

        # Discretization timestep
        log_dt = torch.rand(hidden_size) * (
            torch.log(torch.tensor(dt_max)) - torch.log(torch.tensor(dt_min))
        ) + torch.log(torch.tensor(dt_min))
        self.log_dt = nn.Parameter(log_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SSM layer.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        dt = torch.exp(self.log_dt)

        # Discretize continuous-time parameters
        A_bar = torch.matrix_exp(self.A * dt.mean())
        B_bar = self.B * dt.unsqueeze(0)

        # Initialize state
        state = torch.zeros(batch_size, self.state_size, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            # State update: h_t = A * h_{t-1} + B * x_t
            state = torch.matmul(state, A_bar.T) + torch.matmul(x[:, t, :], B_bar.T)
            # Output: y_t = C * h_t + D * x_t
            output = torch.matmul(state, self.C.T) + self.D * x[:, t, :]
            outputs.append(output)

        return torch.stack(outputs, dim=1)


class HybridBlock(nn.Module):
    """Hybrid block that can use either attention or SSM."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        use_ssm: bool = False,
        state_size: int = 64,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()

        self.use_ssm = use_ssm

        if use_ssm:
            self.mixer = S4Layer(hidden_size, state_size)
        else:
            from src.models.attention.grouped_query import GroupedQueryAttention
            self.mixer = GroupedQueryAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_kv_heads=max(1, num_attention_heads // 3),
                dropout=dropout,
            )

        from src.models.components.ffn import SwiGLU
        self.ffn = SwiGLU(hidden_size, intermediate_size)

        self.input_layernorm = nn.RMSNorm(hidden_size, eps=layer_norm_eps)
        self.post_mixer_layernorm = nn.RMSNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Pre-norm mixer
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.use_ssm:
            hidden_states = self.mixer(hidden_states)
        else:
            hidden_states, _ = self.mixer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=False,
            )

        hidden_states = residual + hidden_states

        # Pre-norm FFN
        residual = hidden_states
        hidden_states = self.post_mixer_layernorm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class HybridSSM(nn.Module):
    """Hybrid model with alternating attention and SSM layers."""

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        num_layers: int = 24,
        num_attention_heads: int = 12,
        intermediate_size: Optional[int] = None,
        ssm_ratio: float = 0.5,  # Fraction of layers using SSM
        state_size: int = 64,
        max_position_embeddings: int = 2048,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        if intermediate_size is None:
            intermediate_size = hidden_size * 4

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Alternate between attention and SSM layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            use_ssm = (i % 2 == 1) and (i / num_layers < ssm_ratio * 2)
            self.layers.append(HybridBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                use_ssm=use_ssm,
                state_size=state_size,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
            ))

        self.norm = nn.RMSNorm(hidden_size, eps=layer_norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return type("Output", (), {"loss": loss, "logits": logits})()
