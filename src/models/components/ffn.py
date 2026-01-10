"""Feed-forward network implementations."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU activation function.

    Combines Swish activation with Gated Linear Unit:
    SwiGLU(x) = (x * W1) * swish(x * W_gate) * W2

    Used in MobileLLM and other modern architectures.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ):
        super().__init__()

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class GatedMLP(nn.Module):
    """Gated MLP with configurable activation.

    Similar to SwiGLU but with configurable activation function.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        bias: bool = False,
    ):
        super().__init__()

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

        self.activation = self._get_activation(activation)

    def _get_activation(self, name: str):
        """Get activation function by name."""
        activations = {
            "gelu": F.gelu,
            "relu": F.relu,
            "silu": F.silu,
            "swish": F.silu,
            "tanh": torch.tanh,
        }
        return activations.get(name, F.gelu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class MLP(nn.Module):
    """Standard MLP without gating.

    Simple two-layer MLP with activation in between.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self.activation = self._get_activation(activation)

    def _get_activation(self, name: str):
        activations = {
            "gelu": F.gelu,
            "relu": F.relu,
            "silu": F.silu,
        }
        return activations.get(name, F.gelu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
