"""Optimizer utilities."""

from typing import Tuple, Optional, Iterable
import torch
from torch.optim import AdamW, Optimizer


def create_optimizer(
    model: torch.nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    exclude_bias_and_norm: bool = True,
) -> Optimizer:
    """Create an AdamW optimizer with optional weight decay exclusions.

    Args:
        model: The model to optimize.
        lr: Learning rate.
        weight_decay: Weight decay coefficient.
        betas: Adam beta parameters.
        eps: Adam epsilon.
        exclude_bias_and_norm: Whether to exclude bias and normalization
            parameters from weight decay.

    Returns:
        Configured optimizer.
    """
    if exclude_bias_and_norm:
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if "bias" in name or "norm" in name or "layernorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
    else:
        param_groups = [{"params": model.parameters(), "weight_decay": weight_decay}]

    return AdamW(param_groups, lr=lr, betas=betas, eps=eps)


def get_parameter_count(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters.

    Args:
        model: The model.
        trainable_only: Whether to count only trainable parameters.

    Returns:
        Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
