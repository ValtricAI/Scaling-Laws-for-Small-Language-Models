"""Learning rate scheduler utilities."""

import math
from typing import Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    num_warmup_steps: int = 1000,
    num_training_steps: int = 100000,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create a learning rate scheduler.

    Args:
        optimizer: The optimizer.
        scheduler_type: Type of scheduler ("cosine", "linear", "constant").
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total training steps.
        min_lr_ratio: Minimum LR as ratio of initial LR (for cosine).

    Returns:
        Learning rate scheduler.
    """
    if scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, min_lr_ratio
        )
    elif scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif scheduler_type == "constant":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create a cosine schedule with linear warmup.

    Args:
        optimizer: The optimizer.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total training steps.
        min_lr_ratio: Minimum LR as ratio of initial LR.

    Returns:
        Learning rate scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """Create a linear schedule with warmup.

    Args:
        optimizer: The optimizer.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total training steps.

    Returns:
        Learning rate scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
) -> LambdaLR:
    """Create a constant schedule with warmup.

    Args:
        optimizer: The optimizer.
        num_warmup_steps: Number of warmup steps.

    Returns:
        Learning rate scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)
