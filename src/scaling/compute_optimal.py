"""Compute-optimal allocation utilities."""

from typing import Dict, Tuple, Optional
import numpy as np


def compute_optimal_allocation(
    compute_budget: float,
    scaling_law: str = "chinchilla",
    **kwargs,
) -> Dict[str, float]:
    """Compute optimal model size and token count for a compute budget.

    Args:
        compute_budget: Total compute budget in FLOPs.
        scaling_law: Which scaling law to use ("chinchilla" or "kaplan").
        **kwargs: Additional parameters for the scaling law.

    Returns:
        Dictionary with optimal allocation.
    """
    if scaling_law == "chinchilla":
        return _chinchilla_optimal(compute_budget, **kwargs)
    elif scaling_law == "kaplan":
        return _kaplan_optimal(compute_budget, **kwargs)
    else:
        raise ValueError(f"Unknown scaling law: {scaling_law}")


def _chinchilla_optimal(
    compute_budget: float,
    alpha: float = 0.34,
    beta: float = 0.28,
    A: float = 406.4,
    B: float = 410.7,
) -> Dict[str, float]:
    """Compute Chinchilla-optimal allocation.

    Chinchilla finding: for compute-optimal training,
    N and D should be scaled equally.
    Approximately: D ≈ 20 * N
    """
    # Ratio from Chinchilla paper analysis
    # N_opt / D_opt = (alpha * B) / (beta * A)
    ratio = (alpha * B) / (beta * A)

    # C = 6 * N * D
    # N = ratio * D
    # C = 6 * ratio * D^2
    D_opt = np.sqrt(compute_budget / (6 * ratio))
    N_opt = ratio * D_opt

    return {
        "num_parameters": N_opt,
        "num_tokens": D_opt,
        "tokens_per_param": D_opt / N_opt,
        "compute_budget": compute_budget,
        "flops_per_token": 6 * N_opt,
    }


def _kaplan_optimal(
    compute_budget: float,
) -> Dict[str, float]:
    """Compute Kaplan-optimal allocation.

    Kaplan finding: allocate more to model size.
    N scales as C^0.73, D scales as C^0.27
    """
    # Kaplan recommendation
    N_opt = (compute_budget / 6) ** 0.73
    D_opt = compute_budget / (6 * N_opt)

    return {
        "num_parameters": N_opt,
        "num_tokens": D_opt,
        "tokens_per_param": D_opt / N_opt,
        "compute_budget": compute_budget,
        "flops_per_token": 6 * N_opt,
    }


def estimate_training_cost(
    num_parameters: float,
    num_tokens: float,
    gpu_flops: float = 312e12,  # A100 theoretical FP16 TFLOPS
    gpu_utilization: float = 0.4,
    num_gpus: int = 1,
) -> Dict[str, float]:
    """Estimate training time and cost.

    Args:
        num_parameters: Number of model parameters.
        num_tokens: Number of training tokens.
        gpu_flops: Theoretical GPU FLOPS (default A100).
        gpu_utilization: Expected GPU utilization.
        num_gpus: Number of GPUs.

    Returns:
        Dictionary with cost estimates.
    """
    total_flops = 6 * num_parameters * num_tokens
    effective_flops = gpu_flops * gpu_utilization * num_gpus
    training_seconds = total_flops / effective_flops
    training_hours = training_seconds / 3600

    return {
        "total_flops": total_flops,
        "training_hours": training_hours,
        "training_days": training_hours / 24,
        "gpu_hours": training_hours * num_gpus,
    }


def suggest_model_config(
    target_params: float,
    depth_width_ratio: str = "deep_narrow",
) -> Dict[str, int]:
    """Suggest model configuration for target parameter count.

    Args:
        target_params: Target number of parameters.
        depth_width_ratio: Architecture style ("deep_narrow", "shallow_wide", "balanced").

    Returns:
        Dictionary with suggested configuration.
    """
    # MobileLLM uses deep and narrow architecture
    if depth_width_ratio == "deep_narrow":
        # More layers, smaller hidden size
        # Roughly: params ≈ 12 * L * d^2 (simplified)
        hidden_size = int(np.sqrt(target_params / 12 / 32))
        hidden_size = (hidden_size // 64) * 64  # Round to multiple of 64
        num_layers = int(target_params / (12 * hidden_size ** 2))

    elif depth_width_ratio == "shallow_wide":
        # Fewer layers, larger hidden size
        hidden_size = int(np.sqrt(target_params / 12 / 12))
        hidden_size = (hidden_size // 64) * 64
        num_layers = int(target_params / (12 * hidden_size ** 2))

    else:  # balanced
        hidden_size = int(np.sqrt(target_params / 12 / 24))
        hidden_size = (hidden_size // 64) * 64
        num_layers = int(target_params / (12 * hidden_size ** 2))

    # Ensure reasonable values
    num_layers = max(4, min(num_layers, 64))
    hidden_size = max(128, min(hidden_size, 4096))

    # Attention heads (typically hidden_size / 64 or 128)
    num_attention_heads = max(4, hidden_size // 64)
    num_kv_heads = max(1, num_attention_heads // 3)  # GQA

    return {
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "num_kv_heads": num_kv_heads,
        "intermediate_size": hidden_size * 4,
        "estimated_params": 12 * num_layers * hidden_size ** 2,
    }
