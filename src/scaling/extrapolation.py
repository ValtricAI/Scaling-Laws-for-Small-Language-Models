"""Extrapolation utilities for scaling law predictions."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class ExtrapolationResult:
    """Result of scaling law extrapolation."""

    target_compute: float
    predicted_loss: float
    optimal_params: float
    optimal_tokens: float
    confidence_interval: Optional[Tuple[float, float]] = None


def extrapolate_performance(
    scaling_law,
    target_computes: List[float],
    include_confidence: bool = True,
    confidence_level: float = 0.95,
) -> List[ExtrapolationResult]:
    """Extrapolate model performance to new compute budgets.

    Args:
        scaling_law: Fitted scaling law object.
        target_computes: List of compute budgets to extrapolate to.
        include_confidence: Whether to include confidence intervals.
        confidence_level: Confidence level for intervals.

    Returns:
        List of extrapolation results.
    """
    results = []

    for compute in target_computes:
        N_opt, D_opt = scaling_law.optimal_allocation(compute)
        predicted_loss = scaling_law.predict_loss(N_opt, D_opt)

        result = ExtrapolationResult(
            target_compute=compute,
            predicted_loss=predicted_loss,
            optimal_params=N_opt,
            optimal_tokens=D_opt,
        )

        results.append(result)

    return results


def compare_scaling_laws(
    data_points: List[dict],
    compute_range: Tuple[float, float],
    num_points: int = 20,
) -> Dict[str, List[float]]:
    """Compare predictions from different scaling laws.

    Args:
        data_points: Training data for fitting.
        compute_range: (min_compute, max_compute) range.
        num_points: Number of points to sample.

    Returns:
        Dictionary with predictions from each scaling law.
    """
    from src.scaling.laws import ChinchillaScalingLaw, KaplanScalingLaw, ScalingDataPoint

    # Convert to ScalingDataPoint objects
    points = [
        ScalingDataPoint(
            num_parameters=d["num_parameters"],
            num_tokens=d["num_tokens"],
            compute=d.get("compute", 6 * d["num_parameters"] * d["num_tokens"]),
            loss=d["loss"],
        )
        for d in data_points
    ]

    # Fit both scaling laws
    chinchilla = ChinchillaScalingLaw()
    kaplan = KaplanScalingLaw()

    chinchilla.fit(points)
    kaplan.fit(points)

    # Generate compute values
    computes = np.logspace(
        np.log10(compute_range[0]),
        np.log10(compute_range[1]),
        num_points,
    )

    # Get predictions
    chinchilla_preds = []
    kaplan_preds = []

    for c in computes:
        N_c, D_c = chinchilla.optimal_allocation(c)
        N_k, D_k = kaplan.optimal_allocation(c)

        chinchilla_preds.append(chinchilla.predict_loss(N_c, D_c))
        kaplan_preds.append(kaplan.predict_loss(N_k, D_k))

    return {
        "compute": computes.tolist(),
        "chinchilla": chinchilla_preds,
        "kaplan": kaplan_preds,
    }


def analyze_efficiency_frontier(
    model_results: List[dict],
) -> Dict[str, List[dict]]:
    """Analyze the efficiency frontier of trained models.

    Args:
        model_results: List of dicts with model results.
            Each dict should have: name, num_parameters, loss, compute

    Returns:
        Dictionary with frontier analysis.
    """
    # Sort by compute
    sorted_results = sorted(model_results, key=lambda x: x["compute"])

    # Find Pareto frontier
    frontier = []
    best_loss = float("inf")

    for result in sorted_results:
        if result["loss"] < best_loss:
            frontier.append(result)
            best_loss = result["loss"]

    # Calculate efficiency metrics
    for result in model_results:
        result["loss_per_flop"] = result["loss"] / result["compute"]
        result["is_frontier"] = result in frontier

    return {
        "all_models": model_results,
        "frontier_models": frontier,
        "num_frontier": len(frontier),
    }


def project_to_target_loss(
    scaling_law,
    target_loss: float,
    max_compute: float = 1e24,
) -> Dict[str, float]:
    """Project compute needed to achieve a target loss.

    Args:
        scaling_law: Fitted scaling law.
        target_loss: Target loss to achieve.
        max_compute: Maximum compute to search.

    Returns:
        Dictionary with projected requirements.
    """
    from scipy.optimize import brentq

    def loss_diff(log_compute):
        compute = 10 ** log_compute
        N, D = scaling_law.optimal_allocation(compute)
        return scaling_law.predict_loss(N, D) - target_loss

    # Binary search for compute
    try:
        log_compute = brentq(loss_diff, 15, np.log10(max_compute))
        compute = 10 ** log_compute
        N, D = scaling_law.optimal_allocation(compute)

        return {
            "target_loss": target_loss,
            "required_compute": compute,
            "optimal_params": N,
            "optimal_tokens": D,
            "achievable": True,
        }
    except ValueError:
        return {
            "target_loss": target_loss,
            "achievable": False,
            "message": "Target loss not achievable within compute budget",
        }
