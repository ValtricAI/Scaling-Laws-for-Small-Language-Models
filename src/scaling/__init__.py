"""Scaling laws analysis utilities."""

from src.scaling.laws import ScalingLaw, ChinchillaScalingLaw, KaplanScalingLaw
from src.scaling.compute_optimal import compute_optimal_allocation
from src.scaling.extrapolation import extrapolate_performance

__all__ = [
    "ScalingLaw",
    "ChinchillaScalingLaw",
    "KaplanScalingLaw",
    "compute_optimal_allocation",
    "extrapolate_performance",
]
