"""Scaling law implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
from scipy.optimize import curve_fit


@dataclass
class ScalingDataPoint:
    """A single data point for scaling law fitting."""

    num_parameters: float  # N
    num_tokens: float  # D
    compute: float  # C = 6 * N * D
    loss: float  # L


class ScalingLaw(ABC):
    """Abstract base class for scaling laws."""

    @abstractmethod
    def predict_loss(self, num_parameters: float, num_tokens: float) -> float:
        """Predict loss given model size and training tokens."""
        pass

    @abstractmethod
    def fit(self, data: List[ScalingDataPoint]) -> None:
        """Fit the scaling law to data."""
        pass

    @abstractmethod
    def optimal_allocation(self, compute_budget: float) -> Tuple[float, float]:
        """Find optimal N and D for a given compute budget."""
        pass


class ChinchillaScalingLaw(ScalingLaw):
    """Chinchilla scaling law: L(N, D) = E + A/N^alpha + B/D^beta

    From "Training Compute-Optimal Large Language Models" (Hoffmann et al., 2022)
    """

    def __init__(
        self,
        E: float = 1.69,
        A: float = 406.4,
        B: float = 410.7,
        alpha: float = 0.34,
        beta: float = 0.28,
    ):
        self.E = E  # Irreducible loss
        self.A = A  # Parameter scaling coefficient
        self.B = B  # Data scaling coefficient
        self.alpha = alpha  # Parameter scaling exponent
        self.beta = beta  # Data scaling exponent

    def predict_loss(self, num_parameters: float, num_tokens: float) -> float:
        """Predict loss given model size and training tokens."""
        return (
            self.E
            + self.A / (num_parameters ** self.alpha)
            + self.B / (num_tokens ** self.beta)
        )

    def fit(self, data: List[ScalingDataPoint]) -> None:
        """Fit the scaling law parameters to data."""

        def loss_func(X, E, A, B, alpha, beta):
            N, D = X
            return E + A / (N ** alpha) + B / (D ** beta)

        N_vals = np.array([d.num_parameters for d in data])
        D_vals = np.array([d.num_tokens for d in data])
        L_vals = np.array([d.loss for d in data])

        popt, _ = curve_fit(
            loss_func,
            (N_vals, D_vals),
            L_vals,
            p0=[self.E, self.A, self.B, self.alpha, self.beta],
            bounds=([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, 1, 1]),
            maxfev=10000,
        )

        self.E, self.A, self.B, self.alpha, self.beta = popt

    def optimal_allocation(self, compute_budget: float) -> Tuple[float, float]:
        """Find optimal N and D for compute budget C = 6*N*D.

        Optimal ratio: N_opt/D_opt = (alpha * B) / (beta * A)
        """
        # Optimal ratio of parameters to data
        ratio = (self.alpha * self.B) / (self.beta * self.A)

        # C = 6 * N * D, so N = C / (6 * D)
        # With N/D = ratio, we get N = ratio * D
        # So C = 6 * ratio * D^2
        # D = sqrt(C / (6 * ratio))
        D_opt = np.sqrt(compute_budget / (6 * ratio))
        N_opt = ratio * D_opt

        return N_opt, D_opt

    def get_params(self) -> dict:
        """Return fitted parameters."""
        return {
            "E": self.E,
            "A": self.A,
            "B": self.B,
            "alpha": self.alpha,
            "beta": self.beta,
        }


class KaplanScalingLaw(ScalingLaw):
    """Kaplan scaling law: L(N) = (N_c / N)^alpha_N

    From "Scaling Laws for Neural Language Models" (Kaplan et al., 2020)
    Simplified power-law form focusing on parameter count.
    """

    def __init__(
        self,
        N_c: float = 8.8e13,
        alpha_N: float = 0.076,
        D_c: float = 5.4e13,
        alpha_D: float = 0.095,
    ):
        self.N_c = N_c
        self.alpha_N = alpha_N
        self.D_c = D_c
        self.alpha_D = alpha_D

    def predict_loss(self, num_parameters: float, num_tokens: float) -> float:
        """Predict loss using power law."""
        L_N = (self.N_c / num_parameters) ** self.alpha_N
        L_D = (self.D_c / num_tokens) ** self.alpha_D
        return L_N + L_D

    def fit(self, data: List[ScalingDataPoint]) -> None:
        """Fit the scaling law to data."""

        def loss_func(X, N_c, alpha_N, D_c, alpha_D):
            N, D = X
            return (N_c / N) ** alpha_N + (D_c / D) ** alpha_D

        N_vals = np.array([d.num_parameters for d in data])
        D_vals = np.array([d.num_tokens for d in data])
        L_vals = np.array([d.loss for d in data])

        popt, _ = curve_fit(
            loss_func,
            (N_vals, D_vals),
            L_vals,
            p0=[self.N_c, self.alpha_N, self.D_c, self.alpha_D],
            maxfev=10000,
        )

        self.N_c, self.alpha_N, self.D_c, self.alpha_D = popt

    def optimal_allocation(self, compute_budget: float) -> Tuple[float, float]:
        """Find optimal allocation (uses Kaplan's recommendation of N ~ C^0.73)."""
        N_opt = compute_budget ** 0.73 / 6
        D_opt = compute_budget / (6 * N_opt)
        return N_opt, D_opt
