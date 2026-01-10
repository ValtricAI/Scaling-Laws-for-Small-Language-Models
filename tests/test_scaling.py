"""Tests for scaling law implementations."""

import pytest
import numpy as np

from src.scaling.laws import ChinchillaScalingLaw, KaplanScalingLaw, ScalingDataPoint
from src.scaling.compute_optimal import compute_optimal_allocation, estimate_training_cost


class TestChinchillaScalingLaw:
    """Tests for Chinchilla scaling law."""

    def test_init(self):
        law = ChinchillaScalingLaw()
        assert law.E == 1.69
        assert law.alpha == 0.34
        assert law.beta == 0.28

    def test_predict_loss(self):
        law = ChinchillaScalingLaw()
        loss = law.predict_loss(num_parameters=1e9, num_tokens=20e9)
        assert loss > 0
        assert loss < 10  # Reasonable loss range

    def test_optimal_allocation(self):
        law = ChinchillaScalingLaw()
        N_opt, D_opt = law.optimal_allocation(compute_budget=1e20)
        assert N_opt > 0
        assert D_opt > 0
        # Check compute constraint approximately holds
        assert abs(6 * N_opt * D_opt - 1e20) / 1e20 < 0.1

    def test_more_compute_lower_loss(self):
        law = ChinchillaScalingLaw()
        N1, D1 = law.optimal_allocation(1e18)
        N2, D2 = law.optimal_allocation(1e20)
        loss1 = law.predict_loss(N1, D1)
        loss2 = law.predict_loss(N2, D2)
        assert loss2 < loss1  # More compute -> lower loss


class TestKaplanScalingLaw:
    """Tests for Kaplan scaling law."""

    def test_init(self):
        law = KaplanScalingLaw()
        assert law.alpha_N == 0.076

    def test_predict_loss(self):
        law = KaplanScalingLaw()
        loss = law.predict_loss(num_parameters=1e9, num_tokens=20e9)
        assert loss > 0

    def test_optimal_allocation(self):
        law = KaplanScalingLaw()
        N_opt, D_opt = law.optimal_allocation(compute_budget=1e20)
        assert N_opt > 0
        assert D_opt > 0


class TestComputeOptimal:
    """Tests for compute optimal allocation."""

    def test_chinchilla_allocation(self):
        result = compute_optimal_allocation(1e20, scaling_law="chinchilla")
        assert "num_parameters" in result
        assert "num_tokens" in result
        assert result["num_parameters"] > 0
        assert result["num_tokens"] > 0

    def test_kaplan_allocation(self):
        result = compute_optimal_allocation(1e20, scaling_law="kaplan")
        assert "num_parameters" in result
        assert "num_tokens" in result

    def test_training_cost_estimate(self):
        result = estimate_training_cost(
            num_parameters=1e9,
            num_tokens=20e9,
            num_gpus=8,
        )
        assert "total_flops" in result
        assert "training_hours" in result
        assert "gpu_hours" in result
        assert result["total_flops"] == 6 * 1e9 * 20e9


class TestScalingDataPoint:
    """Tests for ScalingDataPoint dataclass."""

    def test_create(self):
        point = ScalingDataPoint(
            num_parameters=1e9,
            num_tokens=20e9,
            compute=6 * 1e9 * 20e9,
            loss=2.5,
        )
        assert point.num_parameters == 1e9
        assert point.loss == 2.5
