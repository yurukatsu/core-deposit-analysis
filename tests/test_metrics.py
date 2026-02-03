"""Tests for derived metrics from estimation results."""

import numpy as np
import pytest

from coredeposit.metrics import (
    weibull_median_survival,
    weibull_quantile_survival,
    compute_median_survival,
    compute_quantile_survival,
)
from coredeposit.types import EstimationResult


class TestWeibullMedianSurvival:
    """Test weibull_median_survival function."""

    def test_median_survival_formula(self):
        """Test median survival against known formula."""
        lam = 0.05
        gam = 1.0

        t50 = weibull_median_survival(lam, gam)

        # t_50 = (ln(2))^(1/gamma) / lambda
        expected = np.power(np.log(2), 1.0 / gam) / lam
        assert np.isclose(t50, expected)

    def test_median_survival_exponential(self):
        """Test median survival for exponential distribution (gamma=1)."""
        lam = 0.1
        gam = 1.0

        t50 = weibull_median_survival(lam, gam)

        # For exponential: t_50 = ln(2) / lambda
        expected = np.log(2) / lam
        assert np.isclose(t50, expected)

    def test_median_survival_higher_lambda_shorter(self):
        """Test that higher lambda gives shorter median survival."""
        t50_low = weibull_median_survival(0.02, 1.0)
        t50_high = weibull_median_survival(0.10, 1.0)

        assert t50_high < t50_low

    def test_median_survival_array_input(self):
        """Test with array inputs (MCMC samples)."""
        lam = np.array([0.05, 0.06, 0.04, 0.055])
        gam = np.array([1.0, 1.1, 0.9, 1.05])

        t50 = weibull_median_survival(lam, gam)

        assert t50.shape == (4,)
        assert np.all(t50 > 0)

    def test_median_survival_with_covariates(self):
        """Test median survival with covariates."""
        lam = 0.05
        gam = 1.5  # Use gamma != 1 so covariate effect is visible
        beta = np.array([0.3, -0.2])
        z = np.array([1.0, 0.5])

        t50 = weibull_median_survival(lam, gam, beta, z)
        t50_baseline = weibull_median_survival(lam, gam)

        # With covariates, result should differ from baseline
        # (beta'z = 0.3*1.0 + (-0.2)*0.5 = 0.2 != 0)
        assert not np.isclose(t50, t50_baseline)

    def test_median_survival_positive_beta_shorter(self):
        """Test that positive beta coefficient leads to shorter survival."""
        lam = 0.05
        gam = 1.0
        z = np.array([1.0])

        t50_baseline = weibull_median_survival(lam, gam)
        t50_pos_beta = weibull_median_survival(lam, gam, beta=np.array([0.5]), z=z)
        t50_neg_beta = weibull_median_survival(lam, gam, beta=np.array([-0.5]), z=z)

        # Positive beta -> higher hazard -> shorter survival
        assert t50_pos_beta < t50_baseline < t50_neg_beta


class TestWeibullQuantileSurvival:
    """Test weibull_quantile_survival function."""

    def test_quantile_survival_median(self):
        """Test that q=0.5 gives median."""
        lam = 0.05
        gam = 1.0

        t50_quantile = weibull_quantile_survival(lam, gam, q=0.5)
        t50_median = weibull_median_survival(lam, gam)

        assert np.isclose(t50_quantile, t50_median)

    def test_quantile_survival_ordering(self):
        """Test that higher quantiles give longer survival times."""
        lam = 0.05
        gam = 1.0

        t10 = weibull_quantile_survival(lam, gam, q=0.1)
        t50 = weibull_quantile_survival(lam, gam, q=0.5)
        t90 = weibull_quantile_survival(lam, gam, q=0.9)

        assert t10 < t50 < t90

    def test_quantile_survival_formula(self):
        """Test quantile survival against known formula."""
        lam = 0.05
        gam = 1.2
        q = 0.9

        tq = weibull_quantile_survival(lam, gam, q)

        # t_q = (-ln(1-q))^(1/gamma) / lambda
        expected = np.power(-np.log(1 - q), 1.0 / gam) / lam
        assert np.isclose(tq, expected)


class TestComputeMedianSurvival:
    """Test compute_median_survival function."""

    def test_compute_median_nls_result(self):
        """Test with NLS result (scalars)."""
        result = EstimationResult(
            params={"lambda": 0.05, "gamma": 1.0, "w1": 0.3, "h": 0.8, "m": 12.0},
            diagnostics={"success": True}
        )

        t50 = compute_median_survival(result)

        assert isinstance(t50, float)
        assert t50 > 0

    def test_compute_median_mcmc_result(self):
        """Test with MCMC result (arrays)."""
        np.random.seed(42)
        result = EstimationResult(
            params={
                "lambda": np.random.lognormal(-3.0, 0.3, 100),
                "gamma": np.random.lognormal(0.0, 0.2, 100),
                "w1": np.random.beta(2, 6, 100),
                "h": np.random.beta(2, 6, 100),
                "m": np.random.lognormal(2.5, 0.2, 100),
            },
            diagnostics={"scale": 1.0}
        )

        t50 = compute_median_survival(result)

        assert isinstance(t50, dict)
        assert "mean" in t50
        assert "median" in t50
        assert "samples" in t50
        assert "lower" in t50
        assert "upper" in t50
        assert len(t50["samples"]) == 100

    def test_compute_median_credible_interval(self):
        """Test credible interval computation."""
        np.random.seed(42)
        result = EstimationResult(
            params={
                "lambda": np.random.lognormal(-3.0, 0.3, 1000),
                "gamma": np.ones(1000),
            },
            diagnostics={"scale": 1.0}
        )

        t50_95 = compute_median_survival(result, ci_prob=0.95)
        t50_90 = compute_median_survival(result, ci_prob=0.90)

        # 95% CI should be wider than 90% CI
        width_95 = t50_95["upper"] - t50_95["lower"]
        width_90 = t50_90["upper"] - t50_90["lower"]
        assert width_95 > width_90


class TestComputeQuantileSurvival:
    """Test compute_quantile_survival function."""

    def test_compute_quantile_nls_result(self):
        """Test with NLS result."""
        result = EstimationResult(
            params={"lambda": 0.05, "gamma": 1.0, "w1": 0.3, "h": 0.8, "m": 12.0},
            diagnostics={"success": True}
        )

        t90 = compute_quantile_survival(result, q=0.9)

        assert isinstance(t90, float)
        assert t90 > 0

    def test_compute_quantile_mcmc_result(self):
        """Test with MCMC result."""
        np.random.seed(42)
        result = EstimationResult(
            params={
                "lambda": np.random.lognormal(-3.0, 0.3, 100),
                "gamma": np.random.lognormal(0.0, 0.2, 100),
            },
            diagnostics={"scale": 1.0}
        )

        t90 = compute_quantile_survival(result, q=0.9)

        assert isinstance(t90, dict)
        assert "mean" in t90
        assert len(t90["samples"]) == 100

    def test_compute_quantile_ordering(self):
        """Test quantile ordering."""
        result = EstimationResult(
            params={"lambda": 0.05, "gamma": 1.0},
            diagnostics={"success": True}
        )

        t10 = compute_quantile_survival(result, q=0.1)
        t50 = compute_quantile_survival(result, q=0.5)
        t90 = compute_quantile_survival(result, q=0.9)

        assert t10 < t50 < t90


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
