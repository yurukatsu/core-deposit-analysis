"""Tests for w1 (transactional proportion) functionality.

Tests cover:
1. w1 module functions (constant, logistic)
2. V_model with time-varying w1
3. NLSEstimator with w1_features
"""

import numpy as np
import pytest

from coredeposit import CoreDepositData, NLSEstimator
from coredeposit.model import V_model
from coredeposit.model.w1 import (
    w1_constant,
    w1_logistic,
    compute_w1_features_ma_deviation,
    compute_w1_features_seasonal,
)


class TestW1Functions:
    """Test w1 module functions."""

    def test_w1_constant(self):
        """Test w1_constant returns constant array."""
        w1 = w1_constant(0.3, T=10)
        assert w1.shape == (11,)
        assert np.allclose(w1, 0.3)

    def test_w1_logistic_scalar_feature(self):
        """Test w1_logistic with single feature."""
        x = np.array([0.0, 1.0, -1.0, 2.0])
        w1 = w1_logistic(a=0.0, b=np.array([1.0]), x=x.reshape(-1, 1))

        assert w1.shape == (4,)
        # At x=0, w1 = sigmoid(0) = 0.5
        assert np.isclose(w1[0], 0.5, atol=1e-6)
        # w1 should be increasing with x
        assert w1[1] > w1[0] > w1[2]

    def test_w1_logistic_multiple_features(self):
        """Test w1_logistic with multiple features."""
        x = np.random.randn(12, 3)
        b = np.array([0.5, -0.3, 0.2])
        w1 = w1_logistic(a=-1.0, b=b, x=x)

        assert w1.shape == (12,)
        assert np.all((w1 > 0) & (w1 < 1))

    def test_w1_logistic_intercept_only(self):
        """Test w1_logistic with zero coefficients (constant w1)."""
        x = np.random.randn(10, 2)
        w1 = w1_logistic(a=-1.0, b=np.zeros(2), x=x)

        expected = 1.0 / (1.0 + np.exp(1.0))  # sigmoid(-1)
        assert np.allclose(w1, expected, atol=1e-6)

    def test_compute_w1_features_ma_deviation(self):
        """Test MA deviation feature computation."""
        inflow = np.array([100, 110, 90, 120, 80, 100, 150, 100, 90, 110, 100, 100, 200])
        features = compute_w1_features_ma_deviation(inflow, window=12)

        assert features.shape == (13,)
        # Large spike at end should have positive deviation
        assert features[-1] > 0
        # Values should be bounded
        assert np.all(np.abs(features) <= 5.0)

    def test_compute_w1_features_seasonal(self):
        """Test seasonal dummy feature computation."""
        dummies = compute_w1_features_seasonal(T=23, start_month=1)

        assert dummies.shape == (24, 11)  # 24 months, 11 dummies (Dec is reference)
        # Each row should have at most one 1 (Dec has all zeros)
        assert np.all(dummies.sum(axis=1) <= 1)
        # Check January dummy
        assert dummies[0, 0] == 1.0  # Jan at t=0
        assert dummies[12, 0] == 1.0  # Jan at t=12


class TestVModelWithTimeVaryingW1:
    """Test V_model with time-varying w1."""

    def test_v_model_constant_w1(self):
        """Test V_model with scalar w1 (baseline)."""
        T = 20
        inflow = np.ones(T + 1) * 10
        inflow[0] = 0
        weight = np.ones(T + 1)

        V = V_model(
            lam=0.05, gam=1.0, w1=0.3, h=0.8, m=12,
            V0=100, inflow=inflow, weight=weight
        )

        assert V.shape == (T + 1,)
        assert V[0] == 100  # Initial balance preserved

    def test_v_model_array_w1(self):
        """Test V_model with array w1."""
        T = 20
        inflow = np.ones(T + 1) * 10
        inflow[0] = 0
        weight = np.ones(T + 1)
        w1_array = np.linspace(0.2, 0.4, T + 1)

        V = V_model(
            lam=0.05, gam=1.0, w1=w1_array, h=0.8, m=12,
            V0=100, inflow=inflow, weight=weight
        )

        assert V.shape == (T + 1,)
        assert V[0] == 100

    def test_v_model_constant_vs_array_w1(self):
        """Test that constant w1 gives same result as array of same value."""
        T = 20
        inflow = np.ones(T + 1) * 10
        inflow[0] = 0
        weight = np.ones(T + 1)

        V_scalar = V_model(
            lam=0.05, gam=1.0, w1=0.3, h=0.8, m=12,
            V0=100, inflow=inflow, weight=weight
        )

        V_array = V_model(
            lam=0.05, gam=1.0, w1=np.full(T + 1, 0.3), h=0.8, m=12,
            V0=100, inflow=inflow, weight=weight
        )

        assert np.allclose(V_scalar, V_array)


class TestNLSEstimatorWithW1Features:
    """Test NLSEstimator with time-varying w1."""

    @pytest.fixture
    def synthetic_data_constant_w1(self):
        """Generate synthetic data with constant w1."""
        np.random.seed(42)
        T = 50

        # True parameters
        lam, gam, w1, h, m = 0.05, 1.2, 0.3, 0.8, 12.0
        V0 = 100.0
        inflow = 5 + np.random.randn(T + 1) * 0.5
        inflow[0] = 0
        weight = np.ones(T + 1)

        # Generate observed data
        V_true = np.array(V_model(
            lam=lam, gam=gam, w1=w1, h=h, m=m,
            V0=V0, inflow=inflow, weight=weight
        ))
        V_obs = V_true + np.random.randn(T + 1) * 0.5
        V_obs[0] = V0

        return CoreDepositData(V_obs=V_obs, inflow=inflow, V0=V0)

    @pytest.fixture
    def synthetic_data_varying_w1(self):
        """Generate synthetic data with time-varying w1."""
        np.random.seed(42)
        T = 50

        # True parameters
        lam, gam, h, m = 0.05, 1.2, 0.8, 12.0
        w1_a, w1_b = -0.5, 0.3
        V0 = 100.0

        inflow = 5 + np.random.randn(T + 1) * 0.5
        inflow[0] = 0
        weight = np.ones(T + 1)

        # Feature: standardized inflow
        w1_features = (inflow - inflow.mean()) / (inflow.std() + 1e-8)
        w1_true = np.array(w1_logistic(w1_a, np.array([w1_b]), w1_features.reshape(-1, 1)))

        # Generate observed data
        V_true = np.array(V_model(
            lam=lam, gam=gam, w1=w1_true, h=h, m=m,
            V0=V0, inflow=inflow, weight=weight
        ))
        V_obs = V_true + np.random.randn(T + 1) * 0.5
        V_obs[0] = V0

        return CoreDepositData(
            V_obs=V_obs, inflow=inflow, V0=V0,
            w1_features=w1_features.reshape(-1, 1)
        )

    def test_nls_constant_w1(self, synthetic_data_constant_w1):
        """Test NLS estimation with constant w1."""
        data = synthetic_data_constant_w1
        estimator = NLSEstimator(loss="soft_l1")
        result = estimator.fit(data)

        assert result.diagnostics["success"]
        assert "w1" in result.params
        assert "w1_a" not in result.params
        assert result.diagnostics["w1_time_varying"] is False

        # Check parameter is in reasonable range
        assert 0 < result.params["w1"] < 1
        assert result.params["lambda"] > 0
        assert result.params["gamma"] > 0

    def test_nls_time_varying_w1(self, synthetic_data_varying_w1):
        """Test NLS estimation with time-varying w1."""
        data = synthetic_data_varying_w1
        estimator = NLSEstimator(loss="soft_l1")
        result = estimator.fit(data)

        assert result.diagnostics["success"]
        assert "w1_a" in result.params
        assert "w1_b" in result.params
        assert result.diagnostics["w1_time_varying"] is True

        # w1_b should have shape (1,) for single feature
        assert result.params["w1_b"].shape == (1,)

        # Mean w1 should be reasonable
        assert 0 < result.params["w1"] < 1

    def test_nls_predict_with_time_varying_w1(self, synthetic_data_varying_w1):
        """Test prediction with time-varying w1."""
        data = synthetic_data_varying_w1
        estimator = NLSEstimator(loss="soft_l1")
        result = estimator.fit(data)

        V_pred = estimator.predict(data, result)

        assert V_pred.shape == data.V_obs.shape
        assert V_pred[0] == data.V0

        # Prediction should be close to observed (within noise)
        rmse = np.sqrt(np.mean((V_pred[1:] - data.V_obs[1:]) ** 2))
        assert rmse < 5.0  # Reasonable fit

    def test_nls_predict_requires_w1_features(self, synthetic_data_varying_w1):
        """Test that predict raises error if w1_features missing for time-varying model."""
        data = synthetic_data_varying_w1
        estimator = NLSEstimator(loss="soft_l1")
        result = estimator.fit(data)

        # Create data without w1_features
        data_no_features = CoreDepositData(
            V_obs=data.V_obs, inflow=data.inflow, V0=data.V0
        )

        with pytest.raises(ValueError, match="w1_features required"):
            estimator.predict(data_no_features, result)


class TestMCMCEstimatorWithW1Features:
    """Test MCMCEstimator with time-varying w1 (quick tests)."""

    @pytest.fixture
    def synthetic_data_varying_w1(self):
        """Generate synthetic data with time-varying w1."""
        np.random.seed(42)
        T = 30  # Short for fast testing

        # True parameters
        lam, gam, h, m = 0.05, 1.2, 0.8, 12.0
        w1_a, w1_b = -0.5, 0.3
        V0 = 100.0

        inflow = 5 + np.random.randn(T + 1) * 0.5
        inflow[0] = 0
        weight = np.ones(T + 1)

        # Feature: standardized inflow
        w1_features = (inflow - inflow.mean()) / (inflow.std() + 1e-8)
        w1_true = np.array(w1_logistic(w1_a, np.array([w1_b]), w1_features.reshape(-1, 1)))

        # Generate observed data
        V_true = np.array(V_model(
            lam=lam, gam=gam, w1=w1_true, h=h, m=m,
            V0=V0, inflow=inflow, weight=weight
        ))
        V_obs = V_true + np.random.randn(T + 1) * 0.5
        V_obs[0] = V0

        return CoreDepositData(
            V_obs=V_obs, inflow=inflow, V0=V0,
            w1_features=w1_features.reshape(-1, 1)
        )

    def test_mcmc_time_varying_w1_smoke(self, synthetic_data_varying_w1):
        """Smoke test: MCMC with time-varying w1 runs without errors."""
        from coredeposit import MCMCEstimator

        data = synthetic_data_varying_w1
        estimator = MCMCEstimator(
            num_warmup=50,
            num_samples=50,
            num_chains=1,
            seed=0,
        )
        result = estimator.fit(data)

        # Check that w1_a and w1_b are in params
        assert "w1_a" in result.params
        assert "w1_b" in result.params
        assert "w1" not in result.params  # Should not have scalar w1

        # Check shapes
        assert result.params["w1_a"].shape == (50,)
        assert result.params["w1_b"].shape == (50, 1)

    def test_mcmc_predict_time_varying_w1(self, synthetic_data_varying_w1):
        """Test MCMC prediction with time-varying w1."""
        from coredeposit import MCMCEstimator

        data = synthetic_data_varying_w1
        estimator = MCMCEstimator(
            num_warmup=50,
            num_samples=50,
            num_chains=1,
            seed=0,
        )
        result = estimator.fit(data)

        # Test prediction
        V_pred = estimator.predict(data, result)
        assert V_pred.shape == data.V_obs.shape

        # Test prediction with uncertainty
        pred = estimator.predict(data, result, uncertainty=True)
        assert "mean" in pred
        assert "samples" in pred
        assert pred["samples"].shape == (50, len(data.V_obs))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
