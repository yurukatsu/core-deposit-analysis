"""Tests for type definitions."""

import numpy as np
import pytest

from coredeposit.types import CoreDepositData, EstimationResult


class TestCoreDepositData:
    """Test CoreDepositData dataclass."""

    def test_basic_creation(self):
        """Test basic data creation."""
        V_obs = np.array([100.0, 95.0, 90.0])
        inflow = np.array([0.0, 10.0, 8.0])
        V0 = 100.0

        data = CoreDepositData(V_obs=V_obs, inflow=inflow, V0=V0)

        assert np.array_equal(data.V_obs, V_obs)
        assert np.array_equal(data.inflow, inflow)
        assert data.V0 == V0
        assert data.z is None
        assert data.w1_features is None

    def test_with_covariates(self):
        """Test data with z covariates."""
        V_obs = np.array([100.0, 95.0, 90.0])
        inflow = np.array([0.0, 10.0, 8.0])
        V0 = 100.0
        z = np.array([0.1, 0.2, 0.15])

        data = CoreDepositData(V_obs=V_obs, inflow=inflow, V0=V0, z=z)

        assert data.z is not None
        assert np.array_equal(data.z, z)

    def test_with_2d_covariates(self):
        """Test data with 2D z covariates."""
        V_obs = np.array([100.0, 95.0, 90.0])
        inflow = np.array([0.0, 10.0, 8.0])
        V0 = 100.0
        z = np.array([[0.1, 0.2], [0.2, 0.3], [0.15, 0.25]])

        data = CoreDepositData(V_obs=V_obs, inflow=inflow, V0=V0, z=z)

        assert data.z.shape == (3, 2)

    def test_with_w1_features(self):
        """Test data with w1 features."""
        V_obs = np.array([100.0, 95.0, 90.0])
        inflow = np.array([0.0, 10.0, 8.0])
        V0 = 100.0
        w1_features = np.array([0.0, 0.5, -0.3])

        data = CoreDepositData(
            V_obs=V_obs, inflow=inflow, V0=V0,
            w1_features=w1_features
        )

        assert data.w1_features is not None
        assert np.array_equal(data.w1_features, w1_features)

    def test_with_2d_w1_features(self):
        """Test data with 2D w1 features."""
        V_obs = np.array([100.0, 95.0, 90.0])
        inflow = np.array([0.0, 10.0, 8.0])
        V0 = 100.0
        w1_features = np.array([[0.1, 0.2], [0.5, 0.3], [-0.3, 0.1]])

        data = CoreDepositData(
            V_obs=V_obs, inflow=inflow, V0=V0,
            w1_features=w1_features
        )

        assert data.w1_features.shape == (3, 2)

    def test_with_all_fields(self):
        """Test data with all optional fields."""
        V_obs = np.array([100.0, 95.0, 90.0])
        inflow = np.array([0.0, 10.0, 8.0])
        V0 = 100.0
        z = np.array([0.1, 0.2, 0.15])
        w1_features = np.array([0.0, 0.5, -0.3])

        data = CoreDepositData(
            V_obs=V_obs, inflow=inflow, V0=V0,
            z=z, w1_features=w1_features
        )

        assert data.z is not None
        assert data.w1_features is not None


class TestEstimationResult:
    """Test EstimationResult dataclass."""

    def test_nls_result(self):
        """Test NLS estimation result."""
        params = {
            "lambda": 0.05,
            "gamma": 1.0,
            "w1": 0.3,
            "h": 0.8,
            "m": 12.0,
        }
        diagnostics = {
            "success": True,
            "nfev": 50,
            "cost": 0.001,
        }

        result = EstimationResult(params=params, diagnostics=diagnostics)

        assert result.params["lambda"] == 0.05
        assert result.diagnostics["success"] is True

    def test_mcmc_result(self):
        """Test MCMC estimation result."""
        np.random.seed(42)
        params = {
            "lambda": np.random.lognormal(-3.0, 0.3, 100),
            "gamma": np.random.lognormal(0.0, 0.2, 100),
            "w1": np.random.beta(2, 6, 100),
            "h": np.random.beta(2, 6, 100),
            "m": np.random.lognormal(2.5, 0.2, 100),
            "sigma": np.random.exponential(0.05, 100),
        }
        diagnostics = {
            "scale": 1000.0,
        }

        result = EstimationResult(params=params, diagnostics=diagnostics)

        assert len(result.params["lambda"]) == 100
        assert result.diagnostics["scale"] == 1000.0

    def test_result_with_beta(self):
        """Test result with beta coefficients."""
        params = {
            "lambda": 0.05,
            "gamma": 1.0,
            "w1": 0.3,
            "h": 0.8,
            "m": 12.0,
            "beta": np.array([0.1, -0.2]),
        }
        diagnostics = {"success": True}

        result = EstimationResult(params=params, diagnostics=diagnostics)

        assert "beta" in result.params
        assert result.params["beta"].shape == (2,)

    def test_result_with_w1_params(self):
        """Test result with time-varying w1 parameters."""
        params = {
            "lambda": 0.05,
            "gamma": 1.0,
            "w1": 0.3,  # Mean w1
            "w1_a": -0.5,
            "w1_b": np.array([0.2]),
            "h": 0.8,
            "m": 12.0,
        }
        diagnostics = {
            "success": True,
            "w1_time_varying": True,
        }

        result = EstimationResult(params=params, diagnostics=diagnostics)

        assert "w1_a" in result.params
        assert "w1_b" in result.params
        assert result.diagnostics["w1_time_varying"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
