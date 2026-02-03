"""Tests for MCMC estimator."""

import numpy as np
import pytest

from coredeposit import CoreDepositData, MCMCEstimator
from coredeposit.model import V_model


class TestMCMCEstimatorBasic:
    """Test basic MCMCEstimator functionality."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data with known parameters."""
        np.random.seed(42)
        T = 30

        # True parameters
        lam, gam, w1, h, m = 0.05, 1.0, 0.3, 0.8, 12.0
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

    def test_mcmc_fit_returns_result(self, synthetic_data):
        """Test that fit returns EstimationResult with samples."""
        estimator = MCMCEstimator(
            num_warmup=50,
            num_samples=50,
            num_chains=1,
            seed=0,
        )
        result = estimator.fit(synthetic_data)

        assert "lambda" in result.params
        assert "gamma" in result.params
        assert "w1" in result.params
        assert "h" in result.params
        assert "m" in result.params
        assert "sigma" in result.params

    def test_mcmc_samples_shape(self, synthetic_data):
        """Test that samples have correct shape."""
        estimator = MCMCEstimator(
            num_warmup=50,
            num_samples=100,
            num_chains=1,
            seed=0,
        )
        result = estimator.fit(synthetic_data)

        # Should have num_samples * num_chains samples
        assert len(result.params["lambda"]) == 100

    def test_mcmc_params_in_valid_range(self, synthetic_data):
        """Test that all samples are in valid ranges."""
        estimator = MCMCEstimator(
            num_warmup=50,
            num_samples=50,
            num_chains=1,
            seed=0,
        )
        result = estimator.fit(synthetic_data)

        assert np.all(result.params["lambda"] > 0)
        assert np.all(result.params["gamma"] > 0)
        assert np.all((result.params["w1"] > 0) & (result.params["w1"] < 1))
        assert np.all((result.params["h"] > 0) & (result.params["h"] < 1))
        assert np.all(result.params["m"] > 0)
        assert np.all(result.params["sigma"] > 0)

    def test_mcmc_diagnostics(self, synthetic_data):
        """Test diagnostics are populated."""
        estimator = MCMCEstimator(
            num_warmup=50,
            num_samples=50,
            num_chains=1,
            seed=0,
        )
        result = estimator.fit(synthetic_data)

        assert "scale" in result.diagnostics
        assert "mcmc" in result.diagnostics


class TestMCMCEstimatorLikelihood:
    """Test different likelihood functions."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data."""
        np.random.seed(42)
        T = 20
        V0 = 100.0
        inflow = 5 + np.random.randn(T + 1) * 0.5
        inflow[0] = 0

        V_true = np.array(V_model(
            lam=0.05, gam=1.0, w1=0.3, h=0.8, m=12.0,
            V0=V0, inflow=inflow, weight=np.ones(T + 1)
        ))
        V_obs = V_true + np.random.randn(T + 1) * 0.5
        V_obs[0] = V0

        return CoreDepositData(V_obs=V_obs, inflow=inflow, V0=V0)

    def test_studentt_likelihood(self, synthetic_data):
        """Test Student-t likelihood."""
        estimator = MCMCEstimator(
            num_warmup=50,
            num_samples=50,
            num_chains=1,
            seed=0,
            likelihood="studentt",
        )
        result = estimator.fit(synthetic_data)

        # Student-t should have nu parameter
        assert "nu" in result.params

    def test_normal_likelihood(self, synthetic_data):
        """Test Normal likelihood."""
        estimator = MCMCEstimator(
            num_warmup=50,
            num_samples=50,
            num_chains=1,
            seed=0,
            likelihood="normal",
        )
        result = estimator.fit(synthetic_data)

        # Normal should not have nu parameter
        assert "nu" not in result.params


class TestMCMCEstimatorWithCovariates:
    """Test MCMCEstimator with covariates."""

    @pytest.fixture
    def synthetic_data_with_z(self):
        """Generate synthetic data with covariates."""
        np.random.seed(42)
        T = 30
        V0 = 100.0

        inflow = 5 + np.random.randn(T + 1) * 0.5
        inflow[0] = 0

        # Single covariate
        z = np.random.randn(T + 1) * 0.5
        beta = 0.2
        weight = np.exp(z * beta)

        V_true = np.array(V_model(
            lam=0.05, gam=1.0, w1=0.3, h=0.8, m=12.0,
            V0=V0, inflow=inflow, weight=weight
        ))
        V_obs = V_true + np.random.randn(T + 1) * 0.3
        V_obs[0] = V0

        return CoreDepositData(V_obs=V_obs, inflow=inflow, V0=V0, z=z)

    def test_mcmc_with_covariate(self, synthetic_data_with_z):
        """Test MCMC with single covariate."""
        estimator = MCMCEstimator(
            num_warmup=50,
            num_samples=50,
            num_chains=1,
            seed=0,
        )
        result = estimator.fit(synthetic_data_with_z)

        assert "beta" in result.params
        assert result.params["beta"].shape == (50, 1)


class TestMCMCEstimatorInitParams:
    """Test MCMCEstimator with initial parameters."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data."""
        np.random.seed(42)
        T = 20
        V0 = 100.0
        inflow = 5 + np.random.randn(T + 1) * 0.5
        inflow[0] = 0

        V_true = np.array(V_model(
            lam=0.05, gam=1.0, w1=0.3, h=0.8, m=12.0,
            V0=V0, inflow=inflow, weight=np.ones(T + 1)
        ))
        V_obs = V_true + np.random.randn(T + 1) * 0.5
        V_obs[0] = V0

        return CoreDepositData(V_obs=V_obs, inflow=inflow, V0=V0)

    def test_mcmc_with_init_params(self, synthetic_data):
        """Test MCMC with initial parameters."""
        init_params = {
            "lambda": 0.05,
            "gamma": 1.0,
            "w1": 0.3,
            "h": 0.8,
            "m": 12.0,
            "sigma": 0.01,
        }

        estimator = MCMCEstimator(
            num_warmup=50,
            num_samples=50,
            num_chains=1,
            seed=0,
            init_params=init_params,
        )
        result = estimator.fit(synthetic_data)

        assert "lambda" in result.params
        assert len(result.params["lambda"]) == 50


class TestMCMCEstimatorARErrors:
    """Test MCMCEstimator with AR(1) errors."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data."""
        np.random.seed(42)
        T = 30
        V0 = 100.0
        inflow = 5 + np.random.randn(T + 1) * 0.5
        inflow[0] = 0

        V_true = np.array(V_model(
            lam=0.05, gam=1.0, w1=0.3, h=0.8, m=12.0,
            V0=V0, inflow=inflow, weight=np.ones(T + 1)
        ))
        V_obs = V_true + np.random.randn(T + 1) * 0.5
        V_obs[0] = V0

        return CoreDepositData(V_obs=V_obs, inflow=inflow, V0=V0)

    def test_mcmc_with_ar_errors(self, synthetic_data):
        """Test MCMC with AR(1) errors."""
        estimator = MCMCEstimator(
            num_warmup=50,
            num_samples=50,
            num_chains=1,
            seed=0,
            ar_errors=True,
        )
        result = estimator.fit(synthetic_data)

        # AR(1) model should have rho parameter
        assert "rho" in result.params
        assert len(result.params["rho"]) == 50
        # rho should be in (-1, 1)
        assert np.all(np.abs(result.params["rho"]) < 1)


class TestMCMCEstimatorPredict:
    """Test MCMCEstimator predict method."""

    @pytest.fixture
    def fitted_result(self):
        """Generate data and fit model."""
        np.random.seed(42)
        T = 20
        V0 = 100.0

        inflow = 5 + np.random.randn(T + 1) * 0.5
        inflow[0] = 0

        V_true = np.array(V_model(
            lam=0.05, gam=1.0, w1=0.3, h=0.8, m=12.0,
            V0=V0, inflow=inflow, weight=np.ones(T + 1)
        ))
        V_obs = V_true + np.random.randn(T + 1) * 0.5
        V_obs[0] = V0

        data = CoreDepositData(V_obs=V_obs, inflow=inflow, V0=V0)
        estimator = MCMCEstimator(
            num_warmup=50,
            num_samples=50,
            num_chains=1,
            seed=0,
        )
        result = estimator.fit(data)

        return data, estimator, result

    def test_predict_mean(self, fitted_result):
        """Test prediction returns mean."""
        data, estimator, result = fitted_result
        V_pred = estimator.predict(data, result)

        assert V_pred.shape == data.V_obs.shape

    def test_predict_initial_balance(self, fitted_result):
        """Test prediction at t=0."""
        data, estimator, result = fitted_result
        V_pred = estimator.predict(data, result)

        assert V_pred[0] == data.V0

    def test_predict_with_uncertainty(self, fitted_result):
        """Test prediction with uncertainty."""
        data, estimator, result = fitted_result
        pred = estimator.predict(data, result, uncertainty=True)

        assert isinstance(pred, dict)
        assert "mean" in pred
        assert "samples" in pred
        assert "lower" in pred
        assert "upper" in pred

    def test_predict_samples_shape(self, fitted_result):
        """Test prediction samples shape."""
        data, estimator, result = fitted_result
        pred = estimator.predict(data, result, uncertainty=True)

        # Should have n_samples x T+1
        assert pred["samples"].shape == (50, len(data.V_obs))

    def test_predict_credible_interval(self, fitted_result):
        """Test credible interval."""
        data, estimator, result = fitted_result

        pred_95 = estimator.predict(data, result, uncertainty=True, ci_prob=0.95)
        pred_90 = estimator.predict(data, result, uncertainty=True, ci_prob=0.90)

        # 95% CI should be wider than 90% CI
        width_95 = pred_95["upper"] - pred_95["lower"]
        width_90 = pred_90["upper"] - pred_90["lower"]
        assert np.all(width_95 >= width_90)


class TestMCMCEstimatorMultipleChains:
    """Test MCMCEstimator with multiple chains."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data."""
        np.random.seed(42)
        T = 20
        V0 = 100.0
        inflow = 5 + np.random.randn(T + 1) * 0.5
        inflow[0] = 0

        V_true = np.array(V_model(
            lam=0.05, gam=1.0, w1=0.3, h=0.8, m=12.0,
            V0=V0, inflow=inflow, weight=np.ones(T + 1)
        ))
        V_obs = V_true + np.random.randn(T + 1) * 0.5
        V_obs[0] = V0

        return CoreDepositData(V_obs=V_obs, inflow=inflow, V0=V0)

    def test_mcmc_multiple_chains(self, synthetic_data):
        """Test MCMC with multiple chains."""
        estimator = MCMCEstimator(
            num_warmup=50,
            num_samples=50,
            num_chains=2,
            seed=0,
        )
        result = estimator.fit(synthetic_data)

        # Should have num_samples * num_chains samples
        assert len(result.params["lambda"]) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
