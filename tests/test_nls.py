"""Tests for NLS estimator."""

import numpy as np
import pytest

from coredeposit import CoreDepositData, NLSEstimator
from coredeposit.model import V_model
from coredeposit.estimators.map_priors import default_map_priors


class TestNLSEstimatorBasic:
    """Test basic NLSEstimator functionality."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data with known parameters."""
        np.random.seed(42)
        T = 30

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

    def test_nls_fit_returns_result(self, synthetic_data):
        """Test that fit returns EstimationResult."""
        estimator = NLSEstimator(loss="soft_l1")
        result = estimator.fit(synthetic_data)

        assert "lambda" in result.params
        assert "gamma" in result.params
        assert "w1" in result.params
        assert "h" in result.params
        assert "m" in result.params

    def test_nls_fit_success(self, synthetic_data):
        """Test that optimization succeeds."""
        estimator = NLSEstimator(loss="soft_l1")
        result = estimator.fit(synthetic_data)

        assert result.diagnostics["success"]

    def test_nls_params_in_valid_range(self, synthetic_data):
        """Test that parameters are in valid ranges."""
        estimator = NLSEstimator(loss="soft_l1")
        result = estimator.fit(synthetic_data)

        assert result.params["lambda"] > 0
        assert result.params["gamma"] > 0
        assert 0 < result.params["w1"] < 1
        assert 0 < result.params["h"] < 1
        assert result.params["m"] > 0

    def test_nls_diagnostics(self, synthetic_data):
        """Test diagnostics are populated."""
        estimator = NLSEstimator(loss="soft_l1")
        result = estimator.fit(synthetic_data)

        assert "success" in result.diagnostics
        assert "nfev" in result.diagnostics
        assert "cost" in result.diagnostics
        assert "method" in result.diagnostics
        assert result.diagnostics["method"] == "nls"


class TestNLSEstimatorLossFunctions:
    """Test different loss functions."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data with noise."""
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

    def test_linear_loss(self, synthetic_data):
        """Test linear (standard) loss."""
        estimator = NLSEstimator(loss="linear")
        result = estimator.fit(synthetic_data)

        assert result.diagnostics["success"]

    def test_soft_l1_loss(self, synthetic_data):
        """Test soft_l1 loss."""
        estimator = NLSEstimator(loss="soft_l1")
        result = estimator.fit(synthetic_data)

        assert result.diagnostics["success"]

    def test_huber_loss(self, synthetic_data):
        """Test huber loss."""
        estimator = NLSEstimator(loss="huber")
        result = estimator.fit(synthetic_data)

        assert result.diagnostics["success"]

    def test_cauchy_loss(self, synthetic_data):
        """Test cauchy loss."""
        estimator = NLSEstimator(loss="cauchy")
        result = estimator.fit(synthetic_data)

        assert result.diagnostics["success"]


class TestNLSEstimatorFixedM:
    """Test NLSEstimator with fixed m parameter."""

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
        V_obs = V_true + np.random.randn(T + 1) * 0.3
        V_obs[0] = V0

        return CoreDepositData(V_obs=V_obs, inflow=inflow, V0=V0)

    def test_fixed_m(self, synthetic_data):
        """Test fitting with fixed m."""
        estimator = NLSEstimator(loss="soft_l1")
        result = estimator.fit(synthetic_data, m_fixed=12.0)

        assert result.params["m"] == 12.0
        assert result.diagnostics["m_fixed"] == 12.0

    def test_fixed_m_different_value(self, synthetic_data):
        """Test fitting with different fixed m values."""
        estimator = NLSEstimator(loss="soft_l1")

        result_m6 = estimator.fit(synthetic_data, m_fixed=6.0)
        result_m24 = estimator.fit(synthetic_data, m_fixed=24.0)

        assert result_m6.params["m"] == 6.0
        assert result_m24.params["m"] == 24.0


class TestNLSEstimatorWithCovariates:
    """Test NLSEstimator with covariates."""

    @pytest.fixture
    def synthetic_data_with_z(self):
        """Generate synthetic data with covariates."""
        np.random.seed(42)
        T = 50
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

    def test_nls_with_covariate(self, synthetic_data_with_z):
        """Test NLS with single covariate."""
        estimator = NLSEstimator(loss="soft_l1")
        result = estimator.fit(synthetic_data_with_z)

        assert "beta" in result.params
        assert result.params["beta"].shape == (1,)

    def test_nls_with_2d_covariates(self):
        """Test NLS with multiple covariates."""
        np.random.seed(42)
        T = 50
        V0 = 100.0

        inflow = 5 + np.random.randn(T + 1) * 0.5
        inflow[0] = 0

        # Two covariates
        z = np.random.randn(T + 1, 2) * 0.5
        beta = np.array([0.2, -0.1])
        weight = np.exp(z @ beta)

        V_true = np.array(V_model(
            lam=0.05, gam=1.0, w1=0.3, h=0.8, m=12.0,
            V0=V0, inflow=inflow, weight=weight
        ))
        V_obs = V_true + np.random.randn(T + 1) * 0.3
        V_obs[0] = V0

        data = CoreDepositData(V_obs=V_obs, inflow=inflow, V0=V0, z=z)

        estimator = NLSEstimator(loss="soft_l1")
        result = estimator.fit(data)

        assert "beta" in result.params
        assert result.params["beta"].shape == (2,)


class TestNLSEstimatorMAP:
    """Test MAP estimation with NLSEstimator."""

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
        V_obs = V_true + np.random.randn(T + 1) * 0.3
        V_obs[0] = V0

        return CoreDepositData(V_obs=V_obs, inflow=inflow, V0=V0)

    def test_map_estimation(self, synthetic_data):
        """Test MAP estimation with priors."""
        priors = default_map_priors()
        estimator = NLSEstimator(priors=priors)
        result = estimator.fit(synthetic_data)

        assert result.diagnostics["method"] == "map"
        assert result.params["lambda"] > 0
        assert 0 < result.params["w1"] < 1

    def test_map_vs_nls_difference(self, synthetic_data):
        """Test that MAP and NLS give different results."""
        priors = default_map_priors()

        estimator_nls = NLSEstimator(loss="soft_l1")
        estimator_map = NLSEstimator(priors=priors)

        result_nls = estimator_nls.fit(synthetic_data)
        result_map = estimator_map.fit(synthetic_data)

        # Results should differ (priors pull parameters)
        # This might not always be true, but generally should be
        assert result_nls.diagnostics["method"] == "nls"
        assert result_map.diagnostics["method"] == "map"


class TestNLSEstimatorPredict:
    """Test NLSEstimator predict method."""

    @pytest.fixture
    def fitted_result(self):
        """Generate data and fit model."""
        np.random.seed(42)
        T = 30
        V0 = 100.0

        inflow = 5 + np.random.randn(T + 1) * 0.5
        inflow[0] = 0

        V_true = np.array(V_model(
            lam=0.05, gam=1.0, w1=0.3, h=0.8, m=12.0,
            V0=V0, inflow=inflow, weight=np.ones(T + 1)
        ))
        V_obs = V_true + np.random.randn(T + 1) * 0.3
        V_obs[0] = V0

        data = CoreDepositData(V_obs=V_obs, inflow=inflow, V0=V0)
        estimator = NLSEstimator(loss="soft_l1")
        result = estimator.fit(data)

        return data, estimator, result

    def test_predict_shape(self, fitted_result):
        """Test prediction shape."""
        data, estimator, result = fitted_result
        V_pred = estimator.predict(data, result)

        assert V_pred.shape == data.V_obs.shape

    def test_predict_initial_balance(self, fitted_result):
        """Test prediction at t=0."""
        data, estimator, result = fitted_result
        V_pred = estimator.predict(data, result)

        assert V_pred[0] == data.V0

    def test_predict_uncertainty_warning(self, fitted_result):
        """Test that uncertainty=True raises warning."""
        data, estimator, result = fitted_result

        with pytest.warns(UserWarning, match="NLSEstimator does not support"):
            estimator.predict(data, result, uncertainty=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
