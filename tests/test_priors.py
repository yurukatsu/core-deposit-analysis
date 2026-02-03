"""Tests for prior distributions (MCMC and MAP)."""

import numpy as np
import pytest

from coredeposit.estimators.priors import CoreDepositPriors, default_priors
from coredeposit.estimators.map_priors import MAPPriors, default_map_priors


class TestDefaultPriors:
    """Test default_priors function."""

    def test_default_priors_no_covariates(self):
        """Test default priors without covariates."""
        priors = default_priors(p=0, q=0)

        assert priors.lambda_prior is not None
        assert priors.gamma_prior is not None
        assert priors.w1_prior is not None
        assert priors.h_prior is not None
        assert priors.m_prior is not None
        assert priors.sigma_prior is not None
        assert priors.nu_prior is not None
        assert priors.rho_prior is not None
        assert priors.beta_prior is None
        assert priors.w1_a_prior is None
        assert priors.w1_b_prior is None

    def test_default_priors_with_covariates(self):
        """Test default priors with S2 covariates."""
        priors = default_priors(p=3)

        assert priors.beta_prior is not None
        # beta_prior should be p-dimensional (check event_shape)
        assert priors.beta_prior.event_shape == (3,)

    def test_default_priors_with_w1_features(self):
        """Test default priors with w1 features."""
        priors = default_priors(p=0, q=2)

        assert priors.w1_a_prior is not None
        assert priors.w1_b_prior is not None
        assert priors.beta_prior is None

    def test_default_priors_with_both(self):
        """Test default priors with both S2 covariates and w1 features."""
        priors = default_priors(p=3, q=2)

        assert priors.beta_prior is not None
        assert priors.w1_a_prior is not None
        assert priors.w1_b_prior is not None


class TestCoreDepositPriors:
    """Test CoreDepositPriors dataclass."""

    def test_normal_likelihood(self):
        """Test normal_likelihood static method."""
        import jax.numpy as jnp

        loc = jnp.array([1.0, 2.0, 3.0])
        sigma = 0.1

        likelihood = CoreDepositPriors.normal_likelihood(loc, sigma)

        # Should be able to evaluate log_prob
        assert likelihood is not None

    def test_studentt_likelihood(self):
        """Test studentt_likelihood static method."""
        import jax.numpy as jnp

        loc = jnp.array([1.0, 2.0, 3.0])
        sigma = 0.1
        nu = 5.0

        likelihood = CoreDepositPriors.studentt_likelihood(loc, sigma, nu)

        assert likelihood is not None


class TestMAPPriors:
    """Test MAPPriors class."""

    def test_map_priors_neg_log_prob_valid(self):
        """Test neg_log_prob with valid parameters."""
        priors = default_map_priors()

        neg_log_p = priors.neg_log_prob(
            lam=0.05, gam=1.0, w1=0.25, h=0.8, m=12.0
        )

        assert np.isfinite(neg_log_p)
        assert neg_log_p > 0  # Negative log prob is positive

    def test_map_priors_neg_log_prob_invalid_w1(self):
        """Test neg_log_prob with invalid w1."""
        priors = default_map_priors()

        # w1 <= 0
        neg_log_p = priors.neg_log_prob(
            lam=0.05, gam=1.0, w1=0.0, h=0.8, m=12.0
        )
        assert neg_log_p == np.inf

        # w1 >= 1
        neg_log_p = priors.neg_log_prob(
            lam=0.05, gam=1.0, w1=1.0, h=0.8, m=12.0
        )
        assert neg_log_p == np.inf

    def test_map_priors_neg_log_prob_invalid_h(self):
        """Test neg_log_prob with invalid h."""
        priors = default_map_priors()

        # h <= 0
        neg_log_p = priors.neg_log_prob(
            lam=0.05, gam=1.0, w1=0.25, h=0.0, m=12.0
        )
        assert neg_log_p == np.inf

        # h >= 1
        neg_log_p = priors.neg_log_prob(
            lam=0.05, gam=1.0, w1=0.25, h=1.0, m=12.0
        )
        assert neg_log_p == np.inf

    def test_map_priors_neg_log_prob_negative_lam(self):
        """Test neg_log_prob with negative lambda."""
        priors = default_map_priors()

        neg_log_p = priors.neg_log_prob(
            lam=-0.05, gam=1.0, w1=0.25, h=0.8, m=12.0
        )
        assert neg_log_p == np.inf

    def test_map_priors_with_beta(self):
        """Test neg_log_prob with beta coefficients."""
        priors = default_map_priors(p=2)

        neg_log_p = priors.neg_log_prob(
            lam=0.05, gam=1.0, w1=0.25, h=0.8, m=12.0,
            beta=np.array([0.1, -0.2])
        )

        assert np.isfinite(neg_log_p)


class TestDefaultMAPPriors:
    """Test default_map_priors function."""

    def test_default_map_priors_no_covariates(self):
        """Test default MAP priors without covariates."""
        priors = default_map_priors(p=0)

        assert priors.lambda_dist is not None
        assert priors.gamma_dist is not None
        assert priors.w1_dist is not None
        assert priors.h_dist is not None
        assert priors.m_dist is not None
        assert priors.beta_dist is None

    def test_default_map_priors_with_covariates(self):
        """Test default MAP priors with covariates."""
        priors = default_map_priors(p=3)

        assert priors.beta_dist is not None

    def test_lambda_prior_median(self):
        """Test lambda prior median is approximately 0.05."""
        priors = default_map_priors()

        median = priors.lambda_dist.median()
        assert np.isclose(median, np.exp(-3.0), rtol=0.01)

    def test_gamma_prior_median(self):
        """Test gamma prior median is approximately 1.0."""
        priors = default_map_priors()

        median = priors.gamma_dist.median()
        assert np.isclose(median, 1.0, rtol=0.01)

    def test_m_prior_median(self):
        """Test m prior median is approximately 12."""
        priors = default_map_priors()

        median = priors.m_dist.median()
        assert np.isclose(median, np.exp(2.5), rtol=0.01)

    def test_w1_prior_mean(self):
        """Test w1 prior mean is approximately 0.25."""
        priors = default_map_priors()

        mean = priors.w1_dist.mean()
        # Beta(2, 6) has mean = 2/(2+6) = 0.25
        assert np.isclose(mean, 0.25, rtol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
