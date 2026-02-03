"""Tests for survival function computations."""

import jax.numpy as jnp
import numpy as np
import pytest

from coredeposit.model.survival import S2_matrix, S2_init_vector


class TestS2Matrix:
    """Test S2_matrix function."""

    def test_s2_matrix_shape(self):
        """Test that S2 matrix has correct shape."""
        T = 20
        weight = jnp.ones(T + 1)
        S2 = S2_matrix(lam=0.05, gam=1.0, T=T, weight=weight)

        assert S2.shape == (T + 1, T + 1)

    def test_s2_matrix_lower_triangular(self):
        """Test that S2 matrix is lower triangular (for i >= 1)."""
        T = 10
        weight = jnp.ones(T + 1)
        S2 = S2_matrix(lam=0.05, gam=1.0, T=T, weight=weight)

        # Upper triangle should be zero
        for t in range(T + 1):
            for i in range(t + 1, T + 1):
                assert S2[t, i] == 0.0

    def test_s2_matrix_column_zero_unused(self):
        """Test that column 0 is all zeros (no cohort at time 0)."""
        T = 10
        weight = jnp.ones(T + 1)
        S2 = S2_matrix(lam=0.05, gam=1.0, T=T, weight=weight)

        assert jnp.all(S2[:, 0] == 0.0)

    def test_s2_matrix_survival_decreasing(self):
        """Test that survival decreases over time for each cohort."""
        T = 20
        weight = jnp.ones(T + 1)
        S2 = S2_matrix(lam=0.05, gam=1.0, T=T, weight=weight)

        # For each cohort i, survival should decrease as t increases
        for i in range(1, T + 1):
            survival_over_time = S2[i:, i]
            diffs = jnp.diff(survival_over_time)
            assert jnp.all(diffs <= 0), f"Survival not decreasing for cohort {i}"

    def test_s2_matrix_survival_bounded(self):
        """Test that survival is in [0, 1]."""
        T = 20
        weight = jnp.ones(T + 1)
        S2 = S2_matrix(lam=0.05, gam=1.0, T=T, weight=weight)

        assert jnp.all(S2 >= 0)
        assert jnp.all(S2 <= 1)

    def test_s2_matrix_diagonal_less_than_one(self):
        """Test that diagonal (first period survival) is < 1."""
        T = 20
        weight = jnp.ones(T + 1)
        S2 = S2_matrix(lam=0.05, gam=1.0, T=T, weight=weight)

        # Diagonal elements (t=i, i>=1) should be < 1 due to hazard
        diagonal = jnp.array([S2[i, i] for i in range(1, T + 1)])
        assert jnp.all(diagonal < 1.0)

    def test_s2_matrix_with_covariates(self):
        """Test S2 matrix with covariate weights."""
        T = 20
        # Higher weight -> faster exit
        weight = jnp.exp(0.1 * jnp.arange(T + 1))

        S2_weighted = S2_matrix(lam=0.05, gam=1.0, T=T, weight=weight)
        S2_baseline = S2_matrix(lam=0.05, gam=1.0, T=T, weight=jnp.ones(T + 1))

        # With higher weights, survival should be lower
        # Check at a specific point
        assert S2_weighted[T, 1] < S2_baseline[T, 1]

    def test_s2_matrix_higher_lambda_lower_survival(self):
        """Test that higher lambda leads to lower survival."""
        T = 20
        weight = jnp.ones(T + 1)

        S2_low = S2_matrix(lam=0.02, gam=1.0, T=T, weight=weight)
        S2_high = S2_matrix(lam=0.10, gam=1.0, T=T, weight=weight)

        # Higher lambda -> lower survival
        assert jnp.all(S2_high[1:, 1:] <= S2_low[1:, 1:])


class TestS2InitVector:
    """Test S2_init_vector function."""

    def test_s2_init_shape(self):
        """Test that S2 init vector has correct shape."""
        T = 20
        weight = jnp.ones(T + 1)
        S2_init = S2_init_vector(lam=0.05, gam=1.0, T=T, m=12, weight=weight)

        assert S2_init.shape == (T + 1,)

    def test_s2_init_decreasing(self):
        """Test that initial cohort survival decreases over time."""
        T = 20
        weight = jnp.ones(T + 1)
        S2_init = S2_init_vector(lam=0.05, gam=1.0, T=T, m=12, weight=weight)

        diffs = jnp.diff(S2_init)
        assert jnp.all(diffs <= 0)

    def test_s2_init_bounded(self):
        """Test that survival is in (0, 1]."""
        T = 20
        weight = jnp.ones(T + 1)
        S2_init = S2_init_vector(lam=0.05, gam=1.0, T=T, m=12, weight=weight)

        assert jnp.all(S2_init > 0)
        assert jnp.all(S2_init <= 1)

    def test_s2_init_higher_m_lower_survival(self):
        """Test that higher initial age leads to lower survival."""
        T = 20
        weight = jnp.ones(T + 1)

        # For gamma > 1, older initial balance has higher hazard
        S2_young = S2_init_vector(lam=0.05, gam=1.5, T=T, m=6, weight=weight)
        S2_old = S2_init_vector(lam=0.05, gam=1.5, T=T, m=24, weight=weight)

        # With increasing hazard (gamma > 1), older balance exits faster
        assert S2_old[-1] < S2_young[-1]

    def test_s2_init_with_covariates(self):
        """Test S2 init with covariate weights."""
        T = 20
        weight_high = jnp.ones(T + 1) * 2.0

        S2_base = S2_init_vector(lam=0.05, gam=1.0, T=T, m=12, weight=jnp.ones(T + 1))
        S2_high = S2_init_vector(lam=0.05, gam=1.0, T=T, m=12, weight=weight_high)

        # Higher weight -> lower survival
        assert jnp.all(S2_high <= S2_base)

    def test_s2_init_gamma_effect(self):
        """Test effect of gamma on initial balance survival."""
        T = 50
        weight = jnp.ones(T + 1)
        m = 12  # Start with deposits already aged 12 months

        S2_gamma_1 = S2_init_vector(lam=0.05, gam=1.0, T=T, m=m, weight=weight)
        S2_gamma_2 = S2_init_vector(lam=0.05, gam=2.0, T=T, m=m, weight=weight)

        # Both should be decreasing
        assert jnp.all(jnp.diff(S2_gamma_1) <= 0)
        assert jnp.all(jnp.diff(S2_gamma_2) <= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
