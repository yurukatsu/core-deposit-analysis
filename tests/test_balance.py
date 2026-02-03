"""Tests for V_model balance prediction function."""

import jax.numpy as jnp
import numpy as np
import pytest

from coredeposit.model.balance import V_model


class TestVModel:
    """Test V_model function."""

    def test_v_model_shape(self):
        """Test output shape."""
        T = 20
        inflow = np.ones(T + 1) * 10.0
        inflow[0] = 0.0
        weight = np.ones(T + 1)

        V = V_model(
            lam=0.05, gam=1.0, w1=0.3, h=0.8, m=12,
            V0=100.0, inflow=inflow, weight=weight
        )

        assert V.shape == (T + 1,)

    def test_v_model_initial_balance(self):
        """Test that V[0] = V0."""
        T = 20
        inflow = np.ones(T + 1) * 10.0
        inflow[0] = 0.0
        weight = np.ones(T + 1)
        V0 = 100.0

        V = V_model(
            lam=0.05, gam=1.0, w1=0.3, h=0.8, m=12,
            V0=V0, inflow=inflow, weight=weight
        )

        assert V[0] == V0

    def test_v_model_no_inflow_decreasing(self):
        """Test that balance decreases when there's no inflow."""
        T = 20
        inflow = np.zeros(T + 1)
        weight = np.ones(T + 1)
        V0 = 100.0

        V = V_model(
            lam=0.05, gam=1.0, w1=0.3, h=0.8, m=12,
            V0=V0, inflow=inflow, weight=weight
        )

        # Balance should decrease over time (only initial balance surviving)
        diffs = np.diff(np.array(V))
        assert np.all(diffs <= 0)

    def test_v_model_positive_values(self):
        """Test that balance is always positive."""
        T = 50
        inflow = np.ones(T + 1) * 5.0
        inflow[0] = 0.0
        weight = np.ones(T + 1)

        V = V_model(
            lam=0.05, gam=1.0, w1=0.3, h=0.8, m=12,
            V0=100.0, inflow=inflow, weight=weight
        )

        assert np.all(np.array(V) > 0)

    def test_v_model_high_w1_lower_retention(self):
        """Test that higher w1 leads to lower balance retention."""
        T = 20
        inflow = np.ones(T + 1) * 10.0
        inflow[0] = 0.0
        weight = np.ones(T + 1)

        V_low_w1 = V_model(
            lam=0.05, gam=1.0, w1=0.1, h=0.8, m=12,
            V0=100.0, inflow=inflow, weight=weight
        )

        V_high_w1 = V_model(
            lam=0.05, gam=1.0, w1=0.5, h=0.8, m=12,
            V0=100.0, inflow=inflow, weight=weight
        )

        # With higher w1 (more transactional), less balance retained
        assert np.array(V_high_w1)[-1] < np.array(V_low_w1)[-1]

    def test_v_model_high_lambda_lower_balance(self):
        """Test that higher lambda leads to lower balance."""
        T = 20
        inflow = np.ones(T + 1) * 10.0
        inflow[0] = 0.0
        weight = np.ones(T + 1)

        V_low_lam = V_model(
            lam=0.02, gam=1.0, w1=0.3, h=0.8, m=12,
            V0=100.0, inflow=inflow, weight=weight
        )

        V_high_lam = V_model(
            lam=0.10, gam=1.0, w1=0.3, h=0.8, m=12,
            V0=100.0, inflow=inflow, weight=weight
        )

        # Higher lambda -> faster exit -> lower balance
        assert np.array(V_high_lam)[-1] < np.array(V_low_lam)[-1]

    def test_v_model_with_covariates(self):
        """Test V_model with covariate weights."""
        T = 20
        inflow = np.ones(T + 1) * 10.0
        inflow[0] = 0.0

        weight_base = np.ones(T + 1)
        weight_high = np.ones(T + 1) * 2.0  # Higher hazard

        V_base = V_model(
            lam=0.05, gam=1.0, w1=0.3, h=0.8, m=12,
            V0=100.0, inflow=inflow, weight=weight_base
        )

        V_high = V_model(
            lam=0.05, gam=1.0, w1=0.3, h=0.8, m=12,
            V0=100.0, inflow=inflow, weight=weight_high
        )

        # Higher weight -> lower balance
        assert np.array(V_high)[-1] < np.array(V_base)[-1]

    def test_v_model_array_w1(self):
        """Test V_model with time-varying w1."""
        T = 20
        inflow = np.ones(T + 1) * 10.0
        inflow[0] = 0.0
        weight = np.ones(T + 1)
        w1_array = np.linspace(0.2, 0.4, T + 1)

        V = V_model(
            lam=0.05, gam=1.0, w1=w1_array, h=0.8, m=12,
            V0=100.0, inflow=inflow, weight=weight
        )

        assert V.shape == (T + 1,)
        assert V[0] == 100.0

    def test_v_model_scalar_vs_array_w1_consistent(self):
        """Test that constant array w1 gives same result as scalar."""
        T = 20
        inflow = np.ones(T + 1) * 10.0
        inflow[0] = 0.0
        weight = np.ones(T + 1)

        V_scalar = V_model(
            lam=0.05, gam=1.0, w1=0.3, h=0.8, m=12,
            V0=100.0, inflow=inflow, weight=weight
        )

        V_array = V_model(
            lam=0.05, gam=1.0, w1=np.full(T + 1, 0.3), h=0.8, m=12,
            V0=100.0, inflow=inflow, weight=weight
        )

        assert np.allclose(np.array(V_scalar), np.array(V_array))

    def test_v_model_varying_inflow(self):
        """Test with varying inflow pattern."""
        T = 20
        np.random.seed(42)
        inflow = np.abs(np.random.randn(T + 1) * 5) + 5
        inflow[0] = 0.0
        weight = np.ones(T + 1)

        V = V_model(
            lam=0.05, gam=1.0, w1=0.3, h=0.8, m=12,
            V0=100.0, inflow=inflow, weight=weight
        )

        assert V.shape == (T + 1,)
        assert np.all(np.array(V) > 0)

    def test_v_model_zero_w1(self):
        """Test with w1=0 (all sticky deposits)."""
        T = 20
        inflow = np.ones(T + 1) * 10.0
        inflow[0] = 0.0
        weight = np.ones(T + 1)

        V = V_model(
            lam=0.05, gam=1.0, w1=0.0, h=0.8, m=12,
            V0=100.0, inflow=inflow, weight=weight
        )

        assert V.shape == (T + 1,)
        # All deposits are sticky, so balance should accumulate more
        assert V[-1] > V[0]  # Assuming inflow exceeds exit

    def test_v_model_high_h_fast_transactional_exit(self):
        """Test that high h leads to fast transactional exit."""
        T = 20
        inflow = np.ones(T + 1) * 10.0
        inflow[0] = 0.0
        weight = np.ones(T + 1)

        V_low_h = V_model(
            lam=0.05, gam=1.0, w1=0.5, h=0.3, m=12,
            V0=100.0, inflow=inflow, weight=weight
        )

        V_high_h = V_model(
            lam=0.05, gam=1.0, w1=0.5, h=0.9, m=12,
            V0=100.0, inflow=inflow, weight=weight
        )

        # Higher h -> more transactional deposits exit -> lower balance
        assert np.array(V_high_h)[-1] < np.array(V_low_h)[-1]


class TestVModelCustomS1:
    """Test V_model with custom S1 term function."""

    def test_v_model_custom_s1_term_fn(self):
        """Test V_model with custom S1 term function."""
        T = 10
        inflow = np.ones(T + 1) * 10.0
        inflow[0] = 0.0
        weight = np.ones(T + 1)

        # Custom S1 term function that returns zeros
        def custom_s1_fn(inflow, w1, h, T):
            return jnp.zeros(T + 1)

        V = V_model(
            lam=0.05, gam=1.0, w1=0.3, h=0.8, m=12,
            V0=100.0, inflow=inflow, weight=weight,
            s1_term_fn=custom_s1_fn
        )

        # With zero S1 contribution, balance should still be positive
        # from sticky deposits and initial balance
        assert V.shape == (T + 1,)
        assert np.all(np.array(V) > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
