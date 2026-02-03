"""Tests for S1 (transactional deposit) survival functions."""

import jax.numpy as jnp
import numpy as np
import pytest

from coredeposit.model.s1 import (
    S1_immediate_exit,
    S1_geometric,
    S1_matrix_immediate_exit,
    S1_matrix_geometric,
    S1_term_immediate_exit,
    S1_term_geometric,
)


class TestS1ImmediateExit:
    """Test S1_immediate_exit survival function."""

    def test_s1_at_duration_1(self):
        """Test survival at duration 1."""
        h = 0.8
        s1 = S1_immediate_exit(1, h)
        assert jnp.isclose(s1, 1.0 - h)

    def test_s1_at_duration_2(self):
        """Test survival at duration >= 2 is zero."""
        h = 0.8
        assert S1_immediate_exit(2, h) == 0.0
        assert S1_immediate_exit(10, h) == 0.0

    def test_s1_array_input(self):
        """Test with array input."""
        h = 0.8
        s = jnp.array([1, 2, 3, 1, 5])
        s1 = S1_immediate_exit(s, h)

        expected = jnp.array([0.2, 0.0, 0.0, 0.2, 0.0])
        assert jnp.allclose(s1, expected)


class TestS1Geometric:
    """Test S1_geometric survival function."""

    def test_s1_geometric_formula(self):
        """Test geometric survival formula."""
        h = 0.3
        s = 5

        s1 = S1_geometric(s, h)
        expected = (1 - h) ** s
        assert jnp.isclose(s1, expected)

    def test_s1_geometric_decreasing(self):
        """Test that survival decreases with duration."""
        h = 0.3
        s = jnp.array([1, 2, 3, 4, 5])
        s1 = S1_geometric(s, h)

        assert jnp.all(jnp.diff(s1) < 0)

    def test_s1_geometric_bounded(self):
        """Test survival is in (0, 1]."""
        h = 0.3
        s = jnp.array([1, 5, 10, 20])
        s1 = S1_geometric(s, h)

        assert jnp.all(s1 > 0)
        assert jnp.all(s1 <= 1)


class TestS1MatrixImmediateExit:
    """Test S1_matrix_immediate_exit function."""

    def test_matrix_shape(self):
        """Test matrix shape."""
        T = 10
        S1 = S1_matrix_immediate_exit(T, h=0.8)
        assert S1.shape == (T + 1, T + 1)

    def test_matrix_diagonal_only(self):
        """Test that only diagonal has non-zero values."""
        T = 10
        h = 0.8
        S1 = S1_matrix_immediate_exit(T, h)

        # Off-diagonal should be zero
        for t in range(T + 1):
            for i in range(T + 1):
                if t != i:
                    assert S1[t, i] == 0.0

    def test_matrix_diagonal_values(self):
        """Test diagonal values."""
        T = 10
        h = 0.8
        S1 = S1_matrix_immediate_exit(T, h)

        # Diagonal at i=0 should be 0 (no cohort at t=0)
        assert S1[0, 0] == 0.0

        # Diagonal at i>=1 should be 1-h
        for i in range(1, T + 1):
            assert jnp.isclose(S1[i, i], 1.0 - h)


class TestS1MatrixGeometric:
    """Test S1_matrix_geometric function."""

    def test_matrix_shape(self):
        """Test matrix shape."""
        T = 10
        S1 = S1_matrix_geometric(T, h=0.3)
        assert S1.shape == (T + 1, T + 1)

    def test_matrix_lower_triangular(self):
        """Test that matrix is lower triangular (for i >= 1)."""
        T = 10
        S1 = S1_matrix_geometric(T, h=0.3)

        # Upper triangle should be zero
        for t in range(T + 1):
            for i in range(t + 1, T + 1):
                assert S1[t, i] == 0.0

    def test_matrix_column_zero_unused(self):
        """Test column 0 is all zeros."""
        T = 10
        S1 = S1_matrix_geometric(T, h=0.3)

        assert jnp.all(S1[:, 0] == 0.0)

    def test_matrix_survival_formula(self):
        """Test survival values match formula."""
        T = 10
        h = 0.3
        S1 = S1_matrix_geometric(T, h)

        # Check specific values: S1[t, i] = (1-h)^(t-i+1)
        for i in range(1, T + 1):
            for t in range(i, T + 1):
                s = t - i + 1
                expected = (1 - h) ** s
                assert jnp.isclose(S1[t, i], expected, atol=1e-6)


class TestS1TermImmediateExit:
    """Test S1_term_immediate_exit function."""

    def test_term_shape(self):
        """Test output shape."""
        T = 10
        inflow = jnp.ones(T + 1) * 10.0
        inflow = inflow.at[0].set(0.0)
        w1 = 0.3
        h = 0.8

        term = S1_term_immediate_exit(inflow, w1, h, T)
        assert term.shape == (T + 1,)

    def test_term_first_element_zero(self):
        """Test that first element is zero (no inflow at t=0)."""
        T = 10
        inflow = jnp.ones(T + 1) * 10.0
        term = S1_term_immediate_exit(inflow, w1=0.3, h=0.8, T=T)

        assert term[0] == 0.0

    def test_term_formula(self):
        """Test term computation formula."""
        T = 10
        inflow = jnp.ones(T + 1) * 10.0
        inflow = inflow.at[0].set(0.0)
        w1 = 0.3
        h = 0.8

        term = S1_term_immediate_exit(inflow, w1, h, T)

        # term[t] = w1 * inflow[t] * (1-h) for t >= 1
        expected = w1 * 10.0 * (1 - h)
        assert jnp.allclose(term[1:], expected)

    def test_term_time_varying_w1(self):
        """Test with time-varying w1."""
        T = 10
        inflow = jnp.ones(T + 1) * 10.0
        inflow = inflow.at[0].set(0.0)
        w1 = jnp.linspace(0.2, 0.4, T + 1)
        h = 0.8

        term = S1_term_immediate_exit(inflow, w1, h, T)

        # Check specific values
        for t in range(1, T + 1):
            expected = w1[t] * inflow[t] * (1 - h)
            assert jnp.isclose(term[t], expected)


class TestS1TermGeometric:
    """Test S1_term_geometric function."""

    def test_term_shape(self):
        """Test output shape."""
        T = 10
        inflow = jnp.ones(T + 1) * 10.0
        inflow = inflow.at[0].set(0.0)
        w1 = 0.3
        h = 0.3

        term = S1_term_geometric(inflow, w1, h, T)
        assert term.shape == (T + 1,)

    def test_term_accumulates(self):
        """Test that term includes contributions from past inflows."""
        T = 5
        inflow = jnp.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
        w1 = 0.5
        h = 0.3

        term = S1_term_geometric(inflow, w1, h, T)

        # Only inflow at t=1, should have decaying effect
        # t=1: w1 * 10 * (1-h)^1 = 5 * 0.7
        assert jnp.isclose(term[1], w1 * 10.0 * (1 - h) ** 1)
        # t=2: w1 * 10 * (1-h)^2
        assert jnp.isclose(term[2], w1 * 10.0 * (1 - h) ** 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
