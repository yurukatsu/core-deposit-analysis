"""Tests for Weibull hazard functions."""

import jax.numpy as jnp
import numpy as np
import pytest

from coredeposit.model.hazard import weibull_hazard


class TestWeibullHazard:
    """Test weibull_hazard function."""

    def test_hazard_positive(self):
        """Test that hazard is always positive."""
        durations = jnp.array([0.1, 1.0, 5.0, 10.0, 50.0])
        lam = 0.05
        gam = 1.0

        h = weibull_hazard(durations, lam, gam)

        assert jnp.all(h > 0)

    def test_hazard_exponential_constant(self):
        """Test that hazard is constant for gamma=1 (exponential)."""
        durations = jnp.array([1.0, 5.0, 10.0, 20.0])
        lam = 0.05
        gam = 1.0

        h = weibull_hazard(durations, lam, gam)

        # For gamma=1, h(t) = lam * gam * (lam*t)^(gam-1) = lam * 1 * 1 = lam
        assert jnp.allclose(h, lam)

    def test_hazard_increasing_with_gamma_gt_1(self):
        """Test that hazard increases with duration when gamma > 1."""
        durations = jnp.array([1.0, 5.0, 10.0, 20.0])
        lam = 0.05
        gam = 1.5

        h = weibull_hazard(durations, lam, gam)

        # Hazard should be increasing
        assert jnp.all(jnp.diff(h) > 0)

    def test_hazard_decreasing_with_gamma_lt_1(self):
        """Test that hazard decreases with duration when gamma < 1."""
        durations = jnp.array([1.0, 5.0, 10.0, 20.0])
        lam = 0.05
        gam = 0.5

        h = weibull_hazard(durations, lam, gam)

        # Hazard should be decreasing
        assert jnp.all(jnp.diff(h) < 0)

    def test_hazard_formula(self):
        """Test hazard computation against known formula."""
        duration = 5.0
        lam = 0.1
        gam = 2.0

        h = weibull_hazard(duration, lam, gam)

        # h(t) = lam * gam * (lam * t)^(gam - 1)
        expected = lam * gam * (lam * duration) ** (gam - 1)
        assert jnp.isclose(h, expected)

    def test_hazard_scalar_input(self):
        """Test hazard with scalar input."""
        h = weibull_hazard(10.0, 0.05, 1.2)
        assert h.shape == ()
        assert float(h) > 0

    def test_hazard_array_input(self):
        """Test hazard with array input."""
        durations = jnp.linspace(1, 100, 50)
        h = weibull_hazard(durations, 0.05, 1.2)
        assert h.shape == (50,)

    def test_hazard_clipping_small_duration(self):
        """Test that small durations are clipped to prevent numerical issues."""
        # Very small duration should be clipped
        h = weibull_hazard(1e-12, 0.05, 0.5)
        assert jnp.isfinite(h)

    def test_hazard_zero_duration(self):
        """Test hazard at zero duration (should be clipped)."""
        h = weibull_hazard(0.0, 0.05, 0.5)
        # Should return finite value due to clipping
        assert jnp.isfinite(h)

    def test_hazard_scales_with_lambda(self):
        """Test that hazard scales linearly with lambda when gamma=1."""
        duration = 10.0
        gam = 1.0

        h1 = weibull_hazard(duration, 0.05, gam)
        h2 = weibull_hazard(duration, 0.10, gam)

        # For gamma=1, h = lambda, so ratio should be 2
        assert jnp.isclose(h2 / h1, 2.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
