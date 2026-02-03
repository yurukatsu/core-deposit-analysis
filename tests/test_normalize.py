"""Tests for data normalization functions."""

import numpy as np
import pytest

from coredeposit.normalize import normalize


class TestNormalize:
    """Test normalize function."""

    def test_normalize_basic(self):
        """Test basic normalization."""
        V_obs = np.array([100.0, 110.0, 120.0, 115.0])
        inflow = np.array([0.0, 15.0, 12.0, 10.0])
        V0 = 100.0

        V_norm, I_norm, V0_norm, scale = normalize(V_obs, inflow, V0)

        # Scale should be max(V_obs)
        assert scale == 120.0
        # All values should be divided by scale
        assert np.allclose(V_norm, V_obs / 120.0)
        assert np.allclose(I_norm, inflow / 120.0)
        assert V0_norm == 100.0 / 120.0
        # Max normalized value should be 1.0
        assert np.max(V_norm) == 1.0

    def test_normalize_shapes(self):
        """Test that shapes are preserved."""
        V_obs = np.array([50.0, 60.0, 70.0, 80.0, 90.0])
        inflow = np.array([0.0, 10.0, 10.0, 10.0, 10.0])
        V0 = 50.0

        V_norm, I_norm, V0_norm, scale = normalize(V_obs, inflow, V0)

        assert V_norm.shape == V_obs.shape
        assert I_norm.shape == inflow.shape

    def test_normalize_positive_values(self):
        """Test that normalization produces positive values for positive input."""
        V_obs = np.array([1000.0, 1100.0, 1050.0])
        inflow = np.array([0.0, 150.0, 50.0])
        V0 = 1000.0

        V_norm, I_norm, V0_norm, scale = normalize(V_obs, inflow, V0)

        assert np.all(V_norm > 0)
        assert I_norm[0] == 0.0  # First inflow is zero
        assert np.all(I_norm[1:] > 0)
        assert V0_norm > 0

    def test_normalize_zero_max_raises(self):
        """Test that zero max raises ValueError."""
        V_obs = np.array([0.0, 0.0, 0.0])
        inflow = np.array([0.0, 0.0, 0.0])
        V0 = 0.0

        with pytest.raises(ValueError, match="positive"):
            normalize(V_obs, inflow, V0)

    def test_normalize_negative_max_raises(self):
        """Test that negative max raises ValueError."""
        V_obs = np.array([-100.0, -50.0, -75.0])
        inflow = np.array([0.0, 10.0, 5.0])
        V0 = -100.0

        with pytest.raises(ValueError, match="positive"):
            normalize(V_obs, inflow, V0)

    def test_normalize_roundtrip(self):
        """Test that we can recover original values."""
        V_obs = np.array([100.0, 150.0, 200.0, 180.0])
        inflow = np.array([0.0, 50.0, 60.0, 30.0])
        V0 = 100.0

        V_norm, I_norm, V0_norm, scale = normalize(V_obs, inflow, V0)

        # Recover original values
        V_recovered = V_norm * scale
        I_recovered = I_norm * scale
        V0_recovered = V0_norm * scale

        assert np.allclose(V_recovered, V_obs)
        assert np.allclose(I_recovered, inflow)
        assert np.isclose(V0_recovered, V0)

    def test_normalize_single_value(self):
        """Test normalization with single observation."""
        V_obs = np.array([100.0])
        inflow = np.array([0.0])
        V0 = 100.0

        V_norm, I_norm, V0_norm, scale = normalize(V_obs, inflow, V0)

        assert scale == 100.0
        assert V_norm[0] == 1.0
        assert V0_norm == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
