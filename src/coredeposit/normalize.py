"""Data normalization utilities for numerical stability.

This module provides functions to normalize input data before estimation,
which improves numerical stability during MCMC sampling.
"""

from __future__ import annotations

import numpy as np

from .types import NDArray


def normalize(
    V_obs: NDArray, inflow: NDArray, V0: float
) -> tuple[NDArray, NDArray, float, float]:
    """Normalize data by the maximum observed balance.

    Divides all monetary values by max(V_obs) to bring them into the [0, 1]
    range. This improves numerical stability during MCMC sampling by ensuring
    all values are on a similar scale.

    Parameters
    ----------
    V_obs : NDArray
        Observed deposit balances. Shape: (T+1,).
    inflow : NDArray
        Deposit inflows. Shape: (T+1,).
    V0 : float
        Initial balance.

    Returns
    -------
    V_obs_normalized : NDArray
        Normalized observed balances, V_obs / scale.
    inflow_normalized : NDArray
        Normalized inflows, inflow / scale.
    V0_normalized : float
        Normalized initial balance, V0 / scale.
    scale : float
        The normalization scale factor (max(V_obs)). Use this to convert
        results back to original scale.

    Raises
    ------
    ValueError
        If max(V_obs) <= 0, as normalization requires positive values.

    Examples
    --------
    >>> V_obs = np.array([1000.0, 1100.0, 1200.0])
    >>> inflow = np.array([0.0, 100.0, 100.0])
    >>> V_norm, I_norm, V0_norm, scale = normalize(V_obs, inflow, 1000.0)
    >>> scale
    1200.0
    >>> V_norm
    array([0.833..., 0.916..., 1.0])
    """
    scale = float(np.max(V_obs))
    if scale <= 0:
        raise ValueError("V_obs must be positive to normalize.")
    return V_obs / scale, inflow / scale, V0 / scale, scale
