"""Transactional deposit proportion (w1) functions.

This module provides functions for computing the transactional deposit
proportion w1, which can be constant or time-varying.

Available functions:
- w1_constant: Constant w1 over time (default behavior)
- w1_logistic: Time-varying w1 via logistic regression
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array as JaxArray

from ..types import ArrayLike, Scalar


def sigmoid(x: ArrayLike) -> JaxArray:
    """Sigmoid function.

    Parameters
    ----------
    x : ArrayLike
        Input values.

    Returns
    -------
    JaxArray
        Sigmoid of input, in (0, 1).
    """
    return 1.0 / (1.0 + jnp.exp(-jnp.clip(x, -500, 500)))


def w1_constant(w1_base: Scalar, T: int) -> JaxArray:
    """Constant w1 over time.

    Parameters
    ----------
    w1_base : Scalar
        Base transactional proportion (0 < w1 < 1).
    T : int
        Number of time periods.

    Returns
    -------
    JaxArray
        Array of shape (T+1,) with constant w1 values.

    Examples
    --------
    >>> w1 = w1_constant(0.3, T=10)
    >>> w1.shape
    (11,)
    >>> w1[5]
    0.3
    """
    return jnp.full(T + 1, w1_base)


def w1_logistic(
    a: Scalar,
    b: ArrayLike,
    x: ArrayLike,
) -> JaxArray:
    """Time-varying w1 via logistic regression.

    Computes w1(t) = sigmoid(a + x(t) @ b) for each time point.

    Parameters
    ----------
    a : Scalar
        Intercept parameter. When x=0, w1 = sigmoid(a).
    b : ArrayLike
        Coefficient vector of shape (p,) for p features.
    x : ArrayLike
        Feature matrix of shape (T+1, p) or (T+1,) for single feature.

    Returns
    -------
    JaxArray
        Array of shape (T+1,) with time-varying w1 values in (0, 1).

    Notes
    -----
    The logistic model is:

        logit(w1(t)) = a + b' x(t)
        w1(t) = sigmoid(a + b' x(t))

    Common features include:
    - log(I_t / MA(I)): deviation from moving average inflow
    - Seasonal dummies (month indicators)
    - Spike indicators

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randn(12, 2)  # 12 months, 2 features
    >>> w1 = w1_logistic(a=-1.0, b=np.array([0.5, -0.3]), x=x)
    >>> w1.shape
    (12,)
    >>> np.all((w1 > 0) & (w1 < 1))
    True
    """
    x = jnp.asarray(x)
    b = jnp.asarray(b)

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if b.ndim == 0:
        b = b.reshape(1)

    linear = a + x @ b
    return sigmoid(linear)


def compute_w1_features_ma_deviation(
    inflow: ArrayLike,
    window: int = 12,
) -> JaxArray:
    """Compute log deviation from moving average as w1 feature.

    Parameters
    ----------
    inflow : ArrayLike
        Inflow array of shape (T+1,).
    window : int, optional
        Moving average window size. Default is 12 (1 year).

    Returns
    -------
    JaxArray
        Feature array of shape (T+1,).
        log(inflow / MA(inflow)), clipped for stability.
    """
    inflow = jnp.asarray(inflow)
    T = inflow.shape[0] - 1

    # Simple expanding/rolling mean (JAX-compatible)
    # For first `window` points, use expanding mean
    # After that, use rolling mean
    def compute_ma(i):
        start = jnp.maximum(0, i - window + 1)
        end = i + 1
        return jnp.mean(jnp.where(jnp.arange(T + 1) >= start, inflow, 0.0)) * (T + 1) / (
            end - start
        )

    # Use cumulative sum for efficiency
    cumsum = jnp.cumsum(inflow)
    counts = jnp.arange(1, T + 2)

    # Expanding mean for first `window` elements
    expanding_mean = cumsum / counts

    # For rolling mean after window
    shifted_cumsum = jnp.concatenate([jnp.array([0.0]), cumsum[:-1]])
    rolled_cumsum = jnp.where(
        jnp.arange(T + 1) >= window,
        cumsum - jnp.roll(cumsum, window),
        cumsum,
    )
    rolled_counts = jnp.where(jnp.arange(T + 1) >= window, window, jnp.arange(1, T + 2))
    ma = rolled_cumsum / rolled_counts

    # Avoid division by zero and log(0)
    ma = jnp.maximum(ma, 1e-8)
    inflow_safe = jnp.maximum(inflow, 1e-8)

    deviation = jnp.log(inflow_safe / ma)
    return jnp.clip(deviation, -5.0, 5.0)


def compute_w1_features_seasonal(
    T: int,
    start_month: int = 1,
) -> JaxArray:
    """Compute seasonal (monthly) dummy features.

    Parameters
    ----------
    T : int
        Number of time periods.
    start_month : int, optional
        Starting month (1-12). Default is 1 (January).

    Returns
    -------
    JaxArray
        Feature matrix of shape (T+1, 11) with monthly dummies.
        Month 12 (December) is the reference category.
    """
    months = (jnp.arange(T + 1) + start_month - 1) % 12 + 1  # 1-12
    # Create dummy for months 1-11 (12 is reference)
    dummies = jnp.zeros((T + 1, 11))
    for m in range(1, 12):
        dummies = dummies.at[:, m - 1].set((months == m).astype(float))
    return dummies
