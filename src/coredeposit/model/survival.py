"""Survival function computations for deposit cohorts.

This module provides functions to compute survival probabilities for deposit
cohorts under the Weibull hazard model with time-varying covariates.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array as JaxArray

from ..types import ArrayLike
from .hazard import weibull_hazard


def S2_matrix(
    lam: ArrayLike,
    gam: ArrayLike,
    T: int,
    weight: ArrayLike,
    dt: float = 1.0,
) -> JaxArray:
    """Compute the survival matrix for sticky deposits (type 2).

    Builds a lower triangular matrix where entry S2[t, i] represents the
    survival probability at time t for deposits that entered at time i.

    The survival function for cohort i at time t is:

        S2(t|i) = exp( -∑_{u=i}^{t} h0(u-i+1) × weight[u] × dt )

    where h0 is the baseline Weibull hazard and weight[u] = exp(β'z[u])
    incorporates time-varying covariates.

    Parameters
    ----------
    lam : ArrayLike
        Weibull scale parameter (λ > 0).
    gam : ArrayLike
        Weibull shape parameter (γ > 0).
    T : int
        Number of time periods (the matrix will have shape (T+1, T+1)).
    weight : ArrayLike
        Covariate weights at each time point. Shape: (T+1,).
        Use ones(T+1) for the no-covariate model, or exp(z @ beta) for
        the covariate model.
    dt : float, optional
        Time step size in months. Default is 1.0.

    Returns
    -------
    JaxArray
        Survival matrix of shape (T+1, T+1). Entry S2[t, i] is the survival
        probability at time t for the cohort that entered at time i.
        - S2[t, i] > 0 for t >= i >= 1
        - S2[t, i] = 0 for t < i or i = 0 (column 0 is unused)

    Notes
    -----
    This function uses fully vectorized JAX operations for efficient
    compilation and execution during MCMC sampling.
    """
    # Indices
    t_idx = jnp.arange(T + 1)[:, None]  # (T+1, 1)
    i_idx = jnp.arange(T + 1)[None, :]  # (1, T+1)

    # Duration s = t - i + 1 (time since cohort i entered, observed at time t)
    s = t_idx - i_idx + 1  # (T+1, T+1)

    # Hazard for each (t, i) pair based on duration s
    h0 = weibull_hazard(jnp.maximum(s, 1) * dt, lam, gam) * dt

    # Weight at calendar time t (broadcast to matrix)
    weight_matrix = jnp.broadcast_to(weight[:, None], (T + 1, T + 1))

    # Weighted hazard
    h0_weighted = h0 * weight_matrix

    # Mask: valid only when t >= i and i >= 1
    valid = (t_idx >= i_idx) & (i_idx >= 1)
    h0_weighted = jnp.where(valid, h0_weighted, 0.0)

    # For each column i, we need cumsum from row i to row t
    # Use the trick: cumsum_from_i[t] = cumsum[t] - cumsum[i-1]

    # Full cumsum along rows
    cumsum_full = jnp.cumsum(h0_weighted, axis=0)  # (T+1, T+1)

    # For column i, the value at row (i-1) in cumsum_full is the sum up to row i-1
    # We need to subtract this from all rows >= i

    # Create the offset matrix: offset[t, i] = cumsum_full[i-1, i] for all t
    # But we need to handle i=0 specially (should be 0)
    i_minus_1 = jnp.maximum(i_idx - 1, 0)  # (1, T+1)

    # Get the cumsum values at row i-1 for each column
    # offset_vals[i] = cumsum_full[i-1, i]
    offset_vals = cumsum_full[i_minus_1.ravel(), jnp.arange(T + 1)]  # (T+1,)

    # Zero out for i=0 (no offset needed)
    offset_vals = jnp.where(jnp.arange(T + 1) >= 1, offset_vals, 0.0)

    # Broadcast to matrix and subtract
    offset_matrix = jnp.broadcast_to(offset_vals[None, :], (T + 1, T + 1))
    cum_hazard = cumsum_full - offset_matrix

    # Apply mask again and compute survival
    # Clip cumulative hazard to prevent numerical underflow in exp()
    cum_hazard = jnp.where(valid, cum_hazard, 0.0)
    S2 = jnp.where(valid, jnp.exp(-jnp.minimum(cum_hazard, 700.0)), 0.0)

    return S2


def S2_init_vector(
    lam: ArrayLike,
    gam: ArrayLike,
    T: int,
    m: ArrayLike,
    weight: ArrayLike,
    dt: float = 1.0,
) -> JaxArray:
    """Compute survival probabilities for the initial deposit cohort.

    The initial balance V0 is assumed to have been deposited before the
    observation period. This function computes the survival of this initial
    cohort, which has already aged m months at time 0.

    The survival at time t is:

        S2_init(t) = exp( -∑_{u=0}^{t} h0(m+u) × weight[u] × dt )

    Parameters
    ----------
    lam : ArrayLike
        Weibull scale parameter (λ > 0).
    gam : ArrayLike
        Weibull shape parameter (γ > 0).
    T : int
        Number of time periods.
    m : ArrayLike
        Average age of the initial balance in months at time 0.
        This is a model parameter to be estimated.
    weight : ArrayLike
        Covariate weights at each time point. Shape: (T+1,).
    dt : float, optional
        Time step size in months. Default is 1.0.

    Returns
    -------
    JaxArray
        Survival vector of shape (T+1,). Entry S2_init[t] is the survival
        probability of the initial cohort at time t.
    """
    # Durations: m, m+1, m+2, ..., m+T
    t_vals = jnp.arange(T + 1)
    durations = (m + t_vals) * dt
    h0_vals = weibull_hazard(durations, lam, gam) * dt

    # Weighted hazard
    h0_weighted = h0_vals * weight

    # Cumulative hazard
    cum_hazard = jnp.cumsum(h0_weighted)

    # Survival (clip to prevent numerical underflow)
    S2 = jnp.exp(-jnp.minimum(cum_hazard, 700.0))

    return S2
