"""Core deposit balance model.

This module provides the main balance prediction function that combines
the two-type deposit model (transactional and sticky deposits) with
Weibull survival dynamics.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array as JaxArray

from ..types import ArrayLike
from .survival import S2_matrix, S2_init_vector


def V_model(
    lam: ArrayLike,
    gam: ArrayLike,
    w1: ArrayLike,
    h: ArrayLike,
    m: ArrayLike,
    V0: ArrayLike,
    inflow: ArrayLike,
    weight: ArrayLike,
    dt: float = 1.0,
) -> JaxArray:
    """Compute predicted deposit balances under the two-type model.

    This function implements the core deposit balance model that decomposes
    deposits into two types:

    1. **Transactional deposits (type 1)**: Short-lived deposits that exit
       quickly. Parameterized by w1 (proportion) and h (exit rate).

    2. **Sticky deposits (type 2)**: Long-term deposits with Weibull survival.
       Parameterized by λ (scale) and γ (shape).

    The balance at time t is computed as:

        V(t) = w1 × inflow(t) × (1 - h)
             + (1 - w1) × Σᵢ inflow(i) × S2(t|i)
             + V0 × S2_init(t)

    where:
    - First term: remaining transactional deposits from current period
    - Second term: sticky deposits from all past inflows
    - Third term: surviving initial balance

    Parameters
    ----------
    lam : ArrayLike
        Weibull scale parameter (λ > 0) for sticky deposits.
    gam : ArrayLike
        Weibull shape parameter (γ > 0) for sticky deposits.
    w1 : ArrayLike
        Proportion of inflows that are transactional deposits (0 < w1 < 1).
    h : ArrayLike
        Exit rate for transactional deposits in the first month (0 < h < 1).
        A value of h=0.8 means 80% of transactional deposits exit within
        the first month.
    m : ArrayLike
        Average age of the initial balance in months at time 0.
        This accounts for the fact that V0 has already survived some time
        before the observation period.
    V0 : ArrayLike
        Initial deposit balance at time 0.
    inflow : ArrayLike
        Deposit inflows at each time point. Shape: (T+1,).
    weight : ArrayLike
        Covariate weights affecting the hazard rate. Shape: (T+1,).
        Use ones(T+1) for the no-covariate model, or exp(z @ beta)
        for the model with covariates.
    dt : float, optional
        Time step size in months. Default is 1.0.

    Returns
    -------
    JaxArray
        Predicted deposit balances at each time point. Shape: (T+1,).
        Vhat[0] is set to V0 by construction.

    Notes
    -----
    This function is designed to be JAX-compatible for use with automatic
    differentiation and MCMC sampling. All array operations use JAX
    primitives.

    The model assumes:
    - All initial balance is sticky (type 2) deposits
    - Transactional deposits only survive one period with probability (1-h)
    - Sticky deposit survival follows a Weibull distribution with
      optional time-varying covariates
    """
    inflow = jnp.asarray(inflow)
    weight = jnp.asarray(weight)
    T = int(inflow.shape[0] - 1)

    S2 = S2_matrix(lam=lam, gam=gam, T=T, weight=weight, dt=dt)
    S2_init = S2_init_vector(lam=lam, gam=gam, T=T, m=m, weight=weight, dt=dt)

    # Term 1: Transactional deposits (survive only one period)
    term1 = jnp.zeros(T + 1, dtype=inflow.dtype).at[1:].set(w1 * inflow[1:] * (1.0 - h))

    # Term 2: Sticky deposits from all past inflows
    term2 = (1.0 - w1) * (S2 @ inflow)

    # Term 3: Surviving initial balance
    term0 = V0 * S2_init

    Vhat = term0 + term1 + term2
    Vhat = Vhat.at[0].set(V0)

    return Vhat
