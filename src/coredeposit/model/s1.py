"""Survival functions for transactional deposits (Type 1).

This module provides survival functions for transactional (決済性) deposits.
The default model assumes immediate exit behavior, but alternative models
can be implemented by following the same interface.

To customize S1 behavior, either:
1. Modify the functions in this module directly
2. Pass a custom `s1_term_fn` to `V_model`
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array as JaxArray

from ..types import ArrayLike


def S1_immediate_exit(s: ArrayLike, h: ArrayLike) -> JaxArray:
    """Survival function for immediate exit model (default).

    Transactional deposits exit immediately after one period:
    - S1(1) = 1 - h (survive first month with probability 1-h)
    - S1(s≥2) = 0 (all exit by second month)

    Parameters
    ----------
    s : ArrayLike
        Survival duration in months (1, 2, 3, ...).
    h : ArrayLike
        First-month exit rate (0 < h < 1).

    Returns
    -------
    JaxArray
        Survival probability at duration s.
    """
    s = jnp.asarray(s)
    return jnp.where(s == 1, 1.0 - h, 0.0)


def S1_geometric(s: ArrayLike, h: ArrayLike) -> JaxArray:
    """Survival function for geometric (constant hazard) model.

    Transactional deposits have constant exit rate h per period:
    - S1(s) = (1 - h)^s

    Parameters
    ----------
    s : ArrayLike
        Survival duration in months (1, 2, 3, ...).
    h : ArrayLike
        Per-period exit rate (0 < h < 1).

    Returns
    -------
    JaxArray
        Survival probability at duration s.
    """
    s = jnp.asarray(s)
    return jnp.power(1.0 - h, s)


def S1_matrix_immediate_exit(T: int, h: ArrayLike) -> JaxArray:
    """Compute S1 survival matrix for immediate exit model.

    Builds a matrix where S1[t, i] is the survival probability at time t
    for deposits that entered at time i.

    For immediate exit: S1[i, i] = 1-h, S1[t, i] = 0 for t > i.

    Parameters
    ----------
    T : int
        Number of time periods.
    h : ArrayLike
        First-month exit rate.

    Returns
    -------
    JaxArray
        Survival matrix of shape (T+1, T+1).
    """
    # Only diagonal (t=i, s=1) has non-zero survival
    # S1[t, i] = 1-h when t == i and i >= 1
    diag = jnp.where(jnp.arange(T + 1) >= 1, 1.0 - h, 0.0)
    return jnp.diag(diag)


def S1_matrix_geometric(T: int, h: ArrayLike) -> JaxArray:
    """Compute S1 survival matrix for geometric model.

    Builds a lower triangular matrix where S1[t, i] = (1-h)^(t-i+1).

    Parameters
    ----------
    T : int
        Number of time periods.
    h : ArrayLike
        Per-period exit rate.

    Returns
    -------
    JaxArray
        Survival matrix of shape (T+1, T+1).
    """
    t_idx = jnp.arange(T + 1)[:, None]  # (T+1, 1)
    i_idx = jnp.arange(T + 1)[None, :]  # (1, T+1)

    # Duration s = t - i + 1
    s = t_idx - i_idx + 1

    # Survival: (1-h)^s
    S1 = jnp.power(1.0 - h, s)

    # Mask: valid only when t >= i and i >= 1
    valid = (t_idx >= i_idx) & (i_idx >= 1)
    S1 = jnp.where(valid, S1, 0.0)

    return S1


def S1_term_immediate_exit(
    inflow: ArrayLike,
    w1: ArrayLike,
    h: ArrayLike,
    T: int,
) -> JaxArray:
    """Compute S1 contribution to balance for immediate exit model (default).

    This is the term: w1(t) × inflow(t) × (1 - h) for t >= 1

    Parameters
    ----------
    inflow : ArrayLike
        Deposit inflows at each time point. Shape: (T+1,).
    w1 : ArrayLike
        Proportion of inflows that are transactional deposits.
        Can be scalar (constant) or array of shape (T+1,) for time-varying.
    h : ArrayLike
        First-month exit rate.
    T : int
        Number of time periods.

    Returns
    -------
    JaxArray
        S1 contribution to balance at each time. Shape: (T+1,).
    """
    inflow = jnp.asarray(inflow)
    w1 = jnp.asarray(w1)

    # Handle both scalar and array w1
    if w1.ndim == 0:
        w1_vals = w1
    else:
        w1_vals = w1[1:]

    # Only current period's inflow survives, and only partially
    term = jnp.zeros(T + 1, dtype=inflow.dtype)
    term = term.at[1:].set(w1_vals * inflow[1:] * (1.0 - h))
    return term


def S1_term_geometric(
    inflow: ArrayLike,
    w1: ArrayLike,
    h: ArrayLike,
    T: int,
) -> JaxArray:
    """Compute S1 contribution to balance for geometric model.

    This computes: Σᵢ w1(i) × inflow(i) × S1(t|i)
    where S1(t|i) = (1-h)^(t-i+1).

    Parameters
    ----------
    inflow : ArrayLike
        Deposit inflows at each time point. Shape: (T+1,).
    w1 : ArrayLike
        Proportion of inflows that are transactional deposits.
        Can be scalar (constant) or array of shape (T+1,) for time-varying.
    h : ArrayLike
        Per-period exit rate.
    T : int
        Number of time periods.

    Returns
    -------
    JaxArray
        S1 contribution to balance at each time. Shape: (T+1,).
    """
    S1 = S1_matrix_geometric(T, h)
    inflow = jnp.asarray(inflow)
    w1 = jnp.asarray(w1)

    # Weight inflow by w1 (element-wise for time-varying w1)
    weighted_inflow = w1 * inflow
    return S1 @ weighted_inflow


# Default S1 term function (used by V_model)
S1_term_default = S1_term_immediate_exit
"""Default S1 term function.

Change this to use a different S1 model globally, or pass a custom
function to V_model via the `s1_term_fn` parameter.
"""
