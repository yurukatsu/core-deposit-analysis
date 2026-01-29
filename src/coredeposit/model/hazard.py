"""Hazard function definitions for deposit survival models.

This module provides hazard rate functions used in the survival analysis
of deposit balances.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array as JaxArray

from ..types import ArrayLike


def weibull_hazard(
    duration: ArrayLike,
    lam: ArrayLike,
    gam: ArrayLike,
) -> JaxArray:
    """Compute the Weibull hazard rate.

    The Weibull hazard function is defined as:

        h(t; λ, γ) = λ * γ * (λ * t)^(γ - 1)

    where:
    - λ (lambda) is the scale parameter
    - γ (gamma) is the shape parameter
    - t is the duration (time since deposit)

    The shape parameter γ determines the hazard behavior:
    - γ < 1: decreasing hazard (negative duration dependence)
    - γ = 1: constant hazard (exponential distribution)
    - γ > 1: increasing hazard (positive duration dependence)

    Parameters
    ----------
    duration : ArrayLike
        Time since deposit entry, in months. Can be a scalar or array.
        Values are clipped to a minimum of 1e-8 to avoid numerical issues
        when γ < 1.
    lam : ArrayLike
        Weibull scale parameter (λ > 0). Controls the overall level of
        the hazard rate.
    gam : ArrayLike
        Weibull shape parameter (γ > 0). Controls how the hazard changes
        with duration.

    Returns
    -------
    JaxArray
        The hazard rate h(duration; lam, gam). Same shape as the input
        duration array.

    Notes
    -----
    This function is JAX-compatible and can be used inside JIT-compiled
    functions and MCMC samplers.
    """
    duration = jnp.clip(duration, 1e-8)
    return lam * gam * (lam * duration) ** (gam - 1)
