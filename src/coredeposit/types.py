"""Type definitions for the core deposit analysis package.

This module defines the core data structures and type aliases used throughout
the package.

Type Aliases
------------
NDArray
    Alias for numpy.ndarray.
ArrayLike
    Union of JAX Array and NumPy ndarray. Used for function parameters that
    accept either array type.
Scalar
    Union of JAX Array and float. Used for scalar parameters that may be
    JAX-traced during MCMC sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from jax import Array as JaxArray

NDArray = np.ndarray
ArrayLike = JaxArray | NDArray
Scalar = JaxArray | float


@dataclass
class CoreDepositData:
    """Input data container for core deposit estimation.

    Attributes
    ----------
    V_obs : NDArray
        Observed deposit balances at each time point. Shape: (T+1,) where T is
        the number of periods. V_obs[0] is the initial balance at time 0.
    inflow : NDArray
        Deposit inflows at each time point. Shape: (T+1,). inflow[0] is typically
        0 (no inflow at time 0), and inflow[t] represents new deposits received
        during period t.
    V0 : float
        Initial deposit balance at time 0. Usually equals V_obs[0].
    z : NDArray | None, optional
        Exogenous covariates affecting the hazard rate (S2). Shape: (T+1,) for
        single covariate or (T+1, p) for p covariates. Default is None.
    w1_features : NDArray | None, optional
        Features for time-varying w1 (transactional proportion).
        Shape: (T+1,) for single feature or (T+1, q) for q features.
        When provided, w1 is modeled as: w1(t) = sigmoid(a + b' w1_features(t)).
        Default is None (constant w1).
    """

    V_obs: NDArray
    inflow: NDArray
    V0: float
    z: NDArray | None = None
    w1_features: NDArray | None = None


@dataclass
class EstimationResult:
    """Container for estimation results.

    Attributes
    ----------
    params : dict[str, Any]
        Estimated model parameters. For NLS, contains point estimates as floats.
        For MCMC, contains posterior samples as arrays with shape (n_samples,)
        or (n_samples, p) for vector parameters.

        Common keys:
        - "lambda": Weibull scale parameter (positive)
        - "gamma": Weibull shape parameter (positive)
        - "w1": Proportion of transactional deposits (0-1)
        - "h": First-month exit rate for transactional deposits (0-1)
        - "m": Average age of initial balance in months (positive)
        - "beta": Covariate coefficients (only if covariates are used)
        - "sigma": Observation noise standard deviation (MCMC only)
        - "nu": Student-t degrees of freedom (MCMC with studentt likelihood)

    diagnostics : dict[str, Any]
        Diagnostic information about the estimation.

        For NLS:
        - "success": bool, whether optimization converged
        - "nfev": int, number of function evaluations
        - "cost": float, final residual sum of squares
        - "m_fixed": float | None, fixed value of m if provided

        For MCMC:
        - "scale": float, normalization scale factor applied to data
    """

    params: dict[str, Any]
    diagnostics: dict[str, Any]
