"""Non-linear least squares estimator for core deposit model.

This module provides the NLSEstimator class for point estimation of
core deposit model parameters using scipy's least_squares optimizer.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from ..types import CoreDepositData, EstimationResult
from ..model.balance import V_model
from .base import Estimator


class NLSEstimator(Estimator):
    """Non-linear least squares estimator for the core deposit model.

    This estimator finds point estimates of model parameters by minimizing
    the sum of squared residuals between predicted and observed deposit
    balances. Supports both models with and without covariates.

    The optimization uses parameter transformations to handle constraints:
    - Positive parameters (λ, γ, m): log transform
    - Bounded parameters (w1, h ∈ [0,1]): logit transform
    - Unbounded parameters (β): no transform

    Parameters
    ----------
    loss : str, optional
        Loss function for robust regression. Options:
        - 'linear': Standard least squares (sum of squared residuals)
        - 'soft_l1': Smooth approximation to L1 loss (default)
        - 'huber': Huber loss
        - 'cauchy': Cauchy loss (most robust to outliers)
        Default is 'soft_l1' for robustness against outliers.
    f_scale : float, optional
        Scaling factor for the loss function. Residuals larger than f_scale
        are down-weighted. Only used when loss != 'linear'. Default is 0.05.

    Attributes
    ----------
    loss : str
        The loss function being used.
    f_scale : float
        The scaling factor for robust loss.

    Examples
    --------
    >>> from coredeposit import NLSEstimator, CoreDepositData
    >>> import numpy as np
    >>> data = CoreDepositData(
    ...     V_obs=np.array([100, 95, 90, 85, 80]),
    ...     inflow=np.array([10, 12, 8, 11, 9]),
    ...     V0=100.0,
    ... )
    >>> estimator = NLSEstimator(loss='soft_l1', f_scale=0.05)
    >>> result = estimator.fit(data)
    >>> print(result.params['lambda'])

    >>> # With fixed m parameter
    >>> result = estimator.fit(data, m_fixed=12.0)
    """

    def __init__(self, *, loss: str = "soft_l1", f_scale: float = 0.05):
        """Initialize the NLS estimator.

        Parameters
        ----------
        loss : str, optional
            Loss function for robust regression. Default is 'soft_l1'.
        f_scale : float, optional
            Scaling factor for robust loss functions. Default is 0.05.
        """
        self.loss = loss
        self.f_scale = f_scale

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid function for logit transformation.

        Parameters
        ----------
        x : float
            Input value (unbounded).

        Returns
        -------
        float
            Output value in (0, 1).
        """
        return 1.0 / (1.0 + np.exp(-x))

    def fit(
        self,
        data: CoreDepositData,
        *,
        m_fixed: float | None = None,
    ) -> EstimationResult:
        """Estimate model parameters using non-linear least squares.

        Minimizes the sum of (weighted) squared residuals between predicted
        and observed deposit balances using scipy's least_squares optimizer.

        Parameters
        ----------
        data : CoreDepositData
            Input data containing:

            - V_obs : array of shape (T+1,) with observed deposit balances
            - inflow : array of shape (T+1,) with deposit inflows
            - V0 : initial deposit balance at time 0
            - z : optional covariates of shape (T+1,) or (T+1, p)

        m_fixed : float or None, optional
            If provided, fix the initial age parameter m to this value
            instead of estimating it. Useful for sensitivity analysis or
            when m is known from external information. Default is None
            (estimate m from data).

        Returns
        -------
        EstimationResult
            Result containing:

            params : dict
                Estimated parameters:

                - 'lambda': Weibull scale parameter (float)
                - 'gamma': Weibull shape parameter (float)
                - 'w1': Transactional deposit proportion (float, 0-1)
                - 'h': Transactional exit rate (float, 0-1)
                - 'm': Initial deposit age in months (float)
                - 'beta': Covariate coefficients (array, only if z provided)

            diagnostics : dict
                Optimization diagnostics:

                - 'success': Whether optimization converged (bool)
                - 'nfev': Number of function evaluations (int)
                - 'cost': Final cost function value (float)
                - 'm_fixed': Value of m_fixed if provided (float or None)

        Notes
        -----
        The optimization starts from x0 = zeros, which corresponds to:
        - λ = exp(0) = 1.0
        - γ = exp(0) = 1.0
        - w1 = sigmoid(0) = 0.5
        - h = sigmoid(0) = 0.5
        - m = exp(0) = 1.0 (if not fixed)
        - β = zeros (if covariates provided)

        The residuals are computed for t = 1, ..., T (excluding t=0 which
        is fixed to V0 by construction).
        """
        V_obs = np.asarray(data.V_obs, dtype=float)
        inflow = np.asarray(data.inflow, dtype=float)
        V0 = float(data.V0)

        z = data.z
        if z is not None:
            z = np.asarray(z, dtype=float)
            if z.ndim == 1:
                z = z.reshape(-1, 1)
            p = z.shape[1]
        else:
            p = 0

        T = V_obs.shape[0] - 1

        def unpack(x: np.ndarray) -> tuple:
            lam = np.exp(x[0])
            gam = np.exp(x[1])
            w1 = self._sigmoid(x[2])
            h = self._sigmoid(x[3])

            if m_fixed is None:
                m = np.exp(x[4])
                beta_start = 5
            else:
                m = m_fixed
                beta_start = 4

            if p > 0:
                beta = x[beta_start : beta_start + p]
            else:
                beta = None

            return lam, gam, w1, h, m, beta

        def residuals(x: np.ndarray) -> np.ndarray:
            lam, gam, w1, h, m, beta = unpack(x)

            if beta is None:
                weight = np.ones(T + 1)
            else:
                weight = np.exp(z @ beta)

            Vhat = np.array(
                V_model(
                    lam=lam,
                    gam=gam,
                    w1=w1,
                    h=h,
                    m=m,
                    V0=V0,
                    inflow=inflow,
                    weight=weight,
                )
            )

            return Vhat[1:] - V_obs[1:]

        n_param = 5 + p if m_fixed is None else 4 + p
        x0 = np.zeros(n_param)

        res = least_squares(
            residuals,
            x0,
            loss=self.loss,
            f_scale=self.f_scale,
        )

        lam, gam, w1, h, m, beta = unpack(res.x)

        params = {
            "lambda": float(lam),
            "gamma": float(gam),
            "w1": float(w1),
            "h": float(h),
            "m": float(m),
        }
        if beta is not None:
            params["beta"] = beta

        return EstimationResult(
            params=params,
            diagnostics={
                "success": bool(res.success),
                "nfev": int(res.nfev),
                "cost": float(res.cost),
                "m_fixed": m_fixed,
            },
        )
