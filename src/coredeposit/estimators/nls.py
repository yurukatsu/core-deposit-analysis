"""Non-linear least squares estimator for core deposit model.

This module provides the NLSEstimator class for point estimation of
core deposit model parameters using scipy's least_squares optimizer.
Supports both standard NLS and MAP (Maximum A Posteriori) estimation.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import least_squares, minimize

from ..types import CoreDepositData, EstimationResult, NDArray
from ..model.balance import V_model
from ..model.w1 import w1_logistic
from .base import Estimator
from .map_priors import MAPPriors


class NLSEstimator(Estimator):
    """Non-linear least squares estimator for the core deposit model.

    This estimator finds point estimates of model parameters by minimizing
    the sum of squared residuals between predicted and observed deposit
    balances. Supports both models with and without covariates.

    When `priors` is provided, performs MAP (Maximum A Posteriori) estimation
    by adding prior penalty terms to the objective function.

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
        Note: Only used when priors=None (standard NLS).
    f_scale : float, optional
        Scaling factor for the loss function. Residuals larger than f_scale
        are down-weighted. Only used when loss != 'linear'. Default is 0.05.
        Note: Only used when priors=None (standard NLS).
    priors : MAPPriors or None, optional
        Prior distributions for MAP estimation. If None, performs standard
        NLS without priors. If provided, uses scipy.optimize.minimize to
        minimize: 0.5 * Σ(residual²) + negative_log_prior.
        Default is None.

    Attributes
    ----------
    loss : str
        The loss function being used.
    f_scale : float
        The scaling factor for robust loss.
    priors : MAPPriors or None
        Prior distributions for MAP estimation.

    Examples
    --------
    Standard NLS estimation:

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

    MAP estimation with default priors:

    >>> from coredeposit.estimators import default_map_priors
    >>> priors = default_map_priors()
    >>> estimator = NLSEstimator(priors=priors)
    >>> result = estimator.fit(data)

    With fixed m parameter:

    >>> result = estimator.fit(data, m_fixed=12.0)
    """

    def __init__(
        self,
        *,
        loss: str = "soft_l1",
        f_scale: float = 0.05,
        priors: MAPPriors | None = None,
    ):
        """Initialize the NLS estimator.

        Parameters
        ----------
        loss : str, optional
            Loss function for robust regression. Default is 'soft_l1'.
            Only used when priors=None.
        f_scale : float, optional
            Scaling factor for robust loss functions. Default is 0.05.
            Only used when priors=None.
        priors : MAPPriors or None, optional
            Prior distributions for MAP estimation. Default is None.
        """
        self.loss = loss
        self.f_scale = f_scale
        self.priors = priors

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
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def fit(
        self,
        data: CoreDepositData,
        *,
        m_fixed: float | None = None,
    ) -> EstimationResult:
        """Estimate model parameters using non-linear least squares or MAP.

        When priors=None, minimizes the sum of (weighted) squared residuals.
        When priors is provided, performs MAP estimation by minimizing:
            0.5 * Σ(residual²) - log p(θ)

        Parameters
        ----------
        data : CoreDepositData
            Input data containing:

            - V_obs : array of shape (T+1,) with observed deposit balances
            - inflow : array of shape (T+1,) with deposit inflows
            - V0 : initial deposit balance at time 0
            - z : optional covariates of shape (T+1,) or (T+1, p) for S2 hazard
            - w1_features : optional features of shape (T+1,) or (T+1, q) for
              time-varying w1. When provided, w1(t) = sigmoid(a + b'x(t)).

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
                - 'w1': Transactional deposit proportion (float, 0-1).
                  For time-varying models, this is the mean w1.
                - 'h': Transactional exit rate (float, 0-1)
                - 'm': Initial deposit age in months (float)
                - 'beta': Covariate coefficients (array, only if z provided)
                - 'w1_a': w1 intercept on logit scale (only if w1_features provided)
                - 'w1_b': w1 coefficients (array, only if w1_features provided)

            diagnostics : dict
                Optimization diagnostics:

                - 'success': Whether optimization converged (bool)
                - 'nfev': Number of function evaluations (int)
                - 'cost': Final cost function value (float)
                - 'm_fixed': Value of m_fixed if provided (float or None)
                - 'method': 'nls' or 'map' (str)
                - 'w1_time_varying': Whether w1 is time-varying (bool)

        Notes
        -----
        The optimization starts from x0 = zeros, which corresponds to:
        - λ = exp(0) = 1.0
        - γ = exp(0) = 1.0
        - w1 = sigmoid(0) = 0.5 (or w1_a = 0 for time-varying)
        - h = sigmoid(0) = 0.5
        - m = exp(0) = 1.0 (if not fixed)
        - w1_b = zeros (if w1_features provided)
        - β = zeros (if covariates provided)

        The residuals are computed for t = 1, ..., T (excluding t=0 which
        is fixed to V0 by construction).
        """
        V_obs = np.asarray(data.V_obs, dtype=float)
        inflow = np.asarray(data.inflow, dtype=float)
        V0 = float(data.V0)

        # S2 covariates (hazard)
        z = data.z
        if z is not None:
            z = np.asarray(z, dtype=float)
            if z.ndim == 1:
                z = z.reshape(-1, 1)
            p = z.shape[1]
        else:
            p = 0

        # w1 features (time-varying transactional proportion)
        w1_features = data.w1_features
        if w1_features is not None:
            w1_features = np.asarray(w1_features, dtype=float)
            if w1_features.ndim == 1:
                w1_features = w1_features.reshape(-1, 1)
            q = w1_features.shape[1]
        else:
            q = 0

        T = V_obs.shape[0] - 1

        def unpack(x: np.ndarray) -> tuple:
            lam = np.exp(x[0])
            gam = np.exp(x[1])
            h = self._sigmoid(x[3])

            if m_fixed is None:
                m = np.exp(x[4])
                next_idx = 5
            else:
                m = m_fixed
                next_idx = 4

            # w1 handling: constant vs time-varying
            if q > 0:
                # Time-varying w1: x[2] is w1_a (intercept on logit scale)
                w1_a = x[2]
                w1_b = x[next_idx : next_idx + q]
                next_idx = next_idx + q
                w1 = np.array(w1_logistic(w1_a, w1_b, w1_features))
            else:
                # Constant w1
                w1 = self._sigmoid(x[2])
                w1_a = None
                w1_b = None

            if p > 0:
                beta = x[next_idx : next_idx + p]
            else:
                beta = None

            return lam, gam, w1, h, m, beta, w1_a, w1_b

        def compute_residuals(x: np.ndarray) -> np.ndarray:
            lam, gam, w1, h, m, beta, _, _ = unpack(x)

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

        # Parameter count: lam, gam, w1_a/w1, h, [m], [w1_b...], [beta...]
        n_base = 5 if m_fixed is None else 4
        n_param = n_base + q + p
        x0 = np.zeros(n_param)

        if self.priors is None:
            # Standard NLS using least_squares
            res = least_squares(
                compute_residuals,
                x0,
                loss=self.loss,
                f_scale=self.f_scale,
            )
            success = res.success
            nfev = res.nfev
            cost = float(res.cost)
            x_opt = res.x
            method = "nls"
        else:
            # MAP estimation using minimize
            def objective(x: np.ndarray) -> float:
                residuals = compute_residuals(x)
                sse = 0.5 * np.sum(residuals**2)

                lam, gam, w1, h, m, beta, w1_a, w1_b = unpack(x)
                # For MAP with time-varying w1, use w1_a for the prior
                # (w1_a on logit scale corresponds to baseline w1)
                w1_for_prior = self._sigmoid(w1_a) if w1_a is not None else w1
                neg_log_prior = self.priors.neg_log_prob(
                    lam=lam, gam=gam, w1=w1_for_prior, h=h, m=m, beta=beta
                )

                return sse + neg_log_prior

            res = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                options={"maxiter": 1000, "disp": False},
            )
            success = res.success
            nfev = res.nfev
            cost = float(res.fun)
            x_opt = res.x
            method = "map"

        lam, gam, w1, h, m, beta, w1_a, w1_b = unpack(x_opt)

        params = {
            "lambda": float(lam),
            "gamma": float(gam),
            "h": float(h),
            "m": float(m),
        }

        # w1 parameters
        if w1_a is not None:
            # Time-varying w1: store coefficients
            params["w1_a"] = float(w1_a)
            params["w1_b"] = w1_b
            # Also store mean w1 for convenience
            params["w1"] = float(np.mean(w1))
        else:
            params["w1"] = float(w1)

        if beta is not None:
            params["beta"] = beta

        return EstimationResult(
            params=params,
            diagnostics={
                "success": bool(success),
                "nfev": int(nfev),
                "cost": float(cost),
                "m_fixed": m_fixed,
                "method": method,
                "w1_time_varying": q > 0,
            },
        )

    def predict(
        self,
        data: CoreDepositData,
        result: EstimationResult,
        *,
        uncertainty: bool = False,
        ci_prob: float = 0.95,
    ) -> NDArray | dict[str, NDArray]:
        """Predict deposit balances using NLS/MAP point estimates.

        Parameters
        ----------
        data : CoreDepositData
            Input data for prediction containing inflows and initial balance.
        result : EstimationResult
            Result from a previous call to `fit()`.
        uncertainty : bool, optional
            If True, a warning is issued since NLS/MAP only provides point
            estimates. Default is False.
        ci_prob : float, optional
            Ignored for NLS/MAP (no uncertainty estimation).

        Returns
        -------
        NDArray
            Predicted deposit balances of shape (T+1,).

        Examples
        --------
        >>> estimator = NLSEstimator()
        >>> result = estimator.fit(data)
        >>> V_pred = estimator.predict(data, result)
        """
        _ = ci_prob  # Unused in NLS
        if uncertainty:
            warnings.warn(
                "NLSEstimator does not support uncertainty estimation. "
                "Use MCMCEstimator for posterior predictive distributions.",
                UserWarning,
                stacklevel=2,
            )

        params = result.params
        V0 = float(data.V0)
        inflow = np.asarray(data.inflow, dtype=float)
        T = inflow.shape[0] - 1

        # S2 covariates (hazard)
        z = data.z
        if z is not None:
            z = np.asarray(z, dtype=float)
            if z.ndim == 1:
                z = z.reshape(-1, 1)
            beta = params.get("beta")
            if beta is not None:
                weight = np.exp(z @ beta)
            else:
                weight = np.ones(T + 1)
        else:
            weight = np.ones(T + 1)

        # w1: constant or time-varying
        if "w1_a" in params:
            # Time-varying w1
            w1_features = data.w1_features
            if w1_features is None:
                raise ValueError(
                    "w1_features required in data for model with time-varying w1"
                )
            w1_features = np.asarray(w1_features, dtype=float)
            if w1_features.ndim == 1:
                w1_features = w1_features.reshape(-1, 1)
            w1 = np.array(w1_logistic(params["w1_a"], params["w1_b"], w1_features))
        else:
            w1 = params["w1"]

        Vhat = np.array(
            V_model(
                lam=params["lambda"],
                gam=params["gamma"],
                w1=w1,
                h=params["h"],
                m=params["m"],
                V0=V0,
                inflow=inflow,
                weight=weight,
            )
        )

        return Vhat
