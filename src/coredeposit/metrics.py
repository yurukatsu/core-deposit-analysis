"""Derived metrics from estimation results.

This module provides functions to compute derived quantities from
estimated model parameters, such as median survival times and
other risk metrics.
"""

from __future__ import annotations

import numpy as np

from .types import EstimationResult, NDArray


def weibull_median_survival(
    lam: float | NDArray,
    gam: float | NDArray,
    beta: NDArray | None = None,
    z: NDArray | None = None,
) -> float | NDArray:
    """Compute median survival time for Weibull distribution.

    The median survival time (half-life) is the time at which 50% of
    deposits have exited. For a Weibull distribution with survival
    function S(t) = exp(-(λt)^γ), the median is:

        t_50 = (ln(2))^(1/γ) / λ

    With covariates z and coefficients β, the survival function becomes
    S(t|z) = exp(-(λt)^γ exp(β'z)), and the median is:

        t_50(z) = (ln(2))^(1/γ) / λ × exp(-β'z/γ)

    Parameters
    ----------
    lam : float or NDArray
        Weibull scale parameter (λ > 0).
    gam : float or NDArray
        Weibull shape parameter (γ > 0).
    beta : NDArray or None, optional
        Covariate coefficients. Shape (p,) for NLS or (n_samples, p) for MCMC.
        Default is None (no covariate adjustment).
    z : NDArray or None, optional
        Covariate values at which to evaluate. Shape (p,).
        Default is None (no covariate adjustment).

    Returns
    -------
    float or NDArray
        Median survival time in the same units as the model (typically months).
    """
    # Baseline median survival
    t50_baseline = np.power(np.log(2), 1.0 / gam) / lam

    # Apply covariate adjustment if both beta and z are provided
    if beta is not None and z is not None:
        z = np.asarray(z).ravel()
        beta = np.asarray(beta)

        if beta.ndim == 1:
            # NLS: single beta vector
            linear_pred = np.dot(beta, z)
        else:
            # MCMC: beta has shape (n_samples, p)
            linear_pred = beta @ z

        # t_50(z) = t_50(baseline) * exp(-β'z / γ)
        adjustment = np.exp(-linear_pred / gam)
        return t50_baseline * adjustment

    return t50_baseline


def weibull_quantile_survival(
    lam: float | NDArray,
    gam: float | NDArray,
    q: float = 0.5,
    beta: NDArray | None = None,
    z: NDArray | None = None,
) -> float | NDArray:
    """Compute q-quantile survival time for Weibull distribution.

    The q-quantile survival time is the time at which (1-q) fraction of
    deposits remain. For q=0.5, this is the median (half-life).

    For a Weibull distribution with survival function S(t) = exp(-(λt)^γ),
    solving S(t) = 1-q gives:

        t_q = (-ln(1-q))^(1/γ) / λ

    With covariates z and coefficients β:

        t_q(z) = (-ln(1-q))^(1/γ) / λ × exp(-β'z/γ)

    Parameters
    ----------
    lam : float or NDArray
        Weibull scale parameter (λ > 0).
    gam : float or NDArray
        Weibull shape parameter (γ > 0).
    q : float, optional
        Quantile (0 < q < 1). Default is 0.5 (median).
        - q=0.5: time when 50% have exited (median)
        - q=0.9: time when 90% have exited
        - q=0.1: time when 10% have exited
    beta : NDArray or None, optional
        Covariate coefficients. Default is None.
    z : NDArray or None, optional
        Covariate values at which to evaluate. Default is None.

    Returns
    -------
    float or NDArray
        Quantile survival time in the same units as the model (typically months).
    """
    # Baseline quantile survival
    tq_baseline = np.power(-np.log(1.0 - q), 1.0 / gam) / lam

    # Apply covariate adjustment if both beta and z are provided
    if beta is not None and z is not None:
        z = np.asarray(z).ravel()
        beta = np.asarray(beta)

        if beta.ndim == 1:
            # NLS: single beta vector
            linear_pred = np.dot(beta, z)
        else:
            # MCMC: beta has shape (n_samples, p)
            linear_pred = beta @ z

        adjustment = np.exp(-linear_pred / gam)
        return tq_baseline * adjustment

    return tq_baseline


def compute_median_survival(
    result: EstimationResult,
    z: NDArray | None = None,
    *,
    ci_prob: float = 0.95,
) -> float | dict[str, float]:
    """Compute median survival time for sticky deposits.

    Calculates the median survival time (half-life) from the estimated
    Weibull parameters. For MCMC results, also computes credible intervals.

    Parameters
    ----------
    result : EstimationResult
        Estimation result from NLSEstimator or MCMCEstimator.
    z : NDArray or None, optional
        Covariate values at which to evaluate the median survival.
        Shape (p,) where p is the number of covariates. If the model
        was estimated with covariates but z is None, returns the
        baseline survival (z=0). Default is None.
    ci_prob : float, optional
        Credible interval probability (MCMC only). Default is 0.95.

    Returns
    -------
    float or dict[str, float]
        For NLS results: median survival time as float.

        For MCMC results: dict containing:
        - 'mean': posterior mean of median survival
        - 'median': posterior median of median survival
        - 'samples': all posterior samples
        - 'lower': lower bound of credible interval
        - 'upper': upper bound of credible interval

    Examples
    --------
    >>> # NLS: point estimate (no covariates)
    >>> result_nls = nls.fit(data)
    >>> t50 = compute_median_survival(result_nls)
    >>> print(f"Median survival: {t50:.1f} months")

    >>> # MCMC: with uncertainty (no covariates)
    >>> result_mcmc = mcmc.fit(data)
    >>> t50 = compute_median_survival(result_mcmc, ci_prob=0.95)
    >>> print(f"Median survival: {t50['mean']:.1f} months")
    >>> print(f"95% CI: [{t50['lower']:.1f}, {t50['upper']:.1f}]")

    >>> # With covariates: median survival at specific z values
    >>> z_high = np.array([1.0, 0.5])  # High risk scenario
    >>> t50_high = compute_median_survival(result_mcmc, z=z_high)
    >>> print(f"Median survival (high risk): {t50_high['mean']:.1f} months")
    """
    params = result.params
    lam = params["lambda"]
    gam = params["gamma"]
    beta = params.get("beta")

    # Check if MCMC (arrays) or NLS (scalars)
    if isinstance(lam, np.ndarray) or hasattr(lam, "__len__"):
        # MCMC: compute for all samples
        lam_arr = np.asarray(lam)
        gam_arr = np.asarray(gam)
        beta_arr = np.asarray(beta) if beta is not None else None

        t50_samples = weibull_median_survival(lam_arr, gam_arr, beta_arr, z)

        alpha = (1.0 - ci_prob) / 2.0
        lower_pct = alpha * 100
        upper_pct = (1.0 - alpha) * 100

        return {
            "mean": float(t50_samples.mean()),
            "median": float(np.median(t50_samples)),
            "samples": t50_samples,
            "lower": float(np.percentile(t50_samples, lower_pct)),
            "upper": float(np.percentile(t50_samples, upper_pct)),
        }
    else:
        # NLS: point estimate
        return float(
            weibull_median_survival(float(lam), float(gam), beta, z)
        )


def compute_quantile_survival(
    result: EstimationResult,
    q: float = 0.5,
    z: NDArray | None = None,
    *,
    ci_prob: float = 0.95,
) -> float | dict[str, float]:
    """Compute q-quantile survival time for sticky deposits.

    Calculates the time at which (1-q) fraction of deposits remain.

    Parameters
    ----------
    result : EstimationResult
        Estimation result from NLSEstimator or MCMCEstimator.
    q : float, optional
        Quantile (0 < q < 1). Default is 0.5 (median).
        - q=0.5: time when 50% have exited (median/half-life)
        - q=0.9: time when 90% have exited
        - q=0.1: time when 10% have exited
    z : NDArray or None, optional
        Covariate values at which to evaluate. Default is None.
    ci_prob : float, optional
        Credible interval probability (MCMC only). Default is 0.95.

    Returns
    -------
    float or dict[str, float]
        For NLS results: quantile survival time as float.

        For MCMC results: dict containing:
        - 'mean': posterior mean
        - 'median': posterior median
        - 'samples': all posterior samples
        - 'lower': lower bound of credible interval
        - 'upper': upper bound of credible interval

    Examples
    --------
    >>> # Time when 90% of deposits have exited
    >>> t90 = compute_quantile_survival(result, q=0.9)
    >>> print(f"90% exit time: {t90['mean']:.1f} months")

    >>> # With covariates
    >>> t90 = compute_quantile_survival(result, q=0.9, z=np.array([1.0]))
    """
    params = result.params
    lam = params["lambda"]
    gam = params["gamma"]
    beta = params.get("beta")

    # Check if MCMC (arrays) or NLS (scalars)
    if isinstance(lam, np.ndarray) or hasattr(lam, "__len__"):
        # MCMC: compute for all samples
        lam_arr = np.asarray(lam)
        gam_arr = np.asarray(gam)
        beta_arr = np.asarray(beta) if beta is not None else None

        tq_samples = weibull_quantile_survival(lam_arr, gam_arr, q, beta_arr, z)

        alpha = (1.0 - ci_prob) / 2.0
        lower_pct = alpha * 100
        upper_pct = (1.0 - alpha) * 100

        return {
            "mean": float(tq_samples.mean()),
            "median": float(np.median(tq_samples)),
            "samples": tq_samples,
            "lower": float(np.percentile(tq_samples, lower_pct)),
            "upper": float(np.percentile(tq_samples, upper_pct)),
        }
    else:
        # NLS: point estimate
        return float(
            weibull_quantile_survival(float(lam), float(gam), q, beta, z)
        )
