"""Prior distributions for MAP estimation using scipy/numpy.

This module provides prior distribution specifications for Maximum A Posteriori
(MAP) estimation with NLSEstimator. Unlike the numpyro-based priors used for
MCMC, these priors use scipy.stats for compatibility with scipy.optimize.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class MAPPriors:
    """Container for prior distributions used in MAP estimation.

    This dataclass holds scipy.stats distributions for each parameter.
    The `neg_log_prob` method computes the negative log prior probability,
    which is added to the least squares objective for MAP estimation.

    Attributes
    ----------
    lambda_dist : scipy.stats distribution
        Prior for Weibull scale parameter λ (positive).
    gamma_dist : scipy.stats distribution
        Prior for Weibull shape parameter γ (positive).
    w1_dist : scipy.stats distribution
        Prior for transactional deposit proportion (0-1).
    h_dist : scipy.stats distribution
        Prior for transactional exit rate (0-1).
    m_dist : scipy.stats distribution
        Prior for initial deposit age in months (positive).
    beta_dist : scipy.stats distribution or None
        Prior for covariate coefficients. None if no covariates.

    Examples
    --------
    >>> priors = MAPPriors(
    ...     lambda_dist=stats.lognorm(s=0.6, scale=np.exp(-3.0)),
    ...     gamma_dist=stats.lognorm(s=0.35, scale=1.0),
    ...     w1_dist=stats.beta(2.0, 6.0),
    ...     h_dist=stats.beta(2.0, 6.0),
    ...     m_dist=stats.lognorm(s=0.4, scale=np.exp(2.5)),
    ... )
    >>> neg_log_prior = priors.neg_log_prob(
    ...     lam=0.05, gam=1.0, w1=0.25, h=0.8, m=12.0
    ... )
    """

    lambda_dist: stats.rv_continuous
    gamma_dist: stats.rv_continuous
    w1_dist: stats.rv_continuous
    h_dist: stats.rv_continuous
    m_dist: stats.rv_continuous
    beta_dist: stats.rv_continuous | None = None

    def neg_log_prob(
        self,
        lam: float,
        gam: float,
        w1: float,
        h: float,
        m: float,
        beta: np.ndarray | None = None,
    ) -> float:
        """Compute negative log prior probability.

        Parameters
        ----------
        lam : float
            Weibull scale parameter.
        gam : float
            Weibull shape parameter.
        w1 : float
            Transactional deposit proportion.
        h : float
            Transactional exit rate.
        m : float
            Initial deposit age.
        beta : ndarray or None
            Covariate coefficients.

        Returns
        -------
        float
            Negative log prior probability. Returns inf if any parameter
            is outside the support of its prior distribution.
        """
        neg_log_p = 0.0

        # Lambda prior
        log_p = self.lambda_dist.logpdf(lam)
        if not np.isfinite(log_p):
            return np.inf
        neg_log_p -= log_p

        # Gamma prior
        log_p = self.gamma_dist.logpdf(gam)
        if not np.isfinite(log_p):
            return np.inf
        neg_log_p -= log_p

        # w1 prior (handle boundary)
        if w1 <= 0 or w1 >= 1:
            return np.inf
        log_p = self.w1_dist.logpdf(w1)
        if not np.isfinite(log_p):
            return np.inf
        neg_log_p -= log_p

        # h prior (handle boundary)
        if h <= 0 or h >= 1:
            return np.inf
        log_p = self.h_dist.logpdf(h)
        if not np.isfinite(log_p):
            return np.inf
        neg_log_p -= log_p

        # m prior
        log_p = self.m_dist.logpdf(m)
        if not np.isfinite(log_p):
            return np.inf
        neg_log_p -= log_p

        # Beta prior (if covariates)
        if beta is not None and self.beta_dist is not None:
            for b in beta:
                log_p = self.beta_dist.logpdf(b)
                if not np.isfinite(log_p):
                    return np.inf
                neg_log_p -= log_p

        return neg_log_p


def default_map_priors(p: int = 0) -> MAPPriors:
    """Create default prior distributions for MAP estimation.

    These priors are designed to match the default MCMC priors
    (CoreDepositPriors) but use scipy.stats distributions.

    Parameters
    ----------
    p : int, optional
        Number of covariate dimensions. Default is 0 (no covariates).

    Returns
    -------
    MAPPriors
        Dataclass containing all prior distributions:

        - lambda_dist: LogNormal(μ=-3.0, σ=0.6), median ~0.05
        - gamma_dist: LogNormal(μ=0.0, σ=0.35), median ~1.0
        - w1_dist: Beta(2.0, 6.0), mean ~0.25
        - h_dist: Beta(2.0, 6.0), mean ~0.25
        - m_dist: LogNormal(μ=2.5, σ=0.4), median ~12 months
        - beta_dist: Normal(0, 0.3) for each coefficient, or None

    Notes
    -----
    scipy.stats.lognorm uses a different parameterization than NumPyro:
    - NumPyro: LogNormal(loc=μ, scale=σ) where X = exp(μ + σ*Z)
    - scipy: lognorm(s=σ, scale=exp(μ)) where X = scale * exp(s*Z)

    Examples
    --------
    >>> priors = default_map_priors()
    >>> # Check median of lambda prior
    >>> priors.lambda_dist.median()  # ~0.05
    """
    # scipy.stats.lognorm(s=sigma, scale=exp(mu))
    # corresponds to X = exp(mu + sigma * Z)
    lambda_dist = stats.lognorm(s=0.6, scale=np.exp(-3.0))
    gamma_dist = stats.lognorm(s=0.35, scale=np.exp(0.0))
    w1_dist = stats.beta(2.0, 6.0)
    h_dist = stats.beta(2.0, 6.0)
    m_dist = stats.lognorm(s=0.4, scale=np.exp(2.5))

    beta_dist = stats.norm(0.0, 0.3) if p > 0 else None

    return MAPPriors(
        lambda_dist=lambda_dist,
        gamma_dist=gamma_dist,
        w1_dist=w1_dist,
        h_dist=h_dist,
        m_dist=m_dist,
        beta_dist=beta_dist,
    )
