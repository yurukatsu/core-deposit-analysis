"""Prior distributions for Bayesian MCMC estimation.

This module defines the prior distribution specifications used by the
MCMCEstimator for Bayesian inference of core deposit model parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpyro.distributions as dist


@dataclass
class CoreDepositPriors:
    """Container for prior distributions of all model parameters.

    This dataclass holds the prior distributions for each parameter in the
    core deposit model. These priors are used by the MCMCEstimator during
    Bayesian inference with NumPyro's NUTS sampler.

    Attributes
    ----------
    lambda_prior : numpyro.distributions.Distribution
        Prior for the Weibull scale parameter λ (positive).
        Default: LogNormal(-3.0, 0.6), centered around ~0.05.
    gamma_prior : numpyro.distributions.Distribution
        Prior for the Weibull shape parameter γ (positive).
        Default: LogNormal(0.0, 0.35), centered around 1.0.
    w1_prior : numpyro.distributions.Distribution
        Prior for the transactional deposit proportion (0-1).
        Default: Beta(2.0, 6.0), favoring lower values ~0.25.
    h_prior : numpyro.distributions.Distribution
        Prior for the transactional exit rate (0-1).
        Default: Beta(2.0, 6.0), favoring lower values ~0.25.
    m_prior : numpyro.distributions.Distribution
        Prior for the initial deposit age in months (positive).
        Default: LogNormal(2.5, 0.4), centered around ~12 months.
    sigma_prior : numpyro.distributions.Distribution
        Prior for the observation noise standard deviation (positive).
        Default: HalfNormal(0.05).
    nu_prior : numpyro.distributions.Distribution
        Prior for Student-t degrees of freedom minus 2 (positive).
        Default: Exponential(1.0). Note: 2.0 is added in the model.
    beta_prior : numpyro.distributions.Distribution or None
        Prior for covariate coefficients. None for no-covariate model.
        Default: Normal(0.0, 0.3) for each coefficient.

    Examples
    --------
    >>> priors = CoreDepositPriors(
    ...     lambda_prior=dist.LogNormal(-3.0, 0.6),
    ...     gamma_prior=dist.LogNormal(0.0, 0.35),
    ...     w1_prior=dist.Beta(2.0, 6.0),
    ...     h_prior=dist.Beta(2.0, 6.0),
    ...     m_prior=dist.LogNormal(2.5, 0.4),
    ...     sigma_prior=dist.HalfNormal(0.05),
    ...     nu_prior=dist.Exponential(1.0),
    ... )
    """

    lambda_prior: dist.Distribution
    gamma_prior: dist.Distribution
    w1_prior: dist.Distribution
    h_prior: dist.Distribution
    m_prior: dist.Distribution
    sigma_prior: dist.Distribution
    nu_prior: dist.Distribution
    beta_prior: dist.Distribution | None = None

    @staticmethod
    def normal_likelihood(loc, sigma):
        """Create a Normal likelihood distribution.

        Parameters
        ----------
        loc : ArrayLike
            Mean (location) of the Normal distribution.
        sigma : ArrayLike
            Standard deviation (scale) of the Normal distribution.

        Returns
        -------
        numpyro.distributions.Normal
            Normal distribution with specified location and scale.
        """
        return dist.Normal(loc, sigma)

    @staticmethod
    def studentt_likelihood(loc, sigma, nu):
        """Create a Student-t likelihood distribution.

        The Student-t distribution provides heavier tails than Normal,
        making it more robust to outliers in the observed data.

        Parameters
        ----------
        loc : ArrayLike
            Location parameter (similar to mean).
        sigma : ArrayLike
            Scale parameter (similar to standard deviation).
        nu : ArrayLike
            Degrees of freedom. Lower values give heavier tails.

        Returns
        -------
        numpyro.distributions.StudentT
            Student-t distribution with specified parameters.
        """
        return dist.StudentT(df=nu, loc=loc, scale=sigma)


def default_priors(p: int = 0) -> CoreDepositPriors:
    """Create default prior distributions for the core deposit model.

    This function returns weakly informative priors suitable for typical
    bank deposit data. The priors are designed to:
    - Keep parameters in reasonable ranges
    - Allow the data to dominate the posterior
    - Ensure numerical stability during MCMC sampling

    Parameters
    ----------
    p : int, optional
        Number of covariate dimensions. If p > 0, beta_prior is created
        as a multivariate Normal distribution. Default is 0 (no covariates).

    Returns
    -------
    CoreDepositPriors
        Dataclass containing all prior distributions:

        - lambda_prior: LogNormal(-3.0, 0.6)
            Weibull scale, median ~0.05, supports roughly [0.01, 0.3]
        - gamma_prior: LogNormal(0.0, 0.35)
            Weibull shape, median ~1.0, supports roughly [0.5, 2.0]
        - w1_prior: Beta(2.0, 6.0)
            Transactional proportion, mean ~0.25, favors lower values
        - h_prior: Beta(2.0, 6.0)
            Exit rate, mean ~0.25, favors lower values
        - m_prior: LogNormal(2.5, 0.4)
            Initial age in months, median ~12, supports roughly [5, 30]
        - sigma_prior: HalfNormal(0.05)
            Observation noise, concentrated near 0
        - nu_prior: Exponential(1.0)
            Student-t df (before adding 2), mean 1.0
        - beta_prior: Normal(0, 0.3) for each of p coefficients, or None

    Examples
    --------
    >>> # No covariates
    >>> priors = default_priors()
    >>> priors.beta_prior is None
    True

    >>> # With 3 covariates
    >>> priors = default_priors(p=3)
    >>> priors.beta_prior is not None
    True
    """
    beta_prior = None
    if p > 0:
        beta_prior = dist.Normal(0.0, 0.3).expand((p,)).to_event(1)

    return CoreDepositPriors(
        lambda_prior=dist.LogNormal(-3.0, 0.6),
        gamma_prior=dist.LogNormal(0.0, 0.35),
        beta_prior=beta_prior,
        w1_prior=dist.Beta(2.0, 6.0),
        h_prior=dist.Beta(2.0, 6.0),
        m_prior=dist.LogNormal(2.5, 0.4),
        sigma_prior=dist.HalfNormal(0.05),
        nu_prior=dist.Exponential(1.0),
    )
