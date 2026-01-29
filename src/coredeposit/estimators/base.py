"""Abstract base class for core deposit estimators.

This module defines the common interface that all estimation methods must
implement. Subclasses include NLSEstimator for non-linear least squares
and MCMCEstimator for Bayesian MCMC estimation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import CoreDepositData, EstimationResult


class Estimator(ABC):
    """Abstract base class for all core deposit estimators.

    This class defines the interface that all estimation methods must implement.
    Subclasses should implement the `fit` method to estimate model parameters
    from observed deposit data.

    The core deposit model has the following parameters to estimate:
    - λ (lambda): Weibull scale parameter for sticky deposit survival
    - γ (gamma): Weibull shape parameter for sticky deposit survival
    - w1: Proportion of deposits that are transactional (short-lived)
    - h: Exit rate for transactional deposits in the first period
    - m: Average age of initial deposits at time 0
    - β (beta): Covariate coefficients (optional, for models with covariates)

    Examples
    --------
    >>> from coredeposit import NLSEstimator, CoreDepositData
    >>> data = CoreDepositData(V_obs=balances, inflow=deposits, V0=initial)
    >>> estimator = NLSEstimator()
    >>> result = estimator.fit(data)
    >>> print(result.params)
    """

    @abstractmethod
    def fit(self, data: CoreDepositData) -> EstimationResult:
        """
        Estimate model parameters from the given data.

        Parameters
        ----------
        data : CoreDepositData
            Time series of balances and inflows (and optional covariates).

        Returns
        -------
        EstimationResult
            Estimated parameters and diagnostics.
        """
        raise NotImplementedError
