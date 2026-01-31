"""Abstract base class for core deposit estimators.

This module defines the common interface that all estimation methods must
implement. Subclasses include NLSEstimator for non-linear least squares
and MCMCEstimator for Bayesian MCMC estimation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import CoreDepositData, EstimationResult, NDArray


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

    @abstractmethod
    def predict(
        self,
        data: CoreDepositData,
        result: EstimationResult,
        *,
        uncertainty: bool = False,
        ci_prob: float = 0.95,
    ) -> NDArray | dict[str, NDArray]:
        """
        Predict deposit balances using estimated parameters.

        Parameters
        ----------
        data : CoreDepositData
            Input data for prediction. Can be the same data used for fitting
            (in-sample prediction) or new data with different inflows.
        result : EstimationResult
            Estimation result from a previous call to `fit()`.
        uncertainty : bool, optional
            If True, return uncertainty estimates (MCMC only).
            Default is False.
        ci_prob : float, optional
            Credible interval probability (MCMC only). Default is 0.95 (95% CI).
            For example, 0.90 gives 90% CI (5th to 95th percentile).

        Returns
        -------
        NDArray or dict[str, NDArray]
            If uncertainty=False: predicted balances as array of shape (T+1,).
            If uncertainty=True (MCMC only): dict containing:

            - 'mean': posterior mean prediction, shape (T+1,)
            - 'samples': all posterior samples, shape (n_samples, T+1)
            - 'lower': lower bound of credible interval, shape (T+1,)
            - 'upper': upper bound of credible interval, shape (T+1,)
        """
        raise NotImplementedError
