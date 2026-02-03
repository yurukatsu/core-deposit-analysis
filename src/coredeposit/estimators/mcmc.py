"""Bayesian MCMC estimator for core deposit model.

This module provides the MCMCEstimator class for Bayesian inference of
core deposit model parameters using NumPyro's NUTS (No-U-Turn Sampler).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_value

from ..types import CoreDepositData, EstimationResult, NDArray
from ..normalize import normalize
from ..model.balance import V_model
from .base import Estimator
from .priors import CoreDepositPriors, default_priors


class MCMCEstimator(Estimator):
    """Bayesian MCMC estimator for the core deposit model.

    This estimator performs Bayesian inference using NumPyro's NUTS sampler
    (No-U-Turn Sampler), a variant of Hamiltonian Monte Carlo. It returns
    posterior samples of all model parameters, enabling uncertainty
    quantification and probabilistic predictions.

    The model supports both Normal and Student-t likelihoods. Student-t
    provides robustness against outliers due to its heavier tails.

    Parameters
    ----------
    num_warmup : int, optional
        Number of warmup (burn-in) iterations per chain. These samples are
        used to tune the sampler and are discarded. Default is 1500.
    num_samples : int, optional
        Number of posterior samples to draw per chain after warmup.
        Total samples = num_samples × num_chains. Default is 2000.
    num_chains : int, optional
        Number of independent MCMC chains to run. Multiple chains enable
        convergence diagnostics (R-hat). Default is 2.
    seed : int, optional
        Random seed for reproducibility. Default is 0.
    target_accept : float, optional
        Target acceptance probability for NUTS. Higher values (closer to 1)
        lead to smaller step sizes and more conservative sampling.
        Default is 0.9.
    likelihood : str, optional
        Likelihood function for observations. Options:

        - 'studentt': Student-t distribution (default, robust to outliers)
        - 'normal': Normal distribution

    priors : CoreDepositPriors or None, optional
        Custom prior distributions. If None, uses default_priors() which
        provides weakly informative priors. Default is None.
    init_params : dict or None, optional
        Initial parameter values for MCMC chains. If provided, should be a
        dictionary with keys matching parameter names ('lambda', 'gamma',
        'w1', 'h', 'm', and optionally 'beta'). Useful for initializing
        from NLS estimates to improve convergence. Default is None.
    ar_errors : bool, optional
        If True, model observation errors with AR(1) autocorrelation.
        This accounts for temporal correlation in prediction errors.
        Default is False.

    Attributes
    ----------
    num_warmup : int
        Number of warmup iterations.
    num_samples : int
        Number of posterior samples per chain.
    num_chains : int
        Number of MCMC chains.
    seed : int
        Random seed.
    target_accept : float
        Target acceptance probability.
    likelihood : str
        Likelihood function type.
    priors : CoreDepositPriors or None
        Prior distributions.
    init_params : dict or None
        Initial parameter values.
    ar_errors : bool
        Whether to use AR(1) error model.

    Examples
    --------
    >>> from coredeposit import MCMCEstimator, CoreDepositData
    >>> import numpy as np
    >>> data = CoreDepositData(
    ...     V_obs=np.array([100, 95, 90, 85, 80]),
    ...     inflow=np.array([10, 12, 8, 11, 9]),
    ...     V0=100.0,
    ... )
    >>> estimator = MCMCEstimator(num_warmup=500, num_samples=1000)
    >>> result = estimator.fit(data)
    >>> # Posterior samples for lambda
    >>> lambda_samples = result.params['lambda']
    >>> print(f"lambda: {lambda_samples.mean():.3f} ± {lambda_samples.std():.3f}")

    Notes
    -----
    The data is automatically normalized by max(V_obs) before fitting to
    improve numerical stability. The scale factor is stored in diagnostics.

    For models with covariates, provide z in CoreDepositData. The covariate
    effect enters as weight[t] = exp(β'z[t]) in the hazard function.
    """

    def __init__(
        self,
        *,
        num_warmup: int = 1500,
        num_samples: int = 2000,
        num_chains: int = 2,
        seed: int = 0,
        target_accept: float = 0.9,
        likelihood: str = "studentt",
        priors: CoreDepositPriors | None = None,
        init_params: dict[str, float] | None = None,
        ar_errors: bool = False,
    ):
        """Initialize the MCMC estimator.

        Parameters
        ----------
        num_warmup : int, optional
            Number of warmup iterations per chain. Default is 1500.
        num_samples : int, optional
            Number of posterior samples per chain. Default is 2000.
        num_chains : int, optional
            Number of MCMC chains. Default is 2.
        seed : int, optional
            Random seed for reproducibility. Default is 0.
        target_accept : float, optional
            Target acceptance probability for NUTS. Default is 0.9.
        likelihood : str, optional
            Likelihood function: 'studentt' or 'normal'. Default is 'studentt'.
        priors : CoreDepositPriors or None, optional
            Custom prior distributions. Default is None (use default priors).
        init_params : dict or None, optional
            Initial parameter values for MCMC chains. Keys should match
            parameter names ('lambda', 'gamma', 'w1', 'h', 'm').
            Useful for initializing from NLS estimates. Default is None.
        ar_errors : bool, optional
            If True, model errors with AR(1) autocorrelation. Default is False.
        """
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.seed = seed
        self.target_accept = target_accept
        self.likelihood = likelihood
        self.priors = priors
        self.init_params = init_params
        self.ar_errors = ar_errors

    def fit(self, data: CoreDepositData) -> EstimationResult:
        """Estimate model parameters using Bayesian MCMC.

        Performs Bayesian inference using NumPyro's NUTS sampler. Returns
        posterior samples of all model parameters, enabling uncertainty
        quantification.

        Parameters
        ----------
        data : CoreDepositData
            Input data containing:

            - V_obs : array of shape (T+1,) with observed deposit balances
            - inflow : array of shape (T+1,) with deposit inflows
            - V0 : initial deposit balance at time 0
            - z : optional covariates of shape (T+1,) or (T+1, p)

        Returns
        -------
        EstimationResult
            Result containing:

            params : dict
                Posterior samples for each parameter (arrays of shape
                (num_chains × num_samples,)):

                - 'lambda': Weibull scale parameter samples
                - 'gamma': Weibull shape parameter samples
                - 'w1': Transactional deposit proportion samples
                - 'h': Transactional exit rate samples
                - 'm': Initial deposit age samples
                - 'sigma': Observation noise std samples
                - 'nu': Student-t degrees of freedom samples (if likelihood='studentt')
                - 'rho': AR(1) autocorrelation coefficient (if ar_errors=True)
                - 'beta': Covariate coefficient samples (if z provided)

            diagnostics : dict
                MCMC diagnostics:

                - 'scale': Normalization scale factor (float). To recover
                  original scale predictions, multiply by this value.
                - 'mcmc': NumPyro MCMC object for advanced diagnostics and
                  ArviZ integration.

        Notes
        -----
        A summary with R-hat convergence diagnostics and effective sample
        sizes is printed to stdout via mcmc.print_summary().

        The posterior samples can be used for:
        - Point estimates: np.mean(samples['lambda'])
        - Credible intervals: np.percentile(samples['lambda'], [2.5, 97.5])
        - Posterior predictive checks
        - Model comparison via WAIC or LOO-CV

        For ArviZ visualization::

            import arviz as az
            idata = az.from_numpyro(result.diagnostics["mcmc"])
            az.plot_trace(idata)
        """
        V_obs_np, inflow_np, V0_np, scale = normalize(
            data.V_obs, data.inflow, data.V0
        )

        V_obs = jnp.asarray(V_obs_np, dtype=jnp.float64)
        inflow = jnp.asarray(inflow_np, dtype=jnp.float64)
        V0 = jnp.asarray(V0_np, dtype=jnp.float64)

        T = V_obs.shape[0] - 1

        has_covariate = data.z is not None
        if has_covariate:
            z = jnp.asarray(data.z, dtype=jnp.float64)
            if z.ndim == 1:
                z = z.reshape(-1, 1)
            p = z.shape[1]
        else:
            z = None
            p = 0

        priors = self.priors or default_priors(p)

        def model():
            lam = numpyro.sample("lambda", priors.lambda_prior)
            gam = numpyro.sample("gamma", priors.gamma_prior)
            w1 = numpyro.sample("w1", priors.w1_prior)
            h = numpyro.sample("h", priors.h_prior)
            m = numpyro.sample("m", priors.m_prior)
            sigma = numpyro.sample("sigma", priors.sigma_prior)

            if has_covariate and priors.beta_prior is not None:
                beta = numpyro.sample("beta", priors.beta_prior)
                weight = jnp.exp(z @ beta)
            else:
                weight = jnp.ones(T + 1, dtype=jnp.float64)

            Vhat = V_model(
                lam=lam,
                gam=gam,
                w1=w1,
                h=h,
                m=m,
                V0=V0,
                inflow=inflow,
                weight=weight,
            )

            idx = jnp.arange(1, T + 1)

            if self.ar_errors:
                # AR(1) error model: epsilon[t] = rho * epsilon[t-1] + eta[t]
                rho = numpyro.sample("rho", priors.rho_prior)

                # Compute residuals
                residuals = V_obs - Vhat

                # Conditional likelihood for AR(1) process
                # For t=1: marginal variance is sigma^2 / (1 - rho^2)
                # For t>1: obs[t] ~ N(Vhat[t] + rho*(obs[t-1] - Vhat[t-1]), sigma)
                sigma_marginal = sigma / jnp.sqrt(1 - rho**2 + 1e-8)

                # First observation (marginal distribution)
                numpyro.sample(
                    "obs_0",
                    priors.normal_likelihood(Vhat[1], sigma_marginal),
                    obs=V_obs[1],
                )

                # Subsequent observations (conditional distribution)
                if T > 1:
                    conditional_mean = Vhat[2:] + rho * residuals[1:-1]
                    numpyro.sample(
                        "obs",
                        priors.normal_likelihood(conditional_mean, sigma).to_event(1),
                        obs=V_obs[2:],
                    )
            elif self.likelihood == "normal":
                numpyro.sample(
                    "obs",
                    priors.normal_likelihood(Vhat[idx], sigma),
                    obs=V_obs[idx],
                )
            else:
                nu = numpyro.sample("nu", priors.nu_prior) + 2.0
                numpyro.sample(
                    "obs",
                    priors.studentt_likelihood(Vhat[idx], sigma, nu),
                    obs=V_obs[idx],
                )

        # Use init_to_value strategy if init_params provided
        # This allows partial initialization (only specified params)
        if self.init_params is not None:
            init_values = {
                k: jnp.asarray(v, dtype=jnp.float64)
                for k, v in self.init_params.items()
            }
            init_strategy = init_to_value(values=init_values)
        else:
            init_strategy = None

        kernel = NUTS(
            model,
            target_accept_prob=self.target_accept,
            init_strategy=init_strategy,
        )
        mcmc = MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
        )

        mcmc.run(jax.random.PRNGKey(self.seed))
        mcmc.print_summary()

        return EstimationResult(
            params=mcmc.get_samples(),
            diagnostics={"scale": float(scale), "mcmc": mcmc},
        )

    def predict(
        self,
        data: CoreDepositData,
        result: EstimationResult,
        *,
        uncertainty: bool = False,
        ci_prob: float = 0.95,
    ) -> NDArray | dict[str, NDArray]:
        """Predict deposit balances using posterior samples.

        Generates predictions using the posterior samples from MCMC estimation.
        By default returns the posterior mean prediction. With uncertainty=True,
        returns the full posterior predictive distribution.

        Parameters
        ----------
        data : CoreDepositData
            Input data for prediction containing inflows and initial balance.
            Can be the same data used for fitting or new data.
        result : EstimationResult
            Result from a previous call to `fit()`.
        uncertainty : bool, optional
            If True, return full posterior predictive distribution including
            credible intervals. Default is False.
        ci_prob : float, optional
            Credible interval probability. Default is 0.95 (95% CI).
            For example, 0.90 gives 90% CI, 0.50 gives 50% CI.

        Returns
        -------
        NDArray or dict[str, NDArray]
            If uncertainty=False:
                Predicted balances using posterior mean, shape (T+1,).

            If uncertainty=True:
                Dictionary containing:

                - 'mean': posterior mean prediction, shape (T+1,)
                - 'samples': all posterior predictions, shape (n_samples, T+1)
                - 'lower': lower bound of credible interval, shape (T+1,)
                - 'upper': upper bound of credible interval, shape (T+1,)

        Examples
        --------
        >>> estimator = MCMCEstimator()
        >>> result = estimator.fit(data)
        >>> # Point prediction (posterior mean)
        >>> V_pred = estimator.predict(data, result)
        >>> # With 95% credible interval (default)
        >>> pred = estimator.predict(data, result, uncertainty=True)
        >>> # With 90% credible interval
        >>> pred = estimator.predict(data, result, uncertainty=True, ci_prob=0.90)
        """
        params = result.params
        V0 = float(data.V0)
        inflow = np.asarray(data.inflow, dtype=float)
        T = inflow.shape[0] - 1

        z = data.z
        has_covariate = z is not None
        if has_covariate:
            z = np.asarray(z, dtype=float)
            if z.ndim == 1:
                z = z.reshape(-1, 1)

        # Get sample arrays
        lam_samples = np.asarray(params["lambda"])
        gam_samples = np.asarray(params["gamma"])
        w1_samples = np.asarray(params["w1"])
        h_samples = np.asarray(params["h"])
        m_samples = np.asarray(params["m"])
        n_samples = len(lam_samples)

        if has_covariate and "beta" in params:
            beta_samples = np.asarray(params["beta"])
        else:
            beta_samples = None

        # Compute predictions for each posterior sample
        V_samples = np.zeros((n_samples, T + 1))

        for i in range(n_samples):
            if beta_samples is not None:
                weight = np.exp(z @ beta_samples[i])
            else:
                weight = np.ones(T + 1)

            V_samples[i] = np.array(
                V_model(
                    lam=lam_samples[i],
                    gam=gam_samples[i],
                    w1=w1_samples[i],
                    h=h_samples[i],
                    m=m_samples[i],
                    V0=V0,
                    inflow=inflow,
                    weight=weight,
                )
            )

        mean = V_samples.mean(axis=0)

        if uncertainty:
            alpha = (1.0 - ci_prob) / 2.0
            lower_pct = alpha * 100
            upper_pct = (1.0 - alpha) * 100
            return {
                "mean": mean,
                "samples": V_samples,
                "lower": np.percentile(V_samples, lower_pct, axis=0),
                "upper": np.percentile(V_samples, upper_pct, axis=0),
            }
        else:
            return mean


# Backward compatibility alias
NoCovariateMCMCEstimator = MCMCEstimator
"""Alias for MCMCEstimator.

Kept for backward compatibility. The MCMCEstimator now handles both
covariate and no-covariate models automatically based on whether
`data.z` is provided.

.. deprecated::
    Use MCMCEstimator directly instead.
"""
