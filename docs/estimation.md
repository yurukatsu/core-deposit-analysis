# Estimation Methods

This document describes the two estimation methods available in the coredeposit library: Non-linear Least Squares (NLS) and Bayesian MCMC.

## 1. Overview

| Method | Class | Purpose | Output |
|--------|-------|---------|--------|
| NLS | `NLSEstimator` | Fast point estimation | Parameter point estimates |
| MCMC | `MCMCEstimator` | Bayesian inference with uncertainty | Posterior samples |

Both estimators implement the same interface:

```python
estimator = Estimator(...)
result = estimator.fit(data)
predictions = estimator.predict(data, result)
```

---

## 2. Non-linear Least Squares (NLS)

### 2.1 Objective Function

NLS minimizes the sum of squared residuals:

$$
\min_{\Theta} \sum_{t=1}^{T} \rho\left( \frac{V_{\text{obs}}(t) - V_{\text{model}}(t \mid \Theta)}{f_{\text{scale}}} \right)
$$

where $\rho$ is the loss function and $f_{\text{scale}}$ is a scale parameter for robust losses.

### 2.2 Loss Functions

| Loss | Formula | Robustness |
|------|---------|------------|
| `linear` | $\rho(r) = r^2$ | Low (sensitive to outliers) |
| `soft_l1` | $\rho(r) = 2(\sqrt{1+r^2} - 1)$ | Medium (default) |
| `huber` | $\rho(r) = \begin{cases} r^2 & \|r\| \leq 1 \\ 2\|r\| - 1 & \text{otherwise} \end{cases}$ | High |
| `cauchy` | $\rho(r) = \ln(1 + r^2)$ | Very high |

### 2.3 Parameter Transformations

To enforce parameter constraints during optimization:

| Parameter | Domain | Transformation |
|-----------|--------|----------------|
| $\lambda$ | $(0, \infty)$ | $\lambda = \exp(\tilde{\lambda})$ |
| $\gamma$ | $(0, \infty)$ | $\gamma = \exp(\tilde{\gamma})$ |
| $m$ | $(0, \infty)$ | $m = \exp(\tilde{m})$ |
| $w_1$ | $(0, 1)$ | $w_1 = \sigma(\tilde{w}_1)$ (sigmoid) |
| $h$ | $(0, 1)$ | $h = \sigma(\tilde{h})$ (sigmoid) |
| $\beta$ | $\mathbb{R}^p$ | No transformation |

### 2.4 Usage

```python
from coredeposit import NLSEstimator, CoreDepositData

data = CoreDepositData(V_obs=..., inflow=..., V0=...)

estimator = NLSEstimator(
    loss='soft_l1',  # Loss function
    f_scale=0.05,    # Scale for robust losses
)

result = estimator.fit(data, m_fixed=None)  # Optional: fix m parameter

# Access point estimates
print(result.params['lambda'])  # float
print(result.params['gamma'])   # float
```

---

## 3. Bayesian MCMC

### 3.1 Statistical Model

The observation model is:

$$
V_{\text{obs}}(t) = V_{\text{model}}(t \mid \Theta) + \epsilon_t
$$

where $\epsilon_t$ follows either a Normal or Student-t distribution.

### 3.2 Likelihood Functions

#### Normal Likelihood

$$
V_{\text{obs}}(t) \sim \mathcal{N}(V_{\text{model}}(t), \sigma^2)
$$

#### Student-t Likelihood (default)

$$
V_{\text{obs}}(t) \sim \text{StudentT}(\nu, V_{\text{model}}(t), \sigma)
$$

The Student-t distribution provides robustness against outliers due to heavier tails.

### 3.3 Prior Distributions

Default weakly informative priors:

| Parameter | Prior | Interpretation |
|-----------|-------|----------------|
| $\lambda$ | $\text{LogNormal}(-3.0, 0.6)$ | Median ~0.05 |
| $\gamma$ | $\text{LogNormal}(0.0, 0.35)$ | Median ~1.0 |
| $w_1$ | $\text{Beta}(2.0, 6.0)$ | Mean ~0.25 |
| $h$ | $\text{Beta}(2.0, 6.0)$ | Mean ~0.25 |
| $m$ | $\text{LogNormal}(2.5, 0.4)$ | Median ~12 months |
| $\sigma$ | $\text{HalfNormal}(0.05)$ | Near zero |
| $\nu$ | $\text{Exponential}(1.0) + 2$ | Heavy tails |
| $\rho$ | $\text{Uniform}(-1, 1)$ | AR(1) coefficient |
| $\beta$ | $\mathcal{N}(0, 0.3)$ | Per covariate |

### 3.4 AR(1) Error Model

When `ar_errors=True`, the observation errors follow an AR(1) process:

$$
\epsilon_t = \rho \epsilon_{t-1} + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \sigma^2)
$$

This is implemented via conditional likelihood:

- **First observation** (marginal):
$$
V_{\text{obs}}(1) \sim \mathcal{N}\left(V_{\text{model}}(1), \frac{\sigma^2}{1-\rho^2}\right)
$$

- **Subsequent observations** (conditional):
$$
V_{\text{obs}}(t) \sim \mathcal{N}\left(V_{\text{model}}(t) + \rho (V_{\text{obs}}(t-1) - V_{\text{model}}(t-1)), \sigma^2\right)
$$

### 3.5 Initialization from NLS

MCMC convergence can be improved by initializing from NLS estimates:

```python
from coredeposit import NLSEstimator, MCMCEstimator

# Run NLS first
nls = NLSEstimator()
nls_result = nls.fit(data)

# Use NLS estimates as MCMC initial values
init_params = {
    'lambda': nls_result.params['lambda'],
    'gamma': nls_result.params['gamma'],
    'w1': nls_result.params['w1'],
    'h': nls_result.params['h'],
    'm': nls_result.params['m'],
}

mcmc = MCMCEstimator(init_params=init_params)
result = mcmc.fit(data)
```

### 3.6 Usage

```python
from coredeposit import MCMCEstimator, CoreDepositData

data = CoreDepositData(V_obs=..., inflow=..., V0=..., z=...)

estimator = MCMCEstimator(
    num_warmup=2000,      # Warmup iterations
    num_samples=4000,     # Posterior samples
    num_chains=2,         # Parallel chains
    likelihood='studentt', # 'normal' or 'studentt'
    ar_errors=True,       # AR(1) error model
    init_params=None,     # Optional NLS initialization
)

result = estimator.fit(data)

# Access posterior samples
lambda_samples = result.params['lambda']  # array of shape (num_chains * num_samples,)
print(f"λ: {lambda_samples.mean():.4f} ± {lambda_samples.std():.4f}")

# Credible intervals
lo, hi = np.percentile(lambda_samples, [2.5, 97.5])
print(f"95% CI: [{lo:.4f}, {hi:.4f}]")
```

---

## 4. Predictions

Both estimators support prediction with the same interface:

### 4.1 NLS Prediction

Returns point predictions:

```python
V_pred = estimator.predict(data, result)  # array of shape (T+1,)
```

### 4.2 MCMC Prediction

Returns posterior predictive distribution:

```python
# Point prediction (posterior mean)
V_pred = estimator.predict(data, result)

# With uncertainty quantification
pred = estimator.predict(data, result, uncertainty=True, ci_prob=0.95)
print(pred['mean'])    # Posterior mean
print(pred['samples']) # All posterior samples
print(pred['lower'])   # 2.5th percentile
print(pred['upper'])   # 97.5th percentile
```

---

## 5. Diagnostics

### 5.1 NLS Diagnostics

```python
result.diagnostics['success']  # Optimization converged
result.diagnostics['cost']     # Final cost value
result.diagnostics['nfev']     # Number of function evaluations
```

### 5.2 MCMC Diagnostics

```python
import arviz as az

# Convert to ArviZ InferenceData
idata = az.from_numpyro(result.diagnostics['mcmc'])

# Trace plots
az.plot_trace(idata)

# Summary statistics with R-hat and ESS
az.summary(idata)

# Posterior distributions
az.plot_posterior(idata)
```

Key convergence metrics:
- **R-hat** < 1.01: Chains have converged
- **ESS** > 400: Sufficient effective samples

---

## 6. Derived Metrics

### 6.1 Median Survival Time

The median survival time (half-life) for sticky deposits:

$$
t_{50} = \frac{(\ln 2)^{1/\gamma}}{\lambda}
$$

```python
from coredeposit import compute_median_survival

# For NLS result (returns float)
t50 = compute_median_survival(nls_result)

# For MCMC result (returns dict with uncertainty)
t50 = compute_median_survival(mcmc_result, ci_prob=0.95)
print(f"Median survival: {t50['mean']:.1f} months")
print(f"95% CI: [{t50['lower']:.1f}, {t50['upper']:.1f}]")
```

---

## 7. Method Comparison

| Aspect | NLS | MCMC |
|--------|-----|------|
| Speed | Fast | Slow |
| Uncertainty | No | Yes |
| Outlier handling | Robust losses | Student-t likelihood |
| Autocorrelation | No | AR(1) errors |
| Prior information | No | Yes |
| Convergence | Local minimum | Global (with sufficient sampling) |

**Recommended workflow**:
1. Start with NLS for quick exploration
2. Use NLS result to initialize MCMC
3. Run MCMC for final inference with uncertainty quantification
