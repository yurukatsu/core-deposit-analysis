# Core Deposit Analysis

A Python library for analyzing bank deposit dynamics using a two-type survival model with time-dependent hazard functions.

## Overview

This library implements a core deposit model that decomposes bank deposits into two types:

- **Transactional deposits** (決済性預金): Short-term deposits with high turnover, modeled with immediate exit behavior
- **Sticky deposits** (滞留性預金): Long-term stable deposits, modeled using Weibull hazard functions

The model supports both frequentist (NLS) and Bayesian (MCMC) estimation approaches.

For detailed documentation:
- [docs/model.md](docs/model.md) - Mathematical model specification
- [docs/estimation.md](docs/estimation.md) - Estimation methods (NLS and MCMC)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/core-deposit-analysis.git
cd core-deposit-analysis

# Install with uv (recommended)
uv sync

# For GPU support (CUDA 12)
uv sync --extra cuda

# Or install with pip
pip install -e .
```

## Quick Start

```python
import numpy as np
import pandas as pd
from coredeposit import CoreDepositData, MCMCEstimator, NLSEstimator

# Load your data
df = pd.read_csv("data/your_data.csv")

# Prepare data
data = CoreDepositData(
    V_obs=df["volume"].values,      # Observed deposit balances
    inflow=df["inflow"].values,     # Deposit inflows
    V0=df["volume"].values[0],      # Initial balance
)

# Option 1: Non-linear Least Squares
nls = NLSEstimator()
result_nls = nls.fit(data)
print(result_nls.params)

# Option 2: Bayesian MCMC
mcmc = MCMCEstimator(
    num_warmup=2000,
    num_samples=4000,
    num_chains=2,
)
result_mcmc = mcmc.fit(data)
```

## Parameters

The model estimates the following parameters:

| Parameter | Description |
|-----------|-------------|
| `lambda` | Weibull scale parameter |
| `gamma` | Weibull shape parameter |
| `w1` | Proportion of transactional deposits (0-1) |
| `h` | First-month exit rate for transactional deposits (0-1) |
| `m` | Average age of initial balance in months |
| `beta` | Covariate coefficients (if covariates provided) |
| `w1_a` | w1 logistic intercept (if w1_features provided) |
| `w1_b` | w1 logistic coefficients (if w1_features provided) |
| `sigma` | Observation noise std (MCMC only) |
| `nu` | Student-t degrees of freedom (MCMC with studentt likelihood) |
| `rho` | AR(1) autocorrelation coefficient (MCMC with ar_errors=True) |

## Examples

See [examples/run_mcmc.py](examples/run_mcmc.py) for a complete example with:
- Device configuration (CPU/GPU)
- NLS initialization for MCMC
- AR(1) autocorrelated errors
- ArviZ visualization and diagnostics

### Running Examples

```bash
uv run python examples/run_mcmc.py
```

## Advanced Features

### Device Configuration

Configure JAX/NumPyro device settings for parallel MCMC chains:

```python
import numpyro

# For CPU with multiple devices (must be called before any JAX operations)
numpyro.set_host_device_count(2)

# Run MCMC with parallel chains
mcmc = MCMCEstimator(num_chains=2)
result = mcmc.fit(data)
```

### NLS Initialization for MCMC

Improve MCMC convergence by initializing from NLS estimates:

```python
from coredeposit import NLSEstimator, MCMCEstimator

# Run NLS first
nls = NLSEstimator()
nls_result = nls.fit(data)

# Use NLS estimates as MCMC initial values
init_params = {
    "lambda": nls_result.params["lambda"],
    "gamma": nls_result.params["gamma"],
    "w1": nls_result.params["w1"],
    "h": nls_result.params["h"],
    "m": nls_result.params["m"],
}

mcmc = MCMCEstimator(init_params=init_params)
result = mcmc.fit(data)
```

### AR(1) Autocorrelated Errors

Model temporal correlation in prediction errors:

```python
mcmc = MCMCEstimator(
    ar_errors=True,  # Enable AR(1) error model
    likelihood="studentt",
)
result = mcmc.fit(data)

# Access AR(1) coefficient
rho_samples = result.params["rho"]
print(f"rho: {rho_samples.mean():.3f}")
```

### Using Covariates

Include time-varying covariates that affect the hazard rate:

```python
# Create covariate array (T+1 observations, p covariates)
z = np.column_stack([
    df["interest_rate"].values,
    df["gdp_growth"].values,
])

data = CoreDepositData(
    V_obs=df["volume"].values,
    inflow=df["inflow"].values,
    V0=df["volume"].values[0],
    z=z,  # Add covariates
)

# Fit with covariates
result = mcmc.fit(data)
print(result.params["beta"])  # Covariate coefficients
```

### Time-Varying Transactional Proportion (w1)

Model the transactional proportion as a function of features:

```python
from coredeposit.model.w1 import compute_w1_features_ma_deviation

# Compute features (e.g., deviation from moving average)
w1_features = compute_w1_features_ma_deviation(df["inflow"].values, window=12)

data = CoreDepositData(
    V_obs=df["volume"].values,
    inflow=df["inflow"].values,
    V0=df["volume"].values[0],
    w1_features=w1_features.reshape(-1, 1),  # Shape: (T+1, q)
)

# Fit with time-varying w1
result = mcmc.fit(data)
print(result.params["w1_a"])  # Logistic intercept
print(result.params["w1_b"])  # Feature coefficients
print(result.params["w1"])    # Mean w1 (NLS only)
```

When `w1_features` is provided, w1 is modeled as:
$$w_1(t) = \sigma(a + b^\top x(t))$$

where $\sigma$ is the sigmoid function. Available feature helpers:
- `compute_w1_features_ma_deviation(inflow, window)`: Log deviation from moving average
- `compute_w1_features_seasonal(T, start_month)`: Monthly dummy variables

### Derived Metrics

Compute median survival time (half-life) for sticky deposits:

```python
from coredeposit import compute_median_survival

# For NLS result (returns float)
t50 = compute_median_survival(nls_result)

# For MCMC result (returns dict with uncertainty)
t50 = compute_median_survival(mcmc_result, ci_prob=0.95)
print(f"Median survival: {t50['mean']:.1f} months")
print(f"95% CI: [{t50['lower']:.1f}, {t50['upper']:.1f}]")
```

## ArviZ Integration (MCMC)

The MCMC estimator stores the NumPyro MCMC object for ArviZ visualization:

```python
import arviz as az

result = mcmc.fit(data)
idata = az.from_numpyro(result.diagnostics["mcmc"])

# Summary statistics
print(az.summary(idata, var_names=["lambda", "gamma", "w1", "h", "m"]))

# Trace plots
az.plot_trace(idata)

# Posterior distributions
az.plot_posterior(idata, hdi_prob=0.95)
```

## Requirements

- Python >= 3.13
- JAX
- NumPyro
- NumPy
- SciPy
- Pandas (for examples)
- ArviZ (optional, for MCMC diagnostics)

For GPU support, install with `--extra cuda` which adds `jax[cuda12]`.

## License

MIT License
