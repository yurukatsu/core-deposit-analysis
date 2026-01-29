# Core Deposit Analysis

A Python library for analyzing bank deposit dynamics using a two-type survival model with time-dependent hazard functions.

## Overview

This library implements a core deposit model that decomposes bank deposits into two types:

- **Transactional deposits** (決済性預金): Short-term deposits with high turnover, modeled with immediate exit behavior
- **Sticky deposits** (滞留性預金): Long-term stable deposits, modeled using Weibull hazard functions

The model supports both frequentist (NLS) and Bayesian (MCMC) estimation approaches.

For detailed mathematical specification, see [docs/model.md](docs/model.md).

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/core-deposit-analysis.git
cd core-deposit-analysis

# Install with uv (recommended)
uv sync

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
    inflow=df["input"].values,       # Deposit inflows
    V0=df["volume"].values[0],       # Initial balance
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

## Examples

See the [examples/](examples/) directory for detailed usage examples:

- [example_run.py](examples/example_run.py) - Basic NLS and MCMC estimation
- [example_mcmc.py](examples/example_mcmc.py) - MCMC with ArviZ visualization

### Running Examples

```bash
cd examples
uv run python example_run.py
uv run python example_mcmc.py
```

## Using Covariates

You can include time-varying covariates that affect the hazard rate:

```python
# Create covariate array (T+1 observations, p covariates)
z = np.column_stack([
    df["interest_rate"].values,
    df["gdp_growth"].values,
])

data = CoreDepositData(
    V_obs=df["volume"].values,
    inflow=df["input"].values,
    V0=df["volume"].values[0],
    z=z,  # Add covariates
)

# Fit with covariates
result = mcmc.fit(data)
print(result.params["beta"])  # Covariate coefficients
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

- Python >= 3.11
- JAX
- NumPyro
- NumPy
- SciPy
- Pandas (for examples)
- ArviZ (optional, for MCMC diagnostics)

## License

MIT License
