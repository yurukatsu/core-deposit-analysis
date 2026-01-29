# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Core deposit analysis library implementing a two-type deposit model with time-dependent hazard functions. Models bank deposit dynamics by decomposing deposits into "transactional" (決済性) and "sticky" (滞留性) types, using Weibull hazard functions for survival analysis.

## Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Run examples
uv run python examples/example_run.py
uv run python examples/example_mcmc.py

# Lint
uv run ruff check src/

# Type check
uv run ty check src/
```

## Architecture

**Data Flow**: `CoreDepositData` → Estimator (`MCMCEstimator` or `NLSEstimator`) → `EstimationResult`

**Project Structure**:
```
core-deposit-analysis/
├── docs/
│   └── model.md          # Mathematical model specification
├── examples/
│   ├── example_run.py    # Basic NLS and MCMC usage
│   └── example_mcmc.py   # MCMC with ArviZ visualization
├── data/                  # Sample datasets
└── src/coredeposit/
    ├── __init__.py       # Public API exports + JAX config
    ├── types.py          # CoreDepositData, EstimationResult, type aliases
    ├── normalize.py      # Data normalization for numerical stability
    ├── model/
    │   ├── __init__.py   # Exports: V_model, weibull_hazard, S2_matrix, S2_init_vector
    │   ├── hazard.py     # weibull_hazard()
    │   ├── survival.py   # S2_matrix(), S2_init_vector() - vectorized
    │   └── balance.py    # V_model() - main balance prediction
    └── estimators/
        ├── __init__.py   # Exports: Estimator, NLSEstimator, MCMCEstimator, CoreDepositPriors
        ├── base.py       # Abstract Estimator base class
        ├── priors.py     # CoreDepositPriors dataclass and default_priors()
        ├── nls.py        # NLSEstimator (scipy least_squares)
        └── mcmc.py       # MCMCEstimator (NumPyro NUTS)
```

**Key Components**:

- [types.py](src/coredeposit/types.py) - `CoreDepositData` holds observed balances (`V_obs`), inflows (`inflow`), exogenous variables (`z`), initial balance (`V0`).

- [model/balance.py](src/coredeposit/model/balance.py) - `V_model()` computes predicted deposit balances. Weibull hazard parameters: `lam` (scale), `gam` (shape). Model parameters: `w1` (transactional proportion), `h` (first-month exit rate), `m` (initial balance average age).

- [estimators/mcmc.py](src/coredeposit/estimators/mcmc.py) - Bayesian estimation via NumPyro NUTS. Supports both covariate and no-covariate models. Uses Student-t likelihood by default for robustness.

- [estimators/nls.py](src/coredeposit/estimators/nls.py) - Non-linear least squares via scipy. Supports optional fixed `m` parameter.

**Parameter Transformations** (in NLS):
- `lam`, `gam`, `m`: log-transformed (positive)
- `w1`, `h`: logit-transformed (0-1 range)

**Type Annotations**:
- `ArrayLike = JaxArray | NDArray` - accepts both JAX and NumPy arrays
- `Scalar = JaxArray | float` - scalar values that may be JAX traced
