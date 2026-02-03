# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Core deposit analysis library implementing a two-type deposit model with time-dependent hazard functions. Models bank deposit dynamics by decomposing deposits into "transactional" (決済性) and "sticky" (滞留性) types, using Weibull hazard functions for survival analysis.

## Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Install with CUDA support
uv sync --extra cuda

# Run examples
uv run python examples/run_nls.py
uv run python examples/run_mcmc.py

# Run tests
uv run pytest tests/ -v

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
│   ├── model.md          # Mathematical model specification
│   └── estimation.md     # Estimation methods (NLS, MCMC)
├── examples/
│   ├── run_nls.py        # NLS and MAP estimation example
│   ├── run_mcmc.py       # MCMC estimation with ArviZ visualization
│   └── data/             # Sample datasets (boj.csv, covariate.csv)
├── tests/                # Comprehensive test suite (150+ tests)
│   ├── test_w1.py        # Time-varying w1 tests
│   ├── test_nls.py       # NLS estimator tests
│   ├── test_mcmc.py      # MCMC estimator tests
│   ├── test_balance.py   # V_model tests
│   ├── test_survival.py  # S2_matrix, S2_init_vector tests
│   ├── test_s1.py        # S1 survival function tests
│   ├── test_hazard.py    # weibull_hazard tests
│   ├── test_metrics.py   # Derived metrics tests
│   ├── test_priors.py    # Prior distribution tests
│   ├── test_normalize.py # Normalization tests
│   └── test_types.py     # Type definition tests
└── src/coredeposit/
    ├── __init__.py       # Public API exports + JAX config
    ├── types.py          # CoreDepositData, EstimationResult, type aliases
    ├── normalize.py      # Data normalization for numerical stability
    ├── metrics.py        # Derived metrics (compute_median_survival)
    ├── model/
    │   ├── __init__.py   # Exports: V_model, weibull_hazard, S2_*, S1_*, w1_*
    │   ├── hazard.py     # weibull_hazard()
    │   ├── survival.py   # S2_matrix(), S2_init_vector() - vectorized
    │   ├── s1.py         # S1 survival functions (immediate_exit, geometric)
    │   ├── w1.py         # Time-varying w1 functions (w1_logistic, feature helpers)
    │   └── balance.py    # V_model() - main balance prediction
    └── estimators/
        ├── __init__.py   # Exports: Estimator, NLSEstimator, MCMCEstimator, priors
        ├── base.py       # Abstract Estimator base class
        ├── priors.py     # CoreDepositPriors dataclass for MCMC (NumPyro)
        ├── map_priors.py # MAPPriors dataclass for MAP (scipy.stats)
        ├── nls.py        # NLSEstimator (NLS and MAP via scipy)
        └── mcmc.py       # MCMCEstimator (NumPyro NUTS)
```

**Key Components**:

- [types.py](src/coredeposit/types.py) - `CoreDepositData` holds observed balances (`V_obs`), inflows (`inflow`), exogenous variables (`z`), time-varying w1 features (`w1_features`), initial balance (`V0`).

- [model/balance.py](src/coredeposit/model/balance.py) - `V_model()` computes predicted deposit balances. Weibull hazard parameters: `lam` (scale), `gam` (shape). Model parameters: `w1` (transactional proportion, scalar or array), `h` (first-month exit rate), `m` (initial balance average age). Accepts `s1_term_fn` for custom S1 models.

- [model/w1.py](src/coredeposit/model/w1.py) - Time-varying w1 functions. `w1_logistic(a, b, x)` for logistic regression model. Feature helpers: `compute_w1_features_ma_deviation()`, `compute_w1_features_seasonal()`.

- [model/s1.py](src/coredeposit/model/s1.py) - S1 (transactional deposit) survival functions. Default: immediate exit. Alternative: geometric. Customizable via `s1_term_fn` parameter.

- [estimators/mcmc.py](src/coredeposit/estimators/mcmc.py) - Bayesian estimation via NumPyro NUTS. Supports covariates (`z`), time-varying w1 (`w1_features`). Uses Student-t likelihood by default for robustness. Key options:
  - `init_params`: Initialize from NLS estimates for better convergence
  - `ar_errors`: Model observation errors with AR(1) autocorrelation

- [estimators/nls.py](src/coredeposit/estimators/nls.py) - Non-linear least squares and MAP estimation via scipy. Supports optional fixed `m` parameter, `priors` for MAP, and `w1_features` for time-varying w1.

- [estimators/map_priors.py](src/coredeposit/estimators/map_priors.py) - `MAPPriors` dataclass for MAP estimation using scipy.stats distributions.

- [estimators/priors.py](src/coredeposit/estimators/priors.py) - `CoreDepositPriors` dataclass with default weakly informative priors. Includes `rho_prior` for AR(1) coefficient.

- [metrics.py](src/coredeposit/metrics.py) - `compute_median_survival()` calculates half-life for sticky deposits with uncertainty quantification for MCMC results.

**Parameter Transformations** (in NLS):
- `lam`, `gam`, `m`: log-transformed (positive)
- `w1`, `h`: logit-transformed (0-1 range)

**Type Annotations**:
- `ArrayLike = JaxArray | NDArray` - accepts both JAX and NumPy arrays
- `Scalar = JaxArray | float` - scalar values that may be JAX traced

**Device Configuration** (for MCMC):
- Call `numpyro.set_host_device_count(n)` before any JAX operations to enable parallel chains on CPU
- Use `uv sync --extra cuda` for GPU support with CUDA 12
