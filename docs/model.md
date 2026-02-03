# Core Deposit Model: Mathematical Specification

This document provides a formal mathematical description of the two-type core deposit model with time-dependent hazard functions.

## 1. Notation and Time Convention

### 1.1 Time Index

Calendar time is indexed by integers representing month-ends:

$$
t = 0, 1, 2, \dots
$$

### 1.2 Inflow Definition

Let $I(i)$ denote the total deposit inflow received during the interval $(i-1, i]$. By convention, this inflow is considered to have **survival duration of 1 month** at month-end $i$.

### 1.3 Key Variables

| Symbol | Description |
|--------|-------------|
| $V(t)$ | Deposit balance at month-end $t$ |
| $I(i)$ | Deposit inflow during period $(i-1, i]$ |
| $S(k)$ | Survival function: proportion remaining after $k$ months |
| $m$ | Average age (in months) of initial balance $V(0)$ |

---

## 2. Balance Equation

The deposit balance at time $t$ is approximated by:

$$
V(t) \approx \sum_{i=1}^{t} I(i) \, S(t - i + 1) + V(0) \, S(t + m)
$$

where:
- The first term aggregates contributions from each historical inflow weighted by its survival
- The second term accounts for the surviving portion of the initial balance

---

## 3. Two-Type Mixture Model

### 3.1 Model Structure

Deposits are assumed to consist of two distinct types, each with its own survival function:

| Type | Name | Survival Function | Interpretation |
|------|------|-------------------|----------------|
| 1 | Transactional (決済性) | $S_1$ | Short-term, high-turnover deposits |
| 2 | Sticky (滞留性) | $S_2$ | Long-term, stable deposits |

The balance equation generalizes to:

$$
V(t) = \sum_{j=1}^{2} \sum_{i=1}^{t} w_j(i) \, I(i) \, S_j(t - i + 1) + \sum_{j=1}^{2} w_j(0) \, V(0) \, S_j(t + m)
$$

where $w_j(i)$ denotes the proportion of inflow at time $i$ belonging to type $j$.

### 3.2 Simplifying Assumptions

**Assumption 1**: The initial balance consists entirely of sticky deposits:
$$
w_1(0) = 0, \quad w_2(0) = 1
$$

**Assumption 2** (Constant w1): Type proportions are constant over time:
$$
w_1(i) = w_1, \quad w_2(i) = 1 - w_1 \quad \forall i \geq 1
$$

### 3.3 Time-Varying w1 (Extension)

The constant w1 assumption can be relaxed to allow time-varying transactional proportions. This is useful when inflow composition varies seasonally or with market conditions.

#### Logistic Specification

The transactional proportion at time $i$ is modeled via logistic regression:

$$
w_1(i) = \sigma(a + b^\top x(i))
$$

where:
- $\sigma(z) = 1 / (1 + e^{-z})$ is the sigmoid function
- $a \in \mathbb{R}$ is the intercept (baseline w1 on logit scale)
- $b \in \mathbb{R}^q$ are coefficients for $q$ features
- $x(i) \in \mathbb{R}^q$ are features at time $i$

#### Common Features

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| MA deviation | $\log(I(i) / \text{MA}(I))$ | Positive when inflow exceeds trend |
| Seasonal dummies | Binary indicators for months 1-11 | Monthly seasonality |

#### Parameter Interpretation

- When $a = 0$ and $b = 0$: $w_1(i) = 0.5$ (constant)
- $a < 0$: baseline w1 < 0.5 (more sticky deposits)
- $b_j > 0$: positive feature increases transactional proportion

#### Implementation

```python
from coredeposit import CoreDepositData
from coredeposit.model.w1 import compute_w1_features_ma_deviation

# Compute features
w1_features = compute_w1_features_ma_deviation(inflow, window=12)

# Include in data
data = CoreDepositData(
    V_obs=V_obs, inflow=inflow, V0=V0,
    w1_features=w1_features.reshape(-1, 1)
)
```

When `w1_features` is provided, estimators return:
- `w1_a`: intercept parameter
- `w1_b`: coefficient vector (shape `(q,)` for NLS, `(n_samples, q)` for MCMC)
- `w1`: mean w1 value (NLS only, for convenience)

---

## 4. Hazard Functions

### 4.1 Transactional Deposits

Transactional deposits can be modeled with different survival functions. The default is the **immediate exit** model, but alternatives are available.

#### 4.1.1 Immediate Exit Model (Default)

The discrete-time hazard function is:

$$
h_1(s \mid h) =
\begin{cases}
h & s = 1 \\
1 & s \geq 2
\end{cases}
$$

where $h \in [0, 1]$ is the first-month exit rate.

The corresponding survival function is:

$$
S_1(s \mid h) =
\begin{cases}
1 & s = 0 \\
1 - h & s = 1 \\
0 & s \geq 2
\end{cases}
$$

This model assumes transactional deposits survive at most one period.

#### 4.1.2 Geometric Model (Alternative)

A constant hazard (geometric) model where deposits exit at rate $h$ each period:

$$
h_1(s \mid h) = h \quad \forall s \geq 1
$$

The corresponding survival function is:

$$
S_1(s \mid h) = (1 - h)^s
$$

This model allows transactional deposits to survive multiple periods with exponentially decaying probability.

#### 4.1.3 Implementation

The S1 model is customizable via the `s1_term_fn` parameter in `V_model`:

```python
from coredeposit.model import V_model
from coredeposit.model.s1 import S1_term_geometric

# Use geometric S1 model
V = V_model(..., s1_term_fn=S1_term_geometric)
```

Available S1 functions in `coredeposit.model.s1`:
- `S1_term_immediate_exit` (default)
- `S1_term_geometric`
- `S1_immediate_exit`, `S1_geometric` (survival functions)
- `S1_matrix_immediate_exit`, `S1_matrix_geometric` (survival matrices)

### 4.2 Sticky Deposits

Sticky deposits follow a continuous hazard model with optional time-varying covariates.

#### Continuous-Time Derivation

The survival function $S_2(t \mid i, z)$ for a deposit received at time $i$ satisfies the following ordinary differential equation:

$$
\frac{d}{dt} S_2(t \mid i, z) = -h(t \mid i, z) \, S_2(t \mid i, z), \quad S_2(i-1 \mid i, z) = 1
$$

where $h(t \mid i, z)$ is the instantaneous hazard rate at calendar time $t$.

#### Proportional Hazards with Time-Varying Covariates

We decompose the hazard into two components:

1. **Baseline hazard** $h_0(s \mid \theta)$: depends on survival duration $s = t - (i-1)$
2. **Covariate effect** $\exp(\beta^\top z(t))$: depends on calendar time $t$

This gives the proportional hazards specification:

$$
h(t \mid i, z) = h_0(t - (i-1) \mid \theta) \exp(\beta^\top z(t))
$$

#### Survival Function

Solving the ODE yields:

$$
S_2(t \mid i, z) = \exp\left( -\int_{i-1}^{t} h(u \mid i, z) \, du \right)
$$

Substituting the proportional hazards form:

$$
S_2(t \mid i, z) = \exp\left( -\int_{i-1}^{t} h_0(u - (i-1) \mid \theta) \exp(\beta^\top z(u)) \, du \right)
$$

In the absence of covariates ($z \equiv 0$ or $\beta = 0$), substituting $s = u - (i-1)$:

$$
S_2(t \mid i) = \exp\left( -\int_{0}^{t-i+1} h_0(s \mid \theta) \, ds \right)
$$

---

## 5. Weibull Specification

The baseline hazard follows a Weibull distribution:

$$
h_0(s \mid \lambda, \gamma) = \lambda \gamma (\lambda s)^{\gamma - 1}
$$

where:
- $\lambda > 0$: scale parameter
- $\gamma > 0$: shape parameter

The cumulative hazard is:

$$
H_0(s \mid \lambda, \gamma) = (\lambda s)^\gamma
$$

And the survival function (without covariates) becomes:

$$
S_2(s \mid \lambda, \gamma) = \exp\left( -(\lambda s)^\gamma \right)
$$

**Shape parameter interpretation**:
- $\gamma < 1$: Decreasing hazard (early exits more likely)
- $\gamma = 1$: Constant hazard (exponential distribution)
- $\gamma > 1$: Increasing hazard (later exits more likely)

---

## 6. Complete Balance Model

Combining all components, the final balance model is:

$$
\begin{aligned}
V(t \mid \Theta) = \;& w_1 \, I(t) \, (1 - h) \\
& + (1 - w_1) \sum_{i=1}^{t} I(i) \, S_2(t - i + 1 \mid \lambda, \gamma, \beta, z) \\
& + V(0) \, S_2(t + m \mid \lambda, \gamma, \beta, z)
\end{aligned}
$$

where the parameter vector is:

$$
\Theta = (\lambda, \gamma, w_1, h, m, \beta)
$$

---

## 7. Parameter Summary

| Parameter | Domain | Description |
|-----------|--------|-------------|
| $\lambda$ | $(0, \infty)$ | Weibull scale parameter |
| $\gamma$ | $(0, \infty)$ | Weibull shape parameter |
| $w_1$ | $[0, 1]$ | Proportion of transactional deposits in inflow (constant model) |
| $a$ | $\mathbb{R}$ | w1 logistic intercept (time-varying model) |
| $b$ | $\mathbb{R}^q$ | w1 logistic coefficients (time-varying model) |
| $h$ | $[0, 1]$ | First-month exit rate for transactional deposits |
| $m$ | $(0, \infty)$ | Average age of initial balance (months) |
| $\beta$ | $\mathbb{R}^p$ | S2 hazard covariate coefficients (optional) |
| $\sigma$ | $(0, \infty)$ | Observation noise (MCMC only) |
| $\nu$ | $(0, \infty)$ | Student-t degrees of freedom (MCMC only) |
| $\rho$ | $(-1, 1)$ | AR(1) autocorrelation coefficient (MCMC with ar_errors) |

---
