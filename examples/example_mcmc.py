"""MCMC estimation example with ArviZ visualization.

This script demonstrates:
1. Running MCMC estimation with the core deposit model
2. Visualizing results with ArviZ (trace plots, posterior distributions, etc.)
3. Computing summary statistics and credible intervals
"""

from pathlib import Path

import numpy as np
import numpyro
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

from coredeposit import CoreDepositData, MCMCEstimator

# Create outputs directory
Path("outputs").mkdir(exist_ok=True)

# Use multiple CPU cores for parallel chains
numpyro.set_host_device_count(2)

# -------------------------------------------------
# Load sample data
# -------------------------------------------------
df = pd.read_csv("../data/boj.csv", parse_dates=["date"])
df = df[(df["date"] >= "2000-01-01") & (df["date"] <= "2010-12-31")].reset_index(drop=True)

V = df["volume"].values
inflow = df["input"].values

data = CoreDepositData(
    V_obs=V,
    inflow=inflow,
    V0=V[0],
)

# -------------------------------------------------
# Run MCMC
# -------------------------------------------------
print("Running MCMC estimation...")

estimator = MCMCEstimator(
    num_warmup=2000,
    num_samples=4000,
    num_chains=2,
    likelihood="studentt",
)

result = estimator.fit(data)

# -------------------------------------------------
# Convert to ArviZ InferenceData
# -------------------------------------------------
idata = az.from_numpyro(result.diagnostics["mcmc"])

# -------------------------------------------------
# Summary statistics
# -------------------------------------------------
print("\n" + "=" * 50)
print("Posterior Summary")
print("=" * 50)

summary = az.summary(
    idata,
    var_names=["lambda", "gamma", "w1", "h", "m", "sigma"],
    hdi_prob=0.95,
)
print(summary)

# -------------------------------------------------
# Convergence diagnostics
# -------------------------------------------------
print("\n" + "=" * 50)
print("Convergence Diagnostics")
print("=" * 50)

# R-hat should be close to 1.0 (< 1.01 is good)
rhat = az.rhat(idata)
print("\nR-hat values:")
for var in ["lambda", "gamma", "w1", "h", "m"]:
    print(f"  {var}: {float(rhat[var].values):.4f}")

# Effective sample size
ess = az.ess(idata)
print("\nEffective sample size:")
for var in ["lambda", "gamma", "w1", "h", "m"]:
    print(f"  {var}: {float(ess[var].values):.0f}")

# -------------------------------------------------
# Visualization
# -------------------------------------------------
print("\nGenerating plots...")

# 1. Trace plot (convergence check)
fig, axes = plt.subplots(5, 2, figsize=(12, 10))
az.plot_trace(
    idata,
    var_names=["lambda", "gamma", "w1", "h", "m"],
    axes=axes,
)
plt.tight_layout()
plt.savefig("outputs/mcmc_trace.png", dpi=150)
print("  Saved: outputs/mcmc_trace.png")

# 2. Posterior distributions
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
az.plot_posterior(
    idata,
    var_names=["lambda", "gamma", "w1", "h", "m", "sigma"],
    hdi_prob=0.95,
    ax=axes.flatten(),
)
plt.tight_layout()
plt.savefig("outputs/mcmc_posterior.png", dpi=150)
print("  Saved: outputs/mcmc_posterior.png")

# 3. Pair plot (correlation between parameters)
fig = plt.figure(figsize=(10, 10))
az.plot_pair(
    idata,
    var_names=["lambda", "gamma", "w1", "h"],
    kind="kde",
    marginals=True,
)
plt.tight_layout()
plt.savefig("outputs/mcmc_pair.png", dpi=150)
print("  Saved: outputs/mcmc_pair.png")

# 4. Forest plot (comparing parameters)
fig, ax = plt.subplots(figsize=(8, 4))
az.plot_forest(
    idata,
    var_names=["w1", "h"],
    combined=True,
    hdi_prob=0.95,
    ax=ax,
)
plt.tight_layout()
plt.savefig("outputs/mcmc_forest.png", dpi=150)
print("  Saved: outputs/mcmc_forest.png")

plt.close("all")

# -------------------------------------------------
# Credible intervals
# -------------------------------------------------
print("\n" + "=" * 50)
print("95% Credible Intervals")
print("=" * 50)

samples = result.params
for var in ["lambda", "gamma", "w1", "h", "m"]:
    vals = np.array(samples[var])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    mean = vals.mean()
    print(f"  {var}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

print("\nDone!")
