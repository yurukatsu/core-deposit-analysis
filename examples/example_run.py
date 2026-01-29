import numpy as np
import numpyro
import pandas as pd

from coredeposit import (
    CoreDepositData,
    NLSEstimator,
    MCMCEstimator,
)

numpyro.set_host_device_count(2)

# -------------------------------------------------
# Load sample data
# -------------------------------------------------
df = pd.read_csv("../data/dummy_small.csv")

V = df["volume"].values
inflow = df["input"].values

data_nocov = CoreDepositData(
    V_obs=V,
    inflow=inflow,
    V0=V[0],
)

print("====================================")
print("NLS (full estimation)")
print("====================================")

nls = NLSEstimator()
res_nls = nls.fit(data_nocov)
print(res_nls.params)


print("\n====================================")
print("NLS (grid search for m)")
print("====================================")

grid = np.linspace(6, 60, 10)
best_cost = np.inf
best_res = None

for m in grid:
    res = nls.fit(data_nocov, m_fixed=m)
    cost = res.diagnostics["cost"]
    print(f"m={m:.1f}, cost={cost:.4f}")

    if cost < best_cost:
        best_cost = cost
        best_res = res

print("Best grid result:", best_res.params)


print("\n====================================")
print("MCMC (no covariate)")
print("====================================")

mcmc_nocov = MCMCEstimator(
    num_warmup=1000,
    num_samples=1000,
    num_chains=2,
)
res_mcmc_nocov = mcmc_nocov.fit(data_nocov)


print("\n====================================")
print("MCMC (with fake covariate)")
print("====================================")

# ダミーの z を作る（時間トレンド）
z = np.arange(len(V)).reshape(-1, 1)

data_cov = CoreDepositData(
    V_obs=V,
    inflow=inflow,
    z=z,
    V0=V[0],
)

mcmc_cov = MCMCEstimator(
    num_warmup=1000,
    num_samples=1000,
    num_chains=2,
)
res_mcmc_cov = mcmc_cov.fit(data_cov)
