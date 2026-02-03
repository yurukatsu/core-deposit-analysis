"""MCMC estimation example with ArviZ visualization.

This script demonstrates:
1. Running MCMC estimation with the core deposit model
2. Predicting deposit balances with uncertainty quantification
3. Visualizing results with ArviZ (trace plots, posterior distributions, etc.)
4. Computing summary statistics and credible intervals
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import arviz as az
import jax
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pandas as pd
from numpy.typing import NDArray

from coredeposit import (
    CoreDepositData,
    MCMCEstimator,
    NLSEstimator,
    EstimationResult,
    compute_median_survival,
)


# ---------------------------------------------------------------------------
# Device Configuration
# ---------------------------------------------------------------------------


@dataclass
class DeviceConfig:
    """Configuration for JAX/NumPyro device settings.

    Attributes:
        platform: Compute platform ("cpu", "gpu", or "auto").
            "auto" will use GPU if available, otherwise CPU.
        num_devices: Number of devices to use for parallel chains.
            If None, uses all available devices.
    """

    platform: Literal["cpu", "gpu", "auto"] = "cpu"
    num_devices: int | None = None

    def setup(self) -> str:
        """Configure JAX/NumPyro devices.

        Returns:
            The actual platform being used.
        """
        # Set CPU device count BEFORE any JAX operations
        # This must be done before JAX is initialized
        if self.platform in ("cpu", "auto"):
            num_devices = self.num_devices or 2
            numpyro.set_host_device_count(num_devices)

        # Determine platform
        if self.platform == "auto":
            available = jax.devices()
            actual_platform = available[0].platform if available else "cpu"
        else:
            actual_platform = self.platform

        # Set device count for display
        if actual_platform == "cpu":
            num_devices = self.num_devices or 2
        else:
            num_devices = self.num_devices or len(jax.devices())

        print(f"  Platform: {actual_platform}")
        print(f"  Devices: {num_devices}")

        return actual_platform


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_deposit_data(
    boj_path: str | Path,
    covariate_path: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, NDArray | None]:
    """Load deposit data and covariates.

    Args:
        boj_path: Path to BOJ deposit data CSV (date, volume, inflow).
        covariate_path: Path to covariate data CSV.
        start_date: Start date for filtering (inclusive).
        end_date: End date for filtering (inclusive).

    Returns:
        Tuple of (deposit DataFrame, covariate array or None).
    """
    df_boj = pd.read_csv(boj_path, parse_dates=["date"])
    df_cov = pd.read_csv(covariate_path, parse_dates=["date"])

    # Align covariate dates to month start (BOJ data uses month start)
    df_cov["date"] = df_cov["date"].dt.to_period("M").dt.to_timestamp()

    # Merge on date
    df = pd.merge(df_boj, df_cov[["date", "VJY0001M Index"]], on="date", how="inner")

    # Filter by date range
    if start_date:
        df = df[df["date"] >= start_date]
    if end_date:
        df = df[df["date"] <= end_date]

    df = df.reset_index(drop=True)

    z = df["VJY0001M Index"].values
    return df, z


def create_core_deposit_data(df: pd.DataFrame, z: NDArray | None = None) -> CoreDepositData:
    """Create CoreDepositData from DataFrame.

    Args:
        df: DataFrame with 'volume' and 'inflow' columns.
        z: Optional covariate array.

    Returns:
        CoreDepositData instance.
    """
    return CoreDepositData(
        V_obs=df["volume"].values,
        inflow=df["inflow"].values,
        V0=df["volume"].values[0],
        z=z,
    )


# ---------------------------------------------------------------------------
# MCMC Estimation
# ---------------------------------------------------------------------------


@dataclass
class MCMCConfig:
    """Configuration for MCMC estimation."""

    num_warmup: int = 2000
    num_samples: int = 4000
    num_chains: int = 2
    likelihood: str = "studentt"
    device: DeviceConfig = field(default_factory=DeviceConfig)
    use_nls_init: bool = False  # Use NLS estimates as initial values
    ar_errors: bool = False  # Use AR(1) autocorrelated errors


def run_nls(data: CoreDepositData) -> EstimationResult:
    """Run NLS estimation for initial values.

    Args:
        data: Core deposit data.

    Returns:
        Estimation result with point estimates.
    """
    estimator = NLSEstimator(loss="soft_l1")
    return estimator.fit(data)


def run_mcmc(
    data: CoreDepositData,
    config: MCMCConfig,
    init_params: dict[str, float] | None = None,
) -> EstimationResult:
    """Run MCMC estimation.

    Args:
        data: Core deposit data.
        config: MCMC configuration.
        init_params: Optional initial parameter values.

    Returns:
        Estimation result with posterior samples.
    """
    estimator = MCMCEstimator(
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
        likelihood=config.likelihood,
        init_params=init_params,
        ar_errors=config.ar_errors,
    )
    return estimator.fit(data)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_trace(idata: az.InferenceData, output_path: Path) -> None:
    """Plot MCMC trace plots for convergence check."""
    var_names = ["lambda", "gamma", "w1", "h", "m"]
    fig, axes = plt.subplots(len(var_names), 2, figsize=(12, 10))
    az.plot_trace(idata, var_names=var_names, axes=axes)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_posterior(idata: az.InferenceData, output_path: Path) -> None:
    """Plot posterior distributions."""
    var_names = ["lambda", "gamma", "w1", "h", "m", "sigma"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    az.plot_posterior(idata, var_names=var_names, hdi_prob=0.95, ax=axes.flatten())
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_pair(idata: az.InferenceData, output_path: Path) -> None:
    """Plot pair plot showing parameter correlations."""
    var_names = ["lambda", "gamma", "w1", "h"]
    plt.figure(figsize=(10, 10))
    az.plot_pair(idata, var_names=var_names, kind="kde", marginals=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_forest(idata: az.InferenceData, output_path: Path) -> None:
    """Plot forest plot comparing parameters."""
    fig, ax = plt.subplots(figsize=(8, 4))
    az.plot_forest(idata, var_names=["w1", "h"], combined=True, hdi_prob=0.95, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_prediction(
    V_obs: NDArray,
    estimator: MCMCEstimator,
    data: CoreDepositData,
    result: EstimationResult,
    output_path: Path,
) -> None:
    """Plot prediction vs observed with credible intervals."""
    pred_95 = estimator.predict(data, result, uncertainty=True, ci_prob=0.95)
    pred_90 = estimator.predict(data, result, uncertainty=True, ci_prob=0.90)
    pred_50 = estimator.predict(data, result, uncertainty=True, ci_prob=0.50)

    fig, ax = plt.subplots(figsize=(12, 5))
    t = np.arange(len(V_obs))

    ax.fill_between(t, pred_95["lower"], pred_95["upper"], alpha=0.2, label="95% CI")
    ax.fill_between(t, pred_90["lower"], pred_90["upper"], alpha=0.3, label="90% CI")
    ax.fill_between(t, pred_50["lower"], pred_50["upper"], alpha=0.4, label="50% CI")
    ax.plot(t, pred_95["mean"], label="Predicted (posterior mean)", linewidth=2)
    ax.plot(t, V_obs, "o", markersize=3, alpha=0.7, label="Observed")

    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Deposit Balance")
    ax.set_title("Model Fit with Credible Intervals")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Summary Statistics
# ---------------------------------------------------------------------------


def print_summary(idata: az.InferenceData) -> None:
    """Print posterior summary statistics."""
    print("\n" + "=" * 50)
    print("Posterior Summary")
    print("=" * 50)

    summary = az.summary(
        idata,
        var_names=["lambda", "gamma", "w1", "h", "m", "sigma"],
        hdi_prob=0.95,
    )
    print(summary)


def print_convergence_diagnostics(idata: az.InferenceData) -> None:
    """Print convergence diagnostics (R-hat and ESS)."""
    var_names = ["lambda", "gamma", "w1", "h", "m"]

    print("\n" + "=" * 50)
    print("Convergence Diagnostics")
    print("=" * 50)

    rhat = az.rhat(idata)
    print("\nR-hat values (should be < 1.01):")
    for var in var_names:
        print(f"  {var}: {float(rhat[var].values):.4f}")

    ess = az.ess(idata)
    print("\nEffective sample size:")
    for var in var_names:
        print(f"  {var}: {float(ess[var].values):.0f}")


def print_credible_intervals(result: EstimationResult, ar_errors: bool = False) -> None:
    """Print 95% credible intervals for all parameters."""
    print("\n" + "=" * 50)
    print("95% Credible Intervals")
    print("=" * 50)

    var_names = ["lambda", "gamma", "w1", "h", "m"]
    if ar_errors:
        var_names.append("rho")

    for var in var_names:
        if var in result.params:
            vals = np.array(result.params[var])
            lo, hi = np.percentile(vals, [2.5, 97.5])
            mean = vals.mean()
            print(f"  {var}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")


def print_median_survival(result: EstimationResult) -> None:
    """Print median survival time (half-life) for sticky deposits."""
    print("\n" + "=" * 50)
    print("Median Survival Time (Sticky Deposits)")
    print("=" * 50)

    t50 = compute_median_survival(result, ci_prob=0.95)
    print(f"  Mean:   {t50['mean']:.1f} months")
    print(f"  Median: {t50['median']:.1f} months")
    print(f"  95% CI: [{t50['lower']:.1f}, {t50['upper']:.1f}] months")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run MCMC estimation example."""
    # Configuration
    config = MCMCConfig(
        num_warmup=3000,
        num_samples=6000,
        num_chains=2,
        likelihood="studentt",
        device=DeviceConfig(
            platform="auto",  # "cpu", "gpu", or "auto"
            num_devices=2,   # Number of devices for parallel chains
        ),
        use_nls_init=False,  # Use NLS estimates as MCMC initial values
        ar_errors=False,  # Use AR(1) autocorrelated errors
    )

    # Setup device
    print("Setting up device...")
    config.device.setup()

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    data_dir = Path(__file__).parent / "data"

    # Load data
    print("\nLoading data...")
    df, z = load_deposit_data(
        boj_path=data_dir / "boj.csv",
        covariate_path=data_dir / "covariate.csv",
        start_date="2000-01-01",
        end_date="2010-12-31",
    )
    print(f"  Data range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Number of observations: {len(df)}")

    data = create_core_deposit_data(df, z=z)

    # Run NLS for initial values if requested
    init_params = None
    if config.use_nls_init:
        print("\nRunning NLS for initial values...")
        nls_result = run_nls(data)
        init_params = {
            "lambda": nls_result.params["lambda"],
            "gamma": nls_result.params["gamma"],
            "w1": nls_result.params["w1"],
            "h": nls_result.params["h"],
            "m": nls_result.params["m"],
        }
        print("  NLS estimates:")
        for k, v in init_params.items():
            print(f"    {k}: {v:.4f}")

    # Run MCMC
    print("\nRunning MCMC estimation...")
    result = run_mcmc(data, config, init_params=init_params)

    # Create estimator for prediction
    estimator = MCMCEstimator(
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
        likelihood=config.likelihood,
    )

    # Convert to ArviZ InferenceData
    idata = az.from_numpyro(result.diagnostics["mcmc"])

    # Print statistics
    print_summary(idata)
    print_convergence_diagnostics(idata)
    print_credible_intervals(result, ar_errors=config.ar_errors)
    print_median_survival(result)

    # Generate plots
    print("\nGenerating plots...")
    plot_trace(idata, output_dir / "mcmc_trace.png")
    plot_posterior(idata, output_dir / "mcmc_posterior.png")
    plot_pair(idata, output_dir / "mcmc_pair.png")
    plot_forest(idata, output_dir / "mcmc_forest.png")
    plot_prediction(df["volume"].values, estimator, data, result, output_dir / "mcmc_prediction.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
