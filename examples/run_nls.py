"""NLS and MAP estimation example.

This script demonstrates:
1. Running NLS estimation for fast point estimates
2. Running MAP estimation with prior distributions
3. Comparing NLS and MAP results
4. Using NLS/MAP results as initial values for MCMC
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from coredeposit import (
    CoreDepositData,
    NLSEstimator,
    EstimationResult,
    compute_median_survival,
)
from coredeposit.estimators import MAPPriors, default_map_priors


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
# NLS/MAP Estimation
# ---------------------------------------------------------------------------


@dataclass
class NLSConfig:
    """Configuration for NLS/MAP estimation."""

    loss: str = "soft_l1"
    f_scale: float = 0.05
    use_map: bool = False


def run_nls(data: CoreDepositData, config: NLSConfig) -> EstimationResult:
    """Run NLS estimation.

    Args:
        data: Core deposit data.
        config: NLS configuration.

    Returns:
        Estimation result with point estimates.
    """
    estimator = NLSEstimator(loss=config.loss, f_scale=config.f_scale)
    return estimator.fit(data)


def run_map(
    data: CoreDepositData,
    priors: MAPPriors | None = None,
) -> EstimationResult:
    """Run MAP estimation.

    Args:
        data: Core deposit data.
        priors: MAP priors. If None, uses default priors.

    Returns:
        Estimation result with point estimates.
    """
    if priors is None:
        p = 0 if data.z is None else (1 if data.z.ndim == 1 else data.z.shape[1])
        priors = default_map_priors(p)

    estimator = NLSEstimator(priors=priors)
    return estimator.fit(data)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_fit(
    V_obs: NDArray,
    nls_result: EstimationResult,
    map_result: EstimationResult,
    data: CoreDepositData,
    output_path: Path,
) -> None:
    """Plot NLS and MAP fit comparison."""
    nls_estimator = NLSEstimator()
    map_estimator = NLSEstimator(priors=default_map_priors())

    V_nls = nls_estimator.predict(data, nls_result)
    V_map = map_estimator.predict(data, map_result)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    t = np.arange(len(V_obs))

    # Plot 1: Fit comparison
    ax = axes[0]
    ax.plot(t, V_obs, "o", markersize=3, alpha=0.7, label="Observed")
    ax.plot(t, V_nls, "-", linewidth=2, label="NLS")
    ax.plot(t, V_map, "--", linewidth=2, label="MAP")
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Deposit Balance")
    ax.set_title("Model Fit: NLS vs MAP")
    ax.legend()

    # Plot 2: Residuals
    ax = axes[1]
    ax.plot(t, V_obs - V_nls, "-", linewidth=1.5, alpha=0.7, label="NLS residuals")
    ax.plot(t, V_obs - V_map, "--", linewidth=1.5, alpha=0.7, label="MAP residuals")
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_prior_effect(
    nls_result: EstimationResult,
    map_result: EstimationResult,
    output_path: Path,
) -> None:
    """Plot comparison of NLS and MAP parameter estimates."""
    params = ["lambda", "gamma", "w1", "h", "m"]
    nls_vals = [nls_result.params[p] for p in params]
    map_vals = [map_result.params[p] for p in params]

    x = np.arange(len(params))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, nls_vals, width, label="NLS")
    bars2 = ax.bar(x + width / 2, map_vals, width, label="MAP")

    ax.set_xlabel("Parameter")
    ax.set_ylabel("Estimate")
    ax.set_title("Parameter Estimates: NLS vs MAP")
    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.legend()

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Summary Statistics
# ---------------------------------------------------------------------------


def print_results(name: str, result: EstimationResult) -> None:
    """Print estimation results."""
    print(f"\n{'=' * 50}")
    print(f"{name} Results")
    print("=" * 50)

    print("\nParameter estimates:")
    for param in ["lambda", "gamma", "w1", "h", "m"]:
        print(f"  {param}: {result.params[param]:.4f}")

    if "beta" in result.params:
        print(f"  beta: {result.params['beta']}")

    print("\nDiagnostics:")
    print(f"  Method: {result.diagnostics['method']}")
    print(f"  Success: {result.diagnostics['success']}")
    print(f"  Cost: {result.diagnostics['cost']:.6f}")
    print(f"  Function evaluations: {result.diagnostics['nfev']}")


def print_median_survival_comparison(
    nls_result: EstimationResult,
    map_result: EstimationResult,
) -> None:
    """Print median survival time comparison."""
    print("\n" + "=" * 50)
    print("Median Survival Time (Sticky Deposits)")
    print("=" * 50)

    t50_nls = compute_median_survival(nls_result)
    t50_map = compute_median_survival(map_result)

    print(f"  NLS: {t50_nls:.1f} months")
    print(f"  MAP: {t50_map:.1f} months")


# ---------------------------------------------------------------------------
# Custom Priors Example
# ---------------------------------------------------------------------------


def create_custom_priors() -> MAPPriors:
    """Create custom MAP priors.

    This example shows how to create tighter or different priors
    based on domain knowledge.
    """
    return MAPPriors(
        # Tighter prior on lambda (expect smaller values)
        lambda_dist=stats.lognorm(s=0.4, scale=np.exp(-3.5)),
        # Tighter prior on gamma (expect near 1)
        gamma_dist=stats.lognorm(s=0.25, scale=1.0),
        # Stronger prior favoring lower w1
        w1_dist=stats.beta(2.0, 8.0),
        # Prior for h centered at 0.5
        h_dist=stats.beta(5.0, 5.0),
        # Tighter prior on m (expect ~12 months)
        m_dist=stats.lognorm(s=0.3, scale=np.exp(2.5)),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run NLS and MAP estimation example."""
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    data_dir = Path(__file__).parent / "data"

    # Load data
    print("Loading data...")
    df, z = load_deposit_data(
        boj_path=data_dir / "boj.csv",
        covariate_path=data_dir / "covariate.csv",
        start_date="2002-05-01",
        end_date="2013-04-30",
    )
    print(f"  Data range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Number of observations: {len(df)}")

    data = create_core_deposit_data(df, z=z)

    # Run NLS
    print("\nRunning NLS estimation...")
    nls_config = NLSConfig(loss="soft_l1", f_scale=0.05)
    nls_result = run_nls(data, nls_config)
    print_results("NLS", nls_result)

    # Run MAP with default priors
    print("\nRunning MAP estimation (default priors)...")
    map_result = run_map(data)
    print_results("MAP (default)", map_result)

    # Run MAP with custom priors
    print("\nRunning MAP estimation (custom priors)...")
    custom_priors = create_custom_priors()
    map_custom_result = run_map(data, priors=custom_priors)
    print_results("MAP (custom)", map_custom_result)

    # Compare median survival times
    print_median_survival_comparison(nls_result, map_result)

    # Generate plots
    print("\nGenerating plots...")
    plot_fit(df["volume"].values, nls_result, map_result, data, output_dir / "nls_vs_map_fit.png")
    plot_prior_effect(nls_result, map_result, output_dir / "nls_vs_map_params.png")

    # Show how to use NLS/MAP as MCMC initialization
    print("\n" + "=" * 50)
    print("Using NLS/MAP for MCMC Initialization")
    print("=" * 50)
    print("""
To use NLS/MAP results as MCMC initial values:

    from coredeposit import MCMCEstimator

    init_params = {
        'lambda': nls_result.params['lambda'],
        'gamma': nls_result.params['gamma'],
        'w1': nls_result.params['w1'],
        'h': nls_result.params['h'],
        'm': nls_result.params['m'],
    }

    mcmc = MCMCEstimator(init_params=init_params)
    mcmc_result = mcmc.fit(data)
""")

    print("\nDone!")


if __name__ == "__main__":
    main()
