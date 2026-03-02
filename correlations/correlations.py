#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "split_by_element" / "ag02_daily_means.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
STRONGEST_CSV_PATH = OUTPUT_DIR / "strongest_correlations.csv"
HEATMAP_PATH = OUTPUT_DIR / "correlation_heatmap.png"

# Columns to drop for v2 dataset
COLUMNS_TO_DROP = [
    "GridPhase2Current",
    "GridPhase3Current",
    "BladeALockPitchAngle",
    "BladeBLockPitchAngle",
    "BladeCLockPitchAngle",
    "BladeCPitchBlockManOilPress",
    "GridPhase2Voltage",
    "GridPhase3Voltage",
    "GenCoolingWaterTankPress",
    "GearBearingHSSRotorEndTemperature",
    "GenPhase2Temperature",
    "GearHydrCoolingWaterLevel",
    "BladeBAverageLoad",
    "BladeCAverageLoad",
    "TrafoPhase2CoreTemperature",
    "TrafoPhase3CoreTemperature",
    "ReactivePowerSetPoint",
    "TrafoPhase3Temperature",
    "GridPhase1Current",
    "TotalQProduction",
    "GearBearingHSSGeneratorEndTemperature",
    "NacelleDirection",
    "GearHydrCoolingWaterTemperature",
    "ActivePowerExpected",
    "GearOilInletTemperature",
    "ControllerGroundTemperature",
    "TowerMaxAccY",
    "SpinnerTemperature",
    "RotorSpeed",
    "CapacityFactor",
    "HydraulicOilTemperature",
    "BrakeAccumulatorPress",
    "BladeCPitchBlockCylindPress",
    "WindSpeed",
    "GearBearingIMSFrontTemperature",
    "TrafoPhase2Temperature",
    "NacelleTemperature",
    "TrafoAuxTransformerTemperature",
]


def get_strongest_correlations(correlation_matrix: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Get all correlations for each feature that exceed the threshold in absolute value."""
    rows = []

    for feature in correlation_matrix.columns:
        correlations = correlation_matrix.loc[feature].drop(labels=[feature])
        valid_correlations = correlations.dropna()
        
        # Filter by threshold: correlation > threshold OR correlation < -threshold
        filtered = valid_correlations[(valid_correlations.abs() >= threshold)]
        
        for corr_feature, corr_value in filtered.items():
            rows.append(
                {
                    "feature": feature,
                    "correlated_feature": corr_feature,
                    "correlation": corr_value,
                }
            )

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return result_df
    
    # Sort by absolute correlation descending, then by feature name ascending
    result_df["abs_correlation"] = result_df["correlation"].abs()
    result_df = result_df.sort_values(
        by=["abs_correlation", "feature"],
        ascending=[False, True],
        na_position="last",
    ).drop(columns=["abs_correlation"]).reset_index(drop=True)
    return result_df


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df = df.select_dtypes(include=[np.number])
    df = df.drop(columns=[col for col in COLUMNS_TO_DROP if col in df.columns])

    if df.shape[1] < 2:
        raise ValueError("Need at least two numeric features to compute pairwise correlations.")

    correlation_matrix = df.corr()
    strongest = get_strongest_correlations(correlation_matrix, threshold=0.9)

    print("\nAll correlations per feature with |r| >= 0.9:")
    print(strongest.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nFound {len(strongest)} correlation pairs with |r| >= 0.9")
    print(f"Saved correlations CSV to: {STRONGEST_CSV_PATH}")
    strongest.to_csv(STRONGEST_CSV_PATH, index=False)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Matrix", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(HEATMAP_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved heatmap image to: {HEATMAP_PATH}")
    plt.show()


if __name__ == "__main__":
    main()