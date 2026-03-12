import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    Normalizer,
)


SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "maxabs": MaxAbsScaler,
    "robust": RobustScaler,
    "normalize": Normalizer,
}


# Manual configuration (edit as needed)
INPUT_PATH = "/home/josee/gaufs-coreset/data/split_by_element/ag10.csv"
OUTPUT_AGGREGATED = "/home/josee/gaufs-coreset/data/split_by_element/ag10_daily_means.csv"
NORMALIZATION_METHOD = "standard"  # standard | minmax | maxabs | robust | normalize
OUTPUT_NORMALIZED = None  # set a path or leave None to auto-generate


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Unnamed: 0" in numeric_cols:
        numeric_cols.remove("Unnamed: 0")
    return numeric_cols


def aggregate_by_day(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    return df.groupby("Day")[numeric_cols].mean()


def normalize_data(
    df: pd.DataFrame, numeric_cols: list[str], method: str
) -> pd.DataFrame:
    scaler_cls = SCALERS[method]
    scaler = scaler_cls()
    normalized = df.copy()
    normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return normalized


def build_normalized_output_path(aggregated_path: str, method: str) -> str:
    if aggregated_path.lower().endswith(".csv"):
        base = aggregated_path[:-4]
    else:
        base = aggregated_path
    return f"{base}_{method}_normalized.csv"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    print("Dataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())

    if "Timestamp" in df.columns:
        df["Date"] = pd.to_datetime(df["Timestamp"]).dt.date

    numeric_cols = get_numeric_columns(df)

    aggregated = aggregate_by_day(df, numeric_cols)

    print("\n\nAggregated data by day:")
    print(aggregated)

    aggregated.to_csv(OUTPUT_AGGREGATED)
    print(f"\n\nAggregated data saved to: {OUTPUT_AGGREGATED}")

    normalized_output = (
        OUTPUT_NORMALIZED
        if OUTPUT_NORMALIZED
        else build_normalized_output_path(OUTPUT_AGGREGATED, NORMALIZATION_METHOD)
    )
    normalized = normalize_data(aggregated, numeric_cols, NORMALIZATION_METHOD)
    normalized.to_csv(normalized_output)
    print(f"\n\nNormalized data saved to: {normalized_output}")

    print("\n\nSummary of aggregated data:")
    print(aggregated.describe())


if __name__ == "__main__":
    main()
