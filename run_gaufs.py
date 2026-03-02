"""
Apply GAUFS (Genetic Algorithm Unsupervised Feature Selection) to the wind turbine dataset.
Loads data/split_by_element/ag02.csv, runs GAUFS on numeric features, prints selected 
features and optimal clusters.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots without display

import pandas as pd
import numpy as np
from gaufs import Gaufs

# ---------------------------------------------------------------------------
# 1. Data loading and preparation
# ---------------------------------------------------------------------------
# DATA_FILE = "data/split_by_element/ag02_daily_means_standard_normalized.csv"
DATA_FILE = "results/out_ag02_seed_10/results/final_clustering_no_cluster_0.csv"


def main():
    # Load the wind turbine dataset
    df = pd.read_csv(DATA_FILE)
    
    df.drop(columns=['YawState','ActivePowerSetpoint','AlarmCode',
                    'GearOilInletPress','ReactivePowerSetPoint','TotalQProduction'], 
            inplace=True, errors='ignore')  # Remove index column if present
    df.drop(columns=['cluster'], inplace=True, errors='ignore')

    # Select only numeric features (exclude timestamps, strings, etc.)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove the index column if present
    numeric_cols = [col for col in numeric_cols if col != "Unnamed: 0"]
    
    features_df = df[numeric_cols].copy()
    
    # Drop rows with missing values for this analysis
    features_df = features_df.dropna()
    
    print(f"Dataset shape: {features_df.shape}")
    print(f"Number of features: {len(numeric_cols)}")
    print(f"Features: {numeric_cols[:10]}...")  # Print first 10 feature names


    # ---------------------------------------------------------------------------
    # 2. Run GAUFS
    # ---------------------------------------------------------------------------
    
    seed = 11

    gaufs = Gaufs(
        seed=seed,
        ngen=150,
        npop=1500,
        mutpb = 0.1, # mutation probability
        num_genetic_executions=1,
        unlabeled_data=features_df,
        cluster_number_search_band=(2, 8),

        output_directory=f"results/out_ag02_seed_{seed}/",
    )

    optimal_solution, _ = gaufs.run()

    # ---------------------------------------------------------------------------
    # 3. Extract and print results
    # ---------------------------------------------------------------------------
    selected_features_mask = optimal_solution[0]  # 1=selected, 0=not selected
    optimal_clusters = optimal_solution[1]

    selected_names = [name for name, s in zip(numeric_cols, selected_features_mask) if s]
    num_selected = sum(selected_features_mask)

    fitness = gaufs.optimal_fitness

    print(f"\nSelected {num_selected} out of {len(selected_features_mask)} features:")
    print(f"Features: {selected_names}")
    print(f"Optimal number of clusters: {optimal_clusters}")
    print(f"Fitness score: {fitness}")



if __name__ == "__main__":
    main()
