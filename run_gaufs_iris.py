"""
Apply GAUFS (Genetic Algorithm Unsupervised Feature Selection) to the Iris dataset.
Loads data/iris.csv, runs GAUFS, prints selected features and optimal clusters,
and optionally evaluates against true labels (AMI) with a comparison plot.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots without display

import pandas as pd
from gaufs import Gaufs

# ---------------------------------------------------------------------------
# 1. Data loading and preparation
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
COLUMN_NAMES = FEATURE_NAMES + ["species"]


def main():
    df = pd.read_csv("data/iris.csv", header=None, names=COLUMN_NAMES)
    features_df = df[FEATURE_NAMES].copy()
    species = df["species"].copy()

    # Encode species to numeric for optional AMI/NMI evaluation
    species_encoded = pd.Categorical(species).codes

    # ---------------------------------------------------------------------------
    # 2. Run GAUFS
    # ---------------------------------------------------------------------------
    gaufs = Gaufs(
        unlabeled_data=features_df,
        cluster_number_search_band=(2, 25),
        seed=42,
        output_directory="out2/",
    )

    optimal_solution, _ = gaufs.run()

    # ---------------------------------------------------------------------------
    # 3. Extract and print results
    # ---------------------------------------------------------------------------
    selected_features_mask = optimal_solution[0]  # 1=selected, 0=not selected
    optimal_clusters = optimal_solution[1]

    selected_names = [name for name, s in zip(FEATURE_NAMES, selected_features_mask) if s]
    num_selected = sum(selected_features_mask)

    fitness = gaufs.optimal_fitness

    print(f"Selected {num_selected} out of {len(selected_features_mask)} features: {selected_names}")
    print(f"Optimal number of clusters: {optimal_clusters}")
    print(f"Fitness score: {fitness}")

    # ---------------------------------------------------------------------------
    # 4. Optional: compare with ground truth (AMI) and comparison plot
    # ---------------------------------------------------------------------------
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_mutual_info_score

    # Get cluster predictions using the same default clustering (Ward linkage)
    # on the selected features only
    selected_cols = [i for i, s in enumerate(selected_features_mask) if s]
    X_selected = features_df.iloc[:, selected_cols]

    model = AgglomerativeClustering(n_clusters=optimal_clusters, linkage="ward")
    pred_labels = model.fit_predict(X_selected)

    ami = adjusted_mutual_info_score(species_encoded, pred_labels)
    print(f"Adjusted Mutual Information (vs true species): {ami:.4f}")

    # Generate comparison plot: fitness vs external metric (AMI) with true labels
    from gaufs.evaluation_metrics import AdjustedMutualInformationScore

    ami_metric = AdjustedMutualInformationScore(true_labels=species_encoded)
    try:
        gaufs.get_plot_comparing_solution_with_another_metric(
            new_metric=ami_metric,
            true_number_of_labels=3,  # Iris has 3 species
        )
    except Exception as e:
        print(f"Could not generate comparison plot: {e}")


if __name__ == "__main__":
    main()
