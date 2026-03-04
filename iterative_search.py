"""
Iterative GAUFS execution with minority cluster removal.

This script performs iterative GAUFS analysis:
1. Runs GAUFS to find optimal features and number of clusters
2. Performs clustering using the selected features
3. Removes rows belonging to the minority (smallest) cluster
4. Repeats the process with the remaining data

The iteration continues until a stopping condition is met (min samples, max iterations, 
or only one cluster remaining).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering, KMeans
from gaufs import Gaufs
from cluster_stats_calculator import compute_cluster_statistics, create_iteration_summary_csv


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_INPUT_DATA = "data/split_by_element/ag02_daily_means_standard_normalized_reduces.csv"
DEFAULT_OUTPUT_DIR = "results/iterative_search_2_reduced"
DEFAULT_SEED = 42
DEFAULT_MAX_ITERATIONS = 15
DEFAULT_MIN_SAMPLES = 50
DEFAULT_CLUSTERING_METHOD = "agglomerative"  # or "kmeans"

# GAUFS parameters
DEFAULT_NGEN = 1500
DEFAULT_NPOP = 3000
DEFAULT_MUTPB = 0.1
DEFAULT_NUM_GENETIC_EXECUTIONS = 1
DEFAULT_CLUSTER_BAND = (2, 8)

# Columns to drop (if present in data)
DEFAULT_DROP_COLS = [
    "YawState",
    "ActivePowerSetpoint", 
    "AlarmCode",
    "GearOilInletPress",
    "ReactivePowerSetPoint",
    "TotalQProduction",
]


class IterativeGaufs:
    """Performs iterative GAUFS with minority cluster removal."""
    
    def __init__(
        self,
        input_data_path: str | Path,
        output_dir: str | Path,
        seed: int = DEFAULT_SEED,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        clustering_method: str = DEFAULT_CLUSTERING_METHOD,
        drop_cols: list[str] | None = None,
        # GAUFS parameters
        ngen: int = DEFAULT_NGEN,
        npop: int = DEFAULT_NPOP,
        mutpb: float = DEFAULT_MUTPB,
        num_genetic_executions: int = DEFAULT_NUM_GENETIC_EXECUTIONS,
        cluster_band: tuple[int, int] = DEFAULT_CLUSTER_BAND,
    ):
        self.input_data_path = Path(input_data_path)
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.max_iterations = max_iterations
        self.min_samples = min_samples
        self.clustering_method = clustering_method
        self.drop_cols = drop_cols if drop_cols is not None else DEFAULT_DROP_COLS
        
        # GAUFS parameters
        self.ngen = ngen
        self.npop = npop
        self.mutpb = mutpb
        self.num_genetic_executions = num_genetic_executions
        self.cluster_band = cluster_band
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track iteration history
        self.history = []
        
    def load_and_prepare_data(self, data_path: Path) -> pd.DataFrame:
        """Load data and prepare for GAUFS."""
        df = pd.read_csv(data_path)
        
        # Drop unwanted columns
        df = df.drop(columns=self.drop_cols, errors='ignore')
        
        # Remove cluster column if present from previous iteration
        df = df.drop(columns=['cluster'], errors='ignore')
        
        # Select only numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ["Unnamed: 0", "row_index"]]
        
        features_df = df[numeric_cols].copy()
        
        # Drop rows with missing values
        features_df = features_df.dropna()
        
        return features_df
    
    def run_gaufs(self, data: pd.DataFrame, iteration: int) -> tuple[list, int, float]:
        """
        Run GAUFS on the data.
        
        Returns:
            tuple: (selected_features_mask, optimal_clusters, fitness)
        """
        iteration_output_dir = self.output_dir / f"iteration_{iteration}"
        iteration_output_dir.mkdir(parents=True, exist_ok=True)
        
        gaufs = Gaufs(
            seed=self.seed + iteration,  # Different seed per iteration
            ngen=self.ngen,
            npop=self.npop,
            mutpb=self.mutpb,
            num_genetic_executions=self.num_genetic_executions,
            unlabeled_data=data,
            cluster_number_search_band=self.cluster_band,
            output_directory=str(iteration_output_dir),
            convergence_generations = 1
        )
        
        optimal_solution, fitness = gaufs.run()
        
        selected_features_mask = optimal_solution[0]
        optimal_clusters = optimal_solution[1]
        
        # Extract scalar fitness value (might be wrapped in tuple/list/array)
        while isinstance(fitness, (list, tuple, np.ndarray)):
            fitness = fitness[0]
        fitness = float(fitness)
        
        return selected_features_mask, optimal_clusters, fitness
    
    def cluster_data(
        self,
        X: np.ndarray,
        n_clusters: int,
    ) -> np.ndarray:
        """Perform clustering on the data."""
        if self.clustering_method == "agglomerative":
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
            labels = model.fit_predict(X)
        elif self.clustering_method == "kmeans":
            model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=self.seed)
            labels = model.fit_predict(X)
        else:
            raise ValueError("clustering_method must be 'agglomerative' or 'kmeans'")
        return labels
    
    def remove_minority_cluster(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
    ) -> tuple[pd.DataFrame, int, dict]:
        """
        Remove rows belonging to the minority cluster.
        
        Returns:
            tuple: (filtered_data, minority_cluster_id, cluster_counts)
        """
        # Add cluster labels to data
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = labels
        
        # Count cluster sizes
        cluster_counts = pd.Series(labels).value_counts().to_dict()
        
        # Find minority cluster (smallest cluster)
        minority_cluster = min(cluster_counts, key=cluster_counts.get)
        
        # Remove minority cluster
        filtered_data = data_with_clusters[data_with_clusters['cluster'] != minority_cluster].copy()
        filtered_data = filtered_data.drop(columns=['cluster'])
        
        return filtered_data, minority_cluster, cluster_counts
    
    def run_iteration(self, iteration: int, current_data: pd.DataFrame) -> dict | None:
        """
        Run a single iteration of GAUFS + clustering + minority removal.
        
        Returns:
            dict: Iteration results, or None if stopping condition is met
        """
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration}")
        print(f"{'='*80}")
        print(f"Current dataset shape: {current_data.shape}")
        
        # Check stopping condition: minimum samples
        if len(current_data) < self.min_samples:
            print(f"Stopping: Dataset has {len(current_data)} samples, below minimum {self.min_samples}")
            return None
        
        # Step 1: Run GAUFS
        print(f"\n[Step 1] Running GAUFS...")
        selected_features_mask, optimal_clusters, fitness = self.run_gaufs(current_data, iteration)
        
        feature_names = current_data.columns.tolist()
        selected_features = [name for name, selected in zip(feature_names, selected_features_mask) if selected]
        num_selected = sum(selected_features_mask)
        
        print(f"  - Selected {num_selected}/{len(selected_features_mask)} features")
        print(f"  - Optimal clusters: {optimal_clusters}")
        print(f"  - Fitness: {fitness:.4f}")
        print(f"  - Selected features: {selected_features}")
        
        # Check if only 1 cluster is optimal
        if optimal_clusters <= 1:
            print(f"Stopping: Optimal number of clusters is {optimal_clusters}")
            return None
        
        # Step 2: Perform clustering with selected features
        print(f"\n[Step 2] Performing clustering with selected features...")
        X = current_data[selected_features].values
        labels = self.cluster_data(X, optimal_clusters)
        
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        print(f"  - Cluster sizes:")
        for cluster_id, count in cluster_counts.items():
            print(f"    Cluster {cluster_id}: {count} samples")
        
        # Step 3: Remove minority cluster
        print(f"\n[Step 3] Removing minority cluster...")
        filtered_data, minority_cluster, cluster_counts_dict = self.remove_minority_cluster(
            current_data, labels
        )
        
        minority_size = cluster_counts_dict[minority_cluster]
        print(f"  - Minority cluster: {minority_cluster} ({minority_size} samples)")
        print(f"  - Remaining samples: {len(filtered_data)}")
        
        # Save iteration results
        iteration_dir = self.output_dir / f"iteration_{iteration}"
        
        # Save clustered data
        data_with_labels = current_data.copy()
        data_with_labels['cluster'] = labels
        data_with_labels.to_csv(iteration_dir / "clustered_data.csv", index=False)
        
        # Save filtered data (for next iteration)
        filtered_data.to_csv(iteration_dir / "filtered_data.csv", index=False)
        
        # Save iteration summary
        summary = {
            "iteration": iteration,
            "initial_samples": len(current_data),
            "initial_features": len(feature_names),
            "selected_features": selected_features,
            "num_selected_features": num_selected,
            "optimal_clusters": optimal_clusters,
            "fitness": fitness,
            "cluster_counts": cluster_counts_dict,
            "minority_cluster": minority_cluster,
            "minority_cluster_size": minority_size,
            "remaining_samples": len(filtered_data),
        }
        
        import json
        with open(iteration_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Compute and save cluster statistics
        print(f"\n[Step 4] Computing cluster statistics...")
        compute_cluster_statistics(data_with_labels, iteration, iteration_dir)
        
        print(f"\n  - Saved results to {iteration_dir}")
        
        return summary
    
    def run(self) -> list[dict]:
        """
        Run the complete iterative GAUFS process.
        
        Returns:
            list: History of all iteration results
        """
        print(f"Starting Iterative GAUFS")
        print(f"Input data: {self.input_data_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Min samples: {self.min_samples}")
        print(f"Clustering method: {self.clustering_method}")
        
        # Load initial data
        current_data = self.load_and_prepare_data(self.input_data_path)
        print(f"Initial dataset shape: {current_data.shape}")
        
        # Run iterations
        for iteration in range(1, self.max_iterations + 1):
            result = self.run_iteration(iteration, current_data)
            
            if result is None:
                print(f"\nStopping at iteration {iteration}")
                break
            
            self.history.append(result)
            
            # Load filtered data for next iteration
            iteration_dir = self.output_dir / f"iteration_{iteration}"
            current_data = pd.read_csv(iteration_dir / "filtered_data.csv")
        
        # Save complete history
        print(f"\n{'='*80}")
        print(f"COMPLETE - Ran {len(self.history)} iterations")
        print(f"{'='*80}")
        
        import json
        with open(self.output_dir / "complete_history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        
        # Print summary
        print("\nIteration Summary:")
        print("-" * 80)
        for result in self.history:
            print(f"Iteration {result['iteration']}: "
                  f"{result['initial_samples']} → {result['remaining_samples']} samples, "
                  f"{result['num_selected_features']} features, "
                  f"{result['optimal_clusters']} clusters, "
                  f"fitness={result['fitness']:.4f}")
        
        # Create iteration summary CSV
        print(f"\n{'='*80}")
        print("Generating iteration summary...")
        print(f"{'='*80}")
        create_iteration_summary_csv(self.output_dir, max_iterations=self.max_iterations)
        
        return self.history


def main():
    """Main entry point."""
    # Configuration
    input_data_path = DEFAULT_INPUT_DATA
    output_dir = DEFAULT_OUTPUT_DIR
    seed = DEFAULT_SEED
    max_iterations = DEFAULT_MAX_ITERATIONS
    min_samples = DEFAULT_MIN_SAMPLES
    clustering_method = DEFAULT_CLUSTERING_METHOD
    
    # Create and run iterative GAUFS
    iterative_gaufs = IterativeGaufs(
        input_data_path=input_data_path,
        output_dir=output_dir,
        seed=seed,
        max_iterations=max_iterations,
        min_samples=min_samples,
        clustering_method=clustering_method,
        # GAUFS parameters
        ngen=DEFAULT_NGEN,
        npop=DEFAULT_NPOP,
        mutpb=DEFAULT_MUTPB,
        num_genetic_executions=DEFAULT_NUM_GENETIC_EXECUTIONS,
        cluster_band=DEFAULT_CLUSTER_BAND,
    )
    
    history = iterative_gaufs.run()
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Total iterations completed: {len(history)}")


if __name__ == "__main__":
    main()
