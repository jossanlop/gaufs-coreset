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
from collections import deque
from dataclasses import dataclass, asdict
from sklearn.cluster import AgglomerativeClustering, KMeans
from gaufs import Gaufs
from cluster_stats_calculator import compute_cluster_statistics, create_iteration_summary_csv, create_tree_summary_csv
import json


# ---------------------------------------------------------------------------
# Node Definition for Tree-based Expansion
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """Represents a node in the iterative search tree."""
    node_id: str  # Unique identifier (e.g., "node_0", "node_0_1", "node_0_2")
    parent_id: str | None  # Parent node ID
    depth: int  # Depth in tree (root is 0)
    input_path: Path  # Path to input data for this node
    status: str  # "pending", "running", "completed", "discarded"
    cluster_id: int | None = None  # Which cluster this node represents (for non-root)
    stop_reason: str | None = None  # Reason for stopping expansion


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_INPUT_DATA = "data/split_by_element/ag10_daily_means_standard_normalized_reduced.csv"
DEFAULT_OUTPUT_DIR = "results/iterative_search_ag10"
DEFAULT_SEED = 42
DEFAULT_MAX_ITERATIONS = 15
DEFAULT_MIN_SAMPLES = 50
DEFAULT_CLUSTERING_METHOD = "agglomerative"  # or "kmeans"

# Tree expansion parameters
DEFAULT_MIN_CLUSTER_PERCENTAGE = 0.05  # 5% of parent size
DEFAULT_MIN_CLUSTER_SIZE_ABSOLUTE = 50  # Absolute minimum for largest child
DEFAULT_MAX_TREE_DEPTH = 20  # Maximum depth to prevent infinite recursion

# GAUFS parameters
DEFAULT_NGEN = 1500
DEFAULT_NPOP = 3000
DEFAULT_MUTPB = 0.1
DEFAULT_NUM_GENETIC_EXECUTIONS = 1
DEFAULT_CLUSTER_BAND = (2, 4)

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
        # Tree expansion parameters
        min_cluster_percentage: float = DEFAULT_MIN_CLUSTER_PERCENTAGE,
        min_cluster_size_absolute: int = DEFAULT_MIN_CLUSTER_SIZE_ABSOLUTE,
        max_tree_depth: int = DEFAULT_MAX_TREE_DEPTH,
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
        
        # Tree expansion parameters
        self.min_cluster_percentage = min_cluster_percentage
        self.min_cluster_size_absolute = min_cluster_size_absolute
        self.max_tree_depth = max_tree_depth
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track nodes and history
        self.nodes: dict[str, Node] = {}  # node_id -> Node
        self.node_results: dict[str, dict] = {}  # node_id -> result summary
        self.node_counter = [0]  # Use list to allow increment in nested function
        self.children_map: dict[str, list[str]] = {}  # parent_id -> [child_ids]
        
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
    
    def run_gaufs(self, data: pd.DataFrame, node_id: str, depth: int) -> tuple[list, int, float]:
        """
        Run GAUFS on the data.
        
        Returns:
            tuple: (selected_features_mask, optimal_clusters, fitness)
        """
        node_output_dir = self.output_dir / f"depth_{depth}" / node_id
        node_output_dir.mkdir(parents=True, exist_ok=True)
        
        gaufs = Gaufs(
            seed=self.seed + hash(node_id) % 10000,  # Unique seed per node
            ngen=self.ngen,
            npop=self.npop,
            mutpb=self.mutpb,
            num_genetic_executions=self.num_genetic_executions,
            unlabeled_data=data,
            cluster_number_search_band=self.cluster_band,
            output_directory=str(node_output_dir),
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

    def get_child_cluster_data(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        parent_size: int,
    ) -> dict[int, tuple[pd.DataFrame, bool]]:
        """
        Get child cluster data and determine which should be kept.
        
        Returns:
            dict: cluster_id -> (data_df, should_keep)
                where should_keep indicates if cluster passes pruning threshold
        """
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = labels
        
        # Count cluster sizes
        cluster_counts = pd.Series(labels).value_counts().to_dict()
        
        # Apply pruning: keep clusters > min_cluster_percentage of parent
        min_cluster_count = max(1, int(parent_size * self.min_cluster_percentage))
        
        result = {}
        for cluster_id in sorted(cluster_counts.keys()):
            cluster_size = cluster_counts[cluster_id]
            should_keep = cluster_size >= min_cluster_count
            
            cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id].copy()
            cluster_data = cluster_data.drop(columns=['cluster'])
            
            result[cluster_id] = (cluster_data, should_keep)
        
        return result
    
    def run_node(self, node: Node) -> dict | None:
        """
        Run a single node in the tree.
        
        Returns:
            dict: Node results with children info, or None if stopping condition is met
        """
        print(f"\n{'='*80}")
        print(f"NODE {node.node_id} (depth={node.depth})")
        print(f"{'='*80}")
        
        # Load input data
        current_data = pd.read_csv(node.input_path)
        print(f"Input dataset shape: {current_data.shape}")
        
        # Check stopping condition: minimum samples
        if len(current_data) < self.min_samples:
            print(f"Stopping: Dataset has {len(current_data)} samples, below minimum {self.min_samples}")
            node.status = "discarded"
            node.stop_reason = f"Too few samples ({len(current_data)} < {self.min_samples})"
            return None
        
        # Check stopping condition: maximum depth
        if node.depth >= self.max_tree_depth:
            print(f"Stopping: Maximum tree depth ({self.max_tree_depth}) reached")
            node.status = "discarded"
            node.stop_reason = f"Max tree depth reached"
            return None
        
        # Step 1: Run GAUFS
        print(f"\n[Step 1] Running GAUFS...")
        selected_features_mask, optimal_clusters, fitness = self.run_gaufs(current_data, node.node_id, node.depth)
        
        feature_names = current_data.columns.tolist()
        selected_features = [name for name, selected in zip(feature_names, selected_features_mask) if selected]
        num_selected = sum(selected_features_mask)
        
        print(f"  - Selected {num_selected}/{len(selected_features_mask)} features")
        print(f"  - Optimal clusters: {optimal_clusters}")
        print(f"  - Fitness: {fitness:.4f}")
        
        # Check if only 1 cluster is optimal
        if optimal_clusters <= 1:
            print(f"Stopping: Optimal number of clusters is {optimal_clusters}")
            node.status = "completed"
            node.stop_reason = "Optimal clusters <= 1"
            
            # Save single cluster result
            node_dir = self.output_dir / f"depth_{node.depth}" / node.node_id
            node_dir.mkdir(parents=True, exist_ok=True)
            current_data.to_csv(node_dir / "clustered_data.csv", index=False)
            
            result = {
                "node_id": node.node_id,
                "parent_id": node.parent_id,
                "depth": node.depth,
                "initial_samples": len(current_data),
                "initial_features": len(feature_names),
                "selected_features": selected_features,
                "num_selected_features": num_selected,
                "optimal_clusters": optimal_clusters,
                "fitness": fitness,
                "stop_reason": node.stop_reason,
                "children": [],
            }
            return result
        
        # Step 2: Perform clustering with selected features
        print(f"\n[Step 2] Performing clustering with selected features...")
        X = current_data[selected_features].values
        labels = self.cluster_data(X, optimal_clusters)
        
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        print(f"  - Cluster sizes:")
        for cluster_id, count in cluster_counts.items():
            print(f"    Cluster {cluster_id}: {count} samples")
        
        # Step 3: Get child cluster data and apply pruning
        print(f"\n[Step 3] Preparing child clusters with pruning...")
        child_clusters = self.get_child_cluster_data(current_data, labels, len(current_data))
        
        # Save clustered data for this node
        node_dir = self.output_dir / f"depth_{node.depth}" / node.node_id
        node_dir.mkdir(parents=True, exist_ok=True)
        
        data_with_labels = current_data.copy()
        data_with_labels['cluster'] = labels
        data_with_labels.to_csv(node_dir / "clustered_data.csv", index=False)
        
        # Compute and save cluster statistics
        print(f"\n[Step 4] Computing cluster statistics...")
        compute_cluster_statistics(data_with_labels, 0, node_dir)
        
        # Determine which children to keep
        children_to_process = []
        discarded_children = []
        
        for cluster_id in sorted(child_clusters.keys()):
            cluster_data, should_keep = child_clusters[cluster_id]
            cluster_size = len(cluster_data)
            
            if should_keep:
                children_to_process.append((cluster_id, cluster_data))
                print(f"  - Cluster {cluster_id}: {cluster_size} samples - KEEP")
            else:
                discarded_children.append((cluster_id, cluster_size, "Below 5% threshold"))
                print(f"  - Cluster {cluster_id}: {cluster_size} samples - DISCARD (below 5% threshold)")
        
        # Check stopping condition: if largest child is below absolute minimum
        if children_to_process:
            largest_child_size = max(len(data) for _, data in children_to_process)
            if largest_child_size < self.min_cluster_size_absolute:
                print(f"\nStopping: Largest child cluster ({largest_child_size}) below minimum ({self.min_cluster_size_absolute})")
                node.status = "completed"
                node.stop_reason = f"Largest child below threshold ({largest_child_size} < {self.min_cluster_size_absolute})"
                
                result = {
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "depth": node.depth,
                    "initial_samples": len(current_data),
                    "initial_features": len(feature_names),
                    "selected_features": selected_features,
                    "num_selected_features": num_selected,
                    "optimal_clusters": optimal_clusters,
                    "fitness": fitness,
                    "cluster_counts": {int(k): int(v) for k, v in cluster_counts.items()},
                    "stop_reason": node.stop_reason,
                    "children": [],
                    "discarded_children": discarded_children,
                }
                return result
        else:
            print(f"\nStopping: No children passed pruning threshold")
            node.status = "completed"
            node.stop_reason = "No children passed pruning threshold"
            
            result = {
                "node_id": node.node_id,
                "parent_id": node.parent_id,
                "depth": node.depth,
                "initial_samples": len(current_data),
                "initial_features": len(feature_names),
                "selected_features": selected_features,
                "num_selected_features": num_selected,
                "optimal_clusters": optimal_clusters,
                "fitness": fitness,
                "cluster_counts": {int(k): int(v) for k, v in cluster_counts.items()},
                "stop_reason": node.stop_reason,
                "children": [],
                "discarded_children": discarded_children,
            }
            return result
        
        # Step 4: Create child nodes
        print(f"\n[Step 5] Creating child nodes...")
        child_infos = []
        
        for cluster_id, cluster_data in children_to_process:
            child_id = self._generate_child_node_id(node.node_id)
            child_input_path = node_dir / f"child_{cluster_id}_data.csv"
            cluster_data.to_csv(child_input_path, index=False)
            
            child_node = Node(
                node_id=child_id,
                parent_id=node.node_id,
                depth=node.depth + 1,
                input_path=child_input_path,
                status="pending",
                cluster_id=cluster_id,
            )
            
            self.nodes[child_id] = child_node
            if node.node_id not in self.children_map:
                self.children_map[node.node_id] = []
            self.children_map[node.node_id].append(child_id)
            
            child_infos.append(child_id)
            print(f"  - Created child node {child_id} from cluster {cluster_id} ({len(cluster_data)} samples)")
        
        node.status = "completed"
        
        # Save node summary
        summary = {
            "node_id": node.node_id,
            "parent_id": node.parent_id,
            "depth": node.depth,
            "initial_samples": len(current_data),
            "initial_features": len(feature_names),
            "selected_features": selected_features,
            "num_selected_features": num_selected,
            "optimal_clusters": optimal_clusters,
            "fitness": fitness,
            "cluster_counts": {int(k): int(v) for k, v in cluster_counts.items()},
            "children": child_infos,
            "discarded_children": discarded_children,
        }
        
        with open(node_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n  - Saved results to {node_dir}")
        
        return summary
    
    def _generate_child_node_id(self, parent_id: str) -> str:
        """Generate a unique child node ID."""
        self.node_counter[0] += 1
        return f"node_{self.node_counter[0]}"
    
    def run(self) -> dict:
        """
        Run the complete tree-based iterative GAUFS process.
        
        Returns:
            dict: Complete tree structure with all nodes and results
        """
        print(f"Starting Tree-based Iterative GAUFS")
        print(f"Input data: {self.input_data_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Min samples: {self.min_samples}")
        print(f"Min cluster percentage: {self.min_cluster_percentage}")
        print(f"Min cluster absolute size: {self.min_cluster_size_absolute}")
        print(f"Max tree depth: {self.max_tree_depth}")
        print(f"Clustering method: {self.clustering_method}")
        
        # Load initial data
        initial_data = self.load_and_prepare_data(self.input_data_path)
        print(f"Initial dataset shape: {initial_data.shape}")
        
        # Create root input file
        root_input_path = self.output_dir / "root_data.csv"
        initial_data.to_csv(root_input_path, index=False)
        
        # Create root node
        root_node = Node(
            node_id="node_0",
            parent_id=None,
            depth=0,
            input_path=root_input_path,
            status="pending",
        )
        
        self.nodes["node_0"] = root_node
        self.node_counter[0] = 0  # Start counter from 0
        
        # BFS worklist engine
        worklist = deque([root_node])
        processed_nodes = []
        
        while worklist:
            current_node = worklist.popleft()
            
            print(f"\nProcessing node {current_node.node_id}...")
            current_node.status = "running"
            
            # Run the node
            result = self.run_node(current_node)
            
            if result is not None:
                self.node_results[current_node.node_id] = result
                processed_nodes.append(result)
                
                # Enqueue child nodes if they exist
                if current_node.node_id in self.children_map:
                    for child_id in self.children_map[current_node.node_id]:
                        child_node = self.nodes[child_id]
                        worklist.append(child_node)
                        print(f"  Enqueued child node {child_id}")
        
        # Save complete tree structure
        print(f"\n{'='*80}")
        print(f"TREE COMPLETE - Processed {len(processed_nodes)} nodes")
        print(f"{'='*80}")
        
        tree_summary = {
            "root_node": "node_0",
            "total_nodes": len(self.nodes),
            "total_processed": len(processed_nodes),
            "nodes": {nid: asdict(node) for nid, node in self.nodes.items()},
            "results": self.node_results,
            "children_map": self.children_map,
            "config": {
                "min_cluster_percentage": self.min_cluster_percentage,
                "min_cluster_size_absolute": self.min_cluster_size_absolute,
                "max_tree_depth": self.max_tree_depth,
                "min_samples": self.min_samples,
                "clustering_method": self.clustering_method,
            }
        }
        
        # Save complete history
        with open(self.output_dir / "complete_history.json", "w") as f:
            json.dump(tree_summary, f, indent=2, default=str)
        
        # Print summary
        print("\nTree Structure Summary:")
        print("-" * 80)
        
        depths = {}
        for result in processed_nodes:
            depth = result.get("depth", 0)
            if depth not in depths:
                depths[depth] = []
            depths[depth].append(result)
        
        for depth in sorted(depths.keys()):
            nodes_at_depth = depths[depth]
            print(f"\nDepth {depth}: {len(nodes_at_depth)} node(s)")
            for result in nodes_at_depth:
                print(f"  {result['node_id']}: "
                      f"{result['initial_samples']} samples, "
                      f"{result['num_selected_features']} features, "
                      f"{result['optimal_clusters']} clusters, "
                      f"fitness={result['fitness']:.4f}")
                if result.get("children"):
                    print(f"    Children: {result['children']}")
                if result.get("stop_reason"):
                    print(f"    Stop reason: {result['stop_reason']}")
        
        print(f"\n{'='*80}")
        print("Generating tree summary...")
        print(f"{'='*80}")
        create_tree_summary_csv(self.output_dir)
        
        print(f"\n{'='*80}")
        print(f"Tree saved to: {self.output_dir}")
        print(f"{'='*80}")
        
        return tree_summary


def main():
    """Main entry point."""
    # Configuration
    input_data_path = DEFAULT_INPUT_DATA
    output_dir = DEFAULT_OUTPUT_DIR
    seed = DEFAULT_SEED
    max_iterations = DEFAULT_MAX_ITERATIONS
    min_samples = DEFAULT_MIN_SAMPLES
    clustering_method = DEFAULT_CLUSTERING_METHOD
    
    # Tree expansion parameters
    min_cluster_percentage = DEFAULT_MIN_CLUSTER_PERCENTAGE
    min_cluster_size_absolute = DEFAULT_MIN_CLUSTER_SIZE_ABSOLUTE
    max_tree_depth = DEFAULT_MAX_TREE_DEPTH
    
    # Create and run tree-based iterative GAUFS
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
        # Tree parameters
        min_cluster_percentage=min_cluster_percentage,
        min_cluster_size_absolute=min_cluster_size_absolute,
        max_tree_depth=max_tree_depth,
    )
    
    tree_summary = iterative_gaufs.run()
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Total nodes explored: {tree_summary['total_nodes']}")
    print(f"Total nodes processed: {tree_summary['total_processed']}")


if __name__ == "__main__":
    main()
