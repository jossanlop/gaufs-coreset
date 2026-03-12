"""
Cluster statistics calculator for iterative GAUFS analysis.

Computes per-cluster feature statistics (mean, std, min, max, median, count)
and saves them as CSV files for each iteration.

Includes denormalization capability to convert normalized statistics back to
original feature units using normalization parameters from ag02_daily_means_reduced.csv.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def get_normalization_parameters(
    original_data_path: Path | str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Calculate normalization parameters (mean and std) from original data.
    
    Parameters
    ----------
    original_data_path : Path or str
        Path to the original (denormalized) data file.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str]]
        Tuple of (means, stds, feature_names) for each numeric feature.
    """
    original_df = pd.read_csv(original_data_path)
    
    # Get numeric columns
    numeric_cols = original_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ["Unnamed: 0", "row_index"]]
    
    features_df = original_df[numeric_cols].dropna()
    
    # Calculate parameters using StandardScaler to match the normalization
    scaler = StandardScaler()
    scaler.fit(features_df)
    
    return scaler.mean_, scaler.scale_, numeric_cols  # scale_ is the std


def denormalize_statistics_row(
    stats_row: dict,
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    feature_names: list[str],
) -> dict:
    """
    Denormalize a statistics row using normalization parameters.
    
    Parameters
    ----------
    stats_row : dict
        Dictionary with normalized statistics (format: 'feature_statistic').
    feature_means : np.ndarray
        Means used for normalization, indexed by feature.
    feature_stds : np.ndarray
        Standard deviations used for normalization, indexed by feature.
    feature_names : list[str]
        List of feature names in order matching the means/stds arrays.
    
    Returns
    -------
    dict
        Updated row with denormalized statistics.
    """
    denorm_row = stats_row.copy()
    
    # Create mapping from feature name to mean/std
    feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    
    for feature_name in feature_names:
        if feature_name not in feature_to_idx:
            continue
            
        idx = feature_to_idx[feature_name]
        mean = feature_means[idx]
        std = feature_stds[idx]
        
        # Denormalize each statistic for this feature
        # Formula: original_value = normalized_value * std + mean
        
        if f'{feature_name}_mean' in denorm_row:
            denorm_row[f'{feature_name}_mean'] = denorm_row[f'{feature_name}_mean'] * std + mean
        
        if f'{feature_name}_std' in denorm_row:
            # Std is scaled differently: original_std = normalized_std * std
            denorm_row[f'{feature_name}_std'] = denorm_row[f'{feature_name}_std'] * std
        
        if f'{feature_name}_min' in denorm_row:
            denorm_row[f'{feature_name}_min'] = denorm_row[f'{feature_name}_min'] * std + mean
        
        if f'{feature_name}_max' in denorm_row:
            denorm_row[f'{feature_name}_max'] = denorm_row[f'{feature_name}_max'] * std + mean
        
        if f'{feature_name}_median' in denorm_row:
            denorm_row[f'{feature_name}_median'] = denorm_row[f'{feature_name}_median'] * std + mean
    
    return denorm_row


def compute_cluster_statistics(
    clustered_df: pd.DataFrame,
    iteration_num: int,
    output_dir: Path | str,
    original_data_path: Path | str | None = None,
    denormalize: bool = True,
) -> pd.DataFrame:
    """
    Compute descriptive statistics for each cluster.
    
    For each cluster, calculates mean, std, min, max, median, and count
    for all feature columns (excluding the 'cluster' column).
    
    Parameters
    ----------
    clustered_df : pd.DataFrame
        DataFrame with features and a 'cluster' column containing cluster labels.
        Features should be in normalized space.
    iteration_num : int
        Iteration number (for naming and reference).
    output_dir : Path or str
        Directory where cluster_statistics.csv will be saved.
    original_data_path : Path or str, optional
        Path to original (denormalized) data for normalization parameters.
        If None, will use default: 'data/split_by_element/ag02_daily_means_reduced.csv'
    denormalize : bool, optional
        Whether to denormalize statistics to original units. Default is True.
    
    Returns
    -------
    pd.DataFrame
        Statistics DataFrame with clusters as rows and feature statistics as columns.
    
    Notes
    -----
    Output file: {output_dir}/cluster_statistics.csv
    File structure:
        - Rows: cluster IDs (0, 1, 2, ...)
        - Columns: Feature_mean, Feature_std, Feature_min, Feature_max, 
                   Feature_median, cluster_size
    
    If denormalize=True, statistics are converted back to original units.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate features from cluster labels
    cluster_col = clustered_df['cluster'].copy()
    features_df = clustered_df.drop(columns=['cluster'])
    
    # Get feature names in order
    feature_names = features_df.columns.tolist()
    
    # Combine for grouped computation
    data_with_clusters = features_df.copy()
    data_with_clusters['cluster'] = cluster_col
    
    # Load normalization parameters if denormalizing
    feature_means = None
    feature_stds = None
    feature_names_from_original = None
    
    if denormalize:
        if original_data_path is None:
            # Use default path: the reduced non-normalized data
            original_data_path = Path("data/split_by_element/ag02_daily_means_reduced.csv")
        
        original_data_path = Path(original_data_path)
        if original_data_path.exists():
            print(f"    Loading normalization parameters from {original_data_path}")
            feature_means, feature_stds, feature_names_from_original = get_normalization_parameters(original_data_path)
        else:
            print(f"    Warning: Original data file not found at {original_data_path}, skipping denormalization")
            denormalize = False
    
    # Initialize list to hold statistics for each cluster
    stats_list = []
    
    # Get unique clusters and sort them
    clusters = sorted(data_with_clusters['cluster'].unique())
    
    for cluster_id in clusters:
        cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
        cluster_features = cluster_data.drop(columns=['cluster'])
        
        stats_row = {'cluster': cluster_id}
        
        # Compute statistics for each feature
        for feature in cluster_features.columns:
            feature_values = cluster_features[feature].dropna()
            
            stats_row[f'{feature}_mean'] = feature_values.mean()
            stats_row[f'{feature}_std'] = feature_values.std()
            stats_row[f'{feature}_min'] = feature_values.min()
            stats_row[f'{feature}_max'] = feature_values.max()
            stats_row[f'{feature}_median'] = feature_values.median()
        
        # Denormalize if requested
        if denormalize and feature_means is not None and feature_stds is not None:
            stats_row = denormalize_statistics_row(
                stats_row,
                feature_means,
                feature_stds,
                feature_names_from_original,
            )
        
        # Add cluster size
        stats_row['cluster_size'] = len(cluster_data)
        
        stats_list.append(stats_row)
    
    # Create DataFrame from stats
    stats_df = pd.DataFrame(stats_list)
    stats_df = stats_df.set_index('cluster')
    
    # Save to CSV
    output_file = output_dir / 'cluster_statistics.csv'
    stats_df.to_csv(output_file)
    
    status = "denormalized (original units)" if denormalize else "normalized space"
    print(f"  - Saved cluster statistics ({status}) to {output_file}")
    
    return stats_df


def create_iteration_summary_csv(
    results_dir: Path | str,
    max_iterations: int = 15,
    output_filename: str = "cluster_statistics_summary.csv",
) -> pd.DataFrame:
    """
    Create a summary CSV file with mean statistics and cluster sizes across all iterations.
    
    Aggregates mean values and cluster sizes from cluster_statistics.csv files in each iteration
    into a single CSV with one row per (iteration, cluster) combination.
    
    Parameters
    ----------
    results_dir : Path or str
        Root results directory (e.g., 'results/iterative_search_2_reduced').
    max_iterations : int
        Maximum number of iterations to process. Default is 15.
    output_filename : str
        Name of the output summary file. Default is 'cluster_statistics_summary.csv'.
    
    Returns
    -------
    pd.DataFrame
        Summary DataFrame with columns: iteration, cluster, cluster_size, feature_1_mean, feature_2_mean, ...
    
    Notes
    -----
    Output file is saved to {results_dir}/{output_filename}
    """
    results_dir = Path(results_dir)
    summary_rows = []
    
    for iteration in range(1, max_iterations + 1):
        iteration_dir = results_dir / f'iteration_{iteration}'
        stats_file = iteration_dir / 'cluster_statistics.csv'
        
        if not stats_file.exists():
            continue
        
        # Load cluster statistics for this iteration
        stats_df = pd.read_csv(stats_file, index_col=0)
        
        # Extract only mean columns
        mean_cols = [col for col in stats_df.columns if col.endswith('_mean')]
        
        for cluster_id in stats_df.index:
            row = {'iteration': iteration, 'cluster': cluster_id}
            
            # Add cluster size
            if 'cluster_size' in stats_df.columns:
                row['cluster_size'] = int(stats_df.loc[cluster_id, 'cluster_size'])
            
            # Add mean values for all features
            for col in mean_cols:
                row[col] = stats_df.loc[cluster_id, col]
            
            summary_rows.append(row)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_rows)
    
    # Ensure proper column order: iteration, cluster, cluster_size, then feature means
    cols = ['iteration', 'cluster', 'cluster_size']
    mean_cols = [col for col in summary_df.columns if col.endswith('_mean')]
    summary_df = summary_df[cols + sorted(mean_cols)]
    
    # Save summary
    output_file = results_dir / output_filename
    summary_df.to_csv(output_file, index=False)
    
    print(f"Saved iteration summary to {output_file}")
    print(f"  - Shape: {summary_df.shape}")
    print(f"  - Iterations: {summary_df['iteration'].min()}-{summary_df['iteration'].max()}")
    print(f"  - Total rows: {len(summary_df)}")
    
    return summary_df


def compute_all_iterations_statistics(
    results_dir: Path | str,
    max_iterations: int = 15,
    original_data_path: Path | str | None = None,
    denormalize: bool = True,
) -> dict:
    """
    Compute statistics for all iterations in a results directory.
    
    Parameters
    ----------
    results_dir : Path or str
        Root results directory (e.g., 'results/iterative_search_2_reduced').
    max_iterations : int
        Maximum number of iterations to process.
    original_data_path : Path or str, optional
        Path to original (denormalized) data for normalization parameters.
        If None, will use default: 'data/split_by_element/ag02_daily_means_reduced.csv'
    denormalize : bool, optional
        Whether to denormalize statistics to original units. Default is True.
    
    Returns
    -------
    dict
        Mapping of iteration number to statistics DataFrame.
    """
    results_dir = Path(results_dir)
    all_stats = {}
    
    for iteration in range(1, max_iterations + 1):
        iteration_dir = results_dir / f'iteration_{iteration}'
        clustered_file = iteration_dir / 'clustered_data.csv'
        
        if not clustered_file.exists():
            print(f"Warning: {clustered_file} not found, skipping iteration {iteration}")
            continue
        
        print(f"\nProcessing iteration {iteration}...")
        clustered_df = pd.read_csv(clustered_file)
        stats_df = compute_cluster_statistics(
            clustered_df,
            iteration,
            iteration_dir,
            original_data_path=original_data_path,
            denormalize=denormalize,
        )
        all_stats[iteration] = stats_df
    
    return all_stats

def create_tree_summary_csv(
    results_dir: Path | str,
    output_filename: str = "tree_statistics_summary.csv",
) -> pd.DataFrame:
    """
    Create a summary CSV file for tree-based iterative search results.
    
    Aggregates mean values and cluster sizes from cluster_statistics.csv files in each node
    into a single CSV with one row per (depth, node_id, cluster) combination.
    
    Parameters
    ----------
    results_dir : Path or str
        Root results directory (e.g., 'results/iterative_search_tree').
    output_filename : str
        Name of the output summary file. Default is 'tree_statistics_summary.csv'.
    
    Returns
    -------
    pd.DataFrame
        Summary DataFrame with columns: depth, node_id, cluster, cluster_size, feature_1_mean, ...
    """
    results_dir = Path(results_dir)
    summary_rows = []
    
    # Find all depth directories
    depth_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('depth_')])
    
    for depth_dir in depth_dirs:
        depth = int(depth_dir.name.split('_')[1])
        
        # Find all node directories within this depth
        node_dirs = sorted([d for d in depth_dir.iterdir() if d.is_dir() and d.name.startswith('node_')])
        
        for node_dir in node_dirs:
            node_id = node_dir.name
            stats_file = node_dir / 'cluster_statistics.csv'
            
            if not stats_file.exists():
                continue
            
            try:
                # Load cluster statistics for this node
                stats_df = pd.read_csv(stats_file, index_col=0)
                
                # Extract only mean columns
                mean_cols = [col for col in stats_df.columns if col.endswith('_mean')]
                
                for cluster_id in stats_df.index:
                    row = {'depth': depth, 'node_id': node_id, 'cluster': cluster_id}
                    
                    # Add cluster size
                    if 'cluster_size' in stats_df.columns:
                        row['cluster_size'] = int(stats_df.loc[cluster_id, 'cluster_size'])
                    
                    # Add mean values for all features
                    for col in mean_cols:
                        row[col] = stats_df.loc[cluster_id, col]
                    
                    summary_rows.append(row)
            except Exception as e:
                print(f"Warning: Error processing {stats_file}: {e}")
                continue
    
    if not summary_rows:
        print(f"No cluster statistics found in {results_dir}")
        return pd.DataFrame()
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_rows)
    
    # Ensure proper column order: depth, node_id, cluster, cluster_size, then feature means
    cols = ['depth', 'node_id', 'cluster', 'cluster_size']
    mean_cols = sorted([col for col in summary_df.columns if col.endswith('_mean')])
    col_order = [c for c in cols if c in summary_df.columns] + mean_cols
    summary_df = summary_df[col_order]
    
    # Save summary
    output_file = results_dir / output_filename
    summary_df.to_csv(output_file, index=False)
    
    print(f"Saved tree summary to {output_file}")
    print(f"  - Shape: {summary_df.shape}")
    print(f"  - Depths: {summary_df['depth'].min()}-{summary_df['depth'].max()}")
    print(f"  - Total nodes: {summary_df['node_id'].nunique()}")
    print(f"  - Total rows: {len(summary_df)}")
    
    return summary_df