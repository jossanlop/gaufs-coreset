"""
Build a coreset via uniform sampling (in the spirit of Fast-Coreset-Generation
uniform compression: https://github.com/Andrew-Draganov/Fast-Coreset-Generation).

No dependency on the Fast-Coreset-Generation repo: uses simple random sampling
with importance weights so the coreset is an unbiased estimator. For sensitivity-
based or fast (HST-based) coresets, clone that repo and set FAST_CORESET_REPO.

Loads a CSV (e.g. data/iris.csv), builds a uniform coreset, and saves the result
as a CSV suitable for GAUFS (points expanded by integer weights).

Usage:
  python run_coreset_fast.py [--input data/iris.csv] [--output data/coreset_fast.csv] [--size 50]
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def uniform_coreset(points, m, weights=None, rng=None):
    """Uniform random sampling coreset. Returns (q_points, q_weights, q_labels).

    Each sampled point gets weight n/m so the coreset is an unbiased estimator.
    """
    n = len(points)
    if rng is None:
        rng = np.random.default_rng()
    if weights is None:
        weights = np.ones(n)
    if m >= n:
        q_points = np.array(points)
        q_weights = weights.copy()
        q_weights *= m / np.sum(q_weights)  # normalize so sum = m
        return q_points, q_weights, np.ones(n)

    weights_prob = weights / np.sum(weights)
    inds = rng.choice(np.arange(n), size=m, replace=False, p=weights_prob)
    q_points = np.array(points)[inds]
    q_weights = weights[inds] * float(n) / m
    return q_points, q_weights, np.ones(m)


def coreset_to_dataframe(q_points, q_weights, feature_names=None, max_repeat=1000):
    """Convert (points, weights) to a DataFrame for GAUFS.

    Expands each point by round(weight) times (min 1), capping at max_repeat per point.
    """
    points = np.atleast_2d(q_points)
    weights = np.atleast_1d(q_weights)
    rows = []
    for i, pt in enumerate(points):
        rep = max(1, min(max_repeat, int(round(weights[i]))))
        for _ in range(rep):
            rows.append(pt)
    X = np.array(rows)
    if feature_names is None:
        feature_names = [f"x{j}" for j in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names)


def main():
    parser = argparse.ArgumentParser(description="Uniform coreset for GAUFS")
    parser.add_argument("--input", default="data/iris.csv", help="Input CSV")
    parser.add_argument("--output", default="data/coreset_fast.csv", help="Output coreset CSV")
    parser.add_argument("--size", type=int, default=50, help="Target coreset size (number of points)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-header", action="store_true", help="Input CSV has no header (e.g. iris)")
    args = parser.parse_args()

    # Optional: use Fast-Coreset-Generation repo if available
    use_repo_coreset = None
    fast_repo = os.environ.get("FAST_CORESET_REPO")
    if fast_repo:
        try:
            import sys
            sys.path.insert(0, fast_repo)
            from make_coreset import uniform_coreset as repo_uniform_coreset
            use_repo_coreset = repo_uniform_coreset
            print(f"Using Fast-Coreset-Generation from {fast_repo}")
        except Exception as e:
            print(f"FAST_CORESET_REPO set but import failed: {e}. Using built-in uniform sampling.")

    # Load data
    if args.no_header:
        df = pd.read_csv(args.input, header=None)
        if df.shape[1] > 1 and (df.iloc[:, -1].dtype == object or df.iloc[:, -1].dtype.name == "category"):
            feature_cols = list(range(df.shape[1] - 1))
        else:
            feature_cols = list(range(df.shape[1]))
        feature_names = [f"x{j}" for j in feature_cols]
        X = df.iloc[:, feature_cols].astype(float).values
    else:
        df = pd.read_csv(args.input)
        X = df.select_dtypes(include=[np.number]).values
        feature_names = list(df.select_dtypes(include=[np.number]).columns)

    n, d = X.shape
    points = X.tolist()
    print(f"Loaded {n} rows, {d} dimensions")
    print(f"Building uniform coreset (size={args.size}, seed={args.seed})...")

    rng = np.random.default_rng(args.seed)
    if use_repo_coreset is not None:
        q_points, q_weights, _ = use_repo_coreset(points, args.size)
    else:
        q_points, q_weights, _ = uniform_coreset(points, args.size, rng=rng)

    print(f"Coreset has {len(q_points)} weighted points")

    out_df = coreset_to_dataframe(q_points, q_weights, feature_names=feature_names)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved expanded coreset to {out_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
