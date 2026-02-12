"""
Build a streaming coreset using the Har-Peled & Mazumdar algorithm
(via CoreSets-Algorithms: https://github.com/kvombatkere/CoreSets-Algorithms).

Loads a CSV (e.g. data/iris.csv), builds a coreset, and saves the result as a CSV
suitable for GAUFS (points expanded by integer weights so row count is manageable).

Usage:
  python run_coreset_streaming.py [--input data/iris.csv] [--output data/coreset_streaming.csv] [--max-size 500]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root for local vendored lib
sys.path.insert(0, str(Path(__file__).resolve().parent))
from coreset_streaming_lib import Coreset_Streaming


def run_stream(stream, max_cs_size=1000, chunk_size=1000, d=2):
    """Simulate streaming coreset construction. Returns Coreset_Streaming instance.

    Args:
        stream: List of points (each point a list of length d).
        max_cs_size: Maximum coreset size before resolution doubling.
        chunk_size: Chunk size for merging.
        d: Dimensionality.
    """
    a = Coreset_Streaming(max_cs_size, d=d)
    b = Coreset_Streaming(max_cs_size, d=d)

    n = len(stream)
    if n == 0:
        return a

    # First chunk into a
    first_chunk = min(chunk_size, n)
    for i in range(first_chunk):
        a.add_point(stream[i])
    a.build_coreset()

    # Stream rest into b, merge when chunk full
    for i in range(first_chunk, n):
        b.add_point(stream[i])
        if (i - first_chunk + 1) % chunk_size == 0:
            b.build_coreset()
            if a.can_union(b):
                a.union(b)
                b = Coreset_Streaming(max_cs_size, d=d)
            else:
                while not a.can_union(b):
                    if a.resolution > b.resolution:
                        b.double_resolution()
                    else:
                        a.double_resolution()
                a.union(b)
                b = Coreset_Streaming(max_cs_size, d=d)

    # Merge remaining in b
    if len(b.coreset) > 0:
        b.build_coreset()
        if a.can_union(b):
            a.union(b)
        else:
            while not a.can_union(b):
                if a.resolution > b.resolution:
                    b.double_resolution()
                else:
                    a.double_resolution()
            a.union(b)

    return a


def coreset_to_dataframe(coreset, feature_names=None):
    """Convert coreset (list of (point, weight)) to a DataFrame for GAUFS.

    Expands each point by round(weight) times (min 1) so GAUFS sees an unweighted
    dataset. Caps expansion so total rows stay reasonable for very large weights.
    """
    points = []
    weights = []
    for (point, w) in coreset.coreset:
        points.append(point)
        weights.append(w)

    points = np.array(points)
    weights = np.array(weights)

    # Expand by integer weight; cap at 1000 repeats per point to avoid huge CSVs
    max_repeat = 1000
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
    parser = argparse.ArgumentParser(description="Streaming coreset for GAUFS")
    parser.add_argument("--input", default="data/iris.csv", help="Input CSV (no header or with header)")
    parser.add_argument("--output", default="data/coreset_streaming.csv", help="Output coreset CSV")
    parser.add_argument("--max-size", type=int, default=500, help="Max coreset size before resolution doubling")
    parser.add_argument("--chunk-size", type=int, default=500, help="Streaming chunk size")
    parser.add_argument("--no-header", action="store_true", help="Input CSV has no header (e.g. iris)")
    args = parser.parse_args()

    # Load data
    if args.no_header:
        # Iris-style: all numeric columns, last column can be ignored for coreset
        df = pd.read_csv(args.input, header=None)
        # Use all columns for coreset (or drop last if it's a label)
        if df.shape[1] > 1 and df.iloc[:, -1].dtype == object or df.iloc[:, -1].dtype.name == "category":
            feature_cols = list(range(df.shape[1] - 1))
            feature_names = [f"x{j}" for j in feature_cols]
        else:
            feature_cols = list(range(df.shape[1]))
            feature_names = [f"x{j}" for j in feature_cols]
        X = df.iloc[:, feature_cols].astype(float).values
    else:
        df = pd.read_csv(args.input)
        # Use only numeric columns
        X = df.select_dtypes(include=[np.number]).values
        feature_names = list(df.select_dtypes(include=[np.number]).columns)

    n, d = X.shape
    stream = [list(row) for row in X]

    print(f"Loaded {n} rows, {d} dimensions")
    print(f"Building streaming coreset (max_size={args.max_size}, chunk_size={args.chunk_size})...")

    cs = run_stream(stream, max_cs_size=args.max_size, chunk_size=args.chunk_size, d=d)

    print(f"Coreset has {len(cs.coreset)} weighted points")

    out_df = coreset_to_dataframe(cs, feature_names=feature_names)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved expanded coreset to {out_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
