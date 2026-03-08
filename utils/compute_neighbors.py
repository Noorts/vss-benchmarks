"""
Compute ground-truth nearest neighbors for a test set against a train set
using DuckDB exact nearest-neighbor search.

Outputs a neighbors.parquet file with columns:
  - id: int64 (matching test.parquet ids)
  - neighbors_id: list<int64> (top-K nearest neighbor ids from train.parquet)

Usage:
  python compute_neighbors.py <dataset_dir> [--k 100] [--metric euclidean|cosine]

Example:
  python compute_neighbors.py files/sift-128-euclidean
"""

import argparse
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

DISTANCE_FUNCTIONS = {
    "euclidean": "array_distance",
    "cosine": "array_cosine_similarity",
}

# For cosine similarity, higher is better, so we need DESC ordering.
DISTANCE_ORDER = {
    "euclidean": "ASC",
    "cosine": "DESC",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ground-truth nearest neighbors.")
    parser.add_argument("dataset_dir", type=Path, help="Directory containing train.parquet and test.parquet")
    parser.add_argument("--k", type=int, default=100, help="Number of nearest neighbors (default: 100)")
    parser.add_argument("--metric", default=None, choices=["euclidean", "cosine"],
                        help="Distance metric (default: inferred from directory name)")
    args = parser.parse_args()

    dataset_dir: Path = args.dataset_dir
    train_path = dataset_dir / "train.parquet"
    test_path = dataset_dir / "test.parquet"
    output_path = dataset_dir / "neighbors.parquet"
    k: int = args.k

    # Infer metric from directory name if not specified
    metric = args.metric
    if metric is None:
        dirname = dataset_dir.name.lower()
        if "euclidean" in dirname:
            metric = "euclidean"
        elif "angular" in dirname or "cosine" in dirname:
            metric = "cosine"
        else:
            parser.error(f"Cannot infer metric from directory name '{dataset_dir.name}'. Use --metric.")

    dist_fn = DISTANCE_FUNCTIONS[metric]
    order = DISTANCE_ORDER[metric]

    # Detect dimensions from train set
    train_schema = pq.read_schema(train_path)
    test_table = pq.read_table(test_path)
    dims = len(test_table.column("emb")[0].as_py())
    query_count = test_table.num_rows

    print(f"Dataset:    {dataset_dir}")
    print(f"Metric:     {metric} ({dist_fn}, {order})")
    print(f"Dimensions: {dims}")
    print(f"Queries:    {query_count}")
    print(f"K:          {k}")

    # Load train set into DuckDB
    print("Loading train set into DuckDB...")
    con = duckdb.connect(":memory:")
    con.execute(
        f"CREATE TABLE dataset AS "
        f"SELECT id, emb::FLOAT[{dims}] AS emb "
        f"FROM '{train_path}'"
    )
    row_count = con.execute("SELECT COUNT(*) FROM dataset").fetchone()[0]
    print(f"  {row_count:,} train rows loaded")

    # Load query vectors ordered by id
    ids = test_table.column("id").to_pylist()
    embs = test_table.column("emb").to_pylist()
    order_idx = sorted(range(len(ids)), key=lambda i: ids[i])
    query_ids = [ids[i] for i in order_idx]
    query_vectors = [embs[i] for i in order_idx]

    # Compute nearest neighbors
    print("Computing nearest neighbors...")
    result_ids: list[int] = []
    result_neighbors: list[list[int]] = []

    for i in tqdm(range(query_count), desc="queries", unit="q"):
        vec = query_vectors[i]
        vec_literal = "[" + ", ".join(str(v) for v in vec) + f"]::FLOAT[{dims}]"

        sql = (
            f"SELECT id FROM dataset "
            f"ORDER BY {dist_fn}(emb, {vec_literal}) {order} "
            f"LIMIT {k}"
        )
        rows = con.execute(sql).fetchall()
        result_ids.append(query_ids[i])
        result_neighbors.append([row[0] for row in rows])

    con.close()

    # Write output
    out_table = pa.table({
        "id": pa.array(result_ids, type=pa.int64()),
        "neighbors_id": pa.array(result_neighbors, type=pa.list_(pa.int64())),
    })
    pq.write_table(out_table, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
